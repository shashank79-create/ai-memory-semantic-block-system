"""Local semantic memory store: SentenceTransformers embeddings in topic-block indexes.

Install: pip install sentence-transformers tiktoken numpy   (chromadb only for ChromaBackend)

Notes:
- The default BlockIndexBackend keeps one in-memory vector matrix per (user, topic)
  block, saved in SQLite. Filtering by user and topic picks blocks instead of scanning
  records, so searching one topic block only compares that block's vectors.
  ChromaBackend (same interface) is an option for very large stores.
- Backends use Chroma's conventions: cosine distance is (1 - cosine similarity), and
  `where` dicts take one top-level key, so conditions are combined with $and via `_where`.
- Expiry uses a numeric `ts` metadata field so deletion happens in one query.
  Records written before `ts` existed need `backfill_timestamps()` once.
- Only episodic memories expire. Semantic memories (lasting facts) are kept until
  deleted. Restating a fact refreshes the stored copy instead of adding a new one.
- Soft-deleted memories are removed for good by `purge_expired` once
  `deleted_grace_days` have passed, so `restore` works only within that window.
- Search covers all topics by default. `cross_topic=False` searches only the query's
  topic block plus "general", which is faster but can miss memories filed elsewhere.
- Locks make dedupe and writes safe within one process. They do not coordinate
  multiple processes sharing the same storage folder.
- chromadb, sentence-transformers and tiktoken are imported only when needed, so
  a custom `embedder` and `backend` can be used without them (as the tests do).
"""
from __future__ import annotations

import hashlib
import logging
import math
import random
import re
import threading
import time
import uuid
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Literal, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("semantic_memory")
MemoryType = Literal["episodic", "semantic"]
MEMORY_TYPES = ("episodic", "semantic")

DEFAULT_TOPICS = {
    "watches": ["watch", "rolex", "casio", "seiko", "omega", "timex", "analog", "digital",
                "automatic", "quartz", "wristwatch", "timepiece"],
    "food": ["food", "pizza", "sushi", "pasta", "burger", "restaurant", "dinner", "lunch",
             "breakfast", "cuisine", "recipe", "italian", "chinese", "mexican", "indian",
             "cook", "cooking", "meal", "eat", "snack", "peanut", "allergy", "allergic",
             "vegetarian", "vegan"],
    "travel": ["trip", "hotel", "flight", "vacation", "beach", "travel", "travelling", "traveling",
               "travelled", "traveled", "holiday", "airport", "passport", "visa"],
    "technology": ["python", "ai", "laptop", "code", "phone", "computer", "app", "software",
                   "iphone", "android", "gadget", "programming"],
}
DEFAULT_TOKEN_BUDGET = 512
# Cosine similarity at or above this counts as a duplicate. Measured with MiniLM:
# punctuation/case changes and "I really love X" score 0.99+, while
# "I am allergic to peanuts" vs "I am not allergic to peanuts" scores 0.970, so 0.97
# would drop that correction. Contradictions below this are stored as separate memories.
DUPLICATE_THRESHOLD = 0.985
# Search settings, chosen on eval/dataset.json (python eval/retrieval_eval.py --sweep)
# and checked on the held-out eval/test_set.json. With MiniLM, 36 memories put
# unrelated questions above 0.2 ("Tell me a joke" vs "I'm vegetarian"), so 0.3 is used.
DEFAULT_MIN_SIMILARITY = 0.3
# Memories in the same topic block as the query use this lower cutoff instead. MiniLM
# scores "Which watch should I buy?" vs "I love my Rolex Submariner" at only 0.15, and
# "Suggest a dinner recipe" vs "I am allergic to peanuts" at 0.11; the shared topic
# is what marks them as relevant.
DEFAULT_TOPIC_MIN_SIMILARITY = 0.1
# Results less similar than (best match - DEFAULT_MAX_GAP) are dropped, except memories
# in the query's topic block. None turns it off.
DEFAULT_MAX_GAP: Optional[float] = 0.15
# Only memories at least this similar to a new one are checked for contradictions.
# Measured with MiniLM: "I love my Rolex" vs "I hate my Rolex" 0.87, "My favourite colour
# is blue" vs "...green" 0.85, while compatible facts the NLI model still calls
# contradictions score lower ("I have a dog" vs "I have a cat" 0.77, "I love pizza" vs
# "I love sushi" 0.49). Changes worded very differently ("I live in Delhi" vs "I moved
# to Mumbai", 0.62) are missed.
CONTRADICTION_MIN_SIMILARITY = 0.80
CONTRADICTION_CANDIDATES = 5     # stored memories checked per new memory
DEFAULT_NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
RECENCY_HALF_LIFE_DAYS = 30.0    # recency score halves every 30 days
DEFAULT_DELETED_GRACE_DAYS = 7
SECONDS_PER_DAY = 86400.0
CHROMA_BATCH = 1000              # max ids per delete/update call


class MemoryStoreError(Exception):
    """Base exception for memory operations."""


class EmbeddingError(MemoryStoreError):
    """The embedding model could not encode input."""


class BackendError(MemoryStoreError):
    """The vector database operation failed."""


def _where(*conditions: dict) -> dict:
    """Combine conditions for Chroma, which accepts one top-level key per where dict."""
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": list(conditions)}


def _row(result: dict, key: str, row: int) -> list:
    """Safely read one query row from a Chroma result."""
    column = result.get(key) or []
    return column[row] if row < len(column) else []


def _similarity_from_distance(distance: float) -> float:
    """Convert Chroma cosine distance to similarity in [0, 1]."""
    # Chroma's cosine distance is 1 - cosine similarity; clamp floating-point edges.
    return min(1.0, max(0.0, 1.0 - float(distance)))


def _parse_timestamp(value: str) -> Optional[datetime]:
    try:
        ts = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def _age_days(timestamp: str, now: datetime) -> float:
    parsed = _parse_timestamp(timestamp)
    if parsed is None:
        return 0.0
    return max(0.0, (now - parsed).total_seconds() / SECONDS_PER_DAY)


def _validate(content: str, importance: float, memory_type: str) -> None:
    if not content.strip():
        raise ValueError("content must be non-empty")
    if not math.isfinite(importance) or not 0.0 <= importance <= 1.0:
        raise ValueError("importance must be between 0 and 1")
    if memory_type not in MEMORY_TYPES:
        raise ValueError("memory_type must be 'episodic' or 'semantic'")


def _stem(word: str) -> str:
    """Reduce simple English plurals to their singular form ("watches" -> "watch")."""
    if len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "sses", "xes", "zes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


class Metrics:
    """Thread-safe counters and uniform reservoir samples for latencies."""

    def __init__(self, sample_limit: int = 2048):
        self._lock = threading.Lock()
        self._counters: Counter[str] = Counter()
        self._samples: Dict[str, List[float]] = {}
        self._seen: Counter[str] = Counter()
        self._sample_limit = sample_limit

    def inc(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[name] += amount

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            samples = self._samples.setdefault(name, [])
            self._seen[name] += 1
            if len(samples) < self._sample_limit:
                samples.append(value)
            else:
                # Algorithm R: each observation kept with probability limit / seen.
                # This bounds memory use while keeping latency percentiles representative.
                j = random.randrange(self._seen[name])
                if j < self._sample_limit:
                    samples[j] = value

    def summary(self) -> dict:
        with self._lock:
            counters = dict(self._counters)
            samples = {name: list(values) for name, values in self._samples.items()}

        def percentile(values: List[float], p: float) -> float:
            return float(np.percentile(values, p)) if values else 0.0

        return {
            "counters": counters,
            "latency_p50": {n: percentile(v, 50) for n, v in samples.items()},
            "latency_p95": {n: percentile(v, 95) for n, v in samples.items()},
        }


metrics = Metrics()


@dataclass(slots=True)
class Memory:
    id: str
    user_id: str
    content: str
    topic: str
    timestamp: str
    importance: float = 1.0
    memory_type: MemoryType = "semantic"
    deleted: bool = False
    version: int = 1
    # Set only on memories returned by add/add_many: (id, content) of stored memories this
    # one contradicted. Those were soft-deleted, so `restore` can bring them back.
    replaced: Tuple[Tuple[str, str], ...] = ()


class EmbeddingCache:
    """Thread-safe LRU cache of read-only float32 vectors."""

    def __init__(self, maxsize: int, name: str = "cache"):
        if maxsize < 0:
            raise ValueError("cache size cannot be negative")
        self.maxsize = maxsize
        self.name = name
        # Cached vectors are read-only so callers cannot corrupt later searches.
        self._items: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[np.ndarray]:
        with self._lock:
            value = self._items.get(key)
            if value is None:
                metrics.inc(f"{self.name}.miss")
                return None
            self._items.move_to_end(key)
            metrics.inc(f"{self.name}.hit")
            return value

    def put(self, key: str, value: np.ndarray) -> None:
        if not self.maxsize:
            return
        value = np.asarray(value, dtype=np.float32)
        value.setflags(write=False)
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            if len(self._items) > self.maxsize:
                self._items.popitem(last=False)
                metrics.inc(f"{self.name}.eviction")


class TokenBudget:
    """Counts tokens with tiktoken, or estimates 4 characters per token without it.

    cl100k_base is OpenAI's tokenizer, so counts for other models (such as Gemini)
    are close estimates rather than exact.
    """

    CHARS_PER_TOKEN = 4

    def __init__(self, encoding: str = "cl100k_base"):
        try:
            import tiktoken
            self.encoder = tiktoken.get_encoding(encoding)
        except Exception as exc:  # not installed, or the encoding could not be downloaded
            logger.warning("tiktoken unavailable (%s); estimating tokens from length", exc)
            self.encoder = None

    def count(self, text: str) -> int:
        if self.encoder is None:
            return math.ceil(len(text) / self.CHARS_PER_TOKEN)
        return len(self.encoder.encode(text))

    def truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        if self.encoder is None:
            return text[:max_tokens * self.CHARS_PER_TOKEN]
        tokens = self.encoder.encode(text)
        return text if len(tokens) <= max_tokens else self.encoder.decode(tokens[:max_tokens])


class TopicClassifier:
    """Keyword classifier that also matches simple plurals ("watches" -> "watch")."""

    def __init__(self, topics: Dict[str, Sequence[str]]):
        self._topics = {t: frozenset(_stem(w.casefold()) for w in words)
                        for t, words in topics.items()}

    def classify(self, text: str) -> str:
        # Set intersection counts each matching keyword once, regardless of repetition.
        tokens = {_stem(t) for t in re.findall(r"\w+", text.casefold())}
        topic, best = "general", 0
        for name, words in self._topics.items():
            hits = len(tokens & words)
            if hits > best:
                topic, best = name, hits
        return topic


class ChromaBackend:
    """Thin wrapper that turns Chroma exceptions into BackendError."""

    def __init__(self, path: str, collection_name: str = "memories"):
        try:
            import chromadb
            self.client = chromadb.PersistentClient(path=path)
            self.collection = self.client.get_or_create_collection(
                collection_name, metadata={"hnsw:space": "cosine"}
            )
        except Exception as exc:
            raise BackendError(f"ChromaDB initialization failed: {exc}") from exc

    @staticmethod
    def _call(action: str, fn):
        try:
            return fn()
        except Exception as exc:
            raise BackendError(f"ChromaDB {action} failed: {exc}") from exc

    def add(self, ids, embeddings, metadatas, documents) -> None:
        self._call("add", lambda: self.collection.add(
            ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents))

    def query(self, embeddings: Sequence[Sequence[float]], k: int, where: dict) -> dict:
        """Run one ANN query per embedding in a single call. Results are indexed by row."""
        # Keep result shape consistent when no query can be run.
        empty = {key: [[] for _ in embeddings] for key in ("ids", "documents", "metadatas", "distances")}
        if k <= 0 or not embeddings:
            return empty
        count = self._call("count", self.collection.count)
        if not count:
            return empty
        return self._call("query", lambda: self.collection.query(
            query_embeddings=[list(e) for e in embeddings],
            n_results=min(k, count),
            where=where,
            include=["documents", "metadatas", "distances"],
        ))

    def get(self, where: Optional[dict] = None, ids: Optional[Sequence[str]] = None,
            include: Tuple[str, ...] = ("documents", "metadatas"),
            limit: Optional[int] = None, offset: int = 0) -> dict:
        return self._call("get", lambda: self.collection.get(
            where=where, ids=list(ids) if ids is not None else None,
            include=list(include), limit=limit, offset=offset))

    def delete(self, ids: Sequence[str]) -> None:
        ids = list(ids)
        for start in range(0, len(ids), CHROMA_BATCH):
            chunk = ids[start:start + CHROMA_BATCH]
            self._call("delete", lambda c=chunk: self.collection.delete(ids=c))

    def update_metadata(self, ids: Sequence[str], metadatas: Sequence[dict]) -> None:
        ids, metadatas = list(ids), list(metadatas)
        for start in range(0, len(ids), CHROMA_BATCH):
            self._call("metadata update", lambda s=start: self.collection.update(
                ids=ids[s:s + CHROMA_BATCH], metadatas=metadatas[s:s + CHROMA_BATCH]))


def _flatten_where(where: Optional[dict]) -> List[Tuple[str, object]]:
    """Turn a Chroma-style where dict into a flat list of (field, condition) pairs."""
    if not where:
        return []
    if "$and" in where:
        return [pair for sub in where["$and"] for pair in _flatten_where(sub)]
    if len(where) != 1:
        raise BackendError("where dicts take one top-level key; combine conditions with $and")
    (field, cond), = where.items()
    return [(field, cond)]


_COMPARE = {
    "$lt": lambda a, b: a < b, "$lte": lambda a, b: a <= b,
    "$gt": lambda a, b: a > b, "$gte": lambda a, b: a >= b,
}


def _check(value, cond) -> bool:
    if not isinstance(cond, dict):
        return value == cond
    (op, target), = cond.items()
    if op == "$eq":
        return value == target
    if op == "$ne":
        return value != target
    if op == "$in":
        return value in target
    if op == "$nin":
        return value not in target
    if op in _COMPARE:
        return value is not None and _COMPARE[op](value, target)
    raise BackendError(f"unsupported where operator {op!r}")


def _equal_values(conds: Sequence[Tuple[str, object]], field: str) -> Optional[set]:
    """Values `field` is restricted to by equality/$in conditions, or None if unrestricted."""
    allowed: Optional[set] = None
    for name, cond in conds:
        if name != field:
            continue
        if isinstance(cond, dict):
            (op, target), = cond.items()
            values = {target} if op == "$eq" else set(target) if op == "$in" else None
        else:
            values = {cond}
        if values is not None:
            allowed = values if allowed is None else allowed & values
    return allowed


class _Block:
    """Vectors of one (user, topic) block in a growable float32 matrix."""

    def __init__(self, dim: int):
        self.ids: List[str] = []
        self.pos: Dict[str, int] = {}
        self._vecs = np.empty((16, dim), dtype=np.float32)

    @property
    def matrix(self) -> np.ndarray:
        return self._vecs[:len(self.ids)]

    def add(self, mem_id: str, vec: np.ndarray) -> None:
        n = len(self.ids)
        if n == len(self._vecs):
            grown = np.empty((2 * n, self._vecs.shape[1]), dtype=np.float32)
            grown[:n] = self._vecs
            self._vecs = grown
        self._vecs[n] = vec
        self.pos[mem_id] = n
        self.ids.append(mem_id)

    def remove(self, mem_id: str) -> np.ndarray:
        """Remove a vector (moving the last one into its slot) and return it."""
        i = self.pos.pop(mem_id)
        vec = self._vecs[i].copy()
        last = len(self.ids) - 1
        if i != last:
            self._vecs[i] = self._vecs[last]
            self.ids[i] = self.ids[last]
            self.pos[self.ids[i]] = i
        self.ids.pop()
        return vec


class BlockIndexBackend:
    """Default backend: SQLite on disk, one in-memory vector matrix per (user, topic) block.

    Searches are exact (every vector in the selected blocks is compared, using one
    matrix multiply per block). Filters on user and topic pick blocks directly instead of
    scanning records, so a search inside one topic block only touches that block. Other
    filters (deleted, memory_type, ...) are checked only on the best matches, in order,
    until enough are found.

    Everything is loaded into memory at startup: about 1.5 KB per memory with MiniLM, so
    100,000 memories use roughly 150 MB. For millions of memories, use ChromaBackend.
    Same interface and Chroma-style results as ChromaBackend.
    """

    def __init__(self, path: str):
        import json
        import sqlite3
        self._json = json
        self._lock = threading.RLock()
        try:
            self._db = sqlite3.connect(str(path), check_same_thread=False)
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS memories (id TEXT PRIMARY KEY, user_id TEXT NOT NULL, "
                "topic TEXT NOT NULL, document TEXT NOT NULL, metadata TEXT NOT NULL, "
                "embedding BLOB NOT NULL)")
            rows = self._db.execute(
                "SELECT id, document, metadata, embedding FROM memories ORDER BY rowid").fetchall()
        except sqlite3.Error as exc:
            raise BackendError(f"SQLite initialization failed: {exc}") from exc
        self._meta: Dict[str, dict] = {}   # insertion ordered
        self._docs: Dict[str, str] = {}
        self._blocks: Dict[Tuple[str, str], _Block] = {}
        self._dim: Optional[int] = None
        for mem_id, doc, meta, emb in rows:
            self._insert(mem_id, np.frombuffer(emb, dtype=np.float32), self._json.loads(meta), doc)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def _write(self, sql: str, rows: Sequence[tuple]) -> None:
        import sqlite3
        try:
            with self._db:
                self._db.executemany(sql, rows)
        except sqlite3.Error as exc:
            raise BackendError(f"SQLite write failed: {exc}") from exc

    def _insert(self, mem_id: str, vec: np.ndarray, meta: dict, doc: str) -> None:
        if self._dim is None:
            self._dim = len(vec)
        elif len(vec) != self._dim:
            raise BackendError(f"embedding has {len(vec)} dimensions, expected {self._dim}")
        key = (meta["user_id"], meta["topic"])
        block = self._blocks.get(key)
        if block is None:
            block = self._blocks[key] = _Block(self._dim)
        block.add(mem_id, vec)
        self._meta[mem_id] = meta
        self._docs[mem_id] = doc

    def _select_blocks(self, conds) -> List[_Block]:
        users, topics = _equal_values(conds, "user_id"), _equal_values(conds, "topic")
        return [b for (u, t), b in self._blocks.items()
                if (users is None or u in users) and (topics is None or t in topics) and b.ids]

    def _matches(self, mem_id: str, conds) -> bool:
        meta = self._meta[mem_id]
        return all(_check(meta.get(field), cond) for field, cond in conds)

    def add(self, ids, embeddings, metadatas, documents) -> None:
        with self._lock:
            dupes = [i for i in ids if i in self._meta]
            if dupes or len(set(ids)) != len(ids):
                raise BackendError(f"ids already exist: {dupes[:5]}")
            vecs = [np.asarray(e, dtype=np.float32) for e in embeddings]
            self._write(
                "INSERT INTO memories (id, user_id, topic, document, metadata, embedding) VALUES (?,?,?,?,?,?)",
                [(i, m["user_id"], m["topic"], d, self._json.dumps(m), v.tobytes())
                 for i, v, m, d in zip(ids, vecs, metadatas, documents)])
            for i, v, m, d in zip(ids, vecs, metadatas, documents):
                self._insert(i, v, dict(m), d)

    def query(self, embeddings: Sequence[Sequence[float]], k: int, where: dict) -> dict:
        result = {key: [] for key in ("ids", "documents", "metadatas", "distances")}
        conds = _flatten_where(where)
        with self._lock:
            blocks = self._select_blocks(conds) if k > 0 else []
            for emb in embeddings:
                hits = self._query_one(np.asarray(emb, dtype=np.float32), k, blocks, conds)
                result["ids"].append([h[0] for h in hits])
                result["distances"].append([1.0 - h[1] for h in hits])
                result["metadatas"].append([dict(self._meta[h[0]]) for h in hits])
                result["documents"].append([self._docs[h[0]] for h in hits])
        return result

    def _query_one(self, q: np.ndarray, k: int, blocks: List[_Block], conds) -> List[Tuple[str, float]]:
        if not blocks:
            return []
        sims = np.concatenate([b.matrix @ q for b in blocks])
        offsets = np.cumsum([0] + [len(b.ids) for b in blocks])
        n = len(sims)
        m = min(n, max(4 * k, 32))
        while True:
            # Sort only the best m, check filters in that order, and widen if too few pass.
            top = np.argpartition(-sims, m - 1)[:m] if m < n else np.arange(n)
            hits = []
            for idx in top[np.argsort(-sims[top], kind="stable")]:
                b = int(np.searchsorted(offsets, idx, side="right")) - 1
                mem_id = blocks[b].ids[idx - offsets[b]]
                if self._matches(mem_id, conds):
                    hits.append((mem_id, float(sims[idx])))
                    if len(hits) == k:
                        return hits
            if m == n:
                return hits
            m = min(n, 4 * m)

    def get(self, where: Optional[dict] = None, ids: Optional[Sequence[str]] = None,
            include: Tuple[str, ...] = ("documents", "metadatas"),
            limit: Optional[int] = None, offset: int = 0) -> dict:
        conds = _flatten_where(where)
        with self._lock:
            if ids is not None:
                candidates = [i for i in dict.fromkeys(ids) if i in self._meta]
            else:
                users = _equal_values(conds, "user_id")
                candidates = [i for i, m in self._meta.items() if users is None or m["user_id"] in users]
            selected = [i for i in candidates if self._matches(i, conds)]
            selected = selected[offset:None if limit is None else offset + limit]
            return {"ids": selected,
                    "documents": [self._docs[i] for i in selected],
                    "metadatas": [dict(self._meta[i]) for i in selected]}

    def delete(self, ids: Sequence[str]) -> None:
        with self._lock:
            ids = [i for i in dict.fromkeys(ids) if i in self._meta]
            self._write("DELETE FROM memories WHERE id = ?", [(i,) for i in ids])
            for i in ids:
                meta = self._meta.pop(i)
                self._docs.pop(i)
                self._blocks[(meta["user_id"], meta["topic"])].remove(i)

    def update_metadata(self, ids: Sequence[str], metadatas: Sequence[dict]) -> None:
        with self._lock:
            pairs = [(i, {**self._meta[i], **m}) for i, m in zip(ids, metadatas) if i in self._meta]
            self._write("UPDATE memories SET user_id = ?, topic = ?, metadata = ? WHERE id = ?",
                        [(m["user_id"], m["topic"], self._json.dumps(m), i) for i, m in pairs])
            for i, merged in pairs:
                old = self._meta[i]
                if (old["user_id"], old["topic"]) != (merged["user_id"], merged["topic"]):
                    vec = self._blocks[(old["user_id"], old["topic"])].remove(i)
                    self._insert(i, vec, merged, self._docs[i])
                else:
                    self._meta[i] = merged


class NLIContradictionChecker:
    """Asks a natural-language-inference model whether a new statement contradicts an old one.

    Call it with (old, new) text pairs; it returns one bool per pair. On its own such a
    model is not enough: it rates "I love pizza" vs "I love sushi" and "I have a dog" vs
    "I have a cat" as contradictions. The store therefore only asks about pairs that are
    already very similar (see CONTRADICTION_MIN_SIMILARITY).
    """

    def __init__(self, model_name: str = DEFAULT_NLI_MODEL, threshold: float = 0.9):
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
        except Exception as exc:
            raise EmbeddingError(f"Failed to load NLI model {model_name!r}: {exc}") from exc
        labels = {i: str(l).lower() for i, l in self.model.model.config.id2label.items()}
        matches = [i for i, l in labels.items() if l.startswith("contradiction")]
        if not matches:
            raise EmbeddingError(f"NLI model {model_name!r} has no contradiction label: {labels}")
        self.index = matches[0]
        self.threshold = threshold

    def __call__(self, pairs: Sequence[Tuple[str, str]]) -> List[bool]:
        if not pairs:
            return []
        logits = np.asarray(self.model.predict(list(pairs)), dtype=np.float64).reshape(len(pairs), -1)
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        return [bool(p) for p in probs[:, self.index] >= self.threshold]


class SemanticMemoryStore:
    def __init__(self, storage_path: str = "./memory", embed_model: str = "all-MiniLM-L6-v2",
                 topic_config: Optional[Dict[str, Sequence[str]]] = None, ttl_days: int = 90,
                 deleted_grace_days: int = DEFAULT_DELETED_GRACE_DAYS,
                 cache_size: int = 1000, query_cache_size: int = 100, embedder=None, backend=None,
                 clock: Optional[Callable[[], datetime]] = None,
                 contradiction_checker: Optional[Callable[[Sequence[Tuple[str, str]]], List[bool]]] = None,
                 contradiction_min_similarity: float = CONTRADICTION_MIN_SIMILARITY):
        """`contradiction_checker` (for example NLIContradictionChecker()) turns on replacing
        contradicted memories; without it, contradicting memories are both kept."""
        if ttl_days <= 0:
            raise ValueError("ttl_days must be positive")
        if deleted_grace_days < 0:
            raise ValueError("deleted_grace_days cannot be negative")
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        if embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                embedder = SentenceTransformer(embed_model)
            except Exception as exc:
                raise EmbeddingError(f"Failed to load embedding model {embed_model!r}: {exc}") from exc
            model_name = embed_model
        else:
            model_name = type(embedder).__name__
        self.embedder = embedder
        # Newer sentence-transformers renamed the method; support both names.
        dim_fn = (getattr(self.embedder, "get_embedding_dimension", None)
                  or getattr(self.embedder, "get_sentence_embedding_dimension", None))
        dim = dim_fn() if dim_fn else "unknown"
        self.model_key = f"{model_name}:{dim}"
        self.backend = backend or BlockIndexBackend(str(self.storage_path / "memories.db"))
        self.classifier = TopicClassifier(topic_config or DEFAULT_TOPICS)
        self.ttl_days = ttl_days
        self.deleted_grace_days = deleted_grace_days
        self.contradiction_checker = contradiction_checker
        self.contradiction_min_similarity = contradiction_min_similarity
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.tokenizer = TokenBudget()
        self.embed_cache = EmbeddingCache(cache_size, "embed_cache")
        self.query_cache = EmbeddingCache(query_cache_size, "query_cache")
        # Serializes dedupe plus insert so two threads in this process cannot race.
        # Stats refreshes take it too, so a write cannot slip between count and cache.
        self._write_lock = threading.RLock()
        self._stats_cache: Dict[str, Dict[str, int]] = {}
        self._stats_lock = threading.Lock()

    def _now(self) -> datetime:
        return self._clock()

    # ----- embeddings -------------------------------------------------------

    def _embed_many(self, texts: Sequence[str], cache: EmbeddingCache) -> np.ndarray:
        """Return an (n, dim) array of unit vectors, embedding each distinct miss once."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        keys = [self.model_key + ":" + hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts]
        out: List[Optional[np.ndarray]] = [cache.get(k) for k in keys]
        missing = [i for i, v in enumerate(out) if v is None]
        if missing:
            # Encode unique misses once, then fan vectors back out to duplicate inputs.
            unique = list(dict.fromkeys(texts[i] for i in missing))
            slot = {t: n for n, t in enumerate(unique)}
            started = time.perf_counter()
            try:
                vectors = np.asarray(
                    self.embedder.encode(unique, normalize_embeddings=True, convert_to_numpy=True),
                    dtype=np.float32,
                )
            except Exception as exc:
                raise EmbeddingError(f"Embedding failed: {exc}") from exc
            if vectors.ndim != 2 or vectors.shape[0] != len(unique) or not np.isfinite(vectors).all():
                raise EmbeddingError("model returned invalid embedding vectors")
            for i in missing:
                vec = vectors[slot[texts[i]]]
                cache.put(keys[i], vec)
                out[i] = vec
            metrics.observe("embed.latency", time.perf_counter() - started)
            metrics.inc("embed.count", len(unique))
        return np.stack(out)

    def _embed_query(self, text: str) -> np.ndarray:
        return self._embed_many([text], self.query_cache)[0]

    # ----- helpers ----------------------------------------------------------

    @staticmethod
    def _memory(mem_id: str, document: str, meta: dict) -> Memory:
        return Memory(
            id=mem_id, user_id=meta["user_id"], content=document, topic=meta["topic"],
            timestamp=meta["timestamp"],
            # Older records may hold importance above 1; clamp so scoring stays in range.
            importance=min(1.0, max(0.0, float(meta.get("importance", 1.0)))),
            memory_type=meta.get("memory_type", "semantic"),
            deleted=bool(meta.get("deleted", False)), version=int(meta.get("version", 1)),
        )

    def _bump_stats(self, user_id: str, topics: Sequence[str], delta: int) -> None:
        with self._stats_lock:
            counts = self._stats_cache.get(user_id)
            if counts is None:
                return
            for topic in topics:
                value = counts.get(topic, 0) + delta
                if value > 0:
                    counts[topic] = value
                else:
                    counts.pop(topic, None)  # match a fresh count, which has no empty topics

    def _find_duplicates(self, vectors: np.ndarray, flags: Sequence[bool],
                         user_id: str) -> Tuple[List[bool], Dict[int, str], Dict[int, int]]:
        """Sort a batch into new items, restated stored memories and in-batch repeats.

        Returns (keep, stored, batch):
        - keep[i] is True when item i should be stored as a new memory.
        - stored[i] is the id of the stored memory item i repeats.
        - batch[i] is the index of the earlier new item in this batch that item i repeats.

        Dedupe spans all topics for the user, so one fact filed under two topics is caught.
        """
        keep = [True] * len(flags)
        stored: Dict[int, str] = {}
        batch: Dict[int, int] = {}
        checks = [i for i, flag in enumerate(flags) if flag]
        if not checks:
            return keep, stored, batch
        # One batched ANN query covers every item that asked for dedupe.
        result = self.backend.query(
            [vectors[i].tolist() for i in checks], 1,
            _where({"user_id": user_id}, {"deleted": False}),
        )
        accepted: List[int] = []
        for row, i in enumerate(checks):
            ids = _row(result, "ids", row)
            if ids and _similarity_from_distance(_row(result, "distances", row)[0]) >= DUPLICATE_THRESHOLD:
                keep[i] = False
                stored[i] = ids[0]
                metrics.inc("duplicate.detected")
                continue
            if accepted:
                sims = vectors[accepted] @ vectors[i]
                best = int(np.argmax(sims))
                if float(sims[best]) >= DUPLICATE_THRESHOLD:
                    keep[i] = False
                    batch[i] = accepted[best]
                    metrics.inc("duplicate.detected")
                    continue
            accepted.append(i)
        return keep, stored, batch

    def _reinforce(self, user_id: str, repeats: Dict[str, List[Tuple[float, str]]],
                   now_iso: str, now_ts: float) -> List[Memory]:
        """Refresh stored memories that were restated.

        The timestamp moves to now (so episodic expiry restarts), importance becomes the
        highest value seen, and a restated semantic fact makes the memory semantic.
        """
        if not repeats:
            return []
        found = self.backend.get(
            where=_where({"user_id": user_id}, {"deleted": False}),
            ids=list(repeats), include=("documents", "metadatas"),
        )
        ids, metas, refreshed = [], [], []
        for mem_id, doc, meta in zip(found["ids"], found["documents"], found["metadatas"]):
            importance = max([float(meta.get("importance", 1.0))] + [imp for imp, _ in repeats[mem_id]])
            kinds = {meta.get("memory_type", "semantic")} | {kind for _, kind in repeats[mem_id]}
            new_meta = {
                **meta, "timestamp": now_iso, "ts": now_ts,
                "importance": min(1.0, importance),
                "memory_type": "semantic" if "semantic" in kinds else "episodic",
                "version": int(meta.get("version", 1)) + 1,
            }
            ids.append(mem_id)
            metas.append(new_meta)
            refreshed.append(self._memory(mem_id, doc, new_meta))
        if ids:
            self.backend.update_metadata(ids, metas)
            metrics.inc("duplicate.reinforced", len(ids))
        return refreshed

    def _find_contradictions(self, vectors: np.ndarray, contents: Sequence[str], new: Sequence[int],
                             user_id: str) -> Dict[int, List[Tuple[str, str, dict]]]:
        """Map each new item index to the stored memories it contradicts, as (id, doc, meta).

        Only stored memories whose similarity is between `contradiction_min_similarity` and
        the duplicate threshold are sent to the checker. Items in the same batch are not
        checked against each other.
        """
        if self.contradiction_checker is None or not new:
            return {}
        result = self.backend.query(
            [vectors[i].tolist() for i in new], CONTRADICTION_CANDIDATES,
            _where({"user_id": user_id}, {"deleted": False}),
        )
        candidates: List[Tuple[int, str, str, dict]] = []
        for row, i in enumerate(new):
            for mem_id, doc, meta, dist in zip(_row(result, "ids", row), _row(result, "documents", row),
                                               _row(result, "metadatas", row), _row(result, "distances", row)):
                if self.contradiction_min_similarity <= _similarity_from_distance(dist) < DUPLICATE_THRESHOLD:
                    candidates.append((i, mem_id, doc, meta))
        if not candidates:
            return {}
        verdicts = self.contradiction_checker([(doc, contents[i]) for i, _, doc, _ in candidates])
        metrics.inc("contradiction.checked", len(candidates))
        found: Dict[int, List[Tuple[str, str, dict]]] = {}
        claimed = set()
        for (i, mem_id, doc, meta), contradicts in zip(candidates, verdicts):
            if contradicts and mem_id not in claimed:  # each old memory is replaced once
                claimed.add(mem_id)
                found.setdefault(i, []).append((mem_id, doc, meta))
        return found

    def _supersede(self, user_id: str, old: Sequence[Tuple[str, str, dict]], new_id: str) -> None:
        """Soft-delete memories contradicted by `new_id`, recording which memory replaced them."""
        now_ts = self._now().timestamp()
        self.backend.update_metadata(
            [mem_id for mem_id, _, _ in old],
            [{**meta, "deleted": True, "deleted_ts": now_ts, "superseded_by": new_id,
              "version": int(meta.get("version", 1)) + 1} for _, _, meta in old],
        )
        self._bump_stats(user_id, [meta["topic"] for _, _, meta in old], -1)
        metrics.inc("contradiction.replaced", len(old))

    # ----- writes -----------------------------------------------------------

    def add(self, content: str, user_id: str, topic: Optional[str] = None, importance: float = 1.0,
            memory_type: MemoryType = "semantic", skip_duplicates: bool = True) -> Optional[Memory]:
        """Store one memory, or refresh the stored copy if it repeats one. Returns that memory."""
        created = self.add_many([{
            "content": content, "topic": topic, "importance": importance,
            "memory_type": memory_type, "skip_duplicates": skip_duplicates,
        }], user_id)
        return created[0] if created else None

    def add_many(self, items: Sequence[dict], user_id: str) -> List[Memory]:
        """Store a batch with one embedding call and at most one dedupe query.

        Items repeating a stored memory refresh it instead of being added (see
        `_reinforce`). Items repeating an earlier item in the same batch are merged into
        it. Returns the new and refreshed memories.
        """
        if not user_id.strip():
            raise ValueError("user_id must be non-empty")
        if not items:
            return []
        prepared = []
        for item in items:
            content = str(item["content"])
            importance = float(item.get("importance", 1.0))
            memory_type = item.get("memory_type", "semantic")
            _validate(content, importance, memory_type)
            topic = item.get("topic") or self.classifier.classify(content)
            prepared.append([content, topic, importance, memory_type,
                             bool(item.get("skip_duplicates", True))])

        started = time.perf_counter()
        vectors = self._embed_many([p[0] for p in prepared], self.embed_cache)
        with self._write_lock:
            keep, stored, batch = self._find_duplicates(vectors, [p[4] for p in prepared], user_id)
            # Fold in-batch repeats into the item they repeat.
            for i, target in batch.items():
                prepared[target][2] = max(prepared[target][2], prepared[i][2])
                if prepared[i][3] == "semantic":
                    prepared[target][3] = "semantic"
            contradicted = self._find_contradictions(
                vectors, [p[0] for p in prepared], [i for i, k in enumerate(keep) if k], user_id)
            now = self._now()
            now_iso, now_ts = now.isoformat(), now.timestamp()
            ids, embs, metas, docs, created = [], [], [], [], []
            for i, (content, topic, importance, memory_type, _) in enumerate(prepared):
                if not keep[i]:
                    continue
                mem_id = str(uuid.uuid4())
                ids.append(mem_id)
                embs.append(vectors[i].tolist())
                docs.append(content)
                metas.append({
                    "user_id": user_id, "topic": topic, "timestamp": now_iso, "ts": now_ts,
                    "importance": importance, "memory_type": memory_type,
                    "deleted": False, "deleted_ts": 0.0, "version": 1,
                })
                old = contradicted.get(i, [])
                created.append(Memory(mem_id, user_id, content, topic, now_iso, importance, memory_type,
                                      replaced=tuple((old_id, doc) for old_id, doc, _ in old)))
            if ids:
                self.backend.add(ids, embs, metas, docs)
                self._bump_stats(user_id, [m.topic for m in created], +1)
                # `created` follows item order, so pair each new memory with its item index.
                for mem, i in zip(created, [i for i, k in enumerate(keep) if k]):
                    if i in contradicted:
                        self._supersede(user_id, contradicted[i], mem.id)
            repeats: Dict[str, List[Tuple[float, str]]] = {}
            for i, mem_id in stored.items():
                repeats.setdefault(mem_id, []).append((prepared[i][2], prepared[i][3]))
            refreshed = self._reinforce(user_id, repeats, now_iso, now_ts)
        metrics.inc("add.count", len(created))
        metrics.observe("add.latency", time.perf_counter() - started)
        return created + refreshed

    def _set_deleted(self, user_id: str, memory_ids: Sequence[str], deleted: bool) -> int:
        ids = list(dict.fromkeys(memory_ids))
        if not ids:
            return 0
        with self._write_lock:
            # Filter by owner and current state so IDs from another user cannot be changed.
            found = self.backend.get(
                where=_where({"user_id": user_id}, {"deleted": not deleted}),
                ids=ids, include=("metadatas",),
            )
            rows = list(zip(found["ids"], found["metadatas"]))
            if not rows:
                return 0
            deleted_ts = self._now().timestamp() if deleted else 0.0
            self.backend.update_metadata(
                [i for i, _ in rows],
                [{**m, "deleted": deleted, "deleted_ts": deleted_ts,
                  "version": int(m.get("version", 1)) + 1} for _, m in rows],
            )
            self._bump_stats(user_id, [m["topic"] for _, m in rows], -1 if deleted else +1)
        metrics.inc("delete.count" if deleted else "restore.count", len(rows))
        return len(rows)

    def soft_delete(self, user_id: str, memory_ids: Sequence[str]) -> int:
        """Hide memories now; `purge_expired` removes them after `deleted_grace_days`."""
        return self._set_deleted(user_id, memory_ids, deleted=True)

    def restore(self, user_id: str, memory_ids: Sequence[str]) -> int:
        return self._set_deleted(user_id, memory_ids, deleted=False)

    def purge_expired(self, user_id: str) -> int:
        """Permanently delete expired episodic memories and soft-deleted memories past grace.

        Semantic memories never expire by age; they are removed only after being deleted.
        """
        now = self._now()
        expired_cutoff = (now - timedelta(days=self.ttl_days)).timestamp()
        deleted_cutoff = (now - timedelta(days=self.deleted_grace_days)).timestamp()
        expired_where = _where({"user_id": user_id}, {"deleted": False},
                               {"memory_type": "episodic"}, {"ts": {"$lt": expired_cutoff}})
        deleted_where = _where({"user_id": user_id}, {"deleted": True},
                               {"deleted_ts": {"$lte": deleted_cutoff}})
        with self._write_lock:
            expired = self.backend.get(where=expired_where, include=("metadatas",))
            removed = self.backend.get(where=deleted_where, include=("metadatas",))
            ids = list(expired["ids"]) + list(removed["ids"])
            if not ids:
                return 0
            self.backend.delete(ids)
            # Soft-deleted memories were already taken out of the counts when deleted.
            self._bump_stats(user_id, [m["topic"] for m in expired["metadatas"]], -1)
        metrics.inc("purge.count", len(ids))
        return len(ids)

    def backfill_timestamps(self, page_size: int = 500) -> int:
        """One-time migration: add numeric `ts` and `deleted_ts` to older records."""
        offset, fixed = 0, 0
        now_ts = self._now().timestamp()
        while True:
            page = self.backend.get(include=("metadatas",), limit=page_size, offset=offset)
            ids, metas = page["ids"], page["metadatas"]
            if not ids:
                return fixed
            fix_ids, fix_metas = [], []
            for mem_id, meta in zip(ids, metas):
                patch = {}
                if "ts" not in meta:
                    parsed = _parse_timestamp(meta.get("timestamp", ""))
                    if parsed is not None:
                        patch["ts"] = parsed.timestamp()
                if "deleted_ts" not in meta:
                    # Unknown deletion time: start the grace period now.
                    patch["deleted_ts"] = now_ts if meta.get("deleted") else 0.0
                if patch:
                    fix_ids.append(mem_id)
                    fix_metas.append({**meta, **patch})
            if fix_ids:
                self.backend.update_metadata(fix_ids, fix_metas)
                fixed += len(fix_ids)
            if len(ids) < page_size:
                return fixed
            offset += page_size

    # ----- reads ------------------------------------------------------------

    def search(self, query: str, user_id: str, topic: Optional[str] = None,
               memory_type: Optional[MemoryType] = None, k: int = 5,
               alpha_similarity: float = 0.6, alpha_importance: float = 0.2,
               alpha_recency: float = 0.2,
               min_similarity: float = DEFAULT_MIN_SIMILARITY,
               topic_min_similarity: float = DEFAULT_TOPIC_MIN_SIMILARITY,
               use_query_topic: bool = True, cross_topic: bool = True,
               max_gap: Optional[float] = DEFAULT_MAX_GAP) -> List[Memory]:
        """Return up to k memories, best first.

        Results below `min_similarity` are dropped, so an unrelated query returns nothing
        rather than filler. When `topic` is not given and `use_query_topic` is on, the
        query's own topic is detected; memories in that topic block are also searched
        and need only `topic_min_similarity`.

        `cross_topic=False` searches only the query's topic block and the "general" block
        (when a topic is detected). That is faster, but misses relevant memories filed
        under other topics.
        """
        if k <= 0:
            return []
        weights = (alpha_similarity, alpha_importance, alpha_recency)
        if any(not math.isfinite(w) or w < 0 for w in weights):
            raise ValueError("score weights must be finite and non-negative")
        total = sum(weights)
        if total == 0:
            raise ValueError("at least one search score weight must be positive")

        started = time.perf_counter()
        conditions = [{"user_id": user_id}, {"deleted": False}]
        if topic:
            conditions.append({"topic": topic})
        if memory_type:
            conditions.append({"memory_type": memory_type})
        embedding = [self._embed_query(query).tolist()]
        query_topic = None
        if topic is None and use_query_topic:
            detected = self.classifier.classify(query)
            query_topic = detected if detected != "general" else None

        # Over-fetch (k * 3) so recency and importance can reorder more than the nearest k.
        if query_topic and not cross_topic:
            # Block-only search: the query's topic block plus the "general" block.
            # With BlockIndexBackend this compares only those blocks' vectors.
            scopes = [{"topic": query_topic}, {"topic": "general"}]
        else:
            # All memories, plus the topic block on its own so its memories are found
            # even when other topics fill the nearest results.
            scopes = [None] + ([{"topic": query_topic}] if query_topic else [])
        results = [self.backend.query(embedding, k * 3,
                                      _where(*conditions, *([scope] if scope else [])))
                   for scope in scopes]

        candidates: Dict[str, Tuple[float, str, dict]] = {}
        for result in results:
            rows = zip(_row(result, "ids", 0), _row(result, "documents", 0),
                       _row(result, "metadatas", 0), _row(result, "distances", 0))
            for mem_id, doc, meta, dist in rows:
                candidates[mem_id] = (_similarity_from_distance(dist), doc, meta)

        now = self._now()
        ranked = []
        passing = {
            mem_id: (similarity, doc, meta) for mem_id, (similarity, doc, meta) in candidates.items()
            if similarity >= (topic_min_similarity
                              if query_topic is not None and meta.get("topic") == query_topic
                              else min_similarity)
        }
        if max_gap is not None and passing:
            # Relative cutoff: drop results much less similar than the best match, except
            # memories in the query's topic block (their own cutoff already applied).
            best = max(similarity for similarity, _, _ in passing.values())
            passing = {i: v for i, v in passing.items()
                       if v[0] >= best - max_gap
                       or (query_topic is not None and v[2].get("topic") == query_topic)}
        for mem_id, (similarity, doc, meta) in passing.items():
            mem = self._memory(mem_id, doc, meta)
            # Importance is the caller's weight; age is counted once, through recency.
            recency = 2 ** (-_age_days(mem.timestamp, now) / RECENCY_HALF_LIFE_DAYS)
            score = (alpha_similarity * similarity + alpha_importance * mem.importance
                     + alpha_recency * recency) / total
            ranked.append((score, mem))

        ranked.sort(key=lambda row: row[0], reverse=True)
        metrics.inc("search.count")
        metrics.observe("search.latency", time.perf_counter() - started)
        return [mem for _, mem in ranked[:k]]

    def iter_memories(self, user_id: str, include_deleted: bool = False,
                      page_size: int = 500) -> Iterator[Memory]:
        """Yield a user's memories in pages, so large stores don't load into memory at once."""
        conditions = [{"user_id": user_id}]
        if not include_deleted:
            conditions.append({"deleted": False})
        where = _where(*conditions)
        # Offset pagination limits peak memory for large users' histories.
        offset = 0
        while True:
            page = self.backend.get(where=where, include=("documents", "metadatas"),
                                    limit=page_size, offset=offset)
            ids = page["ids"]
            if not ids:
                return
            for mem_id, doc, meta in zip(ids, page["documents"], page["metadatas"]):
                yield self._memory(mem_id, doc, meta)
            if len(ids) < page_size:
                return
            offset += page_size

    def get_all(self, user_id: str, include_deleted: bool = False) -> List[Memory]:
        return list(self.iter_memories(user_id, include_deleted))

    def stats(self, user_id: str, refresh: bool = False) -> dict:
        with self._stats_lock:
            cached = self._stats_cache.get(user_id)
        if refresh or cached is None:
            with self._write_lock:
                # Metadata only: no document text is loaded to count topics.
                found = self.backend.get(
                    where=_where({"user_id": user_id}, {"deleted": False}),
                    include=("metadatas",),
                )
                cached = dict(Counter(m["topic"] for m in found["metadatas"]))
                with self._stats_lock:
                    self._stats_cache[user_id] = cached
        return {"user_id": user_id, "total_memories": sum(cached.values()), "topics": dict(cached)}


class ContextFormatter:
    PREFIX = "Relevant memories:\n"

    def __init__(self, tokenizer: TokenBudget):
        self.tokenizer = tokenizer

    def format(self, memories: Sequence[Memory], token_budget: int = DEFAULT_TOKEN_BUDGET) -> str:
        """Return as many memories as fit in the budget, or "" if none fit."""
        remaining = token_budget - self.tokenizer.count(self.PREFIX)
        lines = []
        for mem in memories:
            line = f"- [{mem.topic}:{mem.memory_type}] {mem.content}\n"
            cost = self.tokenizer.count(line)
            if cost > remaining:
                break
            lines.append(line)
            remaining -= cost
        return self.PREFIX + "".join(lines) if lines else ""


class WorkingMemory:
    """Recent messages with token counts computed once, on insert."""

    def __init__(self, capacity: int = 10, tokenizer: Optional[TokenBudget] = None):
        if capacity < 0:
            raise ValueError("capacity cannot be negative")
        self.tokenizer = tokenizer or TokenBudget()
        # A bounded deque drops the oldest message automatically when full.
        self._items: deque = deque(maxlen=capacity)

    @property
    def messages(self) -> List[str]:
        return [text for text, _ in self._items]

    def add(self, message: str) -> None:
        self._items.append((message, self.tokenizer.count(message)))

    def get_context(self, token_budget: int = 200) -> str:
        if token_budget <= 0:
            return ""
        chosen, used = [], 0
        for text, cost in reversed(self._items):
            # Each message after the first also costs a newline separator.
            extra = cost + (1 if chosen else 0)
            if used + extra > token_budget:
                break
            chosen.append(text)
            used += extra
        return "\n".join(reversed(chosen))


class MemorySession:
    WORKING_HEADER = "Recent conversation:\n"

    def __init__(self, user_id: str, store: SemanticMemoryStore):
        self.user_id = user_id
        self._store = store
        self.formatter = ContextFormatter(store.tokenizer)
        self.working = WorkingMemory(tokenizer=store.tokenizer)

    def add(self, content: str, topic: Optional[str] = None, importance: float = 1.0,
            memory_type: MemoryType = "semantic") -> Optional[Memory]:
        return self._store.add(content, self.user_id, topic, importance, memory_type)

    def add_turn(self, message: str) -> None:
        """Record a chat message in working memory (recent context, not stored long-term)."""
        self.working.add(message)

    def search(self, query: str, topic: Optional[str] = None, memory_type: Optional[MemoryType] = None,
               k: int = 5, **options) -> List[Memory]:
        """Search this user's memories; `options` go to SemanticMemoryStore.search."""
        return self._store.search(query, self.user_id, topic, memory_type, k, **options)

    def get_all(self) -> List[Memory]:
        return self._store.get_all(self.user_id)

    def delete(self, memory_ids: Sequence[str]) -> int:
        return self._store.soft_delete(self.user_id, memory_ids)

    def restore(self, memory_ids: Sequence[str]) -> int:
        return self._store.restore(self.user_id, memory_ids)

    def purge_expired(self) -> int:
        return self._store.purge_expired(self.user_id)

    def stats(self, refresh: bool = False) -> dict:
        return self._store.stats(self.user_id, refresh)

    def build_context(self, query: str, topic: Optional[str] = None,
                      token_budget: int = DEFAULT_TOKEN_BUDGET, include_working: bool = True,
                      cross_topic: bool = True) -> str:
        """Long-term memories plus recent messages, within `token_budget` tokens in total.

        Recent messages get up to a third of the budget; whatever they leave unused goes
        to long-term memories.
        """
        working_section = ""
        if include_working:
            tokenizer = self._store.tokenizer
            header = "\n\n" + self.WORKING_HEADER
            text = self.working.get_context(token_budget // 3 - tokenizer.count(header))
            if text:
                working_section = header + text
        used = self._store.tokenizer.count(working_section) if working_section else 0
        long_context = self.formatter.format(self.search(query, topic, cross_topic=cross_topic),
                                             token_budget - used)
        if not long_context:
            return working_section.lstrip("\n")
        return long_context + working_section


if __name__ == "__main__":
    store = SemanticMemoryStore()
    session = MemorySession("alice", store)
    for text, topic, importance, kind in [
        ("I love my Rolex Submariner", "watches", 1.0, "semantic"),
        ("Yesterday I visited a watch shop in Geneva", "watches", 0.7, "episodic"),
        ("Neapolitan pizza is amazing", "food", 0.8, "semantic"),
    ]:
        session.add(text, topic, importance, kind)
    session.add_turn("User: What watches do I like?")
    print(session.build_context("What watches do I like?"))
    print(session.stats())
    print(metrics.summary())
