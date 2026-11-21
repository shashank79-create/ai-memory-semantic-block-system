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
DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# BGE retrieval models expect this prefix on search queries (not on stored memories).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_POOL = 50   # candidates from embeddings and BM25 that the reranker re-scores
RRF_K = 60         # reciprocal rank fusion constant
# Hybrid search settings (eval/retrieval_eval.py --sweep-hybrid on eval/dataset.json).
# The reranker (trained on web search) gives personal memories tiny, poorly calibrated
# relevance, so cutting individual memories by an absolute relevance loses many relevant
# ones. Instead:
# - Answer gate: a question gets memories only if its best embedding similarity or its
#   best reranker logit clears these bars (unrelated questions like "Tell me a joke"
#   clear neither). None turns the gate off.
DEFAULT_GATE_SIMILARITY: Optional[float] = 0.6
DEFAULT_GATE_LOGIT: Optional[float] = -9.5
# - Relative cutoff: drop memories whose reranker logit is more than this below the best.
DEFAULT_RELEVANCE_GAP: Optional[float] = 2.0
DEFAULT_MIN_RELEVANCE = 0.0  # absolute cutoff on reranker relevance (0-1); off by default
# Cosine similarity at or above this counts as a duplicate. Measured with BGE-small:
# punctuation/case changes and "I really love X" score 0.97-1.0, while "I am allergic to
# peanuts" vs "I am not allergic to peanuts" scores 0.923 and "I work at Google" vs
# "I work at Google as an engineer" 0.934. Contradictions below this are stored separately.
DUPLICATE_THRESHOLD = 0.96
# Classic search (no reranker) settings below were chosen with MiniLM embeddings on
# eval/dataset.json and checked on eval/test_set.json; they assume MiniLM similarities.
# With MiniLM, 36 memories put unrelated questions above 0.2 ("Tell me a joke" vs
# "I'm vegetarian"), so 0.3 is used.
DEFAULT_MIN_SIMILARITY = 0.3
# Memories in the same topic block as the query use this lower cutoff instead. MiniLM
# scores "Which watch should I buy?" vs "I love my Rolex Submariner" at only 0.15, and
# "Suggest a dinner recipe" vs "I am allergic to peanuts" at 0.11; the shared topic
# is what marks them as relevant.
DEFAULT_TOPIC_MIN_SIMILARITY = 0.1
# Results less similar than (best match - DEFAULT_MAX_GAP) are dropped, except memories
# in the query's topic block. None turns it off.
DEFAULT_MAX_GAP: Optional[float] = 0.15
# Only memories at least this similar to a new one are checked for contradictions,
# because the NLI model also calls compatible facts contradictions ("I have a dog" vs
# "I have a cat"). Measured with BGE-small, no cutoff separates them cleanly: real
# contradictions score 0.82-0.92 ("I love my Rolex" vs "I hate my Rolex" 0.815,
# "...allergic to peanuts" vs "...not allergic..." 0.923), compatible facts 0.76-0.88
# ("dog" vs "cat" 0.869, "Paris" vs "Rome" 0.882). 0.90 is chosen to avoid deleting
# compatible facts, at the cost of missing many contradictions.
CONTRADICTION_MIN_SIMILARITY = 0.90
# EmbeddingTopicClassifier: below this similarity to every topic's examples, a text is
# "general". 0.35 gave the best topic accuracy on eval/dataset.json (78%; keywords: 75%).
DEFAULT_TOPIC_CLASSIFIER_MIN_SIMILARITY = 0.35
CONTRADICTION_CANDIDATES = 5    # stored memories checked per new memory
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


DEFAULT_TOPIC_EXAMPLES = {
    "watches": [
        "my wristwatch", "a luxury watch brand", "collecting watches", "mechanical and automatic watches",
        "a smartwatch or fitness tracker on my wrist", "buying a new watch", "a vintage timepiece",
        "watch dials, straps and movements",
    ],
    "food": [
        "food I like to eat", "my diet and eating habits", "a food allergy or intolerance",
        "cooking a recipe", "drinks like coffee or tea", "eating at a restaurant",
        "a vegetarian or vegan diet", "desserts and sweets", "healthy eating and nutrition",
        "breakfast, lunch and dinner", "a dish my family makes",
    ],
    "travel": [
        "a trip I am planning", "countries and cities I have visited", "flights and airports",
        "hotels and places to stay", "passport and visa", "a holiday destination",
        "sightseeing abroad", "living in another country",
    ],
    "technology": [
        "my laptop or computer", "my smartphone", "programming and writing code",
        "learning a programming language", "software and apps", "building an AI project",
        "gadgets and electronics", "fixing a broken device",
    ],
}


class EmbeddingTopicClassifier:
    """Picks the topic whose example phrases are most similar in meaning to the text.

    Each topic is described by a few example phrases. A text goes to the topic with the
    most similar example, or to "general" when no example reaches `min_similarity`.
    Unlike keyword lists this catches wording the lists never mention ("timepiece",
    "lactose intolerant"), and adding a topic needs only a few phrases.

    It sorts topics slightly more accurately than the keyword lists but gives worse search
    results with topic blocks (see SemanticMemoryStore's `topic_classifier`), so it is
    not the default.
    """

    def __init__(self, embed: Callable[[Sequence[str]], np.ndarray],
                 examples: Optional[Dict[str, Sequence[str]]] = None,
                 min_similarity: float = DEFAULT_TOPIC_CLASSIFIER_MIN_SIMILARITY):
        self._embed = embed
        self.min_similarity = min_similarity
        examples = examples or DEFAULT_TOPIC_EXAMPLES
        self._labels = [topic for topic, phrases in examples.items() for _ in phrases]
        self._vectors = embed([p for phrases in examples.values() for p in phrases])

    def scores(self, text: str) -> Dict[str, float]:
        """Best similarity to each topic's examples."""
        sims = self._vectors @ self._embed([text])[0]
        best: Dict[str, float] = {}
        for label, sim in zip(self._labels, sims):
            best[label] = max(best.get(label, -1.0), float(sim))
        return best

    def classify(self, text: str) -> str:
        scores = self.scores(text)
        topic = max(scores, key=scores.get)
        return topic if scores[topic] >= self.min_similarity else "general"


def _bm25_tokens(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


class BM25Index:
    """Incremental BM25 keyword index over one user's memories (k1=1.5, b=0.75)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self._postings: Dict[str, Dict[str, int]] = {}
        self._terms: Dict[str, List[str]] = {}
        self._lengths: Dict[str, int] = {}
        self._total = 0

    def __len__(self) -> int:
        return len(self._lengths)

    def add(self, mem_id: str, text: str) -> None:
        if mem_id in self._lengths:
            return
        counts = Counter(_bm25_tokens(text))
        for term, tf in counts.items():
            self._postings.setdefault(term, {})[mem_id] = tf
        self._terms[mem_id] = list(counts)
        self._lengths[mem_id] = sum(counts.values())
        self._total += self._lengths[mem_id]

    def remove(self, mem_id: str) -> None:
        length = self._lengths.pop(mem_id, None)
        if length is None:
            return
        self._total -= length
        for term in self._terms.pop(mem_id):
            docs = self._postings[term]
            docs.pop(mem_id, None)
            if not docs:
                del self._postings[term]

    def top(self, query: str, n: int) -> List[Tuple[str, float]]:
        if not self._lengths or n <= 0:
            return []
        count = len(self._lengths)
        avg = (self._total / count) or 1.0
        scores: Dict[str, float] = {}
        for term in set(_bm25_tokens(query)):
            docs = self._postings.get(term)
            if not docs:
                continue
            idf = math.log(1 + (count - len(docs) + 0.5) / (len(docs) + 0.5))
            for mem_id, tf in docs.items():
                norm = self.k1 * (1 - self.b + self.b * self._lengths[mem_id] / avg)
                scores[mem_id] = scores.get(mem_id, 0.0) + idf * tf * (self.k1 + 1) / (tf + norm)
        return sorted(scores.items(), key=lambda kv: -kv[1])[:n]


class CrossEncoderReranker:
    """Scores (query, memory) pairs with a cross-encoder; returns relevance in 0-1.

    A cross-encoder reads the question and the memory together, which ranks far better
    than comparing separate embeddings (+11.8 points Recall@10 on LoCoMo).
    """

    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL):
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
        except Exception as exc:
            raise EmbeddingError(f"Failed to load reranker {model_name!r}: {exc}") from exc

    def __call__(self, query: str, documents: Sequence[str]) -> np.ndarray:
        if not len(documents):
            return np.zeros(0)
        logits = np.asarray(self.model.predict([(query, d) for d in documents]), dtype=np.float64)
        return 1.0 / (1.0 + np.exp(-logits.reshape(-1)))


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
    def __init__(self, storage_path: str = "./memory", embed_model: str = DEFAULT_EMBED_MODEL,
                 topic_config: Optional[Dict[str, Sequence[str]]] = None, ttl_days: int = 90,
                 deleted_grace_days: int = DEFAULT_DELETED_GRACE_DAYS,
                 cache_size: int = 1000, query_cache_size: int = 100, embedder=None, backend=None,
                 clock: Optional[Callable[[], datetime]] = None,
                 contradiction_checker: Optional[Callable[[Sequence[Tuple[str, str]]], List[bool]]] = None,
                 contradiction_min_similarity: float = CONTRADICTION_MIN_SIMILARITY,
                 topic_classifier="keyword", reranker="default",
                 query_prefix: Optional[str] = None):
        """Search mode depends on `reranker`:
        - "default" or a reranker object: hybrid search. Embedding and BM25 keyword
          candidates are merged, then re-scored by the reranker (see `search`).
        - None: classic search. Embedding similarity with topic blocks and cutoffs tuned
          for MiniLM (the original design; kept for comparison and for tests).

        `query_prefix` is prepended to search queries before embedding; it defaults to the
        BGE prefix when the BGE model is loaded by name.

        `contradiction_checker` (for example NLIContradictionChecker()) turns on replacing
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
            if query_prefix is None and "bge" in embed_model.lower():
                query_prefix = BGE_QUERY_PREFIX
        else:
            model_name = type(embedder).__name__
        self.query_prefix = query_prefix or ""
        if reranker == "default":
            reranker = CrossEncoderReranker()
        self.reranker = reranker
        self._bm25: Dict[str, BM25Index] = {}   # per user, built on first hybrid search
        self._bm25_lock = threading.RLock()
        self.embedder = embedder
        # Newer sentence-transformers renamed the method; support both names.
        dim_fn = (getattr(self.embedder, "get_embedding_dimension", None)
                  or getattr(self.embedder, "get_sentence_embedding_dimension", None))
        dim = dim_fn() if dim_fn else "unknown"
        self.model_key = f"{model_name}:{dim}"
        self.backend = backend or BlockIndexBackend(str(self.storage_path / "memories.db"))
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
        # "keyword" (default) uses keyword lists (`topic_config`); "embedding" picks topics by
        # meaning. Keywords are the default because they gave better search results in
        # eval/retrieval_eval.py (74% vs 67% overall on the tuning set, after re-tuning the
        # search settings for each). Any object with classify(text) -> topic also works.
        if topic_classifier == "embedding":
            # Shares the query cache, so a search embeds its question only once.
            self.classifier = EmbeddingTopicClassifier(
                lambda texts: self._embed_many(list(texts), self.query_cache))
        elif topic_classifier == "keyword":
            self.classifier = TopicClassifier(topic_config or DEFAULT_TOPICS)
        else:
            self.classifier = topic_classifier

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
        return self._embed_many([self.query_prefix + text], self.query_cache)[0]

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

    def _bm25_for(self, user_id: str) -> BM25Index:
        """The user's BM25 index, built from the store on first use, then kept in step."""
        with self._bm25_lock:
            index = self._bm25.get(user_id)
            if index is None:
                index = BM25Index()
                found = self.backend.get(where=_where({"user_id": user_id}, {"deleted": False}),
                                         include=("documents",))
                for mem_id, doc in zip(found["ids"], found["documents"]):
                    index.add(mem_id, doc)
                self._bm25[user_id] = index
            return index

    def _bm25_add(self, user_id: str, pairs) -> None:
        with self._bm25_lock:
            index = self._bm25.get(user_id)
            if index is not None:  # not built yet: it will load these on first use
                for mem_id, doc in pairs:
                    index.add(mem_id, doc)

    def _bm25_remove(self, user_id: str, ids: Sequence[str]) -> None:
        with self._bm25_lock:
            index = self._bm25.get(user_id)
            if index is not None:
                for mem_id in ids:
                    index.remove(mem_id)

    def _supersede(self, user_id: str, old: Sequence[Tuple[str, str, dict]], new_id: str) -> None:
        """Soft-delete memories contradicted by `new_id`, recording which memory replaced them."""
        now_ts = self._now().timestamp()
        self.backend.update_metadata(
            [mem_id for mem_id, _, _ in old],
            [{**meta, "deleted": True, "deleted_ts": now_ts, "superseded_by": new_id,
              "version": int(meta.get("version", 1)) + 1} for _, _, meta in old],
        )
        self._bump_stats(user_id, [meta["topic"] for _, _, meta in old], -1)
        self._bm25_remove(user_id, [mem_id for mem_id, _, _ in old])
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
                self._bm25_add(user_id, zip(ids, docs))
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
                ids=ids, include=("documents", "metadatas"),
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
            if deleted:
                self._bm25_remove(user_id, [i for i, _ in rows])
            else:
                self._bm25_add(user_id, zip(found["ids"], found["documents"]))
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
            self._bm25_remove(user_id, ids)
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
               max_gap: Optional[float] = DEFAULT_MAX_GAP,
               min_relevance: Optional[float] = None,
               gate_similarity: Optional[float] = DEFAULT_GATE_SIMILARITY,
               gate_logit: Optional[float] = DEFAULT_GATE_LOGIT,
               relevance_gap: Optional[float] = DEFAULT_RELEVANCE_GAP) -> List[Memory]:
        """Return up to k memories, best first.

        Hybrid search (when the store has a reranker): the top RERANK_POOL memories by
        embedding similarity and by BM25 keyword score are merged with reciprocal rank
        fusion, then re-scored by the reranker. If neither the best embedding similarity
        reaches `gate_similarity` nor the best reranker logit reaches `gate_logit`, the
        question is treated as unrelated to the user and nothing is returned. Memories
        more than `relevance_gap` logits below the best, or below `min_relevance`, are
        dropped. The score combines relevance, importance and recency with the alpha weights. Topic blocks and the
        similarity cutoffs below are not used; `topic` and `memory_type` still filter.

        Classic search (no reranker): results below `min_similarity` are dropped, so an
        unrelated query returns nothing rather than filler. When `topic` is not given and `use_query_topic` is on, the
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
        if self.reranker is not None:
            query_topic = None
            if topic is None and use_query_topic:
                detected = self.classifier.classify(query)
                query_topic = detected if detected != "general" else None
            results = self._hybrid_search(
                query, user_id, conditions, k, weights,
                DEFAULT_MIN_RELEVANCE if min_relevance is None else min_relevance,
                gate_similarity, gate_logit, relevance_gap, query_topic)
            metrics.inc("search.count")
            metrics.observe("search.latency", time.perf_counter() - started)
            return results
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

    def _hybrid_search(self, query: str, user_id: str, conditions: List[dict], k: int,
                       weights: Tuple[float, float, float], min_relevance: float,
                       gate_similarity: Optional[float], gate_logit: Optional[float],
                       relevance_gap: Optional[float], query_topic: Optional[str]) -> List[Memory]:
        where = _where(*conditions)
        embedding = [self._embed_query(query).tolist()]
        dense = self.backend.query(embedding, RERANK_POOL, where)
        docs: Dict[str, Tuple[str, dict]] = {}
        dense_ids = []
        best_similarity = max((_similarity_from_distance(d) for d in _row(dense, "distances", 0)),
                              default=0.0)
        for mem_id, doc, meta in zip(_row(dense, "ids", 0), _row(dense, "documents", 0),
                                     _row(dense, "metadatas", 0)):
            dense_ids.append(mem_id)
            docs[mem_id] = (doc, meta)

        # BM25 covers all of the user's active memories; fetch extra in case filters drop some.
        keyword_ids = [i for i, _ in self._bm25_for(user_id).top(query, RERANK_POOL * 4)]
        if keyword_ids:
            found = self.backend.get(where=where, ids=keyword_ids, include=("documents", "metadatas"))
            allowed = {i: (d, m) for i, d, m in zip(found["ids"], found["documents"], found["metadatas"])}
            keyword_ids = [i for i in keyword_ids if i in allowed][:RERANK_POOL]
            docs.update({i: allowed[i] for i in keyword_ids})

        # Topic block: memories filed under the question's topic. They are re-scored like
        # the rest but skip the answer gate and relative cutoff, because the reranker and
        # embeddings score them low even when they matter ("Suggest a dinner recipe" vs
        # "I am allergic to peanuts").
        block_ids: List[str] = []
        if query_topic is not None:
            block = self.backend.query(embedding, RERANK_POOL, _where(*conditions, {"topic": query_topic}))
            for mem_id, doc, meta in zip(_row(block, "ids", 0), _row(block, "documents", 0),
                                         _row(block, "metadatas", 0)):
                block_ids.append(mem_id)
                docs[mem_id] = (doc, meta)

        fused: Dict[str, float] = {}
        for ranking in (dense_ids, keyword_ids):
            for rank, mem_id in enumerate(ranking):
                fused[mem_id] = fused.get(mem_id, 0.0) + 1.0 / (RRF_K + rank + 1)
        pool = sorted(fused, key=lambda i: -fused[i])[:RERANK_POOL]
        pool += [i for i in block_ids if i not in set(pool)]
        if not pool:
            return []
        in_block = np.array([i in set(block_ids) for i in pool])
        relevance = np.clip(np.asarray(self.reranker(query, [docs[i][0] for i in pool]), dtype=np.float64),
                            1e-12, 1 - 1e-12)
        logits = np.log(relevance / (1 - relevance))
        gated = (gate_similarity is not None and gate_logit is not None
                 and best_similarity < gate_similarity and logits.max() < gate_logit)
        if gated:
            metrics.inc("search.gated")
        keep = (relevance >= min_relevance) & (not gated)
        if relevance_gap is not None:
            keep &= logits >= logits.max() - relevance_gap
        keep |= in_block

        alpha_relevance, alpha_importance, alpha_recency = weights
        total = sum(weights)
        now = self._now()
        ranked = []
        for mem_id, rel, kept in zip(pool, relevance, keep):
            if not kept:
                continue
            mem = self._memory(mem_id, *docs[mem_id])
            recency = 2 ** (-_age_days(mem.timestamp, now) / RECENCY_HALF_LIFE_DAYS)
            score = (alpha_relevance * float(rel) + alpha_importance * mem.importance
                     + alpha_recency * recency) / total
            ranked.append((score, mem))
        ranked.sort(key=lambda row: row[0], reverse=True)
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
