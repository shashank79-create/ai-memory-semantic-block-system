"""Tests for semantic_memory.

Store tests use a word-hash embedder and run on three backends: an in-memory stand-in,
BlockIndexBackend (the default) and real ChromaDB (skipped if not installed). Tests
marked "real" use the actual MiniLM and NLI models (skipped if unavailable).

Run: python -m pytest test_semantic_memory.py
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from semantic_memory import (
    BlockIndexBackend,
    ChromaBackend,
    MemorySession,
    SemanticMemoryStore,
    TopicClassifier,
    DEFAULT_TOPICS,
    _stem,
)

DIM = 256


class WordHashEmbedder:
    """Bag-of-words vectors: texts sharing more words are more similar."""

    def get_sentence_embedding_dimension(self):
        return DIM

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"\w+", text.casefold()):
                bucket = int(hashlib.md5(_stem(word).encode()).hexdigest(), 16) % DIM
                out[row, bucket] += 1.0
            norm = np.linalg.norm(out[row])
            if norm:
                out[row] /= norm
        return out


def _matches(meta: dict, where) -> bool:
    if not where:
        return True
    if "$and" in where:
        return all(_matches(meta, cond) for cond in where["$and"])
    assert len(where) == 1, "Chroma allows one top-level key per where dict"
    (key, cond), = where.items()
    value = meta.get(key)
    if isinstance(cond, dict):
        (op, target), = cond.items()
        if op == "$in":
            return value in target
        if op == "$ne":
            return value != target
        if value is None:
            return False
        return {"$lt": value < target, "$lte": value <= target, "$gt": value > target,
                "$gte": value >= target, "$eq": value == target}[op]
    return value == cond


class FakeBackend:
    """Mimics the ChromaBackend interface with Chroma-style result shapes."""

    def __init__(self):
        self.rows = {}  # id -> (embedding, metadata, document); dicts keep insertion order

    def add(self, ids, embeddings, metadatas, documents):
        for i, e, m, d in zip(ids, embeddings, metadatas, documents):
            self.rows[i] = (np.asarray(e, dtype=np.float32), dict(m), d)

    def query(self, embeddings, k, where):
        result = {key: [] for key in ("ids", "documents", "metadatas", "distances")}
        for emb in embeddings:
            emb = np.asarray(emb, dtype=np.float32)
            hits = sorted(
                ((1.0 - float(vec @ emb), i, m, d) for i, (vec, m, d) in self.rows.items()
                 if _matches(m, where)),
                key=lambda h: h[0],
            )[:k]
            result["ids"].append([h[1] for h in hits])
            result["distances"].append([h[0] for h in hits])
            result["metadatas"].append([dict(h[2]) for h in hits])
            result["documents"].append([h[3] for h in hits])
        return result

    def get(self, where=None, ids=None, include=("documents", "metadatas"), limit=None, offset=0):
        selected = [i for i in self.rows if (ids is None or i in ids) and _matches(self.rows[i][1], where)]
        selected = selected[offset:offset + limit if limit is not None else None]
        return {
            "ids": selected,
            "metadatas": [dict(self.rows[i][1]) for i in selected],
            "documents": [self.rows[i][2] for i in selected],
        }

    def delete(self, ids):
        for i in ids:
            self.rows.pop(i, None)

    def update_metadata(self, ids, metadatas):
        for i, m in zip(ids, metadatas):
            vec, old, doc = self.rows[i]
            self.rows[i] = (vec, {**old, **m}, doc)  # Chroma merges metadata on update


class Clock:
    def __init__(self):
        self.now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def __call__(self):
        return self.now

    def advance(self, days: float):
        self.now += timedelta(days=days)


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture(params=["fake", "blocks", "chroma"])
def store(request, tmp_path, clock):
    """Every store test runs on each backend (ChromaDB only when installed)."""
    if request.param == "chroma":
        pytest.importorskip("chromadb")
        backend = ChromaBackend(str(tmp_path / "chroma"))
    elif request.param == "blocks":
        backend = BlockIndexBackend(str(tmp_path / "memories.db"))
    else:
        backend = FakeBackend()
    return SemanticMemoryStore(storage_path=str(tmp_path), embedder=WordHashEmbedder(),
                               backend=backend, clock=clock)


# ----- topic classification ---------------------------------------------------

@pytest.mark.parametrize("text, topic", [
    ("What watches do I like?", "watches"),
    ("I ate two pizzas", "food"),
    ("Booked flights and hotels", "travel"),
    ("New laptops are fast", "technology"),
    ("The weather is nice", "general"),
])
def test_classifier_matches_plurals(text, topic):
    assert TopicClassifier(DEFAULT_TOPICS).classify(text) == topic


def test_stem_leaves_non_plurals_alone():
    assert [_stem(w) for w in ["glass", "bus", "analysis", "watch", "cities"]] == \
        ["glass", "bus", "analysis", "watch", "city"]


def test_add_classifies_topic_when_not_given(store):
    assert store.add("I want new watches", "alice").topic == "watches"


# ----- duplicates -------------------------------------------------------------

def test_restating_refreshes_instead_of_adding(store, clock):
    first = store.add("I love my Rolex Submariner", "alice", importance=0.5, memory_type="episodic")
    clock.advance(80)
    again = store.add("I love my Rolex Submariner", "alice", importance=0.9)
    assert again.id == first.id
    assert again.version == 2
    assert again.importance == 0.9
    assert again.memory_type == "semantic"  # a restated lasting fact stops expiring
    assert again.timestamp == clock.now.isoformat()
    assert len(store.get_all("alice")) == 1


def test_restated_episodic_memory_expiry_restarts(store, clock):
    store.add("Visited a watch shop", "alice", memory_type="episodic")
    clock.advance(80)
    store.add("Visited a watch shop", "alice", memory_type="episodic")
    clock.advance(20)  # 100 days after first mention, 20 after the restatement
    assert store.purge_expired("alice") == 0
    assert len(store.get_all("alice")) == 1


def test_add_many_dedupes_by_default(store):
    out = store.add_many([{"content": "Pizza is great", "importance": 0.4},
                          {"content": "Pizza is great", "importance": 0.8, "memory_type": "episodic"}],
                         "alice")
    assert len(out) == 1
    assert out[0].importance == 0.8           # highest importance kept
    assert out[0].memory_type == "semantic"   # semantic wins over episodic
    assert len(store.get_all("alice")) == 1


def test_add_many_can_skip_dedupe(store):
    store.add_many([{"content": "Pizza is great", "skip_duplicates": False},
                    {"content": "Pizza is great", "skip_duplicates": False}], "alice")
    assert len(store.get_all("alice")) == 2


def test_dedupe_is_per_user(store):
    store.add("Pizza is great", "alice")
    store.add("Pizza is great", "bob")
    assert len(store.get_all("alice")) == 1
    assert len(store.get_all("bob")) == 1


# ----- expiry and deletion ----------------------------------------------------

def test_only_episodic_memories_expire(store, clock):
    store.add("I am allergic to peanuts", "alice", memory_type="semantic")
    store.add("Had lunch at a cafe", "alice", memory_type="episodic")
    clock.advance(91)
    assert store.purge_expired("alice") == 1
    assert [m.content for m in store.get_all("alice")] == ["I am allergic to peanuts"]
    clock.advance(1000)
    assert store.purge_expired("alice") == 0


def test_soft_deleted_memories_purged_after_grace(store, clock):
    mem = store.add("I like sushi", "alice")
    assert store.soft_delete("alice", [mem.id]) == 1
    clock.advance(3)
    assert store.purge_expired("alice") == 0
    assert store.restore("alice", [mem.id]) == 1  # still restorable inside grace
    assert store.soft_delete("alice", [mem.id]) == 1
    clock.advance(8)
    assert store.purge_expired("alice") == 1
    assert store.restore("alice", [mem.id]) == 0
    assert store.get_all("alice", include_deleted=True) == []


def test_other_user_cannot_delete(store):
    mem = store.add("I like sushi", "alice")
    assert store.soft_delete("bob", [mem.id]) == 0
    assert len(store.get_all("alice")) == 1


def test_stats_not_double_counted_on_purge(store, clock):
    keep = store.add("I like sushi", "alice")
    gone = store.add("Rolex watches are nice", "alice")
    assert store.stats("alice")["total_memories"] == 2
    store.soft_delete("alice", [gone.id])
    assert store.stats("alice")["total_memories"] == 1
    clock.advance(8)
    assert store.purge_expired("alice") == 1
    assert store.stats("alice")["topics"] == {keep.topic: 1}
    assert store.stats("alice", refresh=True)["topics"] == {keep.topic: 1}


def test_importance_must_be_between_0_and_1(store):
    with pytest.raises(ValueError):
        store.add("Too important", "alice", importance=2.0)


def test_backfill_adds_missing_fields(store, clock):
    store.backend.add(["old"], [np.ones(DIM) / np.sqrt(DIM)],
                      [{"user_id": "alice", "topic": "general", "timestamp": "2025-01-01T00:00:00+00:00",
                        "importance": 1.0, "memory_type": "episodic", "deleted": False, "version": 1}],
                      ["legacy memory"])
    assert store.backfill_timestamps() == 1
    assert store.purge_expired("alice") == 1  # now has ts, so it can expire


# ----- search -----------------------------------------------------------------

def test_unrelated_query_returns_nothing(store):
    store.add("Neapolitan pizza is amazing", "alice")
    assert store.search("rolex submariner bracelet", "alice") == []
    assert [m.content for m in store.search("amazing neapolitan pizza", "alice")] == \
        ["Neapolitan pizza is amazing"]


def test_search_respects_topic_and_user(store):
    store.add("Rolex watches are amazing", "alice", topic="watches")
    store.add("Pizza is amazing", "alice", topic="food")
    store.add("Rolex watches are amazing", "bob", topic="watches")
    results = store.search("amazing", "alice", topic="food", min_similarity=0.0)
    assert [m.content for m in results] == ["Pizza is amazing"]


def test_recency_breaks_ties_between_equal_matches(store, clock):
    store.add("rolex watch old", "alice", skip_duplicates=False)
    clock.advance(60)
    store.add("rolex watch new", "alice", skip_duplicates=False)
    results = store.search("rolex watch", "alice")
    assert [m.content for m in results] == ["rolex watch new", "rolex watch old"]


def test_importance_above_one_in_old_records_is_clamped(store):
    store.backend.add(["old"], [WordHashEmbedder().encode(["pizza"])[0]],
                      [{"user_id": "alice", "topic": "food", "timestamp": "2026-01-01T00:00:00+00:00",
                        "ts": 0.0, "importance": 5.0, "memory_type": "semantic", "deleted": False,
                        "deleted_ts": 0.0, "version": 1}], ["pizza"])
    assert store.search("pizza", "alice")[0].importance == 1.0


def test_query_topic_block_finds_low_similarity_memories(store):
    store.add("Submariner bezel is ceramic", "alice", topic="watches")  # shares no words
    store.add("Pizza dough needs time", "alice", topic="food")
    # Zero similarity: found only through the topic block, with its cutoff lowered to 0.
    found = [m.content for m in store.search("which watch next", "alice", topic_min_similarity=0.0)]
    assert found == ["Submariner bezel is ceramic"]
    assert store.search("which watch next", "alice") == []  # default block cutoff is 0.1
    assert store.search("which watch next", "alice", use_query_topic=False,
                        topic_min_similarity=0.0) == []


def test_relative_cutoff_drops_weak_matches_but_spares_topic_block(store):
    store.add("rolex submariner bezel ceramic dial", "alice", topic="watches")
    store.add("rolex dial", "alice", topic="watches")
    store.add("rolex shop", "alice", topic="travel")
    query = "rolex submariner bezel ceramic watch"
    found = {m.content for m in store.search(query, "alice", min_similarity=0.2)}
    # "rolex shop" (other topic) is far below the best match and is dropped;
    # "rolex dial" is as weak but in the query's watches block, so it stays.
    assert found == {"rolex submariner bezel ceramic dial", "rolex dial"}
    found_no_gap = {m.content for m in store.search(query, "alice", min_similarity=0.2, max_gap=None)}
    assert "rolex shop" in found_no_gap


def test_general_query_uses_normal_cutoff(store):
    store.add("Submariner bezel is ceramic", "alice", topic="watches")
    assert store.search("tell me something", "alice") == []


# ----- context building ---------------------------------------------------------

def test_build_context_includes_working_memory_within_budget(store):
    session = MemorySession("alice", store)
    session.add("I love my Rolex Submariner", topic="watches")
    session.add_turn("User: which rolex should I buy?")
    context = session.build_context("rolex submariner", token_budget=120)
    assert "Relevant memories:" in context
    assert "I love my Rolex Submariner" in context
    assert "Recent conversation:\nUser: which rolex should I buy?" in context
    assert store.tokenizer.count(context) <= 120


def test_build_context_without_relevant_memories(store):
    session = MemorySession("alice", store)
    session.add("Pizza is amazing")
    session.add_turn("User: hello")
    assert session.build_context("rolex submariner") == "Recent conversation:\nUser: hello"


def test_build_context_respects_small_budgets(store):
    session = MemorySession("alice", store)
    for n in range(20):
        store.add(f"rolex fact number {n} " + "detail " * 10, "alice", skip_duplicates=False)
        session.add_turn(f"message {n} " + "words " * 10)
    for budget in (0, 5, 30, 100, 300):
        context = session.build_context("rolex fact detail", token_budget=budget)
        assert store.tokenizer.count(context) <= max(budget, 0)


# ----- real embedding model -----------------------------------------------------

@pytest.fixture(scope="module")
def minilm():
    st = pytest.importorskip("sentence_transformers")
    try:
        return st.SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as exc:  # e.g. offline with no cached model
        pytest.skip(f"MiniLM unavailable: {exc}")


@pytest.fixture
def real_store(tmp_path, clock, minilm):
    return SemanticMemoryStore(storage_path=str(tmp_path), embedder=minilm, clock=clock)


def test_real_model_dedupes_rewording_but_keeps_negation(real_store):
    real_store.add("I am allergic to peanuts", "alice")
    real_store.add("I am allergic to peanuts.", "alice")
    real_store.add("I am not allergic to peanuts", "alice")
    assert sorted(m.content for m in real_store.get_all("alice")) == \
        ["I am allergic to peanuts", "I am not allergic to peanuts"]


@pytest.mark.parametrize("query, expected", [
    ("What watches do I like?", "I love my Rolex Submariner"),
    ("Can I eat peanut butter?", "I am allergic to peanuts"),
    ("What computer do I use?", "My laptop is a ThinkPad running Linux"),
    ("Where am I travelling?", "I'm planning a beach vacation in Goa next month"),
])
def test_real_model_finds_related_memories(real_store, query, expected):
    for text in ["I love my Rolex Submariner", "Neapolitan pizza is amazing",
                 "I am allergic to peanuts", "My laptop is a ThinkPad running Linux",
                 "I'm planning a beach vacation in Goa next month"]:
        real_store.add(text, "alice")
    assert expected in [m.content for m in real_store.search(query, "alice", k=3)]


def test_real_model_returns_nothing_for_unrelated_query(real_store):
    for text in ["I love my Rolex Submariner", "Neapolitan pizza is amazing",
                 "My laptop is a ThinkPad running Linux"]:
        real_store.add(text, "alice")
    assert real_store.search("Tell me a joke", "alice") == []
    assert real_store.search("What's the capital of France?", "alice") == []


def test_real_model_brings_in_topic_block_memories(real_store):
    for text in ["I love my Rolex Submariner", "I prefer analog watches over smartwatches",
                 "I am allergic to peanuts", "Neapolitan pizza is my favourite food"]:
        real_store.add(text, "alice")
    watch = [m.content for m in real_store.search("Which watch should I buy next?", "alice")]
    assert {"I love my Rolex Submariner", "I prefer analog watches over smartwatches"} <= set(watch)
    assert "I am allergic to peanuts" not in watch
    dinner = [m.content for m in real_store.search("Suggest a dinner recipe for tonight", "alice")]
    assert {"I am allergic to peanuts", "Neapolitan pizza is my favourite food"} <= set(dinner)
    assert not any("watch" in c.lower() for c in dinner)


# ----- contradictions -------------------------------------------------------------

class NotChecker:
    """Fake NLI: a pair contradicts when exactly one side contains "not"."""

    def __init__(self):
        self.pairs = []

    def __call__(self, pairs):
        self.pairs.extend(pairs)
        return [("not" in old.split()) != ("not" in new.split()) for old, new in pairs]


@pytest.fixture
def checked_store(store):
    store.contradiction_checker = NotChecker()
    return store


def test_contradiction_replaces_old_memory(checked_store):
    old = checked_store.add("my favourite watch brand is rolex", "alice")
    new = checked_store.add("my favourite watch brand is not rolex", "alice")
    assert new.replaced == ((old.id, old.content),)
    assert [m.content for m in checked_store.get_all("alice")] == [new.content]
    assert checked_store.stats("alice")["total_memories"] == 1
    # The replaced memory can be brought back during the grace period.
    assert checked_store.restore("alice", [old.id]) == 1
    assert len(checked_store.get_all("alice")) == 2


def test_dissimilar_memories_are_not_checked(checked_store):
    checked_store.add("pizza is great", "alice")
    checked_store.add("rolex is not cheap", "alice")
    assert checked_store.contradiction_checker.pairs == []
    assert len(checked_store.get_all("alice")) == 2


def test_compatible_similar_memories_are_kept(checked_store):
    checked_store.add("my favourite watch brand is rolex", "alice")
    mem = checked_store.add("my favourite watch brand is rolex and omega", "alice")
    assert checked_store.contradiction_checker.pairs  # was checked
    assert mem.replaced == ()
    assert len(checked_store.get_all("alice")) == 2


def test_contradictions_off_by_default(store):
    store.add("my favourite watch brand is rolex", "alice")
    mem = store.add("my favourite watch brand is not rolex", "alice")
    assert mem.replaced == ()
    assert len(store.get_all("alice")) == 2


def test_contradiction_check_is_per_user(checked_store):
    checked_store.add("my favourite watch brand is rolex", "bob")
    mem = checked_store.add("my favourite watch brand is not rolex", "alice")
    assert mem.replaced == ()
    assert len(checked_store.get_all("bob")) == 1


@pytest.fixture(scope="module")
def nli():
    pytest.importorskip("sentence_transformers")
    from semantic_memory import NLIContradictionChecker
    try:
        return NLIContradictionChecker()
    except Exception as exc:
        pytest.skip(f"NLI model unavailable: {exc}")


@pytest.mark.parametrize("old, new, replaced", [
    ("I am allergic to peanuts", "I am not allergic to peanuts", True),
    ("I love my Rolex Submariner", "I hate my Rolex Submariner", True),
    ("My favourite colour is blue", "My favourite colour is green", True),
    ("I have a dog", "I have a cat", False),
    ("I love pizza", "I love sushi", False),
    ("I work at Google", "I work at Google as an engineer", False),
])
def test_real_nli_contradictions(real_store, nli, old, new, replaced):
    real_store.contradiction_checker = nli
    first = real_store.add(old, "alice")
    second = real_store.add(new, "alice")
    assert bool(second.replaced) == replaced
    expected = [new] if replaced else sorted([old, new])
    assert sorted(m.content for m in real_store.get_all("alice")) == expected
    assert first.id not in [m.id for m in real_store.get_all("alice")] or not replaced



# ----- topic-block search and the block index backend -----------------------------

def test_block_only_search_skips_other_topics(store):
    store.add("rolex submariner is great", "alice", topic="watches")
    store.add("rolex submariner shop visit", "alice", topic="travel")
    store.add("rolex submariner notes", "alice", topic="general")
    found = {m.topic for m in store.search("rolex submariner watch", "alice", cross_topic=False)}
    assert found == {"watches", "general"}
    found_all = {m.topic for m in store.search("rolex submariner watch", "alice")}
    assert found_all == {"watches", "travel", "general"}


def test_block_only_search_on_general_query_searches_everything(store):
    store.add("rolex submariner is great", "alice", topic="watches")
    found = store.search("rolex submariner", "alice", cross_topic=False, use_query_topic=False)
    assert [m.topic for m in found] == ["watches"]


def blocks_store(path, clock):
    return SemanticMemoryStore(storage_path=str(path), embedder=WordHashEmbedder(),
                               backend=BlockIndexBackend(str(path / "memories.db")), clock=clock)


def test_block_backend_persists_across_restarts(tmp_path, clock):
    first = blocks_store(tmp_path, clock)
    keep = first.add("I love my Rolex Submariner", "alice")
    gone = first.add("pizza is great", "alice")
    first.soft_delete("alice", [gone.id])
    first.backend.close()

    second = blocks_store(tmp_path, clock)
    assert [m.id for m in second.get_all("alice")] == [keep.id]
    assert [m.id for m in second.search("rolex submariner", "alice")] == [keep.id]
    assert second.restore("alice", [gone.id]) == 1
    second.backend.close()


def test_block_backend_delete_and_topic_move(tmp_path):
    backend = BlockIndexBackend(str(tmp_path / "memories.db"))
    vecs = np.eye(4, dtype=np.float32)
    meta = lambda topic: {"user_id": "u", "topic": topic, "deleted": False}
    backend.add(["a", "b", "c"], vecs[:3], [meta("x"), meta("x"), meta("x")], ["A", "B", "C"])
    backend.delete(["a"])  # moves "c" into "a"'s slot
    hit = backend.query([vecs[2]], 1, {"user_id": "u"})
    assert hit["ids"] == [["c"]] and hit["distances"][0][0] == pytest.approx(0.0)
    backend.update_metadata(["b"], [{"topic": "y"}])
    assert backend.query([vecs[1]], 5, {"$and": [{"user_id": "u"}, {"topic": "y"}]})["ids"] == [["b"]]
    assert backend.query([vecs[1]], 5, {"$and": [{"user_id": "u"}, {"topic": "x"}]})["ids"] == [["c"]]
    assert backend.get(where={"topic": "y"})["metadatas"] == [{"user_id": "u", "topic": "y", "deleted": False}]
    with pytest.raises(Exception):
        backend.add(["b"], vecs[:1], [meta("x")], ["dup"])
    backend.close()


def test_block_backend_filters_widen_until_enough_results(tmp_path):
    backend = BlockIndexBackend(str(tmp_path / "memories.db"))
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(200, 8)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    # Only every 50th memory passes the filter, so the best-ranked ones mostly fail it.
    metas = [{"user_id": "u", "topic": "t", "keep": i % 50 == 0} for i in range(200)]
    backend.add([f"m{i}" for i in range(200)], vecs, metas, [str(i) for i in range(200)])
    hits = backend.query([vecs[1]], 3, {"$and": [{"user_id": "u"}, {"keep": True}]})["ids"][0]
    expected = sorted((i for i in range(0, 200, 50)), key=lambda i: -float(vecs[i] @ vecs[1]))[:3]
    assert hits == [f"m{i}" for i in expected]
    backend.close()
