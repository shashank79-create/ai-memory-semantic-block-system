"""Measure search time as the number of memories grows.

Compares three ways of finding memories for a question:
- keyword scan: the original notebook's approach (loop over every memory in the topic,
  count shared words)
- brute-force embeddings: exact cosine similarity against every stored vector (numpy)
- semantic_memory: SemanticMemoryStore.search on ChromaDB's vector index

Run: python benchmark.py            (sizes 100 1000 5000 10000)
     python benchmark.py 100 1000   (custom sizes)
"""
from __future__ import annotations

import platform
import random
import statistics
import sys
import tempfile
import time

import numpy as np
from sentence_transformers import SentenceTransformer

from semantic_memory import SemanticMemoryStore

SUBJECTS = {
    "watches": ["Rolex", "Omega", "Seiko", "Casio", "Tudor", "Longines", "analog watch", "smartwatch"],
    "food": ["pizza", "sushi", "pasta", "biryani", "tacos", "ramen", "burgers", "dosa"],
    "travel": ["Goa", "Paris", "Tokyo", "a beach trip", "Manali", "a hotel in Rome", "Bali", "Dubai"],
    "technology": ["my laptop", "Python", "Android", "my iPhone", "Linux", "AI tools", "my code", "gadgets"],
}
VERBS = ["I love", "I like", "I am curious about", "I bought", "I read about", "I talked about",
         "I want to try", "I dislike", "I compared", "I asked a friend about"]
TAILS = ["yesterday", "last week", "on a Sunday", "with my brother", "after work", "for the first time",
         "again", "at a store", "online", "during lunch", "this morning", "with colleagues"]
QUERIES = ["What watches do I like?", "Which food do I enjoy?", "Where do I want to travel?",
           "What do I use for coding?", "Do I like Rolex?", "What did I eat last week?",
           "Tell me about my trips", "Which phone do I have?"]


def make_memories(n: int, seed: int = 0) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        topic = rng.choice(list(SUBJECTS))
        out.append((topic, f"{rng.choice(VERBS)} {rng.choice(SUBJECTS[topic])} {rng.choice(TAILS)} (note {i})"))
    return out


def keyword_search(memories, query, topic, limit=5):
    """The original notebook's SimpleMemory.search, without the user check."""
    words = set(query.lower().split())
    scored = []
    for mem_topic, text in memories:
        if mem_topic != topic:
            continue
        lowered = text.lower()
        score = sum(1 for w in words if w in lowered)
        if score:
            scored.append((score, text))
    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[:limit]


def timed_ms(fn, repeats):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def main(sizes):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    largest = make_memories(max(sizes))
    print(f"Embedding {len(largest)} memories (one-off setup)...", flush=True)
    vectors = model.encode([t for _, t in largest], normalize_embeddings=True,
                           convert_to_numpy=True, batch_size=256).astype(np.float32)
    query_vecs = model.encode(QUERIES, normalize_embeddings=True, convert_to_numpy=True)

    embed_ms = timed_ms(lambda: model.encode([f"fresh query {random.random()}"],
                                             normalize_embeddings=True), 20)

    rows = []
    for n in sizes:
        memories, vecs = largest[:n], vectors[:n]
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            store = SemanticMemoryStore(storage_path=tmp, embedder=model)
            # Insert pre-computed vectors directly: this measures search, not insertion.
            for start in range(0, n, 1000):
                chunk = range(start, min(start + 1000, n))
                store.backend.add(
                    [f"m{i}" for i in chunk], [vecs[i].tolist() for i in chunk],
                    [{"user_id": "u", "topic": memories[i][0], "timestamp": "2026-01-01T00:00:00+00:00",
                      "ts": 0.0, "importance": 1.0, "memory_type": "semantic", "deleted": False,
                      "deleted_ts": 0.0, "version": 1} for i in chunk],
                    [memories[i][1] for i in chunk])
            for q in QUERIES:  # warm the query cache and Chroma's index
                store.search(q, "u")

            kw = statistics.median(timed_ms(
                lambda q=q: keyword_search(memories, q, store.classifier.classify(q)), 20) for q in QUERIES)
            brute = statistics.median(timed_ms(
                lambda qv=qv: np.argpartition(-(vecs @ qv), 5)[:5], 20) for qv in query_vecs)
            ann = statistics.median(timed_ms(lambda q=q: store.search(q, "u"), 20) for q in QUERIES)
            rows.append((n, kw, brute, ann))
            print(f"  {n:>6} memories: keyword {kw:.2f} ms, brute-force {brute:.2f} ms, "
                  f"semantic_memory {ann:.2f} ms", flush=True)
            del store

    print(f"\nMachine: {platform.processor() or platform.machine()}, Python {platform.python_version()}")
    print(f"Embedding one new question with MiniLM: {embed_ms:.1f} ms (not included below)\n")
    print("| Memories | Keyword scan (original) | Brute-force embeddings | semantic_memory search |")
    print("|---:|---:|---:|---:|")
    for n, kw, brute, ann in rows:
        print(f"| {n:,} | {kw:.2f} ms | {brute:.2f} ms | {ann:.2f} ms |")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [100, 1000, 5000, 10000])
