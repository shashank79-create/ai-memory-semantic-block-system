"""Score how well each search method finds the right memories.

Two question sets, each with its own user's memories and the memories that would help
answer each question (unrelated questions list none and should get nothing back):
- eval/dataset.json  - tuning set: used to choose search settings (--sweep)
- eval/test_set.json - held-out set: written before tuning, never used to choose settings

All memories are stored once with the real MiniLM model (topics picked by the store's
classifier), then each method answers every question with up to k=5 memories.

Metrics, answerable questions:
- recall: share of the relevant memories returned
- precision: share of returned memories that are relevant (0 if nothing returned)
- hit rate: questions with at least one relevant memory returned
- MRR: 1 / rank of the first relevant memory, averaged
Unrelated questions:
- rejection: share that correctly got no memories
Overall: F1 of recall and precision, and rejection, weighted by how many questions of
each kind there are.

Run from the repo root:
  python eval/retrieval_eval.py                      # both sets, all methods
  python eval/retrieval_eval.py --sweep              # try settings on the tuning set
  python eval/retrieval_eval.py --show-misses default --set test
"""
from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from semantic_memory import BlockIndexBackend, SemanticMemoryStore  # noqa: E402

K = 5
SETS = {"tuning": "dataset.json", "test": "test_set.json"}

# The original notebook's topic keywords and search (AI_Memory_Blocks.ipynb, first version).
ORIGINAL_TOPICS = {
    "watches": ["watch", "rolex", "casio", "timex", "omega", "seiko", "analog", "digital",
                "automatic", "quartz"],
    "food": ["food", "pizza", "pasta", "burger", "restaurant", "dinner", "lunch", "italian",
             "chinese", "mexican"],
    "technology": ["phone", "laptop", "computer", "app", "software", "iphone", "android", "gadget"],
    "travel": ["trip", "vacation", "hotel", "flight", "beach", "travel"],
    "fashion": ["clothes", "shirt", "shoes", "brand", "style", "wear"],
}


def original_topic(text: str) -> str:
    lowered = text.lower()
    for topic, words in ORIGINAL_TOPICS.items():
        if any(w in lowered for w in words):
            return topic
    return "general"


def keyword_method(memories: dict):
    by_topic = defaultdict(list)
    for mem_id, text in memories.items():
        by_topic[original_topic(text)].append((mem_id, text))

    def search(query: str) -> list:
        words = set(query.lower().split())
        scored = []
        for mem_id, text in by_topic[original_topic(query)]:
            n = sum(1 for w in words if w in text.lower())
            if n:
                scored.append((n, mem_id))
        scored.sort(key=lambda s: -s[0])
        return [mem_id for _, mem_id in scored[:K]]
    return search


def score(results: dict, queries: list) -> dict:
    answerable = [q for q in queries if q["relevant"]]
    unrelated = [q for q in queries if not q["relevant"]]
    recall, precision, hit, mrr = [], [], [], []
    for q in answerable:
        got, want = results[q["q"]], set(q["relevant"])
        found = [m for m in got if m in want]
        recall.append(len(found) / min(len(want), K))
        precision.append(len(found) / len(got) if got else 0.0)
        hit.append(1.0 if found else 0.0)
        ranks = [i for i, m in enumerate(got, 1) if m in want]
        mrr.append(1.0 / ranks[0] if ranks else 0.0)
    out = {"recall": statistics.mean(recall), "precision": statistics.mean(precision),
           "hit rate": statistics.mean(hit), "MRR": statistics.mean(mrr)}
    out["rejection"] = statistics.mean([0.0 if results[q["q"]] else 1.0 for q in unrelated]) if unrelated else 1.0
    f1 = 2 * out["recall"] * out["precision"] / (out["recall"] + out["precision"] or 1)
    out["overall"] = (len(answerable) * f1 + len(unrelated) * out["rejection"]) / len(queries)
    return out


class Bench:
    """One set's memories in a real store, with every method available."""

    def __init__(self, name: str, model):
        data = json.loads((ROOT / "eval" / SETS[name]).read_text(encoding="utf-8"))
        self.memories, self.queries = data["memories"], data["queries"]
        for q in self.queries:
            assert all(m in self.memories for m in q["relevant"]), q
        self.model = model
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.store = SemanticMemoryStore(storage_path=self._tmp.name, embedder=model,
                                         backend=BlockIndexBackend(f"{self._tmp.name}/memories.db"))
        text_to_id = {text: mem_id for mem_id, text in self.memories.items()}
        stored = self.store.add_many(
            [{"content": t, "skip_duplicates": False} for t in self.memories.values()], "u")
        self.store_id = {m.id: text_to_id[m.content] for m in stored}
        self.topics = {text_to_id[m.content]: m.topic for m in stored}
        self.mem_ids = list(self.memories)
        self.mem_vecs = model.encode([self.memories[m] for m in self.mem_ids], normalize_embeddings=True)

    def close(self):
        self.store.backend.close()
        self._tmp.cleanup()

    def top_k_embedding(self, query: str) -> list:
        q = self.model.encode([query], normalize_embeddings=True)[0]
        return [self.mem_ids[i] for i in np.argsort(-(self.mem_vecs @ q))[:K]]

    def via_store(self, **options):
        return lambda query: [self.store_id[m.id] for m in self.store.search(query, "u", k=K, **options)]

    def run(self, fn) -> dict:
        return {q["q"]: fn(q["q"]) for q in self.queries}

    def methods(self) -> dict:
        return {
            "Keyword search (original notebook)": keyword_method(self.memories),
            "Embeddings, top 5 (no cutoff)": self.top_k_embedding,
            "Embeddings + similarity cutoff": self.via_store(use_query_topic=False),
            "Embeddings + topic blocks (default)": self.via_store(),
            "Topic blocks, block-only": self.via_store(cross_topic=False),
        }


METRICS = ["recall", "precision", "hit rate", "MRR", "rejection", "overall"]


def fmt(metric, value):
    return f"{value:.2f}" if metric == "MRR" else f"{value:.0%}"


def print_table(bench: Bench, results: dict):
    print("| Method | " + " | ".join(m if m == "MRR" else m.capitalize() for m in METRICS) + " |")
    print("|---|" + "---:|" * len(METRICS))
    for name, res in results.items():
        s = score(res, bench.queries)
        print(f"| {name} | " + " | ".join(fmt(m, s[m]) for m in METRICS) + " |")


def print_misses(bench: Bench, res: dict):
    for q in bench.queries:
        got, want = res[q["q"]], set(q["relevant"])
        missing, extra = sorted(want - set(got)), [m for m in got if m not in want]
        if missing or extra:
            show = lambda ids: ", ".join(f"{m} [{bench.topics[m]}] {bench.memories[m]!r}" for m in ids) or "-"
            print(f"- {q['q']!r}\n    missing: {show(missing)}\n    extra:   {show(extra)}")


def sweep(bench: Bench):
    grid = itertools.product([0.2, 0.25, 0.3, 0.35, 0.4],    # min_similarity
                             [0.0, 0.1, 0.2, "same"],         # topic_min_similarity
                             [None, 0.1, 0.15, 0.2, 0.25])    # max_gap
    rows = []
    for min_sim, topic_min, gap in grid:
        options = {"min_similarity": min_sim, "max_gap": gap,
                   "topic_min_similarity": min_sim if topic_min == "same" else topic_min}
        rows.append((score(bench.run(bench.via_store(**options)), bench.queries), options))
    rows.sort(key=lambda r: -r[0]["overall"])
    print("Top settings on the tuning set:\n")
    print("| min_similarity | topic_min_similarity | max_gap | " + " | ".join(METRICS) + " |")
    print("|---:|---:|---:|" + "---:|" * len(METRICS))
    for s, o in rows[:12]:
        print(f"| {o['min_similarity']} | {o['topic_min_similarity']} | {o['max_gap']} | "
              + " | ".join(fmt(m, s[m]) for m in METRICS) + " |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true", help="try search settings on the tuning set")
    parser.add_argument("--set", choices=list(SETS), help="only this question set")
    parser.add_argument("--show-misses", metavar="METHOD", help="list questions this method got wrong")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    if args.sweep:
        bench = Bench("tuning", model)
        sweep(bench)
        bench.close()
        return

    for name in [args.set] if args.set else list(SETS):
        bench = Bench(name, model)
        answerable = sum(1 for q in bench.queries if q["relevant"])
        print(f"\n## {name} set: {len(bench.memories)} memories, {answerable} answerable questions, "
              f"{len(bench.queries) - answerable} unrelated, k={K}")
        print("Topics the store filed memories under:", dict(sorted(Counter(bench.topics.values()).items())))
        print()
        results = {method: bench.run(fn) for method, fn in bench.methods().items()}
        print_table(bench, results)
        if args.show_misses:
            method = next(m for m in results if args.show_misses.lower() in m.lower())
            print(f"\nMisses for {method}:")
            print_misses(bench, results[method])
        bench.close()


if __name__ == "__main__":
    main()
