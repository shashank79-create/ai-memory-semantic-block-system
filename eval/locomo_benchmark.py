"""Search quality on LoCoMo, a public long-term conversational memory benchmark.

LoCoMo (Maharana et al., ACL 2024, github.com/snap-research/locomo) has 10 long
conversations between two people over many sessions (5,882 dialogue turns) and 1,986
questions. Each question lists the dialogue turns ("evidence") that answer it, so search
can be scored without an LLM: store every turn as a memory, search with the question,
and check whether the evidence turns come back.

Scored: questions in categories 1-4 that have evidence. Category 5 (adversarial,
unanswerable) is left out, as in the Mem0 paper's LoCoMo evaluation.

Methods compared (all return a ranked list of turns):
- Keyword search (original notebook): topic keywords + word overlap
- BM25: standard keyword ranking (k1=1.5, b=0.75), implemented here
- MiniLM: all-MiniLM-L6-v2 embeddings (the store's model)
- BGE-small: BAAI/bge-small-en-v1.5 embeddings, a stronger small model
- Hybrid BM25 + BGE: reciprocal rank fusion (k=60)
- Hybrid + reranker: top 50 of the hybrid re-scored by cross-encoder/ms-marco-MiniLM-L-6-v2
- semantic_memory default: the store's hybrid search (BGE + BM25 + reranker) with its
  answer gate and relative cutoff, which can return fewer than 10 turns
- semantic_memory hybrid, no gate: the same search returning its full top 10
- semantic_memory classic: the previous design (MiniLM, topic blocks, cutoffs)
- semantic_memory classic, ranking only: classic with cutoffs off, so topic blocks are
  the only difference from plain MiniLM

Metrics: Recall@k (share of evidence turns in the top k), Hit@k (at least one evidence
turn in the top k), MRR@10. 95% confidence intervals by bootstrap over questions
(questions from the same conversation are related, so these intervals are a bit narrow).

Run from the repo root: python eval/locomo_benchmark.py
The dataset (2.8 MB) is downloaded to eval/data/ on first run.
"""
from __future__ import annotations

import json
import math
import random
import re
import statistics
import sys
import tempfile
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from retrieval_eval import keyword_method  # noqa: E402  (original notebook search)
from semantic_memory import BlockIndexBackend, CrossEncoderReranker, SemanticMemoryStore  # noqa: E402

DATA = ROOT / "eval" / "data" / "locomo10.json"
DATA_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
RESULTS = ROOT / "eval" / "results" / "locomo_results.json"
CATEGORY_NAMES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}
KS = (5, 10)
RERANK_POOL = 50
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# ----- data -------------------------------------------------------------------------

def load_locomo():
    if not DATA.exists():
        DATA.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading LoCoMo to {DATA} ...", flush=True)
        urllib.request.urlretrieve(DATA_URL, DATA)
    return json.loads(DATA.read_text(encoding="utf-8"))


def turns_of(sample) -> list:
    """(dia_id, text) for every dialogue turn, with shared-image captions included."""
    conv = sample["conversation"]
    sessions = sorted((k for k in conv if re.fullmatch(r"session_\d+", k)), key=lambda k: int(k.split("_")[1]))
    out = []
    for key in sessions:
        for t in conv[key]:
            text = f"{t['speaker']}: {t['text']}"
            if t.get("blip_caption"):
                text += f" [shares an image: {t['blip_caption']}]"
            out.append((t["dia_id"], text))
    return out


def evidence_of(q, valid: set) -> list:
    """Normalise evidence ids ("D8:6; D9:17", "D:11:26" -> D8:6, D9:17, D11:26)."""
    found = re.findall(r"D:?(\d+):(\d+)", " ".join(q.get("evidence") or []))
    return sorted({f"D{a}:{b}" for a, b in found} & valid)


# ----- methods ----------------------------------------------------------------------

def tokens(text: str) -> list:
    return re.findall(r"\w+", text.lower())


class BM25:
    def __init__(self, docs: list, k1: float = 1.5, b: float = 0.75):
        self.tf = [Counter(tokens(d)) for d in docs]
        self.len = np.array([sum(c.values()) for c in self.tf], dtype=float)
        self.avg = self.len.mean()
        df = Counter(t for c in self.tf for t in c)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(len(self.tf))
        norm = self.k1 * (1 - self.b + self.b * self.len / self.avg)
        for term in set(tokens(query)):
            if term not in self.idf:
                continue
            tf = np.array([c.get(term, 0) for c in self.tf], dtype=float)
            s += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
        return s


def rank(scores: np.ndarray, n: int) -> list:
    return list(np.argsort(-scores, kind="stable")[:n])


def rrf(*rankings, k: int = 60, n: int = 50) -> list:
    fused = defaultdict(float)
    for ranking in rankings:
        for r, idx in enumerate(ranking):
            fused[idx] += 1.0 / (k + r + 1)
    return sorted(fused, key=lambda i: -fused[i])[:n]


# ----- scoring ----------------------------------------------------------------------

def metrics(ranked: list, gold: list) -> dict:
    gold = set(gold)
    out = {}
    for k in KS:
        hits = len(set(ranked[:k]) & gold)
        out[f"Recall@{k}"] = hits / len(gold)
        out[f"Hit@{k}"] = 1.0 if hits else 0.0
    first = next((r for r, d in enumerate(ranked[:10], 1) if d in gold), None)
    out["MRR@10"] = 1.0 / first if first else 0.0
    return out


def bootstrap_ci(values: list, n: int = 2000, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    arr = np.asarray(values)
    means = arr[rng.integers(0, len(arr), size=(n, len(arr)))].mean(axis=1)
    return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_diff_ci(a: list, b: list, n: int = 2000, seed: int = 0) -> tuple:
    return bootstrap_ci(list(np.asarray(a) - np.asarray(b)), n, seed)


# ----- main -------------------------------------------------------------------------

def main():
    from sentence_transformers import SentenceTransformer
    data = load_locomo()
    minilm = SentenceTransformer("all-MiniLM-L6-v2")
    bge = SentenceTransformer("BAAI/bge-small-en-v1.5")
    store_reranker = CrossEncoderReranker()
    reranker = store_reranker.model

    methods = ["Keyword search (original notebook)", "BM25", "MiniLM", "BGE-small",
               "Hybrid BM25 + BGE", "Hybrid + reranker",
               "semantic_memory default", "semantic_memory with gap 2.0", "semantic_memory hybrid, no gate",
               "semantic_memory classic", "semantic_memory classic, ranking only"]
    per_q = {m: [] for m in methods}           # metric dicts, one per question
    timings = {m: [] for m in methods}         # seconds per question
    questions = []                             # (sample_id, category, question)
    returned_counts = []                       # turns returned by the store's default search
    gap_counts = []                            # turns returned with the gap turned on
    skipped = Counter()

    for sample in data:
        turns = turns_of(sample)
        ids = [d for d, _ in turns]
        texts = [t for _, t in turns]
        valid = set(ids)
        qs = []
        for q in sample["qa"]:
            if q.get("category") not in CATEGORY_NAMES:
                skipped["category 5 (adversarial)"] += 1
                continue
            gold = evidence_of(q, valid)
            if not gold:
                skipped["no usable evidence"] += 1
                continue
            qs.append((q, gold))
        print(f"{sample['sample_id']}: {len(turns)} turns, {len(qs)} questions", flush=True)

        bm25 = BM25(texts)
        mini_docs = minilm.encode(texts, normalize_embeddings=True, batch_size=128)
        bge_docs = bge.encode(texts, normalize_embeddings=True, batch_size=128)
        keyword = keyword_method(dict(zip(ids, texts)))

        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        store = SemanticMemoryStore(storage_path=tmp.name, embedder=minilm,
                                    backend=BlockIndexBackend(f"{tmp.name}/memories.db"),
                                    query_cache_size=4096, reranker=None)
        created = store.add_many([{"content": t, "skip_duplicates": False} for t in texts], "u")
        to_dia = {m.id: d for m, d in zip(created, ids)}
        hybrid_store = SemanticMemoryStore(storage_path=f"{tmp.name}/hybrid", embedder=bge,
                                           backend=BlockIndexBackend(f"{tmp.name}/hybrid.db"),
                                           query_cache_size=4096, reranker=store_reranker,
                                           query_prefix=BGE_QUERY_PREFIX)
        created = hybrid_store.add_many([{"content": t, "skip_duplicates": False} for t in texts], "u")
        to_dia.update({m.id: d for m, d in zip(created, ids)})

        for q, gold in qs:
            question = q["question"]
            questions.append((sample["sample_id"], q["category"], question))
            out = {}

            def timed(name, fn):
                start = time.perf_counter()
                out[name] = fn()
                timings[name].append(time.perf_counter() - start)

            timed("Keyword search (original notebook)", lambda: keyword(question))
            bm25_rank = None

            def do_bm25():
                nonlocal bm25_rank
                bm25_rank = rank(bm25.scores(question), RERANK_POOL)
                return [ids[i] for i in bm25_rank]
            timed("BM25", do_bm25)
            timed("MiniLM", lambda: [ids[i] for i in rank(
                mini_docs @ minilm.encode([question], normalize_embeddings=True)[0], 10)])
            bge_rank = None

            def do_bge():
                nonlocal bge_rank
                qv = bge.encode([BGE_QUERY_PREFIX + question], normalize_embeddings=True)[0]
                bge_rank = rank(bge_docs @ qv, RERANK_POOL)
                return [ids[i] for i in bge_rank]
            timed("BGE-small", do_bge)
            hybrid = None

            def do_hybrid():
                nonlocal hybrid
                hybrid = rrf(bm25_rank, bge_rank, n=RERANK_POOL)
                return [ids[i] for i in hybrid]
            timed("Hybrid BM25 + BGE", do_hybrid)

            def do_rerank():
                scores = reranker.predict([(question, texts[i]) for i in hybrid])
                return [ids[hybrid[j]] for j in np.argsort(-np.asarray(scores), kind="stable")]
            timed("Hybrid + reranker", do_rerank)
            timed("semantic_memory default", lambda: [
                to_dia[m.id] for m in hybrid_store.search(question, "u", k=10)])
            timed("semantic_memory with gap 2.0", lambda: [
                to_dia[m.id] for m in hybrid_store.search(question, "u", k=10, relevance_gap=2.0)])
            gap_counts.append(len(out["semantic_memory with gap 2.0"]))
            timed("semantic_memory hybrid, no gate", lambda: [
                to_dia[m.id] for m in hybrid_store.search(question, "u", k=10, gate_similarity=None,
                                                          gate_logit=None, relevance_gap=None)])
            timed("semantic_memory classic", lambda: [
                to_dia[m.id] for m in store.search(question, "u", k=10)])
            timed("semantic_memory classic, ranking only", lambda: [
                to_dia[m.id] for m in store.search(question, "u", k=10, min_similarity=-1.0,
                                                   topic_min_similarity=-1.0, max_gap=None)])
            returned_counts.append(len(out["semantic_memory default"]))
            for name in methods:
                per_q[name].append(metrics(out[name], gold))
        store.backend.close()
        hybrid_store.backend.close()
        tmp.cleanup()

    # The hybrid/reranker timings include the BM25 and BGE searches they build on.
    timings["Hybrid BM25 + BGE"] = [a + b + c for a, b, c in
                                    zip(timings["Hybrid BM25 + BGE"], timings["BM25"], timings["BGE-small"])]
    timings["Hybrid + reranker"] = [a + b for a, b in
                                    zip(timings["Hybrid + reranker"], timings["Hybrid BM25 + BGE"])]

    n = len(questions)
    print(f"\nScored {n} questions from {len(data)} conversations; skipped {dict(skipped)}.\n")
    cols = ["Recall@5", "Recall@10", "Hit@10", "MRR@10"]
    print("| Method | " + " | ".join(cols) + " | ms / question |")
    print("|---|" + "---:|" * (len(cols) + 1))
    summary = {}
    for name in methods:
        row = {}
        for c in cols:
            row[c] = bootstrap_ci([m[c] for m in per_q[name]])
        row["ms"] = 1000 * statistics.median(timings[name])
        summary[name] = row
        cells = [f"{row[c][0]:.1%} ({row[c][1]:.1%}–{row[c][2]:.1%})" if c != "MRR@10"
                 else f"{row[c][0]:.3f}" for c in cols]
        print(f"| {name} | " + " | ".join(cells) + f" | {row['ms']:.1f} |")

    print("\nRecall@10 by question category:\n")
    cats = sorted(CATEGORY_NAMES)
    counts = Counter(c for _, c, _ in questions)
    print("| Method | " + " | ".join(f"{CATEGORY_NAMES[c]} (n={counts[c]})" for c in cats) + " |")
    print("|---|" + "---:|" * len(cats))
    by_cat = {}
    for name in methods:
        vals = [statistics.mean(m["Recall@10"] for m, (_, c, _) in zip(per_q[name], questions) if c == cat)
                for cat in cats]
        by_cat[name] = dict(zip([CATEGORY_NAMES[c] for c in cats], vals))
        print(f"| {name} | " + " | ".join(f"{v:.1%}" for v in vals) + " |")

    print("\nPaired differences in Recall@10 (95% CI; an interval excluding 0 is a real difference):\n")
    pairs = [("semantic_memory default", "semantic_memory classic", "New default vs previous default (classic)"),
             ("semantic_memory default", "semantic_memory hybrid, no gate", "Gate vs no gate"),
             ("semantic_memory with gap 2.0", "semantic_memory default", "Gap 2.0 on top of the gate"),
             ("semantic_memory hybrid, no gate", "Hybrid + reranker", "Store's hybrid vs the standalone baseline"),
             ("semantic_memory classic, ranking only", "MiniLM", "Topic blocks vs plain MiniLM"),
             ("semantic_memory classic", "semantic_memory classic, ranking only", "Classic cutoffs vs none"),
             ("BGE-small", "MiniLM", "BGE-small vs MiniLM"),
             ("Hybrid BM25 + BGE", "BGE-small", "Hybrid vs BGE-small"),
             ("Hybrid + reranker", "Hybrid BM25 + BGE", "Reranker vs hybrid"),
             ("BM25", "semantic_memory default", "BM25 vs new default")]
    diffs = {}
    for a, b, label in pairs:
        mean, lo, hi = paired_diff_ci([m["Recall@10"] for m in per_q[a]], [m["Recall@10"] for m in per_q[b]])
        diffs[label] = (mean, lo, hi)
        print(f"- {label}: {mean:+.1%} ({lo:+.1%} to {hi:+.1%})")

    avg_returned = statistics.mean(returned_counts)
    print(f"\nsemantic_memory default returned {avg_returned:.1f} turns per question on average "
          f"(others always return 10); {sum(1 for c in returned_counts if c == 0)} questions got none.")
    print(f"With gap 2.0: {statistics.mean(gap_counts):.1f} turns on average; "
          f"{sum(1 for c in gap_counts if c == 0)} questions got none.")
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps({"questions": n, "skipped": dict(skipped), "summary": summary,
                                   "recall10_by_category": by_cat, "paired_recall10_diffs": diffs},
                                  indent=1), encoding="utf-8")
    print(f"\nSaved {RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    random.seed(0)
    main()
