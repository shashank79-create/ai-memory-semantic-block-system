# AI Memory Semantic Block System

A long-term memory module for chat assistants, plus a Gemini notebook that uses it. It stores what a user tells the assistant, finds the memories relevant to each new message, and keeps them up to date (duplicates merged, old events expiring, contradicted facts replaced).

This README reports what the measurements show, including where the original ideas did **not** work.

## 🔎 How search works
1. **Candidates:** the 50 memories closest in meaning (BGE-small embeddings) and the 50 best keyword matches (BM25) are merged with reciprocal rank fusion.
2. **Topic block:** memories filed under the question's topic (watches, food, travel, technology; picked by keywords) are added too.
3. **Reranker:** a cross-encoder (`ms-marco-MiniLM-L-6-v2`) reads the question with each candidate and scores how relevant it is.
4. **Cutoffs:** if the question matches nothing well (best embedding similarity below 0.6 *and* best reranker score below −9.5), it is treated as unrelated to the user and gets no memories. Memories scoring more than 2 below the best are dropped. Topic-block memories skip both cutoffs.
5. **Ranking:** relevance, importance and recency combined; the top k are returned.

This replaced the original design (MiniLM embeddings with topic blocks and similarity cutoffs), after the public benchmark below showed it was weaker than plain keyword search. The original is still available as "classic" search (`reranker=None`).

## 📊 Benchmark: LoCoMo
[LoCoMo](https://github.com/snap-research/locomo) (Maharana et al., ACL 2024) is a public benchmark for long-term conversational memory: 10 long conversations (5,882 dialogue turns over many sessions) and 1,986 questions, each listing the turns that answer it. Every turn is stored as a memory and each question is used as a search; a method scores when it returns the answering turns. 1,535 questions are scored: category 5 (adversarial, 446 questions) is left out, as in the Mem0 paper's LoCoMo evaluation, and 5 questions have no usable evidence.

Results (`python eval/locomo_benchmark.py`; 95% bootstrap confidence intervals in brackets):

| Method | Recall@5 | Recall@10 | Hit@10 | MRR@10 |
|---|---:|---:|---:|---:|
| Keyword search (original notebook) | 24.7% (22.6–26.7) | 24.7% (22.6–26.7) | 28.0% | 0.197 |
| `semantic_memory` classic (previous design) | 34.1% (31.9–36.4) | 41.9% (39.6–44.3) | 48.0% | 0.269 |
| BM25 | 43.4% (41.0–45.9) | 51.0% (48.5–53.5) | 56.6% | 0.353 |
| MiniLM embeddings | 35.5% (33.2–37.7) | 44.5% (42.2–46.9) | 50.6% | 0.274 |
| BGE-small embeddings | 48.1% (45.8–50.5) | 58.6% (56.3–60.9) | 65.6% | 0.392 |
| Hybrid BM25 + BGE | 52.5% (50.1–54.9) | 61.9% (59.5–64.2) | 68.9% | 0.438 |
| Hybrid + reranker (standalone baseline) | 68.0% (65.8–70.4) | 73.7% (71.7–75.9) | 80.5% | 0.625 |
| **`semantic_memory`, full ranking** (cutoffs off) | **68.0% (65.8–70.4)** | **73.8% (71.8–76.0)** | **80.6%** | **0.624** |
| `semantic_memory` default (with cutoffs) | 58.5% (56.2–60.9) | 59.7% (57.3–61.9) | 67.5% | 0.589 |

*Recall@k:* share of the answering turns in the top k results. *Hit@10:* questions with at least one answering turn in the top 10. *MRR@10:* 1 / rank of the first answering turn. The hybrid searches take about 40–90 ms per question on a laptop CPU, most of it the reranker.

What this shows:
- **The rebuilt search finds 74% of answering turns in its top 10, up from 42%** (+31.9 points), matching the best standalone baseline.
- **The default cutoffs trade recall for focus.** They return about 3 turns per question instead of 10, which costs 14 points of Recall@10 (−14.2%, CI −15.8 to −12.6). Pass `relevance_gap=None` to get the full ranking. The answer gate itself rejected none of LoCoMo's questions (all of them are about the user).
- **What mattered, measured one step at a time** (paired differences in Recall@10): BGE-small instead of MiniLM +14.1% (+12.0 to +16.2); adding BM25 +3.3% (+1.3 to +5.5); the reranker +11.8% (+10.1 to +13.6).
- **Topic blocks don't change ranking on LoCoMo** (+0.0%): its conversations are about people's lives, so almost everything is "general". Their value is in the cutoffs, for an assistant whose memories fall into clear topics (see the next section and the notebook).
- These are **retrieval** scores. Published LoCoMo results such as Mem0's measure whether an LLM's final *answer* is correct (judged by another LLM), so they are not directly comparable. Answer accuracy with Gemini has not been measured.
- Confidence intervals resample questions; questions from the same conversation are related, so the true uncertainty is somewhat larger.

## 🧪 Small hand-written test
LoCoMo only checks whether answering turns are found. It doesn't check whether search returns *nothing* for questions unrelated to the user, or how much irrelevant material comes back. Two small hand-written sets in `eval/` measure that (36 and 29 memories, 47 and 38 questions, including unrelated ones like "Who wrote Hamlet?"). Settings were chosen on the first set (`python eval/retrieval_eval.py --sweep-hybrid`); results on the second (held-out) set:

| Method | Recall | Precision | Rejects unrelated | Overall |
|---|---:|---:|---:|---:|
| Keyword search (original notebook) | 40% | 18% | 0% | 20% |
| Classic (previous design) | 77% | 68% | 100% | 77% |
| Hybrid, no cutoffs | 86% | 25% | 0% | 31% |
| **Hybrid with cutoffs (default)** | 77% | 50% | 100% | 68% |

*Recall:* share of relevant memories returned (up to 5). *Precision:* share of returned memories that were relevant. *Overall:* F1 of recall and precision, combined with rejection.

- **On this small test, the classic design is still better** (77% vs 68% overall): the hybrid search returns more irrelevant memories (50% precision vs 68%). The reranker is trained on web search and gives short personal facts low, poorly calibrated scores, which makes good cutoffs hard.
- **Without cutoffs, every unrelated question gets memories.** The answer gate fixes that here and costs nothing on LoCoMo.
- **Treat these numbers with caution:** the sets are small (results can move by ±15 points), I wrote both the questions and the system, and while building the hybrid search I looked at held-out results more than once, so that set is no longer fully blind.

Earlier experiments from this test, kept as negative results: picking topics by meaning (`EmbeddingTopicClassifier`) sorted topics slightly more accurately than keywords (78% vs 75%) but made classic search worse (67% vs 74% overall on the tuning set); and on LoCoMo, topic blocks gave no ranking gain.

## ⏱️ Speed
`python benchmark.py` times one search over N memories (laptop CPU, excluding the ~15 ms to embed the question):

| Memories | Brute-force numpy | `semantic_memory` (topic-block index) | `semantic_memory` on ChromaDB |
|---:|---:|---:|---:|
| 1,000 | 0.03 ms | 0.25 ms | 11.4 ms |
| 10,000 | 0.49 ms | 1.87 ms | 61.5 ms |
| 50,000 | 2.13 ms | 3.67 ms | 310.7 ms |

- The in-memory topic-block index is much faster than **my ChromaDB setup**, but that setup filtered every query by user, topic and deleted status, which makes ChromaDB scan every record (an unfiltered ChromaDB query took 3.8 ms at 10,000). With one collection per user it would likely be far closer, so this is not a fair ChromaDB comparison.
- Plain numpy over all vectors is faster still; the index adds ranking, filtering and result building.
- At these sizes, search time doesn't matter: embedding the question takes ~15 ms and an LLM reply takes seconds.

## 🛠️ Built With
- Python
- Google Gemini AI
- SentenceTransformers (BGE-small embeddings, ms-marco cross-encoder reranker, NLI contradiction model)
- NumPy + SQLite (topic-block index); ChromaDB optional
- Jupyter Notebook

## 📁 Files
- `AI_Memory_Blocks.ipynb` - Gemini assistant with long-term memory
- `semantic_memory.py` - the memory store the notebook uses
- `test_semantic_memory.py` - tests for the memory store
- `benchmark.py` - search timing at different memory counts
- `eval/locomo_benchmark.py` - LoCoMo benchmark (downloads the dataset to `eval/data/`); results in `eval/results/`
- `eval/retrieval_eval.py`, `eval/dataset.json`, `eval/test_set.json` - small hand-written test

## 🎯 How to Use
1. Get a free Gemini API key from [aistudio.google.com](https://aistudio.google.com)
2. Open the notebook in Jupyter (from this folder, so it can import `semantic_memory.py`)
3. Run all cells and start chatting! The first cell installs the packages, and the first run downloads four models: BGE-small embeddings (about 130 MB), the reranker (about 90 MB), the contradiction model (about 300 MB) and, for classic search only, MiniLM (about 90 MB).
4. Save lasting facts with `assistant.remember("...")` and chat with `assistant.chat("...")`. Memories are kept in the `memory/` folder between sessions.

## 🧠 Semantic Memory Store (`semantic_memory.py`)
A standalone memory module that replaces the notebook's keyword search with embeddings.

- **Hybrid search:** BGE-small embeddings + BM25 keyword index + cross-encoder reranker (see "How search works" above). `reranker=None` switches to classic search.
- **Storage:** vectors are kept in one in-memory matrix per (user, topic) block and saved in SQLite (`memory/memories.db`). All memories are loaded at startup (about 150 MB per 100,000 memories). The BM25 index is rebuilt from the store on first search. `ChromaBackend` is still available for much larger stores.
- **Topic blocks:** topics are detected from keywords (plurals included). A search also looks inside the question's topic block, and memories there skip the cutoffs.
- **Duplicates:** restating something refreshes the stored memory (time, importance, version) instead of adding a copy.
- **Expiry:** only episodic memories (events) expire, after 90 days by default. Semantic memories (lasting facts) stay until deleted. Deleted memories can be restored for 7 days, then `purge_expired` removes them for good.
- **Contradictions (optional):** pass `contradiction_checker=NLIContradictionChecker()` and a new memory that contradicts a very similar stored one replaces it ("I am not allergic to peanuts" replaces "I am allergic to peanuts"). The replaced memory is soft-deleted, so it can be restored for 7 days. The model alone would call "I love pizza" vs "I love sushi" or "I have a dog" vs "I have a cat" contradictions, so only pairs with BGE similarity of at least 0.90 are checked. That avoids deleting compatible facts but misses many real contradictions ("I love my Rolex" → "I hate my Rolex" scores 0.815), as well as changes worded very differently ("I live in Delhi" → "I moved to Mumbai").
- **Ranking:** reranker relevance, importance and recency.
- **Context:** `MemorySession.build_context` fits long-term memories and recent messages into a token budget.

```bash
pip install sentence-transformers tiktoken numpy google-genai   # add chromadb for ChromaBackend
python semantic_memory.py          # small demo
python -m pytest test_semantic_memory.py
python benchmark.py
python eval/retrieval_eval.py
python eval/locomo_benchmark.py   # downloads the LoCoMo dataset (2.8 MB) on first run
```

Every store test runs on three backends: an in-memory stand-in, the topic-block backend and real ChromaDB. More tests use the real MiniLM and NLI models to check search, duplicates and contradictions. Tests needing ChromaDB or the models are skipped when those aren't available.

The search cutoffs come from the small hand-written test. The duplicate threshold (0.96) was set from measured BGE scores: rewordings like "I really love my Rolex" score 0.97–1.0, while "I am not allergic to peanuts" scores 0.923 against "I am allergic to peanuts", so it isn't treated as a duplicate.

## 📫 Connect With Me
- LinkedIn: [https://www.linkedin.com/in/shashank-yadav-142a42336]

## 📄 License
MIT
