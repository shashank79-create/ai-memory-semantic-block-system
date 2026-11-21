# AI Memory Semantic Block System

A long-term memory module for chat assistants, plus a Gemini notebook that uses it. It stores what a user tells the assistant, finds the memories relevant to each new message, and keeps them up to date (duplicates merged, old events expiring, contradicted facts replaced).

This README reports what the measurements show, including where the original ideas did **not** work.

## 📊 Benchmark: LoCoMo
[LoCoMo](https://github.com/snap-research/locomo) (Maharana et al., ACL 2024) is a public benchmark for long-term conversational memory: 10 long conversations (5,882 dialogue turns over many sessions) and 1,986 questions, each listing the turns that answer it. Every turn is stored as a memory and each question is used as a search; a method scores when it returns the answering turns. 1,535 questions are scored: category 5 (adversarial, 446 questions) is left out, as in the Mem0 paper's LoCoMo evaluation, and 5 questions have no usable evidence.

Results (`python eval/locomo_benchmark.py`; 95% bootstrap confidence intervals in brackets):

| Method | Recall@5 | Recall@10 | Hit@10 | MRR@10 | ms / question |
|---|---:|---:|---:|---:|---:|
| Keyword search (original notebook) | 24.7% (22.6–26.7) | 24.7% (22.6–26.7) | 28.0% | 0.197 | 2.4 |
| BM25 | 43.4% (41.0–45.9) | 51.0% (48.5–53.5) | 56.6% | 0.353 | 1.8 |
| MiniLM embeddings | 35.5% (33.2–37.7) | 44.5% (42.2–46.9) | 50.6% | 0.274 | 16.8 |
| BGE-small embeddings | 48.1% (45.8–50.5) | 58.6% (56.3–60.9) | 65.6% | 0.392 | 28.3 |
| Hybrid BM25 + BGE (reciprocal rank fusion) | 52.5% (50.1–54.9) | 61.9% (59.5–64.2) | 68.9% | 0.438 | 30.7 |
| **Hybrid + cross-encoder reranker** | **68.0% (65.8–70.4)** | **73.7% (71.7–75.9)** | **80.5%** | **0.625** | 93.8 |
| `semantic_memory` default (MiniLM + topic blocks + cutoffs) | 34.1% (31.9–36.4) | 41.9% (39.6–44.3) | 48.0% | 0.269 | 17.6 |

*Recall@k:* share of the answering turns in the top k results. *Hit@10:* questions with at least one answering turn in the top 10. *MRR@10:* 1 / rank of the first answering turn. Times are on a laptop CPU and include embedding the question.

Paired differences in Recall@10 (95% CI):
- **Topic blocks vs plain MiniLM: +0.0%.** In ranking, topic blocks change nothing; they only matter through the cutoffs. LoCoMo's conversations are about people's lives, so most turns and questions land in the "general" block.
- **The store's cutoffs vs no cutoffs: −2.6% (−3.4 to −1.8).** The cutoffs that keep unrelated memories out also drop some relevant ones.
- **BGE-small vs MiniLM: +14.1% (+12.0 to +16.2).** The embedding model matters more than anything else in the store.
- **Reranker vs hybrid: +11.8% (+10.1 to +13.6).** Re-scoring the top 50 with a cross-encoder is the largest single gain, at ~60 ms extra per question.

What this means:
- **The current store is not competitive.** It finds fewer answering turns than plain BM25 keyword ranking, and 32 points fewer than BM25 + BGE + reranker.
- **The original topic-block idea gives no measurable benefit on a public benchmark.** It helped on my small hand-written test (below), where questions are about watches and food, but not on real long conversations.
- **What works is well known:** a stronger embedding model, hybrid keyword + embedding search, and a reranker.
- These are **retrieval** scores. Published LoCoMo results such as Mem0's measure whether an LLM's final *answer* is correct (judged by another LLM), so they are not directly comparable. Answer accuracy with Gemini has not been measured here.
- Confidence intervals resample questions; questions from the same conversation are related, so the true uncertainty is somewhat larger.

## 🧪 Small hand-written test
Before the public benchmark, I wrote two small question sets in `eval/` (36 and 29 memories, 47 and 38 questions), including unrelated questions that should get no memories. They were used to choose the store's cutoffs (`python eval/retrieval_eval.py`). On the held-out set, the store found 77% of relevant memories (original keyword search: 40%) and returned nothing for all 7 unrelated questions (original: 0 of 7).

Treat these numbers with caution: the sets are small (results can move by ±15 points), I wrote both the questions and the system, and the memories are short, tidy sentences. LoCoMo is the more reliable measure. One thing the small test measures that LoCoMo doesn't: whether *unrelated* questions get no memories, which is what the cutoffs are for.

An experiment from this test, kept as a negative result: picking topics by meaning (`EmbeddingTopicClassifier`) sorted topics slightly more accurately than keywords (78% vs 75%) but made search worse (67% vs 74% overall on the tuning set), so keywords stay the default.

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
- SentenceTransformers (MiniLM embeddings, NLI contradiction model)
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
3. Run all cells and start chatting! The first cell installs the packages, and the first run downloads two models: the embedding model (about 90 MB) and the contradiction model (about 300 MB).
4. Save lasting facts with `assistant.remember("...")` and chat with `assistant.chat("...")`. Memories are kept in the `memory/` folder between sessions.

## 🧠 Semantic Memory Store (`semantic_memory.py`)
A standalone memory module that replaces the notebook's keyword search with embeddings.

- **Embeddings + topic-block index:** SentenceTransformers (`all-MiniLM-L6-v2`) vectors, kept in one in-memory matrix per (user, topic) block and saved in SQLite (`memory/memories.db`). All memories are loaded at startup (about 150 MB per 100,000 memories). `ChromaBackend` is still available for much larger stores.
- **Topic blocks:** topics are detected from keywords (plurals included). A search also looks inside the question's topic block, and memories there pass a lower similarity cutoff. This helped on the small hand-written test but made no difference on LoCoMo (see above).
- **Duplicates:** restating something refreshes the stored memory (time, importance, version) instead of adding a copy.
- **Expiry:** only episodic memories (events) expire, after 90 days by default. Semantic memories (lasting facts) stay until deleted. Deleted memories can be restored for 7 days, then `purge_expired` removes them for good.
- **Contradictions (optional):** pass `contradiction_checker=NLIContradictionChecker()` and a new memory that contradicts a very similar stored one replaces it ("I am not allergic to peanuts" replaces "I am allergic to peanuts"). The replaced memory is soft-deleted, so it can be restored for 7 days. The model alone would call "I love pizza" vs "I love sushi" a contradiction, so only pairs with similarity of at least 0.80 are checked. Changes worded very differently ("I live in Delhi" → "I moved to Mumbai") are missed.
- **Ranking:** similarity, importance and recency. A minimum similarity (0.3, or 0.1 inside the question's topic block) and a relative cutoff keep unrelated memories out.
- **Context:** `MemorySession.build_context` fits long-term memories and recent messages into a token budget.

```bash
pip install sentence-transformers tiktoken numpy google-genai   # add chromadb for ChromaBackend
python semantic_memory.py          # small demo
python -m pytest test_semantic_memory.py
python benchmark.py
python eval/retrieval_eval.py
python eval/locomo_benchmark.py   # also downloads BGE-small and a reranker (~220 MB)
```

Every store test runs on three backends: an in-memory stand-in, the topic-block backend and real ChromaDB. More tests use the real MiniLM and NLI models to check search, duplicates and contradictions. Tests needing ChromaDB or the models are skipped when those aren't available.

The search cutoffs come from the small hand-written test. The duplicate threshold (0.985) was set from measured MiniLM scores. For example, "I am not allergic to peanuts" scores 0.970 against "I am allergic to peanuts", so it isn't treated as a duplicate.

## 📫 Connect With Me
- LinkedIn: [https://www.linkedin.com/in/shashank-yadav-142a42336]

## 📄 License
MIT
