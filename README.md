# AI Memory Semantic Block System

An AI assistant with long-term memory. It remembers what you tell it, sorts memories into topic blocks (watches, food, travel, technology), and gives the AI only the memories that matter for each message.

## 🚀 The Problem
Chat assistants either forget everything between sessions or stuff the whole history into the prompt. Searching old messages by keywords misses things said in different words ("timepiece" never finds "watch") and pulls in unrelated ones.

## 💡 My Solution
- **Search by meaning:** memories are stored as embeddings, so a question finds related memories even when the words differ.
- **Topic blocks:** each memory is filed under a topic. A question about watches also searches the whole watches block, which finds memories the embedding model scores low (it rates "Which watch should I buy?" vs "I love my Rolex Submariner" at only 0.15).
- **Memory that stays correct:** repeated facts are merged, lasting facts never expire, one-off events expire after 90 days, and a new fact that contradicts an old one replaces it.
- **Only what matters reaches the AI:** unrelated questions get no memories, just the recent conversation.

## 📊 Performance
Each topic block has its own vector index, so a search compares the question only against the blocks it needs. Measured with `python benchmark.py` (Intel CPU, Python 3.11): median time for one search, not counting the ~15 ms to embed the question.

| Memories | Keyword scan (original notebook) | Brute-force numpy | **Topic blocks, all topics (default)** | **Topic blocks, block-only** | ChromaDB |
|---:|---:|---:|---:|---:|---:|
| 100 | 0.02 ms | 0.01 ms | 0.21 ms | 0.20 ms | 7.56 ms |
| 1,000 | 0.14 ms | 0.03 ms | 0.25 ms | 0.23 ms | 11.38 ms |
| 5,000 | 0.73 ms | 0.22 ms | 0.58 ms | 0.43 ms | 34.06 ms |
| 10,000 | 1.37 ms | 0.49 ms | 1.87 ms | 0.87 ms | 61.51 ms |
| 50,000 | 7.89 ms | 2.13 ms | 3.67 ms | 1.78 ms | 310.67 ms |

What this shows:
- **Topic blocks are 30–85x faster than the ChromaDB version** (1.87 ms vs 61.5 ms at 10,000 memories). ChromaDB's metadata filters (user, deleted, topic) scan every record; blocks are picked directly instead.
- **Block-only search (`cross_topic=False`) roughly halves the time again**, since it compares only the question's topic block and the "general" block. The trade-off is that memories filed under other topics are skipped.
- **The search is exact,** not approximate: every vector in the chosen blocks is compared. It still grows linearly, but slowly: 3.7 ms at 50,000 memories.
- **Search is no longer a noticeable cost.** Embedding the question takes ~15 ms and a Gemini reply takes seconds. The main gain is still *which* memories are found.
- The two raw baselines are faster at small sizes because they skip ranking, filtering and result building.

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

## 🎯 How to Use
1. Get a free Gemini API key from [aistudio.google.com](https://aistudio.google.com)
2. Open the notebook in Jupyter (from this folder, so it can import `semantic_memory.py`)
3. Run all cells and start chatting! The first cell installs the packages, and the first run downloads the embedding model (about 90 MB).
4. Save lasting facts with `assistant.remember("...")` and chat with `assistant.chat("...")`. Memories are kept in the `memory/` folder between sessions.

## 🧠 Semantic Memory Store (`semantic_memory.py`)
A standalone memory module that replaces the notebook's keyword search with embeddings.

- **Embeddings + topic-block index:** SentenceTransformers (`all-MiniLM-L6-v2`) vectors, kept in one in-memory matrix per (user, topic) block and saved in SQLite (`memory/memories.db`). All memories are loaded at startup (about 150 MB per 100,000 memories). `ChromaBackend` is still available for much larger stores.
- **Topic blocks:** topics are detected from keywords (plurals included). A search also looks inside the question's topic block, and memories there count as relevant even when the embedding score is low (MiniLM scores "Which watch should I buy?" vs "I love my Rolex Submariner" at only 0.15). Pass `cross_topic=False` to search only the question's block plus "general", which is faster.
- **Duplicates:** restating something refreshes the stored memory (time, importance, version) instead of adding a copy.
- **Expiry:** only episodic memories (events) expire, after 90 days by default. Semantic memories (lasting facts) stay until deleted. Deleted memories can be restored for 7 days, then `purge_expired` removes them for good.
- **Contradictions (optional):** pass `contradiction_checker=NLIContradictionChecker()` and a new memory that contradicts a very similar stored one replaces it ("I am not allergic to peanuts" replaces "I am allergic to peanuts"). The replaced memory is soft-deleted, so it can be restored for 7 days. The model alone would call "I love pizza" vs "I love sushi" a contradiction, so only pairs with similarity of at least 0.80 are checked. Changes worded very differently ("I live in Delhi" → "I moved to Mumbai") are missed.
- **Ranking:** similarity, importance and recency, with a minimum similarity so unrelated memories aren't returned.
- **Context:** `MemorySession.build_context` fits long-term memories and recent messages into a token budget.

```bash
pip install sentence-transformers tiktoken numpy google-genai   # add chromadb for ChromaBackend
python semantic_memory.py          # small demo
python -m pytest test_semantic_memory.py
python benchmark.py
```

Every store test runs on three backends: an in-memory stand-in, the topic-block backend and real ChromaDB. A few more tests check search and duplicates with the real MiniLM model. Tests needing ChromaDB or MiniLM are skipped when those aren't installed.

The search cutoff (0.2) and duplicate threshold (0.985) were set from measured MiniLM scores. For example, "I am not allergic to peanuts" scores 0.970 against "I am allergic to peanuts", so it isn't treated as a duplicate.

## 📫 Connect With Me
- LinkedIn: [https://www.linkedin.com/in/shashank-yadav-142a42336]

## 📄 License
MIT
