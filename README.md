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
Measured with `python benchmark.py` (Intel CPU, Python 3.11), median time to search once, not counting the ~10 ms to embed the question:

| Memories | Keyword scan (original notebook) | Brute-force embeddings (numpy) | `semantic_memory` search (ChromaDB) |
|---:|---:|---:|---:|
| 100 | 0.02 ms | 0.01 ms | 6.63 ms |
| 1,000 | 0.15 ms | 0.03 ms | 11.43 ms |
| 5,000 | 0.85 ms | 0.22 ms | 32.70 ms |
| 10,000 | 1.76 ms | 0.48 ms | 63.28 ms |

What this shows:
- **Search speed isn't the bottleneck.** All three take milliseconds, while a Gemini reply takes seconds. The gain from this project is *which* memories are found, not speed.
- **The ChromaDB search grows linearly.** Its metadata filters (`user_id`, `deleted`, `topic`) scan every record: an unfiltered query takes 3.8 ms at 10,000 memories, but adding the `user_id` filter raises that to 34 ms. Every search runs two filtered queries (all memories, plus the topic block).
- **Topic blocks don't make search faster** in this design. They are filters on one shared index, so they improve results, not speed.

## 🛠️ Built With
- Python
- Google Gemini AI
- ChromaDB + SentenceTransformers
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

- **Embeddings + vector index:** SentenceTransformers (`all-MiniLM-L6-v2`) and ChromaDB, saved to disk.
- **Topic blocks:** topics are detected from keywords (plurals included). A search also looks inside the question's topic block, and memories there count as relevant even when the embedding score is low (MiniLM scores "Which watch should I buy?" vs "I love my Rolex Submariner" at only 0.15). The speed comes from the vector index; topics improve which memories are found.
- **Duplicates:** restating something refreshes the stored memory (time, importance, version) instead of adding a copy.
- **Expiry:** only episodic memories (events) expire, after 90 days by default. Semantic memories (lasting facts) stay until deleted. Deleted memories can be restored for 7 days, then `purge_expired` removes them for good.
- **Contradictions (optional):** pass `contradiction_checker=NLIContradictionChecker()` and a new memory that contradicts a very similar stored one replaces it ("I am not allergic to peanuts" replaces "I am allergic to peanuts"). The replaced memory is soft-deleted, so it can be restored for 7 days. The model alone would call "I love pizza" vs "I love sushi" a contradiction, so only pairs with similarity of at least 0.80 are checked. Changes worded very differently ("I live in Delhi" → "I moved to Mumbai") are missed.
- **Ranking:** similarity, importance and recency, with a minimum similarity so unrelated memories aren't returned.
- **Context:** `MemorySession.build_context` fits long-term memories and recent messages into a token budget.

```bash
pip install chromadb sentence-transformers tiktoken numpy google-genai
python semantic_memory.py          # small demo
python -m pytest test_semantic_memory.py
python benchmark.py
```

Every store test runs twice: on an in-memory stand-in for ChromaDB and on real ChromaDB. A few more tests check search and duplicates with the real MiniLM model. Tests needing ChromaDB or MiniLM are skipped when those aren't installed.

The search cutoff (0.2) and duplicate threshold (0.985) were set from measured MiniLM scores. For example, "I am not allergic to peanuts" scores 0.970 against "I am allergic to peanuts", so it isn't treated as a duplicate.

## 📫 Connect With Me
- LinkedIn: [https://www.linkedin.com/in/shashank-yadav-142a42336]

## 📄 License
MIT
