# Advanced-RAG-System
# Crawl4AI Documentation RAG Assistant
 
An intelligent, self-improving RAG (Retrieval-Augmented Generation) chatbot for the [Crawl4AI](https://docs.crawl4ai.com) documentation. Ask anything about installation, crawling, extraction, or the API — and get precise, hallucination-free answers.
 
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![Pinecone](https://img.shields.io/badge/VectorDB-Pinecone-green)
![Groq](https://img.shields.io/badge/LLM-Groq%20%7C%20LLaMA3-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
 
---
 
## ✨ Features
 
- 🔍 **Hybrid Search** — Semantic (Pinecone) + BM25 keyword search combined
- 🔀 **Reciprocal Rank Fusion (RRF)** — Intelligently merges both search results
- 🎯 **Cross-Encoder Reranking** — `ms-marco-MiniLM-L-6-v2` for high-precision reranking
- 🧠 **Short + Long-term Memory** — Session memory + Pinecone-backed persistent memory
- ✍️ **Query Rewriting** — Automatically expands vague queries for better retrieval
- 🛡️ **Hallucination Guard** — Refuses to answer if confidence is too low
- 🎭 **Intent Detection** — Adapts response style for how-to, definition, comparison, error queries
- 👍 **Feedback Loop** — Thumbs up/down to track answer quality
- ⚡ **Streaming Responses** — Token-by-token streaming via Groq
 
---
 
## 🏗️ Architecture
 
```
User Query
    ↓
Query Rewriting (LLM)
    ↓
Hybrid Retrieval
├── Semantic Search (Pinecone)
└── BM25 Keyword Search
    ↓
Reciprocal Rank Fusion
    ↓
Cross-Encoder Reranking
    ↓
Confidence Gate (failure detection)
    ↓
Intent Detection → Prompt Building
    ↓
LLM (LLaMA 3.1 via Groq) → Streaming Answer
    ↓
Memory Update (Short + Long-term)
```
 
---
 
## 📁 Project Structure
 
```
├── app.py              # Main RAG pipeline
├── ingest.py           # Data ingestion & Pinecone upload
├── retriever.py        # Hybrid retrieval (semantic + BM25 + RRF)
├── ranker.py           # Cross-encoder reranking
├── memory.py           # Short-term + long-term memory
├── utils.py            # Query rewriting, intent detection, context optimization
├── streamlit_app.py    # Streamlit web UI
├── requirements.txt    # Python dependencies
├── data/
│   └── scraped/        # Crawl4AI documentation (.md files)
└── .env                # API keys (not committed)
```
 
---
 
## 🚀 Setup & Installation
 
### 1. Clone the Repository
 
```bash
git clone https://github.com/yourusername/crawl4ai-rag-assistant.git
cd crawl4ai-rag-assistant
```
 
### 2. Create Virtual Environment
 
```bash
python -m venv venv
 
# Windows
venv\Scripts\activate
 
# macOS / Linux
source venv/bin/activate
```
 
### 3. Install Dependencies
 
```bash
pip install -r requirements.txt
```
 
### 4. Set Up API Keys
 
Create a `.env` file in the root directory:
 
```env
GROQ_API_KEY=your_groq_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```
 
- Get Groq API key: [console.groq.com](https://console.groq.com)
- Get Pinecone API key: [app.pinecone.io](https://app.pinecone.io)
 
> ⚠️ **Never commit your `.env` file to GitHub!** It's already in `.gitignore`.
 
### 5. Add Documentation Data
 
Place your scraped Crawl4AI `.md` files inside:
 
```
data/scraped/
```
 
### 6. Run Data Ingestion (First Time Only)
 
```bash
python ingest.py
```
 
This will:
- Clean and chunk the markdown files
- Generate embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- Upload vectors to Pinecone
- Save chunks locally as `chunks.pkl`
 
### 7. Launch the App
 
```bash
streamlit run streamlit_app.py
```
 
Open your browser at `http://localhost:8501` 🎉
 
---
 
## 🛠️ Tech Stack
 
| Component | Technology |
|-----------|-----------|
| LLM | LLaMA 3.1 8B via [Groq](https://groq.com) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector DB | [Pinecone](https://pinecone.io) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Keyword Search | BM25 (`rank_bm25`) |
| Framework | LangChain |
| UI | Streamlit |
 
---
 
## 💡 How It Works
 
1. **Ingest**: Markdown docs are cleaned, chunked, deduplicated, and embedded into Pinecone.
2. **Query**: User asks a question → query is rewritten for better retrieval.
3. **Retrieve**: Both semantic (Pinecone) and keyword (BM25) search run in parallel.
4. **Fuse**: Reciprocal Rank Fusion merges the two result lists.
5. **Rerank**: Cross-encoder scores each doc against the query for precision.
6. **Guard**: If top score is too low, the bot admits it doesn't know.
7. **Answer**: Intent-aware prompt is built and streamed through LLaMA 3.1.
8. **Remember**: Good answers are stored in long-term memory for future sessions.
 
---
 
## 🔒 Security Notes
 
- Never share or commit your API keys
- Add `.env` to your `.gitignore`
- Regenerate keys if accidentally exposed
 
---
 
## 📄 License
 
MIT License — feel free to use, modify, and distribute.
 
---
 
## 🙏 Acknowledgements
 
- [Crawl4AI](https://github.com/unclecode/crawl4ai) for the amazing web crawling library
- [Groq](https://groq.com) for ultra-fast LLM inference
- [Pinecone](https://pinecone.io) for vector storage
- [LangChain](https://langchain.com) for the RAG framework
