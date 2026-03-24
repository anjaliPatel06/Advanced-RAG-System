import os
import pickle
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from rank_bm25 import BM25Okapi
from ranker import rank_docs
from utils import detect_query_intent

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("rag-index")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings
)

with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

texts = [doc.page_content for doc in chunks]
tokenized_corpus = [text.split() for text in texts]
bm25 = BM25Okapi(tokenized_corpus)


def keyword_search(query, k=5):
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [chunks[i] for i in top_n]


def filter_docs_initial(docs):
    """
    Pre-reranking filter: remove too-short or too-long chunks.
    IMPROVED: Also strip chunks that are pure navigation/TOC residue.
    """
    filtered = []
    for d in docs:
        text = d.page_content.strip()
        if len(text) < 50:
            continue
        if len(text) > 2000:
            continue
        # IMPROVED: Skip chunks that look like leftover nav menus
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        nav_ratio = sum(1 for l in lines if l.startswith("*") or l.startswith("-")) / max(len(lines), 1)
        if nav_ratio > 0.6:
            continue
        filtered.append(d)
    return filtered


def get_relevant_docs(query):
    """
    IMPROVED hybrid retrieval:
    - Intent-aware k: how_to/error queries get more semantic hits
    - Reciprocal Rank Fusion (RRF) instead of naive dedup concat
    - Returns ranked (score, doc) tuples
    """
    intent = detect_query_intent(query)

    semantic_k = 10 if intent in ("how_to", "error") else 8
    keyword_k = 7 if intent in ("definition", "comparison") else 5

    semantic_docs = vector_store.similarity_search(query, k=semantic_k)
    keyword_docs = keyword_search(query, k=keyword_k)

    rrf_scores = {}
    doc_map = {}

    def rrf_add(doc_list, weight=1.0):
        for rank, doc in enumerate(doc_list):
            key = doc.page_content.strip()
            rrf_score = weight / (60 + rank + 1)  # RRF formula
            rrf_scores[key] = rrf_scores.get(key, 0) + rrf_score
            doc_map[key] = doc

    rrf_add(semantic_docs, weight=1.2)   # Slightly prefer semantic
    rrf_add(keyword_docs, weight=1.0)

    sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)
    combined = [doc_map[k] for k in sorted_keys]

    combined = filter_docs_initial(combined)

    ranked_docs = rank_docs(query, combined, top_k=5)

    return ranked_docs
