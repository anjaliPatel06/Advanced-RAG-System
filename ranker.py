from sentence_transformers import CrossEncoder
import numpy as np

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rank_docs(query, docs, top_k=5, threshold=0.3):
    """
    Rerank docs using cross-encoder. 
    IMPROVED:
    - Adaptive threshold: if no docs pass hard threshold, return best available (soft fallback)
    - Score normalization via sigmoid so scores are meaningful across queries
    - Returns list of (score, doc) tuples
    """
    if not docs:
        return []

    pairs = [(query, d.page_content) for d in docs]
    raw_scores = reranker.predict(pairs)

    scores = 1 / (1 + np.exp(-np.array(raw_scores)))

    scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

    filtered = [(score, doc) for score, doc in scored_docs if score > threshold]

    if not filtered and scored_docs:
        filtered = list(scored_docs[:2])

    return filtered[:top_k]


def get_top_score(ranked_docs):
    """Returns the highest reranker score from ranked results."""
    if not ranked_docs:
        return 0.0
    return ranked_docs[0][0]
