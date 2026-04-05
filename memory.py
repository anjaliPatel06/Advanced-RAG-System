import os
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document  # FIXED: was missing, caused NameError

load_dotenv()

# --- Short-term memory (in-process, session-scoped) ---
memory_store = []

# IMPROVED: Feedback log to track which answers were good/bad for self-improvement
feedback_log = []


def get_short_memory():
    """Returns last 3 Q&A pairs for conversational context."""
    return memory_store[-3:]


def update_short_memory(query, answer):
    memory_store.append(f"Q: {query}\nA: {answer}")
    # Keep short memory bounded
    if len(memory_store) > 20:
        memory_store.pop(0)


# IMPROVED: Feedback tracking for self-improvement loop
def record_feedback(query, answer, was_helpful: bool):
    """
    Called externally (e.g., from Streamlit thumbs up/down) to log answer quality.
    This data can later be used to fine-tune retrieval thresholds.
    """
    feedback_log.append({
        "query": query,
        "answer": answer,
        "helpful": was_helpful
    })


def get_feedback_summary():
    if not feedback_log:
        return "No feedback recorded yet."
    helpful = sum(1 for f in feedback_log if f["helpful"])
    total = len(feedback_log)
    return f"Feedback: {helpful}/{total} answers marked helpful ({100*helpful//total}%)"


# --- Long-term memory (Pinecone-backed) ---
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("rag-index")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings
)


def is_useful_answer(answer):
    """Filter answers that are not worth storing long-term."""
    if not answer or len(answer.strip()) < 30:
        return False
    bad_phrases = [
        "i don't know",
        "not enough information",
        "refer to",
        "i cannot",
        "i don't have"
    ]
    lower = answer.lower()
    return not any(p in lower for p in bad_phrases)


def store_long_term(query, answer):
    """
    Store high-quality Q&A pair into Pinecone with 'memory' type tag.
    FIXED: Document import was missing in original code (caused NameError).
    IMPROVED: Only store if answer passes quality check.
    """
    if not is_useful_answer(answer):
        return

    doc = Document(
        page_content=f"Q: {query}\nA: {answer}",
        metadata={"type": "memory", "query": query}
    )
    try:
        vector_store.add_documents([doc])
    except Exception as e:
        print(f"[Memory] Failed to store long-term memory: {e}")


def get_long_term(query, k=3):
    """
    Retrieve relevant past Q&A pairs from Pinecone memory namespace.
    IMPROVED: Wrapped in try/except — if Pinecone has no memory docs yet,
    this won't crash the pipeline.
    """
    try:
        retriever = vector_store.as_retriever(
            search_kwargs={
                "k": k,
                "filter": {"type": "memory"}
            }
        )
        results = retriever.invoke(query)
        return results
    except Exception as e:
        print(f"[Memory] Long-term retrieval failed (safe): {e}")
        return []