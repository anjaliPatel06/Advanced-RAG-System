import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from retriever import get_relevant_docs
from memory import (
    get_short_memory,
    update_short_memory,
    store_long_term,
    get_long_term
)
from utils import (
    optimize_context,
    rewrite_query,
    check_failure,
    extract_docs_from_ranked,
    detect_query_intent
)
from ranker import get_top_score

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    streaming=True
)

GREETINGS = {"hi", "hello", "hey", "hii", "helo", "hiya", "yo"}


def is_greeting(query):
    tokens = set(query.lower().strip().split())
    return bool(tokens & GREETINGS) and len(tokens) <= 3


def build_prompt(query, full_context, intent):
    """
    IMPROVED: Intent-aware prompt construction.
    How-to queries get step-format instruction; comparison gets table hint.
    Hallucination guard is always present.
    """
    intent_instruction = {
        "how_to": "If the answer involves steps, list them clearly and in order.",
        "definition": "Give a clear, concise definition using only the context provided.",
        "comparison": "If comparing, highlight key differences from the context only.",
        "error": "Focus on the error cause and fix as described in the context.",
        "general": "Answer concisely and directly."
    }.get(intent, "Answer concisely and directly.")

    return f"""You are a precise AI assistant for Crawl4AI documentation.

STRICT RULES:
1. Answer ONLY using the provided context below. Never use outside knowledge.
2. If the context does not contain the answer, respond EXACTLY: "I don't have enough information to answer this based on the available documentation."
3. Do NOT suggest users "refer to documentation" — you ARE the documentation assistant.
4. Do NOT fabricate function names, parameters, or examples not present in the context.
5. {intent_instruction}
6. Cite the relevant section or concept when possible.

Context:
{full_context}

Question: {query}

Answer:"""


def rag_pipeline(query):
    """
    Main RAG pipeline with:
    - Greeting shortcut
    - Query rewriting
    - Hybrid retrieval + reranking
    - Confidence-gated failure detection
    - Intent-aware prompting
    - Memory read + write
    - Streaming output
    """
    query = query.strip()

    if is_greeting(query):
        yield "Hello! I'm your Crawl4AI documentation assistant. Ask me anything about installation, crawling, extraction, or the API!"
        return

    rewritten = rewrite_query(query, llm)
    print(f"\n[Query Rewrite] {query!r} → {rewritten!r}")

    short_memory = get_short_memory()

    long_memory_docs = get_long_term(rewritten)
    long_memory_text = "\n".join([d.page_content for d in long_memory_docs])

    ranked_docs = get_relevant_docs(rewritten)

    if check_failure(ranked_docs):
        top = get_top_score(ranked_docs)
        print(f"[Failure] Top reranker score: {top:.3f} — below confidence threshold")
        yield "I don't have enough information to answer this based on the available documentation."
        return

    plain_docs = extract_docs_from_ranked(ranked_docs)
    context = optimize_context(plain_docs, short_memory)

    if long_memory_text.strip():
        context += f"\n\n[Past Relevant Answers]\n{long_memory_text}"

    intent = detect_query_intent(query)

    prompt = build_prompt(query, context, intent)

    print(f"\n--- CONTEXT PREVIEW ---\n{context[:800]}\n-----------------------\n")

    full_answer = ""
    for chunk in llm.stream(prompt):
        token = chunk.content or ""
        full_answer += token
        yield token

    update_short_memory(query, full_answer)
    if is_useful_answer(full_answer):
        store_long_term(query, full_answer)


def is_useful_answer(answer):
    if not answer or len(answer.strip()) < 30:
        return False
    bad = ["not enough information", "i don't have", "i cannot", "refer to"]
    return not any(p in answer.lower() for p in bad)


if __name__ == "__main__":
    while True:
        q = input("\nAsk: ").strip()
        if not q:
            continue
        print("\nAnswer: ", end="")
        for token in rag_pipeline(q):
            print(token, end="", flush=True)
        print()
