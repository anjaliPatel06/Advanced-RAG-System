import re

def rewrite_query(query, llm):
    """
    IMPROVED: Generates ONE focused expanded query instead of verbose multi-line output.
    Also detects if query is too vague and adds domain context (Crawl4AI docs).
    """
    prompt = f"""You are a search query optimizer for a Crawl4AI documentation assistant.

Rewrite the user query into a single, focused search query (1-2 sentences max).
- Add relevant technical keywords if missing
- Keep it specific and searchable
- Do NOT answer the question, only rewrite the query
- Domain: Crawl4AI Python web crawling library documentation

User query: {query}

Rewritten query (one line only):"""

    result = llm.invoke(prompt).content.strip()
    result = re.sub(r'^(Rewritten query\s*[:\-]\s*)', '', result, flags=re.IGNORECASE)
    return result.split("\n")[0].strip()


def optimize_context(docs, memory, max_tokens=3000):
    """
    IMPROVED context optimization:
    - Deduplicates by content hash
    - Enforces approximate token budget (1 token ≈ 4 chars)
    - Prioritizes longer, more informative chunks
    - Clearly separates memory from retrieved context
    """
    seen = set()
    unique_docs = []

    sorted_docs = sorted(docs, key=lambda d: len(d.page_content), reverse=True)

    char_budget = max_tokens * 4  # rough chars-to-tokens ratio
    used_chars = 0

    for d in sorted_docs:
        text = d.page_content.strip()
        if text in seen or len(text) < 50:
            continue
        if used_chars + len(text) > char_budget:
            continue
        unique_docs.append(text)
        seen.add(text)
        used_chars += len(text)

    context_block = "\n\n---\n\n".join(unique_docs)
    memory_text = "\n".join(memory) if memory else ""

    if memory_text:
        return f"[Conversation History]\n{memory_text}\n\n[Retrieved Context]\n{context_block}"
    return f"[Retrieved Context]\n{context_block}"


def check_failure(docs, threshold=2):
    """
    IMPROVED failure detection:
    - Checks both count and top reranker score
    - docs can be list of (score, doc) tuples OR plain docs
    """
    if not docs:
        return True

    if isinstance(docs[0], tuple):
        scores = [s for s, _ in docs]
        valid = [s for s in scores if s > 0.25]
        if len(valid) < threshold:
            return True
        if max(scores) < 0.2:
            return True
        return False

    valid_docs = [d for d in docs if len(d.page_content.strip()) > 50]
    return len(valid_docs) < threshold


def extract_docs_from_ranked(ranked_docs):
    """Unwrap (score, doc) tuples → plain doc list."""
    if not ranked_docs:
        return []
    if isinstance(ranked_docs[0], tuple):
        return [doc for _, doc in ranked_docs]
    return ranked_docs


def detect_query_intent(query):
    """
    IMPROVED: Classify query intent to guide retrieval strategy.
    Returns one of: 'how_to', 'definition', 'comparison', 'error', 'general'
    """
    q = query.lower()
    if any(w in q for w in ["how to", "how do i", "how can i", "steps to", "example"]):
        return "how_to"
    if any(w in q for w in ["what is", "what are", "define", "explain"]):
        return "definition"
    if any(w in q for w in ["difference", "vs", "versus", "compare", "better"]):
        return "comparison"
    if any(w in q for w in ["error", "exception", "fail", "not working", "bug", "issue"]):
        return "error"
    return "general"
