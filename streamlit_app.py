import streamlit as st
from app import rag_pipeline
from memory import record_feedback, get_feedback_summary

st.set_page_config(
    page_title="Crawl4AI RAG Assistant",
    layout="centered"
)

st.title("Crawl4AI Documentation Assistant")
st.caption("Powered by Advanced Self-Improving RAG · Hybrid Search · Cross-Encoder Reranking")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "feedback" not in st.session_state:
    st.session_state.feedback = {}  

with st.sidebar:
    st.header("⚙️ System Info")
    st.markdown("""
    **Retrieval Pipeline:**
    - Semantic (Pinecone) + BM25 keyword
    - Reciprocal Rank Fusion
    - Cross-Encoder Reranking
    - Short + Long-term Memory
    - Query Rewriting
    - Hallucination Guard
    """)

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.feedback = {}
        st.rerun()

    st.divider()
    st.markdown("**Feedback Summary**")
    st.info(get_feedback_summary())

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            feedback_key = f"feedback_{i}"
            if feedback_key not in st.session_state.feedback:
                col1, col2, _ = st.columns([1, 1, 8])
                with col1:
                    if st.button("👍", key=f"up_{i}"):
                        st.session_state.feedback[feedback_key] = True
                        user_query = ""
                        if i > 0 and st.session_state.messages[i - 1]["role"] == "user":
                            user_query = st.session_state.messages[i - 1]["content"]
                        record_feedback(user_query, msg["content"], was_helpful=True)
                        st.rerun()
                with col2:
                    if st.button("👎", key=f"down_{i}"):
                        st.session_state.feedback[feedback_key] = False
                        user_query = ""
                        if i > 0 and st.session_state.messages[i - 1]["role"] == "user":
                            user_query = st.session_state.messages[i - 1]["content"]
                        record_feedback(user_query, msg["content"], was_helpful=False)
                        st.rerun()
            else:
                val = st.session_state.feedback[feedback_key]
                st.caption("Helpful" if val else "Not helpful")

user_input = st.chat_input("Ask about Crawl4AI (installation, crawling, API, extraction...)")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        response = ""

        placeholder.markdown("_Retrieving relevant documentation..._")

        for token in rag_pipeline(user_input):
            response += token
            placeholder.markdown(response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

    st.rerun()
