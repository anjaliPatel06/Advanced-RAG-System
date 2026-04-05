import streamlit as st
from app import rag_pipeline
from memory import record_feedback, get_feedback_summary

st.set_page_config(
    page_title="Crawl4AI Assistant",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', sans-serif;
    background: #faf8f5;
    color: #1c1917;
}

.stApp { background: #faf8f5 !important; }

/* Hide default streamlit sidebar toggle & header */
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }
header[data-testid="stHeader"] { display: none !important; }
#MainMenu { display: none !important; }
footer { display: none !important; }

.main .block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ══════════════════════════════════════
   FULL PAGE LAYOUT
══════════════════════════════════════ */
.app-shell {
    display: grid;
    grid-template-columns: 1fr 300px;
    grid-template-rows: 56px 1fr;
    height: 100vh;
    width: 100%;
    overflow: hidden;
}

/* ── TOPNAV ── */
.topnav {
    grid-column: 1 / -1;
    background: #faf8f5;
    border-bottom: 1px solid #e8e1d9;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 28px;
    height: 56px;
    position: sticky;
    top: 0;
    z-index: 50;
}

.nav-brand {
    display: flex;
    align-items: center;
    gap: 10px;
}

.nav-icon {
    width: 30px;
    height: 30px;
    background: #1c1917;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    flex-shrink: 0;
}

.nav-title {
    font-size: 15px;
    font-weight: 600;
    color: #1c1917;
    letter-spacing: -0.02em;
}

.nav-subtitle {
    font-size: 12px;
    color: #a8a29e;
    font-weight: 400;
}

.nav-right {
    display: flex;
    align-items: center;
    gap: 16px;
}

.nav-status {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    color: #a8a29e;
}

.nav-dot {
    width: 6px;
    height: 6px;
    background: #a3be8c;
    border-radius: 50%;
    animation: glow 2.5s ease-in-out infinite;
}

@keyframes glow {
    0%, 100% { opacity: 0.8; }
    50% { opacity: 1; box-shadow: 0 0 6px #a3be8c; }
}

.nav-pill {
    font-size: 11px;
    font-weight: 500;
    color: #78716c;
    background: #f0ece6;
    border: 1px solid #e8e1d9;
    border-radius: 99px;
    padding: 3px 10px;
    letter-spacing: 0.02em;
}

/* ── CHAT AREA ── */
.chat-area {
    grid-column: 1;
    display: flex;
    flex-direction: column;
    height: calc(100vh - 56px);
    overflow: hidden;
    background: #faf8f5;
}

.chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 32px 48px;
    scroll-behavior: smooth;
}

.chat-messages::-webkit-scrollbar { width: 4px; }
.chat-messages::-webkit-scrollbar-track { background: transparent; }
.chat-messages::-webkit-scrollbar-thumb { background: #e8e1d9; border-radius: 99px; }

/* ── RIGHT PANEL ── */
.right-panel {
    grid-column: 2;
    background: #f5f1eb;
    border-left: 1px solid #e8e1d9;
    height: calc(100vh - 56px);
    overflow-y: auto;
    padding: 24px 20px;
}

.right-panel::-webkit-scrollbar { width: 3px; }
.right-panel::-webkit-scrollbar-thumb { background: #e8e1d9; border-radius: 99px; }

.panel-section { margin-bottom: 28px; }

.panel-label {
    font-size: 10px;
    font-weight: 600;
    color: #a8a29e;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 12px;
    display: block;
}

.pipeline-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 0;
    font-size: 12px;
    color: #78716c;
    border-bottom: 1px solid #ece7e0;
}

.pipeline-row:last-child { border-bottom: none; }

.p-dot {
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: #a3be8c;
    flex-shrink: 0;
}

.model-card {
    background: #faf8f5;
    border: 1px solid #e8e1d9;
    border-radius: 10px;
    overflow: hidden;
}

.model-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 12px;
    border-bottom: 1px solid #f0ece6;
    font-size: 12px;
}

.model-row:last-child { border-bottom: none; }
.model-key { color: #a8a29e; }
.model-val { color: #44403c; font-family: 'DM Mono', monospace; font-size: 11px; }

.feedback-box {
    background: #faf8f5;
    border: 1px solid #e8e1d9;
    border-radius: 10px;
    padding: 12px;
    font-size: 12px;
    color: #78716c;
    font-family: 'DM Mono', monospace;
    line-height: 1.6;
}

/* ══════════════════════════════════════
   EMPTY STATE
══════════════════════════════════════ */
.empty-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    text-align: center;
    padding: 40px;
}

.empty-icon-box {
    width: 56px;
    height: 56px;
    background: #f0ece6;
    border: 1px solid #e8e1d9;
    border-radius: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 26px;
    margin-bottom: 20px;
}

.empty-h {
    font-size: 20px;
    font-weight: 600;
    color: #1c1917;
    letter-spacing: -0.03em;
    margin-bottom: 8px;
}

.empty-p {
    font-size: 13px;
    color: #a8a29e;
    line-height: 1.65;
    max-width: 380px;
    margin-bottom: 32px;
}

.chips {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
    max-width: 560px;
}

.chip {
    background: #fff;
    border: 1px solid #e8e1d9;
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 12.5px;
    color: #57534e;
    cursor: default;
    transition: all 0.15s;
    font-family: 'DM Sans', sans-serif;
}

.chip:hover {
    border-color: #c5b9ab;
    color: #1c1917;
    background: #f5f1eb;
}

/* ══════════════════════════════════════
   CHAT MESSAGES STYLING
══════════════════════════════════════ */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 20px 0 !important;
    border-bottom: 1px solid #f0ece6 !important;
    margin-bottom: 0 !important;
    max-width: 740px;
    margin-left: auto;
    margin-right: auto;
}

[data-testid="stChatMessage"]:last-child {
    border-bottom: none !important;
}

[data-testid="chatAvatarIcon-user"] {
    background: #e8e1d9 !important;
    color: #78716c !important;
    border-radius: 8px !important;
    border: 1px solid #ddd6cc !important;
    width: 30px !important;
    height: 30px !important;
    font-size: 13px !important;
}

[data-testid="chatAvatarIcon-assistant"] {
    background: #1c1917 !important;
    color: #faf8f5 !important;
    border-radius: 8px !important;
    width: 30px !important;
    height: 30px !important;
    font-size: 13px !important;
}

[data-testid="stChatMessage"] p {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    line-height: 1.8 !important;
    color: #292524 !important;
}

[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3 {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    color: #1c1917 !important;
    letter-spacing: -0.02em !important;
    margin: 18px 0 6px !important;
}
[data-testid="stChatMessage"] h1 { font-size: 17px !important; }
[data-testid="stChatMessage"] h2 { font-size: 15px !important; }
[data-testid="stChatMessage"] h3 { font-size: 14px !important; }

[data-testid="stChatMessage"] code {
    font-family: 'DM Mono', monospace !important;
    background: #f0ece6 !important;
    color: #9c6644 !important;
    border: 1px solid #e8e1d9 !important;
    border-radius: 4px !important;
    padding: 1px 6px !important;
    font-size: 12px !important;
}

[data-testid="stChatMessage"] pre {
    background: #1c1917 !important;
    border-radius: 10px !important;
    padding: 18px !important;
    margin: 12px 0 !important;
    border: none !important;
}

[data-testid="stChatMessage"] pre code {
    background: transparent !important;
    border: none !important;
    color: #d6cfc7 !important;
    font-size: 12.5px !important;
    padding: 0 !important;
}

[data-testid="stChatMessage"] ul,
[data-testid="stChatMessage"] ol {
    padding-left: 18px !important;
    color: #292524 !important;
    font-size: 14px !important;
    line-height: 1.9 !important;
}

[data-testid="stChatMessage"] strong {
    color: #1c1917 !important;
    font-weight: 600 !important;
}

[data-testid="stChatMessage"] blockquote {
    border-left: 2px solid #ddd6cc !important;
    padding-left: 14px !important;
    color: #a8a29e !important;
    margin: 10px 0 !important;
    font-style: italic;
}

/* ══════════════════════════════════════
   FEEDBACK BUTTONS
══════════════════════════════════════ */
[data-testid="stButton"] button {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 12px !important;
    font-weight: 400 !important;
    background: transparent !important;
    border: 1px solid #e8e1d9 !important;
    color: #a8a29e !important;
    border-radius: 6px !important;
    padding: 2px 10px !important;
    transition: all 0.12s !important;
}

[data-testid="stButton"] button:hover {
    background: #f0ece6 !important;
    border-color: #c5b9ab !important;
    color: #57534e !important;
}

.stCaption p {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 11px !important;
    color: #a8a29e !important;
}

/* ══════════════════════════════════════
   CHAT INPUT
══════════════════════════════════════ */
[data-testid="stChatInput"] {
    background: #fff !important;
    border: 1px solid #e8e1d9 !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 8px rgba(0,0,0,0.05), 0 4px 16px rgba(0,0,0,0.04) !important;
    max-width: 740px !important;
    margin: 0 auto !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: #c5b9ab !important;
    box-shadow: 0 1px 8px rgba(0,0,0,0.07), 0 4px 20px rgba(0,0,0,0.06) !important;
}

[data-testid="stChatInput"] textarea {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    color: #1c1917 !important;
    background: transparent !important;
    line-height: 1.6 !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #c5b9ab !important;
    font-size: 13.5px !important;
}

.stChatInputContainer {
    background: #faf8f5 !important;
    padding: 14px 48px 20px !important;
    border-top: 1px solid #f0ece6 !important;
    max-width: 100% !important;
}

hr { border-color: #f0ece6 !important; }
em { color: #a8a29e !important; font-style: italic; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "feedback" not in st.session_state:
    st.session_state.feedback = {}

# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT — two columns: chat | right panel
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="topnav">
    <div class="nav-brand">
        <div class="nav-icon">🕸️</div>
        <div>
            <div class="nav-title">Crawl4AI</div>
        </div>
        <span class="nav-pill">Documentation Assistant</span>
    </div>
    <div class="nav-right">
        <div class="nav-status">
            <div class="nav-dot"></div>
            <span>All systems online</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

col_chat, col_panel = st.columns([3, 1], gap="small")

# ── RIGHT PANEL ───────────────────────────────────────────────────────────────
with col_panel:
    st.markdown("""
    <div class="right-panel">

      <div class="panel-section">
        <span class="panel-label">Pipeline</span>
        <div class="pipeline-row"><span class="p-dot"></span>Semantic · Pinecone</div>
        <div class="pipeline-row"><span class="p-dot"></span>BM25 keyword</div>
        <div class="pipeline-row"><span class="p-dot"></span>Rank Fusion · RRF</div>
        <div class="pipeline-row"><span class="p-dot"></span>Cross-encoder rerank</div>
        <div class="pipeline-row"><span class="p-dot"></span>Short + long memory</div>
        <div class="pipeline-row"><span class="p-dot"></span>Query rewriting</div>
        <div class="pipeline-row"><span class="p-dot"></span>Hallucination guard</div>
      </div>

      <div class="panel-section">
        <span class="panel-label">Model</span>
        <div class="model-card">
            <div class="model-row"><span class="model-key">LLM</span><span class="model-val">llama-3.1-8b</span></div>
            <div class="model-row"><span class="model-key">Embed</span><span class="model-val">MiniLM-L6</span></div>
            <div class="model-row"><span class="model-key">Rank</span><span class="model-val">ms-marco</span></div>
            <div class="model-row"><span class="model-key">Infra</span><span class="model-val">Groq + Pinecone</span></div>
        </div>
      </div>

    </div>
    """, unsafe_allow_html=True)

    st.markdown('<span class="panel-label" style="padding-left:0;">Feedback</span>', unsafe_allow_html=True)
    st.markdown(f'<div class="feedback-box">{get_feedback_summary()}</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    if st.button("↺  New conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.feedback = {}
        st.rerun()

# ── CHAT AREA ─────────────────────────────────────────────────────────────────
with col_chat:

    # Empty state
    if not st.session_state.messages:
        st.markdown("""
        <div class="empty-wrap">
            <div class="empty-icon-box">🕸️</div>
            <div class="empty-h">How can I help?</div>
            <div class="empty-p">
                Ask anything about Crawl4AI — installation, async crawling,
                data extraction, JavaScript handling, or the full API reference.
            </div>
            <div class="chips">
                <div class="chip">How do I install Crawl4AI?</div>
                <div class="chip">What is AsyncWebCrawler?</div>
                <div class="chip">How to extract structured data?</div>
                <div class="chip">CSS selectors in extraction</div>
                <div class="chip">What is CrawlResult?</div>
                <div class="chip">Handle JavaScript pages</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Chat messages
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg["role"] == "assistant":
                feedback_key = f"feedback_{i}"
                if feedback_key not in st.session_state.feedback:
                    c1, c2, _ = st.columns([1, 1, 10])
                    with c1:
                        if st.button("👍", key=f"up_{i}"):
                            st.session_state.feedback[feedback_key] = True
                            user_query = ""
                            if i > 0 and st.session_state.messages[i-1]["role"] == "user":
                                user_query = st.session_state.messages[i-1]["content"]
                            record_feedback(user_query, msg["content"], was_helpful=True)
                            st.rerun()
                    with c2:
                        if st.button("👎", key=f"down_{i}"):
                            st.session_state.feedback[feedback_key] = False
                            user_query = ""
                            if i > 0 and st.session_state.messages[i-1]["role"] == "user":
                                user_query = st.session_state.messages[i-1]["content"]
                            record_feedback(user_query, msg["content"], was_helpful=False)
                            st.rerun()
                else:
                    val = st.session_state.feedback[feedback_key]
                    st.caption("✅ Helpful" if val else "❌ Not helpful")

    # Input
    user_input = st.chat_input("Ask anything about Crawl4AI...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            placeholder = st.empty()
            response = ""
            placeholder.markdown("_Searching documentation..._")
            for token in rag_pipeline(user_input):
                response += token
                placeholder.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()