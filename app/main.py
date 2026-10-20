"""
OmniRAG — Streamlit Production UI
────────────────────────────────────
Clean, production-grade chat interface with:
- File ingestion sidebar
- Chat history
- Source attribution display
- Agent trace visibility
- LangSmith trace link
"""

import os
import time
import uuid
import logging
import tempfile
import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="OmniRAG — Multi-Agent Research",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Set env vars before imports
os.environ["LANGCHAIN_TRACING_V2"] = st.secrets.get("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_API_KEY"] = st.secrets.get("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = st.secrets.get("LANGCHAIN_PROJECT", "omnirag-production")
os.environ["GOOGLE_API_KEY"] = st.secrets.get("GOOGLE_API_KEY", "")
os.environ["QDRANT_URL"] = st.secrets.get("QDRANT_URL", "")
os.environ["QDRANT_API_KEY"] = st.secrets.get("QDRANT_API_KEY", "")
os.environ["TAVILY_API_KEY"] = st.secrets.get("TAVILY_API_KEY", "")

from app.graph.workflow import run_query
from app.rag.ingestion import ingest_documents, extract_from_text

logging.basicConfig(level=logging.INFO)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1B3A6B 0%, #2E5EAA 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .agent-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-rag    { background: #E8F5E9; color: #2E7D32; }
    .badge-web    { background: #E3F2FD; color: #1565C0; }
    .badge-both   { background: #F3E5F5; color: #6A1B9A; }
    .source-card {
        background: #F8F9FA;
        border-left: 4px solid #2E5EAA;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
        font-size: 0.85rem;
    }
    .critique-pass { color: #2E7D32; font-weight: 600; }
    .critique-fail { color: #C62828; font-weight: 600; }
    .metric-card {
        background: white;
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = []
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1 style="margin:0;font-size:1.8rem">🧠 OmniRAG</h1>
    <p style="margin:0.3rem 0 0 0;opacity:0.9">
        Multi-Agent Hybrid Research Platform &nbsp;|&nbsp; 
        LangGraph · MCP · LangSmith · Hybrid RAG
    </p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📁 Knowledge Base")
    st.caption("Upload documents to enrich the RAG knowledge base")

    uploaded_files = st.file_uploader(
        "Upload files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        help="Supports PDF, DOCX, TXT, Markdown",
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.ingested_files:
                with st.spinner(f"Processing {uploaded_file.name}…"):
                    try:
                        with tempfile.NamedTemporaryFile(
                            suffix=os.path.splitext(uploaded_file.name)[1],
                            delete=False
                        ) as tmp:
                            tmp.write(uploaded_file.read())
                            tmp_path = tmp.name

                        from app.rag.ingestion import ingest_file
                        count = ingest_file(tmp_path)
                        os.unlink(tmp_path)
                        st.session_state.ingested_files.append(uploaded_file.name)
                        st.success(f"✅ {uploaded_file.name} — {count} chunks ingested")
                    except Exception as e:
                        st.error(f"❌ {uploaded_file.name}: {e}")

    st.divider()

    # Quick text ingestion
    with st.expander("📝 Ingest Text Directly"):
        raw_text = st.text_area("Paste text content", height=120)
        text_label = st.text_input("Label/source name", value="manual-input")
        if st.button("Ingest Text", use_container_width=True):
            if raw_text.strip():
                docs = extract_from_text(raw_text, source=text_label)
                count = ingest_documents(docs)
                st.success(f"✅ Ingested {count} chunks")
            else:
                st.warning("Please enter some text first.")

    st.divider()

    # Ingested files list
    if st.session_state.ingested_files:
        st.subheader("📚 Ingested Files")
        for f in st.session_state.ingested_files:
            st.markdown(f"• {f}")

    st.divider()

    # Settings
    st.subheader("⚙️ Settings")
    show_context = st.toggle("Show retrieved context", value=False)
    show_critique = st.toggle("Show agent critique", value=False)
    show_trace = st.toggle("Show LangSmith trace", value=True)

    st.divider()

    # Stats
    st.subheader("📊 Session Stats")
    col1, col2 = st.columns(2)
    col1.metric("Queries", st.session_state.total_queries)
    col2.metric("Files", len(st.session_state.ingested_files))

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()


# ── Chat Area ─────────────────────────────────────────────────────────────────
chat_col, info_col = st.columns([3, 1])

with chat_col:
    # Render history
    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(turn["query"])
        with st.chat_message("assistant"):
            st.markdown(turn["answer"])

            # Agent route badge
            route = turn.get("route", "")
            badge_class = {
                "rag_agent": "badge-rag",
                "web_agent": "badge-web",
                "both": "badge-both",
            }.get(route, "badge-both")
            st.markdown(
                f'<span class="agent-badge {badge_class}">🔀 {route or "both"}</span>'
                f'<span class="agent-badge" style="background:#FFF9C4;color:#F57F17">'
                f'⚡ {turn.get("iterations", 1)} iteration(s)</span>',
                unsafe_allow_html=True,
            )

            if show_context and (turn.get("rag_context") or turn.get("web_context")):
                with st.expander("📖 Retrieved Context"):
                    if turn.get("rag_context"):
                        st.markdown("**📚 Knowledge Base:**")
                        st.markdown(turn["rag_context"])
                    if turn.get("web_context"):
                        st.markdown("**🌐 Web Search:**")
                        st.markdown(turn["web_context"])

            if show_critique and turn.get("critique"):
                critique = turn["critique"]
                cls = "critique-pass" if "GOOD" in critique.upper() else "critique-fail"
                with st.expander("🔍 Agent Critique"):
                    st.markdown(f'<span class="{cls}">{critique}</span>', unsafe_allow_html=True)

    # Input
    if query := st.chat_input("Ask anything — searches your documents + the web…"):
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("🤔 Thinking… (Routing → Retrieval → Synthesis → Evaluation)")

            start = time.time()
            try:
                result = run_query(query, thread_id=st.session_state.thread_id)
                elapsed = time.time() - start

                answer = result.get("final_answer") or "No answer generated."
                placeholder.markdown(answer)

                # Route badge
                route = result.get("route", "")
                badge_class = {
                    "rag_agent": "badge-rag",
                    "web_agent": "badge-web",
                    "both": "badge-both",
                }.get(route, "badge-both")
                st.markdown(
                    f'<span class="agent-badge {badge_class}">🔀 {route or "both"}</span>'
                    f'<span class="agent-badge" style="background:#FFF9C4;color:#F57F17">'
                    f'⚡ {result.get("iterations", 1)} iter</span>'
                    f'<span class="agent-badge" style="background:#ECEFF1;color:#455A64">'
                    f'⏱️ {elapsed:.1f}s</span>',
                    unsafe_allow_html=True,
                )

                if show_context:
                    with st.expander("📖 Retrieved Context"):
                        if result.get("rag_context"):
                            st.markdown("**📚 Knowledge Base:**")
                            st.markdown(result["rag_context"])
                        if result.get("web_context"):
                            st.markdown("**🌐 Web Search:**")
                            st.markdown(result["web_context"])

                if show_critique and result.get("critique"):
                    critique = result["critique"]
                    cls = "critique-pass" if "GOOD" in critique.upper() else "critique-fail"
                    with st.expander("🔍 Agent Critique"):
                        st.markdown(f'<span class="{cls}">{critique}</span>', unsafe_allow_html=True)

                if show_trace:
                    project = os.environ.get("LANGCHAIN_PROJECT", "omnirag-production")
                    st.caption(f"📡 [View LangSmith Trace](https://smith.langchain.com/projects/{project})")

                # Save to history
                st.session_state.chat_history.append({
                    "query": query,
                    "answer": answer,
                    **result,
                })
                st.session_state.total_queries += 1

            except Exception as e:
                placeholder.error(f"❌ Error: {e}")
                logging.exception("Query failed")


# ── Right Info Panel ──────────────────────────────────────────────────────────
with info_col:
    st.subheader("🏗️ Architecture")
    st.markdown("""
**Pipeline:**
1. 🎯 **Supervisor** routes query
2. 📚 **RAG Agent** — Hybrid retrieval (Dense + BM25 → Rerank)
3. 🌐 **Web Agent** — Tavily live search
4. 🔗 **Synthesis** — Merges both sources
5. ✅ **Critique** — Quality evaluator (loops if needed)

**Tech Stack:**
- LangGraph (multi-agent)
- LangSmith (tracing)
- Qdrant (vector DB)
- Gemini 2.0 Flash (LLM)
- BM25 + Dense Hybrid
- Cross-encoder rerank
- MCP server
- Streamlit UI
""")
    st.divider()
    st.subheader("💡 Try asking:")
    sample_queries = [
        "What are the key findings in the uploaded report?",
        "Summarise the latest AI developments",
        "What does the data show about revenue trends?",
        "Compare the information in my documents with current news",
    ]
    for q in sample_queries:
        if st.button(q, key=q, use_container_width=True):
            st.session_state["_prefill"] = q
            st.rerun()
