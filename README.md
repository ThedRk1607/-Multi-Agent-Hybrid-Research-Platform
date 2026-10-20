# 🧠 OmniRAG — Multi-Agent Hybrid Research Platform




## 📌 What Is OmniRAG?

OmniRAG is an **advanced, production-ready research assistant** that intelligently answers questions by:

1. **Routing** your query to the right specialist agent(s)
2. **Retrieving** from a hybrid vector knowledge base (dense + sparse + reranking)
3. **Searching** the live web for current information
4. **Synthesising** both sources into a grounded, cited answer
5. **Critiquing** its own answer and iterating until quality passes

Every step is **fully traced in LangSmith** for observability, debugging, and evaluation.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       USER QUERY                            │
│                    (Streamlit UI / API)                      │
└───────────────────────────┬─────────────────────────────────┘
                            │
                    ┌───────▼────────┐
                    │   SUPERVISOR   │  ← LangGraph Node
                    │     AGENT      │  ← Routes to best path
                    └───┬───┬────┬───┘
                        │   │    │
              ┌─────────┘   │    └──────────┐
              │             │               │
      ┌───────▼───┐  ┌──────▼──────┐ ┌─────▼──────┐
      │  RAG      │  │    WEB      │ │   BOTH     │
      │  AGENT    │  │   AGENT     │ │ (Parallel) │
      │           │  │             │ │            │
      │ Hybrid    │  │  Tavily     │ │  RAG +     │
      │ Qdrant    │  │  Search     │ │  Web       │
      └─────┬─────┘  └──────┬──────┘ └─────┬──────┘
            │               │               │
            └───────────────┴───────────────┘
                            │
                   ┌────────▼─────────┐
                   │   SYNTHESIS      │  ← Merges all context
                   │     AGENT        │  ← Cites sources
                   └────────┬─────────┘
                            │
                   ┌────────▼─────────┐
                   │   CRITIQUE       │  ← Quality evaluator
                   │     AGENT        │  ← Loops if REVISE
                   └────────┬─────────┘
                            │
                   ┌────────▼─────────┐
                   │  FINAL ANSWER    │
                   │  + LangSmith     │
                   │    Trace Link    │
                   └──────────────────┘
```

### Hybrid RAG Pipeline

```
PDF / DOCX / TXT / MD
         │
    ┌────▼──────────────────────────────┐
    │           DOCLING                  │
    │  Text chunks │ Tables │ Images     │
    └────┬──────────────────────────────┘
         │
    ┌────▼──────────────────────────────┐
    │    DUAL EMBEDDING                  │
    │  Dense: Gemini-Embedding-001       │
    │  Sparse: FastEmbed BM25            │
    └────┬──────────────────────────────┘
         │
    ┌────▼──────────────────────────────┐
    │       QDRANT HYBRID STORE          │
    │   dense vectors + sparse vectors   │
    └────┬──────────────────────────────┘
         │
  QUERY ─┤
         │
    ┌────▼──────────────────────────────┐
    │  1. LLM Filter Extraction          │  ← source, page, type
    │  2. Hybrid Search (RRF fusion)     │  ← dense + sparse
    │  3. Cross-Encoder Rerank           │  ← BAAI/bge-reranker
    └────┬──────────────────────────────┘
         │
    Top-K Relevant Chunks
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose | Free Tier |
|---|---|---|---|
| **LLM** | Gemini 2.0 Flash | Agent reasoning, synthesis | ✅ 15 RPM |
| **Embeddings** | Gemini Embedding 001 | Dense 768-dim vectors | ✅ 1500 RPD |
| **Sparse** | FastEmbed BM25 | Keyword matching | ✅ Local |
| **Vector DB** | Qdrant Cloud | Hybrid storage & search | ✅ 1GB |
| **Reranker** | BAAI/bge-reranker-base | Cross-encoder scoring | ✅ Local |
| **Orchestration** | LangGraph | Multi-agent state machine | ✅ Open source |
| **Tracing** | LangSmith | Observability & evals | ✅ 5K traces/mo |
| **Web Search** | Tavily | Live internet search | ✅ 1K req/mo |
| **MCP Server** | Custom MCP | Reusable tool layer | ✅ Local |
| **Doc Processing** | Docling | PDF/DOCX extraction | ✅ Local |
| **UI** | Streamlit | Web interface | ✅ Streamlit Cloud |
| **CI/CD** | GitHub Actions | Automated deploy | ✅ 2000 min/mo |
| **Containers** | Docker | Reproducible builds | ✅ Docker Hub |
| **Hosting** | Render / Streamlit Cloud | Production deployment | ✅ Free tier |
| **Checkpoints** | SQLite | Conversation memory | ✅ Local |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker (optional)
- Free accounts for: Google AI Studio, LangSmith, Qdrant Cloud, Tavily

### 1. Clone & Setup

```bash
git clone https://github.com/yourusername/omnirag.git
cd omnirag
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys (see Free API Keys section below)
```

### 3. Ingest Documents

```bash
# Ingest a single PDF
python scripts/ingest.py --file data/my_document.pdf

# Ingest an entire folder
python scripts/ingest.py --dir data/docs/

# Ingest raw text
python scripts/ingest.py --text "Your content here" --label "my-notes"
```

### 4. Launch the App

```bash
streamlit run app/main.py
```

Open `http://localhost:8501` 🎉

### Docker Quick Start

```bash
# Copy and fill environment variables
cp .env.example .env

# Start everything
docker-compose up -d

# Check logs
docker-compose logs -f omnirag
```

---

## 🔑 Free API Keys (Step-by-Step)

### Google Gemini API (LLM + Embeddings)
1. Go to [aistudio.google.com](https://aistudio.google.com/apikey)
2. Click **Create API Key**
3. Copy to `GOOGLE_API_KEY` in `.env`
- Free: 15 requests/min, 1M tokens/day

### LangSmith (Tracing)
1. Sign up at [smith.langchain.com](https://smith.langchain.com)
2. Go to Settings → API Keys → Create
3. Copy to `LANGCHAIN_API_KEY` in `.env`
- Free: 5,000 traces/month

### Qdrant Cloud (Vector DB)
1. Sign up at [cloud.qdrant.io](https://cloud.qdrant.io)
2. Create a free cluster (1GB, always free)
3. Copy **Cluster URL** → `QDRANT_URL`
4. Copy **API Key** → `QDRANT_API_KEY`

### Tavily (Web Search)
1. Sign up at [tavily.com](https://tavily.com)
2. Go to dashboard → API Keys
3. Copy to `TAVILY_API_KEY` in `.env`
- Free: 1,000 requests/month

---

## 📊 LangSmith Observability

Every query is automatically traced. You'll see:

```
omnirag-production/
└── run: "What is the revenue trend?"
    ├── supervisor_node        (0.3s)  → route: both
    ├── rag_node               (1.2s)  → 5 docs retrieved
    │   ├── HybridRetriever    (0.8s)
    │   └── CrossEncoderRerank (0.4s)
    ├── web_node               (2.1s)  → 5 web results
    ├── synthesis_node         (1.8s)  → draft answer
    ├── critique_node          (0.9s)  → GOOD ✅
    └── TOTAL                  (6.3s)
```

**View traces at:** `https://smith.langchain.com/projects/omnirag-production`

### LangSmith Evaluations

```python
from langsmith import Client
from langsmith.evaluation import evaluate

client = Client()

# Run evaluation dataset
evaluate(
    lambda inputs: run_query(inputs["question"])["final_answer"],
    data="omnirag-eval-dataset",
    evaluators=["qa", "faithfulness", "hallucination"],
)
```

---

## 🔌 MCP Server

OmniRAG includes a standalone **Model Context Protocol server** with 5 tools:

| Tool | Description |
|---|---|
| `hybrid_search` | Search knowledge base with hybrid retrieval |
| `web_search` | Live Tavily web search |
| `summarise_docs` | Summarise retrieved document chunks |
| `extract_entities` | Named entity recognition on text |
| `calculate` | Safe math expression evaluator |

### Run MCP Server

```bash
# Standalone MCP server (stdio transport)
python -m app.mcp.server

# Via Docker
docker-compose up mcp_server
```

### Connect to Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "omnirag": {
      "command": "python",
      "args": ["-m", "app.mcp.server"],
      "cwd": "/path/to/omnirag",
      "env": {
        "GOOGLE_API_KEY": "your_key",
        "QDRANT_URL": "your_url",
        "QDRANT_API_KEY": "your_key",
        "TAVILY_API_KEY": "your_key"
      }
    }
  }
}
```

---

## 🌐 Deployment

### Option 1: Streamlit Cloud (Recommended — 100% Free)

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo, set `app/main.py` as entry point
4. Add secrets in dashboard (copy from `.env`)
5. Deploy ✅

### Option 2: Render (Free tier)

1. Push to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect repo, set:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run app/main.py --server.port $PORT --server.address 0.0.0.0`
4. Add environment variables from `.env`
5. Deploy ✅

### Option 3: Railway (Free $5 credit/month)

```bash
# Install Railway CLI
npm install -g @railway/cli
railway login
railway init
railway up
railway variables set GOOGLE_API_KEY=xxx LANGCHAIN_API_KEY=xxx # etc
```

### Option 4: Docker (Self-hosted)

```bash
docker build -t omnirag .
docker run -p 8501:8501 --env-file .env omnirag
```

### Option 5: HuggingFace Spaces (Free GPU)

1. Create a Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Select **Streamlit** as SDK
3. Push code to the Space repo
4. Add secrets in Space settings

---

## 📁 Project Structure

```
omnirag/
├── app/
│   ├── config.py               # Centralised settings (pydantic-settings)
│   ├── main.py                 # Streamlit production UI
│   ├── agents/
│   │   ├── rag_agent.py        # Hybrid retrieval + LLM synthesis
│   │   ├── web_agent.py        # Tavily live web search agent
│   │   ├── synthesis_agent.py  # Context merger + answer generator
│   │   └── critique_agent.py   # Quality evaluator (loop trigger)
│   ├── rag/
│   │   ├── ingestion.py        # Docling → chunk → Qdrant pipeline
│   │   ├── retriever.py        # Hybrid retriever + filter extraction
│   │   └── reranker.py         # Cross-encoder BAAI/bge reranker
│   ├── mcp/
│   │   └── server.py           # MCP server with 5 tools
│   └── graph/
│       └── workflow.py         # LangGraph multi-agent state machine
│
├── scripts/
│   └── ingest.py               # CLI ingestion tool
│
├── tests/
│   └── test_agents.py          # Unit tests (pytest)
│
├── .github/workflows/
│   └── ci-cd.yml               # GitHub Actions CI/CD
│
├── .streamlit/
│   ├── config.toml             # Streamlit theme config
│   └── secrets.toml.example    # Secrets template
│
├── data/                       # Local data + SQLite checkpoints
├── Dockerfile                  # Multi-stage production Docker image
├── docker-compose.yml          # Local dev with MCP server
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ Configuration Reference

All settings are in `.env` (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | required | Gemini LLM + embeddings |
| `LANGCHAIN_API_KEY` | required | LangSmith tracing |
| `QDRANT_URL` | required | Qdrant cloud cluster URL |
| `QDRANT_API_KEY` | required | Qdrant authentication |
| `TAVILY_API_KEY` | required | Web search |
| `RETRIEVAL_TOP_K` | `10` | Candidates before reranking |
| `RERANKER_TOP_N` | `5` | Final results after reranking |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `MAX_ITERATIONS` | `5` | Max critique-refine cycles |
| `AGENT_TEMPERATURE` | `0.0` | LLM determinism |

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html

# Run specific test class
pytest tests/test_agents.py::TestMCPCalculator -v

# Run async tests
pytest tests/ -v --asyncio-mode=auto
```

---

## 📈 Performance Benchmarks

| Metric | Value |
|---|---|
| Average response time | ~4–8 seconds |
| Ingestion speed | ~50 chunks/second |
| Reranking latency | ~200ms (local) |
| Hybrid search latency | ~300ms (Qdrant Cloud) |
| Max document size | 200MB (Streamlit limit) |
| Concurrent users | Limited by Gemini free tier (15 RPM) |

---

## 🔒 Security

- API keys loaded from environment variables only — never hardcoded
- Docker runs as non-root user (`omnirag:1000`)
- MCP `calculate` tool uses safe AST evaluator (no `eval()`)
- Streamlit XSRF protection enabled
- Input validation on all agent inputs
- Rate limiting via Gemini/Tavily free tier limits

---

## 🗺️ Roadmap

- [ ] Async parallel agent execution (LangGraph async)
- [ ] Streaming responses (SSE)
- [ ] Multi-modal image understanding in RAG
- [ ] Graph RAG with entity relationships
- [ ] Evaluation dashboard with LangSmith datasets
- [ ] OpenTelemetry metrics export
- [ ] Redis caching for repeated queries
- [ ] Multi-user session isolation

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Commit with conventional commits: `git commit -m "feat: add streaming"`
4. Push and open a PR

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

