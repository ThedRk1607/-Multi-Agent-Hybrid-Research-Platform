"""
OmniRAG Configuration — centralised settings via pydantic-settings.
All values come from environment variables (see .env.example).
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    # ── LLM ─────────────────────────────────────────────────────
    google_api_key: str = Field(..., env="GOOGLE_API_KEY")
    gemini_model: str = Field("gemini-2.0-flash", env="GEMINI_MODEL")
    gemini_embedding_model: str = Field(
        "models/gemini-embedding-001", env="GEMINI_EMBEDDING_MODEL"
    )

    # ── LangSmith Tracing ────────────────────────────────────────
    langchain_tracing_v2: str = Field("true", env="LANGCHAIN_TRACING_V2")
    langchain_api_key: str = Field(..., env="LANGCHAIN_API_KEY")
    langchain_project: str = Field("omnirag-production", env="LANGCHAIN_PROJECT")
    langchain_endpoint: str = Field(
        "https://api.smith.langchain.com", env="LANGCHAIN_ENDPOINT"
    )

    # ── Qdrant Vector DB (free cloud tier) ───────────────────────
    qdrant_url: str = Field(..., env="QDRANT_URL")
    qdrant_api_key: str = Field(..., env="QDRANT_API_KEY")
    qdrant_collection: str = Field("omnirag_hybrid", env="QDRANT_COLLECTION")

    # ── Tavily Web Search (free tier) ────────────────────────────
    tavily_api_key: str = Field(..., env="TAVILY_API_KEY")

    # ── RAG Retrieval Tuning ─────────────────────────────────────
    retrieval_top_k: int = Field(10, env="RETRIEVAL_TOP_K")
    reranker_top_n: int = Field(5, env="RERANKER_TOP_N")
    chunk_size: int = Field(1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(200, env="CHUNK_OVERLAP")

    # ── Agent Settings ───────────────────────────────────────────
    max_iterations: int = Field(5, env="MAX_ITERATIONS")
    agent_temperature: float = Field(0.0, env="AGENT_TEMPERATURE")

    # ── App ──────────────────────────────────────────────────────
    app_title: str = "OmniRAG — Multi-Agent Hybrid Research Platform"
    data_dir: str = "data"
    log_level: str = Field("INFO", env="LOG_LEVEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
