"""
OmniRAG — Hybrid Retriever
──────────────────────────
Dense (Gemini) + Sparse (BM25) → Reciprocal Rank Fusion → Cross-Encoder Rerank.
Supports metadata filtering via LLM-extracted structured filters.
"""

import logging
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.config import settings
from app.rag.ingestion import get_vector_store
from app.rag.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


# ── Filter Extraction Schema ──────────────────────────────────────────────────

class RAGFilters(BaseModel):
    """Structured filters extracted from natural language query."""
    source: Optional[str] = Field(None, description="Document/file name filter")
    content_type: Optional[str] = Field(
        None, description="One of: text, table, image"
    )
    page: Optional[int] = Field(None, description="Specific page number")


FILTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a filter extraction assistant.
Given a user query, extract structured metadata filters if clearly present.
Return JSON matching the RAGFilters schema.
If no filter is apparent, return all nulls.

Schema:
- source: exact document name/identifier
- content_type: 'text', 'table', or 'image' only
- page: integer page number

Be conservative — only set a filter if the user explicitly mentions it."""),
    ("human", "Query: {query}"),
])


# ── Hybrid Retriever ──────────────────────────────────────────────────────────

class HybridRetriever(BaseRetriever):
    """
    Production-grade hybrid retriever:
    1. LLM extracts metadata filters from query
    2. Hybrid search (dense + sparse RRF) on Qdrant
    3. Cross-encoder reranking on top candidates
    """

    top_k: int = Field(default_factory=lambda: settings.retrieval_top_k)
    reranker_top_n: int = Field(default_factory=lambda: settings.reranker_top_n)
    use_reranking: bool = True
    use_filter_extraction: bool = True

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:

        # ── Step 1: Extract structured filters ───────────────────
        filters = {}
        if self.use_filter_extraction:
            filters = self._extract_filters(query)
            if filters:
                logger.info("Extracted filters: %s", filters)

        # ── Step 2: Hybrid vector search ──────────────────────────
        vs = get_vector_store()
        search_kwargs: Dict[str, Any] = {"k": self.top_k}
        if filters:
            search_kwargs["filter"] = self._build_qdrant_filter(filters)

        candidates = vs.similarity_search(query, **search_kwargs)
        logger.info("Retrieved %d candidates via hybrid search.", len(candidates))

        if not candidates:
            return []

        # ── Step 3: Cross-encoder reranking ───────────────────────
        if self.use_reranking and len(candidates) > self.reranker_top_n:
            reranker = CrossEncoderReranker(top_n=self.reranker_top_n)
            candidates = reranker.rerank(query, candidates)
            logger.info("Reranked to top %d results.", len(candidates))

        return candidates

    def _extract_filters(self, query: str) -> Dict[str, Any]:
        """Use LLM to extract metadata filters from query."""
        try:
            llm = ChatGoogleGenerativeAI(
                model=settings.gemini_model,
                google_api_key=settings.google_api_key,
                temperature=0,
            )
            structured_llm = llm.with_structured_output(RAGFilters)
            chain = FILTER_PROMPT | structured_llm
            result: RAGFilters = chain.invoke({"query": query})
            return {k: v for k, v in result.model_dump().items() if v is not None}
        except Exception as e:
            logger.warning("Filter extraction failed: %s", e)
            return {}

    def _build_qdrant_filter(self, filters: Dict[str, Any]) -> Dict:
        """Convert extracted filters to Qdrant filter format."""
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue
        conditions = []
        for key, value in filters.items():
            conditions.append(
                FieldCondition(key=key, match=MatchValue(value=value))
            )
        return Filter(must=conditions) if conditions else None


# ── Multi-Query Expander ──────────────────────────────────────────────────────

class MultiQueryRetriever:
    """
    Generates multiple query variations for broader retrieval coverage,
    then deduplicates results.
    """

    def __init__(self, base_retriever: HybridRetriever, num_queries: int = 3):
        self.base_retriever = base_retriever
        self.num_queries = num_queries

    def retrieve(self, query: str) -> List[Document]:
        queries = self._generate_queries(query)
        all_docs: Dict[str, Document] = {}

        for q in queries:
            docs = self.base_retriever.get_relevant_documents(q)
            for doc in docs:
                key = hashlib.md5(doc.page_content.encode()).hexdigest()
                all_docs[key] = doc

        return list(all_docs.values())[:self.base_retriever.reranker_top_n]

    def _generate_queries(self, query: str) -> List[str]:
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""Generate {self.num_queries} different search queries 
            that retrieve relevant information for the user's question.
            Return ONLY the queries, one per line, no numbering."""),
            ("human", "{query}"),
        ])
        llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=0.3,
        )
        response = (prompt | llm).invoke({"query": query})
        queries = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
        return [query] + queries[:self.num_queries - 1]


import hashlib
