"""
OmniRAG — RAG Agent
────────────────────
Specialised agent that retrieves from the hybrid knowledge base
and formats context for the synthesis agent.
"""

import logging
from dataclasses import dataclass
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings
from app.rag.retriever import HybridRetriever

logger = logging.getLogger(__name__)

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a precise research agent specialising in document retrieval.

Given the retrieved context chunks below, extract and organise ALL relevant information 
that answers the query. Be thorough — include facts, figures, tables, and relationships.
If context is insufficient, say so clearly.

Format your response as structured markdown with clear sections.
Always cite the source and page number for each fact."""),
    ("human", "Query: {query}\n\nRetrieved Context:\n{context}"),
])


@dataclass
class RAGAgent:
    retriever: HybridRetriever

    def run(self, query: str) -> str:
        # Retrieve
        docs = self.retriever.get_relevant_documents(query)
        if not docs:
            logger.warning("RAG agent: no documents retrieved for: %s", query)
            return "No relevant documents found in the knowledge base."

        # Format context
        context_parts = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            context_parts.append(
                f"[Doc {i}] Source: {meta.get('source', 'unknown')} | "
                f"Page: {meta.get('page', '?')} | "
                f"Type: {meta.get('content_type', 'text')} | "
                f"Score: {meta.get('rerank_score', 'N/A'):.3f}\n\n"
                f"{doc.page_content}"
            )
        context = "\n\n" + "─" * 60 + "\n\n".join(context_parts)

        # LLM synthesis of retrieved docs
        llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=0,
        )
        chain = RAG_PROMPT | llm
        response = chain.invoke({"query": query, "context": context})
        logger.info("RAG agent completed for query: %s", query[:60])
        return response.content


def create_rag_agent() -> RAGAgent:
    retriever = HybridRetriever(
        top_k=settings.retrieval_top_k,
        reranker_top_n=settings.reranker_top_n,
        use_reranking=True,
        use_filter_extraction=True,
    )
    return RAGAgent(retriever=retriever)
