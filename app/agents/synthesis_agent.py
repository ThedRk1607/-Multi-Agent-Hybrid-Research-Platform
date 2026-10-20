"""
OmniRAG — Synthesis Agent
──────────────────────────
Merges RAG context + Web context into a final grounded answer.
"""

import logging
from dataclasses import dataclass
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings

logger = logging.getLogger(__name__)

SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a senior research analyst producing a final, well-structured answer.

You have two information sources:
1. Knowledge Base (RAG): Curated internal documents
2. Web Search: Live internet results

Instructions:
- Synthesise BOTH sources into one coherent, comprehensive answer
- Prioritise factual accuracy; cite sources inline [RAG: source/page] or [Web: URL]
- Highlight any contradictions between sources
- Use clear markdown formatting with headers, bullet points, and tables where useful
- End with a "Sources" section listing all references
- If information is missing, explicitly state what is unknown

Be thorough, accurate, and professional."""),
    ("human", """Question: {query}

--- RAG Context ---
{rag_context}

--- Web Context ---
{web_context}

Provide your comprehensive answer:"""),
])


@dataclass
class SynthesisAgent:
    def run(self, query: str, rag_context: str, web_context: str) -> str:
        llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=0.1,
        )
        chain = SYNTHESIS_PROMPT | llm
        response = chain.invoke({
            "query": query,
            "rag_context": rag_context or "No knowledge base context retrieved.",
            "web_context": web_context or "No web search performed.",
        })
        logger.info("Synthesis agent completed.")
        return response.content


def create_synthesis_agent() -> SynthesisAgent:
    return SynthesisAgent()
