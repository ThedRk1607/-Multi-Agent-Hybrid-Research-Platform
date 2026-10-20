"""
OmniRAG — Web Search Agent
───────────────────────────
Performs live web search via Tavily (free tier: 1000 req/month)
and formats results for the synthesis agent.
"""

import logging
from dataclasses import dataclass
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults

from app.config import settings

logger = logging.getLogger(__name__)

WEB_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a web research agent. Given live search results below,
extract, organise, and present ALL relevant information that answers the query.

Include:
- Key facts and figures
- Source URLs
- Publication dates where available
- Conflicting information if present

Format as structured markdown. Be thorough but concise."""),
    ("human", "Query: {query}\n\nSearch Results:\n{results}"),
])


@dataclass
class WebAgent:
    max_results: int = 5

    def run(self, query: str) -> str:
        try:
            tool = TavilySearchResults(
                max_results=self.max_results,
                api_key=settings.tavily_api_key,
            )
            raw_results = tool.invoke({"query": query})

            if not raw_results:
                return "No web results found."

            formatted = []
            for i, r in enumerate(raw_results, 1):
                formatted.append(
                    f"[Web {i}] {r.get('title', 'No Title')}\n"
                    f"URL: {r.get('url', '')}\n"
                    f"Content: {r.get('content', '')}"
                )
            results_text = "\n\n---\n\n".join(formatted)

            llm = ChatGoogleGenerativeAI(
                model=settings.gemini_model,
                google_api_key=settings.google_api_key,
                temperature=0,
            )
            chain = WEB_PROMPT | llm
            response = chain.invoke({"query": query, "results": results_text})
            logger.info("Web agent completed for query: %s", query[:60])
            return response.content

        except Exception as e:
            logger.error("Web agent error: %s", e)
            return f"Web search failed: {e}"


def create_web_agent() -> WebAgent:
    return WebAgent(max_results=5)
