"""
OmniRAG — MCP Server
─────────────────────
Custom Model Context Protocol server exposing tools:
  • hybrid_search      — search the knowledge base
  • web_search         — live Tavily web search  
  • summarise_docs     — summarise retrieved context
  • extract_entities   — NER on retrieved text
  • calculate          — safe Python expression evaluator

Run standalone:  python -m app.mcp.server
"""

import ast
import logging
import operator
import os
import sys
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

logger = logging.getLogger(__name__)

app = Server("omnirag-mcp")

# ── Tool Definitions ──────────────────────────────────────────────────────────

TOOLS = [
    Tool(
        name="hybrid_search",
        description=(
            "Search the internal knowledge base using hybrid dense+sparse retrieval "
            "with cross-encoder reranking. Use this for questions about ingested documents."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "top_k": {"type": "integer", "default": 5, "description": "Results to return"},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="web_search",
        description=(
            "Search the live web using Tavily. Use for current events, recent data, "
            "or anything not in the knowledge base."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Web search query"},
                "max_results": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="summarise_docs",
        description="Summarise a list of document contents into a concise paragraph.",
        inputSchema={
            "type": "object",
            "properties": {
                "documents": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of document text chunks to summarise",
                },
                "focus": {"type": "string", "description": "What aspect to focus on"},
            },
            "required": ["documents"],
        },
    ),
    Tool(
        name="extract_entities",
        description="Extract named entities (people, orgs, dates, numbers) from text.",
        inputSchema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to extract entities from"},
            },
            "required": ["text"],
        },
    ),
    Tool(
        name="calculate",
        description="Safely evaluate a mathematical expression. E.g. '2 + 2 * 10 / 5'.",
        inputSchema={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression"},
            },
            "required": ["expression"],
        },
    ),
]


# ── Tool Handlers ─────────────────────────────────────────────────────────────

@app.list_tools()
async def list_tools() -> list[Tool]:
    return TOOLS


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    if name == "hybrid_search":
        return await _hybrid_search(**arguments)
    elif name == "web_search":
        return await _web_search(**arguments)
    elif name == "summarise_docs":
        return await _summarise_docs(**arguments)
    elif name == "extract_entities":
        return await _extract_entities(**arguments)
    elif name == "calculate":
        return await _calculate(**arguments)
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


# ── Implementations ───────────────────────────────────────────────────────────

async def _hybrid_search(query: str, top_k: int = 5) -> list[TextContent]:
    try:
        from app.rag.retriever import HybridRetriever
        retriever = HybridRetriever(reranker_top_n=top_k)
        docs = retriever.get_relevant_documents(query)
        if not docs:
            return [TextContent(type="text", text="No relevant documents found.")]

        results = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            score = meta.get("rerank_score", "N/A")
            results.append(
                f"[Result {i}] Source: {meta.get('source','unknown')} | "
                f"Page: {meta.get('page','?')} | "
                f"Type: {meta.get('content_type','text')} | "
                f"Rerank Score: {score}\n\n{doc.page_content}"
            )
        return [TextContent(type="text", text="\n\n---\n\n".join(results))]
    except Exception as e:
        logger.error("hybrid_search error: %s", e)
        return [TextContent(type="text", text=f"Search error: {e}")]


async def _web_search(query: str, max_results: int = 5) -> list[TextContent]:
    try:
        from tavily import TavilyClient
        from app.config import settings
        client = TavilyClient(api_key=settings.tavily_api_key)
        response = client.search(query=query, max_results=max_results)
        results = []
        for r in response.get("results", []):
            results.append(
                f"**{r.get('title','No Title')}**\n"
                f"URL: {r.get('url','')}\n"
                f"{r.get('content','')}"
            )
        return [TextContent(type="text", text="\n\n---\n\n".join(results) or "No web results.")]
    except Exception as e:
        logger.error("web_search error: %s", e)
        return [TextContent(type="text", text=f"Web search error: {e}")]


async def _summarise_docs(documents: list, focus: str = "") -> list[TextContent]:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.prompts import ChatPromptTemplate
        from app.config import settings

        combined = "\n\n---\n\n".join(documents[:8])
        focus_clause = f"Focus especially on: {focus}." if focus else ""

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"Summarise the following documents concisely. {focus_clause}"),
            ("human", "{text}"),
        ])
        llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=0,
        )
        response = (prompt | llm).invoke({"text": combined})
        return [TextContent(type="text", text=response.content)]
    except Exception as e:
        return [TextContent(type="text", text=f"Summarisation error: {e}")]


async def _extract_entities(text: str) -> list[TextContent]:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.prompts import ChatPromptTemplate
        from app.config import settings

        prompt = ChatPromptTemplate.from_messages([
            ("system", """Extract named entities from text. Return as structured list:
PEOPLE: ...
ORGANISATIONS: ...
DATES: ...
NUMBERS/METRICS: ...
LOCATIONS: ..."""),
            ("human", "{text}"),
        ])
        llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=0,
        )
        response = (prompt | llm).invoke({"text": text[:3000]})
        return [TextContent(type="text", text=response.content)]
    except Exception as e:
        return [TextContent(type="text", text=f"Entity extraction error: {e}")]


async def _calculate(expression: str) -> list[TextContent]:
    """Safe math evaluator — no exec/eval of arbitrary code."""
    SAFE_OPS = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg,
        ast.Mod: operator.mod,
    }

    def _eval(node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            return SAFE_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        elif isinstance(node, ast.UnaryOp):
            return SAFE_OPS[type(node.op)](_eval(node.operand))
        else:
            raise ValueError(f"Unsupported operation: {type(node)}")

    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval(tree.body)
        return [TextContent(type="text", text=f"{expression} = {result}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Calculation error: {e}")]


# ── Entry Point ───────────────────────────────────────────────────────────────

async def run_server():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_server())
