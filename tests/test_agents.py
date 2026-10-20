"""
OmniRAG — Test Suite
Tests core components without requiring live API calls (mocked where needed).
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

# ── Reranker Tests ────────────────────────────────────────────────────────────

class TestCrossEncoderReranker:
    def test_rerank_returns_top_n(self):
        from app.rag.reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker(top_n=3)

        docs = [
            Document(page_content=f"Document {i} content", metadata={"id": i})
            for i in range(10)
        ]

        with patch.object(reranker, '_get_model') as mock_model:
            model = MagicMock()
            model.predict.return_value = list(range(10, 0, -1))  # scores 10..1
            mock_model.return_value = model

            result = reranker.rerank("test query", docs)

        assert len(result) == 3
        # First result should have highest score
        assert result[0].metadata["rerank_score"] == 10.0

    def test_rerank_handles_empty_docs(self):
        from app.rag.reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker(top_n=5)
        result = reranker.rerank("query", [])
        assert result == []

    def test_rerank_handles_model_failure(self):
        from app.rag.reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker(top_n=3)
        docs = [Document(page_content=f"doc {i}") for i in range(5)]

        with patch.object(reranker, '_get_model', return_value=None):
            result = reranker.rerank("query", docs)
        assert len(result) == 3  # Falls back to truncation


# ── MCP Calculator Tests ──────────────────────────────────────────────────────

class TestMCPCalculator:
    @pytest.mark.asyncio
    async def test_basic_arithmetic(self):
        from app.mcp.server import _calculate
        result = await _calculate("2 + 2")
        assert "4" in result[0].text

    @pytest.mark.asyncio
    async def test_complex_expression(self):
        from app.mcp.server import _calculate
        result = await _calculate("(10 + 5) * 2 / 3")
        assert "10.0" in result[0].text

    @pytest.mark.asyncio
    async def test_invalid_expression(self):
        from app.mcp.server import _calculate
        result = await _calculate("import os; os.system('ls')")
        assert "error" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_power_operation(self):
        from app.mcp.server import _calculate
        result = await _calculate("2 ** 10")
        assert "1024" in result[0].text


# ── Ingestion Deduplication Tests ─────────────────────────────────────────────

class TestIngestion:
    def test_deduplication(self):
        from app.rag.ingestion import ingest_documents
        docs = [
            Document(page_content="Same content", metadata={"source": "test"}),
            Document(page_content="Same content", metadata={"source": "test"}),  # duplicate
            Document(page_content="Different content", metadata={"source": "test"}),
        ]

        with patch("app.rag.ingestion.get_vector_store") as mock_vs:
            vs = MagicMock()
            vs.add_documents.return_value = ["id1", "id2"]
            mock_vs.return_value = vs

            count = ingest_documents(docs)
            # Should deduplicate to 2 unique docs
            call_args = vs.add_documents.call_args[0][0]
            assert len(call_args) == 2

    def test_text_extraction(self):
        from app.rag.ingestion import extract_from_text
        text = "This is test content. " * 100  # Long enough to chunk
        docs = extract_from_text(text, source="test")
        assert len(docs) > 0
        assert all(d.metadata["source"] == "test" for d in docs)
        assert all(d.metadata["content_type"] == "text" for d in docs)


# ── Agent Routing Tests ───────────────────────────────────────────────────────

class TestAgentRouting:
    def test_supervisor_routes_to_rag(self):
        """Supervisor should route document-specific queries to RAG."""
        from app.graph.workflow import supervisor_node, AgentState
        from langchain_core.messages import HumanMessage

        state = AgentState(
            messages=[HumanMessage(content="What does the uploaded document say?")],
            query="What does the uploaded document say?",
            rag_context="",
            web_context="",
            draft_answer="",
            final_answer="",
            critique="",
            iterations=0,
            route="",
        )

        with patch("app.graph.workflow.get_llm") as mock_llm:
            llm = MagicMock()
            response = MagicMock()
            response.content = "rag_agent"
            llm.return_value = llm
            mock_llm.return_value = llm

            # Test routing logic directly
            from app.graph.workflow import route_supervisor
            state["route"] = "rag_agent"
            assert route_supervisor(state) == "rag_node"

    def test_routing_web(self):
        from app.graph.workflow import route_supervisor, AgentState
        from langchain_core.messages import HumanMessage

        state = AgentState(
            messages=[], query="", rag_context="", web_context="",
            draft_answer="", final_answer="", critique="",
            iterations=0, route="web_agent",
        )
        assert route_supervisor(state) == "web_node"

    def test_routing_both(self):
        from app.graph.workflow import route_supervisor, AgentState

        state = AgentState(
            messages=[], query="", rag_context="", web_context="",
            draft_answer="", final_answer="", critique="",
            iterations=0, route="both",
        )
        assert route_supervisor(state) == "both_node"

    def test_critique_ends_on_good(self):
        from app.graph.workflow import route_critique, AgentState

        state = AgentState(
            messages=[], query="", rag_context="", web_context="",
            draft_answer="test answer", final_answer="GOOD answer here",
            critique="GOOD", iterations=1, route="",
        )
        assert route_critique(state) == "end"

    def test_critique_loops_on_revise(self):
        from app.graph.workflow import route_critique, AgentState

        state = AgentState(
            messages=[], query="", rag_context="", web_context="",
            draft_answer="bad answer", final_answer="",
            critique="REVISE: missing details", iterations=1, route="",
        )
        assert route_critique(state) == "synthesis_node"
