"""
OmniRAG — Multi-Agent LangGraph Workflow
─────────────────────────────────────────
Agents:
  ┌─────────────────────────────────────────────┐
  │              SUPERVISOR AGENT               │
  │  Routes queries → right specialist agent    │
  └──────┬──────────────┬────────────┬──────────┘
         │              │            │
    ┌────▼────┐   ┌──────▼────┐ ┌───▼──────────┐
    │   RAG   │   │    WEB    │ │  SYNTHESIS   │
    │  AGENT  │   │  SEARCH   │ │    AGENT     │
    │(Hybrid) │   │  AGENT    │ │(Final Answer)│
    └────┬────┘   └──────┬────┘ └───▲──────────┘
         │               │          │
         └───────────────┴──────────┘
                    │
             ┌──────▼──────┐
             │  CRITIQUE   │
             │   AGENT     │
             │(Evaluator)  │
             └─────────────┘

LangSmith traces every step automatically.
"""

import logging
from typing import Annotated, List, Literal, TypedDict

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

from app.config import settings
from app.agents.rag_agent import create_rag_agent
from app.agents.web_agent import create_web_agent
from app.agents.synthesis_agent import create_synthesis_agent
from app.agents.critique_agent import create_critique_agent

logger = logging.getLogger(__name__)


# ── State Schema ──────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    query: str
    rag_context: str
    web_context: str
    draft_answer: str
    final_answer: str
    critique: str
    iterations: int
    route: str


# ── LLM Factory ──────────────────────────────────────────────────────────────

def get_llm(temperature: float = 0.0):
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=temperature,
    )


# ── Supervisor Node ───────────────────────────────────────────────────────────

SUPERVISOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a supervisor routing research queries to specialist agents.

Available agents:
- rag_agent: For questions about ingested documents/knowledge base
- web_agent: For current events, recent data, or live information  
- both: When the question benefits from both sources
- synthesis: Directly synthesise already gathered context (when context is rich)

Analyse the query and respond with ONLY one of: rag_agent, web_agent, both, synthesis
"""),
    ("human", "Query: {query}\nExisting RAG context: {rag_context}\nExisting Web context: {web_context}"),
])


def supervisor_node(state: AgentState) -> AgentState:
    """Routes to the appropriate agent(s)."""
    llm = get_llm()
    chain = SUPERVISOR_PROMPT | llm
    response = chain.invoke({
        "query": state["query"],
        "rag_context": state.get("rag_context", ""),
        "web_context": state.get("web_context", ""),
    })
    route = response.content.strip().lower()
    if route not in ["rag_agent", "web_agent", "both", "synthesis"]:
        route = "both"
    logger.info("Supervisor routed to: %s", route)
    return {**state, "route": route}


# ── Agent Nodes ───────────────────────────────────────────────────────────────

def rag_node(state: AgentState) -> AgentState:
    agent = create_rag_agent()
    context = agent.run(state["query"])
    return {**state, "rag_context": context}


def web_node(state: AgentState) -> AgentState:
    agent = create_web_agent()
    context = agent.run(state["query"])
    return {**state, "web_context": context}


def synthesis_node(state: AgentState) -> AgentState:
    agent = create_synthesis_agent()
    answer = agent.run(
        query=state["query"],
        rag_context=state.get("rag_context", ""),
        web_context=state.get("web_context", ""),
    )
    return {**state, "draft_answer": answer}


def critique_node(state: AgentState) -> AgentState:
    agent = create_critique_agent()
    critique = agent.evaluate(
        query=state["query"],
        answer=state["draft_answer"],
        context=state.get("rag_context", "") + "\n" + state.get("web_context", ""),
    )
    iterations = state.get("iterations", 0) + 1
    # If critique says GOOD or max iterations reached → finalise
    if "GOOD" in critique.upper() or iterations >= settings.max_iterations:
        final = state["draft_answer"]
    else:
        final = ""  # Will loop back
    return {**state, "critique": critique, "final_answer": final, "iterations": iterations}


# ── Routing Functions ─────────────────────────────────────────────────────────

def route_supervisor(state: AgentState) -> Literal["rag_node", "web_node", "both_node", "synthesis_node"]:
    route = state.get("route", "both")
    if route == "rag_agent":
        return "rag_node"
    elif route == "web_agent":
        return "web_node"
    elif route == "both":
        return "both_node"
    else:
        return "synthesis_node"


def route_critique(state: AgentState) -> Literal["synthesis_node", "end"]:
    """Loop back to synthesis if critique flagged issues, else end."""
    if state.get("final_answer"):
        return "end"
    return "synthesis_node"


# ── Parallel Both Node ────────────────────────────────────────────────────────

def both_node(state: AgentState) -> AgentState:
    """Run RAG and Web in sequence (parallel requires async LangGraph)."""
    state = rag_node(state)
    state = web_node(state)
    return state


# ── Graph Assembly ────────────────────────────────────────────────────────────

def build_graph(checkpointer=None):
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("rag_node", rag_node)
    graph.add_node("web_node", web_node)
    graph.add_node("both_node", both_node)
    graph.add_node("synthesis_node", synthesis_node)
    graph.add_node("critique_node", critique_node)

    # Entry
    graph.add_edge(START, "supervisor")

    # Supervisor → agents (conditional)
    graph.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "rag_node": "rag_node",
            "web_node": "web_node",
            "both_node": "both_node",
            "synthesis_node": "synthesis_node",
        },
    )

    # Agents → synthesis
    graph.add_edge("rag_node", "synthesis_node")
    graph.add_edge("web_node", "synthesis_node")
    graph.add_edge("both_node", "synthesis_node")

    # Synthesis → critique
    graph.add_edge("synthesis_node", "critique_node")

    # Critique → loop or end
    graph.add_conditional_edges(
        "critique_node",
        route_critique,
        {"synthesis_node": "synthesis_node", "end": END},
    )

    return graph.compile(checkpointer=checkpointer)


# ── Public Interface ──────────────────────────────────────────────────────────

def run_query(query: str, thread_id: str = "default") -> dict:
    """
    Run a query through the full multi-agent pipeline.
    Returns dict with final_answer, rag_context, web_context, critique.
    """
    import os, uuid
    db_path = os.path.join(settings.data_dir, "checkpoints.db")
    os.makedirs(settings.data_dir, exist_ok=True)

    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        graph = build_graph(checkpointer)
        config = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}

        initial_state: AgentState = {
            "messages": [HumanMessage(content=query)],
            "query": query,
            "rag_context": "",
            "web_context": "",
            "draft_answer": "",
            "final_answer": "",
            "critique": "",
            "iterations": 0,
            "route": "",
        }

        result = graph.invoke(initial_state, config=config)
        return {
            "final_answer": result.get("final_answer") or result.get("draft_answer"),
            "rag_context": result.get("rag_context", ""),
            "web_context": result.get("web_context", ""),
            "critique": result.get("critique", ""),
            "iterations": result.get("iterations", 0),
            "route": result.get("route", ""),
        }
