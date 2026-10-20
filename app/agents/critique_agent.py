"""
OmniRAG — Critique Agent
─────────────────────────
Evaluates the synthesised answer for:
  • Faithfulness (grounded in context)
  • Completeness (answers the full question)
  • Accuracy (no hallucination)
  • Clarity (well-structured)

Returns "GOOD" if the answer passes, or specific improvement instructions.
"""

import logging
from dataclasses import dataclass
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings

logger = logging.getLogger(__name__)

CRITIQUE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a rigorous fact-checking and quality evaluation agent.

Evaluate the given answer against these criteria:

1. FAITHFULNESS: Every claim is supported by the provided context (no hallucination)
2. COMPLETENESS: The original question is fully answered
3. ACCURACY: Numbers, dates, and facts are correct per context
4. CLARITY: Well-structured, readable, professional

Scoring:
- If ALL criteria pass → respond with exactly: GOOD
- If ANY criterion fails → respond with: REVISE: [specific instructions for improvement]
  List each issue clearly so the synthesis agent can fix it.

Be strict. A "GOOD" means you'd stake your reputation on this answer."""),
    ("human", """Original Question: {query}

Context Used:
{context}

Generated Answer:
{answer}

Your evaluation:"""),
])


@dataclass
class CritiqueAgent:
    def evaluate(self, query: str, answer: str, context: str) -> str:
        if not answer or not answer.strip():
            return "REVISE: Answer is empty. Please generate a complete response."

        llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=0,
        )
        chain = CRITIQUE_PROMPT | llm
        response = chain.invoke({
            "query": query,
            "answer": answer,
            "context": context[:4000],  # Token budget
        })
        critique = response.content.strip()
        logger.info("Critique result: %s", critique[:100])
        return critique


def create_critique_agent() -> CritiqueAgent:
    return CritiqueAgent()
