"""
OmniRAG — Cross-Encoder Reranker
──────────────────────────────────
Uses BAAI/bge-reranker-base to score query-document pairs
and return the top-N most relevant results.
"""

import logging
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Lightweight cross-encoder reranker using sentence-transformers.
    Model: BAAI/bge-reranker-base (free, runs locally).
    """

    _model = None  # lazy singleton

    def __init__(self, top_n: int = 5, model_name: str = "BAAI/bge-reranker-base"):
        self.top_n = top_n
        self.model_name = model_name

    def _get_model(self):
        if CrossEncoderReranker._model is None:
            try:
                from sentence_transformers import CrossEncoder
                CrossEncoderReranker._model = CrossEncoder(self.model_name)
                logger.info("Loaded reranker: %s", self.model_name)
            except Exception as e:
                logger.warning("Reranker load failed (%s). Skipping rerank.", e)
                CrossEncoderReranker._model = None
        return CrossEncoderReranker._model

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """Score and sort documents by relevance to query."""
        model = self._get_model()
        if model is None or not documents:
            return documents[: self.top_n]

        pairs = [(query, doc.page_content) for doc in documents]
        scores = model.predict(pairs)

        scored = sorted(
            zip(scores, documents), key=lambda x: x[0], reverse=True
        )

        reranked = []
        for score, doc in scored[: self.top_n]:
            doc.metadata["rerank_score"] = float(score)
            reranked.append(doc)

        return reranked
