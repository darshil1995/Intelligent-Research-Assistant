from sentence_transformers import CrossEncoder
from src.utils.logger import get_logger
from typing import List
from langchain_core.documents import Document

logger = get_logger(__name__)

# A lightweight but powerful reranker model
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class DocumentReranker:
    def __init__(self):
        logger.info(f"Loading Reranker Model: {RERANK_MODEL}")
        self.model = CrossEncoder(RERANK_MODEL)

    def rerank(self, query: str, documents: List[Document], top_n: int = 3) -> List[Document]:
        """
        Takes a list of retrieved documents and re-orders them by actual relevance.
        """
        if not documents:
            return []

        # Prepare pairs for the Cross-Encoder: (Query, Document_Content)
        pairs = [[query, doc.page_content] for doc in documents]

        # Get relevance scores
        scores = self.model.predict(pairs)

        # Attach scores to documents and sort
        for i, doc in enumerate(documents):
            doc.metadata["rerank_score"] = float(scores[i])

        # Sort by score descending
        reranked_docs = sorted(documents, key=lambda x: x.metadata["rerank_score"], reverse=True)

        logger.info(f"Reranked {len(documents)} docs. Top score: {reranked_docs[0].metadata['rerank_score']:.4f}")

        return reranked_docs[:top_n]