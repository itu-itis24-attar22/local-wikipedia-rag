"""End-to-end RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from app.classifier import Classification
from app.config import UNKNOWN_ANSWER
from app.generator import generate_answer
from app.retriever import RetrievedChunk, retrieve


@dataclass(frozen=True)
class RAGResponse:
    query: str
    answer: str
    classification: Classification
    retrieved_chunks: list[RetrievedChunk]


def answer_question(query: str) -> RAGResponse:
    """Classify, retrieve, and answer one question."""
    retrieval_result = retrieve(query)
    if retrieval_result.classification.force_unknown or not retrieval_result.chunks:
        return RAGResponse(
            query=query,
            answer=UNKNOWN_ANSWER,
            classification=retrieval_result.classification,
            retrieved_chunks=retrieval_result.chunks,
        )

    answer = generate_answer(query, retrieval_result.chunks)
    return RAGResponse(
        query=query,
        answer=answer,
        classification=retrieval_result.classification,
        retrieved_chunks=retrieval_result.chunks,
    )
