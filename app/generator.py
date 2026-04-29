"""Local Ollama generation logic grounded in retrieved context."""

from __future__ import annotations

from typing import Any

import requests

from app.config import (
    GENERATION_MODEL,
    GENERATION_TEMPERATURE,
    OLLAMA_BASE_URL,
    UNKNOWN_ANSWER,
)
from app.embeddings import OllamaError, _ollama_url, ensure_model_available
from app.retriever import RetrievedChunk, format_retrieved_context


class GenerationError(RuntimeError):
    """Raised when local answer generation fails."""


def build_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    """Build a strict grounded-answer prompt."""
    context = format_retrieved_context(chunks)
    return f"""You are a local Wikipedia retrieval assistant.

Use only the retrieved local Wikipedia context below.
Do not use outside knowledge.
If the context does not contain enough information to answer the question, reply exactly:
{UNKNOWN_ANSWER}

When answering, be concise and mention the relevant source entity names from the context.

Retrieved context:
{context}

Question:
{query}

Answer:"""


def generate_answer(
    query: str,
    chunks: list[RetrievedChunk],
    *,
    model: str = GENERATION_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    temperature: float = GENERATION_TEMPERATURE,
) -> str:
    """Generate a grounded answer with a local Ollama model."""
    if not chunks:
        return UNKNOWN_ANSWER
    try:
        ensure_model_available(model, base_url)
    except OllamaError as exc:
        raise GenerationError(str(exc)) from exc

    prompt = build_prompt(query, chunks)
    try:
        response = requests.post(
            _ollama_url("/api/generate", base_url),
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=180,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise GenerationError(f"Local Ollama generation failed for model '{model}'.") from exc

    payload: dict[str, Any] = response.json()
    answer = str(payload.get("response", "")).strip()
    if not answer:
        return UNKNOWN_ANSWER
    return answer
