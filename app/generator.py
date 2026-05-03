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
from app.retriever import RetrievedChunk


class GenerationError(RuntimeError):
    """Raised when local answer generation fails."""


MAX_PROMPT_CHUNKS = 3
MAX_CHUNK_WORDS_FOR_PROMPT = 140
MAX_CHUNKS_PER_ENTITY_FOR_PROMPT = 2


def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " ..."


def _select_prompt_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Keep the generation prompt small while preserving entity diversity."""
    selected: list[RetrievedChunk] = []
    counts_by_entity: dict[str, int] = {}
    for chunk in chunks:
        entity_name = str(chunk.metadata.get("entity_name", ""))
        if counts_by_entity.get(entity_name, 0) >= MAX_CHUNKS_PER_ENTITY_FOR_PROMPT:
            continue
        selected.append(chunk)
        counts_by_entity[entity_name] = counts_by_entity.get(entity_name, 0) + 1
        if len(selected) >= MAX_PROMPT_CHUNKS:
            break
    return selected or chunks[:MAX_PROMPT_CHUNKS]


def format_prompt_context(chunks: list[RetrievedChunk]) -> str:
    """Format a compact grounded context for local generation."""
    blocks: list[str] = []
    for index, chunk in enumerate(_select_prompt_chunks(chunks), start=1):
        metadata = chunk.metadata
        blocks.append(
            "\n".join(
                [
                    f"[{index}] Entity: {metadata.get('entity_name', '')}",
                    f"Type: {metadata.get('entity_type', '')}",
                    f"Section: {metadata.get('section_title', '')}",
                    f"Source: {metadata.get('source_url', '')}",
                    f"Chunk index: {metadata.get('chunk_index', '')}",
                    "Text:",
                    _truncate_words(chunk.text, MAX_CHUNK_WORDS_FOR_PROMPT),
                ]
            )
        )
    return "\n\n".join(blocks)


def build_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    """Build a strict grounded-answer prompt."""
    context = format_prompt_context(chunks)
    return f"""You are a local Wikipedia retrieval assistant.

Use only the retrieved local Wikipedia context below.
Do not use outside knowledge.
If the context is empty or unrelated to the question, reply exactly:
{UNKNOWN_ANSWER}
If the context supports an answer, answer only with supported facts and do not append the fallback sentence.

When answering, be concise. Use at most 3 short sentences and mention the relevant source entity names from the context.
When writing mathematical formulas, preserve exponents using plain text notation, for example E = mc^2.

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
                "options": {
                    "temperature": temperature,
                    "num_ctx": 2048,
                    "num_predict": 160,
                },
            },
            timeout=300,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise GenerationError(f"Local Ollama generation failed for model '{model}'.") from exc

    payload: dict[str, Any] = response.json()
    answer = str(payload.get("response", "")).strip()
    if not answer:
        return UNKNOWN_ANSWER
    return answer
