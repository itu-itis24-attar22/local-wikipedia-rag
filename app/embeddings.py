"""Local embedding helpers backed by Ollama."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import requests

from app.config import EMBEDDING_MODEL, OLLAMA_BASE_URL


class OllamaError(RuntimeError):
    """Raised when the local Ollama service cannot satisfy a request."""


def _ollama_url(path: str, base_url: str = OLLAMA_BASE_URL) -> str:
    return base_url.rstrip("/") + path


def list_ollama_models(base_url: str = OLLAMA_BASE_URL) -> list[str]:
    """Return locally available Ollama model names."""
    try:
        response = requests.get(_ollama_url("/api/tags", base_url), timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise OllamaError(
            "Ollama is not reachable at "
            f"{base_url}. Start Ollama and pull the required models."
        ) from exc
    payload: dict[str, Any] = response.json()
    return [item.get("name", "") for item in payload.get("models", []) if item.get("name")]


def model_available(model_name: str, base_url: str = OLLAMA_BASE_URL) -> bool:
    """Return whether a model is present locally in Ollama."""
    models = list_ollama_models(base_url)
    return any(name == model_name or name.startswith(model_name + ":") for name in models)


def ensure_model_available(model_name: str, base_url: str = OLLAMA_BASE_URL) -> None:
    """Raise a helpful error if an Ollama model is missing."""
    if not model_available(model_name, base_url):
        raise OllamaError(
            f"Ollama model '{model_name}' is not available locally. "
            f"Run: ollama pull {model_name}"
        )


def embed_texts(
    texts: Iterable[str],
    *,
    model: str = EMBEDDING_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    batch_size: int = 16,
) -> list[list[float]]:
    """Generate local embeddings for text inputs using Ollama.

    The function uses Ollama's /api/embed endpoint first and falls back to the
    older /api/embeddings endpoint when necessary.
    """
    ensure_model_available(model, base_url)
    text_list = list(texts)
    if not text_list:
        return []
    embeddings: list[list[float]] = []
    for start in range(0, len(text_list), batch_size):
        batch = text_list[start : start + batch_size]
        embeddings.extend(_embed_batch(batch, model=model, base_url=base_url))
    return embeddings


def embed_text(
    text: str,
    *,
    model: str = EMBEDDING_MODEL,
    base_url: str = OLLAMA_BASE_URL,
) -> list[float]:
    """Generate one local embedding."""
    return embed_texts([text], model=model, base_url=base_url, batch_size=1)[0]


def _embed_batch(
    batch: list[str],
    *,
    model: str,
    base_url: str,
) -> list[list[float]]:
    try:
        response = requests.post(
            _ollama_url("/api/embed", base_url),
            json={"model": model, "input": batch},
            timeout=120,
        )
        if response.status_code != 404:
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
            embeddings = payload.get("embeddings")
            if embeddings:
                return [[float(value) for value in vector] for vector in embeddings]
    except requests.RequestException as exc:
        raise OllamaError(f"Embedding request failed for model '{model}'.") from exc

    if len(batch) != 1:
        vectors: list[list[float]] = []
        for text in batch:
            vectors.extend(_embed_batch([text], model=model, base_url=base_url))
        return vectors

    try:
        response = requests.post(
            _ollama_url("/api/embeddings", base_url),
            json={"model": model, "prompt": batch[0]},
            timeout=120,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise OllamaError(f"Embedding request failed for model '{model}'.") from exc
    payload = response.json()
    embedding = payload.get("embedding")
    if not embedding:
        raise OllamaError("Ollama returned no embedding.")
    return [[float(value) for value in embedding]]
