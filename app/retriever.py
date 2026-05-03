"""Vector retrieval logic over the local Chroma collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.classifier import Classification, classify_query
from app.config import COMPARISON_TOP_K_PER_ENTITY, RETRIEVAL_TOP_K
from app.embeddings import embed_text
from app.vector_store import get_collection


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    text: str
    metadata: dict[str, Any]
    distance: float | None


@dataclass(frozen=True)
class RetrievalPlan:
    query: str
    classification: Classification
    searches: list[dict[str, Any]]


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    classification: Classification
    chunks: list[RetrievedChunk]


def _entity_filter(entity_name: str) -> dict[str, str]:
    return {"entity_name": entity_name}


def _type_filter(entity_type: str) -> dict[str, str]:
    return {"entity_type": entity_type}


def build_retrieval_plan(
    query: str,
    *,
    top_k: int = RETRIEVAL_TOP_K,
    comparison_top_k_per_entity: int = COMPARISON_TOP_K_PER_ENTITY,
) -> RetrievalPlan:
    """Build Chroma metadata filters from the query classification."""
    classification = classify_query(query)
    if classification.force_unknown:
        return RetrievalPlan(query=query, classification=classification, searches=[])

    entity_names = list(dict.fromkeys(classification.matched_entities))
    if entity_names:
        per_entity_k = comparison_top_k_per_entity if classification.is_comparison else top_k
        searches = [
            {"where": _entity_filter(entity_name), "top_k": per_entity_k}
            for entity_name in entity_names
        ]
        return RetrievalPlan(query=query, classification=classification, searches=searches)

    if classification.query_type == "person":
        searches = [{"where": _type_filter("person"), "top_k": top_k}]
    elif classification.query_type == "place":
        searches = [{"where": _type_filter("place"), "top_k": top_k}]
    elif classification.query_type == "both":
        searches = [{"where": None, "top_k": top_k}]
    else:
        searches = []
    return RetrievalPlan(query=query, classification=classification, searches=searches)


def _query_collection(
    query: str,
    *,
    where: dict[str, Any] | None,
    top_k: int,
) -> list[RetrievedChunk]:
    collection = get_collection()
    query_embedding = embed_text(query)
    kwargs: dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where
    result = collection.query(**kwargs)

    ids = result.get("ids", [[]])[0]
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    chunks: list[RetrievedChunk] = []
    for index, chunk_id in enumerate(ids):
        chunks.append(
            RetrievedChunk(
                id=chunk_id,
                text=documents[index],
                metadata=metadatas[index],
                distance=float(distances[index]) if distances else None,
            )
        )
    return chunks


def _entity_anchor_chunk(entity_name: str) -> RetrievedChunk | None:
    """Return the first/lead chunk for a named entity from Chroma."""
    collection = get_collection()
    try:
        result = collection.get(
            where={"$and": [_entity_filter(entity_name), {"chunk_index": 0}]},
            include=["documents", "metadatas"],
        )
        ids = result.get("ids", [])
        if ids:
            return RetrievedChunk(
                id=ids[0],
                text=result.get("documents", [])[0],
                metadata=result.get("metadatas", [])[0],
                distance=-1.0,
            )
    except Exception:
        pass

    result = collection.get(
        where=_entity_filter(entity_name),
        include=["documents", "metadatas"],
    )
    ids = result.get("ids", [])
    documents = result.get("documents", [])
    metadatas = result.get("metadatas", [])

    fallback: RetrievedChunk | None = None
    for index, chunk_id in enumerate(ids):
        metadata = metadatas[index]
        chunk = RetrievedChunk(
            id=chunk_id,
            text=documents[index],
            metadata=metadata,
            distance=-1.0,
        )
        if int(metadata.get("chunk_index", -1)) == 0:
            return chunk
        if fallback is None and str(metadata.get("section_title", "")).lower() == "lead":
            fallback = chunk
    return fallback


def retrieve(
    query: str,
    *,
    top_k: int = RETRIEVAL_TOP_K,
    comparison_top_k_per_entity: int = COMPARISON_TOP_K_PER_ENTITY,
) -> RetrievalResult:
    """Retrieve relevant chunks from Chroma based on a classified query."""
    plan = build_retrieval_plan(
        query,
        top_k=top_k,
        comparison_top_k_per_entity=comparison_top_k_per_entity,
    )
    chunks_by_id: dict[str, RetrievedChunk] = {}
    for entity_name in dict.fromkeys(plan.classification.matched_entities):
        anchor = _entity_anchor_chunk(entity_name)
        if anchor is not None:
            chunks_by_id.setdefault(anchor.id, anchor)
    for search in plan.searches:
        for chunk in _query_collection(query, where=search["where"], top_k=search["top_k"]):
            chunks_by_id.setdefault(chunk.id, chunk)

    chunks = sorted(
        chunks_by_id.values(),
        key=lambda chunk: float("inf") if chunk.distance is None else chunk.distance,
    )
    return RetrievalResult(query=query, classification=plan.classification, chunks=chunks[:top_k])


def format_retrieved_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks for a grounded generation prompt."""
    blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
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
                    chunk.text,
                ]
            )
        )
    return "\n\n".join(blocks)
