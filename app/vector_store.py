"""Chroma vector store integration."""

from __future__ import annotations

import argparse
from typing import Any

from app.config import CHROMA_COLLECTION, CHROMA_DIR, EMBEDDING_MODEL
from app.database import count_rows, list_chunks
from app.embeddings import embed_texts


def _import_chromadb():
    try:
        import chromadb
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "chromadb is not installed. Run: pip install -r requirements.txt"
        ) from exc
    return chromadb


def get_client():
    """Return a persistent Chroma client."""
    chromadb = _import_chromadb()
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection(*, reset: bool = False):
    """Return the single wiki_chunks collection, optionally recreated."""
    client = get_client()
    if reset:
        try:
            client.delete_collection(CHROMA_COLLECTION)
        except Exception:
            pass
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def chunk_id(chunk: dict[str, Any]) -> str:
    """Return a stable vector id for a SQLite chunk."""
    return f"chunk-{chunk['id']}"


def metadata_for_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    """Build Chroma metadata for one SQLite chunk row."""
    return {
        "sqlite_chunk_id": int(chunk["id"]),
        "entity_id": int(chunk["entity_id"]),
        "entity_name": chunk["entity_name"],
        "entity_type": chunk["entity_type"],
        "source_url": chunk["source_url"],
        "chunk_index": int(chunk["chunk_index"]),
        "section_title": chunk.get("section_title") or "",
        "word_count": int(chunk["word_count"]),
    }


def index_chunks(
    *,
    reset: bool = True,
    batch_size: int = 16,
    embedding_model: str = EMBEDDING_MODEL,
) -> dict[str, Any]:
    """Embed SQLite chunks locally and store them in Chroma."""
    chunks = list_chunks()
    if not chunks:
        raise RuntimeError("No SQLite chunks found. Run python -m app.chunker first.")

    collection = get_collection(reset=reset)
    total = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        documents = [chunk["text"] for chunk in batch]
        embeddings = embed_texts(documents, model=embedding_model, batch_size=batch_size)
        collection.upsert(
            ids=[chunk_id(chunk) for chunk in batch],
            embeddings=embeddings,
            documents=documents,
            metadatas=[metadata_for_chunk(chunk) for chunk in batch],
        )
        total += len(batch)
        print(f"Indexed {total}/{len(chunks)} chunks into Chroma.")
    return {"indexed": total, "collection_count": collection.count()}


def collection_count() -> int:
    """Return the number of vectors in the Chroma collection."""
    return int(get_collection().count())


def validate_vector_store() -> list[str]:
    """Return validation errors for Chroma vector storage."""
    errors: list[str] = []
    sqlite_chunks = count_rows("chunks")
    try:
        chroma_vectors = collection_count()
    except Exception as exc:
        errors.append(f"Could not open Chroma collection: {exc}")
        return errors

    if chroma_vectors != sqlite_chunks:
        errors.append(
            f"Chroma vector count ({chroma_vectors}) does not match SQLite chunks ({sqlite_chunks})."
        )
    if not CHROMA_DIR.exists():
        errors.append(f"Chroma directory does not exist: {CHROMA_DIR}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed SQLite chunks and store them in Chroma.")
    parser.add_argument("--reset", action="store_true", help="Recreate the Chroma collection first.")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    try:
        summary = index_chunks(reset=args.reset, batch_size=args.batch_size)
        errors = validate_vector_store()
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from exc
    print(
        f"Vector indexing summary: indexed={summary['indexed']}, "
        f"collection_count={summary['collection_count']}."
    )
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("Vector store validation passed.")


if __name__ == "__main__":
    main()
