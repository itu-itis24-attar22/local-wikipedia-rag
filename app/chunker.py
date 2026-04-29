"""Paragraph-aware chunking for large Wikipedia documents."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import CHUNK_OVERLAP_WORDS, CHUNK_TARGET_WORDS
from app.database import insert_chunks, list_chunks, list_entities


@dataclass(frozen=True)
class RawSection:
    title: str
    paragraphs: list[str]


def word_count(text: str) -> int:
    """Count whitespace-delimited words."""
    return len(text.split())


def parse_raw_sections(raw_path: Path) -> list[RawSection]:
    """Parse a saved raw Wikipedia text file into titled sections."""
    text = raw_path.read_text(encoding="utf-8")
    sections: list[RawSection] = []
    current_title = "Lead"
    current_paragraphs: list[str] = []
    paragraph_lines: list[str] = []
    in_content = False

    def flush_paragraph() -> None:
        if paragraph_lines:
            paragraph = " ".join(line.strip() for line in paragraph_lines if line.strip()).strip()
            if paragraph:
                current_paragraphs.append(paragraph)
            paragraph_lines.clear()

    def flush_section() -> None:
        flush_paragraph()
        if current_paragraphs:
            sections.append(RawSection(current_title, list(current_paragraphs)))
            current_paragraphs.clear()

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            in_content = True
            flush_section()
            current_title = stripped.replace("## ", "", 1).strip() or "Lead"
            continue
        if not in_content:
            continue
        if not stripped:
            flush_paragraph()
            continue
        if stripped.startswith("# ") or stripped.startswith("Type:") or stripped.startswith("Source URL:"):
            continue
        paragraph_lines.append(stripped)

    flush_section()
    return sections


def _emit_chunk(
    chunks: list[dict[str, Any]],
    *,
    section_title: str,
    words: list[str],
) -> None:
    text = " ".join(words).strip()
    if not text:
        return
    chunks.append(
        {
            "chunk_index": len(chunks),
            "section_title": section_title,
            "text": text,
            "word_count": len(words),
        }
    )


def _tail_overlap(words: list[str], overlap_words: int) -> list[str]:
    if overlap_words <= 0:
        return []
    return words[-overlap_words:]


def chunk_sections(
    sections: list[RawSection],
    *,
    target_words: int = CHUNK_TARGET_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
) -> list[dict[str, Any]]:
    """Chunk raw sections while respecting paragraph boundaries where possible."""
    if target_words <= 0:
        raise ValueError("target_words must be positive.")
    if overlap_words < 0:
        raise ValueError("overlap_words cannot be negative.")
    if overlap_words >= target_words:
        raise ValueError("overlap_words must be smaller than target_words.")

    chunks: list[dict[str, Any]] = []
    for section in sections:
        current_words: list[str] = []
        for paragraph in section.paragraphs:
            paragraph_words = paragraph.split()
            if not paragraph_words:
                continue

            if len(paragraph_words) > target_words:
                if current_words:
                    _emit_chunk(chunks, section_title=section.title, words=current_words)
                    current_words = []
                start = 0
                step = target_words - overlap_words
                while start < len(paragraph_words):
                    window = paragraph_words[start : start + target_words]
                    _emit_chunk(chunks, section_title=section.title, words=window)
                    if start + target_words >= len(paragraph_words):
                        break
                    start += step
                continue

            if current_words and len(current_words) + len(paragraph_words) > target_words:
                emitted_words = current_words
                _emit_chunk(chunks, section_title=section.title, words=emitted_words)
                overlap = _tail_overlap(emitted_words, overlap_words)
                current_words = overlap + paragraph_words
                if len(current_words) > target_words:
                    current_words = paragraph_words
            else:
                current_words.extend(paragraph_words)

        if current_words:
            _emit_chunk(chunks, section_title=section.title, words=current_words)

    return chunks


def chunk_raw_file(
    raw_path: Path,
    *,
    target_words: int = CHUNK_TARGET_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
) -> list[dict[str, Any]]:
    """Parse and chunk one raw text file."""
    sections = parse_raw_sections(raw_path)
    return chunk_sections(sections, target_words=target_words, overlap_words=overlap_words)


def rebuild_chunks_from_database(
    *,
    target_words: int = CHUNK_TARGET_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
) -> dict[str, Any]:
    """Read stored entities, chunk their raw files, and replace SQLite chunks."""
    entity_summaries: list[dict[str, Any]] = []
    total_chunks = 0

    for entity in list_entities():
        raw_path = Path(entity["raw_path"])
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw file is missing for {entity['name']}: {raw_path}")
        chunks = chunk_raw_file(
            raw_path,
            target_words=target_words,
            overlap_words=overlap_words,
        )
        if not chunks:
            raise RuntimeError(f"No chunks produced for {entity['name']}")
        inserted = insert_chunks(entity["id"], chunks, replace_existing=True)
        total_chunks += inserted
        entity_summaries.append(
            {
                "entity_id": entity["id"],
                "name": entity["name"],
                "type": entity["type"],
                "chunk_count": inserted,
            }
        )

    return {"total_chunks": total_chunks, "entities": entity_summaries}


def validate_chunks() -> list[str]:
    """Return validation errors for stored chunks."""
    errors: list[str] = []
    chunks = list_chunks()
    entities = list_entities()
    entity_ids = {entity["id"] for entity in entities}
    chunked_entity_ids = {chunk["entity_id"] for chunk in chunks}

    missing = entity_ids - chunked_entity_ids
    if missing:
        errors.append(f"{len(missing)} entities have no chunks.")
    if len(chunks) <= 40:
        errors.append(f"Expected more than 40 chunks, found {len(chunks)}.")
    empty_chunks = [chunk["id"] for chunk in chunks if not chunk["text"].strip()]
    if empty_chunks:
        errors.append(f"Empty chunk ids: {empty_chunks}.")
    missing_metadata = [
        chunk["id"]
        for chunk in chunks
        if not chunk["entity_name"] or not chunk["entity_type"] or not chunk["source_url"]
    ]
    if missing_metadata:
        errors.append(f"Chunks missing joined entity metadata: {missing_metadata}.")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk local raw Wikipedia text into SQLite.")
    parser.add_argument("--target-words", type=int, default=CHUNK_TARGET_WORDS)
    parser.add_argument("--overlap-words", type=int, default=CHUNK_OVERLAP_WORDS)
    args = parser.parse_args()

    summary = rebuild_chunks_from_database(
        target_words=args.target_words,
        overlap_words=args.overlap_words,
    )
    errors = validate_chunks()
    print(f"Chunked {len(summary['entities'])} entities into {summary['total_chunks']} chunks.")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("Chunk validation passed.")


if __name__ == "__main__":
    main()
