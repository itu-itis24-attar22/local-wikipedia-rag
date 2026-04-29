"""Wikipedia ingestion command.

This module fetches configured Wikipedia pages, extracts readable paragraph
text with section labels, saves raw text locally, and stores entity metadata in
SQLite. Chunking and vector indexing are layered on in later phases.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

from app.config import ENTITIES_PATH, PROJECT_ROOT, RAW_DIR, SQLITE_PATH, load_entities_config
from app.database import initialize_database, reset_database, upsert_entity


WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "BLG483E-Local-Wikipedia-RAG/1.0 (student project)"
SKIPPED_SECTION_TITLES = {
    "references",
    "external links",
    "further reading",
    "see also",
    "notes",
    "bibliography",
    "sources",
}


@dataclass(frozen=True)
class EntityConfig:
    name: str
    entity_type: str
    wikipedia_title: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class ParsedWikipediaPage:
    title: str
    source_url: str
    sections: list[dict[str, str]]

    @property
    def word_count(self) -> int:
        return sum(len(section["text"].split()) for section in self.sections)


class WikipediaHTMLParser(HTMLParser):
    """Extract headings and paragraphs from Wikipedia's parsed HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.sections: list[dict[str, str]] = []
        self.current_section = "Lead"
        self._paragraph_parts: list[str] = []
        self._heading_parts: list[str] = []
        self._capture_paragraph = False
        self._capture_heading = False
        self._ignore_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        class_value = attrs_dict.get("class", "")
        if tag in {"style", "script", "table", "figure", "sup", "math"}:
            self._ignore_depth += 1
            return
        if tag == "span" and "mw-editsection" in class_value:
            self._ignore_depth += 1
            return
        if self._ignore_depth:
            return
        if tag == "p":
            self._capture_paragraph = True
            self._paragraph_parts = []
        elif tag in {"h2", "h3", "h4"}:
            self._capture_heading = True
            self._heading_parts = []
        elif tag == "br":
            self._append_text(" ")

    def handle_endtag(self, tag: str) -> None:
        if self._ignore_depth:
            if tag in {"style", "script", "table", "figure", "sup", "math", "span"}:
                self._ignore_depth = max(0, self._ignore_depth - 1)
            return
        if tag == "p" and self._capture_paragraph:
            paragraph = clean_text("".join(self._paragraph_parts))
            if len(paragraph.split()) >= 12 and not paragraph.startswith("Coordinates:"):
                self.sections.append({"section_title": self.current_section, "text": paragraph})
            self._capture_paragraph = False
            self._paragraph_parts = []
        elif tag in {"h2", "h3", "h4"} and self._capture_heading:
            heading = clean_text("".join(self._heading_parts))
            if heading and heading.lower() not in SKIPPED_SECTION_TITLES:
                self.current_section = heading
            self._capture_heading = False
            self._heading_parts = []

    def handle_data(self, data: str) -> None:
        if self._ignore_depth:
            return
        self._append_text(data)

    def _append_text(self, data: str) -> None:
        if self._capture_paragraph:
            self._paragraph_parts.append(data)
        if self._capture_heading:
            self._heading_parts.append(data)


def clean_text(text: str) -> str:
    """Normalize whitespace and common Wikipedia extraction artifacts."""
    text = unescape(text)
    text = re.sub(r"\[\s*\d+\s*\]", "", text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def wiki_source_url(title: str) -> str:
    """Build a canonical-looking English Wikipedia URL for a page title."""
    return "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_"), safe="_()")


def raw_filename(entity_name: str) -> str:
    """Return a stable raw text filename for an entity display name."""
    slug = re.sub(r"[^a-z0-9]+", "_", entity_name.lower()).strip("_")
    return f"{slug}.txt"


def load_entity_configs(path: Path = ENTITIES_PATH) -> list[EntityConfig]:
    """Load all configured entities with explicit type labels."""
    data = load_entities_config(path)
    entities: list[EntityConfig] = []
    for entity_type, key in (("person", "people"), ("place", "places")):
        for item in data[key]:
            entities.append(
                EntityConfig(
                    name=item["name"],
                    entity_type=entity_type,
                    wikipedia_title=item["wikipedia_title"],
                    aliases=tuple(item.get("aliases", [])),
                )
            )
    return entities


def fetch_wikipedia_page(title: str, *, timeout: int = 30) -> ParsedWikipediaPage:
    """Fetch and parse one Wikipedia page."""
    params = {
        "action": "parse",
        "page": title,
        "prop": "text",
        "format": "json",
        "formatversion": "2",
        "redirects": "1",
    }
    response: requests.Response | None = None
    for attempt in range(1, 5):
        response = requests.get(
            WIKIPEDIA_API_URL,
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
        )
        if response.status_code != 429:
            break
        retry_after = response.headers.get("Retry-After")
        if retry_after and retry_after.isdigit():
            wait_seconds = min(int(retry_after), 30)
        else:
            wait_seconds = min(2**attempt, 30)
        print(f"Rate limited by Wikipedia for {title}; retrying in {wait_seconds}s.")
        time.sleep(wait_seconds)
    if response is None:
        raise RuntimeError(f"No response received for {title}")
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    if "error" in payload:
        code = payload["error"].get("code", "unknown")
        info = payload["error"].get("info", "unknown error")
        raise RuntimeError(f"Wikipedia API error for {title}: {code}: {info}")

    parsed = payload["parse"]
    parser = WikipediaHTMLParser()
    parser.feed(parsed["text"])
    sections = parser.sections
    if not sections:
        raise RuntimeError(f"No meaningful paragraphs extracted for {title}")

    parsed_title = parsed.get("title", title)
    return ParsedWikipediaPage(
        title=parsed_title,
        source_url=wiki_source_url(parsed_title),
        sections=sections,
    )


def format_raw_text(entity: EntityConfig, page: ParsedWikipediaPage) -> str:
    """Render parsed sections as a local raw text document."""
    lines = [
        f"# {entity.name}",
        f"Type: {entity.entity_type}",
        f"Wikipedia title: {page.title}",
        f"Source URL: {page.source_url}",
        "",
    ]
    last_section: str | None = None
    for section in page.sections:
        section_title = section["section_title"] or "Lead"
        if section_title != last_section:
            lines.extend([f"## {section_title}", ""])
            last_section = section_title
        lines.extend([section["text"], ""])
    return "\n".join(lines).strip() + "\n"


def save_raw_text(entity: EntityConfig, page: ParsedWikipediaPage, raw_dir: Path = RAW_DIR) -> Path:
    """Save one parsed Wikipedia page under data/raw."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / raw_filename(entity.name)
    raw_path.write_text(format_raw_text(entity, page), encoding="utf-8")
    return raw_path


def raw_file_is_meaningful(path: Path, *, min_words: int = 200) -> bool:
    """Return whether a cached raw file is substantial enough to reuse."""
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    return len(text.split()) >= min_words and "Source URL:" in text


def cached_source_url(path: Path, fallback_title: str) -> str:
    """Read a source URL from a cached raw file, with a deterministic fallback."""
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("Source URL:"):
                return line.replace("Source URL:", "", 1).strip()
    return wiki_source_url(fallback_title)


def clear_raw_files(raw_dir: Path = RAW_DIR) -> None:
    """Remove generated raw text files."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for path in raw_dir.glob("*.txt"):
        path.unlink()


def ingest_entities(
    *,
    reset: bool = False,
    use_cache: bool = True,
    pause_seconds: float = 0.1,
    db_path: Path = SQLITE_PATH,
    raw_dir: Path = RAW_DIR,
) -> dict[str, Any]:
    """Ingest all configured entities and return a summary."""
    if reset:
        reset_database(db_path, force=True)
        clear_raw_files(raw_dir)
    else:
        initialize_database(db_path)

    entities = load_entity_configs()
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for index, entity in enumerate(entities, start=1):
        try:
            raw_path = raw_dir / raw_filename(entity.name)
            if use_cache and raw_file_is_meaningful(raw_path):
                entity_id = upsert_entity(
                    entity.name,
                    entity.entity_type,
                    cached_source_url(raw_path, entity.wikipedia_title),
                    str(raw_path.relative_to(PROJECT_ROOT)),
                    db_path=db_path,
                )
                word_count = len(raw_path.read_text(encoding="utf-8").split())
                successes.append(
                    {
                        "id": entity_id,
                        "name": entity.name,
                        "type": entity.entity_type,
                        "raw_path": str(raw_path),
                        "word_count": word_count,
                    }
                )
                print(
                    f"[{index:02d}/{len(entities)}] Reused cached {entity.entity_type}: "
                    f"{entity.name} ({word_count} words)"
                )
                time.sleep(pause_seconds)
                continue

            page = fetch_wikipedia_page(entity.wikipedia_title)
            raw_path = save_raw_text(entity, page, raw_dir)
            entity_id = upsert_entity(
                entity.name,
                entity.entity_type,
                page.source_url,
                str(raw_path.relative_to(PROJECT_ROOT)),
                db_path=db_path,
            )
            successes.append(
                {
                    "id": entity_id,
                    "name": entity.name,
                    "type": entity.entity_type,
                    "raw_path": str(raw_path),
                    "word_count": page.word_count,
                }
            )
            print(
                f"[{index:02d}/{len(entities)}] Ingested {entity.entity_type}: "
                f"{entity.name} ({page.word_count} words)"
            )
        except Exception as exc:  # pragma: no cover - surfaced in CLI output
            failures.append({"name": entity.name, "type": entity.entity_type, "error": str(exc)})
            print(f"[{index:02d}/{len(entities)}] FAILED {entity.entity_type}: {entity.name}: {exc}")
        time.sleep(pause_seconds)

    people_count = sum(1 for item in successes if item["type"] == "person")
    place_count = sum(1 for item in successes if item["type"] == "place")
    return {
        "total": len(entities),
        "success_count": len(successes),
        "failure_count": len(failures),
        "people_count": people_count,
        "place_count": place_count,
        "successes": successes,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest configured Wikipedia pages locally.")
    parser.add_argument("--reset", action="store_true", help="Reset SQLite metadata and raw text first.")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Fetch every page even when a local raw file already exists.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.1,
        help="Small pause between Wikipedia API requests.",
    )
    parser.add_argument(
        "--skip-chunking",
        action="store_true",
        help="Only fetch raw text and metadata; do not rebuild SQLite chunks.",
    )
    args = parser.parse_args()

    summary = ingest_entities(
        reset=args.reset,
        use_cache=not args.no_cache,
        pause_seconds=args.pause_seconds,
    )
    print(
        "Ingestion summary: "
        f"{summary['people_count']}/20 people, "
        f"{summary['place_count']}/20 places, "
        f"{summary['failure_count']} failures."
    )
    if summary["failures"]:
        sys.exit(1)
    if not args.skip_chunking:
        from app.chunker import rebuild_chunks_from_database, validate_chunks

        chunk_summary = rebuild_chunks_from_database()
        chunk_errors = validate_chunks()
        print(
            f"Chunking summary: {len(chunk_summary['entities'])} entities, "
            f"{chunk_summary['total_chunks']} chunks."
        )
        if chunk_errors:
            for error in chunk_errors:
                print(f"ERROR: {error}")
            sys.exit(1)


if __name__ == "__main__":
    main()
