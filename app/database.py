"""SQLite metadata storage for entities and chunks."""

from __future__ import annotations

import argparse
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import SQLITE_PATH


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    type TEXT NOT NULL CHECK (type IN ('person', 'place')),
    source_url TEXT NOT NULL,
    raw_path TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    UNIQUE(name, type)
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    section_title TEXT,
    text TEXT NOT NULL,
    word_count INTEGER NOT NULL,
    FOREIGN KEY(entity_id) REFERENCES entities(id) ON DELETE CASCADE,
    UNIQUE(entity_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);
CREATE INDEX IF NOT EXISTS idx_chunks_entity_id ON chunks(entity_id);
"""


def connect(db_path: Path = SQLITE_PATH) -> sqlite3.Connection:
    """Open a SQLite connection with rows returned as dictionaries."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON;")
    return connection


def initialize_database(db_path: Path = SQLITE_PATH) -> None:
    """Create the database schema if it does not already exist."""
    with connect(db_path) as connection:
        connection.executescript(SCHEMA_SQL)


def reset_database(db_path: Path = SQLITE_PATH, *, force: bool = False) -> None:
    """Drop and recreate project tables.

    The force flag is intentionally required because reset deletes all local
    metadata and chunk rows. Raw files and Chroma data are reset elsewhere.
    """
    if not force:
        raise ValueError("reset_database requires force=True.")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with connect(db_path) as connection:
        connection.executescript(
            """
            PRAGMA foreign_keys = OFF;
            DROP TABLE IF EXISTS chunks;
            DROP TABLE IF EXISTS entities;
            PRAGMA foreign_keys = ON;
            """
        )
        connection.executescript(SCHEMA_SQL)


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def upsert_entity(
    name: str,
    entity_type: str,
    source_url: str,
    raw_path: str,
    *,
    db_path: Path = SQLITE_PATH,
    ingested_at: str | None = None,
) -> int:
    """Insert or update an entity and return its database id."""
    initialize_database(db_path)
    timestamp = ingested_at or utc_now_iso()
    with connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO entities (name, type, source_url, raw_path, ingested_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(name, type) DO UPDATE SET
                source_url = excluded.source_url,
                raw_path = excluded.raw_path,
                ingested_at = excluded.ingested_at
            """,
            (name, entity_type, source_url, raw_path, timestamp),
        )
        row = connection.execute(
            "SELECT id FROM entities WHERE name = ? AND type = ?",
            (name, entity_type),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"Failed to upsert entity: {name}")
        return int(row["id"])


def clear_chunks_for_entity(entity_id: int, *, db_path: Path = SQLITE_PATH) -> None:
    """Delete all chunks for one entity."""
    with connect(db_path) as connection:
        connection.execute("DELETE FROM chunks WHERE entity_id = ?", (entity_id,))


def insert_chunks(
    entity_id: int,
    chunks: Iterable[dict[str, Any]],
    *,
    db_path: Path = SQLITE_PATH,
    replace_existing: bool = True,
) -> int:
    """Insert chunk rows for one entity and return the number inserted."""
    initialize_database(db_path)
    chunk_rows = list(chunks)
    with connect(db_path) as connection:
        if replace_existing:
            connection.execute("DELETE FROM chunks WHERE entity_id = ?", (entity_id,))
        connection.executemany(
            """
            INSERT INTO chunks (entity_id, chunk_index, section_title, text, word_count)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    entity_id,
                    int(chunk["chunk_index"]),
                    chunk.get("section_title") or "",
                    chunk["text"],
                    int(chunk["word_count"]),
                )
                for chunk in chunk_rows
            ],
        )
    return len(chunk_rows)


def list_entities(
    *,
    db_path: Path = SQLITE_PATH,
    entity_type: str | None = None,
) -> list[dict[str, Any]]:
    """Return stored entities, optionally filtered by type."""
    initialize_database(db_path)
    with connect(db_path) as connection:
        if entity_type:
            rows = connection.execute(
                "SELECT * FROM entities WHERE type = ? ORDER BY name",
                (entity_type,),
            ).fetchall()
        else:
            rows = connection.execute("SELECT * FROM entities ORDER BY type, name").fetchall()
    return [dict(row) for row in rows]


def get_entity_by_name(
    name: str,
    *,
    db_path: Path = SQLITE_PATH,
) -> dict[str, Any] | None:
    """Return one entity by display name."""
    initialize_database(db_path)
    with connect(db_path) as connection:
        row = connection.execute(
            "SELECT * FROM entities WHERE lower(name) = lower(?)",
            (name,),
        ).fetchone()
    return dict(row) if row else None


def list_chunks(
    *,
    db_path: Path = SQLITE_PATH,
    entity_id: int | None = None,
) -> list[dict[str, Any]]:
    """Return stored chunks joined with entity metadata."""
    initialize_database(db_path)
    base_query = """
        SELECT
            chunks.id,
            chunks.entity_id,
            chunks.chunk_index,
            chunks.section_title,
            chunks.text,
            chunks.word_count,
            entities.name AS entity_name,
            entities.type AS entity_type,
            entities.source_url
        FROM chunks
        JOIN entities ON entities.id = chunks.entity_id
    """
    with connect(db_path) as connection:
        if entity_id is None:
            rows = connection.execute(
                base_query + " ORDER BY entities.type, entities.name, chunks.chunk_index"
            ).fetchall()
        else:
            rows = connection.execute(
                base_query + " WHERE chunks.entity_id = ? ORDER BY chunks.chunk_index",
                (entity_id,),
            ).fetchall()
    return [dict(row) for row in rows]


def count_rows(table_name: str, *, db_path: Path = SQLITE_PATH) -> int:
    """Count rows in one project table."""
    if table_name not in {"entities", "chunks"}:
        raise ValueError("table_name must be 'entities' or 'chunks'.")
    initialize_database(db_path)
    with connect(db_path) as connection:
        row = connection.execute(f"SELECT COUNT(*) AS count FROM {table_name}").fetchone()
    return int(row["count"])


def table_names(*, db_path: Path = SQLITE_PATH) -> set[str]:
    """Return the set of table names in the SQLite database."""
    initialize_database(db_path)
    with connect(db_path) as connection:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    return {str(row["name"]) for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage the local SQLite metadata store.")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate project tables.")
    args = parser.parse_args()

    if args.reset:
        reset_database(force=True)
        print(f"Reset SQLite database at {SQLITE_PATH}")
    else:
        initialize_database()
        print(f"Initialized SQLite database at {SQLITE_PATH}")
        print(f"Tables: {', '.join(sorted(table_names()))}")


if __name__ == "__main__":
    main()
