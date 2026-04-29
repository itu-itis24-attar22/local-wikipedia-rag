"""Project validation command."""

from __future__ import annotations

from pathlib import Path

from app.chunker import validate_chunks
from app.config import (
    CHROMA_DIR,
    EMBEDDING_MODEL,
    ENTITIES_PATH,
    GENERATION_MODEL,
    REQUIRED_PEOPLE,
    REQUIRED_PLACES,
    SQLITE_PATH,
    load_entities_config,
    validate_entities_config,
)
from app.database import count_rows, list_entities
from app.embeddings import model_available
from app.ingest import raw_filename
from app.retriever import retrieve
from app.vector_store import collection_count


class ValidationReport:
    def __init__(self) -> None:
        self.errors: list[str] = []

    def pass_check(self, message: str) -> None:
        print(f"PASS: {message}")

    def fail_check(self, message: str) -> None:
        self.errors.append(message)
        print(f"FAIL: {message}")


def validate_entity_file(report: ValidationReport) -> None:
    errors = validate_entities_config(ENTITIES_PATH)
    if errors:
        for error in errors:
            report.fail_check(error)
        return
    data = load_entities_config(ENTITIES_PATH)
    people_names = {item["name"] for item in data["people"]}
    place_names = {item["name"] for item in data["places"]}
    if REQUIRED_PEOPLE <= people_names and REQUIRED_PLACES <= place_names:
        report.pass_check("20 people and 20 places configured with all required entities.")
    else:
        report.fail_check("Required entity set is incomplete.")


def validate_raw_files(report: ValidationReport) -> None:
    data = load_entities_config(ENTITIES_PATH)
    missing: list[str] = []
    small: list[str] = []
    for group in ("people", "places"):
        for entity in data[group]:
            path = Path("data/raw") / raw_filename(entity["name"])
            if not path.exists():
                missing.append(str(path))
            elif len(path.read_text(encoding="utf-8").split()) < 200:
                small.append(str(path))
    if missing:
        report.fail_check(f"Missing raw files: {missing}")
    elif small:
        report.fail_check(f"Raw files with too little text: {small}")
    else:
        report.pass_check("All raw files exist and contain meaningful text.")


def validate_sqlite(report: ValidationReport) -> None:
    if not SQLITE_PATH.exists():
        report.fail_check(f"SQLite database is missing: {SQLITE_PATH}")
        return
    entities = list_entities()
    counts = {"person": 0, "place": 0}
    for entity in entities:
        counts[entity["type"]] += 1
    if counts == {"person": 20, "place": 20}:
        report.pass_check("SQLite contains 20 people and 20 places.")
    else:
        report.fail_check(f"Unexpected SQLite entity counts: {counts}")

    chunk_errors = validate_chunks()
    if chunk_errors:
        for error in chunk_errors:
            report.fail_check(error)
    else:
        report.pass_check(f"SQLite chunks validated: {count_rows('chunks')} chunks.")


def validate_chroma(report: ValidationReport) -> None:
    if not CHROMA_DIR.exists():
        report.fail_check(f"Chroma directory is missing: {CHROMA_DIR}")
        return
    try:
        vectors = collection_count()
        chunks = count_rows("chunks")
    except Exception as exc:
        report.fail_check(f"Chroma validation failed: {exc}")
        return
    if vectors == chunks:
        report.pass_check(f"Chroma vector count matches SQLite chunks: {vectors}.")
    else:
        report.fail_check(f"Chroma vectors ({vectors}) != SQLite chunks ({chunks}).")


def validate_ollama(report: ValidationReport) -> None:
    for model in (GENERATION_MODEL, EMBEDDING_MODEL):
        try:
            if model_available(model):
                report.pass_check(f"Ollama model available: {model}.")
            else:
                report.fail_check(f"Ollama model missing: {model}. Run: ollama pull {model}")
        except Exception as exc:
            report.fail_check(f"Ollama check failed for {model}: {exc}")


def validate_sample_retrieval(report: ValidationReport) -> None:
    try:
        result = retrieve("Where is Hagia Sophia located?", top_k=4)
    except Exception as exc:
        report.fail_check(f"Sample retrieval failed: {exc}")
        return
    entity_names = {chunk.metadata.get("entity_name") for chunk in result.chunks}
    if "Hagia Sophia" in entity_names:
        report.pass_check("Sample retrieval returned Hagia Sophia chunks.")
    else:
        report.fail_check(f"Sample retrieval did not return Hagia Sophia: {entity_names}")


def main() -> None:
    report = ValidationReport()
    validate_entity_file(report)
    validate_raw_files(report)
    validate_sqlite(report)
    validate_chroma(report)
    validate_ollama(report)
    validate_sample_retrieval(report)

    if report.errors:
        print("\nValidation failed:")
        for error in report.errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("\nAll validation checks passed.")


if __name__ == "__main__":
    main()
