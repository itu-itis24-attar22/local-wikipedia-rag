"""Project configuration constants."""

from pathlib import Path
from typing import Any
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CHROMA_DIR = DATA_DIR / "chroma"
SQLITE_PATH = DATA_DIR / "wiki_rag.sqlite"
ENTITIES_PATH = DATA_DIR / "entities.json"

CHROMA_COLLECTION = "wiki_chunks"
GENERATION_MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"

CHUNK_TARGET_WORDS = 700
CHUNK_OVERLAP_WORDS = 100
RETRIEVAL_TOP_K = 6
COMPARISON_TOP_K_PER_ENTITY = 4
GENERATION_TEMPERATURE = 0.2

UNKNOWN_ANSWER = "I don't know based on the local Wikipedia data."

REQUIRED_PEOPLE = {
    "Albert Einstein",
    "Marie Curie",
    "Leonardo da Vinci",
    "William Shakespeare",
    "Ada Lovelace",
    "Nikola Tesla",
    "Lionel Messi",
    "Cristiano Ronaldo",
    "Taylor Swift",
    "Frida Kahlo",
}

REQUIRED_PLACES = {
    "Eiffel Tower",
    "Great Wall of China",
    "Taj Mahal",
    "Grand Canyon",
    "Machu Picchu",
    "Colosseum",
    "Hagia Sophia",
    "Statue of Liberty",
    "Pyramids of Giza",
    "Mount Everest",
}


def load_entities_config(path: Path = ENTITIES_PATH) -> dict[str, list[dict[str, Any]]]:
    """Load the configured people and places from JSON."""
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return data


def validate_entities_config(path: Path = ENTITIES_PATH) -> list[str]:
    """Return validation errors for the entity configuration."""
    data = load_entities_config(path)
    errors: list[str] = []
    people = data.get("people", [])
    places = data.get("places", [])
    people_names = {item.get("name") for item in people}
    place_names = {item.get("name") for item in places}

    if len(people) != 20:
        errors.append(f"Expected 20 people, found {len(people)}.")
    if len(places) != 20:
        errors.append(f"Expected 20 places, found {len(places)}.")

    missing_people = sorted(REQUIRED_PEOPLE - people_names)
    missing_places = sorted(REQUIRED_PLACES - place_names)
    if missing_people:
        errors.append(f"Missing required people: {', '.join(missing_people)}.")
    if missing_places:
        errors.append(f"Missing required places: {', '.join(missing_places)}.")

    for group_name, items in (("people", people), ("places", places)):
        seen: set[str] = set()
        for item in items:
            name = item.get("name")
            title = item.get("wikipedia_title")
            if not name or not title:
                errors.append(f"Every {group_name} entry must include name and wikipedia_title.")
            if name in seen:
                errors.append(f"Duplicate {group_name} entry: {name}.")
            seen.add(name)

    return errors
