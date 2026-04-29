"""Project configuration constants."""

from pathlib import Path


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
