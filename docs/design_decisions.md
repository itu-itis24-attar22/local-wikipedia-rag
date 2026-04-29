# Design Decisions

## One Chroma Collection

The project uses one collection, `wiki_chunks`, with metadata fields for entity name, entity type, source URL, chunk index, and section title. This supports person-only, place-only, and mixed comparison queries without duplicating vector-store logic.

## SQLite Plus Chroma

SQLite is the durable metadata and chunk text store. Chroma is the semantic retrieval index. Keeping both makes validation easier because the system can compare SQLite chunk counts against Chroma vector counts.

## Rule-Based Classifier

The classifier is intentionally simple. Exact entity and alias matches are preferred. Keyword clues route broader queries, and a few hint rules help examples such as Turkey to Hagia Sophia and electricity to Nikola Tesla.

## Chunking

Chunking is paragraph-aware with a target of 700 words and 100 words of overlap. Very large paragraphs are split with a sliding window. Section titles are preserved when available.

## Prompting

The generation prompt includes only retrieved context and tells the local model to use the exact safe fallback when the answer is unavailable.
