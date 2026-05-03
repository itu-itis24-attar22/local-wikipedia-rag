# Local Wikipedia RAG Assistant

## Project Overview

Local Wikipedia RAG Assistant is a localhost-only BLG483E HW3 project. It ingests Wikipedia pages for famous people and famous places, stores cleaned raw text locally, chunks the documents, embeds chunks with Ollama `nomic-embed-text`, stores vectors in a persistent Chroma collection, and answers questions with Ollama `llama3.2:3b`.

The system does not use OpenAI, external LLM APIs, hosted vector databases, or LangChain.

## Features

- 20 famous people and 20 famous places configured in `data/entities.json`
- Required homework entities included
- Clean Wikipedia text saved under `data/raw/`
- SQLite metadata database at `data/wiki_rag.sqlite`
- Paragraph-aware chunking around 700 words with 100-word overlap
- One Chroma collection named `wiki_chunks` with metadata filters
- Rule-based person, place, both, and unknown query classification
- Local-only embeddings and generation through Ollama
- Streamlit chat UI with optional retrieved context
- CLI using the same RAG pipeline
- Safe fallback answer: `I don't know based on the local Wikipedia data.`

## Requirements

- Python 3.10 or newer
- Ollama installed and running locally
- Internet access only for Wikipedia ingestion and initial model pulls

## Installation

```powershell
git clone https://github.com/itu-itis24-attar22/local-wikipedia-rag
cd local-wikipedia-rag
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Ollama Setup

Install Ollama from `https://ollama.com`, start it, then pull the two local models:

```powershell
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

You can confirm Ollama is reachable with:

```powershell
ollama list
```

## Ingest And Build The Index

This command resets local generated data, fetches Wikipedia pages, stores metadata in SQLite, chunks documents, embeds chunks locally, and writes Chroma vectors under `data/chroma/`.

```powershell
python -m app.ingest --reset
```

If Wikipedia rate-limits the run, rerun without reset so cached pages are reused:

```powershell
python -m app.ingest --pause-seconds 1.0
```

For ingestion and chunking without embeddings:

```powershell
python -m app.ingest --reset --skip-indexing
```

## Run The Streamlit UI

```powershell
streamlit run app/ui_streamlit.py
```

The sidebar includes:

- Show retrieved context
- Clear chat
- Reset local data

## Run The CLI

Interactive mode:

```powershell
python -m app.cli
```

One-shot mode:

```powershell
python -m app.cli "Who was Albert Einstein and what is he known for?"
python -m app.cli --show-context "Compare Albert Einstein and Nikola Tesla."
```

## Example Questions

- Who was Albert Einstein and what is he known for?
- What did Marie Curie discover?
- Why is Nikola Tesla famous?
- Compare Lionel Messi and Cristiano Ronaldo.
- Where is the Eiffel Tower located?
- Why is the Great Wall of China important?
- Which famous place is located in Turkey?
- Which person is associated with electricity?
- Who is the president of Mars?

## Reset Instructions

Reset and rebuild everything:

```powershell
python -m app.ingest --reset
```

Reset only the SQLite tables:

```powershell
python -m app.database --reset
```

Rebuild chunks from existing raw files:

```powershell
python -m app.chunker
```

Rebuild Chroma vectors from existing SQLite chunks:

```powershell
python -m app.vector_store --reset
```

## Validation

After ingestion and indexing:

```powershell
python -m app.validate
```

Run tests:

```powershell
pytest
```

## Design Decisions

- The project uses one Chroma collection with metadata instead of separate person and place stores. This keeps comparisons and mixed queries simple while still allowing type filters.
- SQLite stores durable metadata and chunk text. Chroma stores vector search data and repeats the metadata needed for filtering and source display.
- The classifier is rule-based by design because the homework allows simple keyword logic and it keeps query routing inspectable.
- Chunking is paragraph-aware, uses overlap, and handles large paragraphs with sliding windows.
- The prompt instructs the local model to answer only from retrieved context and return the exact safe fallback when context is insufficient.

## Limitations

- Answers are only as complete as the ingested Wikipedia pages and retrieved chunks.
- The local model can still make mistakes, so source context should be checked for important answers.
- First indexing can take time because all embeddings are generated locally.
- Wikipedia ingestion depends on network availability and can be rate-limited.
- The current classifier is intentionally simple and may route ambiguous queries broadly.

## Demo Video

Demo video link: TODO
