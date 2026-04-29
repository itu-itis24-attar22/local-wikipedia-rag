# Product PRD: Local Wikipedia RAG Assistant

## Product Goal

Build a local retrieval augmented generation assistant that answers questions about configured famous people and famous places using only locally stored Wikipedia data, local embeddings, a local vector database, and a local language model.

## Users

- BLG483E instructor evaluating the homework
- Student demonstrating local RAG architecture
- Local user asking questions about the configured Wikipedia entities

## Functional Requirements

- Ingest exactly 20 famous people and 20 famous places from `data/entities.json`
- Include all required homework entities
- Fetch and clean Wikipedia article text
- Store raw text under `data/raw/`
- Store entity metadata and chunks in SQLite
- Chunk documents with a large-document strategy
- Generate embeddings locally with Ollama `nomic-embed-text`
- Store vectors in one Chroma collection named `wiki_chunks`
- Classify queries as person, place, both, or unknown
- Retrieve relevant chunks with metadata filters
- Generate answers with Ollama `llama3.2:3b`
- Return the exact safe fallback when context is missing
- Provide Streamlit chat UI and CLI
- Allow optional retrieved context display
- Allow clearing chat and resetting local generated data

## Non-Functional Requirements

- Runs fully on localhost after Wikipedia data and models are downloaded
- Does not use external LLM APIs
- Does not use cloud vector databases
- Avoids LangChain or frameworks that hide the RAG implementation
- Is understandable from source code and README alone
- Uses reproducible commands for setup, ingestion, validation, UI, and CLI

## Acceptance Criteria

- `python -m app.ingest --reset` builds raw data, SQLite chunks, and Chroma vectors
- SQLite contains 20 people and 20 places after ingestion
- Every entity has at least one chunk
- Chroma vector count equals SQLite chunk count
- Required validation queries retrieve the expected entities
- Failure queries return `I don't know based on the local Wikipedia data.`
- Streamlit UI answers questions and can show context
- README instructions are enough for the instructor to run the project

## Out Of Scope

- Multi-user authentication
- Cloud deployment as part of homework submission
- External hosted LLM or embedding APIs
- Continuous Wikipedia synchronization
- Advanced entity linking beyond the configured entity list
- Long-term chat memory
