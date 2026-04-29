# Local Wikipedia RAG Assistant

Local Wikipedia RAG Assistant is a localhost-only retrieval augmented generation project for BLG483E HW3. It ingests Wikipedia pages for famous people and places, chunks them, embeds them with a local Ollama embedding model, stores vectors in Chroma, stores metadata in SQLite, and answers questions with a local Ollama LLM.

This README will be expanded as the project is implemented.

## Quick Start

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
ollama pull llama3.2:3b
ollama pull nomic-embed-text
python -m app.ingest --reset
streamlit run app/ui_streamlit.py
```

## Demo Video

Demo video link: TODO
