# Demo Script

Target length: about 5 minutes.

1. Project overview
   - Explain that the system is a local Wikipedia RAG assistant for famous people and places.
   - Mention no external LLM API, no cloud vector database, and no LangChain.

2. Architecture walkthrough
   - Show `data/entities.json`.
   - Show the pipeline: Wikipedia pages, raw text, SQLite metadata, chunking, Ollama embeddings, Chroma, classifier, retriever, prompt builder, Ollama generation, UI.

3. Live ingestion and indexing
   - Run `python -m app.ingest --reset`.
   - Point out raw files, SQLite, and Chroma persistence.

4. Live question answering
   - Run `streamlit run app/ui_streamlit.py`.
   - Ask one person question, one place question, one comparison question, and one failure question.
   - Toggle retrieved context.

5. Tradeoffs and limitations
   - Local models are private but slower.
   - Wikipedia ingestion can be rate-limited.
   - Rule-based classifier is simple but inspectable.

6. Closing
   - Show `python -m app.validate`.
   - Mention future improvements: reranking, hybrid search, streaming, and stronger local models.
