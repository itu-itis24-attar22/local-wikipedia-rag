# Limitations

- The configured dataset is intentionally small: 20 people and 20 places.
- Retrieval quality depends on the chunking strategy and embedding model.
- The rule-based classifier can miss unusual phrasings.
- Local model answers can still be imperfect, so retrieved context should be reviewed.
- First indexing can be slow on CPU-only machines.
- Wikipedia rate limits may require rerunning ingestion with a slower pause.
- The Streamlit reset button removes generated local data and requires rebuilding before answering again.
