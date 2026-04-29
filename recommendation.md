# Production Recommendation

## Deployment

For production, package the app as a containerized service with separate volumes for SQLite, Chroma, and raw text. Keep Ollama or another local model server on the same host or private network. Add health checks for the UI, API, vector store, and model server.

## Model Recommendation

`llama3.2:3b` is appropriate for this homework because it is lightweight and local. For production quality, evaluate a stronger local model such as Llama 3.1 8B, Qwen 2.5 7B, or a domain-tuned model if hardware allows. Keep `nomic-embed-text` as a reasonable local embedding baseline, then benchmark against larger local embedding models.

## Retrieval Improvements

- Add hybrid lexical plus vector retrieval
- Use reranking for the final context set
- Add better entity linking for partial names and aliases
- Tune chunk size by retrieval evaluation rather than one fixed default
- Store section hierarchy and article lead separately
- Add citation snippets with highlighted matched text

## Security And Privacy

- Keep all LLM and embedding calls local
- Do not log sensitive user questions by default
- Validate reset operations with confirmation
- Pin dependencies and scan them before deployment
- Restrict any future network ingestion jobs to trusted domains

## Limitations

- Wikipedia can contain incomplete or disputed information
- Local models may still hallucinate despite grounded prompts
- Rule-based classification does not cover every phrasing
- Chroma and SQLite are simple local stores, not high-availability services
- Large-scale ingestion would need job scheduling, retries, and monitoring

## Future Improvements

- Add an evaluation dashboard for the provided query set
- Stream model responses token by token
- Add source citation links in every answer
- Add a small admin page for rebuild status
- Add automated latency and retrieval-quality reports
