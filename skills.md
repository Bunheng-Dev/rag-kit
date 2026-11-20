# rag-kit contributor guidance

This project is a provider-swappable FastAPI RAG toolkit. Keep provider-specific SDK calls behind the interfaces in `app/providers` and keep the API layer independent of concrete vendors.

## Runtime contract

- API framework: FastAPI + Uvicorn.
- LLM, embeddings, vector database, and datasource are selected through environment variables.
- The default providers are Gemini, Gemini embeddings, Qdrant, and MySQL.
- Query-time retrieval reads the vector database; MySQL is used by the ingestion adapter.
- Query and ingestion embeddings must use the same model and `EMBEDDING_DIM`.
- The API contract is `GET /health` and `POST /api/chat/stream` with SSE output.

## Engineering rules

- Do not commit `.env`, credentials, tokens, or generated vector data.
- Keep dependencies lightweight; do not add Torch or ChromaDB for the default path.
- Use typed Python and preserve the compatibility facades in `app/services` unless a migration is intentional.
- New providers need a `base.py` interface implementation and a factory registration.
- Validate collection dimensions before writing vectors.
- Treat the MySQL ingestion script as a schema adapter: replace its source-specific extraction rules for a new dataset.
