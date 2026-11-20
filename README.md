# rag-kit

Reusable, provider-swappable Retrieval-Augmented Generation (RAG) API built with FastAPI. The default stack uses Gemini for generation and embeddings, Qdrant for vector search, and MySQL for ingestion.

## Features

- Provider interfaces under `app/providers/{llm,embedding,vectordb,datasource}`.
- Provider selection through `LLM_PROVIDER`, `EMBEDDING_PROVIDER`, `VECTOR_DB`, and `DATASOURCE`.
- Pydantic Settings configuration with no credentials in tracked files.
- Configurable embedding model and `EMBEDDING_DIM`; Qdrant collections are created or validated automatically.
- Prompt templates loaded from `PROMPT_TEMPLATE_PATH` or inline `PROMPT_TEMPLATE`.
- Existing endpoints: `GET /health` and `POST /api/chat/stream` (SSE).
- Lightweight runtime dependencies: no Torch or ChromaDB.

## Layout

```text
rag-kit/
├── app/
│   ├── api/chat.py
│   ├── core/{config.py,limiter.py,security.py}
│   ├── providers/
│   │   ├── llm/{base.py,gemini.py}
│   │   ├── embedding/{base.py,gemini.py}
│   │   ├── vectordb/{base.py,qdrant.py}
│   │   ├── datasource/{base.py,mysql.py}
│   │   └── factory.py
│   └── services/                 # compatibility facades
├── prompts/system.txt
├── scripts/
├── docker-compose.yml
├── pyproject.toml
└── .env.example
```

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
copy .env.example .env       # Windows
# cp .env.example .env       # Linux/macOS
```

Set a real `GEMINI_API_KEY`, a strong `CHAT_INTERNAL_KEY`, and the desired Qdrant/MySQL connection values in `.env`. Never commit `.env`.

Run the API locally:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 9000 --reload
```

Check `http://localhost:9000/health`. Chat requests require the `X-Chat-API-Key` header and use the existing SSE endpoint:

```bash
curl -N http://localhost:9000/api/chat/stream \
  -H "Content-Type: application/json" \
  -H "X-Chat-API-Key: $CHAT_INTERNAL_KEY" \
  -d '{"message":"What information is available?"}'
```

## Docker profiles

The API service is shared by both profiles. `cloud` adds the optional Cloudflare tunnel and expects external Gemini/Qdrant/MySQL services. `selfhost` adds a local Qdrant container; set `QDRANT_URL=http://qdrant:6333` in `.env`.

```bash
docker compose --profile selfhost up --build
docker compose --profile cloud up --build
```

Do not enable `cloud` without setting `CLOUDFLARE_TOKEN`.

## Ingestion

`scripts/sync_mysql_to_qdrant.py` is the current MySQL schema adapter. It obtains its connection through the `DATASOURCE` provider and writes vectors through the configured vector provider:

```bash
python scripts/sync_mysql_to_qdrant.py
```

The adapter contains the source schema and content-format rules for the copied dataset. Replace it with a project-specific adapter when using a different MySQL schema.

## Provider extension

Implement the relevant abstract interface in `app/providers`, then register the provider name in `app/providers/factory.py`. Keep ingestion and query embeddings on the same model and dimension; changing either requires rebuilding the target vector collection.

## Verification

```bash
python -m compileall -q app scripts
python scripts/test_rag_pipeline.py
```

The pipeline script requires a running API and configured external providers. Compile/import checks do not prove provider credentials, Qdrant connectivity, MySQL connectivity, or production readiness.
