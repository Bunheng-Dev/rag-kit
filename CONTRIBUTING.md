# Contributing to rag-kit

Thanks for helping improve rag-kit. Keep changes focused, provider-neutral, and easy to run locally.

## Development setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Copy `.env.example` to `.env` for local configuration. Never commit `.env`, API keys, database passwords, tokens, or generated vector data.

## Provider changes

Implement the appropriate abstract interface under `app/providers`, add a factory registration, document the environment variables, and include tests that do not require external credentials. Keep query and ingestion embeddings on the same model and dimension.

## Checks before opening a pull request

```bash
python -m compileall -q app scripts tests
ruff check .
pytest -q
```

For API changes, also run the smoke test against a configured local server:

```bash
python scripts/test_rag_pipeline.py
```

Describe any external services, credentials, or manual verification required by the change. Do not include secrets in issues or pull requests.
