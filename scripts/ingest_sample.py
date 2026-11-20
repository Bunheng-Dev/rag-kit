"""Ingest the bundled JSON dataset without requiring MySQL."""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from app.providers.factory import get_datasource_provider
from app.services.embedder import embedder
from app.services.vector_service import vector_service


def main() -> None:
    if settings.DATASOURCE != "file":
        raise RuntimeError(
            "Sample ingestion requires DATASOURCE=file. Set it in .env or run "
            "DATASOURCE=file python scripts/ingest_sample.py."
        )

    documents = get_datasource_provider().read_documents()
    if not documents:
        raise RuntimeError("The sample dataset is empty")

    vector_service.ensure_collection()
    ids = [document["id"] for document in documents]
    texts = [document["text"] for document in documents]
    metadatas = [document["metadata"] for document in documents]
    embeddings = embedder.embed_documents(texts)
    vector_service.upsert_documents(ids, texts, embeddings, metadatas)
    print(f"Ingested {len(documents)} sample documents into '{settings.QDRANT_COLLECTION}'.")


if __name__ == "__main__":
    main()
