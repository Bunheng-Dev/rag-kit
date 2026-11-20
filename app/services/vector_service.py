"""Backward-compatible vector service facade."""

from app.providers.factory import get_vector_db_provider
from app.providers.vectordb.base import VectorDBProvider


class VectorStoreService:
    def __init__(self, provider: VectorDBProvider | None = None) -> None:
        self._provider = provider or get_vector_db_provider()

    def __getattr__(self, name: str):
        return getattr(self._provider, name)


vector_service = VectorStoreService()
