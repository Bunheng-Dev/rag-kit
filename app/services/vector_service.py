"""Backward-compatible vector service facade."""

from app.providers.factory import get_vector_db_provider
from app.providers.vectordb.base import VectorDBProvider


class VectorStoreService:
    def __init__(self, provider: VectorDBProvider | None = None) -> None:
        self._provider = provider

    @property
    def provider(self) -> VectorDBProvider:
        if self._provider is None:
            self._provider = get_vector_db_provider()
        return self._provider

    def __getattr__(self, name: str):
        return getattr(self.provider, name)


vector_service = VectorStoreService()
