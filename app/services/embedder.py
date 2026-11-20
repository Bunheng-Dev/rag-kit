"""Backward-compatible embedding service facade."""

from collections.abc import Sequence

from app.providers.embedding.base import EmbeddingProvider
from app.providers.factory import get_embedding_provider


class EmbeddingService:
    def __init__(self, provider: EmbeddingProvider | None = None) -> None:
        self._provider = provider

    @property
    def provider(self) -> EmbeddingProvider:
        if self._provider is None:
            self._provider = get_embedding_provider()
        return self._provider

    def embed_text(self, text: str) -> list[float]:
        return self.provider.embed_text(text)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self.provider.embed_documents(texts)


embedder = EmbeddingService()
