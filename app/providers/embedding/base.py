from abc import ABC, abstractmethod
from collections.abc import Sequence


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the configured vector dimension."""

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Embed one text value."""

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed multiple text values."""
