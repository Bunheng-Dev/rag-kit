from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence


class VectorDBProvider(ABC):
    @abstractmethod
    def ensure_collection(self) -> None:
        """Create or validate the configured collection."""

    @abstractmethod
    def query(
        self,
        query_vector: Sequence[float],
        n_results: int = 3,
        where: Mapping[str, object] | None = None,
    ) -> dict[str, list[list[dict | str]]]:
        """Return documents and metadata in the API-compatible result shape."""

    @abstractmethod
    def upsert_documents(
        self,
        ids: Sequence[str | int],
        documents: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[dict] | None = None,
    ) -> None:
        """Insert or replace documents."""

    @abstractmethod
    def delete_documents(self, ids: Sequence[str | int]) -> None:
        """Delete documents by source id."""
