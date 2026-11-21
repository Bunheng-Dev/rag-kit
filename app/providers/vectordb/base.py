from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence


class VectorDBProvider(ABC):
    @abstractmethod
    def point_id(self, document_id: str | int) -> str:
        """Return the deterministic point ID for a source document ID."""

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

    @abstractmethod
    def list_point_ids(
        self,
        where: Mapping[str, object] | None = None,
    ) -> list[str]:
        """List point IDs matching a payload filter, without payloads or vectors."""

    @abstractmethod
    def delete_points(self, point_ids: Sequence[str | int]) -> None:
        """Delete points by their database point IDs."""
