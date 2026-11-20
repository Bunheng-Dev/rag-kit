from abc import ABC, abstractmethod
from typing import Any


class DataSourceProvider(ABC):
    @abstractmethod
    def connect(self) -> Any:
        """Return a live source-database connection."""

    def read_documents(self) -> list[dict[str, Any]]:
        """Return normalized documents for file-style ingestion providers."""
        raise NotImplementedError(f"{type(self).__name__} does not support document reads")
