from abc import ABC, abstractmethod
from typing import Any


class DataSourceProvider(ABC):
    @abstractmethod
    def connect(self) -> Any:
        """Return a live source-database connection."""
