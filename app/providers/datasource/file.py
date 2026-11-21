import json
from pathlib import Path
from typing import Any

from app.core.config import PROJECT_ROOT, settings
from app.services.secret_scanner import SecretScanError, guard_documents, print_scan_summary
from .base import DataSourceProvider


class FileDataSourceProvider(DataSourceProvider):
    """Load normalized RAG documents from a local JSON file."""

    def __init__(self, path: Path | None = None) -> None:
        configured_path = Path(settings.SAMPLE_DATA_PATH)
        self.path = path or (configured_path if configured_path.is_absolute() else PROJECT_ROOT / configured_path)

    def connect(self) -> "FileDataSourceProvider":
        return self

    def read_documents(self) -> list[dict[str, Any]]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Sample dataset not found: {self.path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON dataset: {self.path}: {exc}") from exc

        if not isinstance(payload, list):
            raise ValueError(f"Dataset must contain a JSON array: {self.path}")

        documents: list[dict[str, Any]] = []
        for index, item in enumerate(payload):
            if not isinstance(item, dict) or not item.get("id") or not item.get("text"):
                raise ValueError(f"Dataset item {index} must contain non-empty 'id' and 'text' fields")
            metadata = item.get("metadata", {})
            if not isinstance(metadata, dict):
                raise ValueError(f"Dataset item {index} metadata must be an object")
            documents.append({"id": str(item["id"]), "text": str(item["text"]), "metadata": metadata})
        records = [(document["id"], document["text"], document["metadata"]) for document in documents]
        try:
            guarded, summaries = guard_documents(records)
        except SecretScanError as exc:
            print_scan_summary(exc.summaries)
            raise
        print_scan_summary(summaries)
        return [{"id": source_id, "text": text, "metadata": metadata} for source_id, text, metadata in guarded]
