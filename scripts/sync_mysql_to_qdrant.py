"""Synchronize a generic MySQL document table into the configured vector store.

The source table and columns are configured through environment variables so
this script does not assume a particular application schema.
"""

import argparse
import os
import re
import sys
from collections.abc import Mapping
from typing import Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pymysql
import pymysql.cursors

from app.core.config import settings
from app.providers.factory import get_datasource_provider
from app.services.embedder import embedder
from app.services.vector_service import vector_service


BATCH_SIZE = 32
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def get_db_connection() -> pymysql.Connection:
    """Connect through the configured datasource provider."""
    return get_datasource_provider().connect()


def _identifier(value: str, label: str) -> str:
    if not IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"{label} must be a simple SQL identifier")
    return value


def _metadata_columns() -> list[str]:
    raw = settings.MYSQL_DOCUMENT_METADATA_COLUMNS
    return [_identifier(item.strip(), "MYSQL_DOCUMENT_METADATA_COLUMNS") for item in raw.split(",") if item.strip()]


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def load_documents(connection: pymysql.Connection) -> list[tuple[str, str, dict[str, Any]]]:
    table = _identifier(settings.MYSQL_DOCUMENT_TABLE, "MYSQL_DOCUMENT_TABLE")
    id_column = _identifier(settings.MYSQL_DOCUMENT_ID_COLUMN, "MYSQL_DOCUMENT_ID_COLUMN")
    text_column = _identifier(settings.MYSQL_DOCUMENT_TEXT_COLUMN, "MYSQL_DOCUMENT_TEXT_COLUMN")
    metadata_columns = _metadata_columns()
    selected_columns = ", ".join(dict.fromkeys([id_column, text_column, *metadata_columns]))

    with connection.cursor() as cursor:
        cursor.execute(f"SELECT {selected_columns} FROM {table};")
        rows = cursor.fetchall()

    documents: list[tuple[str, str, dict[str, Any]]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("MySQL datasource must return mapping rows")
        document_id = row.get(id_column)
        text = str(row.get(text_column) or "").strip()
        if document_id is None or not text:
            continue
        metadata = {
            column: _json_value(row.get(column))
            for column in metadata_columns
            if row.get(column) is not None
        }
        documents.append((str(document_id), text, metadata))
    return documents


def sync_documents(documents: list[tuple[str, str, dict[str, Any]]]) -> None:
    vector_service.ensure_collection()
    for start in range(0, len(documents), BATCH_SIZE):
        batch = documents[start : start + BATCH_SIZE]
        ids = [item[0] for item in batch]
        texts = [item[1] for item in batch]
        metadatas = [item[2] for item in batch]
        embeddings = embedder.embed_documents(texts)
        vector_service.upsert_documents(ids, texts, embeddings, metadatas)
        print(f"Synced {start + len(batch)}/{len(documents)} documents.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="read and validate source rows without embedding or writing vectors",
    )
    args = parser.parse_args(argv)

    try:
        if settings.DATASOURCE != "mysql":
            raise RuntimeError("MySQL synchronization requires DATASOURCE=mysql")

        connection = get_db_connection()
        try:
            documents = load_documents(connection)
        finally:
            connection.close()

        if args.dry_run:
            print(f"Dry run: {len(documents)} documents would be synced from '{settings.MYSQL_DOCUMENT_TABLE}'.")
            return 0
        if not documents:
            print("No non-empty documents found; nothing to sync.")
            return 0
        sync_documents(documents)
        print(f"Synchronized {len(documents)} documents into '{settings.QDRANT_COLLECTION}'.")
        return 0
    except pymysql.err.OperationalError:
        print(
            "ERROR: MySQL connection failed. Check MYSQL_HOST, MYSQL_PORT, "
            "MYSQL_USER, MYSQL_PASSWORD, and MYSQL_DATABASE.",
            file=sys.stderr,
        )
    except pymysql.err.ProgrammingError:
        print(
            "ERROR: MySQL query failed. Check MYSQL_DOCUMENT_TABLE, "
            "MYSQL_DOCUMENT_ID_COLUMN, MYSQL_DOCUMENT_TEXT_COLUMN, and metadata columns.",
            file=sys.stderr,
        )
    except ValueError as exc:
        print(f"ERROR: invalid configuration: {exc}", file=sys.stderr)
    except Exception as exc:
        print(f"ERROR: synchronization failed ({type(exc).__name__}).", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
