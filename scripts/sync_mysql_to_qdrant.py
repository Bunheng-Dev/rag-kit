"""Synchronize a generic MySQL document table into the configured vector store.

The source table and columns are configured through environment variables so
this script does not assume a particular application schema.
"""

import argparse
import os
import sys
from typing import Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pymysql
import pymysql.cursors

from app.core.config import settings
from app.providers.datasource.mysql import MySQLDataSourceProvider
from app.providers.factory import get_datasource_provider
from app.services.secret_scanner import SecretScanError
from app.services.embedder import embedder
from app.services.vector_service import vector_service


BATCH_SIZE = 32
def get_db_connection() -> pymysql.Connection:
    """Connect through the configured datasource provider."""
    return get_datasource_provider().connect()


def load_documents(connection: pymysql.Connection) -> list[tuple[str, str, dict[str, Any]]]:
    return MySQLDataSourceProvider().read_documents(connection)


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
    except SecretScanError:
        print("ERROR: secret scan blocked ingestion; see the value-free findings above.", file=sys.stderr)
    except ValueError as exc:
        print(f"ERROR: invalid configuration: {exc}", file=sys.stderr)
    except Exception as exc:
        print(f"ERROR: synchronization failed ({type(exc).__name__}).", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
