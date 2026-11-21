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
PRUNE_DELETE_THRESHOLD = 0.5
PRUNE_DELETE_BATCH_SIZE = 100
SYNC_SOURCE_PAYLOAD_KEY = "sync_source"


class PruneSafetyError(RuntimeError):
    """Raised when a prune safety guard prevents deletion."""


def get_db_connection() -> pymysql.Connection:
    """Connect through the configured datasource provider."""
    return get_datasource_provider().connect()


def load_documents(connection: pymysql.Connection) -> list[tuple[str, str, dict[str, Any]]]:
    return MySQLDataSourceProvider().read_documents(connection)


def sync_source_marker() -> str:
    return f"mysql:{settings.MYSQL_DOCUMENT_TABLE}"


def document_point_ids(documents: list[tuple[str, str, dict[str, Any]]]) -> set[str]:
    return {str(vector_service.point_id(document_id)) for document_id, _, _ in documents}


def sync_documents(documents: list[tuple[str, str, dict[str, Any]]]) -> int:
    vector_service.ensure_collection()
    for start in range(0, len(documents), BATCH_SIZE):
        batch = documents[start : start + BATCH_SIZE]
        ids = [item[0] for item in batch]
        texts = [item[1] for item in batch]
        metadatas = [{**item[2], SYNC_SOURCE_PAYLOAD_KEY: sync_source_marker()} for item in batch]
        embeddings = embedder.embed_documents(texts)
        vector_service.upsert_documents(ids, texts, embeddings, metadatas)
        print(f"Synced {start + len(batch)}/{len(documents)} documents.")
    return len(documents)


def prune_documents(
    documents: list[tuple[str, str, dict[str, Any]]],
    *,
    dry_run: bool = False,
    allow_empty: bool = False,
    force: bool = False,
) -> tuple[int, int]:
    if not documents and not allow_empty:
        raise PruneSafetyError(
            "MySQL returned 0 rows; prune aborted. Use --allow-empty-prune only when an empty source is intentional."
        )

    current_ids = document_point_ids(documents)
    existing_ids = vector_service.list_point_ids({SYNC_SOURCE_PAYLOAD_KEY: sync_source_marker()})
    to_delete = [point_id for point_id in existing_ids if point_id not in current_ids]
    if to_delete and len(to_delete) > len(existing_ids) * PRUNE_DELETE_THRESHOLD and not force:
        raise PruneSafetyError(
            f"prune would delete {len(to_delete)} of {len(existing_ids)} scoped points, exceeding the "
            f"{PRUNE_DELETE_THRESHOLD:.0%} safety threshold; re-run with --force to override"
        )

    if dry_run:
        sample = ", ".join(to_delete[:10]) or "none"
        print(
            f"Prune dry-run: upserted 0, existing {len(existing_ids)}, to_delete {len(to_delete)}; "
            f"would delete: {sample}."
        )
        return len(existing_ids), len(to_delete)

    for start in range(0, len(to_delete), PRUNE_DELETE_BATCH_SIZE):
        vector_service.delete_points(to_delete[start : start + PRUNE_DELETE_BATCH_SIZE])
    return len(existing_ids), len(to_delete)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="with --prune, report deletion candidates without embedding, upserting, or deleting",
    )
    parser.add_argument("--prune", action="store_true", help="delete scoped Qdrant points missing from MySQL")
    parser.add_argument(
        "--allow-empty-prune",
        action="store_true",
        help="allow --prune to delete all scoped points when MySQL returns no rows",
    )
    parser.add_argument("--force", action="store_true", help="override the --prune deletion safety threshold")
    args = parser.parse_args(argv)

    if args.dry_run and not args.prune:
        parser.error("--dry-run requires --prune")

    try:
        if settings.DATASOURCE != "mysql":
            raise RuntimeError("MySQL synchronization requires DATASOURCE=mysql")

        connection = get_db_connection()
        try:
            documents = load_documents(connection)
        finally:
            connection.close()

        if args.prune and args.dry_run:
            prune_documents(documents, dry_run=True, allow_empty=args.allow_empty_prune, force=args.force)
            return 0
        if not documents and not args.prune:
            print("No non-empty documents found; nothing to sync.")
            return 0
        upserted = sync_documents(documents)
        if args.prune:
            existing_count, to_delete_count = prune_documents(
                documents,
                allow_empty=args.allow_empty_prune,
                force=args.force,
            )
            print(
                f"Prune summary: upserted {upserted}, existing {existing_count}, "
                f"to_delete {to_delete_count}, deleted {to_delete_count}."
            )
            return 0
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
    except PruneSafetyError as exc:
        print(f"ERROR: prune aborted: {exc}", file=sys.stderr)
    except Exception as exc:
        print(f"ERROR: synchronization failed ({type(exc).__name__}).", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
