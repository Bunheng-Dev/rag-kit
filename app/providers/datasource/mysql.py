import re
from collections.abc import Mapping
from typing import Any

import pymysql
import pymysql.cursors

from app.core.config import settings
from app.services.secret_scanner import SecretScanError, guard_documents, print_scan_summary
from .base import DataSourceProvider


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class MySQLDataSourceProvider(DataSourceProvider):
    def connect(self) -> pymysql.Connection:
        return pymysql.connect(
            host=settings.MYSQL_HOST,
            port=settings.MYSQL_PORT,
            user=settings.MYSQL_USER,
            password=settings.MYSQL_PASSWORD,
            database=settings.MYSQL_DATABASE,
            cursorclass=pymysql.cursors.DictCursor,
            charset="utf8mb4",
            connect_timeout=settings.MYSQL_CONNECT_TIMEOUT,
        )

    @staticmethod
    def _identifier(value: str, label: str) -> str:
        if not _IDENTIFIER_RE.fullmatch(value):
            raise ValueError(f"{label} must be a simple SQL identifier")
        return value

    @classmethod
    def _metadata_columns(cls) -> list[str]:
        raw = settings.MYSQL_DOCUMENT_METADATA_COLUMNS
        columns = [cls._identifier(item.strip(), "MYSQL_DOCUMENT_METADATA_COLUMNS") for item in raw.split(",") if item.strip()]
        blocked = settings.secret_scan_blocked_columns
        skipped = list(dict.fromkeys(column for column in columns if column.lower() in blocked))
        if skipped:
            print(f"Secret scan: skipped blocked MySQL columns: {', '.join(skipped)}")
        return [column for column in columns if column.lower() not in blocked]

    @staticmethod
    def _json_value(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    def read_documents(
        self,
        connection: pymysql.Connection | None = None,
    ) -> list[tuple[str, str, dict[str, Any]]]:
        owned_connection = connection is None
        connection = connection or self.connect()
        try:
            table = self._identifier(settings.MYSQL_DOCUMENT_TABLE, "MYSQL_DOCUMENT_TABLE")
            id_column = self._identifier(settings.MYSQL_DOCUMENT_ID_COLUMN, "MYSQL_DOCUMENT_ID_COLUMN")
            text_column = self._identifier(settings.MYSQL_DOCUMENT_TEXT_COLUMN, "MYSQL_DOCUMENT_TEXT_COLUMN")
            blocked = settings.secret_scan_blocked_columns
            if id_column.lower() in blocked or text_column.lower() in blocked:
                raise ValueError("MYSQL_DOCUMENT_ID_COLUMN and MYSQL_DOCUMENT_TEXT_COLUMN must not be blocked columns")
            metadata_columns = self._metadata_columns()
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
                    column: self._json_value(row.get(column))
                    for column in metadata_columns
                    if row.get(column) is not None
                }
                documents.append((str(document_id), text, metadata))

            try:
                guarded, summaries = guard_documents(documents)
            except SecretScanError as exc:
                print_scan_summary(exc.summaries)
                raise
            print_scan_summary(summaries)
            return guarded
        finally:
            if owned_connection:
                connection.close()
