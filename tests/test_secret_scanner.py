import pytest

from app.core.config import settings
from app.providers.datasource.mysql import MySQLDataSourceProvider
from app.services.secret_scanner import guard_documents, redact, scan_text
from scripts import ingest_sample, sync_mysql_to_qdrant as sync_script


@pytest.mark.parametrize(
    ("value", "expected_type"),
    [
        ("AKIA1234567890ABCDEF", "aws_access_key"),
        ("AIzaSyA12345678901234567890", "google_api_key"),
        ("sk-proj-" + "a" * 24, "openai_api_key"),
        ("ghp_" + "a" * 36, "github_token"),
        ("xoxb-" + "a" * 20, "slack_token"),
        ("eyJ" + "a" * 12 + "." + "b" * 12 + "." + "c" * 12, "jwt"),
        ("-----BEGIN PRIVATE KEY-----x-----END PRIVATE KEY-----", "pem_private_key"),
        ("mysql://user:pass@example.com/db", "connection_string"),
        ("postgres://user:pass@example.com/db", "connection_string"),
        ("mongodb://user:pass@example.com/db", "connection_string"),
        ("redis://user:pass@example.com/0", "connection_string"),
        ("password=supersecret", "credential_assignment"),
        ("Bearer " + "A" * 20, "bearer_token"),
        ("Ab9_xY7zQ2mN8pL4vC6dE1fG3hJ5kR7sT9uW", "high_entropy_string"),
    ],
)
def test_secret_types_are_detected(value: str, expected_type: str) -> None:
    assert expected_type in {finding["type"] for finding in scan_text(value)}


def test_obvious_placeholders_are_ignored() -> None:
    for value in ("your-api-key", "replace-with-secret", "example-token", "xxxx", "<key>"):
        assert scan_text(value) == []


def test_redacted_placeholders_are_safe_to_rescan() -> None:
    result = redact("password=<fake> token=<fake>")
    redacted_secret = redact("password=supersecret token=anothersecret")

    assert scan_text(result) == []
    assert scan_text(redacted_secret) == []


def test_redact_removes_detected_values(monkeypatch) -> None:
    monkeypatch.setattr(settings, "SECRET_SCAN_MODE", "redact")
    secret = "password=supersecret"

    guarded, summaries = guard_documents([("source-1", secret, {"note": secret})])

    assert summaries
    assert secret not in guarded[0][1]
    assert secret not in guarded[0][2]["note"]
    assert "[REDACTED:credential_assignment]" in guarded[0][1]


def test_block_mode_aborts_before_sync_and_does_not_report_values(monkeypatch, capsys) -> None:
    monkeypatch.setattr(settings, "SECRET_SCAN_MODE", "block")
    monkeypatch.setattr(settings, "MYSQL_DOCUMENT_TABLE", "documents")
    monkeypatch.setattr(settings, "MYSQL_DOCUMENT_ID_COLUMN", "id")
    monkeypatch.setattr(settings, "MYSQL_DOCUMENT_TEXT_COLUMN", "text")
    monkeypatch.setattr(settings, "MYSQL_DOCUMENT_METADATA_COLUMNS", "category")
    secret = "password=supersecret"
    connection = _FakeConnection([{"id": "1", "text": secret, "category": "sample"}])
    upserted = []
    monkeypatch.setattr(sync_script, "get_db_connection", lambda: connection)
    monkeypatch.setattr(sync_script, "sync_documents", lambda documents: upserted.extend(documents))

    assert sync_script.main([]) == 1

    output = capsys.readouterr()
    assert not upserted
    assert secret not in output.out + output.err


def test_blocked_mysql_columns_are_not_selected_or_ingested(monkeypatch, capsys) -> None:
    monkeypatch.setattr(settings, "SECRET_SCAN_MODE", "block")
    monkeypatch.setattr(settings, "MYSQL_DOCUMENT_TABLE", "documents")
    monkeypatch.setattr(settings, "MYSQL_DOCUMENT_ID_COLUMN", "id")
    monkeypatch.setattr(settings, "MYSQL_DOCUMENT_TEXT_COLUMN", "text")
    monkeypatch.setattr(settings, "MYSQL_DOCUMENT_METADATA_COLUMNS", "category,password")
    secret = "supersecret-password"

    documents = MySQLDataSourceProvider().read_documents(
        _FakeConnection([{"id": "1", "text": "safe", "category": "sample", "password": secret}])
    )

    assert documents == [("1", "safe", {"category": "sample"})]
    assert "password" not in _FakeConnection.last_query
    assert secret not in capsys.readouterr().out


def test_bundled_sample_passes_normal_ingestion(monkeypatch) -> None:
    monkeypatch.setattr(settings, "SECRET_SCAN_MODE", "block")
    monkeypatch.setattr(settings, "DATASOURCE", "file")
    monkeypatch.setattr(settings, "SAMPLE_DATA_PATH", "data/sample.json")
    monkeypatch.setattr(ingest_sample.vector_service, "ensure_collection", lambda: None)
    monkeypatch.setattr(
        ingest_sample.embedder,
        "embed_documents",
        lambda texts: [[0.0] * settings.EMBEDDING_DIM for _ in texts],
    )
    upserted: list[str] = []
    monkeypatch.setattr(
        ingest_sample.vector_service,
        "upsert_documents",
        lambda ids, texts, embeddings, metadatas: upserted.extend(ids),
    )

    ingest_sample.main()

    assert upserted == ["product-1", "service-1", "faq-1"]


def test_redact_mode_upserts_only_redacted_text_and_metadata(monkeypatch, capsys) -> None:
    monkeypatch.setattr(settings, "DATASOURCE", "mysql")
    monkeypatch.setattr(settings, "SECRET_SCAN_MODE", "redact")
    monkeypatch.setattr(settings, "MYSQL_DOCUMENT_TABLE", "documents")
    monkeypatch.setattr(settings, "MYSQL_DOCUMENT_ID_COLUMN", "id")
    monkeypatch.setattr(settings, "MYSQL_DOCUMENT_TEXT_COLUMN", "text")
    monkeypatch.setattr(settings, "MYSQL_DOCUMENT_METADATA_COLUMNS", "note")
    secret = "password=supersecret"
    connection = _FakeConnection([{"id": "1", "text": secret, "note": secret}])
    captured: dict[str, object] = {}
    monkeypatch.setattr(sync_script, "get_db_connection", lambda: connection)
    monkeypatch.setattr(sync_script.vector_service, "ensure_collection", lambda: None)
    monkeypatch.setattr(
        sync_script.embedder,
        "embed_documents",
        lambda texts: [[0.0] * settings.EMBEDDING_DIM for _ in texts],
    )
    monkeypatch.setattr(
        sync_script.vector_service,
        "upsert_documents",
        lambda ids, texts, embeddings, metadatas: captured.update(texts=list(texts), metadatas=list(metadatas)),
    )

    assert sync_script.main([]) == 0

    output = capsys.readouterr()
    assert secret not in output.out + output.err
    assert secret not in captured["texts"]
    assert secret not in captured["metadatas"]


class _FakeCursor:
    def __init__(self, rows):
        self.rows = rows
        self.query = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, query):
        self.query = query
        _FakeConnection.last_query = query

    def fetchall(self):
        return self.rows


class _FakeConnection:
    last_query = ""

    def __init__(self, rows):
        self.cursor_instance = _FakeCursor(rows)

    def cursor(self):
        return self.cursor_instance

    def close(self):
        return None
