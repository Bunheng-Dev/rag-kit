from scripts import sync_mysql_to_qdrant as sync_script


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows
        self.query = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, query):
        self.query = query

    def fetchall(self):
        return self.rows


class FakeConnection:
    def __init__(self, rows):
        self.cursor_instance = FakeCursor(rows)

    def cursor(self):
        return self.cursor_instance


def test_mysql_rows_become_documents_without_real_database(monkeypatch) -> None:
    monkeypatch.setattr(sync_script.settings, "MYSQL_DOCUMENT_TABLE", "knowledge")
    monkeypatch.setattr(sync_script.settings, "MYSQL_DOCUMENT_ID_COLUMN", "doc_id")
    monkeypatch.setattr(sync_script.settings, "MYSQL_DOCUMENT_TEXT_COLUMN", "body")
    monkeypatch.setattr(sync_script.settings, "MYSQL_DOCUMENT_METADATA_COLUMNS", "kind,score")

    connection = FakeConnection(
        [
            {"doc_id": "a", "body": "First document", "kind": "faq", "score": 3},
            {"doc_id": "b", "body": "", "kind": "ignored", "score": 0},
        ]
    )

    documents = sync_script.load_documents(connection)

    assert documents == [("a", "First document", {"kind": "faq", "score": 3})]
    assert connection.cursor_instance.query == "SELECT doc_id, body, kind, score FROM knowledge;"
