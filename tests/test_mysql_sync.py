from scripts import sync_mysql_to_qdrant as sync_script
from app.providers.vectordb.qdrant import QdrantVectorDBProvider


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


class FakeSyncStore:
    def __init__(
        self,
        existing_ids: list[str],
        *,
        other_source_ids: list[str] | None = None,
        fail_upsert: bool = False,
    ) -> None:
        self.existing_ids = existing_ids
        self.other_source_ids = other_source_ids or []
        self.all_ids = [*existing_ids, *self.other_source_ids]
        self.fail_upsert = fail_upsert
        self.deleted: list[list[str]] = []
        self.upserts: list[tuple[list[str], list[dict]]] = []
        self.filters: list[dict] = []

    @staticmethod
    def point_id(document_id: str | int) -> str:
        return f"point-{document_id}"

    def ensure_collection(self) -> None:
        return None

    def upsert_documents(self, ids, documents, embeddings, metadatas=None) -> None:
        if self.fail_upsert:
            raise RuntimeError("fake upsert failed")
        self.upserts.append((list(ids), list(metadatas or [])))

    def list_point_ids(self, where=None) -> list[str]:
        self.filters.append(dict(where or {}))
        if where == {"sync_source": "mysql:knowledge"}:
            return list(self.existing_ids)
        return list(self.all_ids)

    def delete_points(self, point_ids) -> None:
        self.deleted.append(list(point_ids))


class FakeConnectionForMain:
    def close(self) -> None:
        return None


def configure_sync_main(monkeypatch, documents, store: FakeSyncStore) -> None:
    monkeypatch.setattr(sync_script.settings, "DATASOURCE", "mysql")
    monkeypatch.setattr(sync_script.settings, "MYSQL_DOCUMENT_TABLE", "knowledge")
    monkeypatch.setattr(sync_script.settings, "QDRANT_COLLECTION", "rag_documents")
    monkeypatch.setattr(sync_script, "get_db_connection", lambda: FakeConnectionForMain())
    monkeypatch.setattr(sync_script, "load_documents", lambda connection: documents)
    monkeypatch.setattr(sync_script, "vector_service", store)
    monkeypatch.setattr(sync_script.embedder, "embed_documents", lambda texts: [[0.0] for _ in texts])


def test_prune_deletes_only_missing_scoped_ids(monkeypatch) -> None:
    store = FakeSyncStore(["point-a", "point-b", "point-stale"])
    documents = [("a", "A", {"kind": "faq"}), ("b", "B", {"kind": "faq"})]
    configure_sync_main(monkeypatch, documents, store)

    assert sync_script.main(["--prune"]) == 0
    assert store.deleted == [["point-stale"]]
    assert store.filters == [{"sync_source": "mysql:knowledge"}]
    assert store.upserts[0][1] == [
        {"kind": "faq", "sync_source": "mysql:knowledge"},
        {"kind": "faq", "sync_source": "mysql:knowledge"},
    ]


def test_prune_dry_run_deletes_nothing(monkeypatch) -> None:
    store = FakeSyncStore(["point-a", "point-stale"])
    documents = [("a", "A", {})]
    configure_sync_main(monkeypatch, documents, store)
    monkeypatch.setattr(sync_script.embedder, "embed_documents", lambda texts: (_ for _ in ()).throw(AssertionError()))

    assert sync_script.main(["--prune", "--dry-run"]) == 0
    assert store.deleted == []
    assert store.upserts == []


def test_empty_mysql_aborts_without_deleting_and_allow_empty_overrides(monkeypatch) -> None:
    store = FakeSyncStore(["point-stale"])
    configure_sync_main(monkeypatch, [], store)

    assert sync_script.main(["--prune"]) == 1
    assert store.deleted == []
    assert sync_script.main(["--prune", "--allow-empty-prune", "--force"]) == 0
    assert store.deleted == [["point-stale"]]


def test_prune_threshold_requires_force(monkeypatch) -> None:
    store = FakeSyncStore(["point-a", "point-stale-1", "point-stale-2"])
    documents = [("a", "A", {})]
    configure_sync_main(monkeypatch, documents, store)

    assert sync_script.main(["--prune"]) == 1
    assert store.deleted == []
    assert sync_script.main(["--prune", "--force"]) == 0
    assert store.deleted == [["point-stale-1", "point-stale-2"]]


def test_points_from_other_sources_are_not_deleted(monkeypatch) -> None:
    store = FakeSyncStore(["point-a"], other_source_ids=["json-point-stale"])
    documents = [("a", "A", {})]
    configure_sync_main(monkeypatch, documents, store)

    assert sync_script.main(["--prune"]) == 0
    assert store.deleted == []
    assert store.filters == [{"sync_source": "mysql:knowledge"}]


def test_prune_is_skipped_if_upsert_fails(monkeypatch) -> None:
    store = FakeSyncStore(["point-stale"], fail_upsert=True)
    configure_sync_main(monkeypatch, [("a", "A", {})], store)

    assert sync_script.main(["--prune"]) == 1
    assert store.filters == []
    assert store.deleted == []


def test_qdrant_list_point_ids_handles_pagination() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.offsets = []

        def scroll(self, **kwargs):
            self.offsets.append(kwargs["offset"])
            assert kwargs["with_payload"] is False
            assert kwargs["with_vectors"] is False
            if kwargs["offset"] is None:
                return ([type("Point", (), {"id": "first"})()], "next")
            return ([type("Point", (), {"id": "second"})()], None)

    provider = object.__new__(QdrantVectorDBProvider)
    provider.client = FakeClient()
    provider.collection_name = "rag_documents"

    assert provider.list_point_ids({"sync_source": "mysql:knowledge"}) == ["first", "second"]
    assert provider.client.offsets == [None, "next"]
