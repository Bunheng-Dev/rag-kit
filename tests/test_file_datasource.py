import json

from app.providers.datasource.file import FileDataSourceProvider


def test_file_datasource_reads_normalized_documents(tmp_path) -> None:
    dataset = tmp_path / "sample.json"
    dataset.write_text(
        json.dumps([{"id": "one", "text": "hello", "metadata": {"kind": "test"}}]),
        encoding="utf-8",
    )

    documents = FileDataSourceProvider(dataset).read_documents()

    assert documents == [{"id": "one", "text": "hello", "metadata": {"kind": "test"}}]
