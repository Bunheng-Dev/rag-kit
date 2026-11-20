from app.main import app, ensure_vector_store, health_check


def test_health_reports_degraded_before_vector_startup() -> None:
    app.state.vector_store_ready = False
    payload = health_check()

    assert payload["status"] == "degraded"
    assert payload["vector_database_status"] == "unavailable"


def test_qdrant_failure_is_captured_as_degraded(monkeypatch) -> None:
    def fail_to_connect() -> None:
        raise ConnectionError("connection reset")

    monkeypatch.setattr("app.main.vector_service.ensure_collection", fail_to_connect)
    ensure_vector_store(app)

    assert app.state.vector_store_ready is False
    assert health_check()["status"] == "degraded"
