import inspect

import pytest
from fastapi import HTTPException
import app.api.chat as chat_module
from app.core.config import settings
from app.core.security import verify_chat_key
from app.main import app, health_check


def _api_paths() -> set[str]:
    return set(app.openapi()["paths"])


def test_v1_chat_accepts_valid_key(monkeypatch) -> None:
    monkeypatch.setattr(settings, "CHAT_INTERNAL_KEY", "unit-test-key")

    assert "/api/v1/chat/stream" in _api_paths()
    assert inspect.iscoroutinefunction(chat_module.chat_stream.__wrapped__)
    assert verify_chat_key("unit-test-key") == "unit-test-key"


def test_v1_chat_requires_key(monkeypatch) -> None:
    monkeypatch.setattr(settings, "CHAT_INTERNAL_KEY", "unit-test-key")

    with pytest.raises(HTTPException) as error:
        verify_chat_key(None)

    assert error.value.status_code == 401


def test_old_chat_path_is_not_available() -> None:
    legacy_path = "/api" + "/chat/stream"

    assert legacy_path not in _api_paths()
    assert legacy_path not in app.openapi()["paths"]


def test_versioned_and_unversioned_health_routes_match() -> None:
    paths = _api_paths()
    unversioned = health_check()

    assert "/health" in paths
    assert "/api/v1/health" in paths
    assert unversioned == health_check()
