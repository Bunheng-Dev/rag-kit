import asyncio
import logging
import sys
import time

import httpx
import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import Settings
from app.providers.llm.gemini import GeminiLLMProvider


def test_default_port_and_provider_normalization() -> None:
    config = Settings(_env_file=None, APP_PORT=8000, DATASOURCE="file-provider")

    assert config.APP_PORT == 8000
    assert config.DATASOURCE == "file_provider"
    assert config.CORS_ALLOW_ORIGINS == []
    assert config.cors_origins == []


def test_cors_origins_parse_comma_separated_values() -> None:
    config = Settings(_env_file=None, CORS_ALLOW_ORIGINS="http://localhost:3000, http://localhost:5173")

    assert config.CORS_ALLOW_ORIGINS == ["http://localhost:3000", "http://localhost:5173"]


def test_empty_cors_origins_parse_to_empty_list() -> None:
    config = Settings(_env_file=None, CORS_ALLOW_ORIGINS="")

    assert config.CORS_ALLOW_ORIGINS == []


def test_wildcard_cors_logs_warning_and_disables_credentials(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="app.core.config"):
        config = Settings(_env_file=None, CORS_ALLOW_ORIGINS="http://localhost:3000,*")

    assert "CORS_ALLOW_ORIGINS includes '*'" in caplog.text
    assert config.cors_allow_credentials is False


def make_cors_test_app(config: Settings) -> FastAPI:
    application = FastAPI()
    application.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=config.cors_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.get("/")
    def root() -> dict[str, bool]:
        return {"ok": True}

    return application


class _NoSocketSelector:
    def __init__(self, selector) -> None:
        self._selector = selector

    def select(self, timeout=None):
        if not self._selector.get_map():
            if timeout:
                time.sleep(timeout)
            return []
        return self._selector.select(timeout)

    def __getattr__(self, name):
        return getattr(self._selector, name)


class _NoSocketEventLoop(asyncio.SelectorEventLoop):
    def _make_self_pipe(self):
        self._ssock = None
        self._csock = None
        self._internal_fds = 0

    def _close_self_pipe(self):
        self._ssock = None
        self._csock = None

    def __init__(self):
        super().__init__()
        self._selector = _NoSocketSelector(self._selector)


@pytest.fixture
def anyio_backend():
    if sys.platform == "win32":
        return "asyncio", {"loop_factory": _NoSocketEventLoop}
    return "asyncio"


@pytest.mark.anyio
async def test_cors_response_allows_configured_origin() -> None:
    config = Settings(_env_file=None, CORS_ALLOW_ORIGINS="http://localhost:3000")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=make_cors_test_app(config)),
        base_url="http://testserver",
    ) as client:
        response = await client.get("/", headers={"Origin": "http://localhost:3000"})

    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


@pytest.mark.anyio
async def test_cors_response_rejects_unconfigured_origin() -> None:
    config = Settings(_env_file=None, CORS_ALLOW_ORIGINS="http://localhost:3000")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=make_cors_test_app(config)),
        base_url="http://testserver",
    ) as client:
        response = await client.get("/", headers={"Origin": "http://localhost:5173"})

    assert "access-control-allow-origin" not in response.headers


def test_embedding_dimension_must_be_positive() -> None:
    try:
        Settings(_env_file=None, EMBEDDING_DIM=0)
    except ValueError as exc:
        assert "EMBEDDING_DIM" in str(exc)
    else:
        raise AssertionError("zero embedding dimension should be rejected")


def test_response_language_auto_follows_the_user(monkeypatch) -> None:
    monkeypatch.setattr("app.providers.llm.gemini.settings.RESPONSE_LANGUAGE", "auto")

    assert GeminiLLMProvider._response_language() == "the same language as the user's question"
