from app.core.config import Settings
from app.providers.llm.gemini import GeminiLLMProvider


def test_default_port_and_provider_normalization() -> None:
    config = Settings(_env_file=None, APP_PORT=8000, DATASOURCE="file-provider")

    assert config.APP_PORT == 8000
    assert config.DATASOURCE == "file_provider"


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
