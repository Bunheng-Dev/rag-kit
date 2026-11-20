from functools import lru_cache

from app.core.config import settings
from .datasource.base import DataSourceProvider
from .embedding.base import EmbeddingProvider
from .llm.base import LLMProvider
from .vectordb.base import VectorDBProvider


def _unsupported(kind: str, provider: str, supported: tuple[str, ...]) -> ValueError:
    return ValueError(f"Unsupported {kind} provider '{provider}'. Supported: {', '.join(supported)}")


@lru_cache(maxsize=1)
def get_llm_provider() -> LLMProvider:
    if settings.LLM_PROVIDER == "gemini":
        from .llm.gemini import GeminiLLMProvider

        return GeminiLLMProvider()
    raise _unsupported("LLM", settings.LLM_PROVIDER, ("gemini",))


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    if settings.EMBEDDING_PROVIDER == "gemini":
        from .embedding.gemini import GeminiEmbeddingProvider

        return GeminiEmbeddingProvider()
    raise _unsupported("embedding", settings.EMBEDDING_PROVIDER, ("gemini",))


@lru_cache(maxsize=1)
def get_vector_db_provider() -> VectorDBProvider:
    if settings.VECTOR_DB == "qdrant":
        from .vectordb.qdrant import QdrantVectorDBProvider

        return QdrantVectorDBProvider()
    raise _unsupported("vector database", settings.VECTOR_DB, ("qdrant",))


@lru_cache(maxsize=1)
def get_datasource_provider() -> DataSourceProvider:
    if settings.DATASOURCE == "mysql":
        from .datasource.mysql import MySQLDataSourceProvider

        return MySQLDataSourceProvider()
    raise _unsupported("data source", settings.DATASOURCE, ("mysql",))


# Short aliases keep the factory convenient for scripts and integrations.
get_vector_db = get_vector_db_provider
get_datasource = get_datasource_provider
