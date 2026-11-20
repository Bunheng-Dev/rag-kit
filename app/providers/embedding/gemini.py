from collections.abc import Sequence

from google import genai

from app.core.config import settings
from .base import EmbeddingProvider


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Gemini embeddings keep the default install free from Torch dependencies."""

    def __init__(self) -> None:
        if not settings.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is required for the Gemini embedding provider")
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)

    @property
    def dimension(self) -> int:
        return settings.EMBEDDING_DIM

    def _embed(self, contents: str | Sequence[str], task_type: str) -> list[list[float]]:
        response = self.client.models.embed_content(
            model=settings.EMBEDDING_MODEL,
            contents=list(contents) if not isinstance(contents, str) else contents,
            config={
                "task_type": task_type,
                "output_dimensionality": settings.EMBEDDING_DIM,
            },
        )
        embeddings = getattr(response, "embeddings", None) or []
        values = [list(item.values or []) for item in embeddings]
        if not values:
            raise RuntimeError("Gemini returned no embeddings")
        for vector in values:
            if len(vector) != settings.EMBEDDING_DIM:
                raise ValueError(
                    f"{settings.EMBEDDING_MODEL} returned {len(vector)} dimensions; "
                    f"expected {settings.EMBEDDING_DIM}"
                )
        return values

    def embed_text(self, text: str) -> list[float]:
        return self._embed(text, "RETRIEVAL_QUERY")[0]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._embed(texts, "RETRIEVAL_DOCUMENT")
