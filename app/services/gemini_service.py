"""Backward-compatible LLM service facade."""

from collections.abc import AsyncIterator, Sequence

from app.providers.factory import get_llm_provider


async def stream_rag_response(user_query: str, retrieved_chunks: Sequence[str]) -> AsyncIterator[str]:
    async for chunk in get_llm_provider().stream_response(user_query, retrieved_chunks):
        yield chunk
