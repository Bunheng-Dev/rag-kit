from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence


class LLMProvider(ABC):
    @abstractmethod
    async def stream_response(
        self,
        user_query: str,
        retrieved_chunks: Sequence[str],
    ) -> AsyncIterator[str]:
        """Stream an answer grounded in the retrieved chunks."""
        yield ""
