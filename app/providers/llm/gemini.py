from collections.abc import AsyncIterator, Sequence
from pathlib import Path

from google import genai
from google.genai import errors

from app.core.config import settings
from .base import LLMProvider


DEFAULT_PROMPT = """You are a helpful retrieval-augmented assistant.
Answer the user's question accurately and professionally using only the provided context.
Respond in exactly one language matching the requested target language. If the answer is not
present in the context, say so clearly in that language and do not invent facts.

Context:
{context}
"""


def _load_prompt_template() -> str:
    if settings.PROMPT_TEMPLATE:
        return settings.PROMPT_TEMPLATE
    path: Path = settings.prompt_template_file
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return DEFAULT_PROMPT


class GeminiLLMProvider(LLMProvider):
    def __init__(self) -> None:
        if not settings.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is required for the Gemini LLM provider")
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.prompt_template = _load_prompt_template()

    @staticmethod
    def _response_language() -> str:
        configured = settings.RESPONSE_LANGUAGE.strip()
        return "the same language as the user's question" if not configured or configured.lower() == "auto" else configured

    async def stream_response(
        self,
        user_query: str,
        retrieved_chunks: Sequence[str],
    ) -> AsyncIterator[str]:
        context = "\n---\n".join(retrieved_chunks) if retrieved_chunks else "No specific context available."
        prompt = self.prompt_template.format(context=context)
        prompt += (
            f"\n\nTarget Language: {self._response_language()}."
            " Respond strictly in that language."
            f"\nUser Question: {user_query}\nAnswer:"
        )

        try:
            response = await self.client.aio.models.generate_content_stream(
                model=settings.LLM_MODEL,
                contents=prompt,
            )
            async for chunk in response:
                try:
                    if chunk.text:
                        yield chunk.text
                except (AttributeError, ValueError):
                    continue
        except errors.APIError as exc:
            print(f"[LLM API error]: {exc}", flush=True)
            yield "The language model is temporarily unavailable. Please try again shortly."
        except Exception as exc:
            print(f"[LLM error]: {exc}", flush=True)
            yield "The language model encountered an unexpected error."
