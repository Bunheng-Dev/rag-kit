import json
import re

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.config import settings
from app.core.limiter import limiter
from app.core.security import verify_chat_key
from app.services.embedder import embedder
from app.services.gemini_service import stream_rag_response
from app.services.vector_service import vector_service


router = APIRouter(prefix=f"{settings.API_V1_PREFIX}/chat", tags=["Chat"])


class ChatRequest(BaseModel):
    message: str


def _slugify(text: str) -> str:
    """Generate a URL-friendly fallback identifier."""
    if not text:
        return ""
    value = re.sub(r"[^\w\s-]", "", str(text).lower()).strip()
    value = re.sub(r"[\s_-]+", "-", value)
    return value.strip("-")


def _select_cards(candidate_cards: list[dict], full_text: str, max_items: int) -> list[dict]:
    """Prefer result cards mentioned by the model, then cap the response."""
    if not candidate_cards:
        return []
    normalized_text = full_text.lower()
    mentioned = [
        card
        for card in candidate_cards
        if card["title"].lower() in normalized_text or card["slug"] in normalized_text
    ]
    return (mentioned or candidate_cards)[:max_items]


def _build_cards(metadatas: list[dict]) -> list[dict]:
    cards: list[dict] = []
    media_base_url = settings.MEDIA_BASE_URL.rstrip("/")

    for metadata in metadatas:
        if not metadata:
            continue
        item_id = metadata.get("id") or metadata.get("item_id")
        if item_id is None:
            continue

        title = str(metadata.get("title") or metadata.get("name") or f"Item {item_id}")
        slug = str(metadata.get("slug") or _slugify(title) or f"item-{item_id}")
        thumbnail = str(metadata.get("thumbnail") or "").strip()
        if thumbnail.startswith("/uploads/") and media_base_url:
            thumbnail = f"{media_base_url}{thumbnail}"

        price = float(metadata.get("price") or 0.0)
        raw_discount = metadata.get("discount_price")
        discount = float(raw_discount) if raw_discount is not None else 0.0
        discount_value = round(discount, 2) if discount > 0.0 and (price == 0.0 or discount < price) else None

        cards.append(
            {
                "id": item_id,
                "title": title,
                "slug": slug,
                "thumbnail": thumbnail,
                "price": price,
                "discount_price": discount_value,
                "category_type": str(metadata.get("category_type") or metadata.get("entity_type") or "item"),
                "url": settings.CARD_URL_TEMPLATE.format(
                    base_url=media_base_url,
                    category_type=metadata.get("category_type") or metadata.get("entity_type") or "item",
                    slug=slug,
                    id=item_id,
                ),
            }
        )
    return cards


@router.post("/stream", dependencies=[Depends(verify_chat_key)])
@limiter.limit("10/minute")
async def chat_stream(request: Request, payload: ChatRequest):
    if not getattr(request.app.state, "vector_store_ready", False):
        raise HTTPException(
            status_code=503,
            detail="Vector database is unavailable; check /health and the service logs.",
        )

    query_vector = embedder.embed_text(payload.message)
    search_results = vector_service.search_similar(
        query_vector=query_vector,
        n_results=settings.RETRIEVAL_TOP_K,
    )
    retrieved_chunks = search_results.get("documents", [[]])[0]
    retrieved_metas = search_results.get("metadatas", [[]])[0] if search_results.get("metadatas") else []
    cards = _build_cards(retrieved_metas) if settings.ENABLE_CARDS else []

    async def event_generator():
        full_text_chunks: list[str] = []
        try:
            async for chunk in stream_rag_response(payload.message, retrieved_chunks):
                full_text_chunks.append(chunk)
                yield f"data: {chunk}\n\n"

            if cards:
                final_cards = _select_cards(
                    cards,
                    full_text="".join(full_text_chunks),
                    max_items=settings.CARD_MAX_ITEMS,
                )
                yield f"data: [CARDS]{json.dumps(final_cards)}\n\n"
        except Exception as exc:
            yield f"data: [Error: {exc}]\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
