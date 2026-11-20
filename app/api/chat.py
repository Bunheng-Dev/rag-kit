import json
import re
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.core.config import settings
from app.core.limiter import limiter
from app.core.security import verify_chat_key
from app.services.embedder import embedder
from app.services.vector_service import vector_service
from app.services.gemini_service import stream_rag_response

router = APIRouter(prefix="/api/chat", tags=["Chat"])

class ChatRequest(BaseModel):
    message: str

STAY_KEYWORDS = {
    "hotel", "hotels", "villa", "villas", "resort", "resorts",
    "stay", "stays", "accommodation", "accommodations", "hostel", "hostels",
    "room", "rooms", "lodge", "lodging",
    "សណ្ឋាគារ", "វីឡា", "កន្លែងស្នាក់នៅ", "ផ្ទះសំណាក់", "បន្ទប់"
}

HOTEL_KEYWORDS = {"hotel", "hotels", "resort", "resorts", "boutique", "សណ្ឋាគារ"}
VILLA_KEYWORDS = {"villa", "villas", "វីឡា", "ផ្ទះវីឡា"}

def _slugify(text: str) -> str:
    """Generate a clean URL-friendly slug fallback."""
    if not text:
        return ""
    s = re.sub(r"[^\w\s-]", "", str(text).lower()).strip()
    s = re.sub(r"[\s_-]+", "-", s)
    return s.strip("-")

def _is_stay_query(query: str, top_entity_type: str | None = None) -> bool:
    """Determine if query is asking about stays or if top vector match is a stay."""
    if top_entity_type in ["hotel", "villa"]:
        return True
    q_lower = query.lower()
    return any(kw in q_lower for kw in STAY_KEYWORDS)

def _select_balanced_cards(candidate_cards: list[dict], full_text: str = "", max_cards: int = 5) -> list[dict]:
    """Select and order cards matching LLM text, balancing hotels and villas fairly."""
    full_text_lower = full_text.lower() if full_text else ""

    # Identify candidate cards mentioned in the LLM text
    mentioned = []
    for card in candidate_cards:
        title_lower = card["title"].lower().strip()
        if full_text_lower and (title_lower in full_text_lower or card["slug"] in full_text_lower):
            mentioned.append(card)

    pool = mentioned if mentioned else candidate_cards

    hotels = [c for c in pool if c.get("category_type") == "hotels"]
    villas = [c for c in pool if c.get("category_type") == "private-villas"]

    def text_pos(c):
        pos = full_text_lower.find(c["title"].lower())
        return pos if pos != -1 else 999999

    hotels.sort(key=text_pos)
    villas.sort(key=text_pos)

    if hotels and villas:
        # Both categories present: allocate balanced slots so neither category starves
        h_target = min(len(hotels), max(1, max_cards // 2))
        v_target = min(len(villas), max(1, max_cards // 2))
        remaining = max_cards - (h_target + v_target)
        while remaining > 0:
            if len(hotels) > h_target:
                h_target += 1
                remaining -= 1
            elif len(villas) > v_target:
                v_target += 1
                remaining -= 1
            else:
                break
        selected = hotels[:h_target] + villas[:v_target]
        selected.sort(key=text_pos)
        return selected[:max_cards]
    else:
        pool.sort(key=text_pos)
        return pool[:max_cards]

@router.post("/stream", dependencies=[Depends(verify_chat_key)])
@limiter.limit("10/minute")
async def chat_stream(request: Request, payload: ChatRequest):
    # 1. Embed query
    query_vector = embedder.embed_text(payload.message)

    # 2. Check if the user query is asking about stays
    is_stay = _is_stay_query(payload.message)

    if is_stay:
        q_lower = payload.message.lower()
        asks_hotel = any(kw in q_lower for kw in HOTEL_KEYWORDS)
        asks_villa = any(kw in q_lower for kw in VILLA_KEYWORDS)

        if asks_hotel and asks_villa:
            # Query asks for both hotels and villas: retrieve at most 5 properties combined
            hotel_pos = min((q_lower.find(kw) for kw in HOTEL_KEYWORDS if kw in q_lower), default=999)
            villa_pos = min((q_lower.find(kw) for kw in VILLA_KEYWORDS if kw in q_lower), default=999)

            if villa_pos < hotel_pos:
                # Villas mentioned first: retrieve 3 villas and 2 hotels (5 total)
                search_first = vector_service.search_similar(query_vector=query_vector, n_results=3, where={"entity_type": "villa"})
                search_second = vector_service.search_similar(query_vector=query_vector, n_results=2, where={"entity_type": "hotel"})
            else:
                # Hotels mentioned first: retrieve 3 hotels and 2 villas (5 total)
                search_first = vector_service.search_similar(query_vector=query_vector, n_results=3, where={"entity_type": "hotel"})
                search_second = vector_service.search_similar(query_vector=query_vector, n_results=2, where={"entity_type": "villa"})

            f_docs = search_first.get("documents", [[]])[0]
            f_metas = search_first.get("metadatas", [[]])[0]
            s_docs = search_second.get("documents", [[]])[0]
            s_metas = search_second.get("metadatas", [[]])[0]

            retrieved_chunks = []
            retrieved_metas = []
            for i in range(max(len(f_docs), len(s_docs))):
                if i < len(f_docs):
                    retrieved_chunks.append(f_docs[i])
                    retrieved_metas.append(f_metas[i])
                if i < len(s_docs):
                    retrieved_chunks.append(s_docs[i])
                    retrieved_metas.append(s_metas[i])
            retrieved_chunks = retrieved_chunks[:5]
            retrieved_metas = retrieved_metas[:5]
        elif asks_villa and not asks_hotel:
            # User specifically asked for villas only: retrieve at most 5 villas
            search_results = vector_service.search_similar(query_vector=query_vector, n_results=5, where={"entity_type": "villa"})
            retrieved_chunks = search_results.get("documents", [[]])[0]
            retrieved_metas = search_results.get("metadatas", [[]])[0] if search_results.get("metadatas") else []
        elif asks_hotel and not asks_villa:
            # User specifically asked for hotels only: retrieve at most 5 hotels
            search_results = vector_service.search_similar(query_vector=query_vector, n_results=5, where={"entity_type": "hotel"})
            retrieved_chunks = search_results.get("documents", [[]])[0]
            retrieved_metas = search_results.get("metadatas", [[]])[0] if search_results.get("metadatas") else []
        else:
            # General stay inquiry: retrieve balanced 3 hotels and 2 villas (5 total)
            search_hotels = vector_service.search_similar(query_vector=query_vector, n_results=3, where={"entity_type": "hotel"})
            search_villas = vector_service.search_similar(query_vector=query_vector, n_results=2, where={"entity_type": "villa"})
            h_docs = search_hotels.get("documents", [[]])[0]
            h_metas = search_hotels.get("metadatas", [[]])[0]
            v_docs = search_villas.get("documents", [[]])[0]
            v_metas = search_villas.get("metadatas", [[]])[0]
            retrieved_chunks = []
            retrieved_metas = []
            for i in range(max(len(h_docs), len(v_docs))):
                if i < len(h_docs):
                    retrieved_chunks.append(h_docs[i])
                    retrieved_metas.append(h_metas[i])
                if i < len(v_docs):
                    retrieved_chunks.append(v_docs[i])
                    retrieved_metas.append(v_metas[i])
            retrieved_chunks = retrieved_chunks[:5]
            retrieved_metas = retrieved_metas[:5]
    else:
        # General query: retrieve top 6 chunks across all categories
        search_results = vector_service.search_similar(query_vector=query_vector, n_results=6)
        top_metas = search_results.get("metadatas", [[]])[0]
        top_type = top_metas[0].get("entity_type") if top_metas else None
        if top_type in ["hotel", "villa"]:
            # Query matched a specific property directly; upgrade to stay search
            search_results = vector_service.search_similar(
                query_vector=query_vector,
                n_results=5,
                where={"entity_type": top_type}
            )
            is_stay = True
        retrieved_chunks = search_results.get("documents", [[]])[0]
        retrieved_metas = search_results.get("metadatas", [[]])[0] if search_results.get("metadatas") else []

    # 3. Extract candidate Stays cards strictly scoped to fa_hotels and fa_private_villa
    cards = []
    if is_stay:
        seen_stay_keys = set()

        for meta in retrieved_metas:
            if not meta:
                continue
            entity_type = meta.get("entity_type")
            if entity_type not in ["hotel", "villa"]:
                continue

            item_id = int(meta.get("id") or meta.get("item_id") or 0)
            stay_key = f"{entity_type}_{item_id}"
            if not item_id or stay_key in seen_stay_keys:
                continue
            seen_stay_keys.add(stay_key)

            is_hotel = (entity_type == "hotel")
            category_type = "hotels" if is_hotel else "private-villas"
            title = meta.get("title") or meta.get("name") or ("Hotel" if is_hotel else "Private Villa")

            raw_slug = str(meta.get("slug") or "").strip()
            slug = raw_slug or _slugify(title) or f"stay-{item_id}"

            thumbnail = str(meta.get("thumbnail") or "").strip()
            upload_base = settings.MEDIA_BASE_URL.rstrip("/")
            if thumbnail.startswith("/uploads/") and upload_base:
                thumbnail = f"{upload_base}{thumbnail}"

            price = float(meta.get("price") or 0.0)
            raw_discount = meta.get("discount_price")
            discount_price = float(raw_discount) if raw_discount is not None else 0.0

            # If discount_price is missing or 0 or equal/greater than price, fallback to null
            if discount_price > 0.0 and (price == 0.0 or discount_price < price):
                discount_price_val = round(discount_price, 2)
            else:
                discount_price_val = None

            card_url = settings.CARD_URL_TEMPLATE.format(
                base_url=settings.MEDIA_BASE_URL.rstrip("/"),
                category_type=category_type,
                slug=slug,
                id=item_id,
            )

            card = {
                "id": item_id,
                "title": title,
                "slug": slug,
                "thumbnail": thumbnail,
                "price": price,
                "discount_price": discount_price_val,
                "category_type": category_type,
                "url": card_url,
            }
            cards.append(card)

    # 4. Stream protocol: LLM text tokens first, structured cards after generation completes, [DONE] at end
    async def event_generator():
        full_text_chunks = []
        try:
            # 1. Transmit all LLM text tokens first
            async for chunk in stream_rag_response(payload.message, retrieved_chunks):
                full_text_chunks.append(chunk)
                yield f"data: {chunk}\n\n"

            # 2. Align emitted cards with the model's textual recommendations
            if cards:
                final_cards = _select_balanced_cards(cards, full_text="".join(full_text_chunks), max_cards=5)
                yield f"data: [CARDS]{json.dumps(final_cards)}\n\n"
        except Exception as e:
            yield f"data: [Error: {str(e)}]\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
