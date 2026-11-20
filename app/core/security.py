import secrets
from typing import Optional
from fastapi import Header, HTTPException, status
from app.core.config import settings

def verify_chat_key(x_chat_key: Optional[str] = Header(None, alias="X-Chat-API-Key")) -> str:
    """Validate internal API key for RAG chat access."""
    if not x_chat_key or not secrets.compare_digest(x_chat_key, settings.CHAT_INTERNAL_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Chat-API-Key header"
        )
    return x_chat_key
