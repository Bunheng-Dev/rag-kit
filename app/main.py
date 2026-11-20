from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from app.core.config import settings
from app.core.limiter import limiter
from app.api.chat import router as chat_router
from app.services.vector_service import vector_service

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.on_event("startup")
def ensure_vector_store() -> None:
    vector_service.ensure_collection()


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "service": settings.APP_NAME,
        "vector_database": settings.VECTOR_DB,
        "collection": settings.QDRANT_COLLECTION,
        "embedding_dimension": settings.EMBEDDING_DIM,
    }
