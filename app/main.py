from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from app.core.config import settings
from app.core.limiter import limiter
from app.api.chat import router as chat_router
from app.services.vector_service import vector_service

logger = logging.getLogger(__name__)


def ensure_vector_store(application: FastAPI) -> None:
    try:
        vector_service.ensure_collection()
    except Exception as exc:
        application.state.vector_store_ready = False
        application.state.vector_store_error = str(exc)
        logger.exception(
            "Qdrant is unavailable; the API will start in degraded mode. "
            "Check QDRANT_URL, credentials, network access, and whether the cluster is paused."
        )
    else:
        application.state.vector_store_ready = True
        application.state.vector_store_error = None
        logger.info("Qdrant collection '%s' is ready", settings.QDRANT_COLLECTION)


@asynccontextmanager
async def lifespan(application: FastAPI):
    ensure_vector_store(application)
    yield


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

app.state.limiter = limiter
app.state.vector_store_ready = False
app.state.vector_store_error = None
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


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
    ready = bool(getattr(app.state, "vector_store_ready", False))
    return {
        "status": "online" if ready else "degraded",
        "service": settings.APP_NAME,
        "vector_database": settings.VECTOR_DB,
        "vector_database_status": "ready" if ready else "unavailable",
        "collection": settings.QDRANT_COLLECTION,
        "embedding_dimension": settings.EMBEDDING_DIM,
    }
