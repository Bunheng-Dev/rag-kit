from collections.abc import Mapping, Sequence
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from app.core.config import settings
from .base import VectorDBProvider


class QdrantVectorDBProvider(VectorDBProvider):
    SCROLL_PAGE_SIZE = 100

    def __init__(self) -> None:
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY or None,
            timeout=settings.QDRANT_TIMEOUT,
        )
        self.collection_name = settings.QDRANT_COLLECTION

    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection_name):
            collection = self.client.get_collection(self.collection_name)
            vector_config = collection.config.params.vectors
            if isinstance(vector_config, dict):
                vector_config = vector_config.get("")
            if vector_config is None:
                raise ValueError(f"Collection '{self.collection_name}' must use one unnamed vector")
            if vector_config.size != settings.EMBEDDING_DIM or vector_config.distance != Distance.COSINE:
                raise ValueError(
                    f"Collection '{self.collection_name}' must use "
                    f"{settings.EMBEDDING_DIM}-dimensional cosine vectors"
                )
        else:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=settings.EMBEDDING_DIM, distance=Distance.COSINE),
            )

        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="entity_type",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="sync_source",
            field_schema=PayloadSchemaType.KEYWORD,
        )

    @staticmethod
    def point_id(document_id: str | int) -> str:
        return str(uuid5(NAMESPACE_URL, f"{settings.QDRANT_COLLECTION}:{document_id}"))

    @staticmethod
    def _filter(where: Mapping[str, object] | None = None) -> Filter | None:
        if not where:
            return None
        return Filter(
            must=[FieldCondition(key=key, match=MatchValue(value=value)) for key, value in where.items()]
        )

    def query(
        self,
        query_vector: Sequence[float],
        n_results: int = 3,
        where: Mapping[str, object] | None = None,
    ) -> dict[str, list[list[dict | str]]]:
        if len(query_vector) != settings.EMBEDDING_DIM:
            raise ValueError(f"Query vector must have {settings.EMBEDDING_DIM} dimensions")
        query_filter = self._filter(where)
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=list(query_vector),
            query_filter=query_filter,
            limit=n_results,
            with_payload=True,
        )
        metadatas: list[dict] = []
        documents: list[str] = []
        for point in response.points:
            payload = point.payload or {}
            documents.append(str(payload.get("text", "")))
            metadatas.append({key: value for key, value in payload.items() if key != "text"})
        return {"metadatas": [metadatas], "documents": [documents]}

    def search_similar(
        self,
        query_vector: Sequence[float],
        n_results: int = 3,
        where: Mapping[str, object] | None = None,
    ) -> dict[str, list[list[dict | str]]]:
        return self.query(query_vector, n_results, where)

    def upsert_documents(
        self,
        ids: Sequence[str | int],
        documents: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[dict] | None = None,
    ) -> None:
        if not (len(ids) == len(documents) == len(embeddings)):
            raise ValueError("ids, documents, and embeddings must have equal lengths")
        if metadatas is not None and len(metadatas) != len(ids):
            raise ValueError("metadatas must have the same length as ids")

        points = []
        for index, (document_id, document, embedding) in enumerate(zip(ids, documents, embeddings)):
            if len(embedding) != settings.EMBEDDING_DIM:
                raise ValueError(
                    f"Embedding for document {document_id!r} has {len(embedding)} dimensions; "
                    f"expected {settings.EMBEDDING_DIM}"
                )
            payload = dict(metadatas[index]) if metadatas is not None else {}
            payload["text"] = document
            points.append(PointStruct(id=self.point_id(document_id), vector=list(embedding), payload=payload))
        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)

    def delete_documents(self, ids: Sequence[str | int]) -> None:
        if ids:
            self.delete_points([self.point_id(document_id) for document_id in ids])

    def list_point_ids(self, where: Mapping[str, object] | None = None) -> list[str]:
        point_ids: list[str] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=self._filter(where),
                limit=self.SCROLL_PAGE_SIZE,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            point_ids.extend(str(point.id) for point in points)
            if offset is None:
                return point_ids

    def delete_points(self, point_ids: Sequence[str | int]) -> None:
        if point_ids:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=list(point_ids),
            )
