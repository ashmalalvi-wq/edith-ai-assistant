"""
Qdrant wrapper — the only module that talks to Qdrant directly.

Stores one point per extracted memory (see backend/memory/classifier.py),
never raw chat messages. The payload carries enough structured info
(memory_id, type, project_id) to filter searches without a round-trip to
Postgres, while Postgres remains the source of truth for the memory's
content and lifecycle.
"""
import logging
import uuid

from qdrant_client import AsyncQdrantClient, models

from backend.config.settings import get_settings
from backend.core.exceptions import VectorStoreError

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self) -> None:
        settings = get_settings()
        self._client = AsyncQdrantClient(url=settings.qdrant_url)
        self._collection = settings.qdrant_collection
        self._vector_size = settings.qdrant_vector_size

    async def ensure_collection(self) -> None:
        try:
            exists = await self._client.collection_exists(self._collection)
            if not exists:
                await self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config=models.VectorParams(
                        size=self._vector_size, distance=models.Distance.COSINE
                    ),
                )
        except Exception as exc:  # qdrant-client raises its own exception types
            raise VectorStoreError(str(exc)) from exc

    async def is_reachable(self) -> bool:
        """Cheap connectivity check for health endpoints — just confirms Qdrant
        answers, without ensure_collection()'s create-if-missing side effect
        (that belongs at startup, not on every health poll)."""
        try:
            await self._client.collection_exists(self._collection)
            return True
        except Exception:
            return False

    async def upsert_memory(
        self,
        *,
        point_id: uuid.UUID,
        vector: list[float],
        memory_id: uuid.UUID,
        type: str,
        project_id: uuid.UUID | None,
        content: str,
    ) -> None:
        try:
            await self._client.upsert(
                collection_name=self._collection,
                points=[
                    models.PointStruct(
                        id=str(point_id),
                        vector=vector,
                        payload={
                            "memory_id": str(memory_id),
                            "type": type,
                            "project_id": str(project_id) if project_id else None,
                            "content": content,
                        },
                    )
                ],
            )
        except Exception as exc:
            raise VectorStoreError(str(exc)) from exc

    async def search(
        self,
        *,
        vector: list[float],
        top_k: int,
        min_score: float = 0.0,
        project_id: uuid.UUID | None = None,
    ) -> list[tuple[uuid.UUID, float]]:
        """Returns (memory_id, similarity_score) pairs, best match first."""
        query_filter = None
        if project_id is not None:
            query_filter = models.Filter(
                must=[models.FieldCondition(key="project_id", match=models.MatchValue(value=str(project_id)))]
            )

        try:
            results = await self._client.query_points(
                collection_name=self._collection,
                query=vector,
                limit=top_k,
                query_filter=query_filter,
                score_threshold=min_score or None,
            )
        except Exception as exc:
            raise VectorStoreError(str(exc)) from exc

        return [
            (uuid.UUID(point.payload["memory_id"]), point.score) for point in results.points
        ]

    async def delete_point(self, point_id: uuid.UUID) -> None:
        try:
            await self._client.delete(
                collection_name=self._collection,
                points_selector=models.PointIdsList(points=[str(point_id)]),
            )
        except Exception as exc:
            raise VectorStoreError(str(exc)) from exc


_store_instance: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """FastAPI dependency provider — single instance per process."""
    global _store_instance
    if _store_instance is None:
        _store_instance = VectorStore()
    return _store_instance
