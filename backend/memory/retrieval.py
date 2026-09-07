"""
Memory retrieval — combines structured (Postgres) and semantic (Qdrant)
lookups into a single context block the AI model router can use.

This is provider-agnostic: it only produces text, so the same context
works whether the request ends up going to Ollama, OpenAI, or Anthropic.
"""
import logging
import uuid
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.settings import get_settings
from backend.memory.embeddings.base import EmbeddingProvider
from backend.memory import structured_store
from backend.memory.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievedContext:
    project_summary: str | None = None
    user_preferences: dict = field(default_factory=dict)
    relevant_memories: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.project_summary or self.user_preferences or self.relevant_memories)

    def to_prompt_block(self) -> str:
        """Renders as a compact block to append to the system prompt."""
        if self.is_empty():
            return ""

        lines = ["--- E.D.I.T.H memory context (use naturally, don't quote verbatim) ---"]

        if self.user_preferences:
            prefs = "; ".join(f"{k}: {v}" for k, v in self.user_preferences.items())
            lines.append(f"User preferences: {prefs}")

        if self.project_summary:
            lines.append(f"Active project: {self.project_summary}")

        if self.relevant_memories:
            lines.append("Relevant past information:")
            lines.extend(f"- {memory}" for memory in self.relevant_memories)

        lines.append("--- end memory context ---")
        return "\n".join(lines)


class MemoryRetrievalService:
    def __init__(self, embedding_provider: EmbeddingProvider, vector_store: VectorStore) -> None:
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store

    async def retrieve(
        self,
        session: AsyncSession,
        *,
        user_message: str,
        project_id: uuid.UUID | None = None,
    ) -> RetrievedContext:
        settings = get_settings()
        context = RetrievedContext()

        # 1. Structured lookups.
        user = await structured_store.get_or_create_default_user(session)
        if user.preferences:
            context.user_preferences = user.preferences

        if project_id:
            try:
                project = await structured_store.get_project(session, project_id)
                context.project_summary = f"{project.name} — {project.description or 'no description on file'}"
            except Exception:
                logger.warning("Referenced project %s not found during retrieval", project_id)

        # 2. Semantic lookup in Qdrant.
        try:
            vector = await self._embedding_provider.embed(user_message)
            hits = await self._vector_store.search(
                vector=vector,
                top_k=settings.memory_search_top_k,
                min_score=settings.memory_min_similarity,
                project_id=project_id,
            )
        except Exception:
            # Memory retrieval must never break the chat path — degrade gracefully.
            logger.exception("Semantic memory search failed; continuing without it")
            hits = []

        if hits:
            memory_ids = [memory_id for memory_id, _score in hits]
            memories = await structured_store.get_memories_by_ids(session, memory_ids)
            context.relevant_memories = [m.content for m in memories]

        return context
