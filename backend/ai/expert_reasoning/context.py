"""
Context assembly for expert reasoning tasks — reuses MemoryRetrievalService
(the exact same collaborator MemoryPipeline uses for chat context) rather
than a new retrieval path, so "what E.D.I.T.H. knows" stays consistent
between a normal chat turn and a delegated expert task.

Bounded and relevance-scoped on purpose (Phase 6 spec: "do not send
unnecessary personal data") — active project/profile + top-K relevant
memories for *this* task, not the whole conversation history or other
profiles' data.
"""
from sqlalchemy.ext.asyncio import AsyncSession

from backend.memory import session_memory, structured_store
from backend.memory.profiles import store as profile_store
from backend.memory.retrieval import MemoryRetrievalService


async def build_expert_context(
    session: AsyncSession, retrieval_service: MemoryRetrievalService, *, task: str, project_id=None
) -> str:
    active_session = await session_memory.get_or_create_session(session)
    effective_project_id = project_id or active_session.active_project_id

    lines = []

    if active_session.profile_id is not None:
        try:
            profile = await profile_store.get_profile(session, active_session.profile_id)
            lines.append(f"Active profile: {profile.name} ({profile.type})")
        except Exception:
            pass

    if effective_project_id is not None:
        try:
            project = await structured_store.get_project(session, effective_project_id)
            lines.append(f"Project: {project.name} — {project.description or 'no description on file'}")
        except Exception:
            pass

    try:
        retrieved = await retrieval_service.retrieve(session, user_message=task, project_id=effective_project_id)
        block = retrieved.to_prompt_block()
        if block:
            lines.append(block)
    except Exception:
        pass  # same "never break the calling action over a context-retrieval failure" posture as MemoryPipeline

    return "\n\n".join(lines)
