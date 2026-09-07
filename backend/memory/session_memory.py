"""
Session Memory — the currently-active runtime context (active profile,
project, provider/model, and arbitrary workspace state), distinct from the
durable conversation history in structured_store.

A single local user has exactly one live session at a time, so this store
gets-or-creates that one row rather than exposing full CRUD, mirroring how
structured_store.get_or_create_default_user works for the User table.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.db_models import Session


async def get_or_create_session(session: AsyncSession) -> Session:
    result = await session.execute(select(Session).limit(1))
    active_session = result.scalar_one_or_none()
    if active_session is None:
        active_session = Session()
        session.add(active_session)
        await session.commit()
        await session.refresh(active_session)
    return active_session


async def update_session(
    session: AsyncSession,
    *,
    profile_id: uuid.UUID | None = None,
    active_project_id: uuid.UUID | None = None,
    active_ai_provider: str | None = None,
    active_ai_model: str | None = None,
    context: dict | None = None,
) -> Session:
    active_session = await get_or_create_session(session)
    if profile_id is not None:
        active_session.profile_id = profile_id
    if active_project_id is not None:
        active_session.active_project_id = active_project_id
    if active_ai_provider is not None:
        active_session.active_ai_provider = active_ai_provider
    if active_ai_model is not None:
        active_session.active_ai_model = active_ai_model
    if context is not None:
        active_session.context = context
    active_session.updated_at = datetime.now(timezone.utc)
    await session.commit()
    await session.refresh(active_session)
    return active_session
