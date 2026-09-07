"""
Profile Memory store — CRUD for Profile, the first-class object representing
a context E.D.I.T.H can operate under (Engineering, School, Personal,
Gaming, Job Applications, or a future custom profile).

Same pattern as structured_store.py: every function takes an explicit
AsyncSession, no global session.
"""
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.exceptions import NotFoundError
from backend.models.db_models import Profile


async def create_profile(
    session: AsyncSession,
    *,
    name: str,
    type: str = "custom",
    preferred_ai_provider: str | None = None,
    preferred_ai_model: str | None = None,
) -> Profile:
    profile = Profile(
        name=name,
        type=type,
        preferred_ai_provider=preferred_ai_provider,
        preferred_ai_model=preferred_ai_model,
    )
    session.add(profile)
    await session.commit()
    await session.refresh(profile)
    return profile


async def list_profiles(session: AsyncSession) -> list[Profile]:
    result = await session.execute(select(Profile).order_by(Profile.updated_at.desc()))
    return list(result.scalars().all())


async def get_profile(session: AsyncSession, profile_id: uuid.UUID) -> Profile:
    profile = await session.get(Profile, profile_id)
    if profile is None:
        raise NotFoundError("Profile", str(profile_id))
    return profile


async def update_profile(session: AsyncSession, profile_id: uuid.UUID, **fields) -> Profile:
    profile = await get_profile(session, profile_id)
    for key, value in fields.items():
        if value is not None:
            setattr(profile, key, value)
    await session.commit()
    await session.refresh(profile)
    return profile


async def delete_profile(session: AsyncSession, profile_id: uuid.UUID) -> None:
    profile = await get_profile(session, profile_id)
    await session.delete(profile)
    await session.commit()
