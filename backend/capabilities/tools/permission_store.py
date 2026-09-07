"""Postgres CRUD for per-tool permission policies (tool_permissions table)."""
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.db_models import ToolPermission


async def get_permission(session: AsyncSession, tool_name: str) -> ToolPermission | None:
    return await session.get(ToolPermission, tool_name)


async def list_permissions(session: AsyncSession) -> list[ToolPermission]:
    result = await session.execute(select(ToolPermission))
    return list(result.scalars().all())


async def ensure_default_permission(
    session: AsyncSession, *, tool_name: str, category: str, default_policy: str
) -> ToolPermission:
    """Creates a permission row with the given default if one doesn't already exist."""
    existing = await get_permission(session, tool_name)
    if existing is not None:
        return existing

    permission = ToolPermission(
        tool_name=tool_name, category=category, enabled=True, permission_policy=default_policy
    )
    session.add(permission)
    await session.commit()
    await session.refresh(permission)
    return permission


async def update_permission(
    session: AsyncSession, tool_name: str, *, enabled: bool | None = None, permission_policy: str | None = None
) -> ToolPermission | None:
    permission = await get_permission(session, tool_name)
    if permission is None:
        return None
    if enabled is not None:
        permission.enabled = enabled
    if permission_policy is not None:
        permission.permission_policy = permission_policy
    await session.commit()
    await session.refresh(permission)
    return permission
