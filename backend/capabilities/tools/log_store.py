"""Postgres CRUD for tool execution logs (tool_logs table)."""
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.db_models import ToolLog


async def create_log(
    session: AsyncSession,
    *,
    tool_name: str,
    category: str,
    request_params: dict,
    success: bool,
    result_summary: str | None,
    error: str | None,
    execution_time_ms: float,
    conversation_id: uuid.UUID | None = None,
    profile_id: uuid.UUID | None = None,
    initiated_by: str = "user",
    approval_status: str = "not_required",
) -> ToolLog:
    log = ToolLog(
        tool_name=tool_name,
        category=category,
        request_params=request_params,
        success=success,
        result_summary=result_summary,
        error=error,
        execution_time_ms=execution_time_ms,
        conversation_id=conversation_id,
        profile_id=profile_id,
        initiated_by=initiated_by,
        approval_status=approval_status,
    )
    session.add(log)
    await session.commit()
    await session.refresh(log)
    return log


async def list_logs(
    session: AsyncSession,
    *,
    tool_name: str | None = None,
    conversation_id: uuid.UUID | None = None,
    profile_id: uuid.UUID | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[ToolLog]:
    query = select(ToolLog).order_by(ToolLog.created_at.desc()).limit(limit).offset(offset)
    if tool_name:
        query = query.where(ToolLog.tool_name == tool_name)
    if conversation_id:
        query = query.where(ToolLog.conversation_id == conversation_id)
    if profile_id:
        query = query.where(ToolLog.profile_id == profile_id)
    result = await session.execute(query)
    return list(result.scalars().all())
