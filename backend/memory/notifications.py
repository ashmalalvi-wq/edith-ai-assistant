"""
Notification store — CRUD for the `notifications` table, same explicit-
AsyncSession pattern as structured_store.py. Lives under backend/memory/
(not backend/capabilities/) because it's structured storage like
tasks/projects/memories, not something with its own execution/permission
model — capabilities that generate notifications (weather alerts, task
deadlines, calendar reminders) call create_notification() directly; it's
also exposed to the user/AI as two low-risk read tools
(notification.list/notification.dismiss, see
backend/capabilities/tools/builtin/notification_tool.py).
"""
import uuid
from datetime import datetime, time

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.settings import get_settings
from backend.core.exceptions import NotFoundError
from backend.models.db_models import Notification

_PRIORITY_ORDER = {"low": 0, "medium": 1, "high": 2, "urgent": 3}


def _parse_hhmm(value: str) -> time:
    hour, minute = value.split(":")
    return time(hour=int(hour), minute=int(minute))


def is_quiet_hours(now: datetime | None = None) -> bool:
    """
    Whether `now` (local time) falls inside the configured quiet hours
    window. Handles overnight windows (e.g. 22:00-07:00) as well as
    same-day ones.
    """
    settings = get_settings()
    start = _parse_hhmm(settings.quiet_hours_start)
    end = _parse_hhmm(settings.quiet_hours_end)
    current = (now or datetime.now()).time()

    if start <= end:
        return start <= current < end
    return current >= start or current < end  # overnight window


async def create_notification(
    session: AsyncSession,
    *,
    title: str,
    body: str | None,
    source: str,
    priority: str = "medium",
    profile_id: uuid.UUID | None = None,
    respect_quiet_hours: bool = True,
) -> Notification | None:
    """
    Creates a notification, unless it's non-urgent and quiet hours are
    active — urgent notifications (Phase 5: "weather alerts", critical
    deadlines) still get through. Returns None when suppressed so callers
    can tell the difference from a real failure.
    """
    if respect_quiet_hours and priority != "urgent" and is_quiet_hours():
        return None

    notification = Notification(title=title, body=body, source=source, priority=priority, profile_id=profile_id)
    session.add(notification)
    await session.commit()
    await session.refresh(notification)
    return notification


async def list_notifications(
    session: AsyncSession, *, unread_only: bool = False, profile_id: uuid.UUID | None = None, limit: int = 50
) -> list[Notification]:
    query = select(Notification).order_by(Notification.created_at.desc()).limit(limit)
    if unread_only:
        query = query.where(Notification.read.is_(False))
    if profile_id:
        query = query.where(Notification.profile_id == profile_id)
    result = await session.execute(query)
    notifications = list(result.scalars().all())
    return sorted(notifications, key=lambda n: _PRIORITY_ORDER.get(n.priority, 0), reverse=True)


async def dismiss_notification(session: AsyncSession, notification_id: uuid.UUID) -> Notification:
    notification = await session.get(Notification, notification_id)
    if notification is None:
        raise NotFoundError("Notification", str(notification_id))
    notification.read = True
    await session.commit()
    await session.refresh(notification)
    return notification
