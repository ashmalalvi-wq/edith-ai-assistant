"""Pydantic schemas for Phase 5's notifications/briefing/integrations APIs."""
import uuid
from datetime import datetime

from pydantic import BaseModel


class NotificationOut(BaseModel):
    id: uuid.UUID
    title: str
    body: str | None
    source: str
    priority: str
    read: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class BriefingOut(BaseModel):
    text: str
    weather_line: str
    schedule_line: str
    tasks_line: str
    emails_line: str
    projects_line: str


class IntegrationStatusOut(BaseModel):
    provider: str
    configured: bool  # client credentials / API key present
    connected: bool  # OAuth token stored (n/a for weather, which needs no auth)


class GoogleOAuthAuthorizeResponse(BaseModel):
    authorization_url: str
