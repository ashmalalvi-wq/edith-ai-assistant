"""Pydantic request/response schemas for the projects/conversations/memory/tasks/users APIs."""
import uuid
from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from backend.models.schemas import AIProviderName, ChatMessage


# --- Projects ---

class ProjectCreate(BaseModel):
    name: str
    description: str | None = None
    status: str = "active"


class ProjectUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    status: str | None = None


class ProjectOut(BaseModel):
    id: uuid.UUID
    name: str
    description: str | None
    status: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# --- Conversations ---

class ConversationCreate(BaseModel):
    title: str = "New conversation"
    project_id: uuid.UUID | None = None
    ai_provider: str = "ollama"
    ai_model: str | None = None


class ConversationUpdate(BaseModel):
    title: str | None = None
    project_id: uuid.UUID | None = None


class ConversationSummary(BaseModel):
    id: uuid.UUID
    title: str
    project_id: uuid.UUID | None
    ai_provider: str
    ai_model: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ConversationDetail(ConversationSummary):
    messages: list[ChatMessage]


# --- Chat (Phase 2: history lives server-side, keyed by conversation_id) ---

class ChatAttachmentRef(BaseModel):
    """One image attached to a chat turn — as returned by POST /api/attachments,
    passed straight back through here rather than re-uploaded. Only `path` is
    actually used server-side (to read the file); the rest is bookkeeping."""

    id: str
    filename: str
    path: str
    mimetype: str


class ChatTurnRequest(BaseModel):
    message: str = Field(..., min_length=1)
    conversation_id: uuid.UUID | None = Field(
        default=None, description="Omit to start a new conversation."
    )
    project_id: uuid.UUID | None = Field(
        default=None, description="Associates a new conversation with a project."
    )
    provider: AIProviderName | None = None
    model: str | None = None
    mode: str | None = Field(
        default=None,
        description="'expert' routes straight to the Expert Reasoning Agent (Claude Code) instead of the normal chat pipeline. Omit for normal assistant behavior.",
    )
    expert_task_mode: Literal["analyze", "implement"] | None = Field(
        default=None,
        description="Only relevant when mode='expert'. 'analyze' (default) is read-only (Claude Code runs with --permission-mode plan). 'implement' lets it actually write files (--permission-mode acceptEdits). Ignored outside Expert mode.",
    )
    attachments: list[ChatAttachmentRef] | None = Field(
        default=None, description="Images uploaded via POST /api/attachments before sending this turn."
    )


class PendingApprovalOut(BaseModel):
    """A capability call the model requested that needs user confirmation before it runs."""

    call_id: str
    capability_id: str
    params: dict


class ChatCapabilityResolveRequest(BaseModel):
    approved: bool


class UIActionOut(BaseModel):
    """A frontend-actionable directive a capability call emitted this turn —
    today only widget.open/widget.close/widget.close_all populate this
    (see backend/capabilities/tools/builtin/widget_tool.py and
    backend/api/routes/chat.py's _collect_widget_ui_actions)."""

    action: str
    widget_id: str | None = None


class ChatTurnResponse(BaseModel):
    conversation_id: uuid.UUID
    conversation_title: str
    provider: AIProviderName
    model: str
    # None when the turn paused for capability approval instead of replying.
    message: ChatMessage | None = None
    pending_approval: PendingApprovalOut | None = None
    ui_actions: list[UIActionOut] = Field(default_factory=list)


# --- Memory ---

class MemoryType(str, Enum):
    TEMPORARY = "temporary"
    LONG_TERM = "long_term"
    PROJECT = "project"
    EMAIL = "email"  # Phase 5: email-derived summaries/follow-ups (see email intelligence)
    ENGINEERING = "engineering"  # Phase 6: CAD design decisions, materials, dimensions, engineering notes


class MemoryCreate(BaseModel):
    type: MemoryType
    content: str
    project_id: uuid.UUID | None = None


class MemoryUpdate(BaseModel):
    content: str | None = None
    type: MemoryType | None = None
    project_id: uuid.UUID | None = None


class MemoryOut(BaseModel):
    id: uuid.UUID
    type: MemoryType
    content: str
    project_id: uuid.UUID | None
    source_conversation_id: uuid.UUID | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class MemorySearchResult(BaseModel):
    memory: MemoryOut
    score: float


# --- Tasks ---

class TaskCreate(BaseModel):
    title: str
    description: str | None = None
    project_id: uuid.UUID | None = None
    status: str = "open"
    priority: str = "medium"
    due_date: datetime | None = None


class TaskUpdate(BaseModel):
    title: str | None = None
    description: str | None = None
    status: str | None = None
    priority: str | None = None
    due_date: datetime | None = None


class TaskOut(BaseModel):
    id: uuid.UUID
    title: str
    description: str | None
    project_id: uuid.UUID | None
    status: str
    priority: str
    due_date: datetime | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# --- Profiles (Profile Memory) ---

class ProfileCreate(BaseModel):
    name: str
    type: str = "custom"
    preferred_ai_provider: str | None = None
    preferred_ai_model: str | None = None


class ProfileUpdate(BaseModel):
    name: str | None = None
    type: str | None = None
    preferred_ai_provider: str | None = None
    preferred_ai_model: str | None = None
    active_project_ids: list[uuid.UUID] | None = None
    open_workspaces: list[str] | None = None
    recent_files: list[str] | None = None
    dashboard_layout: dict | None = None
    preferred_capabilities: list[str] | None = None
    automation_preferences: dict | None = None
    cad_files: list[str] | None = None
    printer_settings: dict | None = None
    git_repositories: list[str] | None = None
    research_documents: list[uuid.UUID] | None = None


class ProfileOut(BaseModel):
    id: uuid.UUID
    name: str
    type: str
    preferred_ai_provider: str | None
    preferred_ai_model: str | None
    active_project_ids: list
    open_workspaces: list
    recent_files: list
    dashboard_layout: dict
    preferred_capabilities: list
    automation_preferences: dict
    cad_files: list
    printer_settings: dict
    git_repositories: list
    research_documents: list
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# --- Session Memory ---

class SessionOut(BaseModel):
    id: uuid.UUID
    profile_id: uuid.UUID | None
    active_project_id: uuid.UUID | None
    active_ai_provider: str | None
    active_ai_model: str | None
    context: dict
    updated_at: datetime

    model_config = {"from_attributes": True}


# --- Users / preferences ---

class UserPreferencesUpdate(BaseModel):
    display_name: str | None = None
    default_ai_provider: str | None = None
    default_ai_model: str | None = None
    preferences: dict | None = None


class UserOut(BaseModel):
    id: uuid.UUID
    display_name: str
    default_ai_provider: str
    default_ai_model: str | None
    preferences: dict

    model_config = {"from_attributes": True}
