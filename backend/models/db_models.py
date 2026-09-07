"""
SQLAlchemy ORM models for E.D.I.T.H's structured memory store.

These mirror the schema in database/postgres/init/001_init.sql. IDs are
generated client-side (uuid4) rather than relying on Postgres's
gen_random_uuid() server default, so the same models work unmodified
against the SQLite database used in tests (see backend/tests/conftest.py).
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, ForeignKey, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import Uuid

# All timestamp columns are timezone-aware (Postgres TIMESTAMPTZ) — without
# this, SQLAlchemy defaults to naive TIMESTAMP and asyncpg rejects the
# timezone-aware datetimes _now() produces.
_TZDateTime = DateTime(timezone=True)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    display_name: Mapped[str] = mapped_column(Text, default="User")
    default_ai_provider: Mapped[str] = mapped_column(Text, default="ollama")
    default_ai_model: Mapped[str | None] = mapped_column(Text, nullable=True)
    preferences: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(_TZDateTime, default=_now)
    updated_at: Mapped[datetime] = mapped_column(_TZDateTime, default=_now, onupdate=_now)


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(Text, default="active")
    created_at: Mapped[datetime] = mapped_column(_TZDateTime, default=_now)
    updated_at: Mapped[datetime] = mapped_column(_TZDateTime, default=_now, onupdate=_now)


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(Text, default="New conversation")
    project_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("projects.id", ondelete="SET NULL"), nullable=True
    )
    ai_provider: Mapped[str] = mapped_column(Text, default="ollama")
    ai_model: Mapped[str | None] = mapped_column(Text, nullable=True)
    profile_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("profiles.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(_TZDateTime, default=_now)
    updated_at: Mapped[datetime] = mapped_column(_TZDateTime, default=_now, onupdate=_now)

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan", order_by="Message.created_at"
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("conversations.id", ondelete="CASCADE")
    )
    role: Mapped[str] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(_TZDateTime, default=_now)

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")


class Memory(Base):
    __tablename__ = "memories"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    type: Mapped[str] = mapped_column(Text)  # temporary | long_term | project
    content: Mapped[str] = mapped_column(Text)
    project_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("projects.id", ondelete="SET NULL"), nullable=True
    )
    source_conversation_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("conversations.id", ondelete="SET NULL"), nullable=True
    )
    qdrant_point_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(_TZDateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(_TZDateTime, default=_now)
    updated_at: Mapped[datetime] = mapped_column(_TZDateTime, default=_now, onupdate=_now)


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("projects.id", ondelete="SET NULL"), nullable=True
    )
    title: Mapped[str] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(Text, default="open")
    priority: Mapped[str] = mapped_column(Text, default="medium")  # low | medium | high
    due_date: Mapped[datetime | None] = mapped_column(_TZDateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(_TZDateTime, default=_now)
    updated_at: Mapped[datetime] = mapped_column(_TZDateTime, default=_now, onupdate=_now)


class Profile(Base):
    """
    A first-class context E.D.I.T.H can operate under (Engineering, School,
    Personal, Gaming, Job Applications, or a future custom profile). Basic
    functionality only for now — CRUD plus the fields the long-term vision
    calls for; nothing yet actively switches behavior based on the active
    profile beyond what Session.profile_id references.
    """

    __tablename__ = "profiles"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(Text)
    type: Mapped[str] = mapped_column(Text, default="custom")
    preferred_ai_provider: Mapped[str | None] = mapped_column(Text, nullable=True)
    preferred_ai_model: Mapped[str | None] = mapped_column(Text, nullable=True)
    active_project_ids: Mapped[list] = mapped_column(JSON, default=list)
    open_workspaces: Mapped[list] = mapped_column(JSON, default=list)
    recent_files: Mapped[list] = mapped_column(JSON, default=list)
    dashboard_layout: Mapped[dict] = mapped_column(JSON, default=dict)
    preferred_capabilities: Mapped[list] = mapped_column(JSON, default=list)
    automation_preferences: Mapped[dict] = mapped_column(JSON, default=dict)
    # Phase 4: session restoration. startup_applications is a list of
    # {"name", "command": [...], "args": [...]} objects launched by
    # workspace.resume_profile; recent_conversation_ids is a bounded
    # most-recent-first list maintained by the memory pipeline.
    startup_applications: Mapped[list] = mapped_column(JSON, default=list)
    recent_conversation_ids: Mapped[list] = mapped_column(JSON, default=list)
    workspace_preferences: Mapped[dict] = mapped_column(JSON, default=dict)
    # Phase 6: Engineering profile support. cad_files/git_repositories are
    # lists of absolute paths; printer_settings is a small dict (e.g.
    # {"bambu_ip": "...", "default_material_profile_id": "..."}).
    cad_files: Mapped[list] = mapped_column(JSON, default=list)
    printer_settings: Mapped[dict] = mapped_column(JSON, default=dict)
    git_repositories: Mapped[list] = mapped_column(JSON, default=list)
    research_documents: Mapped[list] = mapped_column(JSON, default=list)  # ResearchDocument ids
    created_at: Mapped[datetime] = mapped_column(_TZDateTime, default=_now)
    updated_at: Mapped[datetime] = mapped_column(_TZDateTime, default=_now, onupdate=_now)


class Session(Base):
    """
    Session Memory: the currently-active runtime context (which profile,
    project, provider/model, and workspace are "live" right now), separate
    from the durable history in Conversation/Message. One row is enough for
    a single local user; profile_id is nullable so this works before any
    profile has been created.
    """

    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    profile_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("profiles.id", ondelete="SET NULL"), nullable=True
    )
    active_project_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("projects.id", ondelete="SET NULL"), nullable=True
    )
    active_ai_provider: Mapped[str | None] = mapped_column(Text, nullable=True)
    active_ai_model: Mapped[str | None] = mapped_column(Text, nullable=True)
    context: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(_TZDateTime, default=_now)
    updated_at: Mapped[datetime] = mapped_column(_TZDateTime, default=_now, onupdate=_now)


class ToolPermission(Base):
    __tablename__ = "tool_permissions"

    tool_name: Mapped[str] = mapped_column(Text, primary_key=True)
    category: Mapped[str] = mapped_column(Text)
    enabled: Mapped[bool] = mapped_column(default=True)
    permission_policy: Mapped[str] = mapped_column(Text)  # always_allow | ask_every_time | never_allow
    updated_at: Mapped[datetime] = mapped_column(_TZDateTime, default=_now, onupdate=_now)


class ToolLog(Base):
    __tablename__ = "tool_logs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    tool_name: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(Text)
    request_params: Mapped[dict] = mapped_column(JSON, default=dict)
    success: Mapped[bool] = mapped_column()
    result_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    execution_time_ms: Mapped[float] = mapped_column()
    # Phase 4: capability activity history — who/what triggered this call.
    # All nullable so pre-Phase-4 rows and dashboard-initiated runs (which
    # have no conversation) remain valid.
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("conversations.id", ondelete="SET NULL"), nullable=True
    )
    profile_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("profiles.id", ondelete="SET NULL"), nullable=True
    )
    initiated_by: Mapped[str] = mapped_column(Text, default="user")  # "user" | "ai"
    approval_status: Mapped[str] = mapped_column(Text, default="not_required")
    # "not_required" | "auto_allowed" | "user_confirmed" | "user_denied"
    created_at: Mapped[datetime] = mapped_column(_TZDateTime, default=_now)


class OAuthCredential(Base):
    """
    Phase 5: stores OAuth tokens for external integrations (Gmail, Google
    Calendar). Tokens are Fernet-encrypted before ever reaching this table
    (see backend/capabilities/integrations/oauth.py) — this model only ever
    holds ciphertext, never a usable access/refresh token in plaintext.
    """

    __tablename__ = "oauth_credentials"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    provider: Mapped[str] = mapped_column(Text, unique=True)  # "gmail" | "google_calendar"
    encrypted_access_token: Mapped[str] = mapped_column(Text)
    encrypted_refresh_token: Mapped[str | None] = mapped_column(Text, nullable=True)
    token_expires_at: Mapped[datetime | None] = mapped_column(_TZDateTime, nullable=True)
    scopes: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(_TZDateTime, default=_now)
    updated_at: Mapped[datetime] = mapped_column(_TZDateTime, default=_now, onupdate=_now)


class Notification(Base):
    """Phase 5: user-facing alerts generated by capabilities (email/calendar/task/weather/system)."""

    __tablename__ = "notifications"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(Text)
    body: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(Text)  # "email" | "calendar" | "task" | "weather" | "system"
    priority: Mapped[str] = mapped_column(Text, default="medium")  # low | medium | high | urgent
    read: Mapped[bool] = mapped_column(default=False)
    profile_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("profiles.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(_TZDateTime, default=_now)


class MCPServer(Base):
    __tablename__ = "mcp_servers"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(Text, unique=True)
    transport: Mapped[str] = mapped_column(Text)  # stdio | sse
    command: Mapped[str | None] = mapped_column(Text, nullable=True)
    args: Mapped[list] = mapped_column(JSON, default=list)
    env: Mapped[dict] = mapped_column(JSON, default=dict)
    url: Mapped[str | None] = mapped_column(Text, nullable=True)
    enabled: Mapped[bool] = mapped_column(default=True)
    status: Mapped[str] = mapped_column(Text, default="disconnected")
    created_at: Mapped[datetime] = mapped_column(_TZDateTime, default=_now)
    updated_at: Mapped[datetime] = mapped_column(_TZDateTime, default=_now, onupdate=_now)


class MaterialProfile(Base):
    """Phase 6: 3D printing knowledge — a named material/print-settings profile (e.g. 'PETG — 0.2mm')."""

    __tablename__ = "material_profiles"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(Text)
    material_type: Mapped[str] = mapped_column(Text)  # PLA | PETG | TPU | ABS | ASA | ...
    nozzle_temp_c: Mapped[int | None] = mapped_column(nullable=True)
    bed_temp_c: Mapped[int | None] = mapped_column(nullable=True)
    layer_height_mm: Mapped[float | None] = mapped_column(nullable=True)
    print_speed_mm_s: Mapped[int | None] = mapped_column(nullable=True)
    supports: Mapped[bool] = mapped_column(default=False)
    nozzle_size_mm: Mapped[float | None] = mapped_column(nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(_TZDateTime, default=_now)
    updated_at: Mapped[datetime] = mapped_column(_TZDateTime, default=_now, onupdate=_now)


class PrintHistory(Base):
    """Phase 6: a record of one print attempt, successful or failed."""

    __tablename__ = "print_history"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    material_profile_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("material_profiles.id", ondelete="SET NULL"), nullable=True
    )
    project_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("projects.id", ondelete="SET NULL"), nullable=True
    )
    file_name: Mapped[str] = mapped_column(Text)
    success: Mapped[bool] = mapped_column()
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    printed_at: Mapped[datetime] = mapped_column(_TZDateTime, default=_now)


class ResearchDocument(Base):
    """Phase 6: a saved technical resource (paper, datasheet, forum post, ...), optionally linked to a project."""

    __tablename__ = "research_documents"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(Text)
    url: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    tags: Mapped[list] = mapped_column(JSON, default=list)
    project_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("projects.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(_TZDateTime, default=_now)


class AutomationRule(Base):
    """
    Phase 7: a trigger -> conditions -> actions automation. Actions always
    execute through CapabilityManager.execute_capability() (see engine.py)
    — this table only stores *what* to do, never bypasses the permission
    system that governs *whether* it's allowed to happen unattended.
    """

    __tablename__ = "automation_rules"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    enabled: Mapped[bool] = mapped_column(default=False)  # never auto-enabled — see automation.create_from_description
    profile_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("profiles.id", ondelete="SET NULL"), nullable=True
    )
    # "schedule" | "capability_completed" | "system_startup" | "manual" | "file_created"
    trigger_type: Mapped[str] = mapped_column(Text)
    trigger_config: Mapped[dict] = mapped_column(JSON, default=dict)
    conditions: Mapped[list] = mapped_column(JSON, default=list)
    actions: Mapped[list] = mapped_column(JSON, default=list)
    is_template: Mapped[bool] = mapped_column(default=False)  # routine-library entries, instantiated (not run directly)
    created_at: Mapped[datetime] = mapped_column(_TZDateTime, default=_now)
    updated_at: Mapped[datetime] = mapped_column(_TZDateTime, default=_now, onupdate=_now)
    last_run_at: Mapped[datetime | None] = mapped_column(_TZDateTime, nullable=True)
    run_count: Mapped[int] = mapped_column(default=0)


class AutomationLog(Base):
    """Phase 7: one execution record for an AutomationRule — every automation run, logged (spec item 11)."""

    __tablename__ = "automation_logs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    rule_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("automation_rules.id", ondelete="SET NULL"), nullable=True
    )
    rule_name: Mapped[str] = mapped_column(Text)  # denormalized snapshot — survives rule rename/deletion
    triggered_by: Mapped[str] = mapped_column(Text)  # event name, "schedule", or "manual"
    trigger_payload: Mapped[dict] = mapped_column(JSON, default=dict)
    conditions_passed: Mapped[bool] = mapped_column(default=True)
    actions_executed: Mapped[list] = mapped_column(JSON, default=list)
    success: Mapped[bool] = mapped_column()
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    execution_time_ms: Mapped[float] = mapped_column(default=0.0)
    created_at: Mapped[datetime] = mapped_column(_TZDateTime, default=_now)
