"""Pydantic schemas for Phase 7's automation API."""
import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class AutomationRuleCreate(BaseModel):
    name: str
    description: str | None = None
    trigger_type: str
    trigger_config: dict = Field(default_factory=dict)
    conditions: list = Field(default_factory=list)
    actions: list = Field(default_factory=list)
    profile_id: uuid.UUID | None = None
    enabled: bool = False


class AutomationRuleUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    trigger_config: dict | None = None
    conditions: list | None = None
    actions: list | None = None


class AutomationRuleOut(BaseModel):
    id: uuid.UUID
    name: str
    description: str | None
    enabled: bool
    trigger_type: str
    trigger_config: dict
    conditions: list
    actions: list
    profile_id: uuid.UUID | None
    is_template: bool
    run_count: int
    last_run_at: datetime | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class AutomationLogOut(BaseModel):
    id: uuid.UUID
    rule_id: uuid.UUID | None
    rule_name: str
    triggered_by: str
    trigger_payload: dict
    conditions_passed: bool
    actions_executed: list
    success: bool
    error: str | None
    execution_time_ms: float
    created_at: datetime

    model_config = {"from_attributes": True}


class CreateFromDescriptionRequest(BaseModel):
    description: str


class InstantiateTemplateRequest(BaseModel):
    profile_id: uuid.UUID | None = None


class AutomationStatusOut(BaseModel):
    paused: bool
    automation_enabled_setting: bool
    scheduled_rule_count: int
    next_run_times: dict[str, str | None]
