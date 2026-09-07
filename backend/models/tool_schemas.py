"""Pydantic request/response schemas for the tools and MCP server APIs."""
import uuid
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class PermissionPolicyOut(str, Enum):
    ALWAYS_ALLOW = "always_allow"
    ASK_EVERY_TIME = "ask_every_time"
    NEVER_ALLOW = "never_allow"


class ToolParameterOut(BaseModel):
    name: str
    type: str
    description: str
    required: bool
    default: object | None = None


class ToolOut(BaseModel):
    name: str
    description: str
    category: str
    version: str
    return_type: str
    parameters: list[ToolParameterOut]
    enabled: bool
    permission_policy: PermissionPolicyOut
    status: str


class ToolRunRequest(BaseModel):
    params: dict = Field(default_factory=dict)
    confirmed: bool = False


class ToolRunResponse(BaseModel):
    success: bool
    output: object | None = None
    error: str | None = None
    execution_time_ms: float


class ToolPermissionUpdate(BaseModel):
    enabled: bool | None = None
    permission_policy: PermissionPolicyOut | None = None


class ToolLogOut(BaseModel):
    id: uuid.UUID
    tool_name: str
    category: str
    request_params: dict
    success: bool
    result_summary: str | None
    error: str | None
    execution_time_ms: float
    conversation_id: uuid.UUID | None = None
    profile_id: uuid.UUID | None = None
    initiated_by: str = "user"
    approval_status: str = "not_required"
    created_at: datetime

    model_config = {"from_attributes": True}


# --- MCP servers ---

class MCPServerCreate(BaseModel):
    name: str
    transport: str = Field(pattern="^(stdio|sse)$")
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    url: str | None = None
    enabled: bool = True


class MCPServerUpdate(BaseModel):
    command: str | None = None
    args: list[str] | None = None
    env: dict[str, str] | None = None
    url: str | None = None
    enabled: bool | None = None


class MCPServerOut(BaseModel):
    id: uuid.UUID
    name: str
    transport: str
    command: str | None
    args: list
    env: dict
    url: str | None
    enabled: bool
    status: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
