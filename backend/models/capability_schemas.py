"""Pydantic request/response schemas for the Capability Manager API."""
from pydantic import BaseModel, Field

from backend.capabilities.types import CapabilityType
from backend.models.tool_schemas import PermissionPolicyOut


class CapabilityOut(BaseModel):
    id: str
    type: CapabilityType
    name: str
    description: str
    category: str
    enabled: bool
    status: str
    metadata: dict


class CapabilityRunRequest(BaseModel):
    params: dict = Field(default_factory=dict)
    confirmed: bool = False


class CapabilityRunResponse(BaseModel):
    success: bool
    output: object | None = None
    error: str | None = None
    execution_time_ms: float


class CapabilityPermissionUpdate(BaseModel):
    enabled: bool | None = None
    permission_policy: PermissionPolicyOut | None = None


# --- Workspace Manager (Phase 4) ---

class WorkspaceAppOut(BaseModel):
    id: str
    name: str
    pid: int
    launched_at: str


class WorkspaceStateOut(BaseModel):
    applications: list[WorkspaceAppOut]


class WorkspaceResumeRequest(BaseModel):
    confirmed: bool = False
