"""
Shared types for the Capability Manager — the vocabulary every capability
source (tools, MCP, and future integrations/widgets/automation) is
described in, regardless of how it's actually implemented underneath.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CapabilityType(str, Enum):
    """What kind of thing a capability is, for grouping/filtering in the UI."""

    TOOL = "tool"
    MCP = "mcp"
    INTEGRATION = "integration"
    WIDGET = "widget"
    AUTOMATION = "automation"


@dataclass
class Capability:
    """A single registered capability's metadata + current status, merged
    from whichever source (ToolManager, MCPManager, ...) actually owns it."""

    id: str
    type: CapabilityType
    name: str
    description: str
    category: str
    enabled: bool
    status: str
    metadata: dict[str, Any] = field(default_factory=dict)
