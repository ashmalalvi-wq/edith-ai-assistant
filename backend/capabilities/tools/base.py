"""
Tool framework core types.

Every tool — built-in or MCP-discovered — is described the same way and
executed through the same interface, so the Tool Manager (manager.py)
never needs to know or care what kind of tool it's running.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ToolCategory(str, Enum):
    """
    Permission/grouping category. Includes categories with no real tools yet
    (browser, fusion, bambu, calendar, email) so the permissions data model
    and UI are ready for the phases that will populate them.
    """

    FILESYSTEM = "filesystem"
    TERMINAL = "terminal"
    VSCODE = "vscode"
    WORKSPACE = "workspace"
    BROWSER = "browser"
    FUSION = "fusion"
    BAMBU = "bambu"
    CALENDAR = "calendar"
    EMAIL = "email"
    WEATHER = "weather"
    TASK = "task"
    NOTIFICATION = "notification"
    BRIEFING = "briefing"
    EXPERT_REASONING = "expert_reasoning"
    GIT = "git"
    PRINTING = "printing"  # 3D printing knowledge (materials/profiles/history) — distinct from BAMBU (live printer control)
    RESEARCH = "research"
    ENGINEERING = "engineering"  # cross-cutting project scaffolding/status aggregation
    AUTOMATION = "automation"
    WIDGET = "widget"  # open/close floating widgets on the Orb Home screen — see widget_tool.py
    MCP = "mcp"


class PermissionPolicy(str, Enum):
    ALWAYS_ALLOW = "always_allow"
    ASK_EVERY_TIME = "ask_every_time"
    NEVER_ALLOW = "never_allow"


@dataclass
class ToolParameter:
    """One entry in a tool's parameter schema — enough for validation and a UI form."""

    name: str
    type: str  # "string" | "integer" | "boolean" | "object" | "array"
    description: str = ""
    required: bool = True
    default: Any = None


@dataclass
class ToolResult:
    success: bool
    output: Any = None
    error: str | None = None
    execution_time_ms: float = 0.0

    def summary(self, max_length: int = 500) -> str:
        text = str(self.output) if self.success else (self.error or "")
        return text if len(text) <= max_length else text[: max_length - 1] + "…"


class Tool(ABC):
    """
    Base class for every tool. A concrete tool declares its metadata as
    class attributes and implements `run()`. `HostExecutor` is injected at
    construction time (see manager.py) rather than imported directly, so
    tools stay swappable/testable independent of how host actions actually
    happen (see backend/capabilities/tools/executor.py).
    """

    name: str
    description: str
    category: ToolCategory
    version: str = "1.0.0"
    parameters: list[ToolParameter] = []
    return_type: str = "object"
    requires_host: bool = False  # True for tools that need HostExecutor (filesystem/terminal/vscode)

    @abstractmethod
    async def run(self, params: dict) -> Any:
        """
        Execute the tool and return its raw output (JSON-serializable).
        Raise ToolExecutionError (or let a lower-level exception like
        SandboxViolationError propagate) on failure — the Tool Manager
        wraps whatever comes out of this into a ToolResult.
        """
        raise NotImplementedError

    def describe(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "version": self.version,
            "return_type": self.return_type,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                    "required": p.required,
                    "default": p.default,
                }
                for p in self.parameters
            ],
        }
