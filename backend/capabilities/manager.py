"""
Capability Manager — the central registry for everything E.D.I.T.H. can do.

This is the "broader Capability Architecture" the Tool Manager grew into:
executable tools and MCP-discovered tools are the first two capability
sources it wraps, with integrations/widgets/automation modules as future
sources (see backend/capabilities/{integrations,widgets,automation}/ —
empty today, structural placeholders only).

Deliberately thin: ToolManager remains the actual execution engine (tool
discovery, sandboxing, permission enforcement, logging all still live
there, untouched). CapabilityManager's job is presenting a single,
type-agnostic view over whatever capability sources exist, and routing
requests to whichever one owns a given capability id.
"""
from sqlalchemy.ext.asyncio import AsyncSession

from backend.capabilities.automation.event_bus import EventName, get_event_bus
from backend.capabilities.tools.base import ToolCategory, ToolResult
from backend.capabilities.tools.manager import ToolManager, get_tool_manager
from backend.capabilities.types import Capability, CapabilityType


def _capability_type_for_category(category: str) -> CapabilityType:
    return CapabilityType.MCP if category == ToolCategory.MCP.value else CapabilityType.TOOL


class CapabilityManager:
    """
    Wraps ToolManager (which already merges built-in tools + MCP tool
    proxies) as the TOOL/MCP capability sources. INTEGRATION/WIDGET/
    AUTOMATION sources have no implementation registered yet — once one
    exists, it plugs in here the same way ToolManager does, and
    list/execute/enable start including it automatically.
    """

    def __init__(self, tool_manager: ToolManager) -> None:
        self._tool_manager = tool_manager

    async def list_capabilities(self, session: AsyncSession) -> list[Capability]:
        tools = await self._tool_manager.list_tools(session)
        return [self._capability_from_tool(t) for t in tools]

    async def get_capability(self, session: AsyncSession, capability_id: str) -> Capability:
        tool = await self._tool_manager.get_tool_status(session, capability_id)
        return self._capability_from_tool(tool)

    async def execute_capability(
        self,
        session: AsyncSession,
        capability_id: str,
        params: dict,
        *,
        confirmed: bool = False,
        conversation_id=None,
        profile_id=None,
        initiated_by: str = "user",
        trigger_depth: int = 0,
    ) -> ToolResult:
        """
        Routes a capability invocation to the source that owns it. Today
        every registered capability is a TOOL or MCP capability, so this
        always delegates to ToolManager — the dispatch point exists so
        future capability types don't require touching call sites.

        conversation_id/profile_id/initiated_by are Phase 4 capability
        activity history — passed straight through to ToolManager's log
        write, purely for auditing (see backend/capabilities/tools/log_store.py).

        Phase 7: publishes CapabilityCompleted on the event bus whenever the
        capability actually ran (payload.success reflects whether it
        succeeded) — but ConfirmationRequiredError/PermissionDeniedError/
        ToolDisabledError all raise *before* the line below, since those
        mean the capability never ran at all. trigger_depth is forwarded
        into the published event so the Automation Engine can cap chains
        where one automation's action triggers another rule.
        """
        result = await self._tool_manager.execute_tool(
            session,
            capability_id,
            params,
            confirmed=confirmed,
            conversation_id=conversation_id,
            profile_id=profile_id,
            initiated_by=initiated_by,
        )
        await get_event_bus().publish(
            EventName.CAPABILITY_COMPLETED,
            {"capability_id": capability_id, "success": result.success, "initiated_by": initiated_by},
            trigger_depth=trigger_depth,
        )
        return result

    async def set_capability_enabled(
        self,
        session: AsyncSession,
        capability_id: str,
        *,
        enabled: bool | None = None,
        permission_policy: str | None = None,
    ) -> Capability:
        tool = await self._tool_manager.update_tool_permission(
            session, capability_id, enabled=enabled, permission_policy=permission_policy
        )
        return self._capability_from_tool(tool)

    async def refresh(self) -> None:
        """Re-runs discovery on every capability source that supports it."""
        await self._tool_manager.refresh()

    @staticmethod
    def _capability_from_tool(tool: dict) -> Capability:
        return Capability(
            id=tool["name"],
            type=_capability_type_for_category(tool["category"]),
            name=tool["name"],
            description=tool["description"],
            category=tool["category"],
            enabled=tool["enabled"],
            status=tool["status"],
            metadata={
                "version": tool["version"],
                "parameters": tool["parameters"],
                "return_type": tool["return_type"],
                "permission_policy": tool["permission_policy"],
            },
        )


_manager_instance: CapabilityManager | None = None


def get_capability_manager() -> CapabilityManager:
    """FastAPI dependency provider — single instance per process."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = CapabilityManager(get_tool_manager())
    return _manager_instance
