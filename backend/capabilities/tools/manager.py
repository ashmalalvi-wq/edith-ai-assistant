"""
Tool Manager — the single point of control for every tool EDITH can run,
whether it's a built-in tool or one discovered from an MCP server.

Responsibilities (per the Phase 3 design): discover tools, register them,
enable/disable, enforce permissions, execute, log, and report status. It
is deliberately independent of the AI model — every provider in
backend/ai/ would call this exact same manager, unmodified.
"""
import importlib
import inspect
import logging
import pkgutil
import time

from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.exceptions import (
    ConfirmationRequiredError,
    PermissionDeniedError,
    ToolDisabledError,
    ToolNotFoundError,
)
from backend.capabilities.tools import builtin as builtin_package
from backend.capabilities.tools import log_store, permission_store
from backend.capabilities.tools.base import PermissionPolicy, Tool, ToolResult
from backend.capabilities.tools.executor import HostExecutor

logger = logging.getLogger(__name__)

# Sensible defaults applied the first time a tool is discovered. After that,
# the row in tool_permissions is the source of truth — this only seeds it.
_DEFAULT_POLICIES: dict[str, PermissionPolicy] = {
    "filesystem.read_file": PermissionPolicy.ALWAYS_ALLOW,
    "filesystem.list_directory": PermissionPolicy.ALWAYS_ALLOW,
    "filesystem.search_directory": PermissionPolicy.ALWAYS_ALLOW,
    "filesystem.create_file": PermissionPolicy.ASK_EVERY_TIME,
    "filesystem.delete_file": PermissionPolicy.ASK_EVERY_TIME,
    "filesystem.move_file": PermissionPolicy.ASK_EVERY_TIME,
    "filesystem.rename_file": PermissionPolicy.ASK_EVERY_TIME,
    "filesystem.create_folder": PermissionPolicy.ASK_EVERY_TIME,
    "terminal.run_command": PermissionPolicy.ASK_EVERY_TIME,
    "vscode.open_workspace": PermissionPolicy.ALWAYS_ALLOW,
    "vscode.open_project": PermissionPolicy.ALWAYS_ALLOW,
    "vscode.open_file": PermissionPolicy.ALWAYS_ALLOW,
    "vscode.reveal_current_project": PermissionPolicy.ALWAYS_ALLOW,
    "vscode.create_file": PermissionPolicy.ASK_EVERY_TIME,
    "vscode.edit_file": PermissionPolicy.ASK_EVERY_TIME,
    "workspace.list_open_applications": PermissionPolicy.ALWAYS_ALLOW,
    "workspace.open_application": PermissionPolicy.ASK_EVERY_TIME,
    "workspace.close_application": PermissionPolicy.ASK_EVERY_TIME,
    "workspace.launch_set": PermissionPolicy.ASK_EVERY_TIME,
    "workspace.open_project": PermissionPolicy.ASK_EVERY_TIME,
    "workspace.resume_profile": PermissionPolicy.ASK_EVERY_TIME,
    "weather.current": PermissionPolicy.ALWAYS_ALLOW,
    "weather.forecast": PermissionPolicy.ALWAYS_ALLOW,
    "task.create": PermissionPolicy.ALWAYS_ALLOW,
    "task.complete": PermissionPolicy.ALWAYS_ALLOW,
    "task.prioritize": PermissionPolicy.ALWAYS_ALLOW,
    "task.list": PermissionPolicy.ALWAYS_ALLOW,
    "gmail.list_emails": PermissionPolicy.ASK_EVERY_TIME,
    "gmail.search_emails": PermissionPolicy.ASK_EVERY_TIME,
    "gmail.summarize_inbox": PermissionPolicy.ASK_EVERY_TIME,
    "gmail.create_draft": PermissionPolicy.ASK_EVERY_TIME,
    "calendar.view": PermissionPolicy.ASK_EVERY_TIME,
    "calendar.search_events": PermissionPolicy.ASK_EVERY_TIME,
    "calendar.summarize_schedule": PermissionPolicy.ASK_EVERY_TIME,
    "calendar.propose_times": PermissionPolicy.ASK_EVERY_TIME,
    "calendar.create_event": PermissionPolicy.ASK_EVERY_TIME,
    "calendar.modify_event": PermissionPolicy.ASK_EVERY_TIME,
    "calendar.delete_event": PermissionPolicy.ASK_EVERY_TIME,
    "notification.list": PermissionPolicy.ALWAYS_ALLOW,
    "notification.dismiss": PermissionPolicy.ALWAYS_ALLOW,
    "briefing.daily": PermissionPolicy.ALWAYS_ALLOW,
    "expert.analyze": PermissionPolicy.ASK_EVERY_TIME,
    "expert.implement": PermissionPolicy.ASK_EVERY_TIME,
    "git.status": PermissionPolicy.ALWAYS_ALLOW,
    "git.log": PermissionPolicy.ALWAYS_ALLOW,
    "git.branches": PermissionPolicy.ALWAYS_ALLOW,
    "git.diff": PermissionPolicy.ALWAYS_ALLOW,
    "git.commit": PermissionPolicy.ASK_EVERY_TIME,
    "git.push": PermissionPolicy.ASK_EVERY_TIME,
    "git.create_branch": PermissionPolicy.ASK_EVERY_TIME,
    "git.merge": PermissionPolicy.ASK_EVERY_TIME,
    "git.delete_branch": PermissionPolicy.ASK_EVERY_TIME,
    "printing.save_material_profile": PermissionPolicy.ALWAYS_ALLOW,
    "printing.list_materials": PermissionPolicy.ALWAYS_ALLOW,
    "printing.log_print_result": PermissionPolicy.ALWAYS_ALLOW,
    "printing.print_history": PermissionPolicy.ALWAYS_ALLOW,
    "research.save_resource": PermissionPolicy.ALWAYS_ALLOW,
    "research.search": PermissionPolicy.ALWAYS_ALLOW,
    "research.link_to_project": PermissionPolicy.ALWAYS_ALLOW,
    "engineering.create_project_structure": PermissionPolicy.ASK_EVERY_TIME,
    "engineering.project_status": PermissionPolicy.ALWAYS_ALLOW,
    "fusion.connect": PermissionPolicy.ALWAYS_ALLOW,
    "fusion.read_project_info": PermissionPolicy.ALWAYS_ALLOW,
    "fusion.list_components": PermissionPolicy.ALWAYS_ALLOW,
    "fusion.get_parameters": PermissionPolicy.ALWAYS_ALLOW,
    "fusion.export_file": PermissionPolicy.ASK_EVERY_TIME,
    "bambu.status": PermissionPolicy.ALWAYS_ALLOW,
    "bambu.filament_info": PermissionPolicy.ALWAYS_ALLOW,
    "automation.create_rule": PermissionPolicy.ASK_EVERY_TIME,
    "automation.create_from_description": PermissionPolicy.ALWAYS_ALLOW,
    "automation.update_rule": PermissionPolicy.ASK_EVERY_TIME,
    "automation.enable": PermissionPolicy.ASK_EVERY_TIME,
    "automation.disable": PermissionPolicy.ALWAYS_ALLOW,
    "automation.run_now": PermissionPolicy.ALWAYS_ALLOW,
    "automation.list_logs": PermissionPolicy.ALWAYS_ALLOW,
    # Opening/closing a floating widget has zero file/network/host effect —
    # same reasoning as task.list being always_allow.
    "widget.open": PermissionPolicy.ALWAYS_ALLOW,
    "widget.close": PermissionPolicy.ALWAYS_ALLOW,
    "widget.close_all": PermissionPolicy.ALWAYS_ALLOW,
}
_DEFAULT_MCP_POLICY = PermissionPolicy.ASK_EVERY_TIME


class ToolManager:
    def __init__(self, host_executor: HostExecutor) -> None:
        self._host_executor = host_executor
        self._builtin_tools: dict[str, Tool] = {}
        self._mcp_manager = None  # set via set_mcp_manager(); avoids a hard import-time dependency

    def set_mcp_manager(self, mcp_manager) -> None:
        self._mcp_manager = mcp_manager

    @property
    def builtin_tool_count(self) -> int:
        return len(self._builtin_tools)

    def discover_builtin_tools(self) -> None:
        """
        Scans backend/capabilities/tools/builtin/ for Tool subclasses and registers one
        instance of each. Adding a new built-in tool is just dropping a
        class in that package — no registration code to edit here.
        """
        self._builtin_tools.clear()
        for _, module_name, _ in pkgutil.iter_modules(builtin_package.__path__):
            module = importlib.import_module(f"{builtin_package.__name__}.{module_name}")
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, Tool)
                    and attr is not Tool
                    and attr.__module__ == module.__name__
                    and not inspect.isabstract(attr)
                ):
                    instance = attr(executor=self._host_executor)
                    self._builtin_tools[instance.name] = instance
                    logger.info("Registered built-in tool: %s", instance.name)

    def _all_tools(self) -> dict[str, Tool]:
        tools = dict(self._builtin_tools)
        if self._mcp_manager is not None:
            for tool in self._mcp_manager.get_tool_proxies():
                tools[tool.name] = tool
        return tools

    async def _ensure_permission(self, session: AsyncSession, tool: Tool):
        default_policy = _DEFAULT_POLICIES.get(tool.name, _DEFAULT_MCP_POLICY)
        return await permission_store.ensure_default_permission(
            session,
            tool_name=tool.name,
            category=tool.category.value,
            default_policy=default_policy.value,
        )

    async def list_tools(self, session: AsyncSession) -> list[dict]:
        """Merged view of every tool's metadata + current permission/status, for the dashboard."""
        results = []
        for tool in self._all_tools().values():
            permission = await self._ensure_permission(session, tool)
            status = await self._tool_status(tool)
            results.append(
                {
                    **tool.describe(),
                    "enabled": permission.enabled,
                    "permission_policy": permission.permission_policy,
                    "status": status,
                }
            )
        return results

    async def _tool_status(self, tool: Tool) -> str:
        if tool.requires_host:
            available = await self._host_executor.is_available()
            return "online" if available else "offline"
        return "online"

    async def get_tool_status(self, session: AsyncSession, tool_name: str) -> dict:
        tool = self._all_tools().get(tool_name)
        if tool is None:
            raise ToolNotFoundError(tool_name)
        permission = await self._ensure_permission(session, tool)
        return {
            **tool.describe(),
            "enabled": permission.enabled,
            "permission_policy": permission.permission_policy,
            "status": await self._tool_status(tool),
        }

    async def execute_tool(
        self,
        session: AsyncSession,
        tool_name: str,
        params: dict,
        *,
        confirmed: bool = False,
        conversation_id=None,
        profile_id=None,
        initiated_by: str = "user",
    ) -> ToolResult:
        tool = self._all_tools().get(tool_name)
        if tool is None:
            raise ToolNotFoundError(tool_name)

        permission = await self._ensure_permission(session, tool)

        if not permission.enabled:
            raise ToolDisabledError(tool_name)
        if permission.permission_policy == PermissionPolicy.NEVER_ALLOW.value:
            raise PermissionDeniedError(tool_name)
        if permission.permission_policy == PermissionPolicy.ASK_EVERY_TIME.value and not confirmed:
            raise ConfirmationRequiredError(tool_name)

        approval_status = (
            "not_required"
            if permission.permission_policy == PermissionPolicy.ALWAYS_ALLOW.value
            else "user_confirmed"
        )

        start = time.perf_counter()
        try:
            output = await tool.run(params)
            result = ToolResult(success=True, output=output, execution_time_ms=(time.perf_counter() - start) * 1000)
        except Exception as exc:  # noqa: BLE001 — tool errors must never crash the manager
            logger.exception("Tool '%s' failed", tool_name)
            result = ToolResult(
                success=False,
                error=str(exc),
                execution_time_ms=(time.perf_counter() - start) * 1000,
            )

        await log_store.create_log(
            session,
            tool_name=tool_name,
            category=tool.category.value,
            request_params=params,
            success=result.success,
            result_summary=result.summary() if result.success else None,
            error=result.error,
            execution_time_ms=result.execution_time_ms,
            conversation_id=conversation_id,
            profile_id=profile_id,
            initiated_by=initiated_by,
            approval_status=approval_status,
        )

        return result

    async def update_tool_permission(
        self,
        session: AsyncSession,
        tool_name: str,
        *,
        enabled: bool | None = None,
        permission_policy: str | None = None,
    ) -> dict:
        tool = self._all_tools().get(tool_name)
        if tool is None:
            raise ToolNotFoundError(tool_name)

        await self._ensure_permission(session, tool)  # make sure a row exists before updating
        permission = await permission_store.update_permission(
            session, tool_name, enabled=enabled, permission_policy=permission_policy
        )
        return {
            **tool.describe(),
            "enabled": permission.enabled,
            "permission_policy": permission.permission_policy,
            "status": await self._tool_status(tool),
        }

    async def refresh(self) -> None:
        """Re-run built-in discovery and reconnect MCP servers per current DB config."""
        self.discover_builtin_tools()
        if self._mcp_manager is not None:
            await self._mcp_manager.reload()


_manager_instance: ToolManager | None = None


def get_tool_manager() -> ToolManager:
    """FastAPI dependency provider — single instance per process."""
    global _manager_instance
    if _manager_instance is None:
        from backend.capabilities.tools.executor import get_host_executor

        _manager_instance = ToolManager(get_host_executor())
        _manager_instance.discover_builtin_tools()

        # Wired here (not at import time) to avoid capabilities.tools and
        # capabilities.mcp having a hard circular import on each other.
        from backend.capabilities.mcp.manager import get_mcp_manager

        _manager_instance.set_mcp_manager(get_mcp_manager())

    return _manager_instance
