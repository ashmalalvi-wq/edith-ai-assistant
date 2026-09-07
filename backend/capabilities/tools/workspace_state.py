"""
Workspace State — in-process registry of applications E.D.I.T.H has
launched, so `workspace.close_application` only ever terminates something
E.D.I.T.H itself started (never an arbitrary PID the user passes in), and
`workspace.list_open_applications` / GET /api/workspaces/state have
something to report.

Deliberately in-memory only (like HostExecutor's singleton) — it resets on
backend restart, which is correct: a fresh process has no idea what's
"open" until it launches something itself. Restoring a *previous* session's
apps is what workspace.resume_profile is for, not this registry.
"""
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class LaunchedApp:
    id: str
    name: str
    command: list[str]
    pid: int
    profile_id: str | None
    launched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class WorkspaceState:
    def __init__(self) -> None:
        self._apps: dict[str, LaunchedApp] = {}

    def register(self, *, name: str, command: list[str], pid: int, profile_id: str | None = None) -> LaunchedApp:
        app = LaunchedApp(id=str(uuid.uuid4()), name=name, command=command, pid=pid, profile_id=profile_id)
        self._apps[app.id] = app
        return app

    def unregister(self, app_id: str) -> LaunchedApp | None:
        return self._apps.pop(app_id, None)

    def find_by_id(self, app_id: str) -> LaunchedApp | None:
        return self._apps.get(app_id)

    def find_by_pid(self, pid: int) -> LaunchedApp | None:
        return next((a for a in self._apps.values() if a.pid == pid), None)

    def find_by_name(self, name: str) -> LaunchedApp | None:
        return next((a for a in self._apps.values() if a.name == name), None)

    def list_apps(self) -> list[LaunchedApp]:
        return list(self._apps.values())


_state_instance: WorkspaceState | None = None


def get_workspace_state() -> WorkspaceState:
    """FastAPI dependency provider — single instance per process."""
    global _state_instance
    if _state_instance is None:
        _state_instance = WorkspaceState()
    return _state_instance
