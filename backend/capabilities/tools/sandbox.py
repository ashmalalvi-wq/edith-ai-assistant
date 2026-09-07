"""
Filesystem sandbox — the single place that decides whether a path is safe
to touch. Used by the filesystem tool, the terminal tool (working
directory), and the VS Code tool (file-writing actions).

Two layers, both must pass:
  1. The resolved path must fall inside one of the configured approved
     directories (backend/config/settings.py: APPROVED_DIRECTORIES).
  2. It must not fall inside a hardcoded blocklist of OS-critical paths,
     even if a misconfigured approved directory were too broad.
"""
from pathlib import Path

from backend.config.settings import get_settings
from backend.core.exceptions import SandboxViolationError

# Second safety layer — checked even if a path technically falls under an
# approved directory, in case that directory was configured too broadly.
_BLOCKED_PREFIXES = [
    r"C:\Windows",
    r"C:\Program Files",
    r"C:\Program Files (x86)",
    r"C:\ProgramData",
    "/etc",
    "/bin",
    "/sbin",
    "/usr/bin",
    "/usr/sbin",
    "/System",
]


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_within_sandbox(raw_path: str) -> Path:
    """
    Resolves `raw_path` to an absolute path and validates it against the
    sandbox. Returns the resolved Path on success, raises
    SandboxViolationError otherwise.
    """
    settings = get_settings()
    resolved = Path(raw_path).expanduser().resolve()

    for blocked in _BLOCKED_PREFIXES:
        if _is_within(resolved, Path(blocked)):
            raise SandboxViolationError(str(resolved))

    approved_roots = [Path(d).expanduser().resolve() for d in settings.approved_directories_list]
    if not any(_is_within(resolved, root) for root in approved_roots):
        raise SandboxViolationError(str(resolved))

    return resolved


def is_within_sandbox(raw_path: str) -> bool:
    """Non-raising check, useful for status/preview UIs."""
    try:
        resolve_within_sandbox(raw_path)
        return True
    except SandboxViolationError:
        return False
