"""
Claude Code provider — the first real ExpertReasoningProvider. Shells out
to the `claude` CLI via the existing HostExecutor (same object
terminal.run_command/vscode.* already use), so there is no new subprocess-
management code and the existing host-mode/sandbox boundaries apply
unmodified.

Verified against the real CLI during development (this machine has
`claude` installed and authenticated): `claude --output-format json
--permission-mode plan -p -- "<prompt>"` returns a JSON object shaped
like `{"type": "result", "subtype": "success", "is_error": bool,
"result": "<text>", "total_cost_usd": float, "session_id": str,
"permission_denials": [...]}`. This module's parsing is written against
that real schema, not guessed.

Session continuity: each run_task() call is still its own one-shot `claude`
process — a genuinely persistent terminal was considered and rejected.
`claude --remote-control` (Claude Code's real feature for viewing/driving
a session from the claude.ai web/mobile app) needs an actual interactive
TTY; confirmed live that it fails immediately without one
("Input must be provided either through stdin or as a prompt argument"),
and it's designed for a human controlling their own session remotely, not
for a backend to automate. Instead this reuses one Claude Code
*conversation* across every Expert Mode call for the life of this backend
process: the first call pins `--session-id <uuid>` (generated once here),
every later call passes `--resume <that same uuid>` instead — both flags
verified live against the real CLI, including that switching
--permission-mode between plan/acceptEdits mid-conversation on a resumed
session works fine (Claude Code just recalls prior context and applies the
new mode). If a call fails outright (non-zero exit, timeout, unparseable
output) the tracked session id is dropped so the next call starts a fresh
session rather than repeatedly resuming a session that may not have been
created (an established session with a normal Claude-side error, e.g. from
a malformed task, is not dropped — only outright process/parse failures).
"""
import asyncio
import json
import logging
import uuid
from pathlib import Path

from backend.capabilities.automation.event_bus import EventBus, EventName
from backend.capabilities.tools.executor import HostExecutor
from backend.capabilities.tools.sandbox import resolve_within_sandbox
from backend.config.settings import get_settings
from backend.ai.expert_reasoning.base import (
    ExpertFileChange,
    ExpertReasoningProvider,
    ExpertTaskRequest,
    ExpertTaskResult,
)

logger = logging.getLogger(__name__)

# The only permission modes E.D.I.T.H. will ever pass to Claude Code.
# "bypassPermissions" is deliberately never in this map — a prompt-injected
# task can never escalate to it, because this module has no code path that
# accepts an arbitrary mode string from the caller.
_PERMISSION_MODE_FOR_TASK_MODE = {
    "analyze": "plan",
    "implement": "acceptEdits",
}


def _snapshot(directory: Path) -> dict[str, float]:
    """path -> mtime for every file under `directory`, used to diff before/after an implement run."""
    if not directory.exists():
        return {}
    return {str(p): p.stat().st_mtime for p in directory.rglob("*") if p.is_file()}


def _diff_snapshots(before: dict[str, float], after: dict[str, float]) -> list[ExpertFileChange]:
    changes = []
    for path in after.keys() - before.keys():
        changes.append(ExpertFileChange(path=path, change_type="created"))
    for path in before.keys() - after.keys():
        changes.append(ExpertFileChange(path=path, change_type="deleted"))
    for path in before.keys() & after.keys():
        if before[path] != after[path]:
            changes.append(ExpertFileChange(path=path, change_type="modified"))
    return changes


class ClaudeCodeProvider(ExpertReasoningProvider):
    name = "claude_code"

    def __init__(self, executor: HostExecutor) -> None:
        self._executor = executor
        # One Claude Code conversation reused for every Expert Mode call for
        # the life of this backend process (see module docstring) — None
        # until the first successful call establishes it.
        self._session_id: str | None = None

    async def is_available(self) -> tuple[bool, str | None]:
        if not await self._executor.is_available():
            return False, "Host mode is disabled — Claude Code requires native (non-Docker) execution."
        # A full check would run `claude --version`; that's a real subprocess
        # spawn for what's meant to be a cheap status check, so this only
        # confirms host access. Actual CLI/auth failures surface at run_task()
        # time as a failed ExpertTaskResult, same as ProviderNotAvailableError
        # for a chat provider whose API key turns out to be invalid.
        return True, None

    def _build_base_command(self, request: ExpertTaskRequest, permission_mode: str) -> list[str]:
        """Everything up to (not including) the output-format flags and the
        prompt itself — shared by the one-shot and streaming call paths."""
        settings = get_settings()
        model = request.model or settings.claude_code_default_model

        command = [settings.claude_code_command, "--permission-mode", permission_mode]
        if model:
            command.extend(["--model", model])

        # First call this process pins a session id; every later call
        # resumes it, so Claude Code sees one continuous conversation
        # across Expert Mode questions instead of starting cold each time.
        # Tracked optimistically and cleared on any outright process
        # failure below (timeout, non-zero exit, unparseable output) —
        # whether that failure was on the pinning call or a resume, the
        # session can't be trusted to exist, so the next call starts fresh
        # rather than repeatedly erroring against a dead session id.
        resuming = self._session_id is not None
        session_id = self._session_id or str(uuid.uuid4())
        command.extend(["--resume" if resuming else "--session-id", session_id])
        self._session_id = session_id

        return command

    @staticmethod
    def _append_prompt(command: list[str], request: ExpertTaskRequest) -> None:
        prompt = f"{request.context}\n\n{request.task}".strip() if request.context else request.task
        # `-p` then `--` then the prompt, always — a real bug found while
        # verifying session continuity: build_expert_context()'s memory
        # block starts with a literal "--- E.D.I.T.H memory context ---"
        # divider, and Claude Code's CLI parser (commander.js-based) treats
        # any positional argument starting with "-" as an unknown option
        # rather than a value, unless "--" marks the end of options first.
        # Confirmed live: without "--" this failed with
        # `error: unknown option '--- E.D.I.T.H memory context...'` every
        # time retrieved memory content happened to be non-empty; with "--"
        # placed right before the prompt it's correctly treated as
        # positional text no matter what it starts with. Not hypothetical —
        # this was actively breaking every Expert Mode call that had any
        # relevant memory to inject.
        command.extend(["-p", "--", prompt])

    def _build_result(
        self, payload: dict, request: ExpertTaskRequest, before: dict[str, float], resolved_dir: Path
    ) -> ExpertTaskResult:
        file_changes = []
        if request.mode == "implement":
            after = _snapshot(resolved_dir)
            file_changes = _diff_snapshots(before, after)

        return ExpertTaskResult(
            success=not payload.get("is_error", False),
            summary=payload.get("result", ""),
            file_changes=file_changes,
            cost_usd=payload.get("total_cost_usd"),
            session_id=payload.get("session_id"),
            error=None if not payload.get("is_error") else payload.get("result"),
        )

    async def run_task(self, request: ExpertTaskRequest) -> ExpertTaskResult:
        if request.mode not in _PERMISSION_MODE_FOR_TASK_MODE:
            return ExpertTaskResult(success=False, summary="", error=f"Unknown expert task mode '{request.mode}'.")

        settings = get_settings()
        permission_mode = _PERMISSION_MODE_FOR_TASK_MODE[request.mode]
        command = self._build_base_command(request, permission_mode)
        command[1:1] = ["--output-format", "json"]
        self._append_prompt(command, request)

        resolved_dir = resolve_within_sandbox(request.working_directory)
        before = _snapshot(resolved_dir) if request.mode == "implement" else {}

        result = await self._executor.run_process(
            command, cwd=str(resolved_dir), timeout=settings.claude_code_timeout_seconds
        )

        if result.timed_out:
            self._session_id = None
            return ExpertTaskResult(success=False, summary="", error="Claude Code task timed out.")
        if result.exit_code != 0:
            self._session_id = None
            return ExpertTaskResult(
                success=False, summary="", error=result.stderr.strip() or f"Claude Code exited with {result.exit_code}."
            )
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            logger.warning("Claude Code output was not valid JSON: %s", result.stdout[:200])
            self._session_id = None
            return ExpertTaskResult(success=False, summary="", error="Could not parse Claude Code's output.")

        return self._build_result(payload, request, before, resolved_dir)

    async def run_task_streaming(
        self, request: ExpertTaskRequest, request_id: str, event_bus: EventBus
    ) -> ExpertTaskResult:
        if request.mode not in _PERMISSION_MODE_FOR_TASK_MODE:
            return ExpertTaskResult(success=False, summary="", error=f"Unknown expert task mode '{request.mode}'.")

        settings = get_settings()
        permission_mode = _PERMISSION_MODE_FOR_TASK_MODE[request.mode]
        command = self._build_base_command(request, permission_mode)
        # --include-partial-messages is what actually produces incremental
        # text_delta events on stream-json output (without it, stream-json
        # still emits one line per whole message, no finer-grained than
        # --output-format json); --verbose is required by the CLI whenever
        # --print + --output-format=stream-json are combined. Verified live
        # against the real CLI with this exact flag set (including
        # --resume/--session-id and --permission-mode) before writing this.
        command[1:1] = ["--output-format", "stream-json", "--include-partial-messages", "--verbose"]
        self._append_prompt(command, request)

        resolved_dir = resolve_within_sandbox(request.working_directory)
        before = _snapshot(resolved_dir) if request.mode == "implement" else {}

        final_payload: dict | None = None
        try:
            async for line in self._executor.stream_process(
                command, cwd=str(resolved_dir), timeout=settings.claude_code_timeout_seconds
            ):
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type")
                if event_type == "stream_event":
                    inner = event.get("event", {})
                    if inner.get("type") == "content_block_delta":
                        delta = inner.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text = delta.get("text", "")
                            if text:
                                await event_bus.publish(
                                    EventName.EXPERT_STREAM_CHUNK, {"request_id": request_id, "text": text}
                                )
                elif event_type == "result":
                    final_payload = event
        except asyncio.TimeoutError:
            self._session_id = None
            return ExpertTaskResult(success=False, summary="", error="Claude Code task timed out.")

        if final_payload is None:
            logger.warning("Claude Code stream ended without a final 'result' event.")
            self._session_id = None
            return ExpertTaskResult(success=False, summary="", error="Could not parse Claude Code's output.")

        return self._build_result(final_payload, request, before, resolved_dir)
