"""
HostExecutor — the abstraction seam between tools and the actual host
operations they perform.

Tools (filesystem/terminal/vscode) never call `open()`, `shutil`, or
`subprocess` directly — they call methods on an injected HostExecutor.
Phase 3 ships exactly one implementation, LocalHostExecutor, which runs
these operations directly on the machine the backend process is on. A
future "Host Tool Bridge" (a companion agent a Dockerized backend could
delegate to over RPC) can implement the same interface without any tool
or Tool Manager code changing.
"""
import asyncio
import csv
import io
import os
import shutil
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from backend.config.settings import get_settings
from backend.core.exceptions import HostUnavailableError, ToolExecutionError
from backend.capabilities.tools.sandbox import resolve_within_sandbox


@dataclass
class FileEntry:
    name: str
    path: str
    is_directory: bool
    size_bytes: int | None = None


@dataclass
class ProcessResult:
    stdout: str
    stderr: str
    exit_code: int | None
    duration_ms: float
    timed_out: bool


@dataclass
class LaunchedProcess:
    pid: int


class HostExecutor(ABC):
    @abstractmethod
    async def is_available(self) -> bool:
        """Whether this executor can currently perform host actions."""
        raise NotImplementedError

    @abstractmethod
    async def read_file(self, path: str) -> str: ...

    @abstractmethod
    async def write_file(self, path: str, content: str) -> None: ...

    @abstractmethod
    async def delete_path(self, path: str) -> None: ...

    @abstractmethod
    async def move_path(self, source: str, destination: str) -> None: ...

    @abstractmethod
    async def make_directory(self, path: str) -> None: ...

    @abstractmethod
    async def list_directory(self, path: str) -> list[FileEntry]: ...

    @abstractmethod
    async def search_directory(self, path: str, pattern: str) -> list[str]: ...

    @abstractmethod
    async def run_process(self, command: list[str], cwd: str, timeout: float) -> ProcessResult: ...

    @abstractmethod
    def stream_process(self, command: list[str], cwd: str, timeout: float) -> AsyncIterator[str]:
        """
        Like run_process, but yields decoded stdout lines as they arrive
        instead of waiting for the process to exit and returning everything
        at once — for callers (e.g. ClaudeCodeProvider's streaming path) that
        need to forward output incrementally. Still enforces `timeout` across
        the whole run, same as run_process; still stdout only, same as
        run_process's ProcessResult (stderr isn't streamed, only captured on
        the non-streaming path).
        """
        ...

    @abstractmethod
    async def launch_application(self, command: list[str], cwd: str) -> LaunchedProcess:
        """
        Starts a process and returns immediately (unlike run_process, which
        waits for it to exit) — for launching desktop applications the user
        wants left running, not commands whose output we need.
        """
        ...

    @abstractmethod
    async def terminate_process(self, pid: int) -> bool:
        """Terminates a process previously started by launch_application. Returns False if it's already gone."""
        ...


class LocalHostExecutor(HostExecutor):
    """Runs host operations directly on the machine the backend is on."""

    async def is_available(self) -> bool:
        return get_settings().host_mode

    def _require_host_mode(self) -> None:
        if not get_settings().host_mode:
            raise HostUnavailableError("filesystem/terminal/vscode")

    async def read_file(self, path: str) -> str:
        self._require_host_mode()
        target = resolve_within_sandbox(path)
        return await asyncio.to_thread(target.read_text, encoding="utf-8")

    async def write_file(self, path: str, content: str) -> None:
        self._require_host_mode()
        target = resolve_within_sandbox(path)

        def _write():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

        await asyncio.to_thread(_write)

    async def delete_path(self, path: str) -> None:
        self._require_host_mode()
        target = resolve_within_sandbox(path)

        def _delete():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()

        await asyncio.to_thread(_delete)

    async def move_path(self, source: str, destination: str) -> None:
        self._require_host_mode()
        src = resolve_within_sandbox(source)
        dest = resolve_within_sandbox(destination)
        await asyncio.to_thread(shutil.move, str(src), str(dest))

    async def make_directory(self, path: str) -> None:
        self._require_host_mode()
        target = resolve_within_sandbox(path)
        await asyncio.to_thread(target.mkdir, parents=True, exist_ok=True)

    async def list_directory(self, path: str) -> list[FileEntry]:
        self._require_host_mode()
        target = resolve_within_sandbox(path)

        def _list():
            entries = []
            for entry in sorted(target.iterdir(), key=lambda p: p.name.lower()):
                entries.append(
                    FileEntry(
                        name=entry.name,
                        path=str(entry),
                        is_directory=entry.is_dir(),
                        size_bytes=entry.stat().st_size if entry.is_file() else None,
                    )
                )
            return entries

        return await asyncio.to_thread(_list)

    async def search_directory(self, path: str, pattern: str) -> list[str]:
        self._require_host_mode()
        target = resolve_within_sandbox(path)

        def _search():
            return [str(p) for p in target.rglob(pattern)]

        return await asyncio.to_thread(_search)

    async def run_process(self, command: list[str], cwd: str, timeout: float) -> ProcessResult:
        self._require_host_mode()
        working_dir = resolve_within_sandbox(cwd)

        loop = asyncio.get_event_loop()
        start = loop.time()

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(working_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            raise ToolExecutionError("terminal.run_command", str(exc)) from exc

        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            timed_out = False
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            stdout, stderr = b"", b""
            timed_out = True

        duration_ms = (loop.time() - start) * 1000
        return ProcessResult(
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
            exit_code=process.returncode,
            duration_ms=duration_ms,
            timed_out=timed_out,
        )

    async def stream_process(self, command: list[str], cwd: str, timeout: float) -> AsyncIterator[str]:
        self._require_host_mode()
        working_dir = resolve_within_sandbox(cwd)

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(working_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            raise ToolExecutionError("terminal.run_command", str(exc)) from exc

        async def _drain_stderr() -> None:
            # Nothing reads this back — stream_process callers only need
            # stdout deltas, and a failing process reports its own error via
            # the caller's own output parsing (see claude_code.py's
            # run_task_streaming, which trusts the final stream-json "result"
            # line the same way run_task() trusts run_process()'s stdout).
            # This loop exists only so the OS pipe buffer never fills and
            # deadlocks the process — the same risk communicate() avoids for
            # run_process, just handled manually here since nothing else
            # drains stderr while stdout is being read incrementally.
            while True:
                chunk = await process.stderr.read(4096)
                if not chunk:
                    return

        stderr_task = asyncio.ensure_future(_drain_stderr())
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout

        try:
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    process.kill()
                    await process.wait()
                    raise asyncio.TimeoutError
                try:
                    line = await asyncio.wait_for(process.stdout.readline(), timeout=remaining)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
                    raise
                if not line:
                    break
                yield line.decode(errors="replace")
            await process.wait()
        finally:
            stderr_task.cancel()

    async def launch_application(self, command: list[str], cwd: str) -> LaunchedProcess:
        self._require_host_mode()
        working_dir = resolve_within_sandbox(cwd)
        image_name = os.path.basename(command[0])

        # Some Windows apps — notably MSIX/Store-packaged ones like Windows
        # 11's modern Notepad — ship a launcher stub at the well-known path
        # (e.g. System32\notepad.exe) that re-execs into the real process
        # under a *different* PID and exits within milliseconds. Snapshot
        # PIDs with this image name before launching so a stub hand-off can
        # be detected and resolved to the real PID below, rather than
        # tracking a PID that's already gone.
        before_pids = await self._list_pids_by_image_name(image_name)

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(working_dir),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                stdin=asyncio.subprocess.DEVNULL,
            )
        except OSError as exc:
            raise ToolExecutionError("workspace.open_application", str(exc)) from exc

        resolved_pid = await self._resolve_real_pid(process.pid, image_name, before_pids)
        return LaunchedProcess(pid=resolved_pid)

    async def _resolve_real_pid(self, pid: int, image_name: str, before_pids: set[int]) -> int:
        # A genuinely long-running app is still alive well before a stub
        # would have exited (stubs exit near-instantly) — one short wait is
        # enough to tell the two cases apart without adding much latency to
        # the common (non-stub) case.
        await asyncio.sleep(0.4)
        if await self._process_exists(pid):
            return pid

        after_pids = await self._list_pids_by_image_name(image_name)
        new_pids = after_pids - before_pids
        if len(new_pids) == 1:
            return new_pids.pop()
        if new_pids:
            # Ambiguous (multiple new instances of the same app appeared in
            # the same window) — best-effort: the real process almost always
            # gets the higher PID, since it's created after the stub.
            return max(new_pids)
        # No replacement found — return the original, now-dead PID. Callers
        # (close_application/list_open_applications) will see it as already
        # gone, which is honest even though it's not the actually-running app.
        return pid

    async def _list_pids_by_image_name(self, image_name: str) -> set[int]:
        try:
            process = await asyncio.create_subprocess_exec(
                "tasklist", "/FI", f"IMAGENAME eq {image_name}", "/FO", "CSV", "/NH",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await process.communicate()
        except OSError:
            return set()

        pids = set()
        for row in csv.reader(io.StringIO(stdout.decode(errors="replace"))):
            if len(row) >= 2:
                try:
                    pids.add(int(row[1]))
                except ValueError:
                    continue
        return pids

    async def _process_exists(self, pid: int) -> bool:
        try:
            process = await asyncio.create_subprocess_exec(
                "tasklist", "/FI", f"PID eq {pid}", "/NH",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await process.communicate()
        except OSError:
            return False
        return str(pid).encode() in stdout

    async def terminate_process(self, pid: int) -> bool:
        self._require_host_mode()
        try:
            import os
            import signal

            os.kill(pid, signal.SIGTERM)
            return True
        except (ProcessLookupError, PermissionError, OSError):
            return False


_executor_instance: LocalHostExecutor | None = None


def get_host_executor() -> HostExecutor:
    """FastAPI dependency provider — single instance per process."""
    global _executor_instance
    if _executor_instance is None:
        _executor_instance = LocalHostExecutor()
    return _executor_instance
