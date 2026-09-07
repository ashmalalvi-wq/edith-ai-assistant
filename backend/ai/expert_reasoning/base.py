"""
Expert Reasoning Provider — the abstraction layer between E.D.I.T.H. and
whatever actually does the heavy-lifting reasoning/development work
(Claude Code today; Claude API, an OpenAI reasoning model, or a local
reasoning model are future implementations of this same interface).

Deliberately shaped like backend/ai/providers/base.py's AIProvider
(is_available() + one call method) — same "interface before
implementations" principle, so a new provider is a new module + one line
in a registry, never a change to the tools that call it.

This is NOT part of the normal chat path. AIModelRouter/AIProviderName
(ollama/openai/anthropic) are untouched — this is what
capabilities/tools/builtin/expert_tool.py delegates to, via the ordinary
Capability Manager, when E.D.I.T.H. needs expert-level reasoning or
development help instead of a normal chat reply.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ExpertFileChange:
    path: str
    change_type: str  # "created" | "modified" | "deleted"


@dataclass
class ExpertTaskRequest:
    """
    One delegated task. `mode` controls how much autonomy the provider is
    given — "analyze" must never touch disk; "implement" may, but only
    after E.D.I.T.H.'s own permission system has already required
    approval for the call that produced this request.
    """

    task: str
    mode: str  # "analyze" | "implement"
    working_directory: str  # absolute path; providers must sandbox this themselves
    context: str = ""  # assembled project/profile/memory context (see expert_tool.py)
    model: str | None = None


@dataclass
class ExpertTaskResult:
    success: bool
    summary: str
    file_changes: list[ExpertFileChange] = field(default_factory=list)
    cost_usd: float | None = None
    session_id: str | None = None
    error: str | None = None


class ExpertReasoningProvider(ABC):
    """Base class for every expert reasoning backend (Claude Code, Claude API, ...)."""

    name: str

    @abstractmethod
    async def is_available(self) -> tuple[bool, str | None]:
        """Whether this provider can currently be used. Returns (available, detail_message)."""
        raise NotImplementedError

    @abstractmethod
    async def run_task(self, request: ExpertTaskRequest) -> ExpertTaskResult:
        """Runs one delegated task and returns its result. Must never crash — failures become ExpertTaskResult(success=False)."""
        raise NotImplementedError

    async def run_task_streaming(self, request: ExpertTaskRequest, request_id: str, event_bus) -> ExpertTaskResult:
        """
        Like run_task, but publishes incremental text chunks to `event_bus`
        (EventName.EXPERT_STREAM_CHUNK, payload {"request_id": request_id,
        "text": delta}) as they become available, for callers that want to
        forward output live instead of waiting for the whole result.

        Default implementation: no real streaming, just runs the whole task
        and publishes it as a single chunk — good enough for a provider that
        doesn't have a streaming-capable backend (same "not every provider
        needs every capability" pattern as supports_tools() on AIProvider).
        ClaudeCodeProvider overrides this with a real implementation.
        """
        from backend.capabilities.automation.event_bus import EventName

        result = await self.run_task(request)
        if result.summary:
            await event_bus.publish(EventName.EXPERT_STREAM_CHUNK, {"request_id": request_id, "text": result.summary})
        return result
