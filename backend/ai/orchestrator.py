"""
Conversation Orchestrator — the AI subsystem's entry point for "turn a
message list into a provider reply."

Phase 1-3: a thin pass-through to AIModelRouter.chat() (still available as
`.chat()`, used by callers that don't need capabilities).

Phase 4: `run_turn()` is the capability-calling loop — it lets the model
request capabilities (via native tool-calling, see capability_reasoning.py),
executes the ones the permission system allows outright, and stops to
surface a PendingApproval for anything that needs confirmation rather than
ever bypassing the existing permission system. The caller (MemoryPipeline)
persists/represents that pending state; `resolve_pending()` continues the
turn once the user has approved or denied it.
"""
import logging
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from backend.ai.capability_keywords import select_relevant_capabilities
from backend.ai.capability_reasoning import CapabilityReasoner, get_capability_reasoner
from backend.ai.router import AIModelRouter, get_ai_router
from backend.core.exceptions import (
    ConfirmationRequiredError,
    PermissionDeniedError,
    ToolDisabledError,
    ToolNotFoundError,
)
from backend.models.schemas import AIProviderName, ChatMessage, ChatRole, ToolCallRequest

logger = logging.getLogger(__name__)

_MAX_CAPABILITY_ITERATIONS = 5


@dataclass
class CapabilityCallOutcome:
    call_id: str
    capability_id: str
    params: dict
    status: str  # "executed" | "pending_approval" | "denied" | "disabled" | "error"
    result: Any = None
    error: str | None = None


@dataclass
class PendingApproval:
    call_id: str
    capability_id: str
    params: dict


@dataclass
class ConversationTurnResult:
    provider: AIProviderName
    model: str
    reply: ChatMessage
    messages: list[ChatMessage]  # full history including any tool exchanges, for persistence
    capability_calls: list[CapabilityCallOutcome] = field(default_factory=list)
    pending_approval: PendingApproval | None = None


class ConversationOrchestrator:
    def __init__(self, ai_router: AIModelRouter, reasoner: CapabilityReasoner | None = None) -> None:
        self._ai_router = ai_router
        self._reasoner = reasoner or get_capability_reasoner()

    async def chat(
        self,
        messages: list[ChatMessage],
        provider: AIProviderName | None = None,
        model: str | None = None,
    ) -> tuple[AIProviderName, str, ChatMessage]:
        """Capability-free pass-through, for callers that don't need the loop below."""
        return await self._ai_router.chat(messages=messages, provider=provider, model=model)

    async def run_turn(
        self,
        session: AsyncSession,
        messages: list[ChatMessage],
        *,
        provider: AIProviderName | None = None,
        model: str | None = None,
        capability_manager=None,
        conversation_id=None,
        profile_id=None,
    ) -> ConversationTurnResult:
        tools = None
        if capability_manager is not None and self._ai_router.provider_supports_tools(provider):
            capabilities = await capability_manager.list_capabilities(session)
            relevant = select_relevant_capabilities(messages, capabilities)
            if relevant:
                tools = self._reasoner.build_tool_schemas(relevant)

        working_messages = list(messages)
        capability_calls: list[CapabilityCallOutcome] = []

        for _ in range(_MAX_CAPABILITY_ITERATIONS):
            used_provider, used_model, reply = await self._ai_router.chat(
                messages=working_messages, provider=provider, model=model, tools=tools
            )

            requested = self._reasoner.extract_requested_calls(reply) if tools else []
            if not requested:
                if not (reply.content or "").strip():
                    # Small local models occasionally emit a blank completion
                    # with no tool call requested (observed live with
                    # llama3.1:8b). Retry once before persisting a reply the
                    # user would just see as "nothing" — same "fail soft, log
                    # loud" principle used elsewhere rather than a silent gap.
                    logger.warning("Empty model reply with no tool calls requested; retrying once.")
                    used_provider, used_model, reply = await self._ai_router.chat(
                        messages=working_messages, provider=provider, model=model, tools=tools
                    )
                    if not (reply.content or "").strip():
                        logger.warning("Model reply still empty after retry; substituting fallback text.")
                        reply = ChatMessage(
                            role=ChatRole.ASSISTANT,
                            content="I didn't generate a response to that — could you try rephrasing or asking again?",
                        )
                working_messages.append(reply)
                return ConversationTurnResult(
                    provider=used_provider,
                    model=used_model,
                    reply=reply,
                    messages=working_messages,
                    capability_calls=capability_calls,
                )

            working_messages.append(reply)

            pending = None
            for call in requested:
                outcome = await self._execute_call(
                    session, capability_manager, call, conversation_id=conversation_id, profile_id=profile_id
                )
                capability_calls.append(outcome)

                if outcome.status == "pending_approval":
                    pending = PendingApproval(
                        call_id=call.id, capability_id=call.name, params=call.arguments
                    )
                    break  # ask about this one before considering any further calls this turn

                working_messages.append(
                    ChatMessage(
                        role=ChatRole.TOOL, content=self._describe_outcome(outcome), tool_call_id=call.id
                    )
                )

            if pending is not None:
                return ConversationTurnResult(
                    provider=used_provider,
                    model=used_model,
                    reply=reply,
                    messages=working_messages,
                    capability_calls=capability_calls,
                    pending_approval=pending,
                )
            # Otherwise loop again so the model sees the tool results and can respond/chain further.

        logger.warning("Capability-calling loop hit the %d-iteration cap; returning as-is.", _MAX_CAPABILITY_ITERATIONS)
        return ConversationTurnResult(
            provider=used_provider,
            model=used_model,
            reply=reply,
            messages=working_messages,
            capability_calls=capability_calls,
        )

    async def resolve_pending(
        self,
        session: AsyncSession,
        messages: list[ChatMessage],
        pending: PendingApproval,
        *,
        approved: bool,
        provider: AIProviderName | None = None,
        model: str | None = None,
        capability_manager=None,
        conversation_id=None,
        profile_id=None,
    ) -> ConversationTurnResult:
        """Executes (or records the denial of) a previously-pending capability call, then continues the turn."""
        if approved and capability_manager is not None:
            try:
                result = await capability_manager.execute_capability(
                    session,
                    pending.capability_id,
                    pending.params,
                    confirmed=True,
                    conversation_id=conversation_id,
                    profile_id=profile_id,
                    initiated_by="ai",
                )
                outcome = CapabilityCallOutcome(
                    call_id=pending.call_id,
                    capability_id=pending.capability_id,
                    params=pending.params,
                    status="executed" if result.success else "error",
                    result=result.output,
                    error=result.error,
                )
            except (ToolDisabledError, PermissionDeniedError, ToolNotFoundError) as exc:
                outcome = CapabilityCallOutcome(
                    call_id=pending.call_id,
                    capability_id=pending.capability_id,
                    params=pending.params,
                    status="error",
                    error=str(exc),
                )
        else:
            outcome = CapabilityCallOutcome(
                call_id=pending.call_id,
                capability_id=pending.capability_id,
                params=pending.params,
                status="denied",
                error="Denied by user.",
            )

        working_messages = [
            *messages,
            ChatMessage(role=ChatRole.TOOL, content=self._describe_outcome(outcome), tool_call_id=pending.call_id),
        ]

        continuation = await self.run_turn(
            session,
            working_messages,
            provider=provider,
            model=model,
            capability_manager=capability_manager,
            conversation_id=conversation_id,
            profile_id=profile_id,
        )
        continuation.capability_calls = [outcome, *continuation.capability_calls]
        return continuation

    @staticmethod
    async def _execute_call(
        session: AsyncSession,
        capability_manager,
        call: ToolCallRequest,
        *,
        conversation_id=None,
        profile_id=None,
    ) -> CapabilityCallOutcome:
        if capability_manager is None:
            return CapabilityCallOutcome(
                call_id=call.id,
                capability_id=call.name,
                params=call.arguments,
                status="error",
                error="Capabilities are not available in this context.",
            )
        try:
            result = await capability_manager.execute_capability(
                session,
                call.name,
                call.arguments,
                confirmed=False,
                conversation_id=conversation_id,
                profile_id=profile_id,
                initiated_by="ai",
            )
            return CapabilityCallOutcome(
                call_id=call.id,
                capability_id=call.name,
                params=call.arguments,
                status="executed" if result.success else "error",
                result=result.output,
                error=result.error,
            )
        except ConfirmationRequiredError:
            return CapabilityCallOutcome(
                call_id=call.id, capability_id=call.name, params=call.arguments, status="pending_approval"
            )
        except PermissionDeniedError:
            return CapabilityCallOutcome(
                call_id=call.id,
                capability_id=call.name,
                params=call.arguments,
                status="denied",
                error="This capability is set to never allow.",
            )
        except ToolDisabledError:
            return CapabilityCallOutcome(
                call_id=call.id,
                capability_id=call.name,
                params=call.arguments,
                status="disabled",
                error="This capability is currently disabled.",
            )
        except ToolNotFoundError:
            return CapabilityCallOutcome(
                call_id=call.id,
                capability_id=call.name,
                params=call.arguments,
                status="error",
                error=f"Unknown capability '{call.name}'.",
            )

    @staticmethod
    def _describe_outcome(outcome: CapabilityCallOutcome) -> str:
        if outcome.status == "executed":
            return str(outcome.result)
        return f"[{outcome.status}] {outcome.error or ''}".strip()


_orchestrator_instance: ConversationOrchestrator | None = None


def get_conversation_orchestrator() -> ConversationOrchestrator:
    """FastAPI dependency provider — reuses a single orchestrator instance per process."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = ConversationOrchestrator(get_ai_router())
    return _orchestrator_instance
