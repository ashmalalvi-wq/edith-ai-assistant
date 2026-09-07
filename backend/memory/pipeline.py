"""
Memory pipeline — the single place that wires together retrieval, the AI
model router, and memory storage for a chat turn.

Flow (Phase 1-3): user message -> retrieve context -> build prompt -> AI
router -> response -> persist messages -> classify -> store new memories

Phase 4: step 5 now runs through ConversationOrchestrator.run_turn(), which
lets the model call capabilities mid-turn. If a capability needs
confirmation, the turn pauses — its resumable state (message history +
which call is pending) is stashed in Session Memory's `context` field,
keyed by conversation id, rather than added as a new table (Session Memory
already exists precisely to hold "current runtime state"). `resolve_pending`
picks that back up once the user approves/denies.

This is the only module the chat API route needs to call; it owns the
ordering so that route stays thin.
"""
import base64
import logging
import uuid
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from backend.ai.orchestrator import (
    ConversationOrchestrator,
    ConversationTurnResult,
    PendingApproval,
    get_conversation_orchestrator,
)
from backend.ai.prompts import PromptBuilder, get_prompt_builder
from backend.ai.task_query_prefetch import detect_task_list_query, format_task_list_context
from backend.capabilities.manager import CapabilityManager, get_capability_manager
from backend.capabilities.tools import log_store as capability_log_store
from backend.config.settings import get_settings
from backend.core.exceptions import ConfirmationRequiredError
from backend.models.db_models import Conversation, Message
from backend.models.memory_schemas import ChatAttachmentRef
from backend.models.schemas import AIProviderName, ChatMessage, ChatRole
from backend.memory.embeddings.base import EmbeddingProvider
from backend.memory.embeddings.ollama_embeddings import get_embedding_provider
from backend.memory import session_memory, structured_store
from backend.memory.classifier import MemoryClassifier, get_classifier
from backend.memory.profiles import store as profile_store
from backend.memory.retrieval import MemoryRetrievalService
from backend.memory.vector_store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)

_TITLE_MAX_LENGTH = 60
_RECENT_CAPABILITY_COUNT = 5


def _derive_title(user_message: str) -> str:
    text = " ".join(user_message.split())
    if len(text) <= _TITLE_MAX_LENGTH:
        return text or "New conversation"
    return text[: _TITLE_MAX_LENGTH - 1].rstrip() + "…"


class MemoryPipeline:
    def __init__(
        self,
        orchestrator: ConversationOrchestrator,
        retrieval_service: MemoryRetrievalService,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        classifier: MemoryClassifier,
        prompt_builder: PromptBuilder | None = None,
        capability_manager: CapabilityManager | None = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._retrieval = retrieval_service
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._classifier = classifier
        self._prompt_builder = prompt_builder or get_prompt_builder()
        self._capability_manager = capability_manager or get_capability_manager()

    async def handle_message(
        self,
        session: AsyncSession,
        *,
        user_content: str,
        conversation_id: uuid.UUID | None,
        project_id: uuid.UUID | None,
        provider: AIProviderName | None,
        model: str | None,
        mode: str | None = None,
        expert_task_mode: str | None = None,
        attachments: list[ChatAttachmentRef] | None = None,
        stream_request_id: str | None = None,
    ) -> tuple[Conversation, Message | None, PendingApproval | None]:
        settings = get_settings()
        active_session = await session_memory.get_or_create_session(session)
        attachments = attachments or []

        # 1. Resolve or create the conversation.
        if conversation_id is not None:
            conversation = await structured_store.get_conversation_with_messages(session, conversation_id)
            effective_project_id = project_id or conversation.project_id
        else:
            conversation = await structured_store.create_conversation(
                session,
                title=_derive_title(user_content),
                project_id=project_id,
                ai_provider=(provider.value if provider else settings.default_ai_provider),
                ai_model=model,
            )
            effective_project_id = project_id
            if active_session.profile_id is not None:
                await structured_store.update_conversation(
                    session, conversation.id, profile_id=active_session.profile_id
                )

        # 2. Persist the user's message. Attachment filenames are appended so
        #    they're visible in conversation history/search even though the
        #    image bytes themselves aren't stored on the Message row (kept
        #    on disk under backend/data/uploads/ instead — see attachments.py).
        display_content = user_content
        if attachments:
            names = ", ".join(a.filename for a in attachments)
            display_content = f"{user_content}\n\n[Attached: {names}]" if user_content.strip() else f"[Attached: {names}]"
        await structured_store.add_message(
            session, conversation_id=conversation.id, role=ChatRole.USER.value, content=display_content
        )

        # 2b. Expert Mode: route straight to the Expert Reasoning Agent
        # (Claude Code) instead of the normal Ollama/cloud chat pipeline —
        # per the architecture's "Advanced tasks" path, this bypasses the
        # local AI model entirely rather than asking it to decide. Claude
        # Code reads images itself via its own filesystem tools, given a
        # path — so attachments are threaded in as absolute paths in the
        # task text rather than base64 (which only the Ollama path uses).
        if mode == "expert":
            expert_task = user_content
            if attachments:
                paths = "\n".join(f"- {a.path}" for a in attachments)
                expert_task = f"{user_content}\n\nAttached image file(s) — read these before responding:\n{paths}"
            return await self._handle_expert_mode(
                session, conversation, expert_task, effective_project_id, active_session,
                task_mode=expert_task_mode or "analyze",
                stream_request_id=stream_request_id,
            )

        # 3. Retrieve relevant context (structured + semantic + profile/session/
        #    capability activity). Never fatal — a retrieval failure degrades
        #    to "no extra context", not a broken chat.
        try:
            context = await self._retrieval.retrieve(
                session, user_message=user_content, project_id=effective_project_id
            )
            memory_block = context.to_prompt_block() if context is not None else ""
        except Exception:
            logger.exception("Memory retrieval failed; continuing without context")
            memory_block = ""

        try:
            workspace_block = await self._gather_workspace_context(session, active_session)
        except Exception:
            logger.exception("Workspace context gathering failed; continuing without it")
            workspace_block = ""

        try:
            task_block = await self._prefetch_task_list_context(
                session, user_content, conversation.id, active_session.profile_id
            )
        except Exception:
            logger.exception("Task list prefetch failed; continuing without it")
            task_block = ""

        # 4. Build the full message list: persona + all context, then conversation history.
        context_block = "\n\n".join(block for block in (memory_block, workspace_block, task_block) if block)
        system_message = self._prompt_builder.build_system_message(context_block or None)

        conversation = await structured_store.get_conversation_with_messages(session, conversation.id)
        history = [
            ChatMessage(role=ChatRole(m.role), content=m.content) for m in conversation.messages
        ]

        # Attach image bytes to the user message we just persisted (it's the
        # last one in history) — only OllamaProvider forwards this, and only
        # a vision-capable model does anything useful with it, but it's
        # harmless to include otherwise (see ChatMessage.images).
        if attachments:
            images_b64 = self._read_attachments_as_base64(attachments)
            for msg in reversed(history):
                if msg.role == ChatRole.USER:
                    msg.images = images_b64
                    break

        full_messages = [system_message, *history]

        # 5. Run the turn — the AI subsystem's orchestrator, which may call
        #    capabilities along the way (provider-agnostic; a provider
        #    without tool-calling support just never receives any).
        turn_result = await self._orchestrator.run_turn(
            session,
            full_messages,
            provider=provider,
            model=model,
            capability_manager=self._capability_manager,
            conversation_id=conversation.id,
            profile_id=active_session.profile_id,
        )

        if turn_result.pending_approval is not None:
            await self._store_pending_approval(session, conversation.id, provider, model, turn_result)
            return conversation, None, turn_result.pending_approval

        return await self._finalize_turn(session, conversation, effective_project_id, user_content, turn_result)

    @staticmethod
    def _read_attachments_as_base64(attachments: list[ChatAttachmentRef]) -> list[str]:
        encoded = []
        for attachment in attachments:
            try:
                encoded.append(base64.b64encode(Path(attachment.path).read_bytes()).decode("ascii"))
            except OSError:
                logger.exception("Could not read attachment %s at %s; skipping it", attachment.id, attachment.path)
        return encoded

    async def _handle_expert_mode(
        self,
        session: AsyncSession,
        conversation: Conversation,
        user_content: str,
        project_id: uuid.UUID | None,
        active_session,
        *,
        task_mode: str = "analyze",
        stream_request_id: str | None = None,
    ) -> tuple[Conversation, Message | None, PendingApproval | None]:
        # "analyze" (Plan) is read-only — Claude Code runs with
        # --permission-mode plan and can never touch files. "implement"
        # (Auto) actually applies changes (--permission-mode acceptEdits),
        # see backend/ai/expert_reasoning/claude_code.py's
        # _PERMISSION_MODE_FOR_TASK_MODE — this only selects which of those
        # two already-existing capabilities to call, nothing new underneath.
        capability_id = "expert.implement" if task_mode == "implement" else "expert.analyze"
        params = {"task": user_content, "project_id": str(project_id) if project_id else None}
        if stream_request_id:
            params["stream_request_id"] = stream_request_id
        try:
            result = await self._capability_manager.execute_capability(
                session,
                capability_id,
                params,
                confirmed=False,
                conversation_id=conversation.id,
                profile_id=active_session.profile_id,
                initiated_by="user",
            )
        except ConfirmationRequiredError:
            call_id = str(uuid.uuid4())
            await self._store_direct_pending_approval(session, conversation.id, call_id, capability_id, params)
            return conversation, None, PendingApproval(call_id=call_id, capability_id=capability_id, params=params)

        return await self._finalize_direct_capability(session, conversation, user_content, project_id, result)

    async def resolve_pending_capability(
        self, session: AsyncSession, conversation_id: uuid.UUID, *, approved: bool, stream_request_id: str | None = None
    ) -> tuple[Conversation, Message | None, PendingApproval | None]:
        """Continues a turn that paused for approval (see handle_message's step 5)."""
        conversation = await structured_store.get_conversation_with_messages(session, conversation_id)
        active_session = await session_memory.get_or_create_session(session)
        pending_state = await self._pop_pending_state(session, conversation_id)
        if pending_state is None:
            raise ValueError(f"No pending capability approval for conversation {conversation_id}.")

        # Expert Mode (or any other direct-capability call, see
        # _handle_expert_mode) stores a simpler shape than the orchestrator
        # loop's — no message history to replay, just the one call.
        if pending_state.get("direct_capability_id"):
            capability_id = pending_state["direct_capability_id"]
            params = dict(pending_state["params"])
            # Always reflects *this* resolve call's stream id (or its
            # absence), never a stale one left over from the original
            # unconfirmed attempt that paused before anything could stream.
            if stream_request_id:
                params["stream_request_id"] = stream_request_id
            else:
                params.pop("stream_request_id", None)
            if not approved:
                message = await structured_store.add_message(
                    session, conversation_id=conversation.id, role=ChatRole.ASSISTANT.value,
                    content=f"Understood — I won't run '{capability_id}'.",
                )
                return conversation, message, None

            result = await self._capability_manager.execute_capability(
                session, capability_id, params, confirmed=True,
                conversation_id=conversation.id, profile_id=active_session.profile_id, initiated_by="user",
            )
            return await self._finalize_direct_capability(session, conversation, None, conversation.project_id, result)

        messages = [ChatMessage.model_validate(m) for m in pending_state["messages"]]
        pending = PendingApproval(
            call_id=pending_state["call_id"],
            capability_id=pending_state["capability_id"],
            params=pending_state["params"],
        )
        provider = AIProviderName(pending_state["provider"]) if pending_state.get("provider") else None
        model = pending_state.get("model")

        turn_result = await self._orchestrator.resolve_pending(
            session,
            messages,
            pending,
            approved=approved,
            provider=provider,
            model=model,
            capability_manager=self._capability_manager,
            conversation_id=conversation.id,
            profile_id=active_session.profile_id,
        )

        if turn_result.pending_approval is not None:
            await self._store_pending_approval(session, conversation.id, provider, model, turn_result)
            return conversation, None, turn_result.pending_approval

        return await self._finalize_turn(session, conversation, conversation.project_id, None, turn_result)

    async def _finalize_turn(
        self,
        session: AsyncSession,
        conversation: Conversation,
        project_id: uuid.UUID | None,
        user_content: str | None,
        turn_result: ConversationTurnResult,
    ) -> tuple[Conversation, Message, None]:
        assistant_message = await structured_store.add_message(
            session, conversation_id=conversation.id, role=ChatRole.ASSISTANT.value, content=turn_result.reply.content
        )
        await structured_store.update_conversation(
            session, conversation.id, ai_provider=turn_result.provider.value, ai_model=turn_result.model
        )

        if user_content is not None:
            await self._store_new_memories(
                session,
                user_content=user_content,
                assistant_content=turn_result.reply.content,
                conversation_id=conversation.id,
                project_id=project_id,
            )

        return conversation, assistant_message, None

    async def _finalize_direct_capability(
        self,
        session: AsyncSession,
        conversation: Conversation,
        user_content: str | None,
        project_id: uuid.UUID | None,
        result,
    ) -> tuple[Conversation, Message, None]:
        """Turns a direct expert.* ToolResult into an assistant message — no LLM call involved (Expert Mode bypasses chat entirely).

        Two distinct "did it work" signals get checked here, not just one:
        `result.success` (the outer ToolResult — False only if the Python
        call itself raised) and `output["success"]` (expert_tool.py's own
        inner flag — False if Claude Code/the provider failed without
        raising, e.g. a non-zero exit or an unavailable provider). Only
        checking the outer one was a real bug found live: an inner failure
        left `output["summary"]` empty while `result.success` stayed True,
        so this fell into the success branch and showed the unhelpful
        "(no response)" instead of the actual error — silently swallowing
        exactly the detail needed to diagnose it.
        """
        output = result.output or {}
        if result.success and output.get("success", True):
            reply_text = output.get("summary") or "(no response)"
        else:
            error = output.get("error") or result.error or "unknown error"
            reply_text = f"The expert agent couldn't complete that: {error}"

        assistant_message = await structured_store.add_message(
            session, conversation_id=conversation.id, role=ChatRole.ASSISTANT.value, content=reply_text
        )

        if user_content is not None:
            await self._store_new_memories(
                session, user_content=user_content, assistant_content=reply_text,
                conversation_id=conversation.id, project_id=project_id,
            )

        return conversation, assistant_message, None

    async def _store_direct_pending_approval(
        self, session: AsyncSession, conversation_id: uuid.UUID, call_id: str, capability_id: str, params: dict
    ) -> None:
        active_session = await session_memory.get_or_create_session(session)
        context = dict(active_session.context or {})
        pending_approvals = dict(context.get("pending_approvals", {}))
        pending_approvals[str(conversation_id)] = {
            "direct_capability_id": capability_id,
            "call_id": call_id,
            "params": params,
        }
        context["pending_approvals"] = pending_approvals
        await session_memory.update_session(session, context=context)

    async def _prefetch_task_list_context(
        self, session: AsyncSession, user_content: str, conversation_id: uuid.UUID, profile_id
    ) -> str:
        """Deterministic structural fix for observed multi-turn tool-calling
        unreliability — see task_query_prefetch.py's module docstring for why
        this exists instead of just trusting the model to call task.list."""
        params = detect_task_list_query(user_content)
        if params is None:
            return ""

        result = await self._capability_manager.execute_capability(
            session,
            "task.list",
            params,
            confirmed=True,
            conversation_id=conversation_id,
            profile_id=profile_id,
            initiated_by="ai",
        )
        if not result.success:
            return ""
        return format_task_list_context(result.output or [])

    async def _gather_workspace_context(self, session: AsyncSession, active_session) -> str:
        """Profile/session/recent-capability-activity context, so the model can
        resolve things like "resume my engineering work" without the user
        having to spell out which profile/project that means."""
        lines = []

        if active_session.profile_id is not None:
            try:
                profile = await profile_store.get_profile(session, active_session.profile_id)
                lines.append(f"Active profile: {profile.name} ({profile.type})")
                # Phase 6: engineering context (item 11 — "continue my pump
                # project" needs CAD files/git repos visible, not just the
                # profile name), so this is worth surfacing whenever a
                # profile has any of it set, not just for type=="engineering".
                if profile.cad_files:
                    lines.append(f"Tracked CAD files: {', '.join(profile.cad_files[:5])}")
                if profile.git_repositories:
                    lines.append(f"Git repositories: {', '.join(profile.git_repositories[:5])}")
                if profile.printer_settings:
                    lines.append(f"Printer settings: {profile.printer_settings}")
            except Exception:
                pass

        if active_session.active_project_id is not None:
            try:
                project = await structured_store.get_project(session, active_session.active_project_id)
                lines.append(f"Active workspace project: {project.name}")
            except Exception:
                pass

        recent_logs = await capability_log_store.list_logs(
            session, profile_id=active_session.profile_id, limit=_RECENT_CAPABILITY_COUNT
        )
        if recent_logs:
            recent_names = ", ".join(dict.fromkeys(log.tool_name for log in recent_logs))
            lines.append(f"Recently used capabilities: {recent_names}")

        if not lines:
            return ""
        return "--- Workspace context ---\n" + "\n".join(lines) + "\n--- end workspace context ---"

    async def _store_pending_approval(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        provider: AIProviderName | None,
        model: str | None,
        turn_result: ConversationTurnResult,
    ) -> None:
        active_session = await session_memory.get_or_create_session(session)
        context = dict(active_session.context or {})
        pending_approvals = dict(context.get("pending_approvals", {}))
        pending_approvals[str(conversation_id)] = {
            "call_id": turn_result.pending_approval.call_id,
            "capability_id": turn_result.pending_approval.capability_id,
            "params": turn_result.pending_approval.params,
            "messages": [m.model_dump(mode="json") for m in turn_result.messages],
            "provider": provider.value if provider else turn_result.provider.value,
            "model": model or turn_result.model,
        }
        context["pending_approvals"] = pending_approvals
        await session_memory.update_session(session, context=context)

    async def _pop_pending_state(self, session: AsyncSession, conversation_id: uuid.UUID) -> dict | None:
        active_session = await session_memory.get_or_create_session(session)
        context = dict(active_session.context or {})
        pending_approvals = dict(context.get("pending_approvals", {}))
        state = pending_approvals.pop(str(conversation_id), None)
        if state is not None:
            context["pending_approvals"] = pending_approvals
            await session_memory.update_session(session, context=context)
        return state

    async def _store_new_memories(
        self,
        session: AsyncSession,
        *,
        user_content: str,
        assistant_content: str,
        conversation_id: uuid.UUID,
        project_id: uuid.UUID | None,
    ) -> None:
        candidates = self._classifier.classify(
            user_message=user_content, assistant_message=assistant_content, project_id=project_id
        )
        for candidate in candidates:
            try:
                memory = await structured_store.create_memory(
                    session,
                    type=candidate.type,
                    content=candidate.content,
                    project_id=project_id,
                    source_conversation_id=conversation_id,
                    expires_at=candidate.expires_at,
                )
                vector = await self._embedding_provider.embed(candidate.content)
                point_id = uuid.uuid4()
                await self._vector_store.upsert_memory(
                    point_id=point_id,
                    vector=vector,
                    memory_id=memory.id,
                    type=memory.type,
                    project_id=project_id,
                    content=candidate.content,
                )
                await structured_store.update_memory(session, memory.id, qdrant_point_id=point_id)
            except Exception:
                # Storing a new memory should never fail the user-facing response.
                logger.exception("Failed to store a new memory; continuing")


_pipeline_instance: MemoryPipeline | None = None


def get_memory_pipeline() -> MemoryPipeline:
    """FastAPI dependency provider — wires up the memory pipeline's collaborators once."""
    global _pipeline_instance
    if _pipeline_instance is None:
        embedding_provider = get_embedding_provider()
        vector_store = get_vector_store()
        retrieval_service = MemoryRetrievalService(embedding_provider, vector_store)
        _pipeline_instance = MemoryPipeline(
            orchestrator=get_conversation_orchestrator(),
            retrieval_service=retrieval_service,
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            classifier=get_classifier(),
            prompt_builder=get_prompt_builder(),
            capability_manager=get_capability_manager(),
        )
    return _pipeline_instance
