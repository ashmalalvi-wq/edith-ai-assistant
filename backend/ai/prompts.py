"""
Prompt construction — the one place that builds the system-level content
E.D.I.T.H sees before every conversation turn (persona + any retrieved
memory context).

Kept separate from provider routing (router.py) and conversation
orchestration (orchestrator.py) so persona/prompt-assembly logic has a
single home — e.g. per-profile personas or future prompt templates can be
added here without touching either of those.
"""
from backend.config.settings import get_settings
from backend.models.schemas import ChatMessage, ChatRole


class PromptBuilder:
    """Builds the system message(s) sent ahead of a conversation."""

    def build_system_message(self, context_block: str | None = None) -> ChatMessage:
        """The persona system prompt, with optional retrieved-memory context appended."""
        settings = get_settings()
        content = settings.system_prompt
        if context_block:
            content = f"{content}\n\n{context_block}"
        return ChatMessage(role=ChatRole.SYSTEM, content=content)

    def with_persona(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        """
        Prepend EDITH's persona as a system message, unless the caller
        already supplied one. Fallback for callers (e.g. the router used
        directly) that didn't pre-build a system message via
        `build_system_message`.
        """
        if messages and messages[0].role == ChatRole.SYSTEM:
            return messages
        return [self.build_system_message(), *messages]


_builder_instance: PromptBuilder | None = None


def get_prompt_builder() -> PromptBuilder:
    """FastAPI dependency provider — reuses a single builder instance per process."""
    global _builder_instance
    if _builder_instance is None:
        _builder_instance = PromptBuilder()
    return _builder_instance
