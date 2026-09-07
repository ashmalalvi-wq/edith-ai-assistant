"""
Common interface that every AI provider must implement.

This is the contract that lets the rest of the application (API routes,
router) work with any AI backend without knowing its implementation
details. To add a new provider: subclass AIProvider, implement the two
methods below, and register it in backend/ai/router.py.
"""
from abc import ABC, abstractmethod

from backend.models.schemas import AIProviderName, ChatMessage


class AIProvider(ABC):
    """Base class for all AI model providers (Ollama, OpenAI, Anthropic, ...)."""

    name: AIProviderName

    @abstractmethod
    async def is_available(self) -> tuple[bool, str | None]:
        """
        Check whether this provider is currently usable (e.g. reachable,
        has an API key configured). Returns (available, detail_message).
        """
        raise NotImplementedError

    @abstractmethod
    async def chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        tools: list[dict] | None = None,
    ) -> ChatMessage:
        """
        Send a conversation to the provider and return the assistant's reply.
        `model` overrides the provider's configured default model when given.

        `tools` (Phase 4), when given, is a list of OpenAI-style function-tool
        schemas (`{"type": "function", "function": {"name", "description",
        "parameters"}}`). A provider that supports native tool-calling should
        return a ChatMessage with `tool_calls` populated instead of (or
        alongside) `content` when the model wants to invoke one. A provider
        without tool-calling support may simply ignore `tools`.
        """
        raise NotImplementedError

    @abstractmethod
    def default_model(self) -> str:
        """The model name to use when the caller doesn't specify one."""
        raise NotImplementedError

    def supports_tools(self) -> bool:
        """Whether this provider's chat() actually honors the `tools` param."""
        return False
