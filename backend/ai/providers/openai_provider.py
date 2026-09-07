"""
OpenAI provider — placeholder for a future phase.

The interface is already wired into the router so the rest of the app
(API routes, frontend selector) can reference "openai" today. Implement
`chat()` with the OpenAI SDK when this phase is built.
"""
from backend.config.settings import get_settings
from backend.core.exceptions import ProviderNotImplementedError
from backend.models.schemas import AIProviderName, ChatMessage
from backend.ai.providers.base import AIProvider


class OpenAIProvider(AIProvider):
    name = AIProviderName.OPENAI

    def __init__(self) -> None:
        settings = get_settings()
        self._api_key = settings.openai_api_key
        self._default_model = settings.openai_default_model

    def default_model(self) -> str:
        return self._default_model

    async def is_available(self) -> tuple[bool, str | None]:
        if not self._api_key:
            return False, "OPENAI_API_KEY is not configured."
        return False, "OpenAI provider is not implemented yet."

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        tools: list[dict] | None = None,
    ) -> ChatMessage:
        raise ProviderNotImplementedError(self.name.value)
