"""
AI Model Router.

This is the single entry point the rest of the application uses to talk
to AI providers. It hides which concrete provider (Ollama, OpenAI,
Anthropic, ...) is behind a given request, so callers only ever depend on
the AIProvider interface. Prompt/persona assembly lives in prompts.py;
conversation-level orchestration lives in orchestrator.py — this module is
provider resolution and the raw chat call only.

To add a new provider in the future: implement AIProvider in a new module
under ai/providers/, then add one line to _PROVIDER_CLASSES below.
"""
import logging

from backend.config.settings import get_settings
from backend.core.exceptions import ProviderNotAvailableError
from backend.models.schemas import AIProviderName, ChatMessage, ProviderStatus
from backend.ai.prompts import get_prompt_builder
from backend.ai.providers.anthropic_provider import AnthropicProvider
from backend.ai.providers.base import AIProvider
from backend.ai.providers.ollama_provider import OllamaProvider
from backend.ai.providers.openai_provider import OpenAIProvider

logger = logging.getLogger(__name__)

_PROVIDER_CLASSES: dict[AIProviderName, type[AIProvider]] = {
    AIProviderName.OLLAMA: OllamaProvider,
    AIProviderName.OPENAI: OpenAIProvider,
    AIProviderName.ANTHROPIC: AnthropicProvider,
}

# Providers with a full chat implementation. Used to report "implemented"
# status to the frontend so it can grey out placeholders.
_IMPLEMENTED_PROVIDERS = {AIProviderName.OLLAMA}


class AIModelRouter:
    """Resolves an AIProviderName to a concrete, ready-to-use provider instance."""

    def __init__(self) -> None:
        self._instances: dict[AIProviderName, AIProvider] = {}

    def get_provider(self, name: AIProviderName | None = None) -> AIProvider:
        settings = get_settings()
        resolved_name = name or AIProviderName(settings.default_ai_provider)

        if resolved_name not in self._instances:
            provider_class = _PROVIDER_CLASSES.get(resolved_name)
            if provider_class is None:
                raise ProviderNotAvailableError(resolved_name.value, "Unknown provider.")
            self._instances[resolved_name] = provider_class()

        return self._instances[resolved_name]

    async def chat(
        self,
        messages: list[ChatMessage],
        provider: AIProviderName | None = None,
        model: str | None = None,
        tools: list[dict] | None = None,
    ) -> tuple[AIProviderName, str, ChatMessage]:
        selected_provider = self.get_provider(provider)

        available, detail = await selected_provider.is_available()
        if not available:
            raise ProviderNotAvailableError(selected_provider.name.value, detail or "")

        prepared_messages = get_prompt_builder().with_persona(messages)
        reply = await selected_provider.chat(prepared_messages, model=model, tools=tools)
        used_model = model or selected_provider.default_model()
        return selected_provider.name, used_model, reply

    def provider_supports_tools(self, provider: AIProviderName | None = None) -> bool:
        return self.get_provider(provider).supports_tools()

    async def list_provider_statuses(self) -> list[ProviderStatus]:
        statuses: list[ProviderStatus] = []
        for name in AIProviderName:
            provider = self.get_provider(name)
            available, detail = await provider.is_available()
            statuses.append(
                ProviderStatus(
                    name=name,
                    available=available,
                    implemented=name in _IMPLEMENTED_PROVIDERS,
                    default_model=provider.default_model(),
                    detail=detail,
                )
            )
        return statuses


_router_instance: AIModelRouter | None = None


def get_ai_router() -> AIModelRouter:
    """FastAPI dependency provider — reuses a single router instance per process."""
    global _router_instance
    if _router_instance is None:
        _router_instance = AIModelRouter()
    return _router_instance
