"""OpenAI reasoning-model (o-series) expert provider — placeholder for a future phase. See claude_api.py's docstring for the pattern this follows."""
from backend.config.settings import get_settings
from backend.core.exceptions import ProviderNotImplementedError
from backend.ai.expert_reasoning.base import ExpertReasoningProvider, ExpertTaskRequest, ExpertTaskResult


class OpenAIReasoningProvider(ExpertReasoningProvider):
    name = "openai_reasoning"

    def __init__(self) -> None:
        self._api_key = get_settings().openai_api_key

    async def is_available(self) -> tuple[bool, str | None]:
        if not self._api_key:
            return False, "OPENAI_API_KEY is not configured."
        return False, "OpenAI reasoning expert provider is not implemented yet."

    async def run_task(self, request: ExpertTaskRequest) -> ExpertTaskResult:
        raise ProviderNotImplementedError(self.name)
