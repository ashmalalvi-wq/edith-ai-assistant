"""
Claude API expert reasoning provider — placeholder for a future phase.

Would call the Messages API directly (extended thinking / a long-context
model) rather than shelling out to the Claude Code CLI — useful when Claude
Code itself isn't installed/authenticated on the host, or for headless
deployments. Not implemented yet; same pattern as
backend/ai/providers/openai_provider.py and anthropic_provider.py being
placeholders until that phase is built.
"""
from backend.config.settings import get_settings
from backend.core.exceptions import ProviderNotImplementedError
from backend.ai.expert_reasoning.base import ExpertReasoningProvider, ExpertTaskRequest, ExpertTaskResult


class ClaudeAPIExpertProvider(ExpertReasoningProvider):
    name = "claude_api"

    def __init__(self) -> None:
        self._api_key = get_settings().anthropic_api_key

    async def is_available(self) -> tuple[bool, str | None]:
        if not self._api_key:
            return False, "ANTHROPIC_API_KEY is not configured."
        return False, "Claude API expert reasoning provider is not implemented yet."

    async def run_task(self, request: ExpertTaskRequest) -> ExpertTaskResult:
        raise ProviderNotImplementedError(self.name)
