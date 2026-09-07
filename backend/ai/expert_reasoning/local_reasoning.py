"""
Local reasoning-model expert provider — placeholder for a future phase.
Would use a larger/slower local Ollama model in place of a cloud call, for
users who want expert-tier reasoning without leaving the machine. See
claude_api.py's docstring for the pattern this follows.
"""
from backend.core.exceptions import ProviderNotImplementedError
from backend.ai.expert_reasoning.base import ExpertReasoningProvider, ExpertTaskRequest, ExpertTaskResult


class LocalReasoningProvider(ExpertReasoningProvider):
    name = "local_reasoning"

    async def is_available(self) -> tuple[bool, str | None]:
        return False, "Local reasoning expert provider is not implemented yet."

    async def run_task(self, request: ExpertTaskRequest) -> ExpertTaskResult:
        raise ProviderNotImplementedError(self.name)
