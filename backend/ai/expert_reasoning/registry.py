"""
Resolves `settings.default_expert_reasoning_provider` to a concrete
ExpertReasoningProvider instance. Mirrors backend/ai/router.py's
_PROVIDER_CLASSES pattern — to add a new expert reasoning provider in the
future: implement ExpertReasoningProvider in a new module, add one line
here.
"""
from backend.capabilities.tools.executor import HostExecutor
from backend.config.settings import get_settings
from backend.ai.expert_reasoning.base import ExpertReasoningProvider
from backend.ai.expert_reasoning.claude_api import ClaudeAPIExpertProvider
from backend.ai.expert_reasoning.claude_code import ClaudeCodeProvider
from backend.ai.expert_reasoning.local_reasoning import LocalReasoningProvider
from backend.ai.expert_reasoning.openai_reasoning import OpenAIReasoningProvider

_instance: ExpertReasoningProvider | None = None


def get_expert_reasoning_provider(executor: HostExecutor | None = None) -> ExpertReasoningProvider:
    """FastAPI dependency provider — single instance per process, like get_ai_router()."""
    global _instance
    if _instance is not None:
        return _instance

    provider_name = get_settings().default_expert_reasoning_provider

    if provider_name == "claude_code":
        if executor is None:
            from backend.capabilities.tools.executor import get_host_executor

            executor = get_host_executor()
        _instance = ClaudeCodeProvider(executor)
    elif provider_name == "claude_api":
        _instance = ClaudeAPIExpertProvider()
    elif provider_name == "openai_reasoning":
        _instance = OpenAIReasoningProvider()
    elif provider_name == "local_reasoning":
        _instance = LocalReasoningProvider()
    else:
        raise ValueError(f"Unknown expert reasoning provider '{provider_name}'.")

    return _instance
