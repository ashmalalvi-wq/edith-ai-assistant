"""
Capability reasoning — turns the Capability Manager's registered
capabilities into a tool schema a provider's native function-calling can
choose from, and turns the model's resulting tool_calls back into
requests the orchestrator can execute.

Phase 4: the "reasoning" is the model's own native tool-calling (see
ai/providers/ollama_provider.py), not a separate heuristic — this module
just does the translation both directions. A provider that doesn't support
tool-calling (OpenAI/Anthropic placeholders today) never receives a
`tools` schema, so this is inert for them.
"""
from abc import ABC, abstractmethod

from backend.capabilities.types import Capability
from backend.models.schemas import ChatMessage, ToolCallRequest


class CapabilityReasoner(ABC):
    """Translates between Capability Manager metadata and a provider's tool-calling wire format."""

    @abstractmethod
    def build_tool_schemas(self, capabilities: list[Capability]) -> list[dict]:
        """Describe every enabled, online capability as an OpenAI-style function tool."""
        raise NotImplementedError

    @abstractmethod
    def extract_requested_calls(self, reply: ChatMessage) -> list[ToolCallRequest]:
        """Pull the capability calls (if any) the model requested in its reply."""
        raise NotImplementedError


_JSON_SCHEMA_TYPES = {"string", "integer", "number", "boolean", "object", "array"}


class NativeFunctionCallingReasoner(CapabilityReasoner):
    def build_tool_schemas(self, capabilities: list[Capability]) -> list[dict]:
        schemas = []
        for capability in capabilities:
            if not capability.enabled or capability.status != "online":
                continue
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": capability.id,
                        "description": capability.description,
                        "parameters": self._parameters_schema(capability),
                    },
                }
            )
        return schemas

    @staticmethod
    def _parameters_schema(capability: Capability) -> dict:
        parameters = capability.metadata.get("parameters", [])
        properties = {}
        required = []
        for param in parameters:
            param_type = param.get("type") if param.get("type") in _JSON_SCHEMA_TYPES else "string"
            properties[param["name"]] = {"type": param_type, "description": param.get("description", "")}
            if param.get("required"):
                required.append(param["name"])
        return {"type": "object", "properties": properties, "required": required}

    def extract_requested_calls(self, reply: ChatMessage) -> list[ToolCallRequest]:
        return list(reply.tool_calls or [])


_reasoner_instance: CapabilityReasoner | None = None


def get_capability_reasoner() -> CapabilityReasoner:
    """FastAPI dependency provider — single instance per process."""
    global _reasoner_instance
    if _reasoner_instance is None:
        _reasoner_instance = NativeFunctionCallingReasoner()
    return _reasoner_instance
