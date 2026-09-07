"""Pydantic schemas shared between the API layer and the AI providers."""
from enum import Enum

from pydantic import BaseModel, Field


class AIProviderName(str, Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class ChatRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"  # a capability's result, fed back to the model mid-turn (Phase 4)


class ToolCallRequest(BaseModel):
    """One capability invocation the model asked for, in a single assistant turn."""

    id: str
    name: str
    arguments: dict = Field(default_factory=dict)


class ChatMessage(BaseModel):
    role: ChatRole
    content: str
    # Phase 4: set on an ASSISTANT message when the model requested capability
    # calls instead of (or alongside) a text reply.
    tool_calls: list[ToolCallRequest] | None = None
    # Phase 4: set on a TOOL message — which tool_call this result answers.
    tool_call_id: str | None = None
    # Phase 8: base64-encoded image data attached to a USER message. Only
    # OllamaProvider currently forwards this (Ollama's /api/chat accepts an
    # "images" field per message for vision-capable models) — providers that
    # don't understand it just ignore the field, same as any unused Pydantic
    # attribute. No vision-capable Ollama model is installed in this
    # environment, so this path is wired but not live-verified end to end;
    # pull one (e.g. `ollama pull llama3.2-vision`) to actually exercise it.
    images: list[str] | None = None


class ProviderStatus(BaseModel):
    name: AIProviderName
    available: bool
    implemented: bool
    default_model: str
    detail: str | None = None


class ProviderListResponse(BaseModel):
    providers: list[ProviderStatus]
    default_provider: AIProviderName
