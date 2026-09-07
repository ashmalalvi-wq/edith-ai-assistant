"""Ollama provider — talks to a local Ollama instance over HTTP."""
import logging

import httpx

from backend.config.settings import get_settings
from backend.core.exceptions import ModelRequestError
from backend.models.schemas import AIProviderName, ChatMessage, ChatRole, ToolCallRequest
from backend.ai.providers.base import AIProvider

logger = logging.getLogger(__name__)


class OllamaProvider(AIProvider):
    name = AIProviderName.OLLAMA

    def __init__(self) -> None:
        settings = get_settings()
        self._base_url = settings.ollama_base_url.rstrip("/")
        self._default_model = settings.ollama_default_model
        self._timeout = settings.ollama_request_timeout

    def default_model(self) -> str:
        return self._default_model

    async def is_available(self) -> tuple[bool, str | None]:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{self._base_url}/api/tags")
                response.raise_for_status()
            return True, None
        except httpx.HTTPError as exc:
            detail = f"Cannot reach Ollama at {self._base_url}: {exc}"
            logger.warning(detail)
            return False, detail

    def supports_tools(self) -> bool:
        return True

    async def chat(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        tools: list[dict] | None = None,
    ) -> ChatMessage:
        target_model = model or self._default_model

        # Ollama's /api/chat returns 400 Bad Request if "images" is sent to a
        # model that doesn't declare the "vision" capability — this isn't a
        # silent no-op, it fails the whole turn. Check once per call (only
        # when there's actually an image to send) and strip + explain rather
        # than crash.
        has_images = any(m.images for m in messages)
        vision_ok = await self._supports_vision(target_model) if has_images else True

        ollama_messages = []
        for m in messages:
            ollama_message = self._to_ollama_message(m)
            if m.images and not vision_ok:
                ollama_message.pop("images", None)
                ollama_message["content"] = (
                    ollama_message["content"]
                    + f"\n\n(Note: {len(m.images)} image(s) were attached, but the current model "
                    f"'{target_model}' doesn't support vision, so they weren't seen. A vision-capable "
                    "model — e.g. `ollama pull llama3.2-vision` — is needed to actually understand images.)"
                )
            ollama_messages.append(ollama_message)

        payload = {
            "model": target_model,
            "messages": ollama_messages,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(f"{self._base_url}/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as exc:
            raise ModelRequestError(self.name.value, str(exc)) from exc

        message = data.get("message", {})
        tool_calls = self._parse_tool_calls(message.get("tool_calls"))
        return ChatMessage(
            role=ChatRole.ASSISTANT, content=message.get("content", "") or "", tool_calls=tool_calls
        )

    async def _supports_vision(self, model: str) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.post(f"{self._base_url}/api/show", json={"model": model})
                response.raise_for_status()
                data = response.json()
            return "vision" in (data.get("capabilities") or [])
        except httpx.HTTPError:
            # Fail closed — if we can't confirm vision support, don't risk
            # sending images and getting the whole turn rejected.
            return False

    @staticmethod
    def _to_ollama_message(message: ChatMessage) -> dict:
        payload = {"role": message.role.value, "content": message.content}
        if message.role == ChatRole.TOOL and message.tool_call_id:
            # Ollama's /api/chat matches tool results back to calls by name,
            # not an id — the id lives only in our own conversation history.
            payload["tool_call_id"] = message.tool_call_id
        if message.images:
            # Only vision-capable models (e.g. llama3.2-vision, qwen2.5vl) do
            # anything with this — Ollama silently ignores it otherwise.
            payload["images"] = message.images
        return payload

    @staticmethod
    def _parse_tool_calls(raw: list[dict] | None) -> list[ToolCallRequest] | None:
        if not raw:
            return None
        calls = []
        for i, call in enumerate(raw):
            function = call.get("function", {})
            calls.append(
                ToolCallRequest(
                    id=str(call.get("id") or f"call_{i}"),
                    name=function.get("name", ""),
                    arguments=function.get("arguments") or {},
                )
            )
        return calls or None

    async def list_models(self) -> list[str]:
        """Extra helper (not part of the base interface) to list installed models."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{self._base_url}/api/tags")
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as exc:
            raise ModelRequestError(self.name.value, str(exc)) from exc

        return [m["name"] for m in data.get("models", [])]
