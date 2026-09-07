"""
Default embedding provider — uses the same local Ollama instance the chat
path already talks to, via its /api/embeddings endpoint. Requires the
configured embedding model to be pulled, e.g.:

    ollama pull nomic-embed-text
"""
import logging

import httpx

from backend.config.settings import get_settings
from backend.core.exceptions import EmbeddingError
from backend.memory.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class OllamaEmbeddingProvider(EmbeddingProvider):
    def __init__(self) -> None:
        settings = get_settings()
        self._base_url = settings.ollama_base_url.rstrip("/")
        self._model = settings.embedding_model
        self._vector_size = settings.qdrant_vector_size
        self._timeout = settings.ollama_request_timeout

    def vector_size(self) -> int:
        return self._vector_size

    async def embed(self, text: str) -> list[float]:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}/api/embeddings",
                    json={"model": self._model, "prompt": text},
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as exc:
            raise EmbeddingError(str(exc)) from exc

        embedding = data.get("embedding")
        if not embedding:
            raise EmbeddingError(f"Ollama returned no embedding for model '{self._model}'.")
        return embedding


_provider_instance: OllamaEmbeddingProvider | None = None


def get_embedding_provider() -> OllamaEmbeddingProvider:
    """FastAPI dependency provider — single instance per process."""
    global _provider_instance
    if _provider_instance is None:
        _provider_instance = OllamaEmbeddingProvider()
    return _provider_instance
