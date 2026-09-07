"""
Embedding provider interface.

Kept separate from AIProvider (backend/ai/providers/) on purpose: chat
models and embedding models are different concerns, and you may eventually
want to pair e.g. Anthropic for chat with a local embedding model, or
switch embedding backends without touching the chat path at all.
"""
from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Base class for all embedding backends."""

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Convert a single piece of text into a vector."""
        raise NotImplementedError

    @abstractmethod
    def vector_size(self) -> int:
        """Dimensionality of vectors this provider produces."""
        raise NotImplementedError
