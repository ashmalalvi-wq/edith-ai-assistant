"""
Multimodal input — extension point for future non-text conversation
content (images, audio, files).

Not implemented yet: ChatMessage (backend/models/schemas.py) is text-only
today, and nothing constructs a MultimodalAttachment anywhere in the
codebase. This module exists so multimodal content has a designated home
when it's built, rather than requiring changes to the provider interface
or orchestrator at that point.
"""
from dataclasses import dataclass


@dataclass
class MultimodalAttachment:
    """Placeholder for a future non-text conversation input."""

    kind: str  # e.g. "image", "audio", "file"
    data: bytes | str
