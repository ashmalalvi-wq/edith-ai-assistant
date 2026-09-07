"""
Rule-based memory classifier.

E.D.I.T.H should not store every message permanently — most of a
conversation is transient. This module decides, deterministically, whether
an exchange contains something worth remembering and what category it
belongs in. It's intentionally simple (regex/keyword triggers) rather than
LLM-based, per the priority on reliability: the same input always produces
the same classification, with no extra model call and no risk of an LLM
hallucinating a "memory" that was never said.

This is a deliberate seam: swapping in an LLM-based classifier later means
implementing `MemoryClassifier` differently, not touching the pipeline that
calls it.
"""
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

MEMORY_TYPE_TEMPORARY = "temporary"
MEMORY_TYPE_LONG_TERM = "long_term"
MEMORY_TYPE_PROJECT = "project"

TEMPORARY_MEMORY_TTL = timedelta(days=7)

# Explicit user instruction to remember something — always stored, always long_term
# (or project, if a project is in scope).
_EXPLICIT_REMEMBER_RE = re.compile(
    r"\b(remember|don't forget|note that|keep in mind|for future reference)\b", re.IGNORECASE
)

# Statements of durable preference — stored as long_term memory.
_PREFERENCE_RE = re.compile(
    r"\b(i prefer|i like|i always|i usually|i never|by default|my default)\b", re.IGNORECASE
)

# Minimum length for an unprompted project-context message to be worth keeping
# (filters out "ok", "thanks", "yes" from ever becoming project memory).
_MIN_PROJECT_MEMORY_LENGTH = 40


@dataclass
class MemoryCandidate:
    content: str
    type: str
    expires_at: datetime | None = None


class MemoryClassifier:
    def classify(
        self, *, user_message: str, assistant_message: str, project_id: object | None
    ) -> list[MemoryCandidate]:
        candidates: list[MemoryCandidate] = []
        text = user_message.strip()
        if not text:
            return candidates

        if _EXPLICIT_REMEMBER_RE.search(text):
            candidates.append(
                MemoryCandidate(
                    content=text,
                    type=MEMORY_TYPE_PROJECT if project_id else MEMORY_TYPE_LONG_TERM,
                )
            )
            return candidates

        if _PREFERENCE_RE.search(text):
            candidates.append(MemoryCandidate(content=text, type=MEMORY_TYPE_LONG_TERM))
            return candidates

        if project_id and len(text) >= _MIN_PROJECT_MEMORY_LENGTH:
            candidates.append(MemoryCandidate(content=text, type=MEMORY_TYPE_PROJECT))
            return candidates

        return candidates


_classifier_instance = MemoryClassifier()


def get_classifier() -> MemoryClassifier:
    return _classifier_instance
