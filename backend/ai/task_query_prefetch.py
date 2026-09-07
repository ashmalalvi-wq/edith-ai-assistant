"""
Deterministic pre-fetch for task-listing questions.

Structural fix for a reliability gap found live (see CLAUDE.md's
"Voice-phrase task list querying + tool-calling reliability" session):
llama3.1:8b answers a *first* task-listing question correctly via native
tool-calling, but on a *second* similar question in the same conversation
it reliably (9/9 in testing) stops calling task.list at all — instead
fabricating an answer, going blank, or deflecting. A short added
system-prompt instruction telling it to always call the tool made zero
measurable difference.

Rather than depend on the model choosing to call task.list, detect a
task-listing intent deterministically here — same "rule-based over
model-based" rationale as MemoryClassifier/capability_keywords.py — and
fetch the real data ourselves on every matching turn, injecting it as
context the model just has to summarize. The model may still also call
task.list itself via normal tool-calling; that's harmless and redundant,
not a conflict — this only guarantees accurate data is present even when
it doesn't.
"""
import re

_TASK_WORDS = ("task", "todo", "to-do")

# A message needs one of these (or a "?") alongside a task word to count as
# a listing *question* rather than, say, a create/complete command.
_LIST_INTENT_PHRASES = (
    "what are my", "what's on my", "whats on my", "show me my", "show my",
    "list my", "pull up my", "what do i have", "read me my", "read my",
    "what tasks",
)

_PRIORITY_WORDS = {
    "high": "high",
    "important": "high",
    "urgent": "high",
    "medium": "medium",
    "moderate": "medium",
    "low": "low",
}


def detect_task_list_query(text: str) -> dict | None:
    """Returns task.list params ({} for no filter) if `text` looks like a
    task-listing question, else None if it doesn't match at all."""
    lowered = text.lower().strip()
    if not any(word in lowered for word in _TASK_WORDS):
        return None
    looks_like_query = "?" in lowered or any(phrase in lowered for phrase in _LIST_INTENT_PHRASES)
    if not looks_like_query:
        return None

    params: dict = {}
    for word, priority in _PRIORITY_WORDS.items():
        if re.search(rf"\b{word}\b", lowered):
            params["priority"] = priority
            break

    if re.search(r"\ball\b", lowered) and "task" in lowered:
        params["status"] = "all"
    elif any(word in lowered for word in ("completed", "finished", "done")):
        params["status"] = "completed"

    return params


def format_task_list_context(tasks: list[dict]) -> str:
    if not tasks:
        return (
            "Live task data for this question (already fetched just now — "
            "there are no matching tasks; say so plainly, do not invent any):\n"
            "None."
        )
    lines = [f"- {t['title']} (priority: {t['priority']}, status: {t['status']})" for t in tasks]
    return (
        "Live task data for this question (already fetched just now — "
        "summarize this real list in your reply; do not call task.list "
        "again this turn, and do not add or omit any tasks):\n" + "\n".join(lines)
    )
