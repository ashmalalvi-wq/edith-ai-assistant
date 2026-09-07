"""
Keyword gate for capability tool-schemas.

Phase 4 originally sent every enabled/online capability (~75 of them) as a
function-calling schema on *every* chat turn, including plain conversation
("hi, how are you?"). That's a lot of extra prompt tokens for a small local
model to read on every message, and in practice it measurably confused
llama3.1:8b into responding with tool-call-shaped text or generic
"no relevant function" replies even when nothing needed a tool.

This module is a cheap, deterministic pre-filter — same reliability
rationale as MemoryClassifier/EmailCategorizer being rule-based rather than
another model call: it looks at the latest user message for keywords/
phrases associated with each capability category and only builds schemas
for categories that plausibly matched. If nothing matches, no `tools` are
sent at all and the model just talks.

This is intentionally coarse (phrase matching, not intent classification).
False negatives just mean the model answers without a tool it could have
used and the user can ask again more specifically; false positives just
mean a few extra schemas got attached. Both are cheap failure modes,
unlike sending the full 75 every time.
"""
import re

from backend.capabilities.types import Capability
from backend.models.schemas import ChatMessage, ChatRole

# Category -> phrases that suggest that category is relevant to the
# current message. Matched as whole-word/phrase substrings, case-insensitive.
_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "filesystem": [
        "file", "files", "folder", "directory", "read file", "create file",
        "delete file", "move file", "rename", "list files", "search files",
    ],
    "terminal": [
        "run command", "terminal", "shell", "powershell", "execute", "cmd",
        "run a script", "run this command",
    ],
    "vscode": [
        "vscode", "vs code", "open project", "open in code", "code editor",
        "open workspace",
    ],
    "workspace": [
        "open application", "launch", "resume my", "workspace", "open project",
        "close application", "start app", "startup applications",
    ],
    "browser": ["browser", "website", "browse to", "open url", "open a tab"],
    "fusion": ["fusion", "fusion 360", "cad", "component", "sketch"],
    "bambu": [
        "bambu", "print status", "printer status", "filament", "printing progress",
        "nozzle", "bed temp",
    ],
    "calendar": [
        "calendar", "event", "schedule a", "meeting", "appointment", "free time",
        "am i busy", "my schedule",
    ],
    "email": ["email", "gmail", "inbox", "draft", "unread mail", "my mail"],
    "weather": [
        "weather", "forecast", "temperature", "rain", "raining", "snow",
        "sunny", "humidity", "how cold", "how hot", "degrees outside",
    ],
    "task": ["task", "todo", "to-do", "reminder", "remind me", "prioritize"],
    "notification": ["notification", "notify me", "alert", "dismiss"],
    "briefing": ["briefing", "daily briefing", "morning summary", "summary of my day"],
    "expert_reasoning": [
        "expert mode", "claude code", "analyze this code", "implement this",
        "debug this", "review my code", "fix this code", "expert reasoning",
    ],
    "git": [
        "git", "commit", "branch", "push", "pull request", "merge", "repo",
        "repository", "diff", "git status", "git log",
    ],
    "printing": ["material profile", "filament profile", "print history", "3d print"],
    "research": ["research", "save this resource", "research note", "reference link"],
    "engineering": ["project status", "project structure", "engineering project", "cad files"],
    "automation": [
        "automation", "automate", "automation rule", "schedule this", "routine",
        "trigger when",
    ],
    "widget": [
        "widget", "widgets", "pull up", "pop up", "show my tasks", "open my tasks",
        "open the tasks widget", "show the tasks widget", "close the tasks widget",
        "hide the tasks widget", "close all widgets", "hide all widgets",
    ],
}


def _latest_user_text(messages: list[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == ChatRole.USER:
            return (message.content or "").lower()
    return ""


def _mcp_name_matches(text: str, capability: Capability) -> bool:
    """No curated keyword list for arbitrary MCP servers — fall back to
    checking whether any distinctive word from the capability's own name
    shows up in the message."""
    words = re.findall(r"[a-z]{4,}", capability.name.lower())
    return any(word in text for word in words)


def select_relevant_capabilities(
    messages: list[ChatMessage], capabilities: list[Capability]
) -> list[Capability]:
    """Return only the capabilities whose category plausibly matches the
    latest user message. Empty list means "attach no tools this turn"."""
    text = _latest_user_text(messages)
    if not text:
        return []

    relevant = []
    for capability in capabilities:
        keywords = _CATEGORY_KEYWORDS.get(capability.category)
        if keywords is not None:
            if any(phrase in text for phrase in keywords):
                relevant.append(capability)
        elif _mcp_name_matches(text, capability):
            relevant.append(capability)
    return relevant
