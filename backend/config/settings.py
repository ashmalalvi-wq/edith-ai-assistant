"""
Centralized application configuration, loaded from environment variables.

Every setting used anywhere in the backend should be read from this module
instead of calling os.getenv() directly elsewhere. This keeps configuration
discoverable in one place and makes it easy to add new settings as future
phases (memory, MCP, automation, etc.) are built.
"""
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/config/settings.py -> backend/ -> project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- General ---
    app_name: str = "E.D.I.T.H"
    app_full_name: str = "Even Dead, I'm The Hero"
    environment: Literal["development", "production"] = "development"
    log_level: str = "INFO"

    # --- Persona ---
    # Injected as a system message ahead of every conversation so the model
    # answers in character as EDITH, regardless of which provider is active.
    system_prompt: str = (
        "You are E.D.I.T.H., a private, local-first AI assistant for your one "
        "authorized user. You are professional, calm, precise, and efficient — "
        "no filler, no over-explaining, no roleplaying, no fictional framing. "
        "You take initiative on tasks within your access and flag risks plainly. "
        "You never pretend to have capabilities, data, people, events, or context "
        "that do not actually exist. If you have not been given relevant memory, "
        "calendar, email, or system context for this turn, say so plainly or ask "
        "for it — never invent people, meetings, locations, projects, or history "
        "to sound natural. Only reference: information explicitly provided in "
        "this conversation, memory context you were actually given, connected-"
        "service data you were actually given (calendar, email, etc.), and "
        "verified system/tool state. When you have nothing relevant to draw on, "
        "answer from general knowledge or ask a clarifying question instead of "
        "fabricating specifics. "
        "When asked what your tasks/to-do list are, call task.list and read back "
        "each task's title in plain language (it defaults to open tasks only; "
        "pass status='all' or 'completed' if that's specifically requested). "
        "When asked for tasks by priority (e.g. 'high priority tasks', 'medium/"
        "moderate priority tasks', 'low priority tasks'), call task.list with "
        "priority set to exactly 'high', 'medium', or 'low' to match, and report "
        "only tasks of that priority — say so plainly if none match."
    )

    # --- API server ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    # Includes "null" because Electron's renderer, when loading the built
    # frontend via file://, sends `Origin: null` on fetch() calls — there is
    # no http(s) origin to name. Electron's dev mode (ELECTRON_START_URL)
    # loads the Vite dev server instead, which is already covered by
    # localhost:5173 below.
    cors_origins: str = "http://localhost:5173,http://localhost:3000,null"

    # --- AI provider selection ---
    # Which provider is used when a request doesn't explicitly specify one.
    default_ai_provider: Literal["ollama", "openai", "anthropic"] = "ollama"

    # --- Ollama ---
    ollama_base_url: str = "http://localhost:11434"
    ollama_default_model: str = "llama3.1:8b"
    ollama_request_timeout: int = 120

    # --- OpenAI (placeholder for future phase) ---
    openai_api_key: str | None = None
    openai_default_model: str = "gpt-4o-mini"

    # --- Anthropic (placeholder for future phase) ---
    anthropic_api_key: str | None = None
    anthropic_default_model: str = "claude-sonnet-4-5"

    # --- Structured memory (PostgreSQL) ---
    database_url: str = (
        "postgresql+asyncpg://edith:edith@localhost:5432/edith"
    )

    # --- Semantic memory (Qdrant) ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "edith_memory"
    qdrant_vector_size: int = 768  # matches nomic-embed-text's output dimension

    # --- Embeddings ---
    # Which embedding backend to use. Only "ollama" is implemented in Phase 2.
    embedding_provider: Literal["ollama"] = "ollama"
    embedding_model: str = "nomic-embed-text"

    # --- Memory retrieval tuning ---
    memory_search_top_k: int = 5
    memory_min_similarity: float = 0.5

    # --- Tools (Phase 3) ---
    # Whether this process can act on the host (filesystem/terminal/VS Code/
    # local MCP servers). True for a natively-run backend (uvicorn); the
    # Dockerized backend service sets this false since it can't see the host.
    host_mode: bool = True

    # Comma-separated absolute paths the filesystem/terminal/VS Code tools may
    # touch. Defaults to just the EDITH project root — safe out of the box,
    # extend via .env (e.g. a Fusion 360 projects folder) as needed.
    approved_directories: str = str(_PROJECT_ROOT)

    terminal_shell: Literal["powershell", "cmd"] = "powershell"
    terminal_timeout_seconds: int = 30

    vscode_command: str = "code"

    # --- Productivity capabilities (Phase 5) ---

    # Fernet key (32 url-safe base64 bytes) used to encrypt OAuth tokens at
    # rest in oauth_credentials. Generate with:
    #   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    # Without this set, Gmail/Calendar report "not configured" — same pattern
    # as OPENAI_API_KEY/ANTHROPIC_API_KEY being optional.
    encryption_key: str | None = None

    # Google OAuth (Gmail + Calendar share one app registration/consent screen).
    google_oauth_client_id: str | None = None
    google_oauth_client_secret: str | None = None
    google_oauth_redirect_uri: str = "http://localhost:8000/api/integrations/google/oauth/callback"

    # Weather (Open-Meteo — no API key required).
    weather_latitude: float | None = None
    weather_longitude: float | None = None
    weather_location_name: str = "your location"

    # Notifications
    quiet_hours_start: str = "22:00"  # HH:MM, 24h, local time
    quiet_hours_end: str = "07:00"

    # --- Expert Reasoning Agent ---
    # Which ExpertReasoningProvider backs expert.analyze/expert.implement.
    # Only "claude_code" is implemented; the others are placeholders,
    # same pattern as default_ai_provider/OpenAI/Anthropic being optional.
    default_expert_reasoning_provider: Literal["claude_code", "claude_api", "openai_reasoning", "local_reasoning"] = (
        "claude_code"
    )
    claude_code_command: str = "claude"
    claude_code_default_model: str | None = None  # None = Claude Code's own default
    claude_code_timeout_seconds: int = 300

    # --- Engineering capabilities (Phase 6) ---
    # Bambu printer LAN access (local MQTTS — see printer's network settings
    # for the access code). All optional; without them, bambu.* tools report
    # "not configured", same pattern as weather/Gmail before their settings
    # are supplied. Coded against Bambu's documented LAN protocol but never
    # verified against real hardware — no printer exists in this environment.
    bambu_printer_ip: str | None = None
    bambu_access_code: str | None = None
    bambu_serial_number: str | None = None
    bambu_mqtt_timeout_seconds: int = 10

    # Name of the MCP server (registered via /api/mcp/servers) whose tools
    # back the fusion.* capability. No official Fusion 360 MCP server
    # exists; this just names which registered server to look for.
    fusion_mcp_server_name: str = "fusion"

    # --- Automation (Phase 7) ---
    # Global kill switch — checked before every rule execution, regardless
    # of individual rule.enabled state. "Users should be able to disable
    # all automations instantly" (spec item 12).
    automation_enabled: bool = True
    automation_max_runs_per_minute: int = 20  # rate limit across all rules combined
    automation_max_trigger_depth: int = 5  # a rule's actions may trigger other rules, capped this deep
    automation_default_timezone: str = "UTC"
    automation_action_timeout_seconds: int = 120  # per-action ceiling, on top of whatever the tool itself enforces
    # Root directory watchdog watches for file-creation triggers. Defaults
    # to the first approved directory (same sandbox everything else uses).
    automation_watch_directory: str | None = None

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def approved_directories_list(self) -> list[str]:
        return [d.strip() for d in self.approved_directories.split(",") if d.strip()]


@lru_cache
def get_settings() -> Settings:
    """Settings are cached so the .env file is only parsed once per process."""
    return Settings()
