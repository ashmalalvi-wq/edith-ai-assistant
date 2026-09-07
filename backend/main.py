"""
E.D.I.T.H backend entry point.

Run with:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
"""
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.routes import (
    attachments,
    automation,
    briefing,
    capabilities,
    chat,
    conversations,
    engineering,
    health,
    integrations,
    mcp_servers,
    memory,
    notifications,
    profiles,
    projects,
    providers,
    research,
    tasks,
    tools,
    users,
    workspaces,
)
from backend.capabilities.automation.engine import get_automation_engine
from backend.capabilities.automation.event_bus import EventName, get_event_bus
from backend.capabilities.automation.file_watcher import get_file_watcher
from backend.capabilities.automation.scheduler import get_automation_scheduler
from backend.capabilities.manager import get_capability_manager
from backend.capabilities.mcp.manager import get_mcp_manager
from backend.capabilities.tools.manager import get_tool_manager
from backend.config.logging_config import configure_logging
from backend.config.settings import get_settings
from backend.core.exceptions import EdithError
from backend.memory.db import init_models
from backend.memory.vector_store import get_vector_store

configure_logging()
logger = logging.getLogger(__name__)

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description=f"E.D.I.T.H — {settings.app_full_name}. Phase 6: engineering assistant.",
    version="0.6.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(EdithError)
async def edith_error_handler(request: Request, exc: EdithError) -> JSONResponse:
    logger.error("EdithError on %s %s: %s", request.method, request.url.path, exc.message)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


app.include_router(health.router)
app.include_router(chat.router)
app.include_router(attachments.router)
app.include_router(providers.router)
app.include_router(projects.router)
app.include_router(conversations.router)
app.include_router(memory.router)
app.include_router(tasks.router)
app.include_router(users.router)
app.include_router(tools.router)
app.include_router(mcp_servers.router)
app.include_router(profiles.router)
app.include_router(capabilities.router)
app.include_router(workspaces.router)
app.include_router(notifications.router)
app.include_router(briefing.router)
app.include_router(integrations.router)
app.include_router(engineering.router)
app.include_router(research.router)
app.include_router(automation.router)


@app.on_event("startup")
async def on_startup() -> None:
    logger.info("%s starting up (environment=%s)", settings.app_name, settings.environment)
    logger.info("Default AI provider: %s", settings.default_ai_provider)
    logger.info("Host mode: %s", settings.host_mode)

    # Safety net for the structured store: database/postgres/init/*.sql are
    # the canonical schema, applied automatically on first container boot.
    # This covers dev setups where that didn't happen (e.g. a pre-existing
    # Postgres volume) — create_all() is a no-op for tables that already exist.
    try:
        await init_models()
        logger.info("Structured memory store ready.")
    except Exception:
        logger.exception("Could not initialize structured memory store — is Postgres running?")

    try:
        await get_vector_store().ensure_collection()
        logger.info("Semantic memory store ready.")
    except Exception:
        logger.exception("Could not initialize semantic memory store — is Qdrant running?")

    tool_manager = get_tool_manager()
    logger.info("Tool Manager ready — %d built-in tools registered.", tool_manager.builtin_tool_count)
    get_capability_manager()  # wraps the Tool Manager as the TOOL/MCP capability sources
    logger.info("Capability Manager ready.")

    try:
        await get_mcp_manager().reload()
        logger.info("MCP servers connected per current configuration.")
    except Exception:
        logger.exception("MCP server connection pass failed on startup.")

    try:
        engine = get_automation_engine()  # subscribes to the event bus on first access
        scheduler = get_automation_scheduler()
        scheduler.start()
        await scheduler.reload()
        get_file_watcher().start()
        logger.info(
            "Automation engine ready (paused=%s); scheduler running with %d schedule-triggered rule(s).",
            engine.is_paused, len(scheduler.next_run_times()),
        )
    except Exception:
        logger.exception("Automation engine/scheduler failed to start.")

    try:
        await get_event_bus().publish(EventName.SYSTEM_STARTUP, {})
    except Exception:
        logger.exception("Failed to publish SystemStartup event.")


@app.on_event("shutdown")
async def on_shutdown() -> None:
    await get_mcp_manager().disconnect_all()
    get_automation_scheduler().shutdown()
    get_file_watcher().stop()
