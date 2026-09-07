"""
Async SQLAlchemy engine/session setup for the structured memory store.

Every other module (structured_store.py, API routes) should get sessions
through `get_session()` (a FastAPI dependency) rather than constructing
their own engine, so there's exactly one connection pool per process.
"""
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.config.settings import get_settings
from backend.models.db_models import Base

_engine = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(settings.database_url, echo=False, future=True)
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(get_engine(), expire_on_commit=False)
    return _session_factory


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency — yields a session and guarantees it's closed."""
    factory = get_session_factory()
    async with factory() as session:
        yield session


@asynccontextmanager
async def session_scope() -> AsyncGenerator[AsyncSession, None]:
    """Context-manager form for use outside of FastAPI's DI (e.g. the memory pipeline)."""
    factory = get_session_factory()
    async with factory() as session:
        yield session


async def init_models() -> None:
    """
    Creates any tables missing from the database.

    The canonical schema lives in database/postgres/init/001_init.sql (applied
    automatically by the Postgres container on first boot). This call is a
    safety net for dev setups where that init script didn't run (e.g. a
    pre-existing Postgres volume, or the SQLite fallback used in tests).
    """
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def dispose_engine() -> None:
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None
