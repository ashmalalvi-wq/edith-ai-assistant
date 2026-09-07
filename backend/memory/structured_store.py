"""
Structured memory store — all PostgreSQL reads/writes for projects,
conversations, messages, memories, tasks, and user preferences.

Every function takes an AsyncSession explicitly rather than pulling one
from a global, so this module is easy to unit test against SQLite
(see backend/tests/conftest.py) and easy to reuse from both API routes
and the memory pipeline.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.exceptions import NotFoundError
from backend.models.db_models import Conversation, Memory, Message, Project, Task, User

# ---------------------------------------------------------------- Users

async def get_or_create_default_user(session: AsyncSession) -> User:
    result = await session.execute(select(User).limit(1))
    user = result.scalar_one_or_none()
    if user is None:
        user = User()
        session.add(user)
        await session.commit()
        await session.refresh(user)
    return user


async def update_user_preferences(session: AsyncSession, **fields) -> User:
    user = await get_or_create_default_user(session)
    for key, value in fields.items():
        if value is not None:
            setattr(user, key, value)
    await session.commit()
    await session.refresh(user)
    return user


# ---------------------------------------------------------------- Projects

async def create_project(session: AsyncSession, *, name: str, description: str | None, status: str) -> Project:
    project = Project(name=name, description=description, status=status)
    session.add(project)
    await session.commit()
    await session.refresh(project)
    return project


async def list_projects(session: AsyncSession) -> list[Project]:
    result = await session.execute(select(Project).order_by(Project.updated_at.desc()))
    return list(result.scalars().all())


async def get_project(session: AsyncSession, project_id: uuid.UUID) -> Project:
    project = await session.get(Project, project_id)
    if project is None:
        raise NotFoundError("Project", str(project_id))
    return project


async def update_project(session: AsyncSession, project_id: uuid.UUID, **fields) -> Project:
    project = await get_project(session, project_id)
    for key, value in fields.items():
        if value is not None:
            setattr(project, key, value)
    await session.commit()
    await session.refresh(project)
    return project


async def delete_project(session: AsyncSession, project_id: uuid.UUID) -> None:
    project = await get_project(session, project_id)
    await session.delete(project)
    await session.commit()


# ---------------------------------------------------------------- Conversations

async def create_conversation(
    session: AsyncSession,
    *,
    title: str,
    project_id: uuid.UUID | None,
    ai_provider: str,
    ai_model: str | None,
) -> Conversation:
    conversation = Conversation(
        title=title, project_id=project_id, ai_provider=ai_provider, ai_model=ai_model
    )
    session.add(conversation)
    await session.commit()
    await session.refresh(conversation)
    return conversation


async def list_conversations(session: AsyncSession, *, search: str | None = None) -> list[Conversation]:
    query = select(Conversation).order_by(Conversation.updated_at.desc())
    if search:
        query = query.where(Conversation.title.ilike(f"%{search}%"))
    result = await session.execute(query)
    return list(result.scalars().all())


async def get_conversation(session: AsyncSession, conversation_id: uuid.UUID) -> Conversation:
    conversation = await session.get(Conversation, conversation_id)
    if conversation is None:
        raise NotFoundError("Conversation", str(conversation_id))
    return conversation


async def get_conversation_with_messages(session: AsyncSession, conversation_id: uuid.UUID) -> Conversation:
    from sqlalchemy.orm import selectinload

    query = (
        select(Conversation)
        .options(selectinload(Conversation.messages))
        .where(Conversation.id == conversation_id)
    )
    result = await session.execute(query)
    conversation = result.scalar_one_or_none()
    if conversation is None:
        raise NotFoundError("Conversation", str(conversation_id))
    return conversation


async def update_conversation(session: AsyncSession, conversation_id: uuid.UUID, **fields) -> Conversation:
    conversation = await get_conversation(session, conversation_id)
    for key, value in fields.items():
        if value is not None:
            setattr(conversation, key, value)
    conversation.updated_at = datetime.now(timezone.utc)
    await session.commit()
    await session.refresh(conversation)
    return conversation


async def delete_conversation(session: AsyncSession, conversation_id: uuid.UUID) -> None:
    conversation = await get_conversation(session, conversation_id)
    await session.delete(conversation)
    await session.commit()


async def add_message(
    session: AsyncSession, *, conversation_id: uuid.UUID, role: str, content: str
) -> Message:
    message = Message(conversation_id=conversation_id, role=role, content=content)
    session.add(message)
    # Touch the parent conversation so "recently updated" ordering reflects new activity.
    conversation = await get_conversation(session, conversation_id)
    conversation.updated_at = datetime.now(timezone.utc)
    await session.commit()
    await session.refresh(message)
    return message


# ---------------------------------------------------------------- Memories

async def create_memory(
    session: AsyncSession,
    *,
    type: str,
    content: str,
    project_id: uuid.UUID | None = None,
    source_conversation_id: uuid.UUID | None = None,
    qdrant_point_id: uuid.UUID | None = None,
    expires_at: datetime | None = None,
) -> Memory:
    memory = Memory(
        type=type,
        content=content,
        project_id=project_id,
        source_conversation_id=source_conversation_id,
        qdrant_point_id=qdrant_point_id,
        expires_at=expires_at,
    )
    session.add(memory)
    await session.commit()
    await session.refresh(memory)
    return memory


async def list_memories(
    session: AsyncSession, *, type: str | None = None, project_id: uuid.UUID | None = None
) -> list[Memory]:
    query = select(Memory).order_by(Memory.updated_at.desc())
    if type:
        query = query.where(Memory.type == type)
    if project_id:
        query = query.where(Memory.project_id == project_id)
    result = await session.execute(query)
    return list(result.scalars().all())


async def get_memory(session: AsyncSession, memory_id: uuid.UUID) -> Memory:
    memory = await session.get(Memory, memory_id)
    if memory is None:
        raise NotFoundError("Memory", str(memory_id))
    return memory


async def get_memories_by_ids(session: AsyncSession, memory_ids: list[uuid.UUID]) -> list[Memory]:
    if not memory_ids:
        return []
    result = await session.execute(select(Memory).where(Memory.id.in_(memory_ids)))
    return list(result.scalars().all())


async def update_memory(session: AsyncSession, memory_id: uuid.UUID, **fields) -> Memory:
    memory = await get_memory(session, memory_id)
    for key, value in fields.items():
        if value is not None:
            setattr(memory, key, value)
    await session.commit()
    await session.refresh(memory)
    return memory


async def delete_memory(session: AsyncSession, memory_id: uuid.UUID) -> Memory:
    memory = await get_memory(session, memory_id)
    await session.delete(memory)
    await session.commit()
    return memory


# ---------------------------------------------------------------- Tasks

async def create_task(
    session: AsyncSession,
    *,
    title: str,
    description: str | None,
    project_id: uuid.UUID | None,
    status: str,
    priority: str = "medium",
    due_date: datetime | None,
) -> Task:
    task = Task(
        title=title,
        description=description,
        project_id=project_id,
        status=status,
        priority=priority,
        due_date=due_date,
    )
    session.add(task)
    await session.commit()
    await session.refresh(task)
    return task


async def list_tasks(
    session: AsyncSession,
    *,
    project_id: uuid.UUID | None = None,
    status: str | None = None,
    priority: str | None = None,
) -> list[Task]:
    query = select(Task).order_by(Task.created_at.desc())
    if project_id:
        query = query.where(Task.project_id == project_id)
    if status:
        query = query.where(Task.status == status)
    if priority:
        query = query.where(Task.priority == priority)
    result = await session.execute(query)
    return list(result.scalars().all())


async def get_task(session: AsyncSession, task_id: uuid.UUID) -> Task:
    task = await session.get(Task, task_id)
    if task is None:
        raise NotFoundError("Task", str(task_id))
    return task


async def update_task(session: AsyncSession, task_id: uuid.UUID, **fields) -> Task:
    task = await get_task(session, task_id)
    for key, value in fields.items():
        if value is not None:
            setattr(task, key, value)
    await session.commit()
    await session.refresh(task)
    return task


async def delete_task(session: AsyncSession, task_id: uuid.UUID) -> None:
    task = await get_task(session, task_id)
    await session.delete(task)
    await session.commit()
