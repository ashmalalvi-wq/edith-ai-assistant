"""Research document store — CRUD for research_documents, same pattern as materials.py."""
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.exceptions import NotFoundError
from backend.models.db_models import ResearchDocument


async def create_research_document(
    session: AsyncSession,
    *,
    title: str,
    url: str | None = None,
    summary: str | None = None,
    tags: list[str] | None = None,
    project_id: uuid.UUID | None = None,
) -> ResearchDocument:
    document = ResearchDocument(title=title, url=url, summary=summary, tags=tags or [], project_id=project_id)
    session.add(document)
    await session.commit()
    await session.refresh(document)
    return document


async def list_research_documents(
    session: AsyncSession, *, project_id: uuid.UUID | None = None, search: str | None = None
) -> list[ResearchDocument]:
    query = select(ResearchDocument).order_by(ResearchDocument.created_at.desc())
    if project_id:
        query = query.where(ResearchDocument.project_id == project_id)
    if search:
        query = query.where(ResearchDocument.title.ilike(f"%{search}%"))
    result = await session.execute(query)
    return list(result.scalars().all())


async def get_research_document(session: AsyncSession, document_id: uuid.UUID) -> ResearchDocument:
    document = await session.get(ResearchDocument, document_id)
    if document is None:
        raise NotFoundError("ResearchDocument", str(document_id))
    return document


async def link_to_project(session: AsyncSession, document_id: uuid.UUID, project_id: uuid.UUID) -> ResearchDocument:
    document = await get_research_document(session, document_id)
    document.project_id = project_id
    await session.commit()
    await session.refresh(document)
    return document
