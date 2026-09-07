"""
3D printing knowledge store — CRUD for material_profiles/print_history,
same explicit-AsyncSession pattern as structured_store.py. Lives under
backend/memory/ (not backend/capabilities/) because it's structured
storage like tasks/projects, not something with its own execution/
permission model or external protocol — the capability wrapper
(capabilities/tools/builtin/printing_tool.py) is what exposes this to
ToolManager/CapabilityManager.
"""
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.exceptions import NotFoundError
from backend.models.db_models import MaterialProfile, PrintHistory

# ---------------------------------------------------------------- Material profiles

async def create_material_profile(
    session: AsyncSession,
    *,
    name: str,
    material_type: str,
    nozzle_temp_c: int | None = None,
    bed_temp_c: int | None = None,
    layer_height_mm: float | None = None,
    print_speed_mm_s: int | None = None,
    supports: bool = False,
    nozzle_size_mm: float | None = None,
    notes: str | None = None,
) -> MaterialProfile:
    profile = MaterialProfile(
        name=name,
        material_type=material_type,
        nozzle_temp_c=nozzle_temp_c,
        bed_temp_c=bed_temp_c,
        layer_height_mm=layer_height_mm,
        print_speed_mm_s=print_speed_mm_s,
        supports=supports,
        nozzle_size_mm=nozzle_size_mm,
        notes=notes,
    )
    session.add(profile)
    await session.commit()
    await session.refresh(profile)
    return profile


async def list_material_profiles(session: AsyncSession, *, material_type: str | None = None) -> list[MaterialProfile]:
    query = select(MaterialProfile).order_by(MaterialProfile.updated_at.desc())
    if material_type:
        query = query.where(MaterialProfile.material_type == material_type)
    result = await session.execute(query)
    return list(result.scalars().all())


async def get_material_profile(session: AsyncSession, profile_id: uuid.UUID) -> MaterialProfile:
    profile = await session.get(MaterialProfile, profile_id)
    if profile is None:
        raise NotFoundError("MaterialProfile", str(profile_id))
    return profile


async def delete_material_profile(session: AsyncSession, profile_id: uuid.UUID) -> None:
    profile = await get_material_profile(session, profile_id)
    await session.delete(profile)
    await session.commit()


# ---------------------------------------------------------------- Print history

async def create_print_history_entry(
    session: AsyncSession,
    *,
    file_name: str,
    success: bool,
    material_profile_id: uuid.UUID | None = None,
    project_id: uuid.UUID | None = None,
    notes: str | None = None,
) -> PrintHistory:
    entry = PrintHistory(
        file_name=file_name,
        success=success,
        material_profile_id=material_profile_id,
        project_id=project_id,
        notes=notes,
    )
    session.add(entry)
    await session.commit()
    await session.refresh(entry)
    return entry


async def list_print_history(
    session: AsyncSession, *, project_id: uuid.UUID | None = None, success: bool | None = None, limit: int = 50
) -> list[PrintHistory]:
    query = select(PrintHistory).order_by(PrintHistory.printed_at.desc()).limit(limit)
    if project_id:
        query = query.where(PrintHistory.project_id == project_id)
    if success is not None:
        query = query.where(PrintHistory.success == success)
    result = await session.execute(query)
    return list(result.scalars().all())
