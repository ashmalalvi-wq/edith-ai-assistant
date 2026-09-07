"""Pydantic schemas for Phase 6's engineering APIs (materials, print history, research documents)."""
import uuid
from datetime import datetime

from pydantic import BaseModel, Field


# --- Materials / print history ---

class MaterialProfileCreate(BaseModel):
    name: str
    material_type: str
    nozzle_temp_c: int | None = None
    bed_temp_c: int | None = None
    layer_height_mm: float | None = None
    print_speed_mm_s: int | None = None
    supports: bool = False
    nozzle_size_mm: float | None = None
    notes: str | None = None


class MaterialProfileOut(BaseModel):
    id: uuid.UUID
    name: str
    material_type: str
    nozzle_temp_c: int | None
    bed_temp_c: int | None
    layer_height_mm: float | None
    print_speed_mm_s: int | None
    supports: bool
    nozzle_size_mm: float | None
    notes: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class PrintHistoryCreate(BaseModel):
    file_name: str
    success: bool
    material_profile_id: uuid.UUID | None = None
    project_id: uuid.UUID | None = None
    notes: str | None = None


class PrintHistoryOut(BaseModel):
    id: uuid.UUID
    file_name: str
    success: bool
    material_profile_id: uuid.UUID | None
    project_id: uuid.UUID | None
    notes: str | None
    printed_at: datetime

    model_config = {"from_attributes": True}


# --- Research documents ---

class ResearchDocumentCreate(BaseModel):
    title: str
    url: str | None = None
    summary: str | None = None
    tags: list[str] = Field(default_factory=list)
    project_id: uuid.UUID | None = None


class ResearchDocumentOut(BaseModel):
    id: uuid.UUID
    title: str
    url: str | None
    summary: str | None
    tags: list
    project_id: uuid.UUID | None
    created_at: datetime

    model_config = {"from_attributes": True}


# --- Aggregated status ---

class EngineeringStatusOut(BaseModel):
    """Generic wrapper for capability-backed status reads (CAD/printer/git)."""

    success: bool
    data: dict | list | None = None
    error: str | None = None
