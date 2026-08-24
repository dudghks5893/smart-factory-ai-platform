"""Local-only FastAPI facade over deterministic synthetic dashboard records."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query

from examples.dashboard_demo.fixtures import SYNTHETIC_INSPECTIONS
from services.api.schemas import HealthResponse, InspectionHistoryResponse, InspectionResponse

app = FastAPI(
    title="SmartFactory Dashboard Synthetic Demo API",
    description="Local portfolio visualization fixture; never a production inspection service.",
    version="1.0.0-demo",
)
_INSPECTIONS_BY_ID = {item.inspection_id: item for item in SYNTHETIC_INSPECTIONS}


# ADD 2026-08-24: Local demo process의 liveness를 production readiness와 혼동 없이 반환한다.
@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


# ADD 2026-08-24: Existing Dashboard list contract에 맞춰 synthetic records를 filter/paginate한다.
@app.get("/v1/inspections", response_model=InspectionHistoryResponse)
async def list_inspections(
    category: Annotated[str | None, Query(min_length=1, max_length=100)] = None,
    is_anomaly: bool | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> InspectionHistoryResponse:
    """Return deterministic newest-first demo records through the production response schema."""
    filtered = tuple(
        item
        for item in SYNTHETIC_INSPECTIONS
        if (category is None or item.category == category)
        and (is_anomaly is None or item.is_anomaly is is_anomaly)
    )
    page = filtered[offset : offset + limit]
    return InspectionHistoryResponse(
        items=list(page),
        limit=limit,
        offset=offset,
        returned_count=len(page),
        has_more=offset + len(page) < len(filtered),
    )


# ADD 2026-08-24: UUID로 synthetic inspection lineage detail을 조회한다.
@app.get("/v1/inspections/{inspection_id}", response_model=InspectionResponse)
async def get_inspection(inspection_id: UUID) -> InspectionResponse:
    """Return one demo detail or the same public 404 status used by an unavailable record."""
    inspection = _INSPECTIONS_BY_ID.get(inspection_id)
    if inspection is None:
        raise HTTPException(status_code=404, detail="Synthetic inspection was not found.")
    return inspection
