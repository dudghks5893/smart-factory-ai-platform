"""Deterministic synthetic inspection fixtures for the local dashboard demo."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid5

from services.api.schemas import InspectionResponse

DEMO_INSPECTION_COUNT = 100
DEMO_ANOMALY_COUNT = 12
DEMO_NORMAL_COUNT = DEMO_INSPECTION_COUNT - DEMO_ANOMALY_COUNT
DEMO_THRESHOLD = 41.2
DEMO_BASE_TIME = datetime(2026, 8, 24, 1, 0, tzinfo=UTC)
DEMO_INTERVAL = timedelta(minutes=3)
DEMO_NAMESPACE = UUID("f573a9b8-0ef0-46d5-95e4-0c7e4692b567")


# ADD 2026-08-24: Synthetic demo lineage label을 deterministic SHA-256 digest로 변환한다.
def _synthetic_digest(label: str) -> str:
    return hashlib.sha256(f"smartfactory-dashboard-demo:{label}".encode()).hexdigest()


# ADD 2026-08-24: 분산된 anomaly 위치와 score variation을 가진 synthetic inspection을 생성한다.
def build_synthetic_inspections() -> tuple[InspectionResponse, ...]:
    """Return 100 newest-first records that cannot be mistaken for runtime persistence data."""
    anomaly_indexes = {round(index * 99 / (DEMO_ANOMALY_COUNT - 1)) for index in range(12)}
    inspections: list[InspectionResponse] = []

    # Screenshot chart가 단조롭지 않도록 고정 수식으로 normal/anomaly score를 분산한다.
    for index in range(DEMO_INSPECTION_COUNT):
        is_anomaly = index in anomaly_indexes
        if is_anomaly:
            anomaly_score = 43.4 + ((index * 17) % 137) / 10
        else:
            anomaly_score = 25.2 + ((index * 37) % 141) / 10
        inspection_id = uuid5(DEMO_NAMESPACE, f"synthetic-inspection-{index:03d}")
        inspections.append(
            InspectionResponse(
                inspection_id=inspection_id,
                created_at=DEMO_BASE_TIME - index * DEMO_INTERVAL,
                model_name="patchcore-demo-synthetic",
                category="metal_nut",
                is_anomaly=is_anomaly,
                anomaly_score=anomaly_score,
                threshold=DEMO_THRESHOLD,
                comparison_operator=">",
                image_sha256=_synthetic_digest(f"image-{index:03d}"),
                image_size_bytes=180_000 + index * 97,
                content_type="image/png",
                model_sha256=_synthetic_digest("model"),
                artifact_metadata_sha256=_synthetic_digest("metadata"),
                threshold_artifact_sha256=_synthetic_digest("threshold"),
                manifest_sha256=_synthetic_digest("manifest"),
                device="demo-cpu",
            )
        )
    return tuple(inspections)


SYNTHETIC_INSPECTIONS = build_synthetic_inspections()
