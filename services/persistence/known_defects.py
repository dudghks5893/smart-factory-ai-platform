"""Known-defect parent/instance persistence domain and SQLAlchemy repository."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID

from sqlalchemy import Select, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from services.persistence.database import PersistenceError
from services.persistence.models import (
    KnownDefectInspectionRecord,
    KnownDefectInstanceRecord,
)
from shared.hashing import is_sha256_digest


@dataclass(frozen=True)
class KnownDefectInstanceCreate:
    """Validated compact fields for one ordered segmentation instance."""

    class_id: int
    class_name: str
    confidence: float
    bbox_x_min: float
    bbox_y_min: float
    bbox_x_max: float
    bbox_y_max: float
    mask_pixel_count: int
    mask_area_ratio: float

    # ADD 2026-08-26: Child class, spatial summary와 mask-area invariant를 검증한다.
    def validate(self, *, image_width: int, image_height: int) -> None:
        if self.class_id < 0 or not self.class_name:
            raise ValueError("Known-defect instance class must be valid.")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Known-defect instance confidence must be finite and in [0, 1].")
        coordinates = (
            self.bbox_x_min,
            self.bbox_y_min,
            self.bbox_x_max,
            self.bbox_y_max,
        )
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError("Known-defect instance bbox must contain finite coordinates.")
        if not (
            0.0 <= self.bbox_x_min <= self.bbox_x_max <= image_width
            and 0.0 <= self.bbox_y_min <= self.bbox_y_max <= image_height
        ):
            raise ValueError("Known-defect instance bbox is outside the image bounds.")
        image_area = image_width * image_height
        if not 0 < self.mask_pixel_count <= image_area:
            raise ValueError("Known-defect mask pixel count is outside the image area.")
        if not math.isfinite(self.mask_area_ratio) or not 0.0 < self.mask_area_ratio <= 1.0:
            raise ValueError("Known-defect mask area ratio must be finite and in (0, 1].")
        expected_ratio = self.mask_pixel_count / image_area
        if not math.isclose(self.mask_area_ratio, expected_ratio, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Known-defect mask area ratio does not match its pixel count.")


@dataclass(frozen=True)
class KnownDefectCreate:
    """Values required to atomically persist one YOLO inference and its children."""

    model_name: str
    task: str
    category: str
    device: str
    diagnostic_confidence: float
    inference_ms: float
    image_width: int
    image_height: int
    image_sha256: str
    model_sha256: str
    artifact_metadata_sha256: str
    dataset_manifest_sha256: str
    dataset_semantic_fingerprint_sha256: str
    instances: tuple[KnownDefectInstanceCreate, ...]

    # ADD 2026-08-26: Parent identity, runtime observation, provenance와 child set을 검증한다.
    def validate(self) -> None:
        for field, value in (
            ("model_name", self.model_name),
            ("category", self.category),
            ("device", self.device),
        ):
            if not value:
                raise ValueError(f"Known-defect {field} must not be empty.")
        if self.task != "segment":
            raise ValueError("Known-defect task must be 'segment'.")
        if not math.isfinite(self.diagnostic_confidence) or not (
            0.0 < self.diagnostic_confidence < 1.0
        ):
            raise ValueError("Known-defect diagnostic confidence must be in (0, 1).")
        if not math.isfinite(self.inference_ms) or self.inference_ms < 0.0:
            raise ValueError("Known-defect inference latency must be finite and non-negative.")
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("Known-defect image dimensions must be positive.")
        for field, digest in (
            ("image_sha256", self.image_sha256),
            ("model_sha256", self.model_sha256),
            ("artifact_metadata_sha256", self.artifact_metadata_sha256),
            ("dataset_manifest_sha256", self.dataset_manifest_sha256),
            (
                "dataset_semantic_fingerprint_sha256",
                self.dataset_semantic_fingerprint_sha256,
            ),
        ):
            if not is_sha256_digest(digest):
                raise ValueError(f"Known-defect {field} must be a SHA-256 hex digest.")
        for instance in self.instances:
            instance.validate(image_width=self.image_width, image_height=self.image_height)


@dataclass(frozen=True)
class KnownDefectInspection:
    """Transport-independent persisted known-defect parent summary."""

    id: UUID
    created_at: datetime
    model_name: str
    task: str
    category: str
    device: str
    diagnostic_confidence: float
    inference_ms: float
    image_width: int
    image_height: int
    image_sha256: str
    model_sha256: str
    artifact_metadata_sha256: str
    dataset_manifest_sha256: str
    dataset_semantic_fingerprint_sha256: str
    instance_count: int


@dataclass(frozen=True)
class KnownDefectInstance:
    """Transport-independent persisted known-defect child."""

    id: UUID
    inspection_id: UUID
    instance_index: int
    class_id: int
    class_name: str
    confidence: float
    bbox_x_min: float
    bbox_y_min: float
    bbox_x_max: float
    bbox_y_max: float
    mask_pixel_count: int
    mask_area_ratio: float


@dataclass(frozen=True)
class KnownDefectInspectionDetail:
    """One persisted parent and all children in deterministic inference order."""

    inspection: KnownDefectInspection
    instances: tuple[KnownDefectInstance, ...]


@dataclass(frozen=True)
class KnownDefectInspectionPage:
    """Parent-only offset page with limit+1 has-more metadata."""

    items: tuple[KnownDefectInspection, ...]
    limit: int
    offset: int
    has_more: bool


class KnownDefectRepository(Protocol):
    """Persistence contract consumed by YOLO routes and test doubles."""

    def check_ready(self) -> None:
        """Fail when parent or child schema cannot serve queries."""
        ...

    def create(self, values: KnownDefectCreate) -> KnownDefectInspectionDetail:
        """Atomically commit one parent and all ordered children."""
        ...

    def get(self, inspection_id: UUID) -> KnownDefectInspectionDetail | None:
        """Return one parent with ordered children or None."""
        ...

    def list(self, *, limit: int, offset: int) -> KnownDefectInspectionPage:
        """Return newest parent summaries without hydrating children."""
        ...


class SqlAlchemyKnownDefectRepository:
    """Request-isolated SQLAlchemy known-defect repository."""

    # ADD 2026-08-26: Request work unit을 생성할 Session factory를 보관한다.
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    # ADD 2026-08-26: Parent와 child migration schema를 readiness 전에 확인한다.
    def check_ready(self) -> None:
        with self._session_factory() as session:
            try:
                session.execute(select(KnownDefectInspectionRecord.id).limit(1))
                session.execute(select(KnownDefectInstanceRecord.id).limit(1))
            except SQLAlchemyError as exc:
                raise PersistenceError("Known-defect schema readiness check failed.") from exc

    # ADD 2026-08-26: Parent와 0..N ordered child를 한 transaction으로 commit한다.
    # MODIFY 2026-08-26: Caller-owned aggregate helper를 standalone commit에서도 재사용한다.
    def create(self, values: KnownDefectCreate) -> KnownDefectInspectionDetail:
        with self._session_factory() as session:
            try:
                detail = add_known_defect_inspection(session, values)
                session.commit()
            except SQLAlchemyError as exc:
                session.rollback()
                raise PersistenceError("Known-defect inspection insert failed.") from exc
        return detail

    # ADD 2026-08-26: Parent lookup과 ordered child query를 한 read Session에서 수행한다.
    def get(self, inspection_id: UUID) -> KnownDefectInspectionDetail | None:
        with self._session_factory() as session:
            try:
                parent = session.get(KnownDefectInspectionRecord, inspection_id)
                if parent is None:
                    return None
                child_statement = (
                    select(KnownDefectInstanceRecord)
                    .where(KnownDefectInstanceRecord.inspection_id == inspection_id)
                    .order_by(KnownDefectInstanceRecord.instance_index.asc())
                )
                children = tuple(session.scalars(child_statement))
            except SQLAlchemyError as exc:
                raise PersistenceError("Known-defect inspection lookup failed.") from exc
            return KnownDefectInspectionDetail(
                inspection=_to_inspection(parent),
                instances=tuple(_to_instance(child) for child in children),
            )

    # ADD 2026-08-26: Parent-only newest-first query로 N+1과 child hydration을 피한다.
    def list(self, *, limit: int, offset: int) -> KnownDefectInspectionPage:
        statement: Select[tuple[KnownDefectInspectionRecord]] = (
            select(KnownDefectInspectionRecord)
            .order_by(
                KnownDefectInspectionRecord.created_at.desc(),
                KnownDefectInspectionRecord.id.desc(),
            )
            .offset(offset)
            .limit(limit + 1)
        )
        with self._session_factory() as session:
            try:
                records = tuple(session.scalars(statement))
            except SQLAlchemyError as exc:
                raise PersistenceError("Known-defect history lookup failed.") from exc
        return KnownDefectInspectionPage(
            items=tuple(_to_inspection(record) for record in records[:limit]),
            limit=limit,
            offset=offset,
            has_more=len(records) > limit,
        )


# ADD 2026-08-26: ORM parent를 timezone-aware immutable domain summary로 변환한다.
def _to_inspection(record: KnownDefectInspectionRecord) -> KnownDefectInspection:
    created_at = record.created_at
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    else:
        created_at = created_at.astimezone(UTC)
    return KnownDefectInspection(
        id=record.id,
        created_at=created_at,
        model_name=record.model_name,
        task=record.task,
        category=record.category,
        device=record.device,
        diagnostic_confidence=record.diagnostic_confidence,
        inference_ms=record.inference_ms,
        image_width=record.image_width,
        image_height=record.image_height,
        image_sha256=record.image_sha256,
        model_sha256=record.model_sha256,
        artifact_metadata_sha256=record.artifact_metadata_sha256,
        dataset_manifest_sha256=record.dataset_manifest_sha256,
        dataset_semantic_fingerprint_sha256=(record.dataset_semantic_fingerprint_sha256),
        instance_count=record.instance_count,
    )


# ADD 2026-08-26: ORM child를 stable inference order가 포함된 immutable domain value로 변환한다.
def _to_instance(record: KnownDefectInstanceRecord) -> KnownDefectInstance:
    return KnownDefectInstance(
        id=record.id,
        inspection_id=record.inspection_id,
        instance_index=record.instance_index,
        class_id=record.class_id,
        class_name=record.class_name,
        confidence=record.confidence,
        bbox_x_min=record.bbox_x_min,
        bbox_y_min=record.bbox_y_min,
        bbox_x_max=record.bbox_x_max,
        bbox_y_max=record.bbox_y_max,
        mask_pixel_count=record.mask_pixel_count,
        mask_area_ratio=record.mask_area_ratio,
    )


# ADD 2026-08-26: Caller-owned transaction에 YOLO aggregate를 flush해 atomic flow에서 재사용한다.
def add_known_defect_inspection(
    session: Session,
    values: KnownDefectCreate,
) -> KnownDefectInspectionDetail:
    """Validate and flush one known-defect aggregate without committing it."""
    values.validate()
    parent = KnownDefectInspectionRecord(
        model_name=values.model_name,
        task=values.task,
        category=values.category,
        device=values.device,
        diagnostic_confidence=values.diagnostic_confidence,
        inference_ms=values.inference_ms,
        image_width=values.image_width,
        image_height=values.image_height,
        image_sha256=values.image_sha256,
        model_sha256=values.model_sha256,
        artifact_metadata_sha256=values.artifact_metadata_sha256,
        dataset_manifest_sha256=values.dataset_manifest_sha256,
        dataset_semantic_fingerprint_sha256=values.dataset_semantic_fingerprint_sha256,
        instance_count=len(values.instances),
    )
    session.add(parent)
    session.flush()
    children = tuple(
        KnownDefectInstanceRecord(
            inspection_id=parent.id,
            instance_index=index,
            **vars(instance),
        )
        for index, instance in enumerate(values.instances)
    )
    session.add_all(children)
    session.flush()
    return KnownDefectInspectionDetail(
        inspection=_to_inspection(parent),
        instances=tuple(_to_instance(child) for child in children),
    )
