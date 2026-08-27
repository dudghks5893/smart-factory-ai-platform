"""Manifest schema for self-contained supervised-derived segmentation packages."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

from ml.datasets.segmentation_annotations import DERIVED_SPLITS


@dataclass(frozen=True)
class DerivedManifestRecord:
    """One copied training sample with immutable source and polygon lineage."""

    dataset_name: str
    dataset_version: str
    derived_task: str
    source_manifest_sha256: str
    source_split: str
    source_manifest_split: str
    source_image_path: str
    source_mask_path: str
    category: str
    sample_id: str
    defect_type: str
    target_class: str
    target_class_id: str
    derived_split: str
    is_negative: bool
    image_width: int
    image_height: int
    image_path: str
    label_path: str
    image_sha256: str
    mask_sha256: str
    polygon_count: int
    component_count: int
    hole_count: int
    polygon_vertex_count: int
    round_trip_iou: str
    pixel_precision: str
    pixel_recall: str


DERIVED_MANIFEST_FIELDS = tuple(DerivedManifestRecord.__dataclass_fields__)


# ADD 2026-08-25: Stable schema의 source-lineage manifest를 기록한다.
def write_derived_manifest(records: list[DerivedManifestRecord], path: Path) -> None:
    """Write deterministic rows and columns for training and audit consumers."""
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=DERIVED_MANIFEST_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


# ADD 2026-08-25: Manifest를 복원한다. → MODIFY 2026-08-28: 제외 split을 객체화 전에 건너뛴다.
def read_derived_manifest(
    path: Path,
    *,
    allowed_splits: set[str] | None = None,
) -> list[DerivedManifestRecord]:
    """Read the generated manifest without accepting missing or extra columns."""
    if allowed_splits is not None and (
        not allowed_splits or not allowed_splits.issubset(DERIVED_SPLITS)
    ):
        raise ValueError("Allowed derived Manifest splits are invalid.")
    records: list[DerivedManifestRecord] = []
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != list(DERIVED_MANIFEST_FIELDS):
            raise ValueError("Unexpected C2-1 derived manifest schema.")
        for row in reader:
            derived_split = row.get("derived_split")
            if derived_split is None:
                raise ValueError("Derived manifest row contains a missing split value.")
            if allowed_splits is not None and derived_split not in allowed_splits:
                continue
            values = {
                field: row[field] for field in DERIVED_MANIFEST_FIELDS if row[field] is not None
            }
            if len(values) != len(DERIVED_MANIFEST_FIELDS):
                raise ValueError("Derived manifest row contains a missing value.")
            if values["is_negative"] not in {"True", "False"}:
                raise ValueError("Derived manifest boolean must be True or False.")
            records.append(
                DerivedManifestRecord(
                    dataset_name=values["dataset_name"],
                    dataset_version=values["dataset_version"],
                    derived_task=values["derived_task"],
                    source_manifest_sha256=values["source_manifest_sha256"],
                    source_split=values["source_split"],
                    source_manifest_split=values["source_manifest_split"],
                    source_image_path=values["source_image_path"],
                    source_mask_path=values["source_mask_path"],
                    category=values["category"],
                    sample_id=values["sample_id"],
                    defect_type=values["defect_type"],
                    target_class=values["target_class"],
                    target_class_id=values["target_class_id"],
                    derived_split=values["derived_split"],
                    is_negative=values["is_negative"] == "True",
                    image_width=int(values["image_width"]),
                    image_height=int(values["image_height"]),
                    image_path=values["image_path"],
                    label_path=values["label_path"],
                    image_sha256=values["image_sha256"],
                    mask_sha256=values["mask_sha256"],
                    polygon_count=int(values["polygon_count"]),
                    component_count=int(values["component_count"]),
                    hole_count=int(values["hole_count"]),
                    polygon_vertex_count=int(values["polygon_vertex_count"]),
                    round_trip_iou=values["round_trip_iou"],
                    pixel_precision=values["pixel_precision"],
                    pixel_recall=values["pixel_recall"],
                )
            )
    return records
