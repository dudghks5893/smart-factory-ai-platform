"""Integrity validation for generated MVTec AD manifests."""

from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from ml.datasets.constants import GOOD_DIR_NAME
from ml.datasets.manifest import ManifestRecord

_ALLOWED_SPLITS = {"train", "validation", "test"}
_ALLOWED_SOURCE_SPLITS = {"train", "test"}


@dataclass
class ManifestValidationReport:
    """Structured integrity result for manifest validation."""

    record_count: int
    errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors


def _validate_record_paths(
    record: ManifestRecord,
    dataset_root: Path,
    errors: list[str],
) -> None:
    image_path = dataset_root / record.image_path
    if not image_path.is_file():
        errors.append(f"missing image: {record.image_path}")
        return

    with Image.open(image_path) as image:
        if image.size != (record.width, record.height):
            errors.append(
                f"image size mismatch: {record.image_path} "
                f"manifest={record.width}x{record.height} actual={image.width}x{image.height}"
            )

    if record.label == 1:
        if not record.mask_path:
            errors.append(f"anomaly sample has no mask path: {record.sample_id}")
            return

        mask_path = dataset_root / record.mask_path
        if not mask_path.is_file():
            errors.append(f"missing mask: {record.mask_path}")
        else:
            with Image.open(mask_path) as mask:
                if mask.size != (record.width, record.height):
                    errors.append(
                        f"mask size mismatch: {record.mask_path} "
                        f"expected={record.width}x{record.height} "
                        f"actual={mask.width}x{mask.height}"
                    )
    elif record.mask_path:
        errors.append(f"normal sample unexpectedly has mask: {record.sample_id}")


def _validate_record_semantics(record: ManifestRecord, errors: list[str]) -> None:
    if record.split not in _ALLOWED_SPLITS:
        errors.append(f"invalid split: {record.sample_id} -> {record.split}")

    if record.source_split not in _ALLOWED_SOURCE_SPLITS:
        errors.append(f"invalid source_split: {record.sample_id} -> {record.source_split}")

    if record.source_split == "train" and record.split not in {"train", "validation"}:
        errors.append(
            f"official train sample assigned to invalid split: {record.sample_id} -> {record.split}"
        )

    if record.source_split == "test" and record.split != "test":
        errors.append(
            f"official test sample must remain test: {record.sample_id} -> {record.split}"
        )

    expected_label = 0 if record.defect_type == GOOD_DIR_NAME else 1
    if record.label != expected_label:
        errors.append(
            f"label/defect mismatch: {record.sample_id} "
            f"defect_type={record.defect_type} label={record.label}"
        )


def validate_manifest_records(
    records: list[ManifestRecord],
    dataset_root: Path,
) -> ManifestValidationReport:
    """Validate sample uniqueness, split semantics, paths, masks, and dimensions."""
    report = ManifestValidationReport(record_count=len(records))
    seen_sample_ids: set[str] = set()
    seen_image_paths: set[str] = set()

    for record in records:
        if record.sample_id in seen_sample_ids:
            report.errors.append(f"duplicate sample_id: {record.sample_id}")
        seen_sample_ids.add(record.sample_id)

        if record.image_path in seen_image_paths:
            report.errors.append(f"duplicate image_path: {record.image_path}")
        seen_image_paths.add(record.image_path)

        _validate_record_semantics(record, report.errors)
        _validate_record_paths(record, dataset_root, report.errors)

    return report
