"""Deterministic component-aware crop train view for C4-2C confirmation."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

from ml.datasets.segmentation_annotations import parse_yolo_segmentation_label
from ml.datasets.yolo_segmentation_manifest import DerivedManifestRecord
from ml.experiments.yolo_sampling import (
    ELIGIBLE_MULTIPLICITY,
    SAMPLING_RULE_VERSION,
    PlannedTrainView,
    plan_component_aware_train_view,
)
from ml.training.yolo_segmentation import YoloDatasetContract, validate_artifact_id
from shared.hashing import is_sha256_digest, sha256_bytes, sha256_file

CROP_TRAIN_VIEW_SCHEMA_VERSION = 1
CROP_SAMPLING_MODE = "component_aware_crop"
CROP_ORDERING_POLICY = (
    "canonical_sample_id_then_component_aware_sample_id_then_small_area_sample_id_ranking"
)
CANONICAL_PATH_PREFIX = "canonical"
GENERATED_PATH_PREFIX = "generated"


@dataclass(frozen=True)
class CropTrainViewEntry:
    """One portable ordered training entry without machine-specific absolute paths."""

    kind: str
    source_sample_id: str
    is_negative: bool
    portable_image_path: str
    source_relative_image_path: str
    generated_relative_path: str | None


@dataclass(frozen=True)
class CropProvenance:
    """One generated crop bound to its canonical source sample and content hashes."""

    source_sample_id: str
    source_relative_image_path: str
    generated_image_path: str
    generated_label_path: str
    crop_box_xyxy: tuple[int, int, int, int]
    source_width: int
    source_height: int
    crop_size: int
    retained_instance_count: int
    target_polygon_index: int
    crop_image_sha256: str
    crop_label_sha256: str


@dataclass(frozen=True)
class CropTrainViewEvidence:
    """Portable identity and exact count contract for the C4-2C training view."""

    schema_version: int
    experiment_id: str
    sampling_mode: str
    sampling_rule_version: str
    canonical_manifest_sha256: str
    crop_size: int
    canonical_entry_count: int
    canonical_positive_count: int
    canonical_negative_count: int
    component_duplicate_count: int
    crop_entry_count: int
    total_entry_count: int
    positive_exposure: int
    negative_exposure: int
    small_aware_count: int
    multi_component_count: int
    eligible_overlap_count: int
    eligible_union_count: int
    small_aware_sample_ids: tuple[str, ...]
    multi_component_sample_ids: tuple[str, ...]
    component_aware_sample_ids: tuple[str, ...]
    observed_train_small_cutoff: float
    ordering_policy: str
    entries: tuple[CropTrainViewEntry, ...]
    crops: tuple[CropProvenance, ...]
    portable_train_list_sha256: str
    train_view_fingerprint_sha256: str
    validation_used_for_sampling: bool
    test_used: bool

    # ADD 2026-08-31: Evidence를 검증한다. → MODIFY 2026-09-01: Actual entry/ID를 대조한다.
    def validate(self) -> None:
        validate_artifact_id(self.experiment_id)
        if (
            self.schema_version != CROP_TRAIN_VIEW_SCHEMA_VERSION
            or self.sampling_mode != CROP_SAMPLING_MODE
            or self.sampling_rule_version != SAMPLING_RULE_VERSION
            or self.crop_size <= 0
        ):
            raise ValueError("C4-2C crop train-view identity is invalid.")
        if self.ordering_policy != CROP_ORDERING_POLICY:
            raise ValueError("C4-2C crop train-view ordering policy changed.")
        if self.validation_used_for_sampling or self.test_used:
            raise ValueError("C4-2C crop train view must remain train-only.")
        canonical_entries = tuple(entry for entry in self.entries if entry.kind == "canonical")
        duplicate_entries = tuple(
            entry for entry in self.entries if entry.kind == "component_aware_duplicate"
        )
        crop_entries = tuple(entry for entry in self.entries if entry.kind == "small_center_crop")
        if len(canonical_entries) + len(duplicate_entries) + len(crop_entries) != len(self.entries):
            raise ValueError("C4-2C train-view contains an unknown entry kind.")
        actual_counts = (
            len(canonical_entries),
            sum(not entry.is_negative for entry in canonical_entries),
            sum(entry.is_negative for entry in canonical_entries),
            len(duplicate_entries),
            len(crop_entries),
            len(self.entries),
            sum(not entry.is_negative for entry in self.entries),
            sum(entry.is_negative for entry in self.entries),
        )
        declared_counts = (
            self.canonical_entry_count,
            self.canonical_positive_count,
            self.canonical_negative_count,
            self.component_duplicate_count,
            self.crop_entry_count,
            self.total_entry_count,
            self.positive_exposure,
            self.negative_exposure,
        )
        if actual_counts != declared_counts:
            raise ValueError("C4-2C declared counts differ from actual ordered entries.")
        if self.canonical_entry_count != (
            self.canonical_positive_count + self.canonical_negative_count
        ):
            raise ValueError("C4-2C canonical train-view counts are inconsistent.")
        if self.total_entry_count != (
            self.canonical_entry_count + self.component_duplicate_count + self.crop_entry_count
        ):
            raise ValueError("C4-2C total train-view count is inconsistent.")
        if (
            self.positive_exposure
            != (
                self.canonical_positive_count
                + self.component_duplicate_count
                + self.crop_entry_count
            )
            or self.negative_exposure != self.canonical_negative_count
        ):
            raise ValueError("C4-2C positive/negative exposure is inconsistent.")
        small_ids = set(self.small_aware_sample_ids)
        multi_ids = set(self.multi_component_sample_ids)
        eligible_ids = set(self.component_aware_sample_ids)
        if (
            len(self.crops) != self.crop_entry_count
            or len(small_ids) != self.small_aware_count
            or len(multi_ids) != self.multi_component_count
            or len(small_ids.intersection(multi_ids)) != self.eligible_overlap_count
            or len(small_ids.union(multi_ids)) != self.eligible_union_count
            or eligible_ids != small_ids.union(multi_ids)
            or self.crop_entry_count != self.small_aware_count
            or self.component_duplicate_count != self.eligible_union_count
        ):
            raise ValueError("C4-2C train-view evidence arrays are incomplete.")
        if self.multi_component_sample_ids != tuple(
            sorted(multi_ids)
        ) or self.component_aware_sample_ids != tuple(sorted(eligible_ids)):
            raise ValueError("Component-aware duplicate IDs must use sample-id order.")
        if any(entry.is_negative for entry in (*duplicate_entries, *crop_entries)):
            raise ValueError("C4-2C duplicate and crop entries must remain positive-only.")
        if not math.isfinite(self.observed_train_small_cutoff):
            raise ValueError("C4-2C small-aware cutoff must be finite.")
        for digest in (
            self.canonical_manifest_sha256,
            self.portable_train_list_sha256,
            self.train_view_fingerprint_sha256,
        ):
            if not is_sha256_digest(digest):
                raise ValueError("C4-2C train-view digest is malformed.")
        if any(
            not is_sha256_digest(crop.crop_image_sha256)
            or not is_sha256_digest(crop.crop_label_sha256)
            for crop in self.crops
        ):
            raise ValueError("C4-2C crop digest is malformed.")

    # ADD 2026-08-31: C4-2C portable evidence를 strict JSON mapping으로 변환한다.
    def to_json_dict(self) -> dict[str, object]:
        self.validate()
        payload = asdict(self)
        json.dumps(payload, sort_keys=True, allow_nan=False)
        return payload


@dataclass(frozen=True)
class CropTrainViewArtifact:
    """Generated crop inputs plus portable list and metadata evidence."""

    output_dir: Path
    train_list_path: Path
    metadata_path: Path
    evidence: CropTrainViewEvidence


@dataclass(frozen=True)
class RuntimeCropTrainViewArtifact:
    """Machine-local absolute train list derived from portable crop evidence."""

    train_list_path: Path
    source_train_list_sha256: str
    entry_count: int


# ADD 2026-08-31: Fast R17 square crop의 exclusive pixel bounds를 source 안으로 이동한다.
def crop_square_bounds(
    *,
    center_x: float,
    center_y: float,
    crop_size: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    if crop_size <= 0 or width < crop_size or height < crop_size:
        raise ValueError("Crop size must fit inside the source image dimensions.")
    half = crop_size / 2.0
    x0 = max(0, min(int(round(center_x - half)), width - crop_size))
    y0 = max(0, min(int(round(center_y - half)), height - crop_size))
    return x0, y0, x0 + crop_size, y0 + crop_size


# ADD 2026-08-31: Source label line order를 보존하며 pixel-space YOLO polygon을 읽는다.
def load_source_polygons(
    record: DerivedManifestRecord,
    *,
    dataset_root: Path,
    valid_class_ids: set[int],
) -> tuple[tuple[int, int, np.ndarray], ...]:
    if record.derived_split != "train" or record.is_negative:
        raise ValueError("C4-2C crops require positive canonical train records.")
    text = (dataset_root / record.label_path).read_text(encoding="utf-8")
    polygons = parse_yolo_segmentation_label(text, valid_class_ids=valid_class_ids)
    result: list[tuple[int, int, np.ndarray]] = []
    for polygon_index, polygon in enumerate(polygons):
        points = np.asarray(polygon.points, dtype=np.float32)
        points[:, 0] *= record.image_width
        points[:, 1] *= record.image_height
        result.append((polygon_index, polygon.class_id, points))
    if not result:
        raise ValueError(f"C4-2C crop source has no polygons: {record.sample_id}")
    return tuple(result)


# ADD 2026-08-31: Fast R17 rasterize-slice-contour contract로 한 small-centered crop을 만든다.
def create_small_center_crop(
    *,
    record: DerivedManifestRecord,
    dataset_root: Path,
    crop_size: int,
    crop_image_path: Path,
    crop_label_path: Path,
    valid_class_ids: set[int],
) -> CropProvenance:
    image_path = dataset_root / record.image_path
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"C4-2C crop source image is unreadable: {image_path}")
    height, width = image_bgr.shape[:2]
    if (width, height) != (record.image_width, record.image_height):
        raise ValueError(f"C4-2C crop source dimensions changed: {record.sample_id}")
    polygons = load_source_polygons(
        record,
        dataset_root=dataset_root,
        valid_class_ids=valid_class_ids,
    )

    # Polygon index is the explicit final tie-break and therefore preserves source label-line order.
    target_index, _, target_points = min(
        polygons,
        key=lambda item: (
            float(abs(cv2.contourArea(item[2].astype(np.float32)))),
            item[1],
            item[0],
        ),
    )
    center_x = float(np.mean(target_points[:, 0]))
    center_y = float(np.mean(target_points[:, 1]))
    x0, y0, x1, y1 = crop_square_bounds(
        center_x=center_x,
        center_y=center_y,
        crop_size=crop_size,
        width=width,
        height=height,
    )
    crop_bgr = image_bgr[y0:y1, x0:x1]
    if crop_bgr.shape[:2] != (crop_size, crop_size):
        raise ValueError(f"C4-2C crop shape is invalid: {record.sample_id}")

    crop_image_path.parent.mkdir(parents=True, exist_ok=True)
    crop_label_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(crop_image_path), crop_bgr):
        raise OSError(f"Failed to write C4-2C crop image: {crop_image_path}")

    label_lines: list[str] = []
    for _, class_id, points in polygons:
        full_mask = np.zeros((height, width), dtype=np.uint8)
        points_i32 = np.rint(points).astype(np.int32)
        points_i32[:, 0] = np.clip(points_i32[:, 0], 0, width - 1)
        points_i32[:, 1] = np.clip(points_i32[:, 1], 0, height - 1)
        cv2.fillPoly(full_mask, [points_i32], color=1)
        crop_mask = full_mask[y0:y1, x0:x1]
        if not np.any(crop_mask):
            continue
        contours, _ = cv2.findContours(
            (crop_mask * 255).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        for contour in contours:
            if contour.shape[0] < 3 or float(cv2.contourArea(contour)) < 1.0:
                continue
            normalized = contour[:, 0, :].astype(np.float32)
            normalized[:, 0] = np.clip(normalized[:, 0] / crop_size, 0.0, 1.0)
            normalized[:, 1] = np.clip(normalized[:, 1] / crop_size, 0.0, 1.0)
            if (
                not np.isfinite(normalized).all()
                or np.any(normalized < 0.0)
                or np.any(normalized > 1.0)
            ):
                raise ValueError("C4-2C crop label coordinate is invalid.")
            coordinates = " ".join(f"{value:.6f}" for point in normalized for value in point)
            label_lines.append(f"{class_id} {coordinates}")
    if not label_lines:
        raise ValueError(f"C4-2C crop removed every GT instance: {record.sample_id}")
    crop_label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
    return CropProvenance(
        source_sample_id=record.sample_id,
        source_relative_image_path=record.image_path,
        generated_image_path=crop_image_path.relative_to(crop_image_path.parents[2]).as_posix(),
        generated_label_path=crop_label_path.relative_to(crop_label_path.parents[2]).as_posix(),
        crop_box_xyxy=(x0, y0, x1, y1),
        source_width=width,
        source_height=height,
        crop_size=crop_size,
        retained_instance_count=len(label_lines),
        target_polygon_index=target_index,
        crop_image_sha256=sha256_file(crop_image_path),
        crop_label_sha256=sha256_file(crop_label_path),
    )


def _portable_canonical_path(image_path: str) -> str:
    path = PurePosixPath(image_path)
    if path.is_absolute() or ".." in path.parts or path.parts[:2] != ("images", "train"):
        raise ValueError("C4-2C canonical train path is not portable.")
    return f"{CANONICAL_PATH_PREFIX}/{path.as_posix()}"


def _fingerprint_payload(
    *,
    experiment_id: str,
    manifest_sha256: str,
    crop_size: int,
    entries: Sequence[CropTrainViewEntry],
    crops: Sequence[CropProvenance],
) -> dict[str, object]:
    return {
        "schema_version": CROP_TRAIN_VIEW_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "manifest_sha256": manifest_sha256,
        "sampling_rule_version": SAMPLING_RULE_VERSION,
        "sampling_mode": CROP_SAMPLING_MODE,
        "multiplicity": ELIGIBLE_MULTIPLICITY,
        "crop_size": crop_size,
        "ordering_policy": CROP_ORDERING_POLICY,
        "entries": [asdict(entry) for entry in entries],
        "crops": [asdict(crop) for crop in crops],
    }


# ADD 2026-08-31: Crop view를 만든다. → MODIFY 2026-09-01: Actual count evidence를 보존한다.
def build_component_aware_crop_train_view(
    *,
    dataset_root: Path,
    train_records: Sequence[DerivedManifestRecord],
    contract: YoloDatasetContract,
    experiment_id: str,
    crop_size: int,
    output_dir: Path,
) -> CropTrainViewArtifact:
    if output_dir.exists():
        raise FileExistsError(f"C4-2C crop train-view output already exists: {output_dir}")
    plan: PlannedTrainView = plan_component_aware_train_view(
        dataset_root=dataset_root,
        train_records=train_records,
        contract=contract,
        experiment_id=experiment_id,
    )
    records_by_id = {record.sample_id: record for record in train_records}
    ranked_small_profiles = sorted(
        (profile for profile in plan.profiles if not profile.is_negative),
        key=lambda profile: (profile.image_min_component_area_ratio, profile.sample_id),
    )[: plan.evidence.small_aware_count]
    small_ids = tuple(profile.sample_id for profile in ranked_small_profiles)
    if set(small_ids) != set(plan.eligibility.small_aware_sample_ids):
        raise RuntimeError("C4-2C small-aware ranking disagrees with shared eligibility.")

    canonical_records = sorted(train_records, key=lambda record: record.sample_id)
    entries = [
        CropTrainViewEntry(
            kind="canonical",
            source_sample_id=record.sample_id,
            is_negative=record.is_negative,
            portable_image_path=_portable_canonical_path(record.image_path),
            source_relative_image_path=record.image_path,
            generated_relative_path=None,
        )
        for record in canonical_records
    ]
    for sample_id in plan.eligibility.eligible_sample_ids:
        record = records_by_id[sample_id]
        entries.append(
            CropTrainViewEntry(
                kind="component_aware_duplicate",
                source_sample_id=sample_id,
                is_negative=record.is_negative,
                portable_image_path=_portable_canonical_path(record.image_path),
                source_relative_image_path=record.image_path,
                generated_relative_path=None,
            )
        )

    crop_provenance: list[CropProvenance] = []
    for sample_id in small_ids:
        record = records_by_id[sample_id]
        stem = f"{sample_id}__small_center_crop{crop_size}"
        image_relative = Path("images/train_crops") / f"{stem}.png"
        label_relative = Path("labels/train_crops") / f"{stem}.txt"
        crop = create_small_center_crop(
            record=record,
            dataset_root=dataset_root,
            crop_size=crop_size,
            crop_image_path=output_dir / image_relative,
            crop_label_path=output_dir / label_relative,
            valid_class_ids=set(contract.classes),
        )
        crop_provenance.append(crop)
        entries.append(
            CropTrainViewEntry(
                kind="small_center_crop",
                source_sample_id=sample_id,
                is_negative=record.is_negative,
                portable_image_path=f"{GENERATED_PATH_PREFIX}/{image_relative.as_posix()}",
                source_relative_image_path=record.image_path,
                generated_relative_path=image_relative.as_posix(),
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    portable_list = output_dir / "train.txt"
    portable_bytes = ("\n".join(entry.portable_image_path for entry in entries) + "\n").encode()
    portable_list.write_bytes(portable_bytes)
    fingerprint_payload = _fingerprint_payload(
        experiment_id=experiment_id,
        manifest_sha256=contract.manifest_sha256,
        crop_size=crop_size,
        entries=entries,
        crops=crop_provenance,
    )
    fingerprint = sha256_bytes(
        json.dumps(
            fingerprint_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    )
    canonical_entries = tuple(entry for entry in entries if entry.kind == "canonical")
    duplicate_entries = tuple(
        entry for entry in entries if entry.kind == "component_aware_duplicate"
    )
    crop_entries = tuple(entry for entry in entries if entry.kind == "small_center_crop")
    small_set = set(small_ids)
    multi_ids = plan.eligibility.multi_component_sample_ids
    multi_set = set(multi_ids)
    eligible_ids = plan.eligibility.eligible_sample_ids
    evidence = CropTrainViewEvidence(
        schema_version=CROP_TRAIN_VIEW_SCHEMA_VERSION,
        experiment_id=experiment_id,
        sampling_mode=CROP_SAMPLING_MODE,
        sampling_rule_version=SAMPLING_RULE_VERSION,
        canonical_manifest_sha256=contract.manifest_sha256,
        crop_size=crop_size,
        canonical_entry_count=len(canonical_entries),
        canonical_positive_count=sum(not entry.is_negative for entry in canonical_entries),
        canonical_negative_count=sum(entry.is_negative for entry in canonical_entries),
        component_duplicate_count=len(duplicate_entries),
        crop_entry_count=len(crop_entries),
        total_entry_count=len(entries),
        positive_exposure=sum(not entry.is_negative for entry in entries),
        negative_exposure=sum(entry.is_negative for entry in entries),
        small_aware_count=len(small_set),
        multi_component_count=len(multi_set),
        eligible_overlap_count=len(small_set.intersection(multi_set)),
        eligible_union_count=len(small_set.union(multi_set)),
        small_aware_sample_ids=small_ids,
        multi_component_sample_ids=multi_ids,
        component_aware_sample_ids=eligible_ids,
        observed_train_small_cutoff=plan.eligibility.observed_train_small_cutoff,
        ordering_policy=CROP_ORDERING_POLICY,
        entries=tuple(entries),
        crops=tuple(crop_provenance),
        portable_train_list_sha256=sha256_bytes(portable_bytes),
        train_view_fingerprint_sha256=fingerprint,
        validation_used_for_sampling=False,
        test_used=False,
    )
    evidence.validate()
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(evidence.to_json_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return CropTrainViewArtifact(output_dir, portable_list, metadata_path, evidence)


# ADD 2026-08-31: Portable crop identities를 machine-local absolute train list로 변환한다.
def build_runtime_crop_train_view_adapter(
    *,
    dataset_root: Path,
    crop_train_view: CropTrainViewArtifact,
    destination: Path,
) -> RuntimeCropTrainViewArtifact:
    crop_train_view.evidence.validate()
    if sha256_file(crop_train_view.train_list_path) != (
        crop_train_view.evidence.portable_train_list_sha256
    ):
        raise ValueError("C4-2C portable train list SHA changed.")
    if destination.exists():
        raise FileExistsError(f"C4-2C runtime train list already exists: {destination}")
    resolved_dataset = dataset_root.resolve()
    resolved_generated = crop_train_view.output_dir.resolve()
    runtime_entries: list[str] = []
    for value in crop_train_view.train_list_path.read_text(encoding="utf-8").splitlines():
        prefix, separator, relative_value = value.partition("/")
        if not separator:
            raise ValueError("C4-2C portable train entry is malformed.")
        relative = PurePosixPath(relative_value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("C4-2C portable train entry escapes its root.")
        base = resolved_dataset if prefix == CANONICAL_PATH_PREFIX else resolved_generated
        if prefix not in {CANONICAL_PATH_PREFIX, GENERATED_PATH_PREFIX}:
            raise ValueError("C4-2C portable train entry has an unknown root.")
        resolved = base.joinpath(*relative.parts).resolve()
        try:
            resolved.relative_to(base)
        except ValueError as exc:
            raise ValueError("C4-2C runtime train entry escapes its root.") from exc
        if not resolved.is_file():
            raise FileNotFoundError(f"C4-2C runtime train image is missing: {resolved}")
        runtime_entries.append(resolved.as_posix())
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(runtime_entries) + "\n", encoding="utf-8")
    return RuntimeCropTrainViewArtifact(
        train_list_path=destination,
        source_train_list_sha256=crop_train_view.evidence.portable_train_list_sha256,
        entry_count=len(runtime_entries),
    )
