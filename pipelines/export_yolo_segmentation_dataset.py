"""Export and validate the MVTec AD-derived YOLO segmentation feasibility dataset."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image

from ml.datasets.defect_annotations import (
    extract_mask_components,
    load_binary_mask,
    resolve_dataset_path,
)
from ml.datasets.manifest import ManifestRecord, read_manifest_csv
from ml.datasets.manifest_validation import validate_manifest_records
from ml.datasets.segmentation_annotations import (
    DERIVED_SPLITS,
    PolygonConversion,
    SegmentationPolygon,
    mask_to_yolo_polygons,
    parse_yolo_segmentation_label,
    rasterize_polygons,
    sample_negative_ids,
    stratified_split_sample_ids,
    summarize_fidelity,
)
from ml.datasets.yolo_segmentation_manifest import (
    DerivedManifestRecord,
    read_derived_manifest,
    write_derived_manifest,
)
from shared.hashing import sha256_bytes, sha256_file

DEFAULT_CONFIG_PATH = Path("configs/data/mvtec_ad_metal_nut_yolo_segmentation.yaml")
MANIFEST_NAME = "manifest.csv"
METADATA_NAME = "metadata.json"
DATASET_YAML_NAME = "dataset.yaml"


@dataclass(frozen=True)
class ExportArtifacts:
    """Completed dataset, validation, visualization, and package paths."""

    dataset_root: Path
    manifest_path: Path
    metadata_path: Path
    dataset_yaml_path: Path
    visualization_paths: tuple[Path, ...]
    package_path: Path
    manifest_sha256: str
    package_sha256: str
    sample_count: int


# ADD 2026-08-25: C2-1 config를 portable repository-relative path와 strict policy로 로드한다.
def load_export_config(config_path: Path) -> dict[str, Any]:
    """Load the explicit class, split, fidelity, and output source of truth."""
    if not config_path.is_file():
        raise FileNotFoundError(f"C2-1 config not found: {config_path}")
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("C2-1 config root must be a mapping.")
    class_mapping = loaded.get("classes")
    if class_mapping != {0: "bent", 1: "color", 2: "scratch"}:
        raise ValueError("C2-1 class mapping must be exactly 0=bent, 1=color, 2=scratch.")
    if "flip" in class_mapping.values():
        raise ValueError("Flip must never enter the segmentation taxonomy.")
    return loaded


# ADD 2026-08-25: Positive anomaly와 good negative source pool을 benchmark와 분리해 선택한다.
def select_source_records(
    records: list[ManifestRecord],
    *,
    category: str,
    class_mapping: dict[int, str],
) -> tuple[list[ManifestRecord], list[ManifestRecord]]:
    """Select local-defect positives and real good images without treating flip as negative."""
    positive_classes = set(class_mapping.values())
    positives = sorted(
        (
            record
            for record in records
            if record.category == category
            and record.label == 1
            and record.defect_type in positive_classes
            and record.source_split == "test"
            and record.split == "test"
        ),
        key=lambda record: (record.defect_type, record.sample_id),
    )
    negatives = sorted(
        (
            record
            for record in records
            if record.category == category and record.label == 0 and record.defect_type == "good"
        ),
        key=lambda record: record.sample_id,
    )
    if {record.defect_type for record in positives} != positive_classes:
        raise ValueError("Source manifest does not contain every configured positive class.")
    if not negatives:
        raise ValueError("Source manifest contains no good images for negative sampling.")
    if any(record.defect_type == "flip" for record in positives + negatives):
        raise ValueError("Flip source image entered the segmentation selection.")
    return positives, negatives


# ADD 2026-08-25: Positive class split과 split-matched negative sample을 한 번에 결정한다.
def build_split_assignments(
    positives: list[ManifestRecord],
    negatives: list[ManifestRecord],
    *,
    seed: int,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    negative_ratio: float,
) -> dict[str, str]:
    """Return one derived split for every selected source sample."""
    positives_by_class: dict[str, list[str]] = defaultdict(list)
    for record in positives:
        positives_by_class[record.defect_type].append(record.sample_id)
    positive_assignments = stratified_split_sample_ids(
        dict(positives_by_class),
        seed=seed,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
    )
    positive_split_counts = Counter(item.derived_split for item in positive_assignments)
    negative_split_counts = {
        split_name: round(positive_split_counts[split_name] * negative_ratio)
        for split_name in DERIVED_SPLITS
    }
    negative_assignments = sample_negative_ids(
        [record.sample_id for record in negatives],
        split_counts=negative_split_counts,
        seed=seed,
    )
    assignments = {
        item.sample_id: item.derived_split
        for item in (*positive_assignments, *negative_assignments)
    }
    if len(assignments) != len(positive_assignments) + len(negative_assignments):
        raise ValueError("Positive and negative source samples overlap.")
    return assignments


# ADD 2026-08-25: Ultralytics-compatible relative dataset YAML을 dependency 없이 생성한다.
def write_dataset_yaml(path: Path, *, class_mapping: dict[int, str]) -> None:
    """Write portable train/val/test paths without a workstation absolute path."""
    content = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": class_mapping,
    }
    path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")


# ADD 2026-08-25: Binary source를 polygon label과 byte-identical copied image로 export한다.
def export_sample(
    *,
    record: ManifestRecord,
    derived_split: str,
    dataset_root: Path,
    output_root: Path,
    dataset_metadata: dict[str, str],
    source_manifest_sha256: str,
    class_mapping: dict[int, str],
) -> tuple[DerivedManifestRecord, PolygonConversion | None]:
    """Export one positive or negative while retaining source hashes and dimensions."""
    source_image = resolve_dataset_path(dataset_root, record.image_path)
    image_relative = Path("images") / derived_split / f"{record.sample_id}{source_image.suffix}"
    label_relative = Path("labels") / derived_split / f"{record.sample_id}.txt"
    destination_image = output_root / image_relative
    destination_label = output_root / label_relative
    destination_image.parent.mkdir(parents=True, exist_ok=True)
    destination_label.parent.mkdir(parents=True, exist_ok=True)

    # 원본 image bytes를 변환하지 않고 self-contained package에 복사한다.
    shutil.copyfile(source_image, destination_image)
    image_sha256 = sha256_file(source_image)
    if sha256_file(destination_image) != image_sha256:
        raise ValueError(f"Copied image hash mismatch: {record.sample_id}")

    target_class = ""
    target_class_id = ""
    mask_sha256 = ""
    conversion: PolygonConversion | None = None
    component_count = 0
    if record.label == 0:
        destination_label.write_bytes(b"")
    else:
        class_by_name = {class_name: class_id for class_id, class_name in class_mapping.items()}
        if record.defect_type not in class_by_name or not record.mask_path:
            raise ValueError(f"Unsupported positive source record: {record.sample_id}")
        source_mask = resolve_dataset_path(dataset_root, record.mask_path)
        mask = load_binary_mask(source_mask, expected_size=(record.width, record.height))

        # Lossless binary mask에서 disconnected contour를 개별 YOLO polygon으로 변환한다.
        class_id = class_by_name[record.defect_type]
        conversion = mask_to_yolo_polygons(mask, class_id=class_id)
        components = extract_mask_components(mask)
        component_count = len(components)
        if len(conversion.polygons) != component_count:
            raise ValueError(f"Contour/component count mismatch: {record.sample_id}")
        destination_label.write_text(
            "\n".join(polygon.to_yolo_line() for polygon in conversion.polygons) + "\n",
            encoding="utf-8",
        )
        target_class = record.defect_type
        target_class_id = str(class_id)
        mask_sha256 = sha256_file(source_mask)

    return (
        DerivedManifestRecord(
            dataset_name=dataset_metadata["name"],
            dataset_version=dataset_metadata["version"],
            derived_task=dataset_metadata["task"],
            source_manifest_sha256=source_manifest_sha256,
            source_split=record.source_split,
            source_manifest_split=record.split,
            source_image_path=record.image_path,
            source_mask_path=record.mask_path,
            category=record.category,
            sample_id=record.sample_id,
            defect_type=record.defect_type,
            target_class=target_class,
            target_class_id=target_class_id,
            derived_split=derived_split,
            is_negative=record.label == 0,
            image_width=record.width,
            image_height=record.height,
            image_path=image_relative.as_posix(),
            label_path=label_relative.as_posix(),
            image_sha256=image_sha256,
            mask_sha256=mask_sha256,
            polygon_count=0 if conversion is None else len(conversion.polygons),
            component_count=component_count,
            hole_count=0 if conversion is None else conversion.hole_count,
            polygon_vertex_count=0 if conversion is None else conversion.vertex_count,
            round_trip_iou="" if conversion is None else f"{conversion.iou:.12f}",
            pixel_precision="" if conversion is None else f"{conversion.precision:.12f}",
            pixel_recall="" if conversion is None else f"{conversion.recall:.12f}",
        ),
        conversion,
    )


# ADD 2026-08-25: Source hash, split uniqueness, label, fidelity와 YAML 계약을 end-to-end 검증한다.
def validate_exported_dataset(
    *,
    output_root: Path,
    source_root: Path,
    source_manifest_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Validate one completed package without importing a training framework."""
    manifest_path = output_root / MANIFEST_NAME
    metadata_path = output_root / METADATA_NAME
    dataset_yaml_path = output_root / DATASET_YAML_NAME
    for required_path in (manifest_path, metadata_path, dataset_yaml_path):
        if not required_path.is_file():
            raise FileNotFoundError(f"Required derived dataset artifact missing: {required_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    records = read_derived_manifest(manifest_path)
    source_manifest_sha256 = sha256_file(source_manifest_path)
    if metadata["source_manifest_sha256"] != source_manifest_sha256:
        raise ValueError("Metadata source manifest hash is stale.")
    if metadata["derived_manifest_sha256"] != sha256_file(manifest_path):
        raise ValueError("Derived manifest hash does not match metadata.")

    dataset_yaml = yaml.safe_load(dataset_yaml_path.read_text(encoding="utf-8"))
    expected_yaml = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": config["classes"],
    }
    if dataset_yaml != expected_yaml:
        raise ValueError("Portable dataset.yaml contract mismatch.")

    source_records = {
        record.image_path: record for record in read_manifest_csv(source_manifest_path)
    }
    sample_ids: set[str] = set()
    source_images: set[str] = set()
    observed_counts: Counter[tuple[str, str]] = Counter()
    conversions_by_defect: dict[str, list[PolygonConversion]] = defaultdict(list)
    valid_class_ids = set(config["classes"])
    gate = config["fidelity_gate"]
    for record in records:
        if record.sample_id in sample_ids or record.source_image_path in source_images:
            raise ValueError("Derived split leakage or duplicate source image detected.")
        sample_ids.add(record.sample_id)
        source_images.add(record.source_image_path)
        if record.derived_split not in DERIVED_SPLITS:
            raise ValueError(f"Invalid derived split: {record.derived_split}")
        if record.defect_type == "flip":
            raise ValueError("Flip image must not be present in segmentation dataset.")

        image_path = resolve_dataset_path(output_root, record.image_path)
        label_path = resolve_dataset_path(output_root, record.label_path)
        source_image = resolve_dataset_path(source_root, record.source_image_path)
        if not image_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"Derived image/label missing: {record.sample_id}")
        with Image.open(image_path) as image:
            if image.size != (record.image_width, record.image_height):
                raise ValueError(f"Derived image dimensions changed: {record.sample_id}")
            image.verify()
        if sha256_file(image_path) != record.image_sha256:
            raise ValueError(f"Derived image hash mismatch: {record.sample_id}")
        if sha256_file(source_image) != record.image_sha256:
            raise ValueError(f"Source image hash mismatch: {record.sample_id}")
        source_record = source_records.get(record.source_image_path)
        if source_record is None or source_record.sample_id != record.sample_id:
            raise ValueError(f"Source manifest lineage mismatch: {record.sample_id}")
        if (
            record.source_manifest_sha256 != source_manifest_sha256
            or record.source_split != source_record.source_split
            or record.source_manifest_split != source_record.split
            or record.source_mask_path != source_record.mask_path
            or record.category != source_record.category
            or record.defect_type != source_record.defect_type
            or (record.image_width, record.image_height)
            != (source_record.width, source_record.height)
        ):
            raise ValueError(f"Source manifest field lineage mismatch: {record.sample_id}")

        label_text = label_path.read_text(encoding="utf-8")
        if record.is_negative:
            if label_text or record.defect_type != "good" or record.polygon_count != 0:
                raise ValueError(f"Invalid empty negative representation: {record.sample_id}")
            observed_counts[(record.derived_split, "negative")] += 1
            continue

        if record.target_class not in config["classes"].values():
            raise ValueError(f"Invalid positive class name: {record.sample_id}")
        polygons = parse_yolo_segmentation_label(
            label_text,
            valid_class_ids=valid_class_ids,
        )
        if len(polygons) != record.polygon_count:
            raise ValueError(f"Positive polygon topology mismatch: {record.sample_id}")
        mask_path = resolve_dataset_path(source_root, record.source_mask_path)
        if sha256_file(mask_path) != record.mask_sha256:
            raise ValueError(f"Source mask hash mismatch: {record.sample_id}")
        source_mask = load_binary_mask(
            mask_path,
            expected_size=(record.image_width, record.image_height),
        )
        source_conversion = mask_to_yolo_polygons(
            source_mask,
            class_id=int(record.target_class_id),
        )
        if (
            source_conversion.hole_count != record.hole_count
            or len(source_conversion.polygons) != record.polygon_count
        ):
            raise ValueError(f"Source contour lineage mismatch: {record.sample_id}")
        reconstructed = rasterize_polygons(
            polygons,
            image_width=record.image_width,
            image_height=record.image_height,
        )
        intersection = int(np.logical_and(source_mask, reconstructed).sum())
        union = int(np.logical_or(source_mask, reconstructed).sum())
        conversion = PolygonConversion(
            polygons=polygons,
            hole_count=source_conversion.hole_count,
            vertex_count=sum(len(polygon.points) for polygon in polygons),
            source_positive_pixels=int(source_mask.sum()),
            reconstructed_positive_pixels=int(reconstructed.sum()),
            intersection_pixels=intersection,
            iou=intersection / union,
            precision=intersection / int(reconstructed.sum()),
            recall=intersection / int(source_mask.sum()),
        )
        if (
            conversion.iou < gate["minimum_sample_iou"]
            or conversion.precision < gate["minimum_sample_precision"]
            or conversion.recall < gate["minimum_sample_recall"]
        ):
            raise ValueError(f"Round-trip fidelity gate failed: {record.sample_id}")
        conversions_by_defect[record.defect_type].append(conversion)
        observed_counts[(record.derived_split, record.defect_type)] += 1

    summaries = {
        defect_type: summarize_fidelity(conversions)
        for defect_type, conversions in sorted(conversions_by_defect.items())
    }
    for defect_type, summary in summaries.items():
        if summary["iou"]["p05"] < gate["minimum_defect_p05_iou"]:
            raise ValueError(f"Defect p05 fidelity gate failed: {defect_type}")
    if metadata["sample_counts"] != _stringify_count_keys(observed_counts):
        raise ValueError("Metadata sample counts do not match exported records.")
    return {
        "record_count": len(records),
        "sample_counts": _stringify_count_keys(observed_counts),
        "fidelity": summaries,
    }


# ADD 2026-08-25: Tuple count key를 metadata/CLI용 stable string key로 변환한다.
def _stringify_count_keys(counts: Counter[tuple[str, str]]) -> dict[str, int]:
    return {
        f"{split_name}:{kind}": counts[(split_name, kind)] for split_name, kind in sorted(counts)
    }


# ADD 2026-08-25: GT와 YOLO round-trip을 class/negative montage로 비교한다.
def render_validation_visualizations(
    *,
    records: list[DerivedManifestRecord],
    output_root: Path,
    source_root: Path,
    visualization_root: Path,
    class_mapping: dict[int, str],
) -> tuple[Path, ...]:
    """Render deterministic examples from generated labels for visual QA."""
    if visualization_root.exists():
        raise FileExistsError(f"Visualization output already exists: {visualization_root}")
    visualization_root.mkdir(parents=True)
    groups: dict[str, list[DerivedManifestRecord]] = defaultdict(list)
    for record in records:
        groups["good" if record.is_negative else record.defect_type].append(record)
    paths: list[Path] = []
    valid_class_ids = set(class_mapping)
    for defect_type in (*class_mapping.values(), "good"):
        selected = sorted(groups[defect_type], key=lambda record: record.sample_id)[:4]
        figure, axes = plt.subplots(len(selected), 4, figsize=(14, 3.2 * len(selected)))
        axes_array = np.atleast_2d(axes)
        for row_index, record in enumerate(selected):
            image_path = resolve_dataset_path(output_root, record.image_path)
            with Image.open(image_path) as image:
                image_array = np.asarray(image.convert("RGB"))
            if record.is_negative:
                ground_truth = np.zeros((record.image_height, record.image_width), dtype=np.bool_)
                polygons: tuple[SegmentationPolygon, ...] = ()
            else:
                mask_path = resolve_dataset_path(source_root, record.source_mask_path)
                ground_truth = load_binary_mask(
                    mask_path,
                    expected_size=(record.image_width, record.image_height),
                )
                label_path = resolve_dataset_path(output_root, record.label_path)
                polygons = parse_yolo_segmentation_label(
                    label_path.read_text(encoding="utf-8"),
                    valid_class_ids=valid_class_ids,
                )
            reconstructed = rasterize_polygons(
                polygons,
                image_width=record.image_width,
                image_height=record.image_height,
            )
            original_axis, gt_axis, polygon_axis, reconstruction_axis = axes_array[row_index]
            original_axis.imshow(image_array)
            gt_axis.imshow(ground_truth, cmap="gray", vmin=0, vmax=1)
            polygon_axis.imshow(image_array)
            polygon_axis.imshow(
                np.ma.masked_where(~reconstructed, reconstructed), cmap="autumn", alpha=0.55
            )
            reconstruction_axis.imshow(reconstructed, cmap="gray", vmin=0, vmax=1)
            original_axis.set_ylabel(record.sample_id, fontsize=8)
            for axis in axes_array[row_index]:
                axis.set_xticks([])
                axis.set_yticks([])
        axes_array[0, 0].set_title("Original")
        axes_array[0, 1].set_title("GT binary mask")
        axes_array[0, 2].set_title("YOLO polygon overlay")
        axes_array[0, 3].set_title("Round-trip mask")
        figure.suptitle(f"C2-1 / {defect_type}", y=0.995)
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
        visualization_path = visualization_root / f"{defect_type}_polygon_validation.png"
        figure.savefig(visualization_path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        paths.append(visualization_path)
    return tuple(paths)


# ADD 2026-08-25: Dataset tree를 stable entry order/timestamp의 Kaggle ZIP으로 묶는다.
def create_deterministic_zip(dataset_root: Path, package_path: Path) -> str:
    """Package only generated images, labels, YAML, manifest, and metadata."""
    if package_path.exists():
        raise FileExistsError(f"Dataset package already exists: {package_path}")
    package_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(item for item in dataset_root.rglob("*") if item.is_file()):
            info = zipfile.ZipInfo(path.relative_to(dataset_root).as_posix())
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    return sha256_file(package_path)


# ADD 2026-08-25: C2-1 split, export, validation, visualization와 packaging을 atomic하게 조율한다.
def export_yolo_segmentation_dataset(
    *,
    config: dict[str, Any],
    created_at_utc: str,
) -> ExportArtifacts:
    """Build the complete supervised-derived package without training a model."""
    dataset = config["dataset"]
    output = config["output"]
    source_root = Path(dataset["source_root"])
    source_manifest_path = Path(dataset["source_manifest"])
    output_root = Path(output["dataset_root"])
    visualization_root = Path(output["visualization_root"])
    package_path = Path(output["package_path"])
    if output_root.exists() or visualization_root.exists() or package_path.exists():
        raise FileExistsError(
            "C2-1 output already exists; use a new version or remove it explicitly."
        )

    # Existing manifest integrity를 확인한 뒤 supervised 대상과 negative pool을 분리한다.
    source_records = read_manifest_csv(source_manifest_path)
    positives, negative_pool = select_source_records(
        source_records,
        category=dataset["category"],
        class_mapping=config["classes"],
    )
    source_validation = validate_manifest_records(positives + negative_pool, source_root)
    if not source_validation.is_valid:
        raise ValueError("C2-1 source validation failed:\n" + "\n".join(source_validation.errors))
    split = config["split"]
    assignments = build_split_assignments(
        positives,
        negative_pool,
        seed=split["seed"],
        train_ratio=split["train_ratio"],
        validation_ratio=split["validation_ratio"],
        test_ratio=split["test_ratio"],
        negative_ratio=config["negatives"]["ratio_to_positive"],
    )
    selected_records = [
        record for record in positives + negative_pool if record.sample_id in assignments
    ]
    source_manifest_sha256 = sha256_file(source_manifest_path)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=f".{output_root.name}-", dir=output_root.parent))
    conversions_by_defect: dict[str, list[PolygonConversion]] = defaultdict(list)
    try:
        derived_records: list[DerivedManifestRecord] = []
        # 각 image를 정확히 한 split에 copy하고 positive mask만 polygon label로 변환한다.
        for record in selected_records:
            derived_record, conversion = export_sample(
                record=record,
                derived_split=assignments[record.sample_id],
                dataset_root=source_root,
                output_root=temporary_root,
                dataset_metadata={
                    "name": dataset["name"],
                    "version": dataset["version"],
                    "task": dataset["task"],
                },
                source_manifest_sha256=source_manifest_sha256,
                class_mapping=config["classes"],
            )
            derived_records.append(derived_record)
            if conversion is not None:
                conversions_by_defect[record.defect_type].append(conversion)
        split_order = {split_name: index for index, split_name in enumerate(DERIVED_SPLITS)}
        derived_records.sort(
            key=lambda record: (
                split_order[record.derived_split],
                record.is_negative,
                record.defect_type,
                record.sample_id,
            )
        )
        write_dataset_yaml(temporary_root / DATASET_YAML_NAME, class_mapping=config["classes"])
        manifest_path = temporary_root / MANIFEST_NAME
        write_derived_manifest(derived_records, manifest_path)
        manifest_sha256 = sha256_file(manifest_path)
        fidelity = {
            defect_type: summarize_fidelity(conversions)
            for defect_type, conversions in sorted(conversions_by_defect.items())
        }
        counts = Counter(
            (
                record.derived_split,
                "negative" if record.is_negative else record.defect_type,
            )
            for record in derived_records
        )
        semantic_contract = {
            "dataset_name": dataset["name"],
            "dataset_version": dataset["version"],
            "source_dataset": dataset["source_dataset"],
            "source_dataset_version": dataset["source_dataset_version"],
            "category": dataset["category"],
            "task": dataset["task"],
            "seed": split["seed"],
            "split_policy": split,
            "negative_sampling_policy": config["negatives"],
            "class_mapping": config["classes"],
            "polygon_policy": config["polygon"],
            "fidelity_gate": config["fidelity_gate"],
            "source_manifest_sha256": source_manifest_sha256,
            "derived_manifest_sha256": manifest_sha256,
            "sample_counts": _stringify_count_keys(counts),
        }
        metadata = {
            **semantic_contract,
            "created_at_utc": created_at_utc,
            "tool_version": "c2_1_v1",
            "semantic_fingerprint_sha256": sha256_bytes(
                json.dumps(
                    semantic_contract,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ),
            "fidelity": fidelity,
        }
        (temporary_root / METADATA_NAME).write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_root.rename(output_root)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise

    # 완성된 package를 source hashes와 fidelity gate로 다시 읽어 검증한다.
    validate_exported_dataset(
        output_root=output_root,
        source_root=source_root,
        source_manifest_path=source_manifest_path,
        config=config,
    )
    final_records = read_derived_manifest(output_root / MANIFEST_NAME)
    visualization_paths = render_validation_visualizations(
        records=final_records,
        output_root=output_root,
        source_root=source_root,
        visualization_root=visualization_root,
        class_mapping=config["classes"],
    )
    package_sha256 = create_deterministic_zip(output_root, package_path)
    return ExportArtifacts(
        dataset_root=output_root,
        manifest_path=output_root / MANIFEST_NAME,
        metadata_path=output_root / METADATA_NAME,
        dataset_yaml_path=output_root / DATASET_YAML_NAME,
        visualization_paths=visualization_paths,
        package_path=package_path,
        manifest_sha256=sha256_file(output_root / MANIFEST_NAME),
        package_sha256=package_sha256,
        sample_count=len(final_records),
    )


# ADD 2026-08-25: Export 또는 existing dataset validation CLI argument를 정의한다.
def parse_args() -> argparse.Namespace:
    """Parse the config and validation-only execution mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


# ADD 2026-08-25: CLI에서 export 후 validation summary 또는 validation-only 결과를 출력한다.
def main() -> None:
    """Run the configured C2-1 dataset lifecycle."""
    args = parse_args()
    config = load_export_config(args.config)
    if args.validate_only:
        report = validate_exported_dataset(
            output_root=Path(config["output"]["dataset_root"]),
            source_root=Path(config["dataset"]["source_root"]),
            source_manifest_path=Path(config["dataset"]["source_manifest"]),
            config=config,
        )
        print(json.dumps(report, indent=2, default=dict, sort_keys=True))
        return

    artifacts = export_yolo_segmentation_dataset(
        config=config,
        created_at_utc=datetime.now(UTC).isoformat(),
    )
    print(json.dumps(asdict(artifacts), indent=2, default=str, sort_keys=True))


if __name__ == "__main__":
    main()
