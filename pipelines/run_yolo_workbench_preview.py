"""Generate actual YOLO Workbench previews inside the locked project environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

from PIL import Image

from ml.datasets.segmentation_annotations import rasterize_segmentation_label_instances
from ml.experiments.yolo_augmentation import (
    prepare_preview_dataset,
    preview_actual_representations,
    preview_actual_training_augmentations,
)
from ml.experiments.yolo_segmentation import load_yolo_experiment_config
from ml.experiments.yolo_workbench import build_research_config, validate_workbench_controls
from ml.experiments.yolo_workbench_runtime import collect_current_environment
from ml.experiments.yolo_workbench_visualization import (
    render_augmentation_gallery,
    render_representation_comparison,
)
from ml.training.yolo_segmentation import (
    load_yolo_segmentation_config,
    validate_experiment_dataset,
)


# ADD 2026-08-28: Locked preview artifact를 notebook이 읽을 strict JSON으로 저장한다.
def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


# ADD 2026-08-28: Actual augmentation/representation을 locked interpreter에서 생성한다.
def run_yolo_workbench_preview(
    *,
    mode: Literal["research", "official"],
    experiment_config_path: Path,
    dataset_root: Path,
    output_root: Path,
    repository_root: Path,
    train_sample_ids: list[str],
    representation_sample_id: str | None,
    research_overrides: dict[str, object],
) -> Path:
    if mode not in {"research", "official"}:
        raise ValueError("Workbench preview mode must be research or official.")
    validate_workbench_controls(mode, overrides=research_overrides)
    environment = collect_current_environment(repository_root)
    environment.validate(repository_root, require_cuda=False)

    experiment = load_yolo_experiment_config(experiment_config_path)
    baseline = load_yolo_segmentation_config(experiment.baseline_config_path)
    active_config = (
        experiment.training_config(baseline)
        if mode == "official"
        else build_research_config(
            baseline,
            overrides=research_overrides,
            output_root=output_root / "research",
        )
    )
    records = list(validate_experiment_dataset(dataset_root, baseline.dataset_contract))
    record_by_id = {record.sample_id: record for record in records}
    if not train_sample_ids or any(
        sample_id not in record_by_id
        or record_by_id[sample_id].derived_split != "train"
        or record_by_id[sample_id].is_negative
        for sample_id in train_sample_ids
    ):
        raise ValueError("Augmentation preview IDs must be positive train samples.")

    # Train mirror와 actual transform output을 locked environment 안에서 함께 생성한다.
    preview_dataset_root = prepare_preview_dataset(
        dataset_root=dataset_root,
        preview_root=output_root / "preview_dataset",
        records=records,
        split="train",
    )
    augmentation_previews = preview_actual_training_augmentations(
        config=active_config,
        preview_root=preview_dataset_root,
        sample_ids=train_sample_ids,
        variants=3,
    )
    albumentations = environment.packages["albumentations"]
    albumentations_active = any(preview.albumentations_active for preview in augmentation_previews)
    if albumentations_active and not albumentations.available:
        raise RuntimeError("Albumentations is active without locked package provenance.")
    originals: dict[str, Image.Image] = {}
    for sample_id in train_sample_ids:
        with Image.open(dataset_root / record_by_id[sample_id].image_path) as source:
            originals[sample_id] = source.convert("RGB")
    augmentation_figure = render_augmentation_gallery(
        originals=originals,
        previews=augmentation_previews,
        output_path=output_root / "visualizations/augmentation/augmentation_preview.png",
    )

    representation_payload: dict[str, object] | None = None
    if representation_sample_id is not None:
        record = record_by_id.get(representation_sample_id)
        if record is None or record.derived_split != "val" or record.is_negative:
            raise ValueError("Representation preview ID must be a positive validation sample.")
        if experiment.intervention_type != "resolution":
            raise ValueError("Representation comparison is valid only for resolution experiments.")
        preview_dataset_root = prepare_preview_dataset(
            dataset_root=dataset_root,
            preview_root=output_root / "preview_dataset",
            records=records,
            split="val",
        )
        candidate = experiment.training_config(baseline)
        representation_previews = preview_actual_representations(
            configs=(baseline, candidate),
            preview_root=preview_dataset_root,
            sample_id=representation_sample_id,
        )
        image_width = record.image_width
        image_height = record.image_height
        label_text = (dataset_root / record.label_path).read_text(encoding="utf-8")
        instances = rasterize_segmentation_label_instances(
            label_text,
            image_width=image_width,
            image_height=image_height,
            valid_class_ids=set(baseline.dataset_contract.classes),
        )
        with Image.open(dataset_root / record.image_path) as source:
            original = source.convert("RGB")
        representation_figure = render_representation_comparison(
            original=original,
            original_mask_pixels=tuple(int(instance.mask.sum()) for instance in instances),
            previews=representation_previews,
            output_path=(output_root / "visualizations/representation/imgsz_640_vs_1024.png"),
        )
        representation_payload = {
            "sample_id": representation_sample_id,
            "source_split": "val",
            "generated_path": str(representation_figure),
            "imgsz": [preview.imgsz for preview in representation_previews],
            "transform_names": [
                list(preview.transform_names) for preview in representation_previews
            ],
        }

    payload = {
        "schema_version": 1,
        "mode": mode,
        "execution_environment": environment.to_json_dict(),
        "test_split_used": False,
        "augmentation": {
            "sample_ids": train_sample_ids,
            "source_split": "train",
            "variants": 3,
            "generated_path": str(augmentation_figure),
            "transform_names": list(augmentation_previews[0].transform_names),
            "albumentations": {
                "available_in_locked_environment": albumentations.available,
                "version": albumentations.version,
                "path": albumentations.path,
                "active_in_actual_transform": albumentations_active,
            },
        },
        "representation": representation_payload,
    }
    return _write_json(output_root / "preview_metadata.json", payload)


# ADD 2026-08-28: Preview mode/paths/sample IDs와 research-only overrides를 정의한다.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("research", "official"), required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--train-sample-id", action="append", default=[])
    parser.add_argument("--representation-sample-id")
    parser.add_argument("--research-imgsz", type=int)
    parser.add_argument("--research-batch", type=int)
    parser.add_argument("--research-epochs", type=int)
    parser.add_argument("--research-patience", type=int)
    return parser.parse_args()


# ADD 2026-08-28: CLI arguments를 official-override-safe preview lifecycle로 전달한다.
def main() -> int:
    args = parse_args()
    research_overrides = {
        field: value
        for field in ("imgsz", "batch", "epochs", "patience")
        if (value := getattr(args, f"research_{field}")) is not None
    }
    metadata_path = run_yolo_workbench_preview(
        mode=args.mode,
        experiment_config_path=args.experiment_config,
        dataset_root=args.dataset,
        output_root=args.output_root,
        repository_root=args.repository_root,
        train_sample_ids=args.train_sample_id,
        representation_sample_id=args.representation_sample_id,
        research_overrides=research_overrides,
    )
    print("YOLO Workbench locked preview: PASS")
    print(f"Preview metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
