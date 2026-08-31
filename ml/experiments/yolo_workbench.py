"""Validation-safe dataset EDA and execution controls for the YOLO workbench."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

from ml.datasets.segmentation_annotations import (
    deterministic_rank,
    rasterize_segmentation_label_instances,
)
from ml.datasets.yolo_segmentation_manifest import (
    DerivedManifestRecord,
    read_derived_manifest,
)
from ml.evaluation.final_benchmark import RepositoryProvenance, resolve_repository_provenance
from ml.evaluation.yolo_segmentation_error_analysis import SizeBucketPolicy
from ml.experiments.yolo_sampling import plan_component_aware_train_view
from ml.experiments.yolo_segmentation import YoloExperimentConfig
from ml.experiments.yolo_workbench_runtime import LockedEnvironmentProvenance
from ml.training.yolo_segmentation import (
    YoloOutputConfig,
    YoloSegmentationBaselineConfig,
    validate_experiment_dataset,
)
from shared.hashing import sha256_file

WorkbenchMode = Literal["research", "official"]
ALLOWED_WORKBENCH_SPLITS = {"train", "val"}
RESEARCH_OVERRIDE_FIELDS = {"imgsz", "batch", "epochs", "patience"}


@dataclass(frozen=True)
class WorkbenchPaths:
    """Explicit repository, dataset, Baseline and ignored output paths."""

    repository_root: Path
    dataset_root: Path
    baseline_artifact_dir: Path
    output_root: Path


@dataclass(frozen=True)
class WorkbenchSample:
    """Compact allowed-split GT description used by EDA and galleries."""

    sample_id: str
    split: str
    class_name: str
    is_negative: bool
    component_count: int
    component_area_ratios: tuple[float, ...]
    size_buckets: tuple[str, ...]
    image_width: int
    image_height: int
    image_path: str
    label_path: str


@dataclass(frozen=True)
class OfficialPreflight:
    """Identity evidence rendered immediately before an explicit official run."""

    experiment_id: str
    intervention_type: str
    git_commit: str | None
    working_tree_dirty: bool
    config_path: str
    config_sha256: str
    manifest_sha256: str
    baseline_model_sha256: str
    baseline_metadata_sha256: str
    model_initialization: str
    model_family: str
    imgsz: int
    batch: int
    epochs: int
    patience: int
    seed: int
    requested_device: str
    train_count: int
    val_count: int
    test_split_used: bool
    telemetry_enabled: bool
    telemetry_interval_seconds: float
    sampling_summary: dict[str, Any] | None
    execution_environment: dict[str, Any]

    # ADD 2026-08-27: Notebook display와 audit용 strict mapping을 반환한다.
    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        json.dumps(payload, allow_nan=False)
        return payload


# ADD 2026-08-27: Workbench mode와 official override prohibition을 fail-fast한다.
def validate_workbench_controls(
    mode: WorkbenchMode,
    *,
    overrides: dict[str, object],
) -> None:
    if mode not in {"research", "official"}:
        raise ValueError("Workbench mode must be research or official.")
    unsupported = set(overrides) - RESEARCH_OVERRIDE_FIELDS
    if unsupported:
        raise ValueError(f"Unsupported research override fields: {sorted(unsupported)}")
    if mode == "official" and overrides:
        raise ValueError("Official experiment mode rejects every notebook training override.")


# ADD 2026-08-27: Official config를 mutate하지 않고 isolated research training config를 만든다.
def build_research_config(
    baseline: YoloSegmentationBaselineConfig,
    *,
    overrides: dict[str, object],
    output_root: Path,
) -> YoloSegmentationBaselineConfig:
    validate_workbench_controls("research", overrides=overrides)
    if any(not isinstance(value, int) or isinstance(value, bool) for value in overrides.values()):
        raise ValueError("Research training overrides must be integer values.")
    imgsz = cast(int, overrides.get("imgsz", baseline.training.imgsz))
    batch = cast(int, overrides.get("batch", baseline.training.batch))
    epochs = cast(int, overrides.get("epochs", baseline.training.epochs))
    patience = cast(int, overrides.get("patience", baseline.training.patience))
    research = replace(
        baseline,
        training=replace(
            baseline.training,
            imgsz=imgsz,
            batch=batch,
            epochs=epochs,
            patience=patience,
        ),
        output=YoloOutputConfig(
            artifact_root=output_root / "artifacts",
            training_runtime_root=output_root / "training",
            evaluation_root=output_root / "validation",
        ),
    )
    research.validate()
    return research


# ADD 2026-08-27: Manifest에서 train/val만 materialize하고 test row를 workbench에서 봉인한다.
def load_workbench_records(dataset_root: Path) -> list[DerivedManifestRecord]:
    records = read_derived_manifest(
        dataset_root / "manifest.csv",
        allowed_splits=ALLOWED_WORKBENCH_SPLITS,
    )
    if not records or any(
        record.derived_split not in ALLOWED_WORKBENCH_SPLITS for record in records
    ):
        raise ValueError("Workbench records must contain only train and validation rows.")
    identities = [(record.derived_split, record.sample_id) for record in records]
    if len(identities) != len(set(identities)):
        raise ValueError("Workbench train/validation sample identities must be unique.")
    return sorted(records, key=lambda record: (record.derived_split, record.sample_id))


# ADD 2026-08-27: Shared polygon rasterization으로 allowed-split GT descriptors를 만든다.
def describe_workbench_samples(
    records: list[DerivedManifestRecord],
    *,
    dataset_root: Path,
    classes: dict[int, str],
    size_policy: SizeBucketPolicy,
) -> list[WorkbenchSample]:
    descriptions: list[WorkbenchSample] = []
    for record in records:
        if record.derived_split not in ALLOWED_WORKBENCH_SPLITS:
            raise ValueError("EDA rejects sealed derived-test records.")
        label_text = (dataset_root / record.label_path).read_text(encoding="utf-8")
        instances = rasterize_segmentation_label_instances(
            label_text,
            image_width=record.image_width,
            image_height=record.image_height,
            valid_class_ids=set(classes),
        )
        if record.is_negative and instances:
            raise ValueError("Good-negative workbench record contains GT instances.")
        if not record.is_negative and len(instances) != record.component_count:
            raise ValueError("Workbench GT component count does not match the Manifest.")
        descriptions.append(
            WorkbenchSample(
                sample_id=record.sample_id,
                split=record.derived_split,
                class_name="good" if record.is_negative else record.target_class,
                is_negative=record.is_negative,
                component_count=record.component_count,
                component_area_ratios=tuple(instance.area_ratio for instance in instances),
                size_buckets=tuple(
                    size_policy.classify(instance.area_ratio) for instance in instances
                ),
                image_width=record.image_width,
                image_height=record.image_height,
                image_path=record.image_path,
                label_path=record.label_path,
            )
        )
    return descriptions


# ADD 2026-08-27: Allowed-split image/class/size/component 분포를 compact JSON으로 집계한다.
def build_eda_summary(
    samples: list[WorkbenchSample],
    *,
    manifest_sha256: str,
    dataset_name: str,
    dataset_version: str,
) -> dict[str, Any]:
    if not samples or any(sample.split not in ALLOWED_WORKBENCH_SPLITS for sample in samples):
        raise ValueError("EDA summary accepts only non-empty train/validation samples.")
    by_split = Counter(sample.split for sample in samples)
    positive_by_split = Counter(sample.split for sample in samples if not sample.is_negative)
    negative_by_split = Counter(sample.split for sample in samples if sample.is_negative)
    class_images = Counter(sample.class_name for sample in samples if not sample.is_negative)
    class_instances: Counter[str] = Counter()
    size_instances: Counter[str] = Counter()
    for sample in samples:
        if not sample.is_negative:
            class_instances[sample.class_name] += sample.component_count
        size_instances.update(sample.size_buckets)
    component_distribution = Counter(sample.component_count for sample in samples)
    positive_samples = [sample for sample in samples if not sample.is_negative]
    payload = {
        "schema_version": 1,
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "manifest_sha256": manifest_sha256,
        "included_splits": ["train", "val"],
        "test_split_status": "SEALED_NOT_USED",
        "image_distribution": {
            split: {
                "total": by_split[split],
                "positive": positive_by_split[split],
                "good_negative": negative_by_split[split],
            }
            for split in ("train", "val")
        },
        "class_image_count": dict(sorted(class_images.items())),
        "class_component_count": dict(sorted(class_instances.items())),
        "size_component_count": {
            bucket: size_instances[bucket] for bucket in ("small", "medium", "large")
        },
        "component_count_distribution": {
            str(count): value for count, value in sorted(component_distribution.items())
        },
        "component_type": {
            "single_component_samples": sum(
                sample.component_count == 1 for sample in positive_samples
            ),
            "multi_component_samples": sum(
                sample.component_count > 1 for sample in positive_samples
            ),
        },
        "image_dimensions": dict(
            sorted(
                Counter(f"{sample.image_width}x{sample.image_height}" for sample in samples).items()
            )
        ),
    }
    json.dumps(payload, allow_nan=False)
    return payload


# ADD 2026-08-27: Class/size/component/good coverage를 stable SHA ranking으로 선택한다.
def select_representative_samples(
    samples: list[WorkbenchSample],
    *,
    seed: int,
    max_count: int = 12,
) -> list[WorkbenchSample]:
    if max_count <= 0 or any(sample.split not in ALLOWED_WORKBENCH_SPLITS for sample in samples):
        raise ValueError("Representative selection input is invalid.")
    ordered = sorted(
        samples,
        key=lambda sample: (
            deterministic_rank(sample.sample_id, seed=seed, namespace="yolo-workbench"),
            sample.sample_id,
        ),
    )
    selectors = [
        *(
            lambda item, class_name=name: item.class_name == class_name
            for name in ("bent", "color", "scratch")
        ),
        *(
            lambda item, bucket=name: bucket in item.size_buckets
            for name in ("small", "medium", "large")
        ),
        lambda item: item.component_count == 1,
        lambda item: item.component_count > 1,
        lambda item: item.is_negative and item.split == "train",
        lambda item: item.is_negative and item.split == "val",
    ]
    selected: list[WorkbenchSample] = []
    for selector in selectors:
        candidate = next((item for item in ordered if selector(item)), None)
        if candidate is not None and candidate not in selected:
            selected.append(candidate)
    for sample in ordered:
        if len(selected) >= max_count:
            break
        if sample not in selected:
            selected.append(sample)
    return selected[:max_count]


# ADD 2026-08-27: Full allowed sample set에서 deterministic val-small hypothesis sample을 고른다.
def select_small_validation_sample(
    samples: list[WorkbenchSample],
    *,
    seed: int,
) -> WorkbenchSample:
    if any(sample.split not in ALLOWED_WORKBENCH_SPLITS for sample in samples):
        raise ValueError("Small validation selection rejects records outside train/validation.")
    eligible = [
        sample
        for sample in samples
        if sample.split == "val" and not sample.is_negative and "small" in sample.size_buckets
    ]
    if not eligible:
        raise ValueError("No positive validation sample contains a small defect component.")
    return min(
        eligible,
        key=lambda sample: (
            deterministic_rank(
                sample.sample_id,
                seed=seed,
                namespace="yolo-workbench:val-small-representation",
            ),
            sample.sample_id,
        ),
    )


# ADD 2026-08-28: Sampling audit를 만든다. → MODIFY 2026-09-01: Actual C4-2C view를 검증한다.
def build_sampling_workbench_summary(
    *,
    experiment: YoloExperimentConfig,
    baseline: YoloSegmentationBaselineConfig,
    dataset_root: Path,
    records: list[DerivedManifestRecord],
) -> dict[str, Any] | None:
    if experiment.sampling_policy is None and experiment.crop_sampling_policy is None:
        return None
    train_records = [record for record in records if record.derived_split == "train"]
    plan = plan_component_aware_train_view(
        dataset_root=dataset_root,
        train_records=train_records,
        contract=baseline.dataset_contract,
        experiment_id=experiment.experiment_id,
    )
    expected = experiment.expected_train_view
    crop_expected = experiment.expected_crop_train_view
    if expected is not None:
        expected.validate_evidence(plan.evidence)
    elif crop_expected is None:
        raise ValueError("Sampling workbench requires expected train-view evidence.")
    small_ids = set(plan.eligibility.small_aware_sample_ids)
    multi_ids = set(plan.eligibility.multi_component_sample_ids)
    eligible_rows = [
        {
            "sample_id": sample_id,
            "small_aware": sample_id in small_ids,
            "multi_component": sample_id in multi_ids,
            "overlap": sample_id in small_ids and sample_id in multi_ids,
            "multiplicity": plan.evidence.sample_multiplicity[sample_id],
        }
        for sample_id in plan.eligibility.eligible_sample_ids
    ]
    summary: dict[str, Any] = {
        "sampling_rule_version": plan.evidence.sampling_rule_version,
        "sampling_validation_source": "TRAIN_ONLY",
        "test_split": "SEALED_NOT_USED",
        "train_list_sha256": plan.evidence.train_list_sha256,
        "canonical_train_entries": plan.evidence.unique_train_count,
        "expanded_train_entries": plan.evidence.expanded_entry_count,
        "positive_exposure": [
            plan.evidence.unique_positive_count,
            plan.evidence.expanded_positive_count,
        ],
        "good_negative_exposure": [
            plan.evidence.unique_good_negative_count,
            plan.evidence.expanded_good_negative_count,
        ],
        "canonical_good_negative_ratio": (
            plan.evidence.unique_good_negative_count / plan.evidence.unique_train_count
        ),
        "expanded_good_negative_ratio": plan.evidence.expanded_good_negative_ratio,
        "small_aware_count": plan.evidence.small_aware_count,
        "multi_component_count": plan.evidence.multi_component_count,
        "eligible_overlap_count": plan.evidence.eligible_overlap_count,
        "eligible_union_count": plan.evidence.eligible_union_count,
        "eligible_multiplicity": plan.evidence.eligible_multiplicity,
        "observed_train_small_cutoff": plan.evidence.observed_train_small_cutoff,
        "eligible_samples": eligible_rows,
    }
    if crop_expected is not None:
        crop_policy = experiment.crop_sampling_policy
        trainer_overrides = experiment.trainer_overrides
        if crop_policy is None or trainer_overrides is None:
            raise ValueError("C4-2C Workbench recipe is incomplete.")
        actual = crop_expected.validate_planned_evidence(plan.evidence)
        summary.update(
            {
                "sampling_mode": crop_policy.sampling_mode,
                "canonical_train_entries": actual["canonical_entries"],
                "component_duplicate_entries": actual["component_duplicate_entries"],
                "crop_entries": actual["small_centered_crop_entries"],
                "expanded_train_entries": actual["total_entries"],
                "positive_exposure": actual["positive_exposure"],
                "good_negative_exposure": actual["negative_exposure"],
                "crop_sample_ids": list(plan.eligibility.small_aware_sample_ids),
                "crop_size": crop_policy.crop_size,
                "trainer_overrides": asdict(trainer_overrides),
                "expected_train_view": asdict(crop_expected),
                "actual_train_view": actual,
                "train_view_validation": "PASS",
                "test_used": False,
            }
        )
    return summary


# ADD 2026-08-27: Official identity를 검증한다. → MODIFY 2026-08-31: C4-2C summary를 포함한다.
def build_official_preflight(
    *,
    experiment: YoloExperimentConfig,
    baseline: YoloSegmentationBaselineConfig,
    paths: WorkbenchPaths,
    requested_device: str,
    overrides: dict[str, object],
    execution_environment: LockedEnvironmentProvenance,
    repository_provenance: RepositoryProvenance | None = None,
) -> OfficialPreflight:
    validate_workbench_controls("official", overrides=overrides)
    if requested_device != "cuda":
        raise ValueError("Official controlled experiment preflight requires explicit CUDA.")
    execution_environment.validate(paths.repository_root, require_cuda=True)
    candidate = experiment.training_config(baseline)
    records = list(validate_experiment_dataset(paths.dataset_root, baseline.dataset_contract))
    counts = Counter(record.derived_split for record in records)
    expected_counts = {
        split: baseline.dataset_contract.sample_counts[split] for split in ("train", "val")
    }
    if counts != expected_counts:
        raise ValueError("Official workbench train/validation identity count changed.")
    model_path = paths.baseline_artifact_dir / "model" / "model.pt"
    metadata_path = paths.baseline_artifact_dir / "model" / "metadata.json"
    source = experiment.baseline_evidence["sources"]
    if sha256_file(model_path) != source["checkpoint_sha256"]:
        raise ValueError("Official preflight Baseline model SHA mismatch.")
    if sha256_file(metadata_path) != source["metadata_sha256"]:
        raise ValueError("Official preflight Baseline metadata SHA mismatch.")
    provenance = repository_provenance or resolve_repository_provenance(paths.repository_root)
    sampling_summary = build_sampling_workbench_summary(
        experiment=experiment,
        baseline=baseline,
        dataset_root=paths.dataset_root,
        records=records,
    )
    return OfficialPreflight(
        experiment_id=experiment.experiment_id,
        intervention_type=experiment.intervention_type,
        git_commit=provenance.git_commit,
        working_tree_dirty=provenance.working_tree_dirty,
        config_path=experiment.config_path.as_posix(),
        config_sha256=sha256_file(experiment.config_path),
        manifest_sha256=sha256_file(paths.dataset_root / "manifest.csv"),
        baseline_model_sha256=sha256_file(model_path),
        baseline_metadata_sha256=sha256_file(metadata_path),
        model_initialization=candidate.model.weights,
        model_family=candidate.model.architecture,
        imgsz=candidate.training.imgsz,
        batch=candidate.training.batch,
        epochs=candidate.training.epochs,
        patience=candidate.training.patience,
        seed=candidate.training.seed,
        requested_device=requested_device,
        train_count=counts["train"],
        val_count=counts["val"],
        test_split_used=False,
        telemetry_enabled=True,
        telemetry_interval_seconds=experiment.telemetry.sample_interval_seconds,
        sampling_summary=sampling_summary,
        execution_environment=execution_environment.to_json_dict(),
    )


# ADD 2026-08-27: EDA payload를 ignored workbench output에 strict JSON으로 저장한다.
def write_eda_summary(payload: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path
