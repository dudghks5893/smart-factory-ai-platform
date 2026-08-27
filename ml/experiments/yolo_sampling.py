"""Deterministic train-only sampling evidence for controlled YOLO experiments."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

from ml.datasets.segmentation_annotations import rasterize_segmentation_label_instances
from ml.datasets.yolo_segmentation_manifest import DerivedManifestRecord
from ml.training.yolo_segmentation import YoloDatasetContract, validate_artifact_id
from shared.hashing import is_sha256_digest, sha256_bytes, sha256_file

TRAIN_VIEW_SCHEMA_VERSION = 1
SAMPLING_RULE_VERSION = "component_aware_bottom_third_union_multi_x2_v1"
SMALL_FRACTION_RULE = "bottom_third"
ELIGIBLE_MULTIPLICITY = 2
TRAIN_VIEW_RELATIVE_ROOT = Path("outputs/experiments/yolo_segmentation")
TRAIN_LIST_PATH_BASE = "canonical_dataset_root"
TRAIN_VIEW_ORDERING_POLICY = (
    "canonical_sample_id_order_then_eligible_second_copy_in_sample_id_order"
)


@dataclass(frozen=True)
class TrainSampleProfile:
    """Train-only identity and source-resolution component evidence for one image."""

    sample_id: str
    image_path: str
    is_negative: bool
    component_count: int
    image_min_component_area_ratio: float | None

    # ADD 2026-08-28: Sampling profile의 train path와 positive/negative geometry 계약을 검증한다.
    def validate(self) -> None:
        _validate_relative_train_path(self.image_path, kind="image", sample_id=self.sample_id)
        if not self.sample_id:
            raise ValueError("Train sampling sample ID must not be blank.")
        if self.is_negative:
            if self.component_count != 0 or self.image_min_component_area_ratio is not None:
                raise ValueError("Good-negative sampling profile must remain component-empty.")
            return
        area_ratio = self.image_min_component_area_ratio
        if (
            self.component_count <= 0
            or area_ratio is None
            or not math.isfinite(area_ratio)
            or not 0.0 < area_ratio <= 1.0
        ):
            raise ValueError("Positive sampling profile has invalid component geometry.")


@dataclass(frozen=True)
class SamplingEligibility:
    """Predeclared bottom-third and multi-component eligibility sets."""

    small_aware_sample_ids: tuple[str, ...]
    multi_component_sample_ids: tuple[str, ...]
    eligible_sample_ids: tuple[str, ...]
    observed_train_small_cutoff: float


@dataclass(frozen=True)
class TrainViewEvidence:
    """Portable machine evidence for one expanded train index view."""

    schema_version: int
    experiment_id: str
    sampling_rule_version: str
    canonical_manifest_sha256: str
    unique_train_count: int
    unique_positive_count: int
    unique_good_negative_count: int
    small_aware_count: int
    multi_component_count: int
    eligible_overlap_count: int
    eligible_union_count: int
    expanded_entry_count: int
    expanded_positive_count: int
    expanded_good_negative_count: int
    expanded_good_negative_ratio: float
    small_fraction_rule: str
    eligible_multiplicity: int
    observed_train_small_cutoff: float
    eligible_sample_ids: tuple[str, ...]
    sample_multiplicity: dict[str, int]
    train_list_sha256: str
    train_list_path_base: str
    ordering_policy: str
    validation_used_for_sampling: bool
    test_split_used: bool

    # ADD 2026-08-28: Train-view count arithmetic, policy flags와 portable hash를 검증한다.
    def validate(self) -> None:
        if self.schema_version != TRAIN_VIEW_SCHEMA_VERSION:
            raise ValueError("Unsupported YOLO train-view evidence schema.")
        validate_artifact_id(self.experiment_id)
        if self.sampling_rule_version != SAMPLING_RULE_VERSION:
            raise ValueError("Unexpected YOLO sampling rule version.")
        if not is_sha256_digest(self.canonical_manifest_sha256) or not is_sha256_digest(
            self.train_list_sha256
        ):
            raise ValueError("Train-view provenance fields must be SHA-256 digests.")
        if self.unique_train_count != self.unique_positive_count + self.unique_good_negative_count:
            raise ValueError("Unique train-view count arithmetic is invalid.")
        if self.eligible_union_count != (
            self.small_aware_count + self.multi_component_count - self.eligible_overlap_count
        ):
            raise ValueError("Sampling eligibility union arithmetic is invalid.")
        if self.expanded_positive_count != self.unique_positive_count + self.eligible_union_count:
            raise ValueError("Expanded positive exposure count is invalid.")
        if self.expanded_good_negative_count != self.unique_good_negative_count:
            raise ValueError("Good-negative multiplicity must remain one.")
        if self.expanded_entry_count != (
            self.expanded_positive_count + self.expanded_good_negative_count
        ):
            raise ValueError("Expanded train-view count arithmetic is invalid.")
        expected_ratio = self.expanded_good_negative_count / self.expanded_entry_count
        if not math.isclose(self.expanded_good_negative_ratio, expected_ratio):
            raise ValueError("Expanded good-negative exposure ratio is invalid.")
        if (
            self.small_fraction_rule != SMALL_FRACTION_RULE
            or self.eligible_multiplicity != ELIGIBLE_MULTIPLICITY
            or self.validation_used_for_sampling
            or self.test_split_used
        ):
            raise ValueError("Train-view sampling or sealed-split policy changed.")
        if self.train_list_path_base != TRAIN_LIST_PATH_BASE:
            raise ValueError("Train-view paths must remain canonical-dataset-root relative.")
        if self.ordering_policy != TRAIN_VIEW_ORDERING_POLICY:
            raise ValueError("Train-view ordering policy changed.")
        if not math.isfinite(self.observed_train_small_cutoff):
            raise ValueError("Observed train small cutoff must be finite.")
        eligible = set(self.eligible_sample_ids)
        if len(eligible) != self.eligible_union_count or self.eligible_sample_ids != tuple(
            sorted(eligible)
        ):
            raise ValueError("Eligible sample IDs are duplicated or incomplete.")
        if len(self.sample_multiplicity) != self.unique_train_count or any(
            value not in {1, ELIGIBLE_MULTIPLICITY} for value in self.sample_multiplicity.values()
        ):
            raise ValueError("Per-sample train multiplicity is invalid.")
        if eligible != {
            sample_id
            for sample_id, multiplicity in self.sample_multiplicity.items()
            if multiplicity == ELIGIBLE_MULTIPLICITY
        }:
            raise ValueError("Eligible IDs and per-sample multiplicity disagree.")

    # ADD 2026-08-28: Train-view evidence를 deterministic strict JSON mapping으로 변환한다.
    def to_json_dict(self) -> dict[str, object]:
        self.validate()
        payload = asdict(self)
        json.dumps(payload, sort_keys=True, allow_nan=False)
        return payload


@dataclass(frozen=True)
class TrainViewArtifact:
    """Ignored deterministic train list and its portable evidence metadata."""

    output_dir: Path
    train_list_path: Path
    metadata_path: Path
    evidence: TrainViewEvidence


# ADD 2026-08-28: Canonical dataset-root relative train path만 sampling input으로 허용한다.
def _validate_relative_train_path(value: str, *, kind: str, sample_id: str) -> PurePosixPath:
    path = PurePosixPath(value)
    expected_prefix = ("images", "train") if kind == "image" else ("labels", "train")
    if (
        not value
        or "\\" in value
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != value
        or path.parts[:2] != expected_prefix
        or len(path.parts) != 3
    ):
        raise ValueError(f"Sampling {kind} path is not portable train content: {sample_id}")
    return path


# ADD 2026-08-28: Train label을 source-resolution mask로 복원해 sampling profile을 만든다.
def profile_train_samples(
    records: Sequence[DerivedManifestRecord],
    *,
    dataset_root: Path,
    valid_class_ids: set[int],
) -> tuple[TrainSampleProfile, ...]:
    if not records or not valid_class_ids:
        raise ValueError("Train sampling requires records and valid segmentation classes.")
    profiles: list[TrainSampleProfile] = []
    observed_ids: set[str] = set()
    observed_paths: set[str] = set()
    resolved_root = dataset_root.resolve()
    for record in sorted(records, key=lambda item: item.sample_id):
        if record.derived_split != "train":
            raise ValueError("Sampling input must contain canonical TRAIN records only.")
        if record.sample_id in observed_ids or record.image_path in observed_paths:
            raise ValueError("Train sampling identities and image paths must be unique.")
        observed_ids.add(record.sample_id)
        observed_paths.add(record.image_path)
        image_relative = _validate_relative_train_path(
            record.image_path,
            kind="image",
            sample_id=record.sample_id,
        )
        label_relative = _validate_relative_train_path(
            record.label_path,
            kind="label",
            sample_id=record.sample_id,
        )
        image_path = dataset_root.joinpath(*image_relative.parts)
        label_path = dataset_root.joinpath(*label_relative.parts)
        try:
            image_path.resolve().relative_to(resolved_root)
            label_path.resolve().relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError("Train sampling content escapes the canonical dataset root.") from exc
        if not image_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"Train sampling content is missing: {record.sample_id}")

        # Eligibility는 train label만 source-resolution mask로 복원해 계산한다.
        label_text = label_path.read_text(encoding="utf-8")
        instances = rasterize_segmentation_label_instances(
            label_text,
            image_width=record.image_width,
            image_height=record.image_height,
            valid_class_ids=valid_class_ids,
        )
        if record.is_negative:
            if label_text or instances or record.component_count != 0 or record.polygon_count != 0:
                raise ValueError("Good-negative train sample contains component labels.")
            minimum_area: float | None = None
        else:
            if not instances or len(instances) != record.component_count:
                raise ValueError("Positive train component count does not match the Manifest.")
            if len(instances) != record.polygon_count:
                raise ValueError(
                    "Positive train polygon count does not match rasterized components."
                )
            try:
                expected_class_id = int(record.target_class_id)
            except ValueError as exc:
                raise ValueError("Positive train target class ID is malformed.") from exc
            if any(instance.class_id != expected_class_id for instance in instances):
                raise ValueError("Positive train polygon class does not match the Manifest.")
            minimum_area = min(instance.area_ratio for instance in instances)
        profile = TrainSampleProfile(
            sample_id=record.sample_id,
            image_path=image_relative.as_posix(),
            is_negative=record.is_negative,
            component_count=len(instances),
            image_min_component_area_ratio=minimum_area,
        )
        profile.validate()
        profiles.append(profile)
    return tuple(profiles)


# ADD 2026-08-28: Train-derived bottom-third와 multi-component 합집합을 deterministic하게 선택한다.
def select_component_aware_eligibility(
    profiles: Sequence[TrainSampleProfile],
) -> SamplingEligibility:
    if not profiles:
        raise ValueError("Sampling eligibility requires non-empty train profiles.")
    for profile in profiles:
        profile.validate()
    if len({profile.sample_id for profile in profiles}) != len(profiles):
        raise ValueError("Sampling profiles must have unique sample IDs.")
    positives = [profile for profile in profiles if not profile.is_negative]
    if not positives:
        raise ValueError("Sampling eligibility requires positive train samples.")
    ordered_by_small = sorted(
        positives,
        key=lambda profile: (
            profile.image_min_component_area_ratio,
            profile.sample_id,
        ),
    )
    small_count = math.ceil(len(ordered_by_small) / 3)
    small_profiles = ordered_by_small[:small_count]
    small_ids = tuple(sorted(profile.sample_id for profile in small_profiles))
    multi_ids = tuple(
        sorted(profile.sample_id for profile in positives if profile.component_count > 1)
    )
    eligible_ids = tuple(sorted(set(small_ids).union(multi_ids)))
    cutoff = small_profiles[-1].image_min_component_area_ratio
    if cutoff is None:
        raise RuntimeError("Positive train sampling cutoff was not calculated.")
    return SamplingEligibility(
        small_aware_sample_ids=small_ids,
        multi_component_sample_ids=multi_ids,
        eligible_sample_ids=eligible_ids,
        observed_train_small_cutoff=cutoff,
    )


# ADD 2026-08-28: Canonical entries 뒤에 eligible second copies를 append해 x2 index view를 만든다.
def expand_component_aware_train_entries(
    profiles: Sequence[TrainSampleProfile],
    eligibility: SamplingEligibility,
) -> tuple[tuple[str, ...], dict[str, int]]:
    canonical = sorted(profiles, key=lambda profile: profile.sample_id)
    if not canonical or len({profile.sample_id for profile in canonical}) != len(canonical):
        raise ValueError("Expanded train view requires unique canonical profiles.")
    profile_by_id = {profile.sample_id: profile for profile in canonical}
    eligible_ids = set(eligibility.eligible_sample_ids)
    if eligibility.eligible_sample_ids != tuple(sorted(eligible_ids)):
        raise ValueError("Eligible sample IDs must be unique deterministic sample_id order.")
    if len({profile.image_path for profile in canonical}) != len(canonical):
        raise ValueError("Expanded train view requires unique canonical image paths.")
    if not eligible_ids.issubset(profile_by_id) or any(
        profile_by_id[sample_id].is_negative for sample_id in eligible_ids
    ):
        raise ValueError("Only canonical positive train samples may be oversampled.")
    canonical_entries = [profile.image_path for profile in canonical]
    duplicate_entries = [
        profile_by_id[sample_id].image_path for sample_id in eligibility.eligible_sample_ids
    ]
    multiplicity = {
        profile.sample_id: ELIGIBLE_MULTIPLICITY if profile.sample_id in eligible_ids else 1
        for profile in canonical
    }
    return tuple([*canonical_entries, *duplicate_entries]), multiplicity


# ADD 2026-08-28: Component-aware train list와 provenance를 ignored experiment output에 쓴다.
def build_component_aware_train_view(
    *,
    repository_root: Path,
    dataset_root: Path,
    train_records: Sequence[DerivedManifestRecord],
    contract: YoloDatasetContract,
    experiment_id: str,
) -> TrainViewArtifact:
    contract.validate()
    validate_artifact_id(experiment_id)
    manifest_path = dataset_root / "manifest.csv"
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 != contract.manifest_sha256:
        raise ValueError("Train-view Manifest SHA does not match the canonical dataset contract.")

    # Canonical train content에서만 eligibility와 expanded exposure를 결정한다.
    profiles = profile_train_samples(
        train_records,
        dataset_root=dataset_root,
        valid_class_ids=set(contract.classes),
    )
    unique_positive_count = sum(not profile.is_negative for profile in profiles)
    unique_good_negative_count = sum(profile.is_negative for profile in profiles)
    if (
        len(profiles) != contract.sample_counts["train"]
        or unique_positive_count != contract.sample_counts["train_positive"]
        or unique_good_negative_count != contract.sample_counts["train_negative"]
    ):
        raise ValueError("Train-view unique counts do not match the canonical dataset contract.")
    eligibility = select_component_aware_eligibility(profiles)
    entries, multiplicity = expand_component_aware_train_entries(profiles, eligibility)
    profile_by_id = {profile.sample_id: profile for profile in profiles}
    eligible_set = set(eligibility.eligible_sample_ids)
    small_set = set(eligibility.small_aware_sample_ids)
    multi_set = set(eligibility.multi_component_sample_ids)
    expanded_positive_count = unique_positive_count + len(eligible_set)
    expanded_good_negative_count = unique_good_negative_count
    train_list_bytes = ("\n".join(entries) + "\n").encode()
    train_list_sha256 = sha256_bytes(train_list_bytes)
    evidence = TrainViewEvidence(
        schema_version=TRAIN_VIEW_SCHEMA_VERSION,
        experiment_id=experiment_id,
        sampling_rule_version=SAMPLING_RULE_VERSION,
        canonical_manifest_sha256=manifest_sha256,
        unique_train_count=len(profiles),
        unique_positive_count=unique_positive_count,
        unique_good_negative_count=unique_good_negative_count,
        small_aware_count=len(small_set),
        multi_component_count=len(multi_set),
        eligible_overlap_count=len(small_set.intersection(multi_set)),
        eligible_union_count=len(eligible_set),
        expanded_entry_count=len(entries),
        expanded_positive_count=expanded_positive_count,
        expanded_good_negative_count=expanded_good_negative_count,
        expanded_good_negative_ratio=expanded_good_negative_count / len(entries),
        small_fraction_rule=SMALL_FRACTION_RULE,
        eligible_multiplicity=ELIGIBLE_MULTIPLICITY,
        observed_train_small_cutoff=eligibility.observed_train_small_cutoff,
        eligible_sample_ids=eligibility.eligible_sample_ids,
        sample_multiplicity={
            sample_id: multiplicity[sample_id] for sample_id in sorted(multiplicity)
        },
        train_list_sha256=train_list_sha256,
        train_list_path_base=TRAIN_LIST_PATH_BASE,
        ordering_policy=TRAIN_VIEW_ORDERING_POLICY,
        validation_used_for_sampling=False,
        test_split_used=False,
    )
    evidence.validate()
    if any(profile_by_id[sample_id].is_negative for sample_id in eligible_set):
        raise RuntimeError("Good-negative multiplicity changed during train-view construction.")

    # Absolute host paths를 evidence에 넣지 않고 Git-ignored experiment output만 생성한다.
    output_dir = repository_root.resolve() / TRAIN_VIEW_RELATIVE_ROOT / experiment_id / "train_view"
    if output_dir.exists():
        raise FileExistsError(f"YOLO train-view output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    train_list_path = output_dir / "train.txt"
    metadata_path = output_dir / "metadata.json"
    train_list_path.write_bytes(train_list_bytes)
    metadata_path.write_text(
        json.dumps(evidence.to_json_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return TrainViewArtifact(
        output_dir=output_dir,
        train_list_path=train_list_path,
        metadata_path=metadata_path,
        evidence=evidence,
    )
