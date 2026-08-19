"""Validated configuration loading for the PatchCore baseline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from ml.training.device import SUPPORTED_DEVICES, DeviceName
from ml.training.preprocessing import PatchCorePreprocessingConfig


@dataclass(frozen=True)
class PatchCoreModelConfig:
    """Anomalib PatchCore model settings."""

    name: str
    implementation: str
    backbone: str
    layers: tuple[str, ...]
    pretrained: bool
    coreset_sampling_ratio: float
    num_neighbors: int

    # ADD 2026-08-19: PatchCore model configuration의 domain invariant를 검증한다.
    def validate(self) -> None:
        if self.name != "patchcore":
            raise ValueError("model.name must be 'patchcore'.")
        if self.implementation != "anomalib":
            raise ValueError("model.implementation must be 'anomalib'.")
        if not self.backbone:
            raise ValueError("model.backbone must not be empty.")
        if not self.layers or len(set(self.layers)) != len(self.layers):
            raise ValueError("model.layers must contain unique layer names.")
        if not 0.0 < self.coreset_sampling_ratio <= 1.0:
            raise ValueError("model.coreset_sampling_ratio must be in (0, 1].")
        if self.num_neighbors <= 0:
            raise ValueError("model.num_neighbors must be positive.")


@dataclass(frozen=True)
class TrainingConfig:
    """PatchCore construction runtime settings."""

    random_seed: int
    device: DeviceName
    batch_size: int
    num_workers: int

    # ADD 2026-08-19: Training runtime configuration을 검증한다.
    def validate(self) -> None:
        if self.random_seed < 0:
            raise ValueError("training.random_seed must be non-negative.")
        if self.device not in SUPPORTED_DEVICES:
            raise ValueError(f"training.device must be one of {SUPPORTED_DEVICES}.")
        if self.batch_size <= 0:
            raise ValueError("training.batch_size must be positive.")
        if self.num_workers < 0:
            raise ValueError("training.num_workers must be non-negative.")


@dataclass(frozen=True)
class OutputConfig:
    """Default roots for generated model and prediction artifacts."""

    artifact_root: Path
    prediction_root: Path


@dataclass(frozen=True)
class PatchCoreBaselineConfig:
    """Complete validated PatchCore baseline configuration."""

    model: PatchCoreModelConfig
    preprocessing: PatchCorePreprocessingConfig
    training: TrainingConfig
    output: OutputConfig

    # ADD 2026-08-19: PatchCore baseline의 모든 configuration section을 검증한다.
    def validate(self) -> None:
        self.model.validate()
        self.preprocessing.validate()
        self.training.validate()


# ADD 2026-08-19: Configuration section이 mapping인지 검증한다.
def _mapping(value: object, section: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{section}' must be a mapping.")
    return cast(dict[str, Any], value)


# ADD 2026-08-19: Configuration 값을 두 정수 tuple로 변환한다.
def _pair(value: object, field: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{field} must contain exactly two integers.")
    try:
        return int(value[0]), int(value[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain exactly two integers.") from exc


# ADD 2026-08-19: Configuration 값을 세 실수 tuple로 변환한다.
def _triple(value: object, field: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{field} must contain exactly three numbers.")
    try:
        return float(value[0]), float(value[1]), float(value[2])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain exactly three numbers.") from exc


# ADD 2026-08-19: Configuration 값이 boolean인지 검증한다.
def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean.")
    return value


# ADD 2026-08-19: Configuration 값을 non-empty string tuple로 검증한다.
def _string_tuple(value: object, field: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise ValueError(f"{field} must contain one or more non-empty strings.")
    return tuple(value)


# ADD 2026-08-19: Load and validate the PatchCore baseline YAML configuration.
def load_patchcore_config(path: Path) -> PatchCoreBaselineConfig:
    """Load and validate the PatchCore baseline YAML configuration."""
    # 외부 YAML 파일을 읽고 root 구조를 검증한다.
    try:
        with path.open(encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    except OSError as exc:
        raise ValueError(f"Cannot read PatchCore config: {path}") from exc

    root = _mapping(raw, "root")

    # 각 section을 typed configuration으로 변환하며 field 오류를 하나의 config 오류로 감싼다.
    try:
        model = _mapping(root["model"], "model")
        preprocessing = _mapping(root["preprocessing"], "preprocessing")
        training = _mapping(root["training"], "training")
        output = _mapping(root["output"], "output")

        device = str(training["device"])
        if device not in SUPPORTED_DEVICES:
            raise ValueError(f"training.device must be one of {SUPPORTED_DEVICES}.")

        config = PatchCoreBaselineConfig(
            model=PatchCoreModelConfig(
                name=str(model["name"]),
                implementation=str(model["implementation"]),
                backbone=str(model["backbone"]),
                layers=_string_tuple(model["layers"], "model.layers"),
                pretrained=_boolean(model["pretrained"], "model.pretrained"),
                coreset_sampling_ratio=float(model["coreset_sampling_ratio"]),
                num_neighbors=int(model["num_neighbors"]),
            ),
            preprocessing=PatchCorePreprocessingConfig(
                resize_size=_pair(preprocessing["resize_size"], "preprocessing.resize_size"),
                center_crop_size=_pair(
                    preprocessing["center_crop_size"],
                    "preprocessing.center_crop_size",
                ),
                image_mean=_triple(
                    preprocessing["image_mean"],
                    "preprocessing.image_mean",
                ),
                image_std=_triple(preprocessing["image_std"], "preprocessing.image_std"),
            ),
            training=TrainingConfig(
                random_seed=int(training["random_seed"]),
                device=cast(DeviceName, device),
                batch_size=int(training["batch_size"]),
                num_workers=int(training["num_workers"]),
            ),
            output=OutputConfig(
                artifact_root=Path(output["artifact_root"]),
                prediction_root=Path(output["prediction_root"]),
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid PatchCore config: {path}") from exc

    # 모델이나 device 생성 전에 전체 domain invariant를 fail-fast 검증한다.
    config.validate()
    return config
