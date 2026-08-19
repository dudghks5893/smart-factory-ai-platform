"""PatchCore adapter and portable artifact contract."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import anomalib
import torch
import torchvision  # type: ignore[import-untyped]
from anomalib.data import InferenceBatch
from anomalib.models.image.patchcore.torch_model import PatchcoreModel
from torch import Tensor

from ml.training.batches import require_batch_tensor
from ml.training.config import PatchCoreBaselineConfig, PatchCoreModelConfig
from ml.training.preprocessing import PatchCorePreprocessingConfig, PatchCorePreprocessor
from ml.training.reproducibility import seed_training
from shared.hashing import sha256_file

ARTIFACT_SCHEMA_VERSION = 1
MODEL_FILENAME = "model.pt"
METADATA_FILENAME = "metadata.json"


@dataclass(frozen=True)
class PatchCoreArtifactMetadata:
    """Metadata required to reproduce and audit a PatchCore artifact."""

    schema_version: int
    model_name: str
    implementation: str
    backbone: str
    layers: tuple[str, ...]
    num_neighbors: int
    coreset_sampling_ratio: float
    pretrained_used_during_training: bool
    preprocessing: PatchCorePreprocessingConfig
    random_seed: int
    category: str
    train_sample_count: int
    manifest_sha256: str
    anomalib_version: str
    torch_version: str
    torchvision_version: str
    python_version: str
    created_at: str

    # ADD 2026-08-19: Convert metadata to a stable JSON-compatible mapping.
    def to_json_dict(self) -> dict[str, Any]:
        """Convert metadata to a stable JSON-compatible mapping."""
        raw = asdict(self)
        raw["layers"] = list(self.layers)
        preprocessing = cast(dict[str, object], raw["preprocessing"])
        preprocessing["resize_size"] = list(self.preprocessing.resize_size)
        preprocessing["center_crop_size"] = list(self.preprocessing.center_crop_size)
        preprocessing["image_mean"] = list(self.preprocessing.image_mean)
        preprocessing["image_std"] = list(self.preprocessing.image_std)
        return raw

    # ADD 2026-08-19: Validate and construct artifact metadata loaded from JSON.
    @classmethod
    def from_json_dict(cls, raw: object) -> PatchCoreArtifactMetadata:
        """Validate and construct artifact metadata loaded from JSON."""
        if not isinstance(raw, dict):
            raise ValueError("Artifact metadata root must be a mapping.")

        # JSON field를 typed metadata로 변환하고 누락되거나 잘못된 값을 즉시 거부한다.
        try:
            preprocessing_raw = raw["preprocessing"]
            if not isinstance(preprocessing_raw, dict):
                raise TypeError("preprocessing must be a mapping")

            metadata = cls(
                schema_version=int(raw["schema_version"]),
                model_name=str(raw["model_name"]),
                implementation=str(raw["implementation"]),
                backbone=str(raw["backbone"]),
                layers=_metadata_string_tuple(raw["layers"], "layers"),
                num_neighbors=int(raw["num_neighbors"]),
                coreset_sampling_ratio=float(raw["coreset_sampling_ratio"]),
                pretrained_used_during_training=_metadata_boolean(
                    raw["pretrained_used_during_training"],
                    "pretrained_used_during_training",
                ),
                preprocessing=PatchCorePreprocessingConfig(
                    resize_size=_metadata_pair(preprocessing_raw["resize_size"], "resize_size"),
                    center_crop_size=_metadata_pair(
                        preprocessing_raw["center_crop_size"],
                        "center_crop_size",
                    ),
                    image_mean=_metadata_triple(preprocessing_raw["image_mean"], "image_mean"),
                    image_std=_metadata_triple(preprocessing_raw["image_std"], "image_std"),
                ),
                random_seed=int(raw["random_seed"]),
                category=str(raw["category"]),
                train_sample_count=int(raw["train_sample_count"]),
                manifest_sha256=str(raw["manifest_sha256"]),
                anomalib_version=str(raw["anomalib_version"]),
                torch_version=str(raw["torch_version"]),
                torchvision_version=str(raw["torchvision_version"]),
                python_version=str(raw["python_version"]),
                created_at=str(raw["created_at"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Artifact metadata is missing or contains invalid fields.") from exc

        metadata.validate()
        return metadata

    # ADD 2026-08-19: Validate compatibility-critical artifact fields.
    def validate(self) -> None:
        """Validate compatibility-critical artifact fields."""
        if self.schema_version != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported artifact schema_version: {self.schema_version}; "
                f"expected {ARTIFACT_SCHEMA_VERSION}."
            )
        PatchCoreModelConfig(
            name=self.model_name,
            implementation=self.implementation,
            backbone=self.backbone,
            layers=self.layers,
            pretrained=self.pretrained_used_during_training,
            coreset_sampling_ratio=self.coreset_sampling_ratio,
            num_neighbors=self.num_neighbors,
        ).validate()
        self.preprocessing.validate()
        if self.random_seed < 0:
            raise ValueError("Artifact random_seed must be non-negative.")
        if not self.category:
            raise ValueError("Artifact category must not be empty.")
        if self.train_sample_count <= 0:
            raise ValueError("Artifact train_sample_count must be positive.")
        if len(self.manifest_sha256) != 64:
            raise ValueError("Artifact manifest_sha256 must be a SHA-256 hex digest.")
        try:
            int(self.manifest_sha256, 16)
        except ValueError as exc:
            raise ValueError("Artifact manifest_sha256 must be a SHA-256 hex digest.") from exc


@dataclass(frozen=True)
class PatchCorePrediction:
    """Raw PatchCore outputs before thresholding or score calibration."""

    scores: Tensor
    anomaly_maps: Tensor


class PatchCoreAdapter:
    """Integrate the low-level Anomalib PatchcoreModel with manifest DataLoaders."""

    # ADD 2026-08-19: PatchcoreModel을 지정 device와 pretrained 정책으로 초기화한다.
    def __init__(
        self,
        *,
        model_config: PatchCoreModelConfig,
        preprocessing_config: PatchCorePreprocessingConfig,
        device: torch.device,
        pre_trained: bool,
    ) -> None:
        # 외부 weight 접근 전에 model/preprocessing configuration을 검증한다.
        model_config.validate()
        preprocessing_config.validate()
        self.model_config = model_config
        self.preprocessing_config = preprocessing_config
        self.device = device

        # Training과 artifact restore가 공유하는 low-level PatchcoreModel을 생성한다.
        self.model = PatchcoreModel(
            backbone=model_config.backbone,
            layers=model_config.layers,
            pre_trained=pre_trained,
            num_neighbors=model_config.num_neighbors,
        ).to(device)

    # ADD 2026-08-19: Seed all required RNGs before constructing the training model.
    @classmethod
    def for_training(
        cls,
        config: PatchCoreBaselineConfig,
        device: torch.device,
    ) -> PatchCoreAdapter:
        """Seed all required RNGs before constructing the training model."""
        # KCenterGreedy와 random projection 재현성을 위해 모델 생성 전에 RNG를 고정한다.
        config.validate()
        seed_training(config.training.random_seed)
        return cls(
            model_config=config.model,
            preprocessing_config=config.preprocessing,
            device=device,
            pre_trained=config.model.pretrained,
        )

    # ADD 2026-08-19: Load an artifact without downloading pretrained backbone weights.
    # MODIFY 2026-08-19: metadata 중복 로드 → 사전 검증 metadata를 재사용하도록 확장했다.
    @classmethod
    def load_artifact(
        cls,
        artifact_dir: Path,
        device: torch.device,
        metadata: PatchCoreArtifactMetadata | None = None,
    ) -> tuple[PatchCoreAdapter, PatchCoreArtifactMetadata]:
        """Load an artifact without downloading pretrained backbone weights."""
        # Artifact metadata를 검증하고 동일한 구조의 모델을 외부 다운로드 없이 생성한다.
        metadata = metadata or read_artifact_metadata(artifact_dir)
        model_config = PatchCoreModelConfig(
            name=metadata.model_name,
            implementation=metadata.implementation,
            backbone=metadata.backbone,
            layers=metadata.layers,
            pretrained=metadata.pretrained_used_during_training,
            coreset_sampling_ratio=metadata.coreset_sampling_ratio,
            num_neighbors=metadata.num_neighbors,
        )
        adapter = cls(
            model_config=model_config,
            preprocessing_config=metadata.preprocessing,
            device=device,
            pre_trained=False,
        )

        # Tensor state_dict만 안전 모드로 로드하고 architecture와 정확히 일치시킨다.
        model_path = artifact_dir / MODEL_FILENAME
        if not model_path.is_file():
            raise FileNotFoundError(f"PatchCore state_dict not found: {model_path}")

        loaded = torch.load(model_path, map_location=device, weights_only=True)
        if not isinstance(loaded, dict) or not all(
            isinstance(key, str) and isinstance(value, Tensor) for key, value in loaded.items()
        ):
            raise ValueError("PatchCore model.pt must contain only a tensor state_dict.")

        state_dict = cast(dict[str, Tensor], loaded)
        adapter.model.load_state_dict(state_dict, strict=True)
        adapter.model.to(device)
        adapter.model.eval()
        if adapter.model.memory_bank.numel() == 0:
            raise ValueError("Loaded PatchCore artifact has an empty memory bank.")
        return adapter, metadata

    # ADD 2026-08-19: Collect normal embeddings and construct the PatchCore memory bank.
    # MODIFY 2026-08-19: 중복 batch tensor 검사 → 공통 batch validation helper를 사용한다.
    def fit(
        self,
        train_loader: object,
        preprocessor: PatchCorePreprocessor,
    ) -> int:
        """Collect normal embeddings and construct the PatchCore memory bank."""
        if not hasattr(train_loader, "__iter__"):
            raise TypeError("train_loader must be iterable.")

        # Optimizer 없이 normal image embedding을 순서대로 수집한다.
        self.model.train()
        train_sample_count = 0

        for batch in train_loader:
            if not isinstance(batch, dict):
                raise TypeError("Each training batch must be a mapping.")
            images = require_batch_tensor(batch, "image")
            labels = require_batch_tensor(batch, "label")
            if torch.any(labels != 0):
                raise ValueError("PatchCore training batches must contain only normal labels (0).")

            # PatchCore geometry/normalization 후 frozen backbone feature를 추출한다.
            transformed_images, _ = preprocessor(images)
            transformed_images = transformed_images.to(self.device)
            embeddings = self.model(transformed_images)
            if not isinstance(embeddings, Tensor):
                raise TypeError("PatchCore training forward must return embedding tensors.")
            train_sample_count += images.shape[0]

        if train_sample_count == 0:
            raise ValueError("PatchCore training loader is empty.")

        # 수집된 train embedding에서 배포용 coreset memory bank를 생성한다.
        self.model.subsample_embedding(sampling_ratio=self.model_config.coreset_sampling_ratio)
        if self.model.memory_bank.numel() == 0:
            raise RuntimeError("PatchCore coreset sampling produced an empty memory bank.")
        self.model.eval()
        return train_sample_count

    # ADD 2026-08-19: Return raw image scores and anomaly maps without thresholding.
    # MODIFY 2026-08-19: no_grad inference → inference_mode로 autograd overhead를 줄였다.
    def predict(
        self,
        images: Tensor,
        preprocessor: PatchCorePreprocessor,
    ) -> PatchCorePrediction:
        """Return raw image scores and anomaly maps without thresholding."""
        if self.model.memory_bank.numel() == 0:
            raise RuntimeError("PatchCore prediction requires a non-empty memory bank.")

        # Artifact에 기록된 geometry와 normalization으로 inference 입력을 준비한다.
        transformed_images, _ = preprocessor(images)
        self.model.eval()

        # Threshold 적용 전 raw anomaly score와 map을 gradient 없이 추론한다.
        with torch.inference_mode():
            prediction = self.model(transformed_images.to(self.device))

        if not isinstance(prediction, InferenceBatch):
            raise TypeError("PatchCore inference forward must return InferenceBatch.")
        if prediction.pred_score is None or prediction.anomaly_map is None:
            raise ValueError("PatchCore inference output is missing score or anomaly map.")

        # Accelerator memory를 batch마다 해제할 수 있도록 필요한 output만 CPU로 이동한다.
        return PatchCorePrediction(
            scores=prediction.pred_score.detach().cpu(),
            anomaly_maps=prediction.anomaly_map.detach().cpu(),
        )

    # ADD 2026-08-19: Save a portable CPU state_dict and inspectable JSON metadata.
    # MODIFY 2026-08-19: PatchCore 내부 hashing → shared streaming hash helper를 사용한다.
    def save_artifact(
        self,
        *,
        artifact_dir: Path,
        category: str,
        train_sample_count: int,
        manifest_path: Path,
        random_seed: int,
    ) -> PatchCoreArtifactMetadata:
        """Save a portable CPU state_dict and inspectable JSON metadata."""
        if self.model.memory_bank.numel() == 0:
            raise RuntimeError("Cannot save a PatchCore artifact with an empty memory bank.")
        if train_sample_count <= 0:
            raise ValueError("train_sample_count must be positive.")
        if artifact_dir.exists():
            raise FileExistsError(f"Artifact directory already exists: {artifact_dir}")

        # Model/Data/config provenance를 포함한 inspectable metadata를 구성한다.
        metadata = PatchCoreArtifactMetadata(
            schema_version=ARTIFACT_SCHEMA_VERSION,
            model_name=self.model_config.name,
            implementation=self.model_config.implementation,
            backbone=self.model_config.backbone,
            layers=self.model_config.layers,
            num_neighbors=self.model_config.num_neighbors,
            coreset_sampling_ratio=self.model_config.coreset_sampling_ratio,
            pretrained_used_during_training=self.model_config.pretrained,
            preprocessing=self.preprocessing_config,
            random_seed=random_seed,
            category=category,
            train_sample_count=train_sample_count,
            manifest_sha256=sha256_file(manifest_path),
            anomalib_version=anomalib.__version__,
            torch_version=torch.__version__,
            torchvision_version=torchvision.__version__,
            python_version=platform.python_version(),
            created_at=datetime.now(UTC).isoformat(),
        )
        metadata.validate()

        # Python model object 없이 CPU state_dict와 JSON metadata만 저장한다.
        artifact_dir.mkdir(parents=True, exist_ok=False)
        cpu_state_dict = {
            key: tensor.detach().cpu() for key, tensor in self.model.state_dict().items()
        }
        torch.save(cpu_state_dict, artifact_dir / MODEL_FILENAME)
        (artifact_dir / METADATA_FILENAME).write_text(
            json.dumps(metadata.to_json_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return metadata


# ADD 2026-08-19: Read and validate metadata.json from a PatchCore artifact.
# MODIFY 2026-08-19: metadata 파일 검사 → 전체 artifact layout을 먼저 검증한다.
def read_artifact_metadata(artifact_dir: Path) -> PatchCoreArtifactMetadata:
    """Read and validate metadata.json from a PatchCore artifact."""
    # JSON parsing 전에 artifact directory와 필수 파일 구성을 확인한다.
    validate_artifact_layout(artifact_dir)
    metadata_path = artifact_dir / METADATA_FILENAME
    try:
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read PatchCore metadata: {metadata_path}") from exc
    return PatchCoreArtifactMetadata.from_json_dict(raw)


# ADD 2026-08-19: Fail fast when an artifact directory or one of its required files is missing.
def validate_artifact_layout(artifact_dir: Path) -> None:
    """Fail fast when an artifact directory or one of its required files is missing."""
    if not artifact_dir.is_dir():
        raise FileNotFoundError(f"PatchCore artifact directory not found: {artifact_dir}")

    for filename in (MODEL_FILENAME, METADATA_FILENAME):
        artifact_path = artifact_dir / filename
        if not artifact_path.is_file():
            raise FileNotFoundError(f"PatchCore artifact file not found: {artifact_path}")


# ADD 2026-08-19: Artifact metadata 값을 두 정수 tuple로 변환한다.
def _metadata_pair(value: object, field: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"Artifact preprocessing.{field} must contain two integers.")
    return int(value[0]), int(value[1])


# ADD 2026-08-19: Artifact metadata 값을 세 실수 tuple로 변환한다.
def _metadata_triple(value: object, field: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"Artifact preprocessing.{field} must contain three numbers.")
    return float(value[0]), float(value[1]), float(value[2])


# ADD 2026-08-19: Artifact metadata 값이 boolean인지 검증한다.
def _metadata_boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"Artifact {field} must be a boolean.")
    return value


# ADD 2026-08-19: Artifact metadata 값을 non-empty string tuple로 검증한다.
def _metadata_string_tuple(value: object, field: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise ValueError(f"Artifact {field} must contain non-empty strings.")
    return tuple(value)
