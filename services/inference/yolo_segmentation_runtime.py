"""Validated YOLO segmentation runtime isolated from transport and smoke tooling."""

from __future__ import annotations

import math
import os
import tempfile
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Protocol, cast

import numpy as np
import torch
from numpy.typing import NDArray

from ml.training.device import SUPPORTED_DEVICES, resolve_device
from ml.training.yolo_segmentation import (
    METADATA_FILENAME,
    MODEL_FILENAME,
    YoloArtifactMetadata,
    validate_yolo_artifact,
)
from shared.hashing import sha256_file

EXPECTED_CLASSES = {0: "bent", 1: "color", 2: "scratch"}


class YoloPredictionModel(Protocol):
    """Small Ultralytics prediction surface used by the project runtime."""

    names: dict[int, str]
    task: str

    # ADD 2026-08-26: 이미 로드된 model에서 one-image prediction을 수행한다.
    def predict(self, **kwargs: object) -> Sequence[object]:
        """Return one framework result collection."""
        ...


type FrameworkLoader = Callable[[Path, str], tuple[YoloPredictionModel, str]]


@dataclass(frozen=True)
class YoloSegmentationRuntimeConfig:
    """Runtime bundle location and explicit accelerator selection."""

    artifact_dir: Path
    device: str

    # ADD 2026-08-26: Runtime bundle path와 supported device 값을 사전 검증한다.
    def validate(self) -> None:
        if not str(self.artifact_dir):
            raise ValueError("artifact_dir must not be empty.")
        if self.device not in SUPPORTED_DEVICES:
            raise ValueError(f"device must be one of {SUPPORTED_DEVICES}.")


@dataclass(frozen=True)
class YoloSegmentationProvenance:
    """Validated model and supervised-derived dataset identity."""

    dataset_manifest_sha256: str
    dataset_semantic_fingerprint_sha256: str
    artifact_metadata_sha256: str
    model_sha256: str
    framework_version: str


@dataclass(frozen=True)
class YoloSegmentationInstance:
    """One framework-neutral segmentation instance with an original-size CPU mask."""

    class_id: int
    class_name: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]
    mask: NDArray[np.bool_]

    # ADD 2026-08-26: Instance class, confidence, bbox와 binary mask contract를 검증한다.
    def validate(self, *, image_width: int, image_height: int) -> None:
        if EXPECTED_CLASSES.get(self.class_id) != self.class_name:
            raise ValueError("YOLO instance class mapping is invalid.")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("YOLO instance confidence must be finite and in [0, 1].")
        if self.mask.dtype != np.bool_ or self.mask.shape != (image_height, image_width):
            raise ValueError("YOLO instance mask must be an original-size boolean array.")
        if not self.mask.any():
            raise ValueError("YOLO instance mask must contain at least one positive pixel.")
        x1, y1, x2, y2 = self.box_xyxy
        if not all(math.isfinite(value) for value in self.box_xyxy):
            raise ValueError("YOLO instance bbox must contain finite coordinates.")
        if not (0.0 <= x1 <= x2 <= image_width and 0.0 <= y1 <= y2 <= image_height):
            raise ValueError("YOLO instance bbox is outside the original image bounds.")

    @property
    def mask_pixel_count(self) -> int:
        """Return mask area without serializing the full binary array."""
        return int(np.count_nonzero(self.mask))

    @property
    def mask_area_ratio(self) -> float:
        """Return mask area divided by the original image area."""
        return self.mask_pixel_count / int(self.mask.size)


@dataclass(frozen=True)
class YoloSegmentationResult:
    """Normalized result for one source image and one already-loaded model."""

    image_width: int
    image_height: int
    device: str
    inference_ms: float
    instances: tuple[YoloSegmentationInstance, ...]

    # ADD 2026-08-26: Sample dimensions, latency와 모든 normalized instance를 검증한다.
    def validate(self) -> None:
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("YOLO result image dimensions must be positive.")
        if not self.device:
            raise ValueError("YOLO result device must not be blank.")
        if not math.isfinite(self.inference_ms) or self.inference_ms < 0.0:
            raise ValueError("YOLO result inference latency must be finite and non-negative.")
        for instance in self.instances:
            instance.validate(image_width=self.image_width, image_height=self.image_height)


@dataclass
class YoloSegmentationAdapter:
    """One reusable YOLO segmentation model with project-owned result normalization."""

    model: YoloPredictionModel
    metadata: YoloArtifactMetadata
    device: str
    imgsz: int
    provenance: YoloSegmentationProvenance
    _inference_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ADD 2026-08-26: Shared model inference를 accelerator 비의존 CPU result로 정규화한다.
    def predict(
        self,
        image_rgb: NDArray[np.uint8],
        *,
        diagnostic_confidence: float,
    ) -> YoloSegmentationResult:
        """Run Ultralytics preprocessing/inference without retaining raw Results or device masks."""
        validate_diagnostic_confidence(diagnostic_confidence)
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != np.uint8:
            raise ValueError("YOLO runtime input must be an HWC uint8 RGB array.")
        image_height, image_width = image_rgb.shape[:2]

        # Shared model state와 accelerator queue를 보호하고 device timing을 동기화한다.
        with self._inference_lock, torch.inference_mode():
            _synchronize_device(self.device)
            started = perf_counter()
            # Ultralytics ndarray의 BGR contract에 맞춰 project RGB를 경계에서 변환한다.
            image_bgr = np.ascontiguousarray(image_rgb[..., ::-1])
            raw_results = self.model.predict(
                source=image_bgr,
                conf=diagnostic_confidence,
                imgsz=self.imgsz,
                device=self.device,
                retina_masks=True,
                save=False,
                stream=False,
                verbose=False,
            )
            _synchronize_device(self.device)
            inference_ms = (perf_counter() - started) * 1000.0

        results = list(raw_results)
        if len(results) != 1:
            raise ValueError("YOLO runtime must return exactly one result for one input image.")
        return normalize_yolo_segmentation_result(
            results[0],
            image_width=image_width,
            image_height=image_height,
            device=self.device,
            inference_ms=inference_ms,
            classes=self.metadata.classes,
        )


# ADD 2026-08-26: Diagnostic operating point가 probability bounds 안인지 검증한다.
def validate_diagnostic_confidence(value: float) -> None:
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("Diagnostic confidence must be finite and in (0, 1).")


# ADD 2026-08-26: Tensor-like framework value를 detached CPU NumPy array로 변환한다.
def _to_numpy(value: object, *, field_name: str) -> NDArray[np.generic]:
    if value is None:
        raise ValueError(f"Ultralytics result is missing {field_name}.")
    detached = getattr(value, "detach", None)
    materialized = detached() if callable(detached) else value
    cpu = getattr(materialized, "cpu", None)
    materialized = cpu() if callable(cpu) else materialized
    numpy = getattr(materialized, "numpy", None)
    array = np.asarray(numpy() if callable(numpy) else materialized)
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"Ultralytics {field_name} must be numeric.")
    return array


# ADD 2026-08-26: Raw Ultralytics boxes/masks를 bounded original-image result로 변환한다.
def normalize_yolo_segmentation_result(
    raw_result: object,
    *,
    image_width: int,
    image_height: int,
    device: str,
    inference_ms: float,
    classes: dict[int, str],
) -> YoloSegmentationResult:
    """Normalize one segmentation result and reject malformed framework alignment."""
    if classes != EXPECTED_CLASSES:
        raise ValueError("YOLO runtime classes must be exactly bent/color/scratch.")
    raw_shape = getattr(raw_result, "orig_shape", None)
    if raw_shape is not None and tuple(raw_shape) != (image_height, image_width):
        raise ValueError("Ultralytics result original shape does not match the runtime input.")
    boxes = getattr(raw_result, "boxes", None)
    if boxes is None:
        raise ValueError("Ultralytics segmentation result is missing boxes.")
    class_ids = _to_numpy(getattr(boxes, "cls", None), field_name="boxes.cls").reshape(-1)
    confidences = _to_numpy(getattr(boxes, "conf", None), field_name="boxes.conf").reshape(-1)
    coordinates = _to_numpy(getattr(boxes, "xyxy", None), field_name="boxes.xyxy")
    if coordinates.ndim != 2 or coordinates.shape[1:] != (4,):
        raise ValueError("Ultralytics boxes.xyxy must have shape (instances, 4).")
    instance_count = len(class_ids)
    if len(confidences) != instance_count or len(coordinates) != instance_count:
        raise ValueError("Ultralytics box class/confidence/coordinate counts do not match.")

    masks = getattr(raw_result, "masks", None)
    if instance_count == 0:
        if masks is not None:
            mask_values = _to_numpy(getattr(masks, "data", None), field_name="masks.data")
            if mask_values.size:
                raise ValueError("Empty YOLO boxes must not contain segmentation masks.")
        normalized = YoloSegmentationResult(
            image_width=image_width,
            image_height=image_height,
            device=device,
            inference_ms=inference_ms,
            instances=(),
        )
        normalized.validate()
        return normalized
    if masks is None:
        raise ValueError("YOLO segmentation instances are missing masks.")
    mask_values = _to_numpy(getattr(masks, "data", None), field_name="masks.data")
    if mask_values.shape != (instance_count, image_height, image_width):
        raise ValueError("Ultralytics retina masks must match the original image dimensions.")
    if not np.isfinite(mask_values).all():
        raise ValueError("Ultralytics masks must contain finite values.")

    instances: list[YoloSegmentationInstance] = []
    for index in range(instance_count):
        raw_class_id = float(class_ids[index])
        if not raw_class_id.is_integer():
            raise ValueError("Ultralytics class IDs must be integers.")
        class_id = int(raw_class_id)
        if class_id not in classes:
            raise ValueError("Ultralytics result contains an unknown class ID.")
        raw_box = coordinates[index].astype(np.float64, copy=False)
        tolerance = 1e-3
        if (
            not np.isfinite(raw_box).all()
            or raw_box[0] < -tolerance
            or raw_box[1] < -tolerance
            or raw_box[2] > image_width + tolerance
            or raw_box[3] > image_height + tolerance
        ):
            raise ValueError("Ultralytics bbox is outside the original image bounds.")
        clipped_box = np.clip(
            raw_box,
            np.array([0.0, 0.0, 0.0, 0.0]),
            np.array([image_width, image_height, image_width, image_height]),
        )
        mask = np.asarray(mask_values[index] >= 0.5, dtype=np.bool_)
        mask.setflags(write=False)
        instance = YoloSegmentationInstance(
            class_id=class_id,
            class_name=classes[class_id],
            confidence=float(confidences[index]),
            box_xyxy=cast(tuple[float, float, float, float], tuple(clipped_box.tolist())),
            mask=mask,
        )
        instance.validate(image_width=image_width, image_height=image_height)
        instances.append(instance)
    normalized = YoloSegmentationResult(
        image_width=image_width,
        image_height=image_height,
        device=device,
        inference_ms=inference_ms,
        instances=tuple(instances),
    )
    normalized.validate()
    return normalized


# ADD 2026-08-26: CUDA/MPS asynchronous queue를 latency measurement 전에 동기화한다.
def _synchronize_device(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


# ADD 2026-08-26: Pinned Ultralytics runtime을 lazy import하고 checkpoint를 한 번 복원한다.
def _load_ultralytics_model(
    model_path: Path,
    task: str,
) -> tuple[YoloPredictionModel, str]:
    config_root = Path(tempfile.gettempdir()) / "smartfactory-ultralytics"
    config_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_root))
    from ultralytics import YOLO
    from ultralytics import __version__ as ultralytics_version

    return cast(YoloPredictionModel, YOLO(str(model_path), task=task)), ultralytics_version


# ADD 2026-08-26: Runtime lineage/device/framework를 검증하고 model을 한 번 복원한다.
def load_yolo_segmentation_runtime(
    config: YoloSegmentationRuntimeConfig,
    *,
    framework_loader: FrameworkLoader = _load_ultralytics_model,
) -> YoloSegmentationAdapter:
    """Load a CUDA-produced checkpoint without downloading weights or silently changing device."""
    config.validate()
    device = resolve_device(config.device)
    model_artifact_dir = config.artifact_dir / "model"

    # Framework load 전에 project-owned metadata와 checkpoint SHA를 검증한다.
    metadata = validate_yolo_artifact(model_artifact_dir)
    model_path = model_artifact_dir / MODEL_FILENAME
    metadata_path = model_artifact_dir / METADATA_FILENAME
    if metadata.framework != "ultralytics":
        raise ValueError("YOLO runtime artifact framework must be ultralytics.")
    training = metadata.training_config.get("training")
    if not isinstance(training, dict) or not isinstance(training.get("imgsz"), int):
        raise ValueError("YOLO runtime artifact is missing the training image size.")
    imgsz = int(training["imgsz"])
    if imgsz <= 0 or imgsz % 32 != 0:
        raise ValueError("YOLO runtime artifact image size must be a positive multiple of 32.")

    # Validated local checkpoint를 복원하고 installed framework/model class contract를 대조한다.
    model, installed_framework_version = framework_loader(model_path, metadata.task)
    if installed_framework_version != metadata.framework_version:
        raise ValueError(
            "Installed Ultralytics version does not match artifact framework version: "
            f"installed={installed_framework_version}, artifact={metadata.framework_version}."
        )
    if getattr(model, "task", None) != "segment":
        raise ValueError("Loaded YOLO model task must be segment.")
    model_names = {int(key): str(value) for key, value in getattr(model, "names", {}).items()}
    if model_names != metadata.classes:
        raise ValueError("Loaded YOLO model classes do not match artifact metadata.")
    return YoloSegmentationAdapter(
        model=model,
        metadata=metadata,
        device=str(device),
        imgsz=imgsz,
        provenance=YoloSegmentationProvenance(
            dataset_manifest_sha256=metadata.dataset_manifest_sha256,
            dataset_semantic_fingerprint_sha256=(metadata.dataset_semantic_fingerprint_sha256),
            artifact_metadata_sha256=sha256_file(metadata_path),
            model_sha256=sha256_file(model_path),
            framework_version=installed_framework_version,
        ),
    )
