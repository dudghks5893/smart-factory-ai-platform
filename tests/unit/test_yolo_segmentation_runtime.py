"""Unit contracts for the YOLO segmentation runtime and local smoke tooling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from PIL import Image

from ml.training.yolo_segmentation import (
    ARTIFACT_SCHEMA_VERSION,
    YoloArtifactMetadata,
    write_yolo_artifact,
)
from pipelines.smoke_yolo_segmentation_runtime import (
    _parse_args,
    compare_device_results,
    resolve_default_smoke_images,
    smoke_yolo_segmentation_runtime,
)
from services.inference.yolo_segmentation_runtime import (
    FrameworkLoader,
    YoloPredictionModel,
    YoloSegmentationAdapter,
    YoloSegmentationProvenance,
    YoloSegmentationResult,
    YoloSegmentationRuntimeConfig,
    load_yolo_segmentation_runtime,
    normalize_yolo_segmentation_result,
    validate_diagnostic_confidence,
)

CLASSES = {0: "bent", 1: "color", 2: "scratch"}


class FakeBoxes:
    """Array-backed box fields matching the Ultralytics Results surface."""

    # ADD 2026-08-26: Fake class/confidence/bbox arrays를 보관한다.
    def __init__(
        self,
        *,
        class_ids: np.ndarray,
        confidences: np.ndarray,
        coordinates: np.ndarray,
    ) -> None:
        self.cls = class_ids
        self.conf = confidences
        self.xyxy = coordinates


class FakeMasks:
    """Array-backed mask field matching the Ultralytics Results surface."""

    # ADD 2026-08-26: Fake predicted mask array를 보관한다.
    def __init__(self, data: np.ndarray) -> None:
        self.data = data


class FakeResult:
    """Minimal boxes, masks and original-shape prediction result."""

    # ADD 2026-08-26: Normalization test용 raw result field를 구성한다.
    def __init__(
        self,
        *,
        boxes: FakeBoxes,
        masks: FakeMasks | None,
        orig_shape: tuple[int, int] = (8, 8),
    ) -> None:
        self.boxes = boxes
        self.masks = masks
        self.orig_shape = orig_shape


class FakeModel:
    """Reusable fake model that records each inference call."""

    # ADD 2026-08-26: Loaded task/classes와 deterministic result를 보관한다.
    def __init__(
        self,
        result: FakeResult,
        *,
        names: dict[int, str] | None = None,
        task: str = "segment",
    ) -> None:
        self.names = names or CLASSES
        self.task = task
        self.result = result
        self.predict_calls: list[dict[str, object]] = []

    # ADD 2026-08-26: Predict arguments를 기록하고 one-result sequence를 반환한다.
    def predict(self, **kwargs: object) -> list[object]:
        self.predict_calls.append(kwargs)
        return [self.result]


# ADD 2026-08-26: Valid C2-2 artifact metadata fixture를 생성한다.
def _metadata() -> YoloArtifactMetadata:
    return YoloArtifactMetadata(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        model_name="yolo11n-seg.pt",
        task="segment",
        architecture="yolo11n-seg",
        category="metal_nut",
        classes=CLASSES,
        seed=42,
        dataset_manifest_sha256="a" * 64,
        dataset_semantic_fingerprint_sha256="b" * 64,
        training_config={"training": {"imgsz": 640}},
        created_at="2026-08-25T00:00:00+00:00",
        framework="ultralytics",
        framework_version="8.4.128",
        torch_version="2.13.0+cu130",
        device="cuda:0",
        best_epoch=60,
        source_checkpoint="weights/best.pt",
        checkpoint_sha256="0" * 64,
    )


# ADD 2026-08-26: Empty normalized result fixture를 생성한다.
def _empty_result() -> FakeResult:
    return FakeResult(
        boxes=FakeBoxes(
            class_ids=np.array([], dtype=np.float32),
            confidences=np.array([], dtype=np.float32),
            coordinates=np.empty((0, 4), dtype=np.float32),
        ),
        masks=None,
    )


# ADD 2026-08-26: One or two valid instance를 가진 raw result fixture를 생성한다.
def _prediction_result(*, multiple: bool = False) -> FakeResult:
    count = 2 if multiple else 1
    masks = np.zeros((count, 8, 8), dtype=np.float32)
    masks[0, 1:5, 1:5] = 1.0
    if multiple:
        masks[1, 3:7, 3:7] = 1.0
    return FakeResult(
        boxes=FakeBoxes(
            class_ids=np.array([0, 2][:count], dtype=np.float32),
            confidences=np.array([0.9, 0.7][:count], dtype=np.float32),
            coordinates=np.array([[1, 1, 5, 5], [3, 3, 7, 7]][:count], dtype=np.float32),
        ),
        masks=FakeMasks(masks),
    )


# ADD 2026-08-26: Runtime bundle layout에 valid checkpoint/metadata를 기록한다.
def _artifact_bundle(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source_checkpoint = tmp_path / "best.pt"
    source_checkpoint.write_bytes(b"cuda-trained-portable-yolo-checkpoint")
    artifact_root = tmp_path / "runtime-artifact"
    write_yolo_artifact(
        source_checkpoint=source_checkpoint,
        artifact_dir=artifact_root / "model",
        metadata=_metadata(),
    )
    return artifact_root


# ADD 2026-08-26: Metadata/SHA/framework/model class를 검증한 runtime load를 확인한다.
def test_runtime_load_validates_artifact_and_model_contract(tmp_path: Path) -> None:
    artifact_root = _artifact_bundle(tmp_path)
    model = FakeModel(_empty_result())

    # Injected loader로 network/model execution 없이 restore contract만 검증한다.
    def framework_loader(model_path: Path, task: str) -> tuple[YoloPredictionModel, str]:
        assert model_path == artifact_root / "model" / "model.pt"
        assert task == "segment"
        return model, "8.4.128"

    runtime = load_yolo_segmentation_runtime(
        YoloSegmentationRuntimeConfig(artifact_dir=artifact_root, device="cpu"),
        framework_loader=framework_loader,
    )
    assert runtime.device == "cpu"
    assert runtime.imgsz == 640
    assert runtime.metadata.classes == CLASSES
    assert runtime.provenance.model_sha256 == runtime.metadata.checkpoint_sha256

    (artifact_root / "model" / "model.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checkpoint SHA"):
        load_yolo_segmentation_runtime(
            YoloSegmentationRuntimeConfig(artifact_dir=artifact_root, device="cpu"),
            framework_loader=framework_loader,
        )


# ADD 2026-08-26: Invalid metadata task와 loaded class mapping을 restore 전에/중에 거부한다.
def test_runtime_load_rejects_invalid_task_and_loaded_classes(tmp_path: Path) -> None:
    artifact_root = _artifact_bundle(tmp_path)
    metadata_path = artifact_root / "model" / "metadata.json"
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    raw["task"] = "detect"
    metadata_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="task/classes"):
        load_yolo_segmentation_runtime(
            YoloSegmentationRuntimeConfig(artifact_dir=artifact_root, device="cpu"),
            framework_loader=cast(
                FrameworkLoader, lambda _path, _task: (FakeModel(_empty_result()), "8.4.128")
            ),
        )

    artifact_root = _artifact_bundle(tmp_path / "second")
    wrong_classes = {0: "bent", 1: "color", 2: "wrong"}
    with pytest.raises(ValueError, match="classes"):
        load_yolo_segmentation_runtime(
            YoloSegmentationRuntimeConfig(artifact_dir=artifact_root, device="cpu"),
            framework_loader=cast(
                FrameworkLoader,
                lambda _path, _task: (
                    FakeModel(_empty_result(), names=wrong_classes),
                    "8.4.128",
                ),
            ),
        )


# ADD 2026-08-26: Unsupported device와 invalid diagnostic confidence를 fail-fast 검증한다.
def test_runtime_device_and_diagnostic_confidence_validation() -> None:
    with pytest.raises(ValueError, match="device must be"):
        YoloSegmentationRuntimeConfig(Path("artifact"), "metal").validate()
    for value in (0.0, 1.0, float("nan")):
        with pytest.raises(ValueError, match="Diagnostic confidence"):
            validate_diagnostic_confidence(value)


# ADD 2026-08-26: Empty prediction을 valid zero-instance result로 정규화한다.
def test_normalize_empty_prediction() -> None:
    normalized = normalize_yolo_segmentation_result(
        _empty_result(),
        image_width=8,
        image_height=8,
        device="cpu",
        inference_ms=2.5,
        classes=CLASSES,
    )
    assert normalized.instances == ()
    assert normalized.image_width == 8


# ADD 2026-08-26: Multi-instance class/bbox/mask를 CPU-owned summary로 정규화한다.
def test_normalize_multi_instance_prediction() -> None:
    normalized = normalize_yolo_segmentation_result(
        _prediction_result(multiple=True),
        image_width=8,
        image_height=8,
        device="mps",
        inference_ms=3.5,
        classes=CLASSES,
    )
    assert [instance.class_name for instance in normalized.instances] == ["bent", "scratch"]
    assert normalized.instances[0].mask.shape == (8, 8)
    assert normalized.instances[0].mask.dtype == np.bool_
    assert normalized.instances[0].mask.flags.writeable is False
    assert normalized.instances[0].mask_pixel_count == 16


# ADD 2026-08-26: Out-of-bounds bbox, wrong mask shape와 missing masks를 거부한다.
@pytest.mark.parametrize("failure", ["bbox", "shape", "missing"])
def test_normalize_rejects_malformed_ultralytics_result(failure: str) -> None:
    result = _prediction_result()
    if failure == "bbox":
        result.boxes.xyxy = np.array([[-1, 1, 5, 5]], dtype=np.float32)
        match = "bbox"
    elif failure == "shape":
        assert result.masks is not None
        result.masks.data = np.ones((1, 4, 4), dtype=np.float32)
        match = "retina masks"
    else:
        result.masks = None
        match = "missing masks"
    with pytest.raises(ValueError, match=match):
        normalize_yolo_segmentation_result(
            result,
            image_width=8,
            image_height=8,
            device="cpu",
            inference_ms=1.0,
            classes=CLASSES,
        )


# ADD 2026-08-26: Adapter가 same model을 재사용하고 Ultralytics preprocessing arguments를 전달한다.
def test_adapter_reuses_model_and_passes_runtime_contract() -> None:
    model = FakeModel(_prediction_result())
    adapter = YoloSegmentationAdapter(
        model=model,
        metadata=_metadata(),
        device="cpu",
        imgsz=640,
        provenance=YoloSegmentationProvenance("a" * 64, "b" * 64, "c" * 64, "d" * 64, "8.4.128"),
    )
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[0, 0] = [10, 20, 30]
    first = adapter.predict(image, diagnostic_confidence=0.25)
    second = adapter.predict(image, diagnostic_confidence=0.25)
    assert len(model.predict_calls) == 2
    assert first.instances[0].class_name == second.instances[0].class_name == "bent"
    assert model.predict_calls[0]["imgsz"] == 640
    assert model.predict_calls[0]["retina_masks"] is True
    source = cast(np.ndarray, model.predict_calls[0]["source"])
    assert source[0, 0].tolist() == [30, 20, 10]


# ADD 2026-08-26: Device comparison이 confidence/bbox/mask delta를 exact equality 없이 계산한다.
def test_compare_device_results_reports_semantic_and_numeric_deltas() -> None:
    primary = normalize_yolo_segmentation_result(
        _prediction_result(),
        image_width=8,
        image_height=8,
        device="mps",
        inference_ms=1.0,
        classes=CLASSES,
    )
    shifted = _prediction_result()
    shifted.boxes.conf = np.array([0.89], dtype=np.float32)
    shifted.boxes.xyxy = np.array([[1.25, 1, 5, 5]], dtype=np.float32)
    reference = normalize_yolo_segmentation_result(
        shifted,
        image_width=8,
        image_height=8,
        device="cpu",
        inference_ms=2.0,
        classes=CLASSES,
    )
    comparison = compare_device_results(
        source=Path("bent/000.png"),
        primary=primary,
        reference=reference,
    )
    assert comparison.class_sets_equal is True
    assert comparison.instance_counts_equal is True
    assert comparison.max_confidence_abs_delta == pytest.approx(0.01)
    assert comparison.max_bbox_abs_delta_pixels == 0.25
    assert comparison.min_mask_iou == 1.0


# ADD 2026-08-26: Preset image와 CLI confidence/device argument validation을 확인한다.
def test_smoke_preset_and_cli_argument_validation(tmp_path: Path) -> None:
    for defect_type in ("good", "bent", "color", "scratch"):
        path = tmp_path / "test" / defect_type / "000.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8)).save(path)
    assert [path.parent.name for path in resolve_default_smoke_images(tmp_path)] == [
        "good",
        "bent",
        "color",
        "scratch",
    ]
    with pytest.raises(SystemExit):
        _parse_args(["--artifact-dir", "artifact", "--confidence", "1.0"])
    with pytest.raises(SystemExit):
        _parse_args(["--artifact-dir", "artifact", "--device", "metal"])


class FakeSmokeRuntime:
    """Lightweight runtime fake used to verify smoke orchestration and output files."""

    # ADD 2026-08-26: Device별 provenance와 prediction call counter를 구성한다.
    def __init__(self, device: str) -> None:
        self.device = device
        self.provenance = YoloSegmentationProvenance(
            "a" * 64,
            "b" * 64,
            "c" * 64,
            "d" * 64,
            "8.4.128",
        )
        self.predict_count = 0

    # ADD 2026-08-26: Smoke image마다 empty valid normalized result를 반환한다.
    def predict(
        self,
        image_rgb: np.ndarray,
        *,
        diagnostic_confidence: float,
    ) -> YoloSegmentationResult:
        self.predict_count += 1
        assert diagnostic_confidence == 0.25
        return YoloSegmentationResult(
            image_width=image_rgb.shape[1],
            image_height=image_rgb.shape[0],
            device=self.device,
            inference_ms=float(self.predict_count),
            instances=(),
        )


# ADD 2026-08-26: Multi-image smoke가 model을 한 번 load하고 summary/visualization을 기록한다.
def test_smoke_orchestration_reuses_loaded_runtime(tmp_path: Path) -> None:
    image_paths: list[Path] = []
    for defect_type in ("good", "bent"):
        path = tmp_path / "test" / defect_type / "000.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8)).save(path)
        image_paths.append(path)
    loaded: list[FakeSmokeRuntime] = []

    # Injected runtime은 accelerator/artifact 없이 orchestration call count만 관찰한다.
    def runtime_loader(config: YoloSegmentationRuntimeConfig) -> YoloSegmentationAdapter:
        runtime = FakeSmokeRuntime(config.device)
        loaded.append(runtime)
        return cast(YoloSegmentationAdapter, runtime)

    suite = smoke_yolo_segmentation_runtime(
        artifact_dir=tmp_path / "artifact",
        image_paths=image_paths,
        requested_device="cpu",
        diagnostic_confidence=0.25,
        output_dir=tmp_path / "outputs",
        runtime_loader=runtime_loader,
    )
    assert len(loaded) == 1
    assert loaded[0].predict_count == 2
    assert suite.images[0].latency_phase == "cold_first_inference"
    assert suite.images[1].latency_phase == "subsequent_inference"
    assert (tmp_path / "outputs" / "good_cpu.png").is_file()
    assert Path(suite.summary_path).is_file()
