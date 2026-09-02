"""Strict C5-4A TensorRT INT8 explicit-Q/DQ quantization contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from ml.training.yolo_segmentation import validate_artifact_id
from shared.hashing import is_sha256_digest

INT8_CONFIG_SCHEMA_VERSION = 1
DEFAULT_TENSORRT_INT8_CONFIG = Path("configs/export/yolo_segmentation_tensorrt_int8.yaml")

EXPECTED_ONNX_SHA256 = "f916325bb126d174de9c1fdfc24802eec11c46014f723fbf3ba3b3c1755c1490"
EXPECTED_ONNX_METADATA_SHA256 = "3286861db66cb4c4f886d2fd71f8f13b749b019bd0d57249f54a025d43b11fcd"
EXPECTED_ONNX_EXPORT_CONFIG_SHA256 = (
    "f1c2ef5045fdd89d964b2dc79c501580c9f55c2a1d38f38f13cf4794bafd0e85"
)
EXPECTED_ONNX_EXPORT_COMMIT = "643ed9386a61bd2bf0c041f92a10b809b6d52c3e"
EXPECTED_FROZEN_MANIFEST_SHA256 = "2a26b1bc03a1876f828e12a625c69c76af5e8c5713e3f64be699feffe2e8aa09"
EXPECTED_MODEL_SHA256 = "e3fd10cdd708d31421feacfc5d694cb638e0ea60672e08796391b33aecf67155"
EXPECTED_DATASET_MANIFEST_SHA256 = (
    "1746338c091c18e96a11399c81ea9be0d7350105c4860cfa6a4162144ddb9905"
)


@dataclass(frozen=True)
class Int8SourceIdentity:
    """Immutable source identities inherited from the accepted C4/C5 artifacts."""

    onnx_sha256: str
    onnx_metadata_sha256: str
    onnx_export_config_sha256: str
    onnx_export_commit: str
    frozen_manifest_sha256: str
    model_sha256: str
    dataset_manifest_sha256: str

    # ADD 2026-09-02: INT8가 accepted C4/C5 source bytes에서 이탈하지 않도록 고정한다.
    def validate(self) -> None:
        expected = {
            "onnx_sha256": EXPECTED_ONNX_SHA256,
            "onnx_metadata_sha256": EXPECTED_ONNX_METADATA_SHA256,
            "onnx_export_config_sha256": EXPECTED_ONNX_EXPORT_CONFIG_SHA256,
            "onnx_export_commit": EXPECTED_ONNX_EXPORT_COMMIT,
            "frozen_manifest_sha256": EXPECTED_FROZEN_MANIFEST_SHA256,
            "model_sha256": EXPECTED_MODEL_SHA256,
            "dataset_manifest_sha256": EXPECTED_DATASET_MANIFEST_SHA256,
        }
        for field, expected_value in expected.items():
            observed = getattr(self, field)
            if not isinstance(observed, str) or not observed:
                raise ValueError(f"C5-4A source identity {field} is invalid.")
            if field != "onnx_export_commit" and not is_sha256_digest(observed):
                raise ValueError(f"C5-4A source identity {field} is not SHA-256.")
            if observed != expected_value:
                raise ValueError(f"C5-4A source identity {field} changed from frozen evidence.")


@dataclass(frozen=True)
class Int8QuantizerPolicy:
    """Explicit-Q/DQ PTQ toolchain contract."""

    implementation: str
    package: str
    version: str
    api: str
    quantize_mode: str
    calibration_method: str
    explicit_qdq: bool
    high_precision_dtype: str
    simplify: bool

    # ADD 2026-09-02: Deprecated implicit calibrator 대신 pinned ModelOpt explicit Q/DQ를 강제한다.
    def validate(self) -> None:
        if type(self.explicit_qdq) is not bool or type(self.simplify) is not bool:
            raise TypeError("C5-4A quantizer booleans must be strict booleans.")
        expected = {
            "implementation": "nvidia_modelopt",
            "package": "nvidia-modelopt",
            "version": "0.46.0",
            "api": "modelopt.onnx.quantization.quantize",
            "quantize_mode": "int8",
            "calibration_method": "entropy",
            "high_precision_dtype": "fp16",
        }
        for field, expected_value in expected.items():
            if getattr(self, field) != expected_value:
                raise ValueError(f"C5-4A quantizer field {field} changed without review.")
        if self.explicit_qdq is not True:
            raise ValueError("C5-4A requires explicit Q/DQ quantization.")
        if self.simplify is not False:
            raise ValueError("C5-4A does not simplify the accepted ONNX during quantization.")


@dataclass(frozen=True)
class Int8CalibrationPolicy:
    """Train-only deterministic PTQ calibration boundary."""

    split: str
    sample_count: int
    ordering: str
    batch_size: int
    validation_used: bool
    test_used: bool
    test_split_used: bool
    input_name: str
    decode: str
    resize: str
    color: str
    layout: str
    dtype: str
    scale: float

    # ADD 2026-09-02: 84장 train만 calibration에 사용하고 val/test leakage를 차단한다.
    def validate(self) -> None:
        boolean_values = (self.validation_used, self.test_used, self.test_split_used)
        if any(type(value) is not bool for value in boolean_values):
            raise TypeError("C5-4A calibration leakage flags must be strict booleans.")
        if type(self.sample_count) is not int or type(self.batch_size) is not int:
            raise TypeError("C5-4A calibration counts must be integers.")
        if (
            self.split != "train"
            or self.sample_count != 84
            or self.ordering != "manifest_sample_id_ascending"
            or self.batch_size != 1
        ):
            raise ValueError("C5-4A calibration must use all 84 train samples deterministically.")
        if (
            self.validation_used is not False
            or self.test_used is not False
            or self.test_split_used is not False
        ):
            raise ValueError("C5-4A calibration must not use validation or final-test content.")
        expected_preprocess = {
            "input_name": "images",
            "decode": "opencv_bgr",
            "resize": "ultralytics_letterbox_640",
            "color": "bgr_to_rgb",
            "layout": "nchw",
            "dtype": "float32",
        }
        for field, expected_value in expected_preprocess.items():
            if getattr(self, field) != expected_value:
                raise ValueError(f"C5-4A calibration preprocess field {field} changed.")
        if abs(self.scale - (1.0 / 255.0)) > 1e-15:
            raise ValueError("C5-4A calibration scale must remain 1/255.")


@dataclass(frozen=True)
class Int8BenchmarkPolicy:
    """Same end-to-end benchmark boundary used by C5-3."""

    warmup_iterations: int
    measured_iterations: int
    sample_selector: str
    scope: str

    # ADD 2026-09-02: INT8와 FP16 latency를 동일한 measurement boundary에서 비교한다.
    def validate(self) -> None:
        if type(self.warmup_iterations) is not int or type(self.measured_iterations) is not int:
            raise TypeError("C5-4A benchmark iteration counts must be integers.")
        if self.warmup_iterations != 10 or self.measured_iterations != 50:
            raise ValueError("C5-4A benchmark iteration policy changed from C5-3.")
        if (
            self.sample_selector != "first_validation_sample"
            or self.scope != "ultralytics_end_to_end_single_image"
        ):
            raise ValueError("C5-4A benchmark boundary changed from C5-3.")


@dataclass(frozen=True)
class Int8CharacterizationPolicy:
    """Validation-only INT8 characterization before numeric tolerance approval."""

    split: str
    sample_count: int
    test_used: bool
    test_split_used: bool
    reference_backend: str
    candidate_backend: str
    comparison_baseline: str
    prediction_initial_confidence: float
    diagnostic_confidence: float
    prediction_iou: float
    max_detections: int
    retina_masks: bool
    mask_threshold: float
    mask_resize: str
    association: str
    acceptance_mode: str
    numeric_thresholds: None
    benchmark: Int8BenchmarkPolicy

    # ADD 2026-09-02: INT8 품질 tolerance를 관측 전에 소급 정의하지 않고
    # val-only characterization으로 제한한다.
    def validate(self) -> None:
        if (
            self.split != "val"
            or self.sample_count != 28
            or self.test_used is not False
            or self.test_split_used is not False
        ):
            raise ValueError("C5-4A characterization must remain validation-only on 28 samples.")
        if (
            self.reference_backend != "pytorch_fp32_gpu"
            or self.candidate_backend != "tensorrt_int8"
            or self.comparison_baseline != "accepted_tensorrt_fp16"
        ):
            raise ValueError("C5-4A characterization backend identity is invalid.")
        if (
            self.prediction_initial_confidence != 0.001
            or self.diagnostic_confidence != 0.25
            or self.prediction_iou != 0.7
            or self.max_detections != 300
            or self.retina_masks is not False
            or self.mask_threshold != 0.5
            or self.mask_resize != "opencv_inter_nearest"
            or self.association != "greedy_max_mask_iou_positive_overlap"
        ):
            raise ValueError("C5-4A prediction normalization changed from C4/C5.")
        if (
            self.acceptance_mode != "metrics_only_pending_tensorrt_int8_tolerance_approval"
            or self.numeric_thresholds is not None
        ):
            raise ValueError("C5-4A must characterize INT8 before defining numeric acceptance.")
        self.benchmark.validate()


@dataclass(frozen=True)
class YoloTensorRtInt8Config:
    """C5-4A static INT8 Q/DQ PTQ design contract."""

    schema_version: int
    quantization_id: str
    format: str
    task: str
    precision: str
    batch: int
    imgsz: int
    dynamic: bool
    workspace_gib: int
    device: int
    source: Int8SourceIdentity
    quantizer: Int8QuantizerPolicy
    calibration: Int8CalibrationPolicy
    characterization: Int8CharacterizationPolicy
    config_path: Path

    # ADD 2026-09-02: C5-4A top-level static INT8 contract를 fail-closed 검증한다.
    def validate(self) -> None:
        if (
            type(self.schema_version) is not int
            or type(self.batch) is not int
            or type(self.imgsz) is not int
            or type(self.dynamic) is not bool
            or type(self.workspace_gib) is not int
            or type(self.device) is not int
        ):
            raise TypeError("C5-4A config scalar types are invalid.")
        if self.schema_version != INT8_CONFIG_SCHEMA_VERSION:
            raise ValueError("Unsupported C5-4A config schema version.")
        validate_artifact_id(self.quantization_id)
        if (
            self.format != "onnx_qdq_to_engine"
            or self.task != "segment"
            or self.precision != "int8"
        ):
            raise ValueError("C5-4A supports only explicit-Q/DQ TensorRT INT8 segmentation.")
        if (
            self.batch != 1
            or self.imgsz != 640
            or self.dynamic is not False
            or self.workspace_gib != 4
            or self.device != 0
        ):
            raise ValueError("C5-4A static TensorRT build parameters changed without review.")
        self.source.validate()
        self.quantizer.validate()
        self.calibration.validate()
        self.characterization.validate()


def _mapping(raw: object, *, field: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"C5-4A {field} must be a mapping.")
    return cast(dict[str, Any], raw)


# ADD 2026-09-02: Repository INT8 YAML을 nested typed contract로 복원한다.
def load_yolo_tensorrt_int8_config(path: Path) -> YoloTensorRtInt8Config:
    try:
        raw_obj: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError("Cannot read C5-4A TensorRT INT8 config.") from exc

    raw = _mapping(raw_obj, field="config root")
    try:
        source = Int8SourceIdentity(**_mapping(raw.pop("source"), field="source"))
        quantizer = Int8QuantizerPolicy(**_mapping(raw.pop("quantizer"), field="quantizer"))
        calibration = Int8CalibrationPolicy(**_mapping(raw.pop("calibration"), field="calibration"))
        characterization_raw = _mapping(raw.pop("characterization"), field="characterization")
        benchmark = Int8BenchmarkPolicy(
            **_mapping(
                characterization_raw.pop("benchmark"),
                field="characterization.benchmark",
            )
        )
        characterization = Int8CharacterizationPolicy(
            **characterization_raw,
            benchmark=benchmark,
        )
        config = YoloTensorRtInt8Config(
            **raw,
            source=source,
            quantizer=quantizer,
            calibration=calibration,
            characterization=characterization,
            config_path=path,
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("C5-4A TensorRT INT8 config fields do not match schema.") from exc

    config.validate()
    return config
