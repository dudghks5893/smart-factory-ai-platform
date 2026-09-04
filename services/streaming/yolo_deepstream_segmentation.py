"""C6-5D DeepStream YOLO11 segmentation decoder foundation contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from services.streaming.yolo_deepstream_tensorrt_int8_inference import (
    EXPECTED_PLAN_BYTES,
    EXPECTED_PLAN_PATH,
    EXPECTED_PLAN_SHA256,
)
from shared.hashing import is_sha256_digest

DEFAULT_DEEPSTREAM_SEGMENTATION_CONFIG = Path("configs/streaming/yolo_deepstream_segmentation.json")
DEFAULT_DEEPSTREAM_SEGMENTATION_LABELS = Path(
    "configs/streaming/yolo_deepstream_segmentation_labels.txt"
)
DEFAULT_DEEPSTREAM_SEGMENTATION_PARSER = Path("services/streaming/deepstream/yolo11_seg_parser.cpp")

EXPECTED_DECODER_ID = "c6_5d_yolo11n_seg_deepstream_decoder_v1"
EXPECTED_C6_5C_CLOSURE_COMMIT = "44f7802e89cf138f7db1160125b98180dadaef8e"
EXPECTED_CLASSES = {0: "bent", 1: "color", 2: "scratch"}
EXPECTED_PARSER_SYMBOL = "NvDsInferParseYolo11Seg"
EXPECTED_PARSER_LIBRARY = "/work/libc6_5d_yolo11_seg_parser.so"
EXPECTED_LABEL_FILE = "/work/c6_5d_labels.txt"


@dataclass(frozen=True)
class SegmentationEngineIdentity:
    """Immutable C6-5C L4 TensorRT plan consumed by C6-5D."""

    relative_path: Path
    sha256: str
    size_bytes: int
    immutable: bool

    # ADD 2026-09-05: C6-5D가 sealed C6-5C L4 plan만 사용하게 한다.
    def validate(self) -> None:
        if (
            self.relative_path != EXPECTED_PLAN_PATH
            or self.sha256 != EXPECTED_PLAN_SHA256
            or self.size_bytes != EXPECTED_PLAN_BYTES
            or self.immutable is not True
        ):
            raise ValueError("C6-5D engine identity changed from sealed C6-5C plan.")
        if not is_sha256_digest(self.sha256):
            raise ValueError("C6-5D engine SHA-256 is invalid.")


@dataclass(frozen=True)
class TensorOutput0Contract:
    """YOLO11-seg channel-major detection tensor contract."""

    name: str
    channels: int
    anchors: int
    layout: str

    # ADD 2026-09-05: Characterized output0 shape/layout을 39x8400으로 고정한다.
    def validate(self) -> None:
        if (
            self.name != "output0"
            or self.channels != 39
            or self.anchors != 8400
            or self.layout != "channel_major"
        ):
            raise ValueError("C6-5D output0 contract changed.")


@dataclass(frozen=True)
class TensorOutput1Contract:
    """YOLO11-seg CHW prototype tensor contract."""

    name: str
    channels: int
    height: int
    width: int
    layout: str

    # ADD 2026-09-05: Characterized output1 shape/layout을 32x160x160으로 고정한다.
    def validate(self) -> None:
        if (
            self.name != "output1"
            or self.channels != 32
            or self.height != 160
            or self.width != 160
            or self.layout != "chw"
        ):
            raise ValueError("C6-5D output1 contract changed.")


@dataclass(frozen=True)
class SegmentationModelContract:
    """Frozen network, class mapping, and raw tensor semantics."""

    network_width: int
    network_height: int
    classes: dict[int, str]
    output0: TensorOutput0Contract
    output1: TensorOutput1Contract

    # ADD 2026-09-05: Decoder model geometry/classes/raw outputs를 exact contract로 검증한다.
    def validate(self) -> None:
        if (
            self.network_width != 640
            or self.network_height != 640
            or self.classes != EXPECTED_CLASSES
        ):
            raise ValueError("C6-5D model contract changed.")
        self.output0.validate()
        self.output1.validate()


@dataclass(frozen=True)
class SegmentationPostprocessContract:
    """Frozen confidence, NMS, and mask reconstruction behavior."""

    confidence_threshold: float
    nms_iou_threshold: float
    max_detections: int
    mask_threshold: float
    class_agnostic_nms: bool
    mask_representation: str
    mask_sampling: str

    # ADD 2026-09-05: Characterized YOLO11-seg decode/NMS/mask policy를 고정한다.
    def validate(self) -> None:
        if type(self.class_agnostic_nms) is not bool:
            raise TypeError("C6-5D class_agnostic_nms must be bool.")
        if (
            self.confidence_threshold != 0.25
            or self.nms_iou_threshold != 0.7
            or self.max_detections != 300
            or self.mask_threshold != 0.5
            or self.class_agnostic_nms is not False
            or self.mask_representation != "bbox_local_sigmoid_probability"
            or self.mask_sampling != "bilinear_half_pixel"
        ):
            raise ValueError("C6-5D postprocess contract changed.")


@dataclass(frozen=True)
class DeepStreamInstanceSegmentationContract:
    """Gst-nvinfer instance-segmentation parser and mask metadata contract."""

    network_type: int
    cluster_mode: int
    output_instance_mask: bool
    output_tensor_meta: bool
    maintain_aspect_ratio: bool
    symmetric_padding: bool
    parser_symbol: str
    parser_library: str
    label_file: str
    segmentation_threshold: float

    # ADD 2026-09-05: DeepStream instance-mask parser boundary와 preprocessing을 고정한다.
    def validate(self) -> None:
        bool_values = (
            self.output_instance_mask,
            self.output_tensor_meta,
            self.maintain_aspect_ratio,
            self.symmetric_padding,
        )
        if any(type(value) is not bool for value in bool_values):
            raise TypeError("C6-5D DeepStream flags must be bool.")
        if (
            self.network_type != 3
            or self.cluster_mode != 4
            or self.output_instance_mask is not True
            or self.output_tensor_meta is not True
            or self.maintain_aspect_ratio is not True
            or self.symmetric_padding is not True
            or self.parser_symbol != EXPECTED_PARSER_SYMBOL
            or self.parser_library != EXPECTED_PARSER_LIBRARY
            or self.label_file != EXPECTED_LABEL_FILE
            or self.segmentation_threshold != 0.5
        ):
            raise ValueError("C6-5D DeepStream parser contract changed.")


@dataclass(frozen=True)
class SegmentationFoundationPolicy:
    """Scope restrictions for the C6-5D decoder foundation gate."""

    network_allowed: bool
    engine_rebuild_allowed: bool
    segmentation_decode_allowed: bool
    instance_metadata_allowed: bool
    overlay_allowed: bool
    annotated_video_allowed: bool
    dataset_used: bool
    validation_used: bool
    test_used: bool
    final_test_used: bool

    # ADD 2026-09-05: Foundation을 decode/metadata까지만 허용하고 overlay/final-test를 봉인한다.
    def validate(self) -> None:
        values = (
            self.network_allowed,
            self.engine_rebuild_allowed,
            self.segmentation_decode_allowed,
            self.instance_metadata_allowed,
            self.overlay_allowed,
            self.annotated_video_allowed,
            self.dataset_used,
            self.validation_used,
            self.test_used,
            self.final_test_used,
        )
        if any(type(value) is not bool for value in values):
            raise TypeError("C6-5D policy flags must be bool.")
        if (
            self.network_allowed is not False
            or self.engine_rebuild_allowed is not False
            or self.segmentation_decode_allowed is not True
            or self.instance_metadata_allowed is not True
            or self.overlay_allowed is not False
            or self.annotated_video_allowed is not False
            or self.dataset_used is not False
            or self.validation_used is not False
            or self.test_used is not False
            or self.final_test_used is not False
        ):
            raise ValueError("C6-5D foundation policy changed.")


@dataclass(frozen=True)
class DeepStreamSegmentationConfig:
    """Top-level C6-5D segmentation decoder foundation contract."""

    schema_version: int
    decoder_id: str
    c6_5c_closure_commit: str
    engine: SegmentationEngineIdentity
    model: SegmentationModelContract
    postprocess: SegmentationPostprocessContract
    deepstream: DeepStreamInstanceSegmentationContract
    policy: SegmentationFoundationPolicy
    config_path: Path

    # ADD 2026-09-05: C6-5D top-level lineage와 모든 nested contract를 검증한다.
    def validate(self) -> None:
        if (
            self.schema_version != 1
            or self.decoder_id != EXPECTED_DECODER_ID
            or self.c6_5c_closure_commit != EXPECTED_C6_5C_CLOSURE_COMMIT
        ):
            raise ValueError("C6-5D top-level decoder identity changed.")
        self.engine.validate()
        self.model.validate()
        self.postprocess.validate()
        self.deepstream.validate()
        self.policy.validate()


# ADD 2026-09-05: JSON object를 strict mapping으로 변환한다.
def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be JSON object.")
    return cast(dict[str, Any], value)


# ADD 2026-09-05: JSON object field set을 exact schema와 대조한다.
def _require_fields(value: dict[str, Any], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields do not match schema.")


# ADD 2026-09-05: JSON integer scalar를 strict int로 변환한다.
def _integer(value: object, *, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be int.")
    return cast(int, value)


# ADD 2026-09-05: JSON numeric scalar를 finite float로 변환한다.
def _number(value: object, *, label: str) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{label} must be numeric.")
    converted = float(cast(int | float, value))
    if converted != converted or converted in {float("inf"), float("-inf")}:
        raise ValueError(f"{label} must be finite.")
    return converted


# ADD 2026-09-05: JSON boolean scalar를 strict bool로 변환한다.
def _boolean(value: object, *, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be bool.")
    return cast(bool, value)


# ADD 2026-09-05: JSON string scalar를 strict str로 변환한다.
def _string(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be str.")
    return value


# ADD 2026-09-05: Class mapping의 string JSON keys를 strict integer IDs로 변환한다.
def _class_mapping(value: object) -> dict[int, str]:
    raw = _mapping(value, label="model.classes")
    classes: dict[int, str] = {}
    for key, name in raw.items():
        try:
            class_id = int(key)
        except ValueError as exc:
            raise ValueError("model.classes keys must be integer strings.") from exc
        if str(class_id) != key:
            raise ValueError("model.classes keys must use canonical integer strings.")
        classes[class_id] = _string(name, label=f"model.classes[{key}]")
    return classes


# ADD 2026-09-05: Frozen C6-5D JSON을 typed decoder contract로 로드한다.
def load_deepstream_segmentation_config(
    path: Path = DEFAULT_DEEPSTREAM_SEGMENTATION_CONFIG,
) -> DeepStreamSegmentationConfig:
    raw = _mapping(json.loads(path.read_text(encoding="utf-8")), label="C6-5D config")
    _require_fields(
        raw,
        {
            "schema_version",
            "decoder_id",
            "c6_5c_closure_commit",
            "engine",
            "model",
            "postprocess",
            "deepstream",
            "policy",
        },
        label="C6-5D config",
    )

    engine = _mapping(raw["engine"], label="engine")
    model = _mapping(raw["model"], label="model")
    output0 = _mapping(model.get("output0"), label="model.output0")
    output1 = _mapping(model.get("output1"), label="model.output1")
    postprocess = _mapping(raw["postprocess"], label="postprocess")
    deepstream = _mapping(raw["deepstream"], label="deepstream")
    policy = _mapping(raw["policy"], label="policy")

    _require_fields(
        engine,
        {"relative_path", "sha256", "size_bytes", "immutable"},
        label="engine",
    )
    _require_fields(
        model,
        {
            "network_width",
            "network_height",
            "classes",
            "output0",
            "output1",
        },
        label="model",
    )
    _require_fields(
        output0,
        {"name", "channels", "anchors", "layout"},
        label="model.output0",
    )
    _require_fields(
        output1,
        {"name", "channels", "height", "width", "layout"},
        label="model.output1",
    )
    _require_fields(
        postprocess,
        {
            "confidence_threshold",
            "nms_iou_threshold",
            "max_detections",
            "mask_threshold",
            "class_agnostic_nms",
            "mask_representation",
            "mask_sampling",
        },
        label="postprocess",
    )
    _require_fields(
        deepstream,
        {
            "network_type",
            "cluster_mode",
            "output_instance_mask",
            "output_tensor_meta",
            "maintain_aspect_ratio",
            "symmetric_padding",
            "parser_symbol",
            "parser_library",
            "label_file",
            "segmentation_threshold",
        },
        label="deepstream",
    )
    _require_fields(
        policy,
        {
            "network_allowed",
            "engine_rebuild_allowed",
            "segmentation_decode_allowed",
            "instance_metadata_allowed",
            "overlay_allowed",
            "annotated_video_allowed",
            "dataset_used",
            "validation_used",
            "test_used",
            "final_test_used",
        },
        label="policy",
    )

    config = DeepStreamSegmentationConfig(
        schema_version=_integer(raw["schema_version"], label="schema_version"),
        decoder_id=_string(raw["decoder_id"], label="decoder_id"),
        c6_5c_closure_commit=_string(
            raw["c6_5c_closure_commit"],
            label="c6_5c_closure_commit",
        ),
        engine=SegmentationEngineIdentity(
            relative_path=Path(_string(engine["relative_path"], label="engine.relative_path")),
            sha256=_string(engine["sha256"], label="engine.sha256"),
            size_bytes=_integer(engine["size_bytes"], label="engine.size_bytes"),
            immutable=_boolean(engine["immutable"], label="engine.immutable"),
        ),
        model=SegmentationModelContract(
            network_width=_integer(model["network_width"], label="model.network_width"),
            network_height=_integer(model["network_height"], label="model.network_height"),
            classes=_class_mapping(model["classes"]),
            output0=TensorOutput0Contract(
                name=_string(output0["name"], label="model.output0.name"),
                channels=_integer(output0["channels"], label="model.output0.channels"),
                anchors=_integer(output0["anchors"], label="model.output0.anchors"),
                layout=_string(output0["layout"], label="model.output0.layout"),
            ),
            output1=TensorOutput1Contract(
                name=_string(output1["name"], label="model.output1.name"),
                channels=_integer(output1["channels"], label="model.output1.channels"),
                height=_integer(output1["height"], label="model.output1.height"),
                width=_integer(output1["width"], label="model.output1.width"),
                layout=_string(output1["layout"], label="model.output1.layout"),
            ),
        ),
        postprocess=SegmentationPostprocessContract(
            confidence_threshold=_number(
                postprocess["confidence_threshold"],
                label="postprocess.confidence_threshold",
            ),
            nms_iou_threshold=_number(
                postprocess["nms_iou_threshold"],
                label="postprocess.nms_iou_threshold",
            ),
            max_detections=_integer(
                postprocess["max_detections"],
                label="postprocess.max_detections",
            ),
            mask_threshold=_number(
                postprocess["mask_threshold"],
                label="postprocess.mask_threshold",
            ),
            class_agnostic_nms=_boolean(
                postprocess["class_agnostic_nms"],
                label="postprocess.class_agnostic_nms",
            ),
            mask_representation=_string(
                postprocess["mask_representation"],
                label="postprocess.mask_representation",
            ),
            mask_sampling=_string(
                postprocess["mask_sampling"],
                label="postprocess.mask_sampling",
            ),
        ),
        deepstream=DeepStreamInstanceSegmentationContract(
            network_type=_integer(deepstream["network_type"], label="deepstream.network_type"),
            cluster_mode=_integer(deepstream["cluster_mode"], label="deepstream.cluster_mode"),
            output_instance_mask=_boolean(
                deepstream["output_instance_mask"],
                label="deepstream.output_instance_mask",
            ),
            output_tensor_meta=_boolean(
                deepstream["output_tensor_meta"],
                label="deepstream.output_tensor_meta",
            ),
            maintain_aspect_ratio=_boolean(
                deepstream["maintain_aspect_ratio"],
                label="deepstream.maintain_aspect_ratio",
            ),
            symmetric_padding=_boolean(
                deepstream["symmetric_padding"],
                label="deepstream.symmetric_padding",
            ),
            parser_symbol=_string(
                deepstream["parser_symbol"],
                label="deepstream.parser_symbol",
            ),
            parser_library=_string(
                deepstream["parser_library"],
                label="deepstream.parser_library",
            ),
            label_file=_string(
                deepstream["label_file"],
                label="deepstream.label_file",
            ),
            segmentation_threshold=_number(
                deepstream["segmentation_threshold"],
                label="deepstream.segmentation_threshold",
            ),
        ),
        policy=SegmentationFoundationPolicy(
            network_allowed=_boolean(policy["network_allowed"], label="policy.network_allowed"),
            engine_rebuild_allowed=_boolean(
                policy["engine_rebuild_allowed"],
                label="policy.engine_rebuild_allowed",
            ),
            segmentation_decode_allowed=_boolean(
                policy["segmentation_decode_allowed"],
                label="policy.segmentation_decode_allowed",
            ),
            instance_metadata_allowed=_boolean(
                policy["instance_metadata_allowed"],
                label="policy.instance_metadata_allowed",
            ),
            overlay_allowed=_boolean(policy["overlay_allowed"], label="policy.overlay_allowed"),
            annotated_video_allowed=_boolean(
                policy["annotated_video_allowed"],
                label="policy.annotated_video_allowed",
            ),
            dataset_used=_boolean(policy["dataset_used"], label="policy.dataset_used"),
            validation_used=_boolean(policy["validation_used"], label="policy.validation_used"),
            test_used=_boolean(policy["test_used"], label="policy.test_used"),
            final_test_used=_boolean(
                policy["final_test_used"],
                label="policy.final_test_used",
            ),
        ),
        config_path=path,
    )
    config.validate()
    return config


# ADD 2026-09-05: Exact class order를 DeepStream label-file bytes로 생성한다.
def build_labels_text(config: DeepStreamSegmentationConfig) -> str:
    config.validate()
    return "\n".join(config.model.classes[index] for index in range(3)) + "\n"


# ADD 2026-09-05: Sealed engine과 custom instance-mask parser용 nvinfer INI를 생성한다.
def build_nvinfer_config_text(config: DeepStreamSegmentationConfig) -> str:
    config.validate()
    deepstream = config.deepstream
    postprocess = config.postprocess
    return "\n".join(
        (
            "[property]",
            "gpu-id=0",
            "net-scale-factor=0.00392156862745098",
            "model-color-format=0",
            "model-engine-file=/model/model.plan",
            f"labelfile-path={deepstream.label_file}",
            "batch-size=1",
            "network-mode=1",
            "process-mode=1",
            "interval=0",
            "gie-unique-id=1",
            f"network-type={deepstream.network_type}",
            "num-detected-classes=3",
            f"cluster-mode={deepstream.cluster_mode}",
            "output-tensor-meta=1",
            "output-instance-mask=1",
            f"segmentation-threshold={deepstream.segmentation_threshold}",
            "maintain-aspect-ratio=1",
            "symmetric-padding=1",
            f"custom-lib-path={deepstream.parser_library}",
            f"parse-bbox-instance-mask-func-name={deepstream.parser_symbol}",
            "",
            "[class-attrs-all]",
            f"pre-cluster-threshold={postprocess.confidence_threshold}",
            "",
        )
    )


# ADD 2026-09-05: C++ parser의 required ABI와 raw tensor constants를 검증한다.
def validate_parser_source_text(source: str) -> None:
    required_fragments = (
        "constexpr int kPredictionChannels = 39;",
        "constexpr int kAnchorCount = 8400;",
        "constexpr int kClassCount = 3;",
        "constexpr int kMaskChannels = 32;",
        "constexpr int kPrototypeHeight = 160;",
        "constexpr int kPrototypeWidth = 160;",
        "constexpr float kNmsIouThreshold = 0.7F;",
        "constexpr std::size_t kMaxDetections = 300;",
        "const float score = output0[(4 + class_id) * kAnchorCount + anchor];",
        "output0[(7 + coeff) * kAnchorCount + anchor]",
        "NvDsInferParseYolo11Seg(",
        "CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(NvDsInferParseYolo11Seg);",
    )
    if any(fragment not in source for fragment in required_fragments):
        raise ValueError("C6-5D parser source is missing a frozen decoder fragment.")
    forbidden_fragments = (
        "onnx-file=",
        "int8-calib-file=",
        "engine-create-func-name",
    )
    if any(fragment in source for fragment in forbidden_fragments):
        raise ValueError("C6-5D parser source contains forbidden build/rebuild configuration.")


# ADD 2026-09-05: C++ parser file을 읽어 static source contract를 검증한다.
def validate_parser_source_file(
    path: Path = DEFAULT_DEEPSTREAM_SEGMENTATION_PARSER,
) -> None:
    validate_parser_source_text(path.read_text(encoding="utf-8"))
