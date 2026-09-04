"""Unit tests for the C6-5D DeepStream segmentation decoder foundation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from services.streaming.yolo_deepstream_segmentation import (
    DEFAULT_DEEPSTREAM_SEGMENTATION_CONFIG,
    DEFAULT_DEEPSTREAM_SEGMENTATION_LABELS,
    DEFAULT_DEEPSTREAM_SEGMENTATION_PARSER,
    EXPECTED_C6_5C_CLOSURE_COMMIT,
    EXPECTED_CLASSES,
    EXPECTED_DECODER_ID,
    EXPECTED_PARSER_SYMBOL,
    build_labels_text,
    build_nvinfer_config_text,
    load_deepstream_segmentation_config,
    validate_parser_source_file,
    validate_parser_source_text,
)


# ADD 2026-09-05: Frozen JSON의 C6-5C lineage와 tensor contract를 검증한다.
def test_default_segmentation_config_loads_exact_contract() -> None:
    config = load_deepstream_segmentation_config()

    assert config.config_path == DEFAULT_DEEPSTREAM_SEGMENTATION_CONFIG
    assert config.decoder_id == EXPECTED_DECODER_ID
    assert config.c6_5c_closure_commit == EXPECTED_C6_5C_CLOSURE_COMMIT
    assert config.model.classes == EXPECTED_CLASSES
    assert (config.model.output0.channels, config.model.output0.anchors) == (39, 8400)
    assert (
        config.model.output1.channels,
        config.model.output1.height,
        config.model.output1.width,
    ) == (32, 160, 160)


# ADD 2026-09-05: DeepStream label file이 class ID 순서와 byte-for-byte 일치하는지 검증한다.
def test_labels_file_matches_class_mapping() -> None:
    config = load_deepstream_segmentation_config()

    expected = build_labels_text(config)

    assert expected == "bent\ncolor\nscratch\n"
    assert DEFAULT_DEEPSTREAM_SEGMENTATION_LABELS.read_text(encoding="utf-8") == expected


# ADD 2026-09-05: Generated nvinfer INI의 instance-mask parser contract를 검증한다.
def test_nvinfer_config_uses_instance_mask_parser() -> None:
    config = load_deepstream_segmentation_config()

    text = build_nvinfer_config_text(config)

    required = (
        "model-engine-file=/model/model.plan",
        "network-type=3",
        "cluster-mode=4",
        "output-tensor-meta=1",
        "output-instance-mask=1",
        "segmentation-threshold=0.5",
        "maintain-aspect-ratio=1",
        "symmetric-padding=1",
        f"parse-bbox-instance-mask-func-name={EXPECTED_PARSER_SYMBOL}",
        "pre-cluster-threshold=0.25",
    )
    assert all(fragment in text for fragment in required)
    assert "onnx-file=" not in text
    assert "int8-calib-file=" not in text
    assert "engine-create-func-name=" not in text


# ADD 2026-09-05: Decoder threshold 변경이 frozen postprocess validation에서 거부되는지 검증한다.
def test_postprocess_contract_rejects_threshold_change() -> None:
    config = load_deepstream_segmentation_config()
    changed = replace(
        config.postprocess,
        confidence_threshold=0.20,
    )

    with pytest.raises(ValueError, match="postprocess contract changed"):
        changed.validate()


# ADD 2026-09-05: Overlay나 sealed final-test 활성화가 foundation scope에서 거부되는지 검증한다.
def test_foundation_policy_rejects_overlay_or_final_test() -> None:
    config = load_deepstream_segmentation_config()

    with pytest.raises(ValueError, match="foundation policy changed"):
        replace(config.policy, overlay_allowed=True).validate()

    with pytest.raises(ValueError, match="foundation policy changed"):
        replace(config.policy, final_test_used=True).validate()


# ADD 2026-09-05: Repository-owned parser가 exact ABI/raw-layout constants를 포함하는지 검증한다.
def test_cpp_parser_source_satisfies_static_contract() -> None:
    validate_parser_source_file()

    source = DEFAULT_DEEPSTREAM_SEGMENTATION_PARSER.read_text(encoding="utf-8")
    assert "NvDsInferParseYolo11Seg(" in source
    assert "kNmsIouThreshold = 0.7F" in source
    assert "kMaxDetections = 300" in source


# ADD 2026-09-05: Parser source의 required raw-layout fragment를 검증한다.
def test_cpp_parser_source_rejects_missing_fragment() -> None:
    source = DEFAULT_DEEPSTREAM_SEGMENTATION_PARSER.read_text(encoding="utf-8")
    damaged = source.replace(
        "constexpr int kAnchorCount = 8400;",
        "constexpr int kAnchorCount = 8401;",
        1,
    )

    with pytest.raises(ValueError, match="missing a frozen decoder fragment"):
        validate_parser_source_text(damaged)


# ADD 2026-09-05: Decoder config loader가 noncanonical class key를 거부하는지 검증한다.
def test_config_loader_rejects_noncanonical_class_key(tmp_path: Path) -> None:
    raw = DEFAULT_DEEPSTREAM_SEGMENTATION_CONFIG.read_text(encoding="utf-8")
    damaged = raw.replace('"0": "bent"', '"00": "bent"', 1)
    path = tmp_path / "decoder.json"
    path.write_text(damaged, encoding="utf-8")

    with pytest.raises(ValueError, match="canonical integer strings"):
        load_deepstream_segmentation_config(path)


# ADD 2026-09-05: Parser Docker builder의 pinned runtime contract를 검증한다.
def test_parser_dockerfile_pins_reproducible_build_contract() -> None:
    dockerfile = Path("services/streaming/deepstream/Dockerfile.yolo11_seg_parser")
    text = dockerfile.read_text(encoding="utf-8")

    required = (
        "nvidia/cuda@sha256:d266e59b88c295bc5fa0e4cef9064eaff84939381b09e2c3d76a5532a303e42d",
        "nvcr.io/nvidia/deepstream@sha256:"
        "4f80b374e4a5086552825fe0f5bdd015c8cfd3dbe430cdde5ce9572e80e01583",
        "/usr/local/cuda-13.2/targets/x86_64-linux/include",
        "COPY services/streaming/deepstream/yolo11_seg_parser.cpp",
        "-DC6_5D_PARSER_SELF_TEST",
        "NvDsInferParseYolo11Seg",
        "/work/libc6_5d_yolo11_seg_parser.so",
        "/work/c6_5d_labels.txt",
        "C6_5D_REPRODUCIBLE_PARSER_BUILD=PASS",
    )

    assert all(fragment in text for fragment in required)
    assert "apt-get" not in text
    assert "curl " not in text
    assert "wget " not in text
