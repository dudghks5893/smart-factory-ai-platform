"""Focused contracts for frozen YOLO ONNX export metadata and identity."""

from __future__ import annotations

import json
import zipfile
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from ml.deployment.yolo_onnx import (
    DEFAULT_EXPORT_CONFIG,
    EXPECTED_FROZEN_MANIFEST_SHA256,
    OnnxGraphContract,
    OnnxTensorContract,
    YoloOnnxExportMetadata,
    load_yolo_onnx_export_config,
    prepare_frozen_yolo_source,
)
from ml.experiments.yolo_final_candidate import (
    OFFICIAL_METADATA_ENTRY,
    OFFICIAL_MODEL_ENTRY,
    FinalCandidateManifest,
    OfficialCandidateEvidence,
    load_final_candidate_manifest,
    materialize_official_candidate_artifact,
)
from shared.hashing import sha256_file

FROZEN_MANIFEST = Path("configs/model/yolo_segmentation_final_candidate.json")


# ADD 2026-09-02: Frozen pointer field로 synthetic matching Official evidence를 만든다.
def _official_evidence(candidate: FinalCandidateManifest) -> OfficialCandidateEvidence:
    return OfficialCandidateEvidence(
        experiment_id=candidate.selected_experiment_id,
        status="COMPLETED",
        decision="ACCEPT",
        decision_reason="fixture",
        repository_git_commit=candidate.repository_git_commit,
        dataset_manifest_sha256=candidate.dataset_manifest_sha256,
        experiment_config_sha256=candidate.experiment_config_sha256,
        official_package_sha256=candidate.official_package_sha256,
        model_sha256=candidate.model_sha256,
        metadata_sha256=candidate.metadata_sha256,
        packaged_experiment_result_sha256=candidate.packaged_experiment_result_sha256,
        model_size_bytes=candidate.model_size_bytes,
        task=candidate.task,
        model_family=candidate.model_family,
        selected_model_name=candidate.selected_model_name,
        framework=candidate.framework,
        framework_version=candidate.framework_version,
        seed=candidate.seed,
        best_epoch=candidate.best_epoch,
        validation_metrics=candidate.validation_metrics,
        primary_confirmation_checks=candidate.primary_confirmation_checks,
        test_used=False,
        test_split_used=False,
    )


# ADD 2026-09-02: Valid static FP32 export metadata fixture를 만든다.
def _metadata() -> YoloOnnxExportMetadata:
    config = load_yolo_onnx_export_config(DEFAULT_EXPORT_CONFIG)
    graph = OnnxGraphContract(
        opset=18,
        inputs=(OnnxTensorContract("images", "TensorProto.FLOAT", (1, 3, 640, 640)),),
        outputs=(
            OnnxTensorContract("output0", "TensorProto.FLOAT", (1, 39, 8400)),
            OnnxTensorContract("output1", "TensorProto.FLOAT", (1, 32, 160, 160)),
        ),
    )
    config_payload = asdict(config)
    config_payload.pop("config_path")
    config_payload["output_root"] = str(config.output_root)
    return YoloOnnxExportMetadata(
        schema_version=1,
        artifact_type="yolo_segmentation_onnx",
        export_state="ONNX_EXPORT_COMPLETED",
        export_id=config.export_id,
        created_at="2026-09-02T12:00:00+09:00",
        source_experiment_id="c4_2c_yolo11n_seg_crop350_nomosaic_maskratio2_seed42",
        frozen_manifest_sha256="a" * 64,
        official_package_sha256="b" * 64,
        source_model_sha256="c" * 64,
        source_metadata_sha256="d" * 64,
        source_model_family="yolo11n-seg",
        source_task="segment",
        dataset_manifest_sha256="e" * 64,
        export_config_sha256="f" * 64,
        export_config=config_payload,
        onnx_sha256="1" * 64,
        onnx_size_bytes=1234,
        graph=asdict(graph),
        environment={
            "python_version": "3.12.14",
            "platform": "fixture",
            "torch_version": "2.13.0",
            "ultralytics_version": "8.4.128",
            "onnx_version": "1.22.0",
            "python_implementation": "cpython",
        },
        repository={"git_commit": "1" * 40, "working_tree_dirty": False},
        test_used=False,
        test_split_used=False,
    )


# ADD 2026-09-02: Repository config가 conservative static FP32 contract인지 검증한다.
def test_export_config_is_explicit_static_fp32_and_metrics_only() -> None:
    config = load_yolo_onnx_export_config(DEFAULT_EXPORT_CONFIG)
    assert (config.batch, config.imgsz, config.opset) == (1, 640, 18)
    assert (config.dynamic, config.simplify, config.nms) == (False, False, False)
    assert config.precision == "fp32"
    assert config.parity.numeric_thresholds is None
    assert config.parity.test_used is False
    assert config.parity.test_split_used is False


# ADD 2026-09-02: Frozen manifest bytes가 expected C4-3 identity인지 고정한다.
def test_frozen_manifest_identity_is_exact() -> None:
    assert sha256_file(FROZEN_MANIFEST) == EXPECTED_FROZEN_MANIFEST_SHA256


# ADD 2026-09-02: Altered frozen pointer를 package/model 접근 전에 거부한다.
def test_prepare_source_rejects_changed_frozen_manifest(tmp_path: Path) -> None:
    changed = tmp_path / "candidate.json"
    payload = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
    payload["model_sha256"] = "0" * 64
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Manifest SHA-256"):
        prepare_frozen_yolo_source(
            repository_root=tmp_path,
            manifest_path=Path("candidate.json"),
            package_path=tmp_path / "must-not-open.zip",
        )


# ADD 2026-09-02: Official package의 different model bytes를 artifact로 materialize하지 않는다.
def test_materializer_rejects_different_model_bytes(tmp_path: Path) -> None:
    candidate = load_final_candidate_manifest(FROZEN_MANIFEST)
    package = tmp_path / "official.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr(OFFICIAL_MODEL_ENTRY, b"not-the-frozen-model")
        archive.writestr(OFFICIAL_METADATA_ENTRY, b"not-the-frozen-metadata")
    candidate = replace(candidate, official_package_sha256=sha256_file(package))
    output = tmp_path / "artifact"
    with pytest.raises(RuntimeError, match="changed verified identity"):
        materialize_official_candidate_artifact(
            package_path=package,
            candidate=candidate,
            evidence=_official_evidence(candidate),
            artifact_dir=output,
        )
    assert not output.exists()


# ADD 2026-09-02: Export metadata schema와 serialization이 deterministic strict JSON인지 검증한다.
def test_export_metadata_round_trip_is_deterministic() -> None:
    metadata = _metadata()
    first = metadata.to_json_bytes()
    restored = YoloOnnxExportMetadata.from_json_dict(json.loads(first))
    assert restored.to_json_bytes() == first
    assert json.loads(first)["graph"]["inputs"][0]["shape"] == [1, 3, 640, 640]


# ADD 2026-09-02: Malformed segmentation graph output schema를 거부한다.
def test_export_metadata_rejects_invalid_output_schema() -> None:
    raw = asdict(_metadata())
    raw["graph"]["outputs"] = []
    with pytest.raises(ValueError, match="output0 and output1"):
        YoloOnnxExportMetadata.from_json_dict(raw)


# ADD 2026-09-02: Config scalar bool/int coercion을 허용하지 않는다.
def test_export_config_rejects_boolean_batch(tmp_path: Path) -> None:
    raw = DEFAULT_EXPORT_CONFIG.read_text(encoding="utf-8").replace("batch: 1", "batch: true")
    path = tmp_path / "export.yaml"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises((TypeError, ValueError), match="scalar types"):
        load_yolo_onnx_export_config(path)
