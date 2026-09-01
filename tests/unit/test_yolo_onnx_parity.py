"""Synthetic validation-only contracts for YOLO PyTorch and ONNX parity."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from ml.datasets.yolo_segmentation_manifest import DerivedManifestRecord
from ml.deployment.yolo_onnx import FrozenYoloSource, YoloOnnxExportMetadata
from ml.deployment.yolo_onnx_parity import (
    BackendPrediction,
    RuntimeTensorObservation,
    build_parity_evidence,
    build_sample_parity,
    load_parity_validation_records,
    match_backend_predictions,
    observe_runtime_tensor,
    predict_backend,
)
from ml.evaluation.final_benchmark import RepositoryProvenance
from ml.evaluation.yolo_segmentation_error_analysis import PredictedInstance, mask_box
from ml.training.yolo_segmentation import load_yolo_segmentation_config

BASELINE_CONFIG = Path("configs/model/yolo_segmentation_baseline.yaml")


# ADD 2026-09-02: Validation/test split을 바꿔 쓸 수 있는 manifest record fixture를 만든다.
def _record(*, split: str = "val") -> DerivedManifestRecord:
    return DerivedManifestRecord(
        dataset_name="fixture",
        dataset_version="v1",
        derived_task="yolo_segmentation",
        source_manifest_sha256="a" * 64,
        source_split="train",
        source_manifest_split="train",
        source_image_path="raw/image.png",
        source_mask_path="raw/mask.png",
        category="metal_nut",
        sample_id="sample-001",
        defect_type="bent",
        target_class="bent",
        target_class_id="0",
        derived_split=split,
        is_negative=False,
        image_width=8,
        image_height=8,
        image_path=f"images/{split}/sample.png",
        label_path=f"labels/{split}/sample.txt",
        image_sha256="b" * 64,
        mask_sha256="c" * 64,
        polygon_count=1,
        component_count=1,
        hole_count=0,
        polygon_vertex_count=4,
        round_trip_iou="1.0",
        pixel_precision="1.0",
        pixel_recall="1.0",
    )


# ADD 2026-09-02: Small boolean mask에서 normalized predicted instance를 만든다.
def _instance(
    *,
    class_id: int,
    confidence: float,
    row: int,
    column: int,
) -> PredictedInstance:
    mask = np.zeros((8, 8), dtype=np.bool_)
    mask[row : row + 2, column : column + 2] = True
    return PredictedInstance(
        class_id=class_id,
        confidence=confidence,
        mask=mask,
        box_xyxy=mask_box(mask),
    )


# ADD 2026-09-02: Finite synthetic output tensor observation fixture를 만든다.
def _tensors() -> tuple[RuntimeTensorObservation, ...]:
    return (
        RuntimeTensorObservation("boxes.xyxy", "float32", (1, 4), True),
        RuntimeTensorObservation("boxes.cls", "float32", (1,), True),
        RuntimeTensorObservation("boxes.conf", "float32", (1,), True),
        RuntimeTensorObservation("masks.data", "float32", (1, 8, 8), True),
    )


# ADD 2026-09-02: Greedy maximum-overlap matching과 class disagreement evidence를 검증한다.
def test_parity_matching_is_spatial_and_reports_class_agreement() -> None:
    left = (
        _instance(class_id=0, confidence=0.9, row=0, column=0),
        _instance(class_id=1, confidence=0.8, row=5, column=5),
    )
    right = (
        _instance(class_id=2, confidence=0.89, row=0, column=0),
        _instance(class_id=1, confidence=0.79, row=5, column=5),
    )
    matches = match_backend_predictions(left, right)
    assert [(item.pytorch_index, item.onnx_index) for item in matches] == [(0, 0), (1, 1)]
    assert [item.class_agreement for item in matches] == [False, True]
    assert all(item.mask_iou == 1.0 for item in matches)


# ADD 2026-09-02: Unmatched counts와 deterministic sample evidence conservation을 검증한다.
def test_sample_parity_preserves_prediction_counts() -> None:
    pytorch = BackendPrediction(
        instances=(
            _instance(class_id=0, confidence=0.9, row=0, column=0),
            _instance(class_id=1, confidence=0.8, row=5, column=5),
        ),
        tensors=_tensors(),
    )
    onnx = BackendPrediction(
        instances=(_instance(class_id=0, confidence=0.89, row=0, column=0),),
        tensors=_tensors(),
    )
    evidence = build_sample_parity(record=_record(), pytorch=pytorch, onnx=onnx)
    assert evidence.matched_instance_count == 1
    assert evidence.unmatched_pytorch_count == 1
    assert evidence.unmatched_onnx_count == 0
    assert evidence.pytorch_predictions[0].class_id == 0
    assert evidence.pytorch_predictions[0].confidence == 0.9
    assert evidence.pytorch_predictions[0].mask_shape == (8, 8)
    assert len(evidence.pytorch_predictions[0].mask_sha256) == 64


# ADD 2026-09-02: NaN/Inf backend output을 evidence publication 전에 거부한다.
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_runtime_tensor_rejects_non_finite(value: float) -> None:
    with pytest.raises(ValueError, match="NaN or Inf"):
        observe_runtime_tensor("output", np.asarray([value], dtype=np.float32))


# ADD 2026-09-02: Test row는 source image 존재 여부를 확인하기 전 거부한다.
def test_parity_prediction_rejects_test_split_without_content_access(tmp_path: Path) -> None:
    class MustNotRun:
        predictor: object = object()
        names: Mapping[int, str] = {0: "bent", 1: "color", 2: "scratch"}

        def predict(self, **_: object) -> list[object]:
            raise AssertionError("test image prediction must not run")

    with pytest.raises(ValueError, match="validation rows only"):
        predict_backend(
            model=MustNotRun(),
            record=_record(split="test"),
            dataset_root=tmp_path,
            imgsz=640,
        )


# ADD 2026-09-02: Sample evidence도 non-validation split을 fail-fast한다.
def test_sample_parity_rejects_test_split() -> None:
    prediction = BackendPrediction(instances=(), tensors=_tensors())
    with pytest.raises(ValueError, match="non-validation"):
        build_sample_parity(
            record=replace(_record(), derived_split="test"),
            pytorch=prediction,
            onnx=prediction,
        )


# ADD 2026-09-02: Dataset boundary가 shared validator에 val-only split을 전달하는지 검증한다.
def test_parity_dataset_boundary_requests_validation_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    record = _record()
    observed_splits: frozenset[str] | None = None

    # Validation split argument를 기록하되 real dataset content는 열지 않는다.
    def validator(
        _: Path,
        __: object,
        *,
        content_splits: frozenset[str],
    ) -> tuple[DerivedManifestRecord, ...]:
        nonlocal observed_splits
        observed_splits = content_splits
        return (record,)

    monkeypatch.setattr(
        "ml.deployment.yolo_onnx_parity.validate_experiment_dataset",
        validator,
    )
    contract = load_yolo_segmentation_config(BASELINE_CONFIG).dataset_contract
    assert load_parity_validation_records(tmp_path, contract) == (record,)
    assert observed_splits == frozenset({"val"})


# ADD 2026-09-02: Aggregate parity evidence serialization이 deterministic strict JSON인지 검증한다.
def test_parity_evidence_serialization_is_deterministic() -> None:
    prediction = BackendPrediction(
        instances=(_instance(class_id=0, confidence=0.9, row=0, column=0),),
        tensors=_tensors(),
    )
    sample = build_sample_parity(record=_record(), pytorch=prediction, onnx=prediction)
    source = cast(
        FrozenYoloSource,
        SimpleNamespace(
            candidate=SimpleNamespace(
                selected_experiment_id="c4_2c_yolo11n_seg_crop350_nomosaic_maskratio2_seed42",
                model_sha256="b" * 64,
            ),
            manifest_sha256="a" * 64,
        ),
    )
    metadata = cast(
        YoloOnnxExportMetadata,
        SimpleNamespace(onnx_sha256="c" * 64, export_config_sha256="d" * 64),
    )
    evidence = build_parity_evidence(
        parity_id="fixture-parity",
        created_at="2026-09-02T12:00:00+09:00",
        source=source,
        export_metadata=metadata,
        samples=(sample,),
        provenance=RepositoryProvenance(git_commit="1" * 40, working_tree_dirty=False),
    )
    assert evidence.to_json_bytes() == evidence.to_json_bytes()
    assert b'"numeric_acceptance": "PENDING_APPROVED_TOLERANCES"' in evidence.to_json_bytes()
