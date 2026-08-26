"""Integration contracts for atomic dual-model inspection serving."""

from __future__ import annotations

from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import cast
from uuid import UUID

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from torch import Tensor

from ml.training.yolo_segmentation import ARTIFACT_SCHEMA_VERSION, YoloArtifactMetadata
from services.api import images as image_module
from services.api.app import create_app, load_combined_inspection_repository
from services.api.config import ServingSettings
from services.api.schemas import (
    CombinedInspectionCreatedEvent,
    InspectionCreatedEvent,
    KnownDefectCreatedEvent,
)
from services.api.websockets import (
    CombinedInspectionEventBroadcaster,
    InspectionEventBroadcaster,
    KnownDefectEventBroadcaster,
)
from services.inference.runtime import (
    InferenceResult,
    ModelRuntime,
    PatchCoreRuntimeConfig,
    ServingProvenance,
)
from services.inference.yolo_segmentation_runtime import (
    YoloSegmentationAdapter,
    YoloSegmentationInstance,
    YoloSegmentationProvenance,
    YoloSegmentationResult,
)
from services.persistence.combined_inspections import (
    CombinedInspection,
    CombinedInspectionCreate,
    CombinedInspectionPage,
    CombinedInspectionRepository,
)
from services.persistence.database import DatabaseManager, PersistenceError
from tests.persistence_helpers import prepare_sqlite_database


class _PatchCoreRuntime:
    model_name = "patchcore"
    category = "metal_nut"
    device = "cpu"
    provenance = ServingProvenance(
        manifest_sha256="a" * 64,
        artifact_metadata_sha256="b" * 64,
        model_sha256="c" * 64,
        threshold_artifact_sha256="d" * 64,
    )

    def __init__(self, *, is_anomaly: bool = True) -> None:
        self.is_anomaly = is_anomaly

    def predict(self, image: Tensor) -> InferenceResult:
        assert image.shape == (1, 3, 8, 8)
        return InferenceResult(
            model_name=self.model_name,
            category=self.category,
            is_anomaly=self.is_anomaly,
            anomaly_score=50.0 if self.is_anomaly else 30.0,
            threshold=40.0,
            comparison_operator=">",
        )


class _FailingPatchCoreRuntime(_PatchCoreRuntime):
    def predict(self, image: Tensor) -> InferenceResult:
        raise RuntimeError("private PatchCore detail")


class _YoloRuntime:
    device = "cpu"

    def __init__(self, *, fail: bool = False, instance_count: int = 1) -> None:
        self.metadata = _yolo_metadata()
        self.provenance = YoloSegmentationProvenance(
            dataset_manifest_sha256="1" * 64,
            dataset_semantic_fingerprint_sha256="2" * 64,
            artifact_metadata_sha256="3" * 64,
            model_sha256="4" * 64,
            framework_version="8.4.128",
        )
        self.fail = fail
        self.instance_count = instance_count

    def predict(
        self,
        image_rgb: np.ndarray,
        *,
        diagnostic_confidence: float,
    ) -> YoloSegmentationResult:
        if self.fail:
            raise RuntimeError("private YOLO detail")
        assert image_rgb.shape == (8, 8, 3)
        assert diagnostic_confidence == 0.25
        instances: list[YoloSegmentationInstance] = []
        for index in range(self.instance_count):
            mask = np.zeros((8, 8), dtype=np.bool_)
            mask[1 + index : 5 + index, 1:5] = True
            instances.append(
                YoloSegmentationInstance(
                    class_id=0 if index == 0 else 2,
                    class_name="bent" if index == 0 else "scratch",
                    confidence=0.95 - index * 0.1,
                    box_xyxy=(1.0, 1.0 + index, 5.0, 5.0 + index),
                    mask=mask,
                )
            )
        return YoloSegmentationResult(
            image_width=8,
            image_height=8,
            device=self.device,
            inference_ms=4.5,
            instances=tuple(instances),
        )


class _FailingCombinedRepository:
    def check_ready(self) -> None:
        return None

    def create(self, values: CombinedInspectionCreate) -> CombinedInspection:
        raise PersistenceError("private combined database detail")

    def get(self, combined_inspection_id: UUID) -> CombinedInspection | None:
        return None

    def list(self, *, limit: int, offset: int) -> CombinedInspectionPage:
        return CombinedInspectionPage(items=(), limit=limit, offset=offset, has_more=False)


class _InspectionBroadcaster(InspectionEventBroadcaster):
    def __init__(self, order: list[str] | None = None) -> None:
        super().__init__()
        self.events: list[InspectionCreatedEvent] = []
        self.order = order

    async def broadcast(self, event: InspectionCreatedEvent) -> None:
        self.events.append(event)
        if self.order is not None:
            self.order.append(event.type)
        await super().broadcast(event)


class _KnownDefectBroadcaster(KnownDefectEventBroadcaster):
    def __init__(self, order: list[str] | None = None) -> None:
        super().__init__()
        self.events: list[KnownDefectCreatedEvent] = []
        self.order = order

    async def broadcast(self, event: KnownDefectCreatedEvent) -> None:
        self.events.append(event)
        if self.order is not None:
            self.order.append(event.type)
        await super().broadcast(event)


class _CombinedBroadcaster(CombinedInspectionEventBroadcaster):
    def __init__(self, order: list[str] | None = None) -> None:
        super().__init__()
        self.events: list[CombinedInspectionCreatedEvent] = []
        self.order = order

    async def broadcast(self, event: CombinedInspectionCreatedEvent) -> None:
        self.events.append(event)
        if self.order is not None:
            self.order.append(event.type)
        await super().broadcast(event)


def _yolo_metadata() -> YoloArtifactMetadata:
    return YoloArtifactMetadata(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        model_name="yolo11n-seg.pt",
        task="segment",
        architecture="yolo11n-seg",
        category="metal_nut",
        classes={0: "bent", 1: "color", 2: "scratch"},
        seed=42,
        dataset_manifest_sha256="1" * 64,
        dataset_semantic_fingerprint_sha256="2" * 64,
        training_config={"training": {"imgsz": 640}},
        created_at="2026-08-26T00:00:00+00:00",
        framework="ultralytics",
        framework_version="8.4.128",
        torch_version="2.13.0",
        device="cuda:0",
        best_epoch=60,
        source_checkpoint="weights/best.pt",
        checkpoint_sha256="4" * 64,
    )


def _settings(tmp_path: Path, *, yolo_enabled: bool = True) -> ServingSettings:
    return ServingSettings(
        artifact_dir=tmp_path / "patchcore",
        thresholds_path=tmp_path / "thresholds.json",
        database_url=prepare_sqlite_database(tmp_path),
        model_device="cpu",
        yolo_segmentation_enabled=yolo_enabled,
        yolo_segmentation_artifact_dir=(tmp_path / "yolo" if yolo_enabled else None),
        yolo_segmentation_device="cpu",
        yolo_segmentation_diagnostic_confidence=0.25,
    )


def _png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (8, 8), color=(120, 80, 40)).save(buffer, format="PNG")
    return buffer.getvalue()


def _patchcore_loader(_config: PatchCoreRuntimeConfig) -> ModelRuntime:
    return _PatchCoreRuntime()


def _app(
    tmp_path: Path,
    *,
    patchcore_runtime: ModelRuntime | None = None,
    yolo_runtime: _YoloRuntime | None = None,
    combined_repository_loader: (
        Callable[[DatabaseManager], CombinedInspectionRepository] | None
    ) = None,
    inspection_broadcaster: InspectionEventBroadcaster | None = None,
    known_broadcaster: KnownDefectEventBroadcaster | None = None,
    combined_broadcaster: CombinedInspectionEventBroadcaster | None = None,
) -> FastAPI:
    patchcore = patchcore_runtime or _PatchCoreRuntime()
    runtime = yolo_runtime or _YoloRuntime()
    return create_app(
        settings=_settings(tmp_path),
        runtime_loader=lambda _config: patchcore,
        yolo_runtime_loader=lambda _config: cast(YoloSegmentationAdapter, runtime),
        inspection_event_broadcaster=inspection_broadcaster,
        known_defect_event_broadcaster=known_broadcaster,
        combined_inspection_event_broadcaster=combined_broadcaster,
        combined_inspection_repository_loader=(
            combined_repository_loader or load_combined_inspection_repository
        ),
    )


# ADD 2026-08-26: POST/GET correlation, shared image와 양쪽 committed event를 함께 검증한다.
def test_combined_inspection_persists_recoverable_children_and_events(tmp_path: Path) -> None:
    event_order: list[str] = []
    inspection_broadcaster = _InspectionBroadcaster(event_order)
    known_broadcaster = _KnownDefectBroadcaster(event_order)
    combined_broadcaster = _CombinedBroadcaster(event_order)
    app = _app(
        tmp_path,
        inspection_broadcaster=inspection_broadcaster,
        known_broadcaster=known_broadcaster,
        combined_broadcaster=combined_broadcaster,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/combined-inspections",
            files={"image": ("sample.png", _png_bytes(), "image/png")},
        )
        recovery = client.get(
            f"/v1/combined-inspections/{response.json()['combined_inspection_id']}"
        )
        patch_history = client.get("/v1/inspections")
        yolo_history = client.get("/v1/known-defects")
        patch_detail = client.get(
            f"/v1/inspections/{response.json()['patchcore']['inspection_id']}"
        )
        yolo_detail = client.get(
            f"/v1/known-defects/{response.json()['known_defects']['inspection_id']}"
        )
        missing = client.get(f"/v1/combined-inspections/{UUID(int=0)}")
        combined_history = client.get("/v1/combined-inspections?limit=1&offset=0")

    assert response.status_code == recovery.status_code == 200
    assert response.json() == recovery.json()
    payload = response.json()
    assert UUID(payload["combined_inspection_id"])
    assert payload["image"]["width"] == payload["image"]["height"] == 8
    assert payload["patchcore"]["is_anomaly"] is True
    assert payload["known_defects"]["instance_count"] == 1
    assert payload["known_defects"]["instances"][0]["class_name"] == "bent"
    assert payload["timings"]["yolo_inference_ms"] == 4.5
    assert patch_history.json()["returned_count"] == 1
    assert yolo_history.json()["returned_count"] == 1
    assert patch_detail.json()["image_sha256"] == payload["image"]["sha256"]
    assert yolo_detail.json()["image_sha256"] == payload["image"]["sha256"]
    assert len(inspection_broadcaster.events) == len(known_broadcaster.events) == 1
    assert len(combined_broadcaster.events) == 1
    assert event_order == [
        "inspection.created",
        "known_defect.created",
        "combined_inspection.created",
    ]
    assert payload["decision"] == {
        "disposition": "REJECT",
        "policy": {"name": "model_agreement", "version": "1"},
        "reason_code": "CONFIRMED_KNOWN_DEFECT",
        "reason": ("PatchCore anomaly evidence and known-defect instances are both present."),
        "evidence": {
            "patchcore": {"prediction": "ANOMALY", "score": 50.0, "threshold": 40.0},
            "known_defects": {"instance_count": 1, "classes": ["bent"]},
        },
    }
    assert combined_history.json()["items"][0] == {
        "combined_inspection_id": payload["combined_inspection_id"],
        "created_at": payload["created_at"],
        "patchcore_prediction": "ANOMALY",
        "known_defect_instance_count": 1,
        "disposition": "REJECT",
        "reason_code": "CONFIRMED_KNOWN_DEFECT",
        "policy": {"name": "model_agreement", "version": "1"},
    }
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "combined_inspection_not_found"


# ADD 2026-08-26: Empty와 multi-instance YOLO output을 combined schema가 그대로 보존함을 검증한다.
@pytest.mark.parametrize("instance_count", [0, 2])
def test_combined_inspection_preserves_yolo_cardinality(
    tmp_path: Path,
    instance_count: int,
) -> None:
    app = _app(tmp_path, yolo_runtime=_YoloRuntime(instance_count=instance_count))

    with TestClient(app) as client:
        response = client.post(
            "/v1/combined-inspections",
            files={"image": ("sample.png", _png_bytes(), "image/png")},
        )

    assert response.status_code == 200
    assert response.json()["known_defects"]["instance_count"] == instance_count
    assert len(response.json()["known_defects"]["instances"]) == instance_count
    expected = "REVIEW" if instance_count == 0 else "REJECT"
    assert response.json()["decision"]["disposition"] == expected


# ADD 2026-08-26: Combined HTTP contract가 Policy v1 truth table과 stable codes를 직렬화한다.
@pytest.mark.parametrize(
    ("is_anomaly", "instance_count", "disposition", "reason_code"),
    [
        (False, 0, "PASS", "NO_ANOMALY_EVIDENCE"),
        (True, 0, "REVIEW", "UNKNOWN_ANOMALY"),
        (False, 1, "REVIEW", "MODEL_DISAGREEMENT"),
        (True, 1, "REJECT", "CONFIRMED_KNOWN_DEFECT"),
    ],
)
def test_combined_api_decision_truth_table(
    tmp_path: Path,
    is_anomaly: bool,
    instance_count: int,
    disposition: str,
    reason_code: str,
) -> None:
    app = _app(
        tmp_path,
        patchcore_runtime=_PatchCoreRuntime(is_anomaly=is_anomaly),
        yolo_runtime=_YoloRuntime(instance_count=instance_count),
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/combined-inspections",
            files={"image": ("sample.png", _png_bytes(), "image/png")},
        )

    assert response.status_code == 200
    assert response.json()["decision"]["disposition"] == disposition
    assert response.json()["decision"]["reason_code"] == reason_code
    assert response.json()["decision"]["policy"] == {
        "name": "model_agreement",
        "version": "1",
    }


# ADD 2026-08-26: Combined WebSocket이 commit 뒤 compact decision summary만 전달함을 검증한다.
def test_combined_websocket_emits_committed_decision_summary(tmp_path: Path) -> None:
    app = _app(tmp_path)

    with TestClient(app) as client:
        with client.websocket_connect("/v1/ws/combined-inspections") as websocket:
            response = client.post(
                "/v1/combined-inspections",
                files={"image": ("sample.png", _png_bytes(), "image/png")},
            )
            event = websocket.receive_json()

    assert response.status_code == 200
    assert event["schema_version"] == "1"
    assert event["type"] == "combined_inspection.created"
    assert event["inspection"] == {
        "combined_inspection_id": response.json()["combined_inspection_id"],
        "created_at": response.json()["created_at"],
        "patchcore_prediction": "ANOMALY",
        "known_defect_instance_count": 1,
        "known_defect_classes": ["bent"],
        "disposition": "REJECT",
        "reason_code": "CONFIRMED_KNOWN_DEFECT",
        "policy_name": "model_agreement",
        "policy_version": "1",
    }
    assert "image" not in event["inspection"]
    assert "instances" not in event["inspection"]


# ADD 2026-08-26: Combined history의 bounded newest-first pagination을 검증한다.
def test_combined_history_pagination_is_bounded_and_deterministic(tmp_path: Path) -> None:
    app = _app(tmp_path)

    with TestClient(app) as client:
        created = [
            client.post(
                "/v1/combined-inspections",
                files={"image": ("sample.png", _png_bytes(), "image/png")},
            ).json()
            for _ in range(3)
        ]
        first_page = client.get("/v1/combined-inspections?limit=2&offset=0").json()
        second_page = client.get("/v1/combined-inspections?limit=2&offset=2").json()

    expected_ids = [
        item["combined_inspection_id"]
        for item in sorted(
            created,
            key=lambda item: (item["created_at"], item["combined_inspection_id"]),
            reverse=True,
        )
    ]
    assert [item["combined_inspection_id"] for item in first_page["items"]] == expected_ids[:2]
    assert first_page["returned_count"] == 2
    assert first_page["has_more"] is True
    assert [item["combined_inspection_id"] for item in second_page["items"]] == [expected_ids[2]]
    assert second_page["has_more"] is False


# ADD 2026-08-26: Combined UUID uniqueness와 malformed upload 거부를 함께 검증한다.
def test_combined_identity_is_unique_and_malformed_image_is_rejected(tmp_path: Path) -> None:
    app = _app(tmp_path)

    with TestClient(app) as client:
        first = client.post(
            "/v1/combined-inspections",
            files={"image": ("sample.png", _png_bytes(), "image/png")},
        )
        second = client.post(
            "/v1/combined-inspections",
            files={"image": ("sample.png", _png_bytes(), "image/png")},
        )
        malformed = client.post(
            "/v1/combined-inspections",
            files={"image": ("bad.png", b"not-an-image", "image/png")},
        )

    assert first.json()["combined_inspection_id"] != second.json()["combined_inspection_id"]
    assert malformed.status_code == 400
    assert malformed.json()["error"]["code"] == "invalid_image"


# ADD 2026-08-26: Combined request가 Pillow RGB decode boundary를 정확히 한 번 호출함을 검증한다.
def test_combined_inspection_decodes_upload_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_decode = image_module._decode_uploaded_rgb_image
    decode_calls = 0

    def recording_decode(
        content: bytes,
        *,
        content_type: str | None,
        max_upload_bytes: int,
    ) -> Image.Image:
        nonlocal decode_calls
        decode_calls += 1
        return original_decode(
            content,
            content_type=content_type,
            max_upload_bytes=max_upload_bytes,
        )

    monkeypatch.setattr(image_module, "_decode_uploaded_rgb_image", recording_decode)
    app = _app(tmp_path)

    with TestClient(app) as client:
        response = client.post(
            "/v1/combined-inspections",
            files={"image": ("sample.png", _png_bytes(), "image/png")},
        )

    assert response.status_code == 200
    assert decode_calls == 1


# ADD 2026-08-26: YOLO failure가 correlation 또는 어느 child row도 남기지 않음을 검증한다.
def test_combined_model_failure_persists_no_partial_result(tmp_path: Path) -> None:
    app = _app(tmp_path, yolo_runtime=_YoloRuntime(fail=True))

    with TestClient(app) as client:
        response = client.post(
            "/v1/combined-inspections",
            files={"image": ("sample.png", _png_bytes(), "image/png")},
        )
        patch_history = client.get("/v1/inspections")
        yolo_history = client.get("/v1/known-defects")

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "inference_failed"
    assert patch_history.json()["returned_count"] == 0
    assert yolo_history.json()["returned_count"] == 0


# ADD 2026-08-26: PatchCore failure의 safe response와 zero persistence를 검증한다.
def test_combined_patchcore_failure_persists_no_partial_result(tmp_path: Path) -> None:
    app = _app(tmp_path, patchcore_runtime=_FailingPatchCoreRuntime())

    with TestClient(app) as client:
        response = client.post(
            "/v1/combined-inspections",
            files={"image": ("sample.png", _png_bytes(), "image/png")},
        )
        patch_history = client.get("/v1/inspections")
        yolo_history = client.get("/v1/known-defects")

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "inference_failed"
    assert patch_history.json()["returned_count"] == 0
    assert yolo_history.json()["returned_count"] == 0


# ADD 2026-08-26: Atomic repository failure가 양쪽 child history와 event를 모두 억제함을 검증한다.
def test_combined_persistence_failure_has_no_partial_rows_or_events(tmp_path: Path) -> None:
    inspection_broadcaster = _InspectionBroadcaster()
    known_broadcaster = _KnownDefectBroadcaster()
    combined_broadcaster = _CombinedBroadcaster()
    app = _app(
        tmp_path,
        combined_repository_loader=lambda _database: _FailingCombinedRepository(),
        inspection_broadcaster=inspection_broadcaster,
        known_broadcaster=known_broadcaster,
        combined_broadcaster=combined_broadcaster,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/combined-inspections",
            files={"image": ("sample.png", _png_bytes(), "image/png")},
        )
        assert client.get("/v1/inspections").json()["returned_count"] == 0
        assert client.get("/v1/known-defects").json()["returned_count"] == 0

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "persistence_unavailable"
    assert inspection_broadcaster.events == known_broadcaster.events == []
    assert combined_broadcaster.events == []


# ADD 2026-08-26: Disabled YOLO는 combined만 unavailable이고 PatchCore endpoint는 유지됨을 검증한다.
def test_combined_disabled_yolo_does_not_disable_patchcore(tmp_path: Path) -> None:
    app = create_app(
        settings=_settings(tmp_path, yolo_enabled=False), runtime_loader=_patchcore_loader
    )

    with TestClient(app) as client:
        combined = client.post(
            "/v1/combined-inspections",
            files={"image": ("sample.png", _png_bytes(), "image/png")},
        )
        patchcore = client.post(
            "/v1/predictions",
            files={"image": ("sample.png", _png_bytes(), "image/png")},
        )

    assert combined.status_code == 503
    assert combined.json()["error"]["code"] == "known_defect_model_disabled"
    assert patchcore.status_code == 200
