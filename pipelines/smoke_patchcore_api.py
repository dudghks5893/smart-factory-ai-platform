"""Smoke-test a real PatchCore artifact through the FastAPI HTTP boundary."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from fastapi.testclient import TestClient

from ml.training.device import SUPPORTED_DEVICES
from services.api.app import RuntimeLoader, create_app
from services.api.config import (
    DEFAULT_MAX_UPLOAD_BYTES,
    ServingSettings,
    required_database_url,
)
from services.api.schemas import HealthResponse, ReadinessResponse
from services.api.tooling import prepare_image_upload, validate_prediction_payload
from services.inference.runtime import load_patchcore_runtime


@dataclass(frozen=True)
class SmokeOutputSummary:
    """Validated normal and anomaly results from one API lifecycle."""

    model_name: str
    category: str
    device: str
    threshold: float
    normal_score: float
    anomaly_score: float


# ADD 2026-08-20: Real artifact lifecycle과 normal/anomaly HTTP 계약을 smoke 검증한다.
# MODIFY 2026-08-20: Required inspection database를 같은 FastAPI lifecycle에 연결한다.
def smoke_patchcore_api(
    *,
    artifact_dir: Path,
    thresholds_path: Path,
    normal_image_path: Path,
    anomaly_image_path: Path,
    database_url: str,
    requested_device: str = "auto",
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
    runtime_loader: RuntimeLoader = load_patchcore_runtime,
) -> SmokeOutputSummary:
    """Exercise startup, health, readiness, and two HTTP inference requests."""
    settings = ServingSettings(
        artifact_dir=artifact_dir,
        thresholds_path=thresholds_path,
        database_url=database_url,
        model_device=requested_device,
        max_upload_bytes=max_upload_bytes,
    )
    settings.validate()

    # HTTP timing과 무관한 smoke input disk loading을 application startup 전에 완료한다.
    normal_upload = prepare_image_upload(
        normal_image_path,
        max_upload_bytes=max_upload_bytes,
    )
    anomaly_upload = prepare_image_upload(
        anomaly_image_path,
        max_upload_bytes=max_upload_bytes,
    )
    app = create_app(settings=settings, runtime_loader=runtime_loader)

    # Lifespan에서 artifact를 한 번 복원한 뒤 실제 FastAPI route로 두 image를 요청한다.
    with TestClient(app) as client:
        health = client.get("/health")
        _require_success(health.status_code, "/health", health.text)
        HealthResponse.model_validate(health.json())

        ready = client.get("/ready")
        _require_success(ready.status_code, "/ready", ready.text)
        readiness = ReadinessResponse.model_validate(ready.json())

        normal_response = client.post(
            "/v1/predictions",
            files={"image": normal_upload.as_multipart_file()},
        )
        _require_success(normal_response.status_code, "normal prediction", normal_response.text)
        normal = validate_prediction_payload(
            normal_response.json(),
            expected_is_anomaly=False,
        )

        anomaly_response = client.post(
            "/v1/predictions",
            files={"image": anomaly_upload.as_multipart_file()},
        )
        _require_success(
            anomaly_response.status_code,
            "anomaly prediction",
            anomaly_response.text,
        )
        anomaly = validate_prediction_payload(
            anomaly_response.json(),
            expected_is_anomaly=True,
        )

    if (normal.model_name, normal.category) != (readiness.model_name, readiness.category):
        raise ValueError("Normal prediction identity does not match readiness metadata.")
    if (anomaly.model_name, anomaly.category) != (readiness.model_name, readiness.category):
        raise ValueError("Anomaly prediction identity does not match readiness metadata.")
    if anomaly.threshold != normal.threshold:
        raise ValueError("Normal and anomaly responses returned different thresholds.")

    return SmokeOutputSummary(
        model_name=readiness.model_name,
        category=readiness.category,
        device=readiness.device,
        threshold=normal.threshold,
        normal_score=normal.anomaly_score,
        anomaly_score=anomaly.anomaly_score,
    )


# ADD 2026-08-20: HTTP failure body를 포함한 actionable smoke error로 변환한다.
def _require_success(status_code: int, operation: str, response_text: str) -> None:
    if status_code != 200:
        raise RuntimeError(f"{operation} failed with HTTP {status_code}: {response_text}")


# ADD 2026-08-20: Real PatchCore API smoke CLI 인자를 정의한다.
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the PatchCore FastAPI application.")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--normal-image", type=Path, required=True)
    parser.add_argument("--anomaly-image", type=Path, required=True)
    parser.add_argument("--device", choices=SUPPORTED_DEVICES, default="auto")
    parser.add_argument("--max-upload-bytes", type=int, default=DEFAULT_MAX_UPLOAD_BYTES)
    return parser.parse_args()


# ADD 2026-08-20: CLI smoke를 실행하고 검증된 model, threshold와 score를 출력한다.
def main() -> int:
    args = _parse_args()
    summary = smoke_patchcore_api(
        artifact_dir=args.artifact_dir,
        thresholds_path=args.thresholds,
        normal_image_path=args.normal_image,
        anomaly_image_path=args.anomaly_image,
        database_url=required_database_url(),
        requested_device=args.device,
        max_upload_bytes=args.max_upload_bytes,
    )
    print("PatchCore FastAPI smoke: PASS")
    print(f"Model/category/device: {summary.model_name}/{summary.category}/{summary.device}")
    print(f"Threshold: {summary.threshold}")
    print(f"Normal score: {summary.normal_score}")
    print(f"Anomaly score: {summary.anomaly_score}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
