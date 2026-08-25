"""Smoke-test a real YOLO segmentation artifact through the FastAPI HTTP boundary."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

from fastapi.testclient import TestClient

from ml.training.device import SUPPORTED_DEVICES
from pipelines.smoke_yolo_segmentation_runtime import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_DIAGNOSTIC_CONFIDENCE,
    resolve_default_smoke_images,
)
from services.api.app import RuntimeLoader, YoloRuntimeLoader, create_app
from services.api.config import (
    DEFAULT_MAX_UPLOAD_BYTES,
    ServingSettings,
    required_database_url,
)
from services.api.schemas import HealthResponse, KnownDefectResponse, ReadinessResponse
from services.api.tooling import prepare_image_upload
from services.inference.runtime import load_patchcore_runtime
from services.inference.yolo_segmentation_runtime import (
    load_yolo_segmentation_runtime,
    validate_diagnostic_confidence,
)

DEFAULT_OUTPUT_PATH = Path("outputs/analysis/yolo_segmentation/api_smoke/smoke_summary.json")


@dataclass(frozen=True)
class KnownDefectApiInstanceSummary:
    """One compact API instance including spatial evidence but no raw mask payload."""

    class_id: int
    class_name: str
    confidence: float
    box: tuple[float, float, float, float]
    mask_pixel_count: int
    mask_area_ratio: float


@dataclass(frozen=True)
class KnownDefectApiImageSummary:
    """One actual multipart response and application-level timing observation."""

    source: str
    status_code: int
    device: str
    expected_class: str | None
    expected_class_detected: bool | None
    instance_count: int
    classes: tuple[str, ...]
    confidences: tuple[float, ...]
    instances: tuple[KnownDefectApiInstanceSummary, ...]
    inference_ms: float
    http_application_ms: float


@dataclass(frozen=True)
class KnownDefectApiSmokeSummary:
    """One FastAPI lifecycle with actual YOLO model identity and image responses."""

    endpoint: str
    transport: str
    model_name: str
    category: str
    requested_device: str
    actual_device: str
    diagnostic_confidence: float
    model_sha256: str
    images: tuple[KnownDefectApiImageSummary, ...]
    output_path: str


# ADD 2026-08-26: Positive parent directory를 optional expected-class observation으로 변환한다.
def _expected_class(image_path: Path) -> str | None:
    return (
        image_path.parent.name if image_path.parent.name in {"bent", "color", "scratch"} else None
    )


# ADD 2026-08-26: HTTP status failure를 safe response body와 함께 actionable smoke error로 변환한다.
def _require_success(status_code: int, operation: str, response_text: str) -> None:
    if status_code != 200:
        raise RuntimeError(f"{operation} failed with HTTP {status_code}: {response_text}")


# ADD 2026-08-26: Real FastAPI lifespan에서 YOLO singleton과 multi-image multipart smoke를 조율한다.
def smoke_yolo_segmentation_api(
    *,
    patchcore_artifact_dir: Path,
    patchcore_thresholds_path: Path,
    yolo_artifact_dir: Path,
    image_paths: list[Path],
    database_url: str,
    yolo_requested_device: str,
    diagnostic_confidence: float,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    patchcore_requested_device: str = "cpu",
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
    runtime_loader: RuntimeLoader = load_patchcore_runtime,
    yolo_runtime_loader: YoloRuntimeLoader = load_yolo_segmentation_runtime,
) -> KnownDefectApiSmokeSummary:
    """Exercise startup/readiness and actual known-defect HTTP responses without persistence."""
    validate_diagnostic_confidence(diagnostic_confidence)
    if not image_paths:
        raise ValueError("YOLO API smoke requires at least one image.")
    uploads = [
        prepare_image_upload(image_path, max_upload_bytes=max_upload_bytes)
        for image_path in image_paths
    ]
    settings = ServingSettings(
        artifact_dir=patchcore_artifact_dir,
        thresholds_path=patchcore_thresholds_path,
        database_url=database_url,
        model_device=patchcore_requested_device,
        max_upload_bytes=max_upload_bytes,
        yolo_segmentation_enabled=True,
        yolo_segmentation_artifact_dir=yolo_artifact_dir,
        yolo_segmentation_device=yolo_requested_device,
        yolo_segmentation_diagnostic_confidence=diagnostic_confidence,
    )
    settings.validate()
    app = create_app(
        settings=settings,
        runtime_loader=runtime_loader,
        yolo_runtime_loader=yolo_runtime_loader,
    )

    image_summaries: list[KnownDefectApiImageSummary] = []
    model_name = ""
    category = ""
    actual_device = ""
    model_sha256 = ""
    # Full lifespan에서 model별 singleton을 한 번 복원하고 실제 multipart route를 호출한다.
    with TestClient(app) as client:
        health = client.get("/health")
        _require_success(health.status_code, "/health", health.text)
        HealthResponse.model_validate(health.json())
        ready = client.get("/ready")
        _require_success(ready.status_code, "/ready", ready.text)
        ReadinessResponse.model_validate(ready.json())
        runtime = app.state.yolo_segmentation_runtime
        model_sha256 = runtime.provenance.model_sha256

        for image_path, upload in zip(image_paths, uploads, strict=True):
            # Disk read는 timing 전에 완료하고 multipart application boundary만 관찰한다.
            started = perf_counter()
            response = client.post(
                "/v1/known-defects",
                files={"image": upload.as_multipart_file()},
            )
            http_application_ms = (perf_counter() - started) * 1000.0
            _require_success(response.status_code, str(image_path), response.text)
            payload = KnownDefectResponse.model_validate(response.json())
            if payload.diagnostic_confidence != diagnostic_confidence:
                raise ValueError("API response diagnostic confidence changed unexpectedly.")
            model_name = payload.model.name
            category = payload.model.category
            actual_device = payload.model.device
            expected_class = _expected_class(image_path)
            image_summaries.append(
                KnownDefectApiImageSummary(
                    source=str(image_path),
                    status_code=response.status_code,
                    device=payload.model.device,
                    expected_class=expected_class,
                    expected_class_detected=(
                        None
                        if expected_class is None
                        else any(
                            instance.class_name == expected_class for instance in payload.instances
                        )
                    ),
                    instance_count=len(payload.instances),
                    classes=tuple(instance.class_name for instance in payload.instances),
                    confidences=tuple(instance.confidence for instance in payload.instances),
                    instances=tuple(
                        KnownDefectApiInstanceSummary(
                            class_id=instance.class_id,
                            class_name=instance.class_name,
                            confidence=instance.confidence,
                            box=(
                                instance.box.x_min,
                                instance.box.y_min,
                                instance.box.x_max,
                                instance.box.y_max,
                            ),
                            mask_pixel_count=instance.mask.pixel_count,
                            mask_area_ratio=instance.mask.area_ratio,
                        )
                        for instance in payload.instances
                    ),
                    inference_ms=payload.inference_ms,
                    http_application_ms=http_application_ms,
                )
            )
    summary = KnownDefectApiSmokeSummary(
        endpoint="POST /v1/known-defects",
        transport="in_process_asgi_testclient",
        model_name=model_name,
        category=category,
        requested_device=yolo_requested_device,
        actual_device=actual_device,
        diagnostic_confidence=diagnostic_confidence,
        model_sha256=model_sha256,
        images=tuple(image_summaries),
        output_path=str(output_path),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


# ADD 2026-08-26: Actual API response와 inference/HTTP timing 경계를 readable text로 출력한다.
def format_yolo_api_smoke_summary(summary: KnownDefectApiSmokeSummary) -> str:
    lines = [
        "YOLO segmentation FastAPI smoke: COMPLETE",
        f"Endpoint: {summary.endpoint}",
        f"Transport: {summary.transport}",
        f"Model/category: {summary.model_name}/{summary.category}",
        f"Requested/actual device: {summary.requested_device}/{summary.actual_device}",
        f"Model SHA256: {summary.model_sha256}",
        f"Diagnostic confidence: {summary.diagnostic_confidence} (not production threshold)",
    ]
    for image in summary.images:
        lines.extend(
            (
                "",
                f"IMAGE: {image.source}",
                f"HTTP status: {image.status_code}",
                f"Device: {image.device}",
                f"Instances: {image.instance_count}",
                f"Classes: {image.classes}",
                f"Confidences: {image.confidences}",
                f"Expected class detected: {image.expected_class_detected}",
                f"Inference: {image.inference_ms:.6f} ms",
                f"HTTP application E2E: {image.http_application_ms:.6f} ms",
            )
        )
        for instance in image.instances:
            lines.append(
                "Instance: "
                f"{instance.class_name} confidence={instance.confidence:.9f} "
                f"box={instance.box} mask_pixels={instance.mask_pixel_count} "
                f"mask_area_ratio={instance.mask_area_ratio:.9f}"
            )
    lines.extend(("", f"Summary JSON: {summary.output_path}"))
    return "\n".join(lines)


# ADD 2026-08-26: Diagnostic confidence CLI value를 bounded float로 변환한다.
def _diagnostic_confidence(value: str) -> float:
    try:
        parsed = float(value)
        validate_diagnostic_confidence(parsed)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return parsed


# ADD 2026-08-26: Real PatchCore/YOLO artifact와 local image/API settings arguments를 정의한다.
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patchcore-artifact-dir", type=Path, required=True)
    parser.add_argument("--patchcore-thresholds", type=Path, required=True)
    parser.add_argument("--yolo-artifact-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--image", type=Path, action="append")
    parser.add_argument("--yolo-device", choices=SUPPORTED_DEVICES, default="mps")
    parser.add_argument("--patchcore-device", choices=SUPPORTED_DEVICES, default="cpu")
    parser.add_argument(
        "--confidence",
        type=_diagnostic_confidence,
        default=DEFAULT_DIAGNOSTIC_CONFIDENCE,
        help="Diagnostic confidence only; this is not a production threshold.",
    )
    parser.add_argument("--max-upload-bytes", type=int, default=DEFAULT_MAX_UPLOAD_BYTES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


# ADD 2026-08-26: CLI preset image와 required database를 actual FastAPI smoke로 조율한다.
def main() -> int:
    args = _parse_args()
    image_paths = args.image or resolve_default_smoke_images(args.dataset_root)
    summary = smoke_yolo_segmentation_api(
        patchcore_artifact_dir=args.patchcore_artifact_dir,
        patchcore_thresholds_path=args.patchcore_thresholds,
        yolo_artifact_dir=args.yolo_artifact_dir,
        image_paths=image_paths,
        database_url=required_database_url(),
        yolo_requested_device=args.yolo_device,
        diagnostic_confidence=args.confidence,
        output_path=args.output,
        patchcore_requested_device=args.patchcore_device,
        max_upload_bytes=args.max_upload_bytes,
    )
    print(format_yolo_api_smoke_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
