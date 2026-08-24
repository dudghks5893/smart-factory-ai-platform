"""Smoke-test one image through the production PatchCore inference runtime."""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ml.training.device import SUPPORTED_DEVICES
from services.api.config import DEFAULT_MAX_UPLOAD_BYTES
from services.api.images import decode_uploaded_image
from services.api.tooling import prepare_image_upload
from services.inference.runtime import (
    InferenceResult,
    ModelRuntime,
    PatchCoreRuntimeConfig,
    load_patchcore_runtime,
)

type RuntimeLoader = Callable[[PatchCoreRuntimeConfig], ModelRuntime]
type SmokeLabel = Literal["NORMAL", "ANOMALY"]


@dataclass(frozen=True)
class RuntimeSmokeSummary:
    """Validated identity, device, score and decision for one local image smoke."""

    model_name: str
    category: str
    device: str
    image_path: Path
    anomaly_score: float
    image_threshold: float
    comparison_operator: str
    is_anomaly: bool
    result: SmokeLabel


# ADD 2026-08-24: Existing serving runtime으로 단일 local image의 pure inference를 검증한다.
def smoke_patchcore_runtime(
    *,
    artifact_dir: Path,
    thresholds_path: Path,
    image_path: Path,
    requested_device: str,
    runtime_loader: RuntimeLoader = load_patchcore_runtime,
) -> RuntimeSmokeSummary:
    """Decode one image, restore validated artifacts, and return the strict-threshold result."""
    # 비용이 큰 artifact restore 전에 image path, suffix, size와 decode 가능 여부를 검증한다.
    upload = prepare_image_upload(image_path, max_upload_bytes=DEFAULT_MAX_UPLOAD_BYTES)
    image = decode_uploaded_image(
        upload.content,
        content_type=upload.content_type,
        max_upload_bytes=DEFAULT_MAX_UPLOAD_BYTES,
    )

    # Production serving과 같은 loader로 lineage를 검증하고 model/preprocessing을 복원한다.
    runtime = runtime_loader(
        PatchCoreRuntimeConfig(
            artifact_dir=artifact_dir,
            thresholds_path=thresholds_path,
            device=requested_device,
        )
    )

    # 동일 runtime inference와 strict validation threshold 판정을 수행한다.
    prediction = runtime.predict(image)
    _validate_prediction(prediction)
    return RuntimeSmokeSummary(
        model_name=prediction.model_name,
        category=prediction.category,
        device=runtime.device,
        image_path=image_path,
        anomaly_score=prediction.anomaly_score,
        image_threshold=prediction.threshold,
        comparison_operator=prediction.comparison_operator,
        is_anomaly=prediction.is_anomaly,
        result="ANOMALY" if prediction.is_anomaly else "NORMAL",
    )


# ADD 2026-08-24: Runtime 결과의 finite score와 strict comparison consistency를 검증한다.
def _validate_prediction(prediction: InferenceResult) -> None:
    if not math.isfinite(prediction.anomaly_score) or not math.isfinite(prediction.threshold):
        raise ValueError("PatchCore runtime smoke score and threshold must be finite.")
    if prediction.comparison_operator != ">":
        raise ValueError("PatchCore runtime smoke requires comparison_operator='>'.")
    if prediction.is_anomaly is not (prediction.anomaly_score > prediction.threshold):
        raise ValueError("PatchCore runtime result violates the strict score > threshold contract.")


# ADD 2026-08-24: Local smoke 결과를 stable human-readable CLI output으로 변환한다.
def format_runtime_smoke_summary(summary: RuntimeSmokeSummary) -> str:
    """Render the complete smoke contract without exposing model internals."""
    return "\n".join(
        (
            "PatchCore runtime smoke: PASS",
            f"Model: {summary.model_name}",
            f"Category: {summary.category}",
            f"Device: {summary.device}",
            f"Image: {summary.image_path}",
            f"Score: {summary.anomaly_score}",
            f"Threshold: {summary.image_threshold}",
            f"Comparison: score {summary.comparison_operator} threshold",
            f"Is anomaly: {str(summary.is_anomaly).lower()}",
            f"Result: {summary.result}",
        )
    )


# ADD 2026-08-24: Pure runtime smoke CLI의 required path와 explicit device 인자를 정의한다.
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test one image through the validated PatchCore runtime."
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", choices=SUPPORTED_DEVICES, default="auto")
    return parser.parse_args()


# ADD 2026-08-24: CLI 입력을 production runtime smoke와 출력 contract로 조율한다.
def main() -> int:
    args = _parse_args()
    summary = smoke_patchcore_runtime(
        artifact_dir=args.artifact_dir,
        thresholds_path=args.thresholds,
        image_path=args.image,
        requested_device=args.device,
    )
    print(format_runtime_smoke_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
