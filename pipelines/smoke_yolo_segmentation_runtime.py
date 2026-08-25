"""Smoke-test a validated YOLO segmentation artifact on local MVTec AD images."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from ml.training.device import SUPPORTED_DEVICES
from services.inference.yolo_segmentation_runtime import (
    YoloSegmentationAdapter,
    YoloSegmentationInstance,
    YoloSegmentationResult,
    YoloSegmentationRuntimeConfig,
    load_yolo_segmentation_runtime,
    validate_diagnostic_confidence,
)

DEFAULT_DATASET_ROOT = Path("data/raw/mvtec_ad/metal_nut")
DEFAULT_OUTPUT_DIR = Path("outputs/analysis/yolo_segmentation/runtime_smoke")
DEFAULT_DIAGNOSTIC_CONFIDENCE = 0.25
SMOKE_DEFECT_TYPES = ("good", "bent", "color", "scratch")
CLASS_COLORS = {
    "bent": np.array([239, 68, 68], dtype=np.float32),
    "color": np.array([59, 130, 246], dtype=np.float32),
    "scratch": np.array([34, 197, 94], dtype=np.float32),
}

type RuntimeLoader = Callable[[YoloSegmentationRuntimeConfig], YoloSegmentationAdapter]


@dataclass(frozen=True)
class SmokeInstanceSummary:
    """Serializable instance evidence without a giant nested mask payload."""

    class_id: int
    class_name: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]
    mask_pixel_count: int
    mask_area_ratio: float


@dataclass(frozen=True)
class ImageSmokeSummary:
    """Runtime and model-quality observations for one actual image."""

    source: str
    expected_class: str | None
    expected_class_detected: bool | None
    device: str
    latency_phase: str
    inference_ms: float
    image_width: int
    image_height: int
    predicted_instance_count: int
    instances: tuple[SmokeInstanceSummary, ...]
    visualization: str | None


@dataclass(frozen=True)
class DeviceComparisonSummary:
    """Non-gating semantic and numeric deltas between two device results."""

    source: str
    primary_device: str
    reference_device: str
    primary_class_set: tuple[str, ...]
    reference_class_set: tuple[str, ...]
    class_sets_equal: bool
    primary_instance_count: int
    reference_instance_count: int
    instance_counts_equal: bool
    matched_instance_count: int
    max_confidence_abs_delta: float | None
    max_bbox_abs_delta_pixels: float | None
    min_mask_iou: float | None


@dataclass(frozen=True)
class RuntimeSmokeSuite:
    """Complete one- or two-device smoke evidence and artifact identity."""

    artifact_dir: str
    diagnostic_confidence: float
    diagnostic_confidence_policy: str
    requested_device: str
    actual_device: str
    model_load_ms: float
    model_sha256: str
    dataset_manifest_sha256: str
    dataset_semantic_fingerprint_sha256: str
    images: tuple[ImageSmokeSummary, ...]
    comparison_requested_device: str | None
    comparison_actual_device: str | None
    comparison_model_load_ms: float | None
    comparison_images: tuple[ImageSmokeSummary, ...]
    comparisons: tuple[DeviceComparisonSummary, ...]
    summary_path: str


# ADD 2026-08-26: MVTec root에서 실제 good/bent/color/scratch smoke image를 검증해 선택한다.
def resolve_default_smoke_images(dataset_root: Path) -> list[Path]:
    paths = [dataset_root / "test" / defect_type / "000.png" for defect_type in SMOKE_DEFECT_TYPES]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"YOLO runtime smoke image not found: {missing[0]}")
    return paths


# ADD 2026-08-26: Diagnostic confidence CLI value를 argparse-compatible validator로 변환한다.
def _diagnostic_confidence(value: str) -> float:
    try:
        parsed = float(value)
        validate_diagnostic_confidence(parsed)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return parsed


# ADD 2026-08-26: Actual image를 한 장씩 RGB uint8 array로 읽고 dataset preload를 방지한다.
def _load_rgb_image(path: Path) -> NDArray[np.uint8]:
    if not path.is_file():
        raise FileNotFoundError(f"YOLO runtime smoke image not found: {path}")
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    except OSError as exc:
        raise ValueError(f"Cannot decode YOLO runtime smoke image: {path}") from exc


# ADD 2026-08-26: Parent defect directory에서 optional expected-class observation을 추론한다.
def _expected_class(path: Path) -> str | None:
    return path.parent.name if path.parent.name in {"bent", "color", "scratch"} else None


# ADD 2026-08-26: Runtime instance를 mask summary만 포함하는 audit-friendly record로 축약한다.
def _summarize_instance(instance: YoloSegmentationInstance) -> SmokeInstanceSummary:
    return SmokeInstanceSummary(
        class_id=instance.class_id,
        class_name=instance.class_name,
        confidence=instance.confidence,
        box_xyxy=instance.box_xyxy,
        mask_pixel_count=instance.mask_pixel_count,
        mask_area_ratio=instance.mask_area_ratio,
    )


# ADD 2026-08-26: Runtime result를 accuracy claim 없는 smoke observation으로 변환한다.
def _summarize_result(
    *,
    image_path: Path,
    result: YoloSegmentationResult,
    is_cold_start: bool,
    visualization_path: Path | None,
) -> ImageSmokeSummary:
    expected_class = _expected_class(image_path)
    return ImageSmokeSummary(
        source=str(image_path),
        expected_class=expected_class,
        expected_class_detected=(
            None
            if expected_class is None
            else any(instance.class_name == expected_class for instance in result.instances)
        ),
        device=result.device,
        latency_phase="cold_first_inference" if is_cold_start else "subsequent_inference",
        inference_ms=result.inference_ms,
        image_width=result.image_width,
        image_height=result.image_height,
        predicted_instance_count=len(result.instances),
        instances=tuple(_summarize_instance(instance) for instance in result.instances),
        visualization=str(visualization_path) if visualization_path is not None else None,
    )


# ADD 2026-08-26: Original image 위에 predicted mask/bbox/class/confidence를 render한다.
def save_runtime_visualization(
    *,
    image_rgb: NDArray[np.uint8],
    result: YoloSegmentationResult,
    output_path: Path,
) -> None:
    overlay = image_rgb.astype(np.float32, copy=True)
    for instance in result.instances:
        color = CLASS_COLORS[instance.class_name]
        overlay[instance.mask] = overlay[instance.mask] * 0.55 + color * 0.45
    rendered = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(rendered)
    for instance in result.instances:
        draw_color = tuple(int(value) for value in CLASS_COLORS[instance.class_name])
        draw.rectangle(instance.box_xyxy, outline=draw_color, width=3)
        x1, y1, _, _ = instance.box_xyxy
        draw.text(
            (max(0.0, x1), max(0.0, y1 - 12.0)),
            f"{instance.class_name} {instance.confidence:.4f}",
            fill=draw_color,
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(output_path)


# ADD 2026-08-26: Two binary masks의 overlap을 empty-safe IoU로 계산한다.
def _mask_iou(left: NDArray[np.bool_], right: NDArray[np.bool_]) -> float:
    if left.shape != right.shape:
        raise ValueError("Device comparison masks must have the same shape.")
    union = np.logical_or(left, right).sum()
    return 1.0 if union == 0 else float(np.logical_and(left, right).sum() / union)


# ADD 2026-08-26: Same-class mask IoU로 instance를 대응시켜 device numeric delta를 관찰한다.
def compare_device_results(
    *,
    source: Path,
    primary: YoloSegmentationResult,
    reference: YoloSegmentationResult,
) -> DeviceComparisonSummary:
    if (primary.image_width, primary.image_height) != (
        reference.image_width,
        reference.image_height,
    ):
        raise ValueError("Device comparison image dimensions do not match.")
    available = set(range(len(reference.instances)))
    matched: list[tuple[YoloSegmentationInstance, YoloSegmentationInstance, float]] = []
    for primary_instance in primary.instances:
        candidates = [
            index
            for index in available
            if reference.instances[index].class_id == primary_instance.class_id
        ]
        if not candidates:
            continue
        best_index = max(
            candidates,
            key=lambda index: _mask_iou(primary_instance.mask, reference.instances[index].mask),
        )
        reference_instance = reference.instances[best_index]
        matched.append(
            (
                primary_instance,
                reference_instance,
                _mask_iou(primary_instance.mask, reference_instance.mask),
            )
        )
        available.remove(best_index)
    confidence_deltas = [abs(left.confidence - right.confidence) for left, right, _ in matched]
    bbox_deltas = [
        max(
            abs(left_value - right_value)
            for left_value, right_value in zip(
                left.box_xyxy,
                right.box_xyxy,
                strict=True,
            )
        )
        for left, right, _ in matched
    ]
    mask_ious = [iou for _, _, iou in matched]
    primary_class_set = tuple(sorted({instance.class_name for instance in primary.instances}))
    reference_class_set = tuple(sorted({instance.class_name for instance in reference.instances}))
    return DeviceComparisonSummary(
        source=str(source),
        primary_device=primary.device,
        reference_device=reference.device,
        primary_class_set=primary_class_set,
        reference_class_set=reference_class_set,
        class_sets_equal=primary_class_set == reference_class_set,
        primary_instance_count=len(primary.instances),
        reference_instance_count=len(reference.instances),
        instance_counts_equal=len(primary.instances) == len(reference.instances),
        matched_instance_count=len(matched),
        max_confidence_abs_delta=max(confidence_deltas) if confidence_deltas else None,
        max_bbox_abs_delta_pixels=max(bbox_deltas) if bbox_deltas else None,
        min_mask_iou=min(mask_ious) if mask_ious else None,
    )


# ADD 2026-08-26: Artifact restore를 측정하되 image별 model reuse를 위해 adapter를 한 번만 반환한다.
def _load_runtime(
    *,
    artifact_dir: Path,
    requested_device: str,
    runtime_loader: RuntimeLoader,
) -> tuple[YoloSegmentationAdapter, float]:
    started = perf_counter()
    runtime = runtime_loader(
        YoloSegmentationRuntimeConfig(artifact_dir=artifact_dir, device=requested_device)
    )
    return runtime, (perf_counter() - started) * 1000.0


# ADD 2026-08-26: Device별 reusable runtime으로 multi-image smoke와 comparison을 조율한다.
def smoke_yolo_segmentation_runtime(
    *,
    artifact_dir: Path,
    image_paths: list[Path],
    requested_device: str,
    diagnostic_confidence: float,
    output_dir: Path,
    comparison_device: str | None = None,
    save_visualizations: bool = True,
    runtime_loader: RuntimeLoader = load_yolo_segmentation_runtime,
) -> RuntimeSmokeSuite:
    """Run local development smoke without classifying four samples as an accuracy metric."""
    validate_diagnostic_confidence(diagnostic_confidence)
    if not image_paths:
        raise ValueError("YOLO runtime smoke requires at least one image.")
    if len(set(image_paths)) != len(image_paths):
        raise ValueError("YOLO runtime smoke image paths must be unique.")
    if comparison_device == requested_device:
        raise ValueError("Comparison device must differ from the primary requested device.")

    # Artifact를 device별 한 번만 복원하고 모든 image에서 같은 model instance를 재사용한다.
    runtime, model_load_ms = _load_runtime(
        artifact_dir=artifact_dir,
        requested_device=requested_device,
        runtime_loader=runtime_loader,
    )
    comparison_runtime: YoloSegmentationAdapter | None = None
    comparison_model_load_ms: float | None = None
    if comparison_device is not None:
        comparison_runtime, comparison_model_load_ms = _load_runtime(
            artifact_dir=artifact_dir,
            requested_device=comparison_device,
            runtime_loader=runtime_loader,
        )
        if comparison_runtime.provenance.model_sha256 != runtime.provenance.model_sha256:
            raise ValueError("Device comparison runtimes did not load the same checkpoint.")

    image_summaries: list[ImageSmokeSummary] = []
    comparison_image_summaries: list[ImageSmokeSummary] = []
    comparisons: list[DeviceComparisonSummary] = []
    for index, image_path in enumerate(image_paths):
        # 각 source를 필요한 시점에만 읽어 dataset 전체 preload를 피한다.
        image_rgb = _load_rgb_image(image_path)
        result = runtime.predict(
            image_rgb,
            diagnostic_confidence=diagnostic_confidence,
        )
        visualization_path = (
            output_dir / f"{image_path.parent.name}_{runtime.device}.png"
            if save_visualizations
            else None
        )
        if visualization_path is not None:
            save_runtime_visualization(
                image_rgb=image_rgb,
                result=result,
                output_path=visualization_path,
            )
        image_summaries.append(
            _summarize_result(
                image_path=image_path,
                result=result,
                is_cold_start=index == 0,
                visualization_path=visualization_path,
            )
        )
        if comparison_runtime is not None:
            comparison_result = comparison_runtime.predict(
                image_rgb,
                diagnostic_confidence=diagnostic_confidence,
            )
            comparison_visualization_path = (
                output_dir / f"{image_path.parent.name}_{comparison_runtime.device}.png"
                if save_visualizations
                else None
            )
            if comparison_visualization_path is not None:
                save_runtime_visualization(
                    image_rgb=image_rgb,
                    result=comparison_result,
                    output_path=comparison_visualization_path,
                )
            comparison_image_summaries.append(
                _summarize_result(
                    image_path=image_path,
                    result=comparison_result,
                    is_cold_start=index == 0,
                    visualization_path=comparison_visualization_path,
                )
            )
            comparisons.append(
                compare_device_results(
                    source=image_path,
                    primary=result,
                    reference=comparison_result,
                )
            )

    # Giant mask arrays를 제외한 smoke evidence만 ignored output JSON에 기록한다.
    summary_path = output_dir / "smoke_summary.json"
    suite = RuntimeSmokeSuite(
        artifact_dir=str(artifact_dir),
        diagnostic_confidence=diagnostic_confidence,
        diagnostic_confidence_policy="c2_2_test_diagnostic_operating_point_not_production",
        requested_device=requested_device,
        actual_device=runtime.device,
        model_load_ms=model_load_ms,
        model_sha256=runtime.provenance.model_sha256,
        dataset_manifest_sha256=runtime.provenance.dataset_manifest_sha256,
        dataset_semantic_fingerprint_sha256=(
            runtime.provenance.dataset_semantic_fingerprint_sha256
        ),
        images=tuple(image_summaries),
        comparison_requested_device=comparison_device,
        comparison_actual_device=(comparison_runtime.device if comparison_runtime else None),
        comparison_model_load_ms=comparison_model_load_ms,
        comparison_images=tuple(comparison_image_summaries),
        comparisons=tuple(comparisons),
        summary_path=str(summary_path),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(asdict(suite), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return suite


# ADD 2026-08-26: Multi-image runtime과 device comparison을 readable text로 출력한다.
def format_runtime_smoke_suite(suite: RuntimeSmokeSuite) -> str:
    lines = [
        "YOLO segmentation runtime smoke: COMPLETE",
        f"Artifact: {suite.artifact_dir}",
        f"Model SHA256: {suite.model_sha256}",
        f"Diagnostic confidence: {suite.diagnostic_confidence} (not production threshold)",
        f"Requested device: {suite.requested_device}",
        f"Actual device: {suite.actual_device}",
        f"Model load: {suite.model_load_ms:.3f} ms",
    ]
    for summary in (*suite.images, *suite.comparison_images):
        lines.extend(
            (
                "",
                f"IMAGE: {summary.source}",
                f"device={summary.device}",
                f"latency_phase={summary.latency_phase}",
                f"inference_ms={summary.inference_ms:.6f}",
                f"instances={summary.predicted_instance_count}",
                f"expected_class={summary.expected_class or 'none'}",
                f"expected_class_detected={summary.expected_class_detected}",
            )
        )
        for index, instance in enumerate(summary.instances):
            lines.extend(
                (
                    f"[{index}] class={instance.class_name} id={instance.class_id}",
                    f"[{index}] confidence={instance.confidence:.10f}",
                    f"[{index}] bbox={instance.box_xyxy}",
                    f"[{index}] mask_pixels={instance.mask_pixel_count}",
                    f"[{index}] mask_area_ratio={instance.mask_area_ratio:.10f}",
                )
            )
    for comparison in suite.comparisons:
        lines.extend(
            (
                "",
                f"COMPARISON: {comparison.source}",
                f"class_sets_equal={comparison.class_sets_equal}",
                f"instance_counts_equal={comparison.instance_counts_equal}",
                f"matched_instances={comparison.matched_instance_count}",
                f"max_confidence_abs_delta={comparison.max_confidence_abs_delta}",
                f"max_bbox_abs_delta_pixels={comparison.max_bbox_abs_delta_pixels}",
                f"min_mask_iou={comparison.min_mask_iou}",
            )
        )
    lines.extend(("", f"Summary JSON: {suite.summary_path}"))
    return "\n".join(lines)


# ADD 2026-08-26: Local MPS/CPU smoke의 artifact, image, confidence와 output arguments를 정의한다.
def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--image", type=Path, action="append")
    parser.add_argument("--device", choices=SUPPORTED_DEVICES, default="mps")
    parser.add_argument("--compare-device", choices=SUPPORTED_DEVICES)
    parser.add_argument(
        "--confidence",
        type=_diagnostic_confidence,
        default=DEFAULT_DIAGNOSTIC_CONFIDENCE,
        help="Diagnostic confidence only; this is not a production threshold.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-visualizations", action="store_true")
    return parser.parse_args(argv)


# ADD 2026-08-26: CLI image를 smoke flow와 human-readable output으로 조율한다.
def main() -> int:
    args = _parse_args()
    image_paths = args.image or resolve_default_smoke_images(args.dataset_root)
    suite = smoke_yolo_segmentation_runtime(
        artifact_dir=args.artifact_dir,
        image_paths=image_paths,
        requested_device=args.device,
        comparison_device=args.compare_device,
        diagnostic_confidence=args.confidence,
        output_dir=args.output_dir,
        save_visualizations=not args.no_visualizations,
    )
    print(format_runtime_smoke_suite(suite))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
