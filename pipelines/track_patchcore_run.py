"""Backfill project-native PatchCore artifacts into one canonical MLflow run."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from ml.tracking.patchcore import (
    DEFAULT_TRACKING_ROOT,
    PatchCoreTrackingInputs,
    build_tracking_pointer,
    prepare_patchcore_tracking,
    validate_tracking_id,
    write_tracking_pointer,
)
from services.tracking.mlflow import (
    LoggedRun,
    MlflowTrackingAdapter,
    MlflowTrackingConfig,
    TrackingPayload,
)


class TrackingAdapter(Protocol):
    """Adapter contract used by the orchestration pipeline."""

    # ADD 2026-08-20: Prepared payload를 외부 tracking backend에 기록하는 계약을 정의한다.
    def track(self, payload: TrackingPayload) -> LoggedRun: ...


@dataclass(frozen=True)
class PatchCoreTrackingResult:
    """Completed MLflow run and immutable local pointer location."""

    experiment_id: str
    run_id: str
    run_name: str
    pointer_path: Path


# ADD 2026-08-20: Existing PatchCore stage artifact를 검증해 하나의 MLflow run으로 backfill한다.
def track_patchcore_run(
    *,
    inputs: PatchCoreTrackingInputs,
    tracking_config: MlflowTrackingConfig,
    tracking_id: str,
    output_root: Path = DEFAULT_TRACKING_ROOT,
    adapter: TrackingAdapter | None = None,
) -> PatchCoreTrackingResult:
    """Track one canonical lineage and persist its project-side run pointer."""
    validate_tracking_id(tracking_id)
    output_dir = output_root / tracking_id
    if output_dir.exists():
        raise FileExistsError(f"MLflow tracking output already exists: {output_dir}")

    # MLflow 접근 전에 모든 source artifact와 cross-stage provenance를 검증한다.
    prepared = prepare_patchcore_tracking(inputs)
    run_name = tracking_config.run_name or (
        f"{prepared.payload.parameters['category']}_baseline_"
        f"seed{prepared.payload.parameters['random_seed']}"
    )
    resolved_config = replace(tracking_config, run_name=run_name)

    # 검증된 allowlist payload를 단일 run으로 기록하고 성공 identity를 반환받는다.
    tracker = adapter or MlflowTrackingAdapter(resolved_config)
    logged_run = tracker.track(prepared.payload)

    # 외부 logging 완료 후 credential 없는 immutable run pointer만 project output에 저장한다.
    pointer = build_tracking_pointer(
        tracking_id=tracking_id,
        experiment_name=resolved_config.experiment_name,
        experiment_id=logged_run.experiment_id,
        run_id=logged_run.run_id,
        run_name=run_name,
        identity=prepared.identity,
    )
    pointer_path = write_tracking_pointer(pointer, output_dir)
    return PatchCoreTrackingResult(
        experiment_id=logged_run.experiment_id,
        run_id=logged_run.run_id,
        run_name=run_name,
        pointer_path=pointer_path,
    )


# ADD 2026-08-20: PatchCore lineage backfill의 required/optional CLI 입력을 정의한다.
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track existing PatchCore artifacts as one validated MLflow lineage run."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--tracking-id", required=True)
    parser.add_argument("--manifest-summary", type=Path)
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--per-defect-metrics", type=Path)
    parser.add_argument("--model-benchmark", type=Path)
    parser.add_argument("--api-benchmark", type=Path)
    parser.add_argument("--tracking-uri")
    parser.add_argument("--experiment-name")
    parser.add_argument("--run-name")
    parser.add_argument("--artifact-location")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_TRACKING_ROOT)
    return parser.parse_args()


# ADD 2026-08-20: CLI/environment 설정을 결합해 canonical MLflow backfill을 실행한다.
def main() -> int:
    args = _parse_args()

    # CLI override가 없으면 safe local default 또는 MLFLOW_* environment를 사용한다.
    tracking_config = MlflowTrackingConfig.from_environment(
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        artifact_location=args.artifact_location,
    )
    inputs = PatchCoreTrackingInputs(
        config_path=args.config,
        manifest_path=args.manifest,
        artifact_dir=args.artifact_dir,
        manifest_summary_path=args.manifest_summary,
        thresholds_path=args.thresholds,
        metrics_path=args.metrics,
        per_defect_metrics_path=args.per_defect_metrics,
        model_benchmark_path=args.model_benchmark,
        api_benchmark_path=args.api_benchmark,
    )

    # 이미 생성된 stage artifact를 재학습/재평가 없이 MLflow에 기록한다.
    result = track_patchcore_run(
        inputs=inputs,
        tracking_config=tracking_config,
        tracking_id=args.tracking_id,
        output_root=args.output_root,
    )
    print("PatchCore MLflow tracking: PASS")
    print(f"Experiment ID: {result.experiment_id}")
    print(f"Run ID: {result.run_id}")
    print(f"Run name: {result.run_name}")
    print(f"Pointer: {result.pointer_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
