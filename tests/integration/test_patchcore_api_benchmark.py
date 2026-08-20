"""Integration test for the FastAPI HTTP benchmark pipeline with a fake runtime."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from torch import Tensor

from ml.datasets.manifest import ManifestRecord, write_manifest_csv
from pipelines.benchmark_patchcore_api import benchmark_patchcore_api
from services.inference.runtime import (
    InferenceResult,
    ModelRuntime,
    PatchCoreRuntimeConfig,
    ServingProvenance,
)
from shared.hashing import sha256_file
from tests.persistence_helpers import prepare_sqlite_database


class _BenchmarkRuntime:
    """Fake runtime carrying production-shaped identity and provenance."""

    model_name = "patchcore"
    category = "metal_nut"
    device = "cpu"

    # ADD 2026-08-20: Benchmark fake의 provenance와 inference 호출 횟수를 초기화한다.
    def __init__(self, manifest_sha256: str) -> None:
        self.predict_calls = 0
        self.provenance = ServingProvenance(
            manifest_sha256=manifest_sha256,
            artifact_metadata_sha256="b" * 64,
            model_sha256="c" * 64,
            threshold_artifact_sha256="d" * 64,
        )

    # ADD 2026-08-20: HTTP benchmark에서 schema-valid normal prediction을 반환한다.
    def predict(self, image: Tensor) -> InferenceResult:
        self.predict_calls += 1
        return InferenceResult(
            model_name=self.model_name,
            category=self.category,
            is_anomaly=False,
            anomaly_score=0.25,
            threshold=0.5,
            comparison_operator=">",
        )


# ADD 2026-08-20: Tiny test manifest와 실제 PNG payload 두 개를 생성한다.
def _write_test_inputs(tmp_path: Path) -> tuple[Path, Path]:
    dataset_root = tmp_path / "dataset"
    image_paths = [Path("metal_nut/test/good/000.png"), Path("metal_nut/test/bent/001.png")]
    records: list[ManifestRecord] = []
    for index, image_path in enumerate(image_paths):
        absolute_path = dataset_root / image_path
        absolute_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), color=(index * 100, 20, 30)).save(
            absolute_path,
            format="PNG",
        )
        records.append(
            ManifestRecord(
                sample_id=f"sample-{index}",
                category="metal_nut",
                source_split="test",
                split="test",
                defect_type="good" if index == 0 else "bent",
                label=index,
                image_path=str(image_path),
                mask_path="",
                width=8,
                height=8,
            )
        )
    manifest_path = tmp_path / "manifest.csv"
    write_manifest_csv(records, manifest_path)
    return dataset_root, manifest_path


# ADD 2026-08-20: One lifespan/runtime으로 warmup과 measured HTTP 요청 및 JSON 저장을 검증한다.
def test_api_benchmark_pipeline_reuses_fake_runtime_and_writes_result(tmp_path: Path) -> None:
    dataset_root, manifest_path = _write_test_inputs(tmp_path)
    runtime = _BenchmarkRuntime(sha256_file(manifest_path))
    load_calls: list[PatchCoreRuntimeConfig] = []

    # ADD 2026-08-20: Benchmark lifespan의 runtime load 횟수와 config를 기록한다.
    def load(config: PatchCoreRuntimeConfig) -> ModelRuntime:
        load_calls.append(config)
        return runtime

    output_dir = tmp_path / "benchmark-output"
    summary = benchmark_patchcore_api(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        artifact_dir=tmp_path / "artifact",
        thresholds_path=tmp_path / "thresholds.json",
        database_url=prepare_sqlite_database(tmp_path),
        output_dir=output_dir,
        requested_device="cpu",
        warmup_count=1,
        runtime_loader=load,
    )

    payload = json.loads(summary.benchmark_path.read_text(encoding="utf-8"))
    assert summary.measured_count == 2
    assert summary.successful_request_count == 2
    assert summary.failed_request_count == 0
    assert runtime.predict_calls == 3
    assert len(load_calls) == 1
    assert payload["provenance"]["manifest_sha256"] == sha256_file(manifest_path)
    assert payload["conditions"]["measured_count"] == 2
    assert payload["conditions"]["warmup_count"] == 1
