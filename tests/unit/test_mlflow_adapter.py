from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from services.tracking.mlflow import (
    MlflowTrackingAdapter,
    MlflowTrackingConfig,
    MlflowTrackingError,
    TrackedArtifact,
    TrackingPayload,
)


class FailingArtifactClient:
    """Minimal client double that fails after a run has been created."""

    def __init__(self) -> None:
        self.statuses: list[str | None] = []

    # ADD 2026-08-20: Existing experiment lookup test response를 반환한다.
    def get_experiment_by_name(self, name: str) -> object:
        return SimpleNamespace(experiment_id="1")

    # ADD 2026-08-20: 이 scenario에서 호출되면 실패하도록 unexpected create를 표시한다.
    def create_experiment(
        self,
        name: str,
        artifact_location: str | None = None,
        tags: dict[str, object] | None = None,
    ) -> str:
        raise AssertionError("existing experiment must be reused")

    # ADD 2026-08-20: Artifact failure 전에 stable run identity를 반환한다.
    def create_run(
        self,
        experiment_id: str,
        tags: dict[str, object] | None = None,
        run_name: str | None = None,
    ) -> object:
        return SimpleNamespace(info=SimpleNamespace(run_id="failed-run"))

    # ADD 2026-08-20: Metadata batch logging 성공을 simulation한다.
    def log_batch(
        self,
        run_id: str,
        metrics: list[object],
        params: list[object],
        tags: list[object],
        synchronous: bool | None = None,
    ) -> None:
        return None

    # ADD 2026-08-20: External artifact logging 실패를 simulation한다.
    def log_artifact(
        self,
        run_id: str,
        local_path: str,
        artifact_path: str | None = None,
    ) -> None:
        raise OSError("artifact store unavailable")

    # ADD 2026-08-20: Adapter가 실패 run을 명시적으로 종료했는지 기록한다.
    def set_terminated(self, run_id: str, status: str | None = None) -> None:
        self.statuses.append(status)


# ADD 2026-08-20: MLflow artifact 오류가 FAILED run과 caller-visible exception을 남기는지 검증한다.
def test_mlflow_adapter_marks_partial_run_failed_and_raises(tmp_path: Path) -> None:
    artifact = tmp_path / "metadata.json"
    artifact.write_text("{}\n", encoding="utf-8")
    client = FailingArtifactClient()
    adapter = MlflowTrackingAdapter(
        MlflowTrackingConfig("sqlite:///:memory:", "test", "failed-run"),
        client_factory=lambda _: client,
    )
    payload = TrackingPayload(
        parameters={"category": "metal_nut"},
        metrics={"image.auroc": 1.0},
        tags={"lineage.model_sha256": "a" * 64},
        artifacts=(TrackedArtifact(artifact, "model"),),
    )

    with pytest.raises(MlflowTrackingError, match="logging failed"):
        adapter.track(payload)

    assert client.statuses == ["FAILED"]
