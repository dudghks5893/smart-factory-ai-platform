"""Explicit MLflow tracking adapter without domain-specific lineage logic."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Protocol, cast

from mlflow import MlflowClient
from mlflow.entities import Metric, Param, RunTag

DEFAULT_MLFLOW_TRACKING_URI = "sqlite:///outputs/mlflow/mlflow.db"
DEFAULT_MLFLOW_EXPERIMENT_NAME = "smartfactory-patchcore"

type ParameterValue = str | int | float | bool


class TrackingClient(Protocol):
    """MLflow client operations required by the project adapter."""

    # ADD 2026-08-20: 이름으로 experiment를 조회하는 client 계약을 정의한다.
    def get_experiment_by_name(self, name: str) -> object | None: ...

    # ADD 2026-08-20: Artifact location과 함께 experiment를 생성하는 client 계약을 정의한다.
    def create_experiment(
        self,
        name: str,
        artifact_location: str | None = None,
        tags: dict[str, object] | None = None,
    ) -> str: ...

    # ADD 2026-08-20: Experiment 아래 named run을 생성하는 client 계약을 정의한다.
    def create_run(
        self,
        experiment_id: str,
        tags: dict[str, object] | None = None,
        run_name: str | None = None,
    ) -> object: ...

    # ADD 2026-08-20: Scalar metric/parameter/tag batch를 기록하는 client 계약을 정의한다.
    def log_batch(
        self,
        run_id: str,
        metrics: list[Metric],
        params: list[Param],
        tags: list[RunTag],
        synchronous: bool | None = None,
    ) -> object | None: ...

    # ADD 2026-08-20: Allowlisted local file을 artifact store에 기록하는 client 계약을 정의한다.
    def log_artifact(
        self,
        run_id: str,
        local_path: str,
        artifact_path: str | None = None,
    ) -> None: ...

    # ADD 2026-08-20: Run lifecycle terminal status를 기록하는 client 계약을 정의한다.
    def set_terminated(self, run_id: str, status: str | None = None) -> None: ...


@dataclass(frozen=True)
class MlflowTrackingConfig:
    """Connection and naming settings for one MLflow tracking operation."""

    tracking_uri: str
    experiment_name: str
    run_name: str | None = None
    artifact_location: str | None = None

    # ADD 2026-08-20: CLI override와 environment에서 안전한 MLflow 설정을 구성한다.
    @classmethod
    def from_environment(
        cls,
        *,
        tracking_uri: str | None = None,
        experiment_name: str | None = None,
        run_name: str | None = None,
        artifact_location: str | None = None,
    ) -> MlflowTrackingConfig:
        """Resolve explicit values before environment variables and local defaults."""
        resolved_uri = (
            tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or DEFAULT_MLFLOW_TRACKING_URI
        )
        resolved_experiment = (
            experiment_name or os.getenv("MLFLOW_EXPERIMENT_NAME") or DEFAULT_MLFLOW_EXPERIMENT_NAME
        )
        resolved_run_name = run_name or os.getenv("MLFLOW_RUN_NAME")
        resolved_artifact_location = artifact_location or os.getenv("MLFLOW_ARTIFACT_ROOT")
        if resolved_artifact_location is None and resolved_uri.startswith("sqlite:///"):
            resolved_artifact_location = Path("outputs/mlflow/artifacts").resolve().as_uri()

        config = cls(
            tracking_uri=resolved_uri,
            experiment_name=resolved_experiment,
            run_name=resolved_run_name,
            artifact_location=resolved_artifact_location,
        )
        config.validate()
        return config

    # ADD 2026-08-20: MLflow connection과 identity 설정의 빈 값을 거부한다.
    def validate(self) -> None:
        """Validate settings before constructing an external tracking client."""
        if not self.tracking_uri.strip():
            raise ValueError("MLflow tracking_uri must not be empty.")
        if not self.experiment_name.strip():
            raise ValueError("MLflow experiment_name must not be empty.")
        if self.run_name is not None and not self.run_name.strip():
            raise ValueError("MLflow run_name must not be empty when provided.")


@dataclass(frozen=True)
class TrackedArtifact:
    """One explicitly allowlisted local file and its MLflow artifact directory."""

    local_path: Path
    artifact_path: str


@dataclass(frozen=True)
class TrackingPayload:
    """Validated scalar metadata and allowlisted files for one MLflow run."""

    parameters: Mapping[str, ParameterValue]
    metrics: Mapping[str, float]
    tags: Mapping[str, str]
    artifacts: tuple[TrackedArtifact, ...]


@dataclass(frozen=True)
class LoggedRun:
    """Stable identity returned after a successfully completed MLflow run."""

    experiment_id: str
    run_id: str


class MlflowTrackingError(RuntimeError):
    """Raised when a requested MLflow operation cannot complete."""


class MlflowTrackingAdapter:
    """Create one run, log a prepared payload, and terminate it explicitly."""

    # ADD 2026-08-20: Tracking 설정과 injectable MLflow client factory를 초기화한다.
    def __init__(
        self,
        config: MlflowTrackingConfig,
        *,
        client_factory: Callable[[str], object] | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self._client_factory = client_factory or (lambda uri: MlflowClient(uri))

    # ADD 2026-08-20: Experiment를 선택하고 단일 run payload를 fail-fast 방식으로 기록한다.
    def track(self, payload: TrackingPayload) -> LoggedRun:
        """Log one complete run and mark partial runs failed on any logging error."""
        _validate_payload(payload)
        _prepare_sqlite_parent(self.config.tracking_uri)

        run_id: str | None = None
        client: TrackingClient | None = None
        try:
            # MLflow backend에 연결하고 canonical experiment를 생성하거나 재사용한다.
            active_client = cast(TrackingClient, self._client_factory(self.config.tracking_uri))
            client = active_client
            experiment_id = self._get_or_create_experiment(active_client)
            run = active_client.create_run(
                experiment_id,
                run_name=self.config.run_name,
            )
            run_id = _run_id(run)

            # 검증된 scalar metadata를 한 번에 기록한 뒤 allowlist artifact만 업로드한다.
            timestamp = int(time() * 1000)
            active_client.log_batch(
                run_id,
                metrics=[
                    Metric(key, value, timestamp, 0) for key, value in payload.metrics.items()
                ],
                params=[Param(key, str(value)) for key, value in payload.parameters.items()],
                tags=[RunTag(key, value) for key, value in payload.tags.items()],
                synchronous=True,
            )
            for artifact in payload.artifacts:
                active_client.log_artifact(
                    run_id,
                    str(artifact.local_path),
                    artifact_path=artifact.artifact_path,
                )
            active_client.set_terminated(run_id, status="FINISHED")
        except Exception as exc:
            if client is not None and run_id is not None:
                try:
                    client.set_terminated(run_id, status="FAILED")
                except Exception:
                    pass
            raise MlflowTrackingError("MLflow run logging failed.") from exc

        return LoggedRun(experiment_id=experiment_id, run_id=run_id)

    # ADD 2026-08-20: 이름으로 experiment를 재사용하거나 configured artifact store에 생성한다.
    def _get_or_create_experiment(self, client: TrackingClient) -> str:
        experiment = client.get_experiment_by_name(self.config.experiment_name)
        if experiment is not None:
            return _experiment_id(experiment)
        return client.create_experiment(
            self.config.experiment_name,
            artifact_location=self.config.artifact_location,
            tags={"project": "smart-factory-ai-platform"},
        )


# ADD 2026-08-20: Payload의 scalar metric과 allowlisted file 계약을 검증한다.
def _validate_payload(payload: TrackingPayload) -> None:
    if not payload.parameters:
        raise ValueError("MLflow tracking parameters must not be empty.")
    if not payload.tags:
        raise ValueError("MLflow tracking tags must not be empty.")
    for key, value in payload.metrics.items():
        if not key or not isinstance(value, float):
            raise ValueError("MLflow metrics must use non-empty names and float values.")
    for artifact in payload.artifacts:
        if not artifact.local_path.is_file():
            raise FileNotFoundError(f"MLflow artifact input not found: {artifact.local_path}")
        if not artifact.artifact_path.strip():
            raise ValueError("MLflow artifact_path must not be empty.")


# ADD 2026-08-20: Local SQLite URI의 database parent를 client 초기화 전에 생성한다.
def _prepare_sqlite_parent(tracking_uri: str) -> None:
    prefix = "sqlite:///"
    if not tracking_uri.startswith(prefix):
        return
    database_path = tracking_uri.removeprefix(prefix)
    if not database_path or database_path == ":memory:":
        return
    Path(database_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


# ADD 2026-08-20: MLflow experiment object에서 stable experiment ID를 추출한다.
def _experiment_id(experiment: object) -> str:
    experiment_id = getattr(experiment, "experiment_id", None)
    if not isinstance(experiment_id, str) or not experiment_id:
        raise TypeError("MLflow experiment response has no valid experiment_id.")
    return experiment_id


# ADD 2026-08-20: MLflow run object에서 stable run ID를 추출한다.
def _run_id(run: object) -> str:
    info = getattr(run, "info", None)
    run_id = getattr(info, "run_id", None)
    if not isinstance(run_id, str) or not run_id:
        raise TypeError("MLflow run response has no valid run_id.")
    return run_id
