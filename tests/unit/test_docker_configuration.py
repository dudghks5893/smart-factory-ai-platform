from __future__ import annotations

import tomllib
from pathlib import Path

import yaml


# ADD 2026-08-20: Repository root를 Docker configuration test에 반환한다.
def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


# ADD 2026-08-20: Docker image가 pinned/reproducible/non-root artifact policy를 지키는지 검증한다.
# MODIFY 2026-08-25: API browser monitor asset과 Dashboard/RAG target isolation을 검증한다.
# MODIFY 2026-08-26: Vision runtime의 required OpenCV shared libraries를 검증한다.
def test_dockerfile_runtime_and_context_policy() -> None:
    root = _project_root()
    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
    dockerignore = (root / ".dockerignore").read_text(encoding="utf-8").splitlines()

    assert "python:3.12.14-slim-bookworm" in dockerfile
    assert "ghcr.io/astral-sh/uv:0.12.5" in dockerfile
    assert (
        "uv sync --locked --no-dev --no-group dashboard --no-group rag --no-install-project"
        in dockerfile
    )
    assert "uv sync --locked --only-group dashboard --no-install-project" in dockerfile
    assert "uv sync --locked --only-group rag --no-install-project" in dockerfile
    assert "FROM application AS runtime" in dockerfile
    assert "FROM application AS test" in dockerfile
    assert "FROM application-base AS dashboard-runtime" in dockerfile
    assert "FROM application-base AS rag-runtime" in dockerfile
    assert "FROM application-base AS vision-application-base" in dockerfile
    assert "FROM vision-application-base AS application" in dockerfile
    assert "libgl1 libglib2.0-0 libxcb1" in dockerfile
    assert "COPY apps/live_monitor ./apps/live_monitor" in dockerfile
    assert "USER app" in dockerfile
    assert '"--workers", "1"' in dockerfile
    assert "COPY ." not in dockerfile
    assert {
        ".git",
        ".venv",
        "data",
        "artifacts",
        "outputs",
        "models",
        "checkpoints",
        "mlruns",
        ".env",
    } <= set(dockerignore)


# ADD 2026-08-20: Compose startup ordering, pin, volume과 external model mount를 검증한다.
# MODIFY 2026-08-21: Dashboard/RAG observer image와 read-only artifact contract를 추가한다.
# MODIFY 2026-08-26: Optional YOLO config와 read-only runtime artifact mount를 검증한다.
# MODIFY 2026-08-26: Host MPS API가 local PostgreSQL에 접근할 loopback port를 검증한다.
def test_compose_postgres_migration_and_api_contract() -> None:
    compose = yaml.safe_load((_project_root() / "compose.yaml").read_text(encoding="utf-8"))
    services = compose["services"]

    assert set(services) == {
        "postgres",
        "migrate",
        "api",
        "test",
        "prometheus",
        "grafana",
        "dashboard",
        "rag",
    }
    assert services["postgres"]["image"] == "postgres:17.6-bookworm"
    assert services["postgres"]["ports"] == ["127.0.0.1:${POSTGRES_PORT:-5432}:5432"]
    assert "postgres_data" in compose["volumes"]
    assert services["migrate"]["command"] == ["alembic", "upgrade", "head"]
    assert services["migrate"]["depends_on"]["postgres"]["condition"] == "service_healthy"
    assert services["api"]["depends_on"]["migrate"]["condition"] == (
        "service_completed_successfully"
    )
    assert services["api"]["environment"]["MODEL_DEVICE"] == "${MODEL_DEVICE:-cpu}"
    assert services["api"]["environment"]["YOLO_SEGMENTATION_ENABLED"] == (
        "${YOLO_SEGMENTATION_ENABLED:-false}"
    )
    assert services["api"]["environment"]["YOLO_SEGMENTATION_ARTIFACT_DIR"] == (
        "/runtime/yolo-segmentation"
    )
    assert services["api"]["environment"]["YOLO_SEGMENTATION_DEVICE"] == (
        "${YOLO_SEGMENTATION_DEVICE:-cpu}"
    )
    assert services["api"]["environment"]["YOLO_SEGMENTATION_CONFIDENCE"] == (
        "${YOLO_SEGMENTATION_CONFIDENCE:-0.25}"
    )
    assert all(volume["read_only"] for volume in services["api"]["volumes"])
    assert services["api"]["volumes"][2]["target"] == "/runtime/yolo-segmentation"
    assert services["test"]["build"]["target"] == "test"
    assert services["test"]["profiles"] == ["test"]
    assert services["prometheus"]["image"] == "prom/prometheus:v3.12.0"
    assert services["grafana"]["image"] == "grafana/grafana:13.1.0"
    dashboard = services["dashboard"]
    assert dashboard["build"]["target"] == "dashboard-runtime"
    assert dashboard["read_only"] is True
    assert dashboard["volumes"][0]["read_only"] is True
    assert dashboard["volumes"][0]["target"] == "/runtime/drift"
    assert dashboard["ports"] == ["${DASHBOARD_PORT:-8501}:8501"]
    assert "depends_on" not in dashboard
    rag = services["rag"]
    assert rag["profiles"] == ["rag"]
    assert rag["build"]["target"] == "rag-runtime"
    assert rag["read_only"] is True
    assert rag["cap_drop"] == ["ALL"]
    assert rag["security_opt"] == ["no-new-privileges:true"]
    assert rag["volumes"][0]["read_only"] is True
    assert rag["volumes"][0]["target"] == "/runtime/rag/index"
    assert rag["ports"] == ["${RAG_PORT:-8001}:8001"]
    assert "depends_on" not in rag
    assert {"postgres_data", "prometheus_data", "grafana_data"} == set(compose["volumes"])


# ADD 2026-08-20: CUDA index가 Linux x86_64에만 적용되는 source marker를 검증한다.
def test_pytorch_cuda_source_is_linux_x86_64_only() -> None:
    with (_project_root() / "pyproject.toml").open("rb") as file:
        project = tomllib.load(file)

    expected_marker = "sys_platform == 'linux' and platform_machine == 'x86_64'"
    arm_cpu_marker = (
        "sys_platform == 'linux' and (platform_machine == 'aarch64' or platform_machine == 'arm64')"
    )
    assert project["tool"]["uv"]["sources"]["torch"] == [
        {"index": "pytorch-cu130", "marker": expected_marker},
        {"index": "pytorch-cpu", "marker": arm_cpu_marker},
    ]
    assert project["tool"]["uv"]["sources"]["torchvision"] == [
        {"index": "pytorch-cu130", "marker": expected_marker},
        {"index": "pytorch-cpu", "marker": arm_cpu_marker},
    ]
