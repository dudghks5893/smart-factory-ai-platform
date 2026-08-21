from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


# ADD 2026-08-20: Repository root의 CI workflow를 YAML 1.1 key coercion 없이 로드한다.
def _load_ci_workflow() -> tuple[dict[str, Any], str]:
    workflow_path = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yml"
    workflow_text = workflow_path.read_text(encoding="utf-8")
    workflow = yaml.load(workflow_text, Loader=yaml.BaseLoader)
    assert isinstance(workflow, dict)
    return workflow, workflow_text


# ADD 2026-08-20: CI trigger, 최소 권한과 stale PR concurrency contract를 검증한다.
def test_ci_workflow_trigger_permission_and_execution_policy() -> None:
    workflow, _ = _load_ci_workflow()

    assert set(workflow["on"]) == {"pull_request", "push", "workflow_dispatch"}
    assert workflow["on"]["push"]["branches"] == ["main"]
    assert workflow["permissions"] == {"contents": "read"}
    assert "github.event.pull_request.number || github.ref" in workflow["concurrency"]["group"]
    assert workflow["concurrency"]["cancel-in-progress"] == (
        "${{ github.event_name == 'pull_request' }}"
    )


# ADD 2026-08-20: Quality, actual PostgreSQL과 runtime image job의 핵심 책임을 검증한다.
# MODIFY 2026-08-21: Dashboard와 API image build 및 Kubernetes render contract를 검증한다.
def test_ci_workflow_job_contracts() -> None:
    workflow, _ = _load_ci_workflow()
    jobs = workflow["jobs"]

    assert set(jobs) == {"quality", "postgres-integration", "docker", "kubernetes"}
    assert {job["name"] for job in jobs.values()} == {
        "quality",
        "postgres-integration",
        "docker",
        "kubernetes",
    }
    assert all(job["runs-on"] == "ubuntu-24.04-arm" for job in jobs.values())
    assert all(int(job["timeout-minutes"]) <= 45 for job in jobs.values())
    assert jobs["postgres-integration"]["needs"] == "quality"
    assert jobs["docker"]["needs"] == "quality"
    assert jobs["kubernetes"]["needs"] == "quality"

    quality_commands = {step.get("run") for step in jobs["quality"]["steps"]}
    assert {"uv lock --check", "uv sync --locked", "make check"} <= quality_commands

    postgres = jobs["postgres-integration"]
    assert postgres["services"]["postgres"]["image"] == "postgres:17.6-bookworm"
    postgres_commands = "\n".join(
        step.get("run", "") for step in postgres["steps"] if "run" in step
    )
    assert "uv run alembic upgrade head" in postgres_commands
    assert "tests/integration/test_postgres_container.py" in postgres_commands

    docker_steps = jobs["docker"]["steps"]
    assert any(step.get("run") == "docker compose config --quiet" for step in docker_steps)
    docker_builds = [
        step for step in docker_steps if step.get("uses", "").startswith("docker/build")
    ]
    assert {step["with"]["target"] for step in docker_builds} == {
        "runtime",
        "dashboard-runtime",
    }
    assert all(step["with"]["platforms"] == "linux/arm64" for step in docker_builds)
    assert all(step["with"]["push"] == "false" for step in docker_builds)

    kubernetes_steps = jobs["kubernetes"]["steps"]
    kubectl_setup = next(
        step for step in kubernetes_steps if step.get("uses", "").startswith("azure/setup-kubectl")
    )
    assert kubectl_setup["with"]["version"] == "v1.34.1"
    assert any(step.get("run") == "make k8s-check" for step in kubernetes_steps)


# ADD 2026-08-20: Pinned uv, CPU lock reuse와 artifact/secret 비생성 정책을 검증한다.
def test_ci_workflow_dependency_cache_and_artifact_policy() -> None:
    workflow, workflow_text = _load_ci_workflow()

    assert workflow["env"] == {"UV_PYTHON": "3.12.14", "UV_VERSION": "0.12.5"}
    assert workflow_text.count("enable-cache: true") == 2
    assert "cache-dependency-glob: uv.lock" in workflow_text
    assert "cache-from: type=gha,scope=runtime-arm64" in workflow_text
    assert "cache-to: type=gha,mode=max,scope=runtime-arm64" in workflow_text
    assert "cache-from: type=gha,scope=dashboard-arm64" in workflow_text
    assert "cache-to: type=gha,mode=max,scope=dashboard-arm64" in workflow_text
    assert "UV_TORCH_BACKEND" not in workflow_text
    assert "actions/upload-artifact" not in workflow_text
    assert "secrets." not in workflow_text
    assert workflow["jobs"]["docker"]["env"]["DOCKER_BUILD_RECORD_UPLOAD"] == "false"
