"""Contract tests for Kubernetes and GCP deployment foundation manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


# ADD 2026-08-21: Repository root를 Kubernetes configuration tests에 반환한다.
def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


# ADD 2026-08-21: 단일 Kubernetes YAML object를 typed mapping으로 로드한다.
def _load_yaml(relative_path: str) -> dict[str, Any]:
    loaded = yaml.safe_load((_project_root() / relative_path).read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


# ADD 2026-08-21: Named container를 pod template에서 반환한다.
def _container(workload: dict[str, Any], name: str) -> dict[str, Any]:
    containers = workload["spec"]["template"]["spec"]["containers"]
    return next(container for container in containers if container["name"] == name)


# ADD 2026-08-21: API Deployment의 replica, worker, rollout과 probe contract를 검증한다.
def test_kubernetes_api_deployment_lifecycle_contract() -> None:
    deployment = _load_yaml("infra/k8s/base/deployment.yaml")
    container = _container(deployment, "api")
    dockerfile = (_project_root() / "Dockerfile").read_text(encoding="utf-8")

    assert deployment["apiVersion"] == "apps/v1"
    assert deployment["kind"] == "Deployment"
    assert deployment["spec"]["replicas"] == 1
    assert deployment["spec"]["strategy"] == {
        "type": "RollingUpdate",
        "rollingUpdate": {"maxUnavailable": 0, "maxSurge": 1},
    }
    assert deployment["spec"]["template"]["spec"]["terminationGracePeriodSeconds"] == 60
    assert "command" not in container and "args" not in container
    assert '"--workers", "1"' in dockerfile
    assert container["startupProbe"]["httpGet"]["path"] == "/health"
    assert container["startupProbe"]["failureThreshold"] == 30
    assert container["livenessProbe"]["httpGet"]["path"] == "/health"
    assert container["readinessProbe"]["httpGet"]["path"] == "/ready"


# ADD 2026-08-21: API resource baseline과 non-root read-only security context를 검증한다.
def test_kubernetes_api_resource_and_security_contract() -> None:
    deployment = _load_yaml("infra/k8s/base/deployment.yaml")
    pod_spec = deployment["spec"]["template"]["spec"]
    container = _container(deployment, "api")

    assert container["resources"] == {
        "requests": {"cpu": "500m", "memory": "1Gi"},
        "limits": {"cpu": "2", "memory": "3Gi"},
    }
    assert pod_spec["automountServiceAccountToken"] is False
    assert pod_spec["securityContext"]["runAsNonRoot"] is True
    assert pod_spec["securityContext"]["runAsUser"] == 10001
    assert pod_spec["securityContext"]["runAsGroup"] == 10001
    assert pod_spec["securityContext"]["seccompProfile"]["type"] == "RuntimeDefault"
    assert container["securityContext"]["allowPrivilegeEscalation"] is False
    assert container["securityContext"]["readOnlyRootFilesystem"] is True
    assert container["securityContext"]["capabilities"]["drop"] == ["ALL"]


# ADD 2026-08-21: Writable tmpfs와 external read-only artifact PVC mount를 검증한다.
def test_kubernetes_temporary_and_artifact_volume_contract() -> None:
    deployment = _load_yaml("infra/k8s/base/deployment.yaml")
    pod_spec = deployment["spec"]["template"]["spec"]
    container = _container(deployment, "api")
    mounts = {mount["mountPath"]: mount for mount in container["volumeMounts"]}
    volumes = {volume["name"]: volume for volume in pod_spec["volumes"]}

    assert mounts["/tmp"]["name"] == "temporary-files"
    assert volumes["temporary-files"]["emptyDir"] == {
        "medium": "Memory",
        "sizeLimit": "256Mi",
    }
    assert mounts["/runtime/model"]["readOnly"] is True
    assert mounts["/runtime/thresholds"]["readOnly"] is True
    assert volumes["model-artifacts"]["persistentVolumeClaim"] == {
        "claimName": "smartfactory-model-artifacts",
        "readOnly": True,
    }


# ADD 2026-08-21: Non-secret ConfigMap과 DATABASE_URL Secret reference 분리를 검증한다.
def test_kubernetes_configuration_and_secret_boundary() -> None:
    configmap = _load_yaml("infra/k8s/base/configmap.yaml")
    deployment = _load_yaml("infra/k8s/base/deployment.yaml")
    container = _container(deployment, "api")
    database_env = next(item for item in container["env"] if item["name"] == "DATABASE_URL")
    manifest_text = "\n".join(
        path.read_text(encoding="utf-8") for path in (_project_root() / "infra/k8s").rglob("*.yaml")
    ).lower()

    assert configmap["data"] == {
        "MODEL_DEVICE": "cpu",
        "MAX_UPLOAD_BYTES": "10485760",
        "PATCHCORE_ARTIFACT_DIR": "/runtime/model",
        "PATCHCORE_THRESHOLDS_PATH": "/runtime/thresholds/thresholds.json",
    }
    assert database_env["valueFrom"]["secretKeyRef"] == {
        "name": "smartfactory-api-secrets",
        "key": "database-url",
    }
    assert "value" not in database_env
    assert "postgresql+psycopg://" not in manifest_text
    assert "change-me" not in manifest_text
    assert "password:" not in manifest_text
    assert configmap["data"].keys().isdisjoint({"model.pt", "thresholds.json"})


# ADD 2026-08-21: Service가 internal ClusterIP만 노출하는지 검증한다.
def test_kubernetes_service_is_internal_cluster_ip() -> None:
    service = _load_yaml("infra/k8s/base/service.yaml")

    assert service["kind"] == "Service"
    assert service["spec"]["type"] == "ClusterIP"
    assert service["spec"]["ports"] == [
        {"name": "http", "port": 80, "targetPort": "http", "protocol": "TCP"}
    ]


# ADD 2026-08-21: Alembic migration이 secured one-shot Job으로 분리됐는지 검증한다.
def test_kubernetes_migration_job_is_separate_and_immutable_safe() -> None:
    deployment = _load_yaml("infra/k8s/base/deployment.yaml")
    job = _load_yaml("infra/k8s/base/migration-job.yaml")
    pod_spec = job["spec"]["template"]["spec"]
    container = _container(job, "migrate")

    assert job["apiVersion"] == "batch/v1"
    assert job["kind"] == "Job"
    assert container["command"] == ["alembic", "upgrade", "head"]
    assert pod_spec["restartPolicy"] == "Never"
    assert job["spec"]["backoffLimit"] == 2
    assert "initContainers" not in deployment["spec"]["template"]["spec"]
    assert container["env"][0]["valueFrom"]["secretKeyRef"]["key"] == "database-url"
    assert container["securityContext"]["readOnlyRootFilesystem"] is True
    assert container["securityContext"]["allowPrivilegeEscalation"] is False


# ADD 2026-08-21: CPU base와 GPU overlay가 device/resource 책임을 정확히 분리하는지 검증한다.
def test_kubernetes_cpu_and_gpu_overlay_contract() -> None:
    configmap = _load_yaml("infra/k8s/base/configmap.yaml")
    gpu_config_patch = _load_yaml("infra/k8s/overlays/gcp-gpu/configmap-patch.yaml")
    gpu_deployment_patch = _load_yaml("infra/k8s/overlays/gcp-gpu/deployment-patch.yaml")
    gpu_container = _container(gpu_deployment_patch, "api")

    assert configmap["data"]["MODEL_DEVICE"] == "cpu"
    assert gpu_config_patch["data"]["MODEL_DEVICE"] == "cuda"
    assert gpu_container["resources"]["limits"]["nvidia.com/gpu"] == 1
    assert set(gpu_container["resources"]["limits"]) == {
        "cpu",
        "memory",
        "nvidia.com/gpu",
    }
    assert "nodeSelector" not in gpu_deployment_patch["spec"]["template"]["spec"]


# ADD 2026-08-21: Kustomize resource graph와 금지 workload 부재를 검증한다.
def test_kubernetes_kustomize_and_scope_contract() -> None:
    base = _load_yaml("infra/k8s/base/kustomization.yaml")
    kinds = {
        document["kind"]
        for path in (_project_root() / "infra/k8s").rglob("*.yaml")
        if (document := yaml.safe_load(path.read_text(encoding="utf-8"))).get("kind")
        != "Kustomization"
    }

    assert base["namespace"] == "smartfactory"
    assert set(base["resources"]) == {
        "namespace.yaml",
        "configmap.yaml",
        "migration-job.yaml",
        "deployment.yaml",
        "service.yaml",
    }
    assert kinds.isdisjoint(
        {"StatefulSet", "LoadBalancer", "HorizontalPodAutoscaler", "CronJob", "Secret"}
    )
    assert not list((_project_root() / "infra/k8s").rglob("*secret*.yaml"))


# ADD 2026-08-21: 문서화된 release가 migration completion 뒤에만 API를 적용하는지 검증한다.
def test_documented_kubernetes_release_sequence_is_migration_gated() -> None:
    root = _project_root()
    namespace = _load_yaml("infra/k8s/base/namespace.yaml")
    configmap = _load_yaml("infra/k8s/base/configmap.yaml")
    job = _load_yaml("infra/k8s/base/migration-job.yaml")
    deployment = _load_yaml("infra/k8s/base/deployment.yaml")
    service = _load_yaml("infra/k8s/base/service.yaml")
    runbook = (root / "docs/deployment/KUBERNETES_GCP.md").read_text(encoding="utf-8")
    makefile = (root / "Makefile").read_text(encoding="utf-8")

    assert namespace["metadata"]["labels"]["app.kubernetes.io/component"] == "configuration"
    assert configmap["metadata"]["labels"]["app.kubernetes.io/component"] == "configuration"
    assert job["metadata"]["labels"]["app.kubernetes.io/component"] == "migration"
    assert deployment["metadata"]["labels"]["app.kubernetes.io/component"] == "api"
    assert service["metadata"]["labels"]["app.kubernetes.io/component"] == "api"

    ordered_contract = (
        "app.kubernetes.io/component=configuration",
        "kubectl get secret smartfactory-api-secrets",
        "kubectl get pvc smartfactory-model-artifacts",
        "kubectl delete job smartfactory-api-migrate",
        "app.kubernetes.io/component=migration",
        "kubectl wait --for=condition=complete job/smartfactory-api-migrate",
        "app.kubernetes.io/component=api",
        "kubectl rollout status deployment/smartfactory-api",
    )
    positions = [runbook.index(contract) for contract in ordered_contract]
    assert positions == sorted(positions)
    assert runbook.count('kubectl apply -k "${PROFILE}" -l ') == 3
    assert not [
        line
        for line in runbook.splitlines()
        if line.startswith("kubectl apply -k ") and " -l " not in line
    ]
    assert "kubectl apply" not in makefile
