"""Locked subprocess boundaries for model-affecting YOLO Workbench computation."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

ENVIRONMENT_JSON_PREFIX = "SMARTFACTORY_LOCKED_ENVIRONMENT_JSON="
REQUIRED_FRAMEWORK_VERSIONS = {
    "torch": "2.13.0",
    "torchvision": "0.28.0",
    "ultralytics": "8.4.128",
}
OPTIONAL_AUGMENTATION_PACKAGES = ("albumentations",)

type SubprocessRunner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class PackageProvenance:
    """Installed package identity from the official execution interpreter."""

    available: bool
    version: str | None
    path: str | None


@dataclass(frozen=True)
class LockedEnvironmentProvenance:
    """Machine-readable identity and import boundary for one locked Python process."""

    python_executable: str
    python_version: str
    packages: dict[str, PackageProvenance]
    cuda_available: bool
    gpu_name: str | None
    external_site_package_paths: tuple[str, ...]

    # ADD 2026-08-28: Locked execution evidence를 strict JSON mapping으로 반환한다.
    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        json.dumps(payload, allow_nan=False)
        return payload

    # ADD 2026-08-28: Official interpreter/package/CUDA가 repository venv에 격리됐는지 검증한다.
    def validate(self, repository_root: Path, *, require_cuda: bool) -> None:
        venv_root = Path(os.path.abspath(repository_root / ".venv"))
        executable = Path(os.path.abspath(self.python_executable))
        if not executable.is_relative_to(venv_root):
            raise ValueError("Official execution Python is outside the repository .venv.")
        if self.external_site_package_paths:
            raise ValueError("Official execution can search external site-packages.")
        for package_name, expected_base_version in REQUIRED_FRAMEWORK_VERSIONS.items():
            package = self.packages.get(package_name)
            if package is None or not package.available or package.path is None:
                raise ValueError(f"Official execution package is unavailable: {package_name}")
            if (
                package.version is None
                or package.version.split("+", maxsplit=1)[0] != expected_base_version
            ):
                raise ValueError(f"Official execution package version mismatch: {package_name}")
            if not Path(os.path.abspath(package.path)).is_relative_to(venv_root):
                raise ValueError(f"Official execution package is outside .venv: {package_name}")
        for package_name in OPTIONAL_AUGMENTATION_PACKAGES:
            package = self.packages.get(package_name)
            if package is None:
                raise ValueError(f"Optional package provenance is missing: {package_name}")
            if package.available and (
                package.path is None
                or not Path(os.path.abspath(package.path)).is_relative_to(venv_root)
            ):
                raise ValueError(f"Optional augmentation package is outside .venv: {package_name}")
        if require_cuda and (not self.cuda_available or not self.gpu_name):
            raise ValueError("Official controlled experiment requires available CUDA/GPU.")


@dataclass(frozen=True)
class ExperimentReviewArtifacts:
    """Deterministic paths consumed by Workbench review sections after subprocess training."""

    experiment_dir: Path
    experiment_result_path: Path
    comparison_path: Path
    telemetry_path: Path
    candidate_artifact_dir: Path
    package_path: Path
    package_metadata_path: Path


@dataclass(frozen=True)
class PreviewArtifacts:
    """Locked preview image and metadata paths consumed by the notebook controller."""

    metadata_path: Path
    augmentation_figure: Path
    representation_figure: Path | None
    metadata: dict[str, Any]


# ADD 2026-08-28: Current interpreter의 package provenance를 repository-local config로 수집한다.
def collect_current_environment(repository_root: Path) -> LockedEnvironmentProvenance:
    # Ultralytics import-time settings 접근도 ignored repository namespace에 격리한다.
    os.environ.setdefault(
        "YOLO_CONFIG_DIR",
        str(repository_root / "outputs/workbench/yolo_segmentation/.ultralytics-config"),
    )
    required_modules = {
        package_name: importlib.import_module(package_name)
        for package_name in REQUIRED_FRAMEWORK_VERSIONS
    }
    packages: dict[str, PackageProvenance] = {}
    for package_name in (*REQUIRED_FRAMEWORK_VERSIONS, *OPTIONAL_AUGMENTATION_PACKAGES):
        spec = importlib.util.find_spec(package_name)
        if spec is None:
            packages[package_name] = PackageProvenance(False, None, None)
            continue
        version = importlib.metadata.version(package_name)
        packages[package_name] = PackageProvenance(True, version, spec.origin)

    torch = required_modules["torch"]
    site_paths = tuple(
        sorted(
            {
                os.path.abspath(path)
                for path in sys.path
                if path and ("site-packages" in path or "dist-packages" in path)
            }
        )
    )
    venv_root = Path(os.path.abspath(Path(sys.executable).parent.parent))
    external_paths = tuple(path for path in site_paths if not Path(path).is_relative_to(venv_root))
    cuda_available = bool(torch.cuda.is_available())
    return LockedEnvironmentProvenance(
        python_executable=os.path.abspath(sys.executable),
        python_version=sys.version,
        packages=packages,
        cuda_available=cuda_available,
        gpu_name=str(torch.cuda.get_device_name(0)) if cuda_available else None,
        external_site_package_paths=external_paths,
    )


# ADD 2026-08-28: uv subprocess stdout에서 host warning과 분리된 environment JSON을 읽는다.
def _parse_environment_stdout(stdout: str) -> LockedEnvironmentProvenance:
    lines = [line for line in stdout.splitlines() if line.startswith(ENVIRONMENT_JSON_PREFIX)]
    if len(lines) != 1:
        raise RuntimeError("Locked environment subprocess did not emit one provenance payload.")
    raw = json.loads(lines[0].removeprefix(ENVIRONMENT_JSON_PREFIX))
    packages = {
        str(name): PackageProvenance(**value) for name, value in dict(raw["packages"]).items()
    }
    return LockedEnvironmentProvenance(
        python_executable=str(raw["python_executable"]),
        python_version=str(raw["python_version"]),
        packages=packages,
        cuda_available=bool(raw["cuda_available"]),
        gpu_name=None if raw["gpu_name"] is None else str(raw["gpu_name"]),
        external_site_package_paths=tuple(raw["external_site_package_paths"]),
    )


# ADD 2026-08-28: Official execution interpreter를 uv boundary에서 검사하고 provenance를 복원한다.
def inspect_locked_environment(
    repository_root: Path,
    *,
    require_cuda: bool,
    subprocess_runner: SubprocessRunner = subprocess.run,
) -> LockedEnvironmentProvenance:
    command = [
        "uv",
        "run",
        "--locked",
        "python",
        "-m",
        "ml.experiments.yolo_workbench_runtime",
        "--repository-root",
        str(repository_root),
    ]
    completed = subprocess_runner(
        command,
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    provenance = _parse_environment_stdout(completed.stdout)
    provenance.validate(repository_root, require_cuda=require_cuda)
    return provenance


# ADD 2026-08-28: Successful runner output의 deterministic review path와 필수 artifact를 복원한다.
def resolve_experiment_review_artifacts(
    *,
    experiment_id: str,
    experiment_root: Path,
    artifact_root: Path,
    package_root: Path,
) -> ExperimentReviewArtifacts:
    experiment_dir = experiment_root / experiment_id
    artifacts = ExperimentReviewArtifacts(
        experiment_dir=experiment_dir,
        experiment_result_path=experiment_dir / "experiment_result.json",
        comparison_path=experiment_dir / "comparison_to_baseline.json",
        telemetry_path=experiment_dir / "resource_telemetry.json",
        candidate_artifact_dir=artifact_root / experiment_id,
        package_path=package_root / f"{experiment_id}.zip",
        package_metadata_path=experiment_dir / "package_metadata.json",
    )
    required = (
        artifacts.experiment_result_path,
        artifacts.comparison_path,
        artifacts.telemetry_path,
        artifacts.candidate_artifact_dir / "model.pt",
        artifacts.candidate_artifact_dir / "metadata.json",
        artifacts.package_path,
        artifacts.package_metadata_path,
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Official experiment review artifact is missing: {missing[0]}")
    return artifacts


# ADD 2026-08-28: Official runtime을 재검증한 뒤 locked repository CLI로 training을 실행한다.
def run_official_training_subprocess(
    *,
    experiment_config_path: Path,
    experiment_id: str,
    experiment_root: Path,
    artifact_root: Path,
    package_root: Path,
    dataset_root: Path,
    baseline_artifact_dir: Path,
    repository_root: Path,
    subprocess_runner: SubprocessRunner = subprocess.run,
) -> ExperimentReviewArtifacts:
    # Training 직전 동일 uv boundary를 다시 검사해 stale notebook preflight를 허용하지 않는다.
    inspect_locked_environment(
        repository_root,
        require_cuda=True,
        subprocess_runner=subprocess_runner,
    )
    command = [
        "uv",
        "run",
        "--locked",
        "python",
        "-m",
        "pipelines.run_yolo_segmentation_experiment",
        "--experiment-config",
        str(experiment_config_path),
        "--dataset",
        str(dataset_root),
        "--baseline-artifact-dir",
        str(baseline_artifact_dir),
        "--device",
        "cuda",
        "--repository-root",
        str(repository_root),
    ]
    subprocess_runner(command, cwd=repository_root, check=True)
    return resolve_experiment_review_artifacts(
        experiment_id=experiment_id,
        experiment_root=experiment_root,
        artifact_root=artifact_root,
        package_root=package_root,
    )


# ADD 2026-08-28: Actual preview를 locked CLI에서 만들고 notebook용 path/metadata만 복원한다.
def run_locked_preview_subprocess(
    *,
    mode: Literal["research", "official"],
    experiment_config_path: Path,
    dataset_root: Path,
    output_root: Path,
    repository_root: Path,
    train_sample_ids: list[str],
    representation_sample_id: str | None,
    research_overrides: dict[str, object],
    subprocess_runner: SubprocessRunner = subprocess.run,
) -> PreviewArtifacts:
    if mode == "official" and research_overrides:
        raise ValueError("Official preview rejects every notebook training override.")
    command = [
        "uv",
        "run",
        "--locked",
        "python",
        "-m",
        "pipelines.run_yolo_workbench_preview",
        "--mode",
        mode,
        "--experiment-config",
        str(experiment_config_path),
        "--dataset",
        str(dataset_root),
        "--output-root",
        str(output_root),
        "--repository-root",
        str(repository_root),
    ]
    for sample_id in train_sample_ids:
        command.extend(("--train-sample-id", sample_id))
    if representation_sample_id is not None:
        command.extend(("--representation-sample-id", representation_sample_id))
    for field, value in sorted(research_overrides.items()):
        command.extend((f"--research-{field}", str(value)))
    subprocess_runner(command, cwd=repository_root, check=True)

    metadata_path = output_root / "preview_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    augmentation_figure = Path(metadata["augmentation"]["generated_path"])
    representation = metadata.get("representation")
    representation_path = (
        representation.get("generated_path") if isinstance(representation, dict) else None
    )
    artifacts = PreviewArtifacts(
        metadata_path=metadata_path,
        augmentation_figure=augmentation_figure,
        representation_figure=None if representation_path is None else Path(representation_path),
        metadata=metadata,
    )
    required = [metadata_path, augmentation_figure]
    if artifacts.representation_figure is not None:
        required.append(artifacts.representation_figure)
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Locked preview artifact is missing: {missing[0]}")
    return artifacts


# ADD 2026-08-28: Environment probe CLI의 repository path와 JSON output을 정의한다.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    return parser.parse_args()


# ADD 2026-08-28: uv process 내부 provenance를 검증 후 host-warning-safe prefix로 출력한다.
def main() -> int:
    args = parse_args()
    provenance = collect_current_environment(args.repository_root)
    provenance.validate(args.repository_root, require_cuda=False)
    print(ENVIRONMENT_JSON_PREFIX + json.dumps(provenance.to_json_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
