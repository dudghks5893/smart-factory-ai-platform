"""Tests for locked YOLO Workbench subprocess and environment contracts."""

from __future__ import annotations

import inspect
import json
import subprocess
from pathlib import Path

import pytest

from ml.experiments.yolo_workbench_runtime import (
    ENVIRONMENT_JSON_PREFIX,
    ExperimentReviewArtifacts,
    LockedEnvironmentProvenance,
    PackageProvenance,
    inspect_locked_environment,
    run_locked_preview_subprocess,
    run_official_training_subprocess,
)
from pipelines.run_yolo_workbench_preview import run_yolo_workbench_preview


# ADD 2026-08-28: Repository venv 기반 framework/optional package provenance fixture를 만든다.
def _environment(
    repository_root: Path,
    *,
    cuda_available: bool = True,
    albumentations_path: Path | None = None,
    external_site_paths: tuple[str, ...] = (),
) -> LockedEnvironmentProvenance:
    site_packages = repository_root / ".venv/lib/python3.12/site-packages"
    return LockedEnvironmentProvenance(
        python_executable=str(repository_root / ".venv/bin/python3"),
        python_version="3.12.13",
        packages={
            "torch": PackageProvenance(
                True, "2.13.0+cu130", str(site_packages / "torch/__init__.py")
            ),
            "torchvision": PackageProvenance(
                True, "0.28.0+cu130", str(site_packages / "torchvision/__init__.py")
            ),
            "ultralytics": PackageProvenance(
                True, "8.4.128", str(site_packages / "ultralytics/__init__.py")
            ),
            "albumentations": PackageProvenance(
                albumentations_path is not None,
                "2.0.8" if albumentations_path is not None else None,
                None if albumentations_path is None else str(albumentations_path),
            ),
        },
        cuda_available=cuda_available,
        gpu_name="Tesla T4" if cuda_available else None,
        external_site_package_paths=external_site_paths,
    )


# ADD 2026-08-28: Environment probe가 uv boundary와 machine-readable evidence를 사용하는지 검증한다.
def test_locked_environment_preflight_uses_uv_and_accepts_host_warning(tmp_path: Path) -> None:
    repository_root = tmp_path / "repo"
    provenance = _environment(repository_root)

    def fake_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert command[:5] == [
            "uv",
            "run",
            "--locked",
            "python",
            "-m",
        ]
        assert command[5] == "ml.experiments.yolo_workbench_runtime"
        assert kwargs == {
            "cwd": repository_root,
            "check": True,
            "capture_output": True,
            "text": True,
        }
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "controller text\n"
                + ENVIRONMENT_JSON_PREFIX
                + json.dumps(provenance.to_json_dict())
                + "\n"
            ),
            stderr="Error in sitecustomize: ModuleNotFoundError: No module named 'wrapt'\n",
        )

    actual = inspect_locked_environment(
        repository_root,
        require_cuda=True,
        subprocess_runner=fake_runner,
    )

    assert actual == provenance
    assert actual.packages["albumentations"].available is False


# ADD 2026-08-28: External site-package search와 system-only Albumentations를 fail-fast한다.
def test_locked_environment_rejects_system_package_contamination(tmp_path: Path) -> None:
    repository_root = tmp_path / "repo"
    external_albumentations = Path(
        "/usr/local/lib/python3.12/dist-packages/albumentations/__init__.py"
    )
    with pytest.raises(ValueError, match="external site-packages"):
        _environment(
            repository_root,
            external_site_paths=("/usr/local/lib/python3.12/dist-packages",),
        ).validate(repository_root, require_cuda=True)
    with pytest.raises(ValueError, match="Optional augmentation package is outside"):
        _environment(
            repository_root,
            albumentations_path=external_albumentations,
        ).validate(repository_root, require_cuda=True)


# ADD 2026-08-28: Preview pipeline이 dataset/model 작업 전에 contaminated environment를 거부한다.
def test_preview_pipeline_rejects_system_only_albumentations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_root = tmp_path / "repo"
    contaminated = _environment(
        repository_root,
        albumentations_path=Path(
            "/usr/local/lib/python3.12/dist-packages/albumentations/__init__.py"
        ),
    )
    monkeypatch.setattr(
        "pipelines.run_yolo_workbench_preview.collect_current_environment",
        lambda repository_root: contaminated,
    )

    with pytest.raises(ValueError, match="Optional augmentation package is outside"):
        run_yolo_workbench_preview(
            mode="official",
            experiment_config_path=repository_root / "config.yaml",
            dataset_root=repository_root / "dataset",
            output_root=repository_root / "outputs",
            repository_root=repository_root,
            train_sample_ids=["train-a"],
            representation_sample_id=None,
            research_overrides={},
        )


# ADD 2026-08-28: Official runner가 exact uv CLI args를 전달하고 review paths를 복원하는지 검증한다.
@pytest.mark.parametrize(
    ("experiment_id", "config_name"),
    [
        (
            "c4_2a_yolo11n_seg_imgsz1024_seed42",
            "c4_2a_yolo11n_seg_imgsz1024_seed42.yaml",
        ),
        (
            "c4_2b_yolo11n_seg_component_aware_sampling_x2_seed42",
            "c4_2b_yolo11n_seg_component_aware_sampling_x2_seed42.yaml",
        ),
    ],
)
def test_official_training_subprocess_uses_exact_cli_and_restores_review_paths(
    tmp_path: Path,
    experiment_id: str,
    config_name: str,
) -> None:
    repository_root = tmp_path / "repo"
    config_path = Path("configs/experiments/yolo_segmentation") / config_name
    assert config_path.is_file()
    dataset_root = repository_root / "data/dataset"
    baseline_dir = repository_root / "artifacts/baseline"
    experiment_root = repository_root / "outputs/experiments"
    artifact_root = repository_root / "artifacts/candidates"
    package_root = repository_root / "outputs/packages"
    provenance = _environment(repository_root)
    commands: list[list[str]] = []

    def fake_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[5] == "ml.experiments.yolo_workbench_runtime":
            assert kwargs == {
                "cwd": repository_root,
                "check": True,
                "capture_output": True,
                "text": True,
            }
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=ENVIRONMENT_JSON_PREFIX + json.dumps(provenance.to_json_dict()),
                stderr="",
            )
        assert command == [
            "uv",
            "run",
            "--locked",
            "python",
            "-m",
            "pipelines.run_yolo_segmentation_experiment",
            "--experiment-config",
            str(config_path),
            "--dataset",
            str(dataset_root),
            "--baseline-artifact-dir",
            str(baseline_dir),
            "--device",
            "cuda",
            "--repository-root",
            str(repository_root),
        ]
        assert kwargs == {"cwd": repository_root, "check": True}
        experiment_dir = experiment_root / experiment_id
        experiment_dir.mkdir(parents=True)
        for filename in (
            "experiment_result.json",
            "comparison_to_baseline.json",
            "resource_telemetry.json",
            "package_metadata.json",
        ):
            (experiment_dir / filename).write_text("{}\n", encoding="utf-8")
        candidate_dir = artifact_root / experiment_id
        candidate_dir.mkdir(parents=True)
        (candidate_dir / "model.pt").write_bytes(b"model")
        (candidate_dir / "metadata.json").write_text("{}\n", encoding="utf-8")
        package_root.mkdir(parents=True)
        (package_root / f"{experiment_id}.zip").write_bytes(b"zip")
        return subprocess.CompletedProcess(command, 0)

    artifacts = run_official_training_subprocess(
        experiment_config_path=config_path,
        experiment_id=experiment_id,
        experiment_root=experiment_root,
        artifact_root=artifact_root,
        package_root=package_root,
        dataset_root=dataset_root,
        baseline_artifact_dir=baseline_dir,
        repository_root=repository_root,
        subprocess_runner=fake_runner,
    )

    assert isinstance(artifacts, ExperimentReviewArtifacts)
    assert artifacts.experiment_dir == experiment_root / experiment_id
    assert artifacts.candidate_artifact_dir == artifact_root / experiment_id
    assert artifacts.package_path == package_root / f"{experiment_id}.zip"
    assert len(commands) == 2
    assert "overrides" not in inspect.signature(run_official_training_subprocess).parameters


# ADD 2026-08-28: Official subprocess nonzero exit를 artifact 복원 없이 그대로 전파한다.
def test_official_training_subprocess_fails_fast(tmp_path: Path) -> None:
    provenance = _environment(tmp_path)

    def failing_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if command[5] == "ml.experiments.yolo_workbench_runtime":
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=ENVIRONMENT_JSON_PREFIX + json.dumps(provenance.to_json_dict()),
                stderr="",
            )
        raise subprocess.CalledProcessError(1, command)

    with pytest.raises(subprocess.CalledProcessError):
        run_official_training_subprocess(
            experiment_config_path=tmp_path / "config.yaml",
            experiment_id="fixture",
            experiment_root=tmp_path / "experiments",
            artifact_root=tmp_path / "artifacts",
            package_root=tmp_path / "packages",
            dataset_root=tmp_path / "dataset",
            baseline_artifact_dir=tmp_path / "baseline",
            repository_root=tmp_path,
            subprocess_runner=failing_runner,
        )


# ADD 2026-08-28: Preview도 uv CLI에서 실행되고 system-only package metadata를 사용하지 않는다.
def test_locked_preview_subprocess_uses_uv_boundary_and_reads_artifacts(tmp_path: Path) -> None:
    repository_root = tmp_path / "repo"
    output_root = repository_root / "outputs/workbench"

    def fake_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert command[:6] == [
            "uv",
            "run",
            "--locked",
            "python",
            "-m",
            "pipelines.run_yolo_workbench_preview",
        ]
        assert "--train-sample-id" in command
        assert "--representation-sample-id" in command
        assert not any(argument.startswith("--research-") for argument in command)
        assert kwargs == {"cwd": repository_root, "check": True}
        augmentation_path = output_root / "augmentation.png"
        representation_path = output_root / "representation.png"
        output_root.mkdir(parents=True)
        augmentation_path.write_bytes(b"png")
        representation_path.write_bytes(b"png")
        (output_root / "preview_metadata.json").write_text(
            json.dumps(
                {
                    "augmentation": {
                        "generated_path": str(augmentation_path),
                        "albumentations": {
                            "available_in_locked_environment": False,
                            "active_in_actual_transform": False,
                        },
                    },
                    "representation": {"generated_path": str(representation_path)},
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0)

    artifacts = run_locked_preview_subprocess(
        mode="official",
        experiment_config_path=repository_root / "config.yaml",
        dataset_root=repository_root / "dataset",
        output_root=output_root,
        repository_root=repository_root,
        train_sample_ids=["train-a"],
        representation_sample_id="val-a",
        research_overrides={},
        subprocess_runner=fake_runner,
    )

    assert artifacts.augmentation_figure.is_file()
    assert artifacts.representation_figure is not None
    assert artifacts.metadata["augmentation"]["albumentations"] == {
        "available_in_locked_environment": False,
        "active_in_actual_transform": False,
    }


# ADD 2026-08-28: Preview subprocess nonzero exit를 artifact read 전에 그대로 전파한다.
def test_locked_preview_subprocess_fails_fast(tmp_path: Path) -> None:
    def failing_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(1, command)

    with pytest.raises(subprocess.CalledProcessError):
        run_locked_preview_subprocess(
            mode="official",
            experiment_config_path=tmp_path / "config.yaml",
            dataset_root=tmp_path / "dataset",
            output_root=tmp_path / "output",
            repository_root=tmp_path,
            train_sample_ids=["train-a"],
            representation_sample_id=None,
            research_overrides={},
            subprocess_runner=failing_runner,
        )


# ADD 2026-08-28: Official preview override가 subprocess 시작 전에 거부되는지 검증한다.
def test_locked_official_preview_rejects_overrides(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="rejects"):
        run_locked_preview_subprocess(
            mode="official",
            experiment_config_path=tmp_path / "config.yaml",
            dataset_root=tmp_path / "dataset",
            output_root=tmp_path / "output",
            repository_root=tmp_path,
            train_sample_ids=["train-a"],
            representation_sample_id=None,
            research_overrides={"imgsz": 768},
        )


# ADD 2026-08-28: C4-2B preview가 representation 없이 augmentation artifact를 복원한다.
def test_locked_preview_allows_sampling_experiment_without_representation(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"

    def fake_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        output_root.mkdir(parents=True)
        augmentation_path = output_root / "augmentation.png"
        augmentation_path.write_bytes(b"png")
        (output_root / "preview_metadata.json").write_text(
            json.dumps(
                {
                    "augmentation": {"generated_path": str(augmentation_path)},
                    "representation": None,
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0)

    artifacts = run_locked_preview_subprocess(
        mode="official",
        experiment_config_path=tmp_path / "c4_2b.yaml",
        dataset_root=tmp_path / "dataset",
        output_root=output_root,
        repository_root=tmp_path,
        train_sample_ids=["train-a"],
        representation_sample_id=None,
        research_overrides={},
        subprocess_runner=fake_runner,
    )

    assert artifacts.augmentation_figure.is_file()
    assert artifacts.representation_figure is None
