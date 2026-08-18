from pathlib import Path

REQUIRED_DIRECTORIES = (
    "apps/dashboard",
    "services/api",
    "services/inference",
    "services/rag",
    "ml/datasets",
    "ml/training",
    "ml/evaluation",
    "pipelines",
    "configs",
    "manuals",
    "monitoring",
    "infra",
    "tests",
    "docs",
)


def test_required_project_directories_exist() -> None:
    project_root = Path(__file__).resolve().parents[2]

    missing_directories = [
        directory for directory in REQUIRED_DIRECTORIES if not (project_root / directory).is_dir()
    ]

    assert not missing_directories, f"Missing required project directories: {missing_directories}"
