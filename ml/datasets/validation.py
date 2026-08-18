"""Validation utilities for the MVTec AD dataset."""

from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from ml.datasets.constants import (
    GOOD_DIR_NAME,
    IMAGE_SUFFIX,
    MASK_SUFFIX,
    MVTEC_AD_CATEGORIES,
)


@dataclass
class DatasetValidationReport:
    """Structured result for a single MVTec AD category validation."""

    category: str
    train_good_count: int = 0
    test_good_count: int = 0
    test_anomaly_count: int = 0
    mask_count: int = 0
    defect_counts: dict[str, int] = field(default_factory=dict)
    corrupted_files: list[str] = field(default_factory=list)
    missing_masks: list[str] = field(default_factory=list)
    unexpected_masks: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not (
            self.errors or self.corrupted_files or self.missing_masks or self.unexpected_masks
        )


def _list_png_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == IMAGE_SUFFIX
    )


def _validate_image(path: Path) -> str | None:
    try:
        with Image.open(path) as image:
            image.verify()

        with Image.open(path) as image:
            width, height = image.size
            if width <= 0 or height <= 0:
                return f"invalid image size: {path}"
    except (OSError, UnidentifiedImageError) as exc:
        return f"cannot read image: {path} ({exc})"

    return None


def _relative(path: Path, dataset_root: Path) -> str:
    try:
        return str(path.relative_to(dataset_root))
    except ValueError:
        return str(path)


def validate_mvtec_category(dataset_root: Path, category: str) -> DatasetValidationReport:
    """Validate one MVTec AD category without modifying the raw dataset."""
    if category not in MVTEC_AD_CATEGORIES:
        raise ValueError(f"Unsupported MVTec AD category: {category}")

    report = DatasetValidationReport(category=category)
    category_root = dataset_root / category
    train_root = category_root / "train"
    train_good_root = train_root / GOOD_DIR_NAME
    test_root = category_root / "test"
    test_good_root = test_root / GOOD_DIR_NAME
    ground_truth_root = category_root / "ground_truth"

    required_directories = (
        category_root,
        train_root,
        train_good_root,
        test_root,
        test_good_root,
        ground_truth_root,
    )
    for directory in required_directories:
        if not directory.is_dir():
            report.errors.append(f"missing directory: {_relative(directory, dataset_root)}")

    if report.errors:
        return report

    unexpected_train_directories = sorted(
        directory.name
        for directory in train_root.iterdir()
        if directory.is_dir() and directory.name != GOOD_DIR_NAME
    )
    if unexpected_train_directories:
        report.errors.append(
            "unexpected train defect directories: " + ", ".join(unexpected_train_directories)
        )

    train_good_images = _list_png_files(train_good_root)
    test_good_images = _list_png_files(test_good_root)

    anomaly_directories = sorted(
        directory
        for directory in test_root.iterdir()
        if directory.is_dir() and directory.name != GOOD_DIR_NAME
    )
    ground_truth_directories = sorted(
        directory for directory in ground_truth_root.iterdir() if directory.is_dir()
    )

    anomaly_types = {directory.name for directory in anomaly_directories}
    ground_truth_types = {directory.name for directory in ground_truth_directories}

    missing_ground_truth_types = sorted(anomaly_types - ground_truth_types)
    unexpected_ground_truth_types = sorted(ground_truth_types - anomaly_types)

    if missing_ground_truth_types:
        report.errors.append(
            "missing ground-truth directories: " + ", ".join(missing_ground_truth_types)
        )
    if unexpected_ground_truth_types:
        report.errors.append(
            "unexpected ground-truth directories: " + ", ".join(unexpected_ground_truth_types)
        )

    anomaly_images: list[Path] = []
    expected_masks: set[Path] = set()

    for defect_directory in anomaly_directories:
        defect_type = defect_directory.name
        defect_images = _list_png_files(defect_directory)
        report.defect_counts[defect_type] = len(defect_images)
        anomaly_images.extend(defect_images)

        for image_path in defect_images:
            expected_masks.add(ground_truth_root / defect_type / f"{image_path.stem}{MASK_SUFFIX}")

    actual_masks = {
        mask_path
        for defect_directory in ground_truth_directories
        for mask_path in _list_png_files(defect_directory)
    }

    report.train_good_count = len(train_good_images)
    report.test_good_count = len(test_good_images)
    report.test_anomaly_count = len(anomaly_images)
    report.mask_count = len(actual_masks)

    report.missing_masks = sorted(
        _relative(path, dataset_root) for path in expected_masks - actual_masks
    )
    report.unexpected_masks = sorted(
        _relative(path, dataset_root) for path in actual_masks - expected_masks
    )

    files_to_validate = [
        *train_good_images,
        *test_good_images,
        *anomaly_images,
        *sorted(actual_masks),
    ]
    for path in files_to_validate:
        validation_error = _validate_image(path)
        if validation_error is not None:
            report.corrupted_files.append(validation_error)

    return report
