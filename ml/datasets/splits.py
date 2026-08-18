"""Deterministic split utilities for dataset preparation."""

from pathlib import Path
from random import Random


def deterministic_train_validation_split(
    image_paths: list[Path],
    validation_ratio: float,
    random_seed: int,
) -> tuple[list[Path], list[Path]]:
    """Split normal training images reproducibly into train and validation sets."""
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1.")

    sorted_paths = sorted(image_paths)
    if len(sorted_paths) < 2:
        raise ValueError("At least two images are required to create a validation split.")

    shuffled_paths = sorted_paths.copy()
    Random(random_seed).shuffle(shuffled_paths)

    validation_count = max(1, round(len(shuffled_paths) * validation_ratio))
    validation_count = min(validation_count, len(shuffled_paths) - 1)

    validation_paths = sorted(shuffled_paths[:validation_count])
    train_paths = sorted(shuffled_paths[validation_count:])

    return train_paths, validation_paths
