"""Tests for deterministic dataset split behavior."""

from pathlib import Path

import pytest

from ml.datasets.splits import deterministic_train_validation_split


# ADD 2026-08-18: split is deterministic for same seed 테스트 시나리오를 검증한다.
def test_split_is_deterministic_for_same_seed() -> None:
    images = [Path(f"{index:03d}.png") for index in range(20)]

    first_train, first_validation = deterministic_train_validation_split(
        images,
        validation_ratio=0.2,
        random_seed=42,
    )
    second_train, second_validation = deterministic_train_validation_split(
        images,
        validation_ratio=0.2,
        random_seed=42,
    )

    assert first_train == second_train
    assert first_validation == second_validation


# ADD 2026-08-18: split is disjoint and preserves all samples 테스트 시나리오를 검증한다.
def test_split_is_disjoint_and_preserves_all_samples() -> None:
    images = [Path(f"{index:03d}.png") for index in range(20)]

    train, validation = deterministic_train_validation_split(
        images,
        validation_ratio=0.2,
        random_seed=42,
    )

    assert len(train) == 16
    assert len(validation) == 4
    assert set(train).isdisjoint(validation)
    assert set(train) | set(validation) == set(images)


# ADD 2026-08-18: split rejects invalid validation ratio 테스트 시나리오를 검증한다.
def test_split_rejects_invalid_validation_ratio() -> None:
    images = [Path("000.png"), Path("001.png")]

    with pytest.raises(ValueError, match="validation_ratio"):
        deterministic_train_validation_split(
            images,
            validation_ratio=1.0,
            random_seed=42,
        )
