"""Tests for PatchCore training RNG controls."""

import random

import numpy as np
import pytest
import torch

from ml.training.reproducibility import seed_training


# ADD 2026-08-19: seed training repeats python numpy and torch values 테스트 시나리오를 검증한다.
def test_seed_training_repeats_python_numpy_and_torch_values() -> None:
    seed_training(42)
    first = (random.random(), np.random.random(), torch.rand(3))

    seed_training(42)
    second = (random.random(), np.random.random(), torch.rand(3))

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


# ADD 2026-08-19: seed training rejects negative seed 테스트 시나리오를 검증한다.
def test_seed_training_rejects_negative_seed() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        seed_training(-1)
