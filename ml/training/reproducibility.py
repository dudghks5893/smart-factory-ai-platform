"""Random seed controls for reproducible PatchCore artifact construction."""

import random

import numpy as np
import torch


# ADD 2026-08-19: Seed Python, NumPy, and PyTorch RNGs before PatchCore training starts.
def seed_training(random_seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs before PatchCore training starts."""
    if random_seed < 0:
        raise ValueError("random_seed must be non-negative.")

    # KCenterGreedy와 SparseRandomProjection이 사용하는 모든 host RNG를 고정한다.
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # CUDA 환경에서는 모든 device RNG와 cuDNN algorithm 선택도 결정적으로 설정한다.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
