"""Device selection shared across local and accelerator environments."""

from typing import Literal

import torch

type DeviceName = Literal["auto", "cpu", "mps", "cuda"]

SUPPORTED_DEVICES: tuple[DeviceName, ...] = ("auto", "cpu", "mps", "cuda")


# ADD 2026-08-19: Resolve an execution device without silently falling back for explicit requests.
def resolve_device(requested: str) -> torch.device:
    """Resolve an execution device without silently falling back for explicit requests."""
    if requested not in SUPPORTED_DEVICES:
        choices = ", ".join(SUPPORTED_DEVICES)
        raise ValueError(f"Unsupported device '{requested}'. Expected one of: {choices}.")

    if requested == "auto":
        # auto에서만 CUDA, MPS, CPU 우선순위에 따라 fallback한다.
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # 명시적으로 요청한 accelerator가 없으면 조용히 CPU로 변경하지 않는다.
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in the current environment.")

    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available in the current environment.")

    return torch.device(requested)
