"""Tests for explicit and automatic device resolution."""

import pytest
import torch

from ml.training.device import resolve_device


# ADD 2026-08-19: auto prefers cuda over mps 테스트 시나리오를 검증한다.
def test_auto_prefers_cuda_over_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert resolve_device("auto") == torch.device("cuda")


# ADD 2026-08-19: auto uses mps then cpu 테스트 시나리오를 검증한다.
def test_auto_uses_mps_then_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("auto") == torch.device("mps")

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device("auto") == torch.device("cpu")


# ADD 2026-08-19: explicit unavailable accelerator raises 테스트 시나리오를 검증한다.
def test_explicit_unavailable_accelerator_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        resolve_device("cuda")
    with pytest.raises(RuntimeError, match="MPS was requested"):
        resolve_device("mps")


# ADD 2026-08-19: invalid device raises 테스트 시나리오를 검증한다.
def test_invalid_device_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported device"):
        resolve_device("tpu")
