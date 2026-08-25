"""Concurrency contracts for dual-model inference orchestration."""

from __future__ import annotations

import asyncio
from threading import Barrier

import pytest

from services.inference.combined import run_combined_inference


# ADD 2026-08-26: Barrier로 두 model call이 실제로 서로 다른 worker에서 겹치는지 검증한다.
def test_combined_inference_overlaps_runtime_calls_without_sleep() -> None:
    rendezvous = Barrier(2, timeout=2.0)

    def patchcore_call() -> str:
        rendezvous.wait()
        return "patchcore"

    def known_defect_call() -> str:
        rendezvous.wait()
        return "yolo"

    result = asyncio.run(run_combined_inference(patchcore_call, known_defect_call))

    assert result.patchcore == "patchcore"
    assert result.known_defect == "yolo"
    assert result.patchcore_inference_ms >= 0.0
    assert result.orchestration_ms >= result.patchcore_inference_ms


# ADD 2026-08-26: 어느 한 branch failure도 successful combined result로 반환되지 않음을 검증한다.
def test_combined_inference_propagates_model_failure() -> None:
    rendezvous = Barrier(2, timeout=2.0)

    def failing_call() -> str:
        rendezvous.wait()
        raise RuntimeError("private model detail")

    def successful_call() -> str:
        rendezvous.wait()
        return "yolo"

    with pytest.raises(RuntimeError, match="private model detail"):
        asyncio.run(run_combined_inference(failing_call, successful_call))
