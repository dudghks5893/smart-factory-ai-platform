"""Parallel orchestration for independent, already-loaded inspection runtimes."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

from fastapi.concurrency import run_in_threadpool


@dataclass(frozen=True)
class CombinedInferenceResult[PatchResultT, YoloResultT]:
    """Both model results and measured branch/orchestration wall times."""

    patchcore: PatchResultT
    known_defect: YoloResultT
    patchcore_inference_ms: float
    orchestration_ms: float


# ADD 2026-08-26: Independent runtime calls를 별도 threadpool worker에서 동시에 실행한다.
async def run_combined_inference[PatchResultT, YoloResultT](
    patchcore_call: Callable[[], PatchResultT],
    known_defect_call: Callable[[], YoloResultT],
) -> CombinedInferenceResult[PatchResultT, YoloResultT]:
    """Await both branches and return only when neither raised an exception."""

    # PatchCore branch wall time은 runtime 내부 timing이 없으므로 worker 경계에서 측정한다.
    def timed_patchcore_call() -> tuple[PatchResultT, float]:
        started = perf_counter()
        result = patchcore_call()
        return result, (perf_counter() - started) * 1000.0

    started = perf_counter()
    patchcore_task = run_in_threadpool(timed_patchcore_call)
    known_defect_task = run_in_threadpool(known_defect_call)
    patchcore_outcome, known_defect_outcome = await asyncio.gather(
        patchcore_task,
        known_defect_task,
        return_exceptions=True,
    )
    if isinstance(patchcore_outcome, BaseException):
        raise patchcore_outcome
    if isinstance(known_defect_outcome, BaseException):
        raise known_defect_outcome
    patchcore, patchcore_ms = patchcore_outcome
    known_defect = known_defect_outcome
    orchestration_ms = (perf_counter() - started) * 1000.0
    if not all(math.isfinite(value) and value >= 0.0 for value in (patchcore_ms, orchestration_ms)):
        raise RuntimeError("Combined inference produced an invalid timing observation.")
    return CombinedInferenceResult(
        patchcore=patchcore,
        known_defect=known_defect,
        patchcore_inference_ms=patchcore_ms,
        orchestration_ms=orchestration_ms,
    )
