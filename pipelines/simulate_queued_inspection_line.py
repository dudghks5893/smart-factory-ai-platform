"""Run a bounded production-line queue against the production prediction API."""

from __future__ import annotations

import argparse
import math
import queue
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

from pipelines.simulate_inspection_line import (
    DEFAULT_ANOMALY_SOURCE_RATIO,
    DEFAULT_API_BASE_URL,
    DEFAULT_CATEGORY,
    DEFAULT_DATASET_ROOT,
    DEFAULT_EVENT_COUNT,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    PRODUCTION_DEMO_PROFILE,
    Clock,
    EventWriter,
    LineEventResult,
    LineScheduleEvent,
    PredictionTransport,
    Sleeper,
    _nonnegative_float,
    _positive_float,
    _positive_int,
    _ratio,
    build_prediction_url,
    build_production_demo_schedule,
    request_line_prediction,
)

DEFAULT_CAPTURE_INTERVAL_SECONDS = 0.2
DEFAULT_QUEUE_SIZE = 8
QUEUE_OPERATION_TIMEOUT_SECONDS = 0.05


@dataclass
class QueuedLineEvent:
    """One captured event and its monotonic producer timestamps."""

    event: LineScheduleEvent
    scheduled_at: float
    captured_at: float
    enqueued_at: float | None = None
    enqueue_ready: threading.Event = field(default_factory=threading.Event, repr=False)


@dataclass(frozen=True)
class QueuedLineEventResult:
    """One API result plus queue and end-to-end timing boundaries."""

    prediction: LineEventResult
    scheduled_at: float
    captured_at: float
    enqueued_at: float
    dequeued_at: float
    request_ended_at: float
    queue_wait_ms: float
    end_to_end_ms: float
    queue_depth: int


@dataclass(frozen=True)
class QueuedLineSimulationSummary:
    """Producer, bounded queue, worker and observed prediction metrics."""

    requested_events: int
    enqueued_events: int
    successful_inspections: int
    failed_inspections: int
    normal_source_events: int
    anomaly_source_events: int
    normal_predictions: int
    anomaly_predictions: int
    unique_inspection_ids: int
    capture_interval_seconds: float
    queue_size: int
    maximum_queue_depth: int
    producer_blocked_count: int
    producer_blocked_seconds: float
    average_queue_wait_ms: float
    p95_queue_wait_ms: float
    average_request_ms: float
    elapsed_seconds: float
    throughput_per_second: float

    # ADD 2026-08-24: Configured cadence를 inference throughput과 분리해 해석하도록 제공한다.
    @property
    def configured_capture_rate(self) -> float | None:
        if self.capture_interval_seconds == 0:
            return None
        return 1 / self.capture_interval_seconds


class QueuedLineSimulationError(RuntimeError):
    """Fail-fast queued simulation error with a completed partial summary."""

    # ADD 2026-08-24: Worker/producer 실패 시 partial metrics를 CLI에 보존한다.
    def __init__(self, message: str, summary: QueuedLineSimulationSummary) -> None:
        super().__init__(message)
        self.summary = summary


@dataclass
class _ProducerState:
    enqueued_events: int = 0
    maximum_queue_depth: int = 0
    blocked_count: int = 0
    blocked_seconds: float = 0.0


@dataclass
class _WorkerState:
    results: list[QueuedLineEventResult] = field(default_factory=list)
    inspection_ids: set[UUID] = field(default_factory=set)
    failure: Exception | None = None


@dataclass(frozen=True)
class _StopWorker:
    """Sentinel placed after every captured event has been enqueued."""


type _QueueEntry = QueuedLineEvent | _StopWorker


# ADD 2026-08-24: Deterministic producer와 단일 HTTP worker를 bounded queue로 조율한다.
def simulate_queued_inspection_line(
    *,
    events: Sequence[LineScheduleEvent],
    api_base_url: str,
    capture_interval_seconds: float = DEFAULT_CAPTURE_INTERVAL_SECONDS,
    queue_size: int = DEFAULT_QUEUE_SIZE,
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    transport: PredictionTransport | None = None,
    sleeper: Sleeper = time.sleep,
    clock: Clock = time.perf_counter,
    event_writer: EventWriter = print,
) -> tuple[tuple[QueuedLineEventResult, ...], QueuedLineSimulationSummary]:
    """Capture on producer cadence and process every event through one HTTP worker."""
    _validate_queued_parameters(
        events=events,
        capture_interval_seconds=capture_interval_seconds,
        queue_size=queue_size,
        request_timeout_seconds=request_timeout_seconds,
    )
    prediction_url = build_prediction_url(api_base_url)
    work_queue: queue.Queue[_QueueEntry] = queue.Queue(maxsize=queue_size)
    stop_event = threading.Event()
    producer_state = _ProducerState()
    worker_state = _WorkerState()
    started_at = clock()

    # 한 non-daemon worker만 시작해 HTTP processing order를 schedule order와 동일하게 유지한다.
    worker = threading.Thread(
        target=_run_worker,
        kwargs={
            "work_queue": work_queue,
            "prediction_url": prediction_url,
            "request_timeout_seconds": request_timeout_seconds,
            "transport": transport,
            "clock": clock,
            "event_writer": event_writer,
            "stop_event": stop_event,
            "worker_state": worker_state,
        },
        name="inspection-inference-worker",
        daemon=False,
    )
    worker.start()
    producer_failure: Exception | None = None
    try:
        _produce_events(
            events=events,
            capture_interval_seconds=capture_interval_seconds,
            work_queue=work_queue,
            stop_event=stop_event,
            producer_state=producer_state,
            sleeper=sleeper,
            clock=clock,
            started_at=started_at,
        )
        _enqueue_stop_signal(work_queue=work_queue, stop_event=stop_event)
    except Exception as exc:
        producer_failure = exc
        stop_event.set()

    # 정상/실패 모두 worker를 명시적으로 join하고 남은 queue bookkeeping을 정리한다.
    worker.join()
    _discard_remaining_entries(work_queue)
    failure = worker_state.failure if worker_state.failure is not None else producer_failure
    summary = _build_queued_summary(
        events=events,
        results=worker_state.results,
        producer_state=producer_state,
        failed_inspections=1 if failure is not None else 0,
        capture_interval_seconds=capture_interval_seconds,
        queue_size=queue_size,
        elapsed_seconds=max(0.0, clock() - started_at),
    )
    if failure is not None:
        raise QueuedLineSimulationError(
            f"Queued line simulation stopped: {failure}", summary
        ) from failure
    if summary.enqueued_events != len(events) or summary.successful_inspections != len(events):
        raise QueuedLineSimulationError(
            "Queued line simulation completed with inconsistent event counts.", summary
        )
    return tuple(worker_state.results), summary


# ADD 2026-08-24: Absolute deadline으로 request latency와 독립적인 cadence를 유지한다.
def _produce_events(
    *,
    events: Sequence[LineScheduleEvent],
    capture_interval_seconds: float,
    work_queue: queue.Queue[_QueueEntry],
    stop_event: threading.Event,
    producer_state: _ProducerState,
    sleeper: Sleeper,
    clock: Clock,
    started_at: float,
) -> None:
    for index, event in enumerate(events):
        scheduled_at = started_at + index * capture_interval_seconds
        _wait_for_capture_deadline(
            deadline=scheduled_at,
            stop_event=stop_event,
            sleeper=sleeper,
            clock=clock,
        )
        if stop_event.is_set():
            raise RuntimeError("Inference worker failed before producer completion.")
        captured_at = clock()
        queued_event = QueuedLineEvent(
            event=event,
            scheduled_at=scheduled_at,
            captured_at=captured_at,
        )

        # Queue가 가득 차면 기다리되 worker failure는 bounded polling으로 감지한다.
        blocked, blocked_seconds, observed_depth = _enqueue_event(
            work_queue=work_queue,
            queued_event=queued_event,
            stop_event=stop_event,
            clock=clock,
        )
        producer_state.enqueued_events += 1
        producer_state.maximum_queue_depth = max(producer_state.maximum_queue_depth, observed_depth)
        if blocked:
            producer_state.blocked_count += 1
            producer_state.blocked_seconds += blocked_seconds


# ADD 2026-08-24: Queue-full에서 no-drop blocking을 적용하면서 worker failure deadlock을 방지한다.
def _enqueue_event(
    *,
    work_queue: queue.Queue[_QueueEntry],
    queued_event: QueuedLineEvent,
    stop_event: threading.Event,
    clock: Clock,
) -> tuple[bool, float, int]:
    put_started_at = clock()
    blocked = work_queue.full()
    observed_depth = work_queue.maxsize if blocked else work_queue.qsize()
    while True:
        if stop_event.is_set():
            raise RuntimeError("Inference worker stopped before event enqueue completed.")
        try:
            work_queue.put(queued_event, timeout=QUEUE_OPERATION_TIMEOUT_SECONDS)
            break
        except queue.Full:
            blocked = True
            observed_depth = work_queue.maxsize
    enqueued_at = clock()
    queued_event.enqueued_at = enqueued_at
    queued_event.enqueue_ready.set()
    observed_depth = max(observed_depth, work_queue.qsize())
    blocked_seconds = max(0.0, enqueued_at - put_started_at) if blocked else 0.0
    return blocked, blocked_seconds, observed_depth


# ADD 2026-08-24: 단일 worker가 dequeue, 실제 HTTP, validation, aggregation을 순서대로 수행한다.
def _run_worker(
    *,
    work_queue: queue.Queue[_QueueEntry],
    prediction_url: str,
    request_timeout_seconds: float,
    transport: PredictionTransport | None,
    clock: Clock,
    event_writer: EventWriter,
    stop_event: threading.Event,
    worker_state: _WorkerState,
) -> None:
    while True:
        if stop_event.is_set():
            return
        try:
            entry = work_queue.get(timeout=QUEUE_OPERATION_TIMEOUT_SECONDS)
        except queue.Empty:
            continue
        try:
            if isinstance(entry, _StopWorker):
                return
            entry.enqueue_ready.wait()
            if entry.enqueued_at is None:
                raise RuntimeError("Queued event is missing its enqueue timestamp.")
            dequeued_at = clock()

            # B1과 동일한 image load, HTTP transport와 response schema validation을 재사용한다.
            prediction = request_line_prediction(
                event=entry.event,
                prediction_url=prediction_url,
                request_timeout_seconds=request_timeout_seconds,
                transport=transport,
                clock=clock,
            )
            if prediction.inspection_id in worker_state.inspection_ids:
                raise ValueError("Prediction API returned a duplicate inspection_id.")
            request_ended_at = clock()
            result = QueuedLineEventResult(
                prediction=prediction,
                scheduled_at=entry.scheduled_at,
                captured_at=entry.captured_at,
                enqueued_at=entry.enqueued_at,
                dequeued_at=dequeued_at,
                request_ended_at=request_ended_at,
                queue_wait_ms=max(0.0, (dequeued_at - entry.enqueued_at) * 1000),
                end_to_end_ms=max(0.0, (request_ended_at - entry.captured_at) * 1000),
                queue_depth=work_queue.qsize(),
            )
            worker_state.inspection_ids.add(prediction.inspection_id)
            worker_state.results.append(result)
            event_writer(format_queued_event(result))
        except Exception as exc:
            worker_state.failure = exc
            stop_event.set()
            if isinstance(entry, QueuedLineEvent):
                event_writer(_format_failed_queued_event(entry.event, exc))
            return
        finally:
            work_queue.task_done()


# ADD 2026-08-24: Producer cadence deadline까지 sleep하되 worker failure를 먼저 확인한다.
def _wait_for_capture_deadline(
    *,
    deadline: float,
    stop_event: threading.Event,
    sleeper: Sleeper,
    clock: Clock,
) -> None:
    remaining = deadline - clock()
    if remaining <= 0:
        return
    if stop_event.is_set():
        raise RuntimeError("Inference worker failed before the next capture deadline.")
    sleeper(remaining)


# ADD 2026-08-24: 정상 producer 완료 뒤 sentinel을 전달하며 queue 공간을 기다린다.
def _enqueue_stop_signal(
    *,
    work_queue: queue.Queue[_QueueEntry],
    stop_event: threading.Event,
) -> None:
    while True:
        if stop_event.is_set():
            raise RuntimeError("Inference worker failed before normal shutdown.")
        try:
            work_queue.put(_StopWorker(), timeout=QUEUE_OPERATION_TIMEOUT_SECONDS)
            return
        except queue.Full:
            continue


# ADD 2026-08-24: 실패 후 처리되지 않은 queue entry의 unfinished-task bookkeeping을 해제한다.
def _discard_remaining_entries(work_queue: queue.Queue[_QueueEntry]) -> None:
    while True:
        try:
            work_queue.get_nowait()
        except queue.Empty:
            return
        work_queue.task_done()


# ADD 2026-08-24: Queue behavior와 HTTP capacity를 분리한 final summary를 계산한다.
def _build_queued_summary(
    *,
    events: Sequence[LineScheduleEvent],
    results: Sequence[QueuedLineEventResult],
    producer_state: _ProducerState,
    failed_inspections: int,
    capture_interval_seconds: float,
    queue_size: int,
    elapsed_seconds: float,
) -> QueuedLineSimulationSummary:
    queue_waits = [result.queue_wait_ms for result in results]
    request_times = [result.prediction.elapsed_ms for result in results]
    successful = len(results)
    return QueuedLineSimulationSummary(
        requested_events=len(events),
        enqueued_events=producer_state.enqueued_events,
        successful_inspections=successful,
        failed_inspections=failed_inspections,
        normal_source_events=sum(event.source_kind == "normal" for event in events),
        anomaly_source_events=sum(event.source_kind == "anomaly" for event in events),
        normal_predictions=sum(not result.prediction.is_anomaly for result in results),
        anomaly_predictions=sum(result.prediction.is_anomaly for result in results),
        unique_inspection_ids=len({result.prediction.inspection_id for result in results}),
        capture_interval_seconds=capture_interval_seconds,
        queue_size=queue_size,
        maximum_queue_depth=producer_state.maximum_queue_depth,
        producer_blocked_count=producer_state.blocked_count,
        producer_blocked_seconds=producer_state.blocked_seconds,
        average_queue_wait_ms=(sum(queue_waits) / successful if successful else 0.0),
        p95_queue_wait_ms=_percentile(queue_waits, 0.95),
        average_request_ms=(sum(request_times) / successful if successful else 0.0),
        elapsed_seconds=elapsed_seconds,
        throughput_per_second=(successful / elapsed_seconds if elapsed_seconds > 0 else 0.0),
    )


# ADD 2026-08-24: Queue event result를 concise one-line operations output으로 변환한다.
def format_queued_event(result: QueuedLineEventResult) -> str:
    prediction = result.prediction
    label = "ANOMALY" if prediction.is_anomaly else "NORMAL"
    return (
        f"[{prediction.event.sequence:03d}/{prediction.event.total:03d}] "
        f"result={label} score={prediction.anomaly_score:.6f} "
        f"queue_wait={result.queue_wait_ms:.3f}ms "
        f"request={prediction.elapsed_ms:.3f}ms queue_depth={result.queue_depth} "
        f"inspection_id={prediction.inspection_id}"
    )


# ADD 2026-08-24: Producer/backpressure와 worker processing metrics를 final output으로 렌더링한다.
def format_queued_summary(summary: QueuedLineSimulationSummary) -> str:
    configured_rate = (
        "unbounded"
        if summary.configured_capture_rate is None
        else f"{summary.configured_capture_rate:.3f} events/sec"
    )
    return "\n".join(
        (
            f"Requested events: {summary.requested_events}",
            f"Enqueued events: {summary.enqueued_events}",
            f"Successful inspections: {summary.successful_inspections}",
            f"Failed inspections: {summary.failed_inspections}",
            (
                "Input sources (normal/anomaly): "
                f"{summary.normal_source_events}/{summary.anomaly_source_events}"
            ),
            (
                "Observed predictions (NORMAL/ANOMALY): "
                f"{summary.normal_predictions}/{summary.anomaly_predictions}"
            ),
            f"Unique inspection IDs: {summary.unique_inspection_ids}",
            f"Capture interval: {summary.capture_interval_seconds:.3f} seconds",
            f"Configured capture rate: {configured_rate}",
            f"Queue size: {summary.queue_size}",
            f"Maximum observed queue depth: {summary.maximum_queue_depth}",
            f"Producer blocked count: {summary.producer_blocked_count}",
            f"Producer total blocked time: {summary.producer_blocked_seconds:.6f} seconds",
            f"Average queue wait: {summary.average_queue_wait_ms:.3f} ms",
            f"p95 queue wait: {summary.p95_queue_wait_ms:.3f} ms",
            f"Average HTTP request time: {summary.average_request_ms:.3f} ms",
            f"Total wall-clock elapsed time: {summary.elapsed_seconds:.3f} seconds",
            f"Processing throughput: {summary.throughput_per_second:.3f} inspections/sec",
        )
    )


# ADD 2026-08-24: Linear interpolation으로 small-sample queue wait percentile을 계산한다.
def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = position - lower_index
    return ordered[lower_index] * (1 - weight) + ordered[upper_index] * weight


# ADD 2026-08-24: Queued runtime count, cadence, queue와 timeout bounds를 사전 검증한다.
def _validate_queued_parameters(
    *,
    events: Sequence[LineScheduleEvent],
    capture_interval_seconds: float,
    queue_size: int,
    request_timeout_seconds: float,
) -> None:
    if not events:
        raise ValueError("Queued line simulation requires at least one scheduled event.")
    if not math.isfinite(capture_interval_seconds) or capture_interval_seconds < 0:
        raise ValueError("capture_interval_seconds must be finite and non-negative.")
    if queue_size <= 0:
        raise ValueError("queue_size must be positive.")
    if not math.isfinite(request_timeout_seconds) or request_timeout_seconds <= 0:
        raise ValueError("request_timeout_seconds must be finite and positive.")


# ADD 2026-08-24: Failed worker event를 image metadata 없이 concise error output으로 변환한다.
def _format_failed_queued_event(event: LineScheduleEvent, exc: Exception) -> str:
    return f"[{event.sequence:03d}/{event.total:03d}] http=FAILED error={type(exc).__name__}"


# ADD 2026-08-24: Queued production-demo CLI argument contract를 정의한다.
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Queue deterministic real MVTec images for one production HTTP worker."
    )
    parser.add_argument(
        "--profile", choices=(PRODUCTION_DEMO_PROFILE,), default=PRODUCTION_DEMO_PROFILE
    )
    parser.add_argument("--api-base-url", default=DEFAULT_API_BASE_URL)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--category", default=DEFAULT_CATEGORY)
    parser.add_argument("--count", type=_positive_int, default=DEFAULT_EVENT_COUNT)
    parser.add_argument(
        "--anomaly-source-ratio",
        type=_ratio,
        default=DEFAULT_ANOMALY_SOURCE_RATIO,
    )
    parser.add_argument(
        "--capture-interval-seconds",
        type=_nonnegative_float,
        default=DEFAULT_CAPTURE_INTERVAL_SECONDS,
    )
    parser.add_argument("--queue-size", type=_positive_int, default=DEFAULT_QUEUE_SIZE)
    parser.add_argument(
        "--request-timeout-seconds",
        type=_positive_float,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
    )
    return parser.parse_args()


# ADD 2026-08-24: Schedule, bounded producer/worker run과 fail-fast exit status를 조율한다.
def main() -> int:
    args = _parse_args()

    # B1과 동일한 validated deterministic schedule을 producer input으로 구성한다.
    events = build_production_demo_schedule(
        dataset_root=args.dataset_root,
        manifest_path=args.manifest,
        category=args.category,
        count=args.count,
        anomaly_source_ratio=args.anomaly_source_ratio,
    )
    print(f"Production line profile: {args.profile}")
    print("Execution mode: queued-single-worker")
    try:
        _, summary = simulate_queued_inspection_line(
            events=events,
            api_base_url=args.api_base_url,
            capture_interval_seconds=args.capture_interval_seconds,
            queue_size=args.queue_size,
            request_timeout_seconds=args.request_timeout_seconds,
        )
    except QueuedLineSimulationError as exc:
        print("Queued production line simulation: FAILED")
        print(format_queued_summary(exc.summary))
        print(f"Error: {exc}")
        return 1

    print("Queued production line simulation: PASS")
    print(format_queued_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
