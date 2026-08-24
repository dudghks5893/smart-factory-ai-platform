"""Unit contracts for the bounded inspection queue and single inference worker."""

from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path
from uuid import UUID

import pytest
from PIL import Image

from pipelines.simulate_inspection_line import (
    LineEventResult,
    LineScheduleEvent,
    PredictionRequestError,
    simulate_inspection_line,
)
from pipelines.simulate_queued_inspection_line import (
    QueuedLineEvent,
    QueuedLineEventResult,
    QueuedLineSimulationError,
    _build_queued_summary,
    _parse_args,
    _produce_events,
    _ProducerState,
    _StopWorker,
    format_queued_summary,
    simulate_queued_inspection_line,
)
from services.api.tooling import PreparedImageUpload


# ADD 2026-08-24: Queue simulator test용 real PNG schedule을 작은 fixture로 생성한다.
def _events(tmp_path: Path, count: int) -> tuple[LineScheduleEvent, ...]:
    events: list[LineScheduleEvent] = []
    for index in range(count):
        image_path = tmp_path / f"{index:03d}.png"
        Image.new("RGB", (8, 8), color=(index, 80, 120)).save(image_path)
        events.append(
            LineScheduleEvent(
                sequence=index + 1,
                total=count,
                source_kind="anomaly" if (index + 1) % 10 == 0 else "normal",
                image_path=image_path,
                relative_image_path=f"metal_nut/test/good/{index:03d}.png",
            )
        )
    return tuple(events)


# ADD 2026-08-24: Fake transport 응답용 strict-threshold payload를 생성한다.
def _payload(index: int, *, inspection_id: UUID | None = None) -> dict[str, object]:
    score = 50.0 if index % 10 == 0 else 30.0
    return {
        "inspection_id": str(inspection_id or UUID(int=index)),
        "model_name": "patchcore",
        "category": "metal_nut",
        "is_anomaly": score > 40.0,
        "anomaly_score": score,
        "threshold": 40.0,
        "comparison_operator": ">",
    }


# ADD 2026-08-24: Producer cadence/order와 마지막 capture 뒤 no-sleep을 검증한다.
def test_producer_preserves_independent_cadence_order_and_no_final_sleep(tmp_path: Path) -> None:
    events = _events(tmp_path, 3)
    work_queue: queue.Queue[QueuedLineEvent | _StopWorker] = queue.Queue(maxsize=8)
    producer_state = _ProducerState()
    stop_event = threading.Event()
    current_time = 10.0
    sleeps: list[float] = []

    def clock() -> float:
        return current_time

    def sleeper(seconds: float) -> None:
        nonlocal current_time
        sleeps.append(seconds)
        current_time += seconds

    _produce_events(
        events=events,
        capture_interval_seconds=0.05,
        work_queue=work_queue,
        stop_event=stop_event,
        producer_state=producer_state,
        sleeper=sleeper,
        clock=clock,
        started_at=10.0,
    )
    queued = [work_queue.get_nowait() for _ in events]

    assert all(isinstance(item, QueuedLineEvent) for item in queued)
    assert [item.event.sequence for item in queued if isinstance(item, QueuedLineEvent)] == [
        1,
        2,
        3,
    ]
    assert [item.captured_at for item in queued if isinstance(item, QueuedLineEvent)] == [
        10.0,
        10.05,
        10.1,
    ]
    assert sleeps == pytest.approx([0.05, 0.05])
    assert producer_state.enqueued_events == 3


# ADD 2026-08-24: Bounded queue의 blocking과 no-drop 계약을 검증한다.
def test_bounded_queue_applies_backpressure_without_data_loss(tmp_path: Path) -> None:
    events = _events(tmp_path, 12)
    request_order: list[int] = []

    def transport(_url: str, upload: PreparedImageUpload, _timeout: float) -> object:
        index = int(Path(upload.filename).stem) + 1
        request_order.append(index)
        time.sleep(0.01)
        return _payload(index)

    results, summary = simulate_queued_inspection_line(
        events=events,
        api_base_url="http://api.local:8000",
        capture_interval_seconds=0,
        queue_size=2,
        transport=transport,
        event_writer=lambda _line: None,
    )

    assert request_order == list(range(1, 13))
    assert [result.prediction.event.sequence for result in results] == list(range(1, 13))
    assert summary.requested_events == 12
    assert summary.enqueued_events == 12
    assert summary.successful_inspections == 12
    assert summary.failed_inspections == 0
    assert summary.unique_inspection_ids == 12
    assert summary.maximum_queue_depth == 2
    assert summary.producer_blocked_count > 0
    assert summary.producer_blocked_seconds > 0


# ADD 2026-08-24: Sustainable capture에서 queue와 blocking이 낮은지 검증한다.
def test_sustainable_cadence_does_not_block_producer(tmp_path: Path) -> None:
    events = _events(tmp_path, 4)
    response_index = 0

    def transport(_url: str, _upload: PreparedImageUpload, _timeout: float) -> object:
        nonlocal response_index
        response_index += 1
        return _payload(response_index)

    _, summary = simulate_queued_inspection_line(
        events=events,
        api_base_url="http://api.local:8000",
        capture_interval_seconds=0.01,
        queue_size=8,
        transport=transport,
        event_writer=lambda _line: None,
    )

    assert summary.successful_inspections == 4
    assert summary.maximum_queue_depth <= 1
    assert summary.producer_blocked_count == 0
    assert summary.producer_blocked_seconds == 0


# ADD 2026-08-24: HTTP failure가 producer를 중단하고 non-daemon worker를 남기지 않는지 검증한다.
def test_http_failure_propagates_without_producer_deadlock(tmp_path: Path) -> None:
    events = _events(tmp_path, 20)
    calls = 0
    started_at = time.perf_counter()

    def failing_transport(_url: str, _upload: PreparedImageUpload, _timeout: float) -> object:
        nonlocal calls
        calls += 1
        raise PredictionRequestError("Prediction API returned HTTP 503.")

    with pytest.raises(QueuedLineSimulationError) as exc_info:
        simulate_queued_inspection_line(
            events=events,
            api_base_url="http://api.local:8000",
            capture_interval_seconds=0,
            queue_size=1,
            transport=failing_transport,
            event_writer=lambda _line: None,
        )

    assert time.perf_counter() - started_at < 1
    assert calls == 1
    assert exc_info.value.summary.failed_inspections == 1
    assert exc_info.value.summary.successful_inspections == 0
    assert not any(thread.name == "inspection-inference-worker" for thread in threading.enumerate())


# ADD 2026-08-24: Duplicate inspection ID가 run failure로 전파되는지 검증한다.
def test_duplicate_inspection_id_is_rejected(tmp_path: Path) -> None:
    events = _events(tmp_path, 3)
    duplicate_id = UUID(int=42)

    with pytest.raises(QueuedLineSimulationError, match="duplicate inspection_id") as exc_info:
        simulate_queued_inspection_line(
            events=events,
            api_base_url="http://api.local:8000",
            capture_interval_seconds=0,
            queue_size=2,
            transport=lambda _url, _upload, _timeout: _payload(1, inspection_id=duplicate_id),
            event_writer=lambda _line: None,
        )

    assert exc_info.value.summary.successful_inspections == 1
    assert exc_info.value.summary.unique_inspection_ids == 1
    assert exc_info.value.summary.failed_inspections == 1


# ADD 2026-08-24: Malformed response를 success로 집계하지 않는지 검증한다.
def test_malformed_response_stops_worker(tmp_path: Path) -> None:
    events = _events(tmp_path, 2)

    with pytest.raises(QueuedLineSimulationError) as exc_info:
        simulate_queued_inspection_line(
            events=events,
            api_base_url="http://api.local:8000",
            capture_interval_seconds=0,
            queue_size=1,
            transport=lambda _url, _upload, _timeout: {"unexpected": True},
            event_writer=lambda _line: None,
        )

    assert exc_info.value.summary.successful_inspections == 0
    assert exc_info.value.summary.failed_inspections == 1


# ADD 2026-08-24: Queue wait average/p95와 HTTP average 계산을 검증한다.
def test_queue_timing_aggregation_and_summary_output(tmp_path: Path) -> None:
    events = _events(tmp_path, 4)
    queued_results: list[QueuedLineEventResult] = []
    for index, queue_wait_ms in enumerate((1.0, 2.0, 3.0, 4.0), start=1):
        prediction = LineEventResult(
            event=events[index - 1],
            inspection_id=UUID(int=index),
            model_name="patchcore",
            category="metal_nut",
            is_anomaly=False,
            anomaly_score=30.0,
            threshold=40.0,
            elapsed_ms=float(index * 10),
        )
        queued_results.append(
            QueuedLineEventResult(
                prediction=prediction,
                scheduled_at=0,
                captured_at=0,
                enqueued_at=0,
                dequeued_at=queue_wait_ms / 1000,
                request_ended_at=0.1,
                queue_wait_ms=queue_wait_ms,
                end_to_end_ms=100,
                queue_depth=0,
            )
        )
    producer_state = _ProducerState(enqueued_events=4, maximum_queue_depth=2)

    summary = _build_queued_summary(
        events=events,
        results=queued_results,
        producer_state=producer_state,
        failed_inspections=0,
        capture_interval_seconds=0.05,
        queue_size=2,
        elapsed_seconds=0.5,
    )
    output = format_queued_summary(summary)

    assert summary.average_queue_wait_ms == pytest.approx(2.5)
    assert summary.p95_queue_wait_ms == pytest.approx(3.85)
    assert summary.average_request_ms == pytest.approx(25.0)
    assert summary.throughput_per_second == pytest.approx(8.0)
    assert summary.configured_capture_rate == pytest.approx(20.0)
    assert "Maximum observed queue depth: 2" in output


# ADD 2026-08-24: Sentinel이 모든 event 처리 후 worker를 종료하는지 검증한다.
def test_graceful_sentinel_shutdown_leaves_no_worker(tmp_path: Path) -> None:
    events = _events(tmp_path, 2)
    response_index = 0

    def transport(_url: str, _upload: PreparedImageUpload, _timeout: float) -> object:
        nonlocal response_index
        response_index += 1
        return _payload(response_index)

    results, summary = simulate_queued_inspection_line(
        events=events,
        api_base_url="http://api.local:8000",
        capture_interval_seconds=0,
        queue_size=1,
        transport=transport,
        event_writer=lambda _line: None,
    )

    assert len(results) == 2
    assert summary.enqueued_events == summary.successful_inspections == 2
    assert not any(thread.name == "inspection-inference-worker" for thread in threading.enumerate())


# ADD 2026-08-24: B1 response-complete-then-sleep 순차 regression을 검증한다.
def test_b1_sequential_mode_regression(tmp_path: Path) -> None:
    events = _events(tmp_path, 2)
    calls: list[str] = []
    response_index = 0

    def transport(_url: str, _upload: PreparedImageUpload, _timeout: float) -> object:
        nonlocal response_index
        response_index += 1
        calls.append("request")
        return _payload(response_index)

    def sleeper(_seconds: float) -> None:
        calls.append("sleep")

    results, summary = simulate_inspection_line(
        events=events,
        api_base_url="http://api.local:8000",
        interval_seconds=0.2,
        transport=transport,
        sleeper=sleeper,
        event_writer=lambda _line: None,
    )

    assert calls == ["request", "sleep", "request"]
    assert len(results) == summary.successful_inspections == 2


# ADD 2026-08-24: Invalid queue/capture CLI values가 worker 시작 전에 거부되는지 검증한다.
@pytest.mark.parametrize(
    "arguments",
    [
        ["--queue-size", "0"],
        ["--capture-interval-seconds", "-0.1"],
    ],
)
def test_queued_cli_rejects_invalid_arguments(
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["simulate_queued_inspection_line", *arguments])

    with pytest.raises(SystemExit) as exc_info:
        _parse_args()

    assert exc_info.value.code == 2
