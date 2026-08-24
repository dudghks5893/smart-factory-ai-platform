"""Unit contracts for the deterministic production-line simulator."""

from __future__ import annotations

import sys
from pathlib import Path
from uuid import UUID

import pytest
from PIL import Image

from ml.datasets.manifest import ManifestRecord, write_manifest_csv
from pipelines.simulate_inspection_line import (
    LineSimulationError,
    PredictionRequestError,
    _encode_multipart_body,
    _parse_args,
    build_production_demo_schedule,
    format_line_summary,
    simulate_inspection_line,
)
from services.api.tooling import PreparedImageUpload


# ADD 2026-08-24: Simulator schedule test용 실제 PNG와 manifest를 구성한다.
def _dataset_fixture(tmp_path: Path) -> tuple[Path, Path]:
    dataset_root = tmp_path / "mvtec_ad"
    records: list[ManifestRecord] = []
    for source_kind, defect_type, label, count in (
        ("normal", "good", 0, 2),
        ("anomaly", "bent", 1, 2),
    ):
        for index in range(count):
            image_relative = f"metal_nut/test/{defect_type}/{index:03d}.png"
            image_path = dataset_root / image_relative
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), color=(index * 20, 80, 120)).save(image_path)
            mask_relative = ""
            if label == 1:
                mask_relative = f"metal_nut/ground_truth/{defect_type}/{index:03d}_mask.png"
                mask_path = dataset_root / mask_relative
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("L", (8, 8), color=255).save(mask_path)
            records.append(
                ManifestRecord(
                    sample_id=f"metal_nut_test_{source_kind}_{index}",
                    category="metal_nut",
                    source_split="test",
                    split="test",
                    defect_type=defect_type,
                    label=label,
                    image_path=image_relative,
                    mask_path=mask_relative,
                    width=8,
                    height=8,
                )
            )
    manifest_path = tmp_path / "manifest.csv"
    write_manifest_csv(records, manifest_path)
    return dataset_root, manifest_path


# ADD 2026-08-24: 동일 input이 count/source ratio/order/path까지 같은 schedule을 만드는지 검증한다.
def test_production_demo_schedule_is_deterministic_and_evenly_interleaved(tmp_path: Path) -> None:
    dataset_root, manifest_path = _dataset_fixture(tmp_path)

    first = build_production_demo_schedule(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        category="metal_nut",
        count=20,
        anomaly_source_ratio=0.1,
    )
    second = build_production_demo_schedule(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        category="metal_nut",
        count=20,
        anomaly_source_ratio=0.1,
    )

    assert first == second
    assert len(first) == 20
    assert [event.sequence for event in first if event.source_kind == "anomaly"] == [10, 20]
    assert sum(event.source_kind == "normal" for event in first) == 18
    assert all(event.image_path.is_file() for event in first)


# ADD 2026-08-24: 부족한 good image가 manifest order로 deterministic하게 재사용되는지 검증한다.
def test_schedule_cycles_real_normal_images_without_synthetic_generation(tmp_path: Path) -> None:
    dataset_root, manifest_path = _dataset_fixture(tmp_path)

    events = build_production_demo_schedule(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        category="metal_nut",
        count=5,
        anomaly_source_ratio=0,
    )

    assert [event.image_path.name for event in events] == [
        "000.png",
        "001.png",
        "000.png",
        "001.png",
        "000.png",
    ]


# ADD 2026-08-24: Manifest traversal path가 dataset 외부 image를 선택하지 못하는지 검증한다.
def test_schedule_rejects_manifest_path_outside_dataset(tmp_path: Path) -> None:
    dataset_root, manifest_path = _dataset_fixture(tmp_path)
    records = [
        ManifestRecord(
            sample_id="escape",
            category="metal_nut",
            source_split="test",
            split="test",
            defect_type="good",
            label=0,
            image_path="../outside.png",
            mask_path="",
            width=8,
            height=8,
        )
    ]
    write_manifest_csv(records, manifest_path)

    with pytest.raises(ValueError, match="stay under dataset root"):
        build_production_demo_schedule(
            dataset_root=dataset_root,
            manifest_path=manifest_path,
            category="metal_nut",
            count=1,
            anomaly_source_ratio=0,
        )


# ADD 2026-08-24: Multipart가 image 외 label/score/threshold를 포함하지 않는지 검증한다.
def test_multipart_payload_contains_only_image_field() -> None:
    upload = PreparedImageUpload("000.png", "image/png", b"real-image-bytes")

    body = _encode_multipart_body(upload, boundary="fixed-boundary")
    header = body.split(b"\r\n\r\n", maxsplit=1)[0]

    assert b'name="image"' in header
    assert b"ground_truth" not in header
    assert b"defect_type" not in header
    assert b"label" not in header
    assert b"score" not in header
    assert b"threshold" not in header


# ADD 2026-08-24: Fake transport로 순차 request, response, interval과 summary를 검증한다.
def test_simulation_is_sequential_and_aggregates_observed_predictions(tmp_path: Path) -> None:
    dataset_root, manifest_path = _dataset_fixture(tmp_path)
    events = build_production_demo_schedule(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        category="metal_nut",
        count=3,
        anomaly_source_ratio=1 / 3,
    )
    calls: list[str] = []
    sleeps: list[float] = []
    outputs: list[str] = []
    request_count = 0

    def transport(url: str, upload: PreparedImageUpload, timeout: float) -> object:
        nonlocal request_count
        request_count += 1
        calls.append(f"request:{upload.filename}:{url}:{timeout}")
        score = 50.0 if request_count == 3 else 30.0
        return _prediction_payload(index=request_count, score=score)

    def sleeper(seconds: float) -> None:
        calls.append(f"sleep:{seconds}")
        sleeps.append(seconds)

    results, summary = simulate_inspection_line(
        events=events,
        api_base_url="http://api.local:8000",
        interval_seconds=0.2,
        request_timeout_seconds=7.0,
        transport=transport,
        sleeper=sleeper,
        event_writer=outputs.append,
    )

    assert [call.split(":", maxsplit=1)[0] for call in calls] == [
        "request",
        "sleep",
        "request",
        "sleep",
        "request",
    ]
    assert sleeps == [0.2, 0.2]
    assert len(results) == 3
    assert (summary.normal_source_events, summary.anomaly_source_events) == (2, 1)
    assert (summary.normal_predictions, summary.anomaly_predictions) == (2, 1)
    assert summary.successful_inspections == 3
    assert summary.unique_inspection_ids == 3
    assert "input_source=ANOMALY result=ANOMALY" in outputs[-1]


# ADD 2026-08-24: 첫 HTTP failure가 retry/sleep 없이 partial summary와 함께 중단되는지 검증한다.
def test_simulation_fails_fast_on_http_failure(tmp_path: Path) -> None:
    dataset_root, manifest_path = _dataset_fixture(tmp_path)
    events = build_production_demo_schedule(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        category="metal_nut",
        count=2,
        anomaly_source_ratio=0,
    )
    calls = 0
    sleeps: list[float] = []

    def failing_transport(_url: str, _upload: PreparedImageUpload, _timeout: float) -> object:
        nonlocal calls
        calls += 1
        raise PredictionRequestError("Prediction API returned HTTP 503.")

    with pytest.raises(LineSimulationError, match="event 1") as exc_info:
        simulate_inspection_line(
            events=events,
            api_base_url="http://api.local:8000",
            transport=failing_transport,
            sleeper=sleeps.append,
            event_writer=lambda _line: None,
        )

    assert calls == 1
    assert sleeps == []
    assert exc_info.value.summary.successful_inspections == 0
    assert exc_info.value.summary.failed_inspections == 1


# ADD 2026-08-24: Schema가 아닌 HTTP JSON을 valid prediction으로 집계하지 않는지 검증한다.
def test_simulation_rejects_malformed_response(tmp_path: Path) -> None:
    dataset_root, manifest_path = _dataset_fixture(tmp_path)
    events = build_production_demo_schedule(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        category="metal_nut",
        count=1,
        anomaly_source_ratio=0,
    )

    with pytest.raises(LineSimulationError) as exc_info:
        simulate_inspection_line(
            events=events,
            api_base_url="http://api.local:8000",
            transport=lambda _url, _upload, _timeout: {"unexpected": True},
            sleeper=lambda _seconds: None,
            event_writer=lambda _line: None,
        )

    assert exc_info.value.summary.successful_inspections == 0
    assert exc_info.value.summary.failed_inspections == 1


# ADD 2026-08-24: Summary가 input source와 observed prediction ratio를 별도로 표현하는지 검증한다.
def test_summary_output_separates_source_and_prediction_counts(tmp_path: Path) -> None:
    dataset_root, manifest_path = _dataset_fixture(tmp_path)
    events = build_production_demo_schedule(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        category="metal_nut",
        count=2,
        anomaly_source_ratio=0.5,
    )
    response_index = 0

    def transport(_url: str, _upload: PreparedImageUpload, _timeout: float) -> object:
        nonlocal response_index
        response_index += 1
        return _prediction_payload(index=response_index, score=30.0)

    _, summary = simulate_inspection_line(
        events=events,
        api_base_url="http://api.local:8000",
        interval_seconds=0,
        transport=transport,
        sleeper=lambda _seconds: None,
        event_writer=lambda _line: None,
    )
    output = format_line_summary(summary)

    assert "Input sources (normal/anomaly): 1/1" in output
    assert "Observed predictions (NORMAL/ANOMALY): 2/0" in output
    assert "Observed anomaly ratio: 0.0%" in output


# ADD 2026-08-24: Invalid CLI numeric argument가 실제 run 전에 argparse에서 거부되는지 검증한다.
@pytest.mark.parametrize(
    "arguments",
    [
        ["--count", "0"],
        ["--anomaly-source-ratio", "1.1"],
        ["--interval-seconds", "-0.1"],
        ["--request-timeout-seconds", "0"],
    ],
)
def test_cli_rejects_invalid_numeric_arguments(
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["simulate_inspection_line", *arguments])

    with pytest.raises(SystemExit) as exc_info:
        _parse_args()

    assert exc_info.value.code == 2


# ADD 2026-08-24: Test transport용 strict-threshold production response payload를 생성한다.
def _prediction_payload(*, index: int, score: float) -> dict[str, object]:
    threshold = 40.0
    return {
        "inspection_id": str(UUID(int=index)),
        "model_name": "patchcore",
        "category": "metal_nut",
        "is_anomaly": score > threshold,
        "anomaly_score": score,
        "threshold": threshold,
        "comparison_operator": ">",
    }
