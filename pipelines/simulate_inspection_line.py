"""Send deterministic real MVTec images through the production prediction API."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from uuid import UUID, uuid4

from ml.datasets.constants import GOOD_DIR_NAME
from ml.datasets.manifest import ManifestRecord, read_manifest_csv
from ml.datasets.manifest_validation import validate_manifest_records
from services.api.config import DEFAULT_MAX_UPLOAD_BYTES
from services.api.schemas import InferenceResponse
from services.api.tooling import (
    PreparedImageUpload,
    prepare_image_upload,
    validate_prediction_payload,
)

DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_DATASET_ROOT = Path("data/raw/mvtec_ad")
DEFAULT_MANIFEST_PATH = Path("data/interim/manifests/mvtec_ad_metal_nut.csv")
DEFAULT_CATEGORY = "metal_nut"
DEFAULT_EVENT_COUNT = 100
DEFAULT_ANOMALY_SOURCE_RATIO = 0.1
DEFAULT_INTERVAL_SECONDS = 1.0
DEFAULT_REQUEST_TIMEOUT_SECONDS = 30.0
MAX_RESPONSE_BYTES = 1024 * 1024
PRODUCTION_DEMO_PROFILE = "production-demo"

type SourceKind = Literal["normal", "anomaly"]
type PredictionTransport = Callable[[str, PreparedImageUpload, float], object]
type Sleeper = Callable[[float], None]
type Clock = Callable[[], float]
type EventWriter = Callable[[str], None]


class PredictionRequestError(RuntimeError):
    """One external prediction request failed before a valid response was available."""


class LineSimulationError(RuntimeError):
    """Fail-fast simulation error carrying the completed partial-run summary."""

    # ADD 2026-08-24: 실패 지점까지의 summary를 CLI error reporting에 보존한다.
    def __init__(self, message: str, summary: LineSimulationSummary) -> None:
        super().__init__(message)
        self.summary = summary


@dataclass(frozen=True)
class LineScheduleEvent:
    """One real image selected for a deterministic sequential trigger."""

    sequence: int
    total: int
    source_kind: SourceKind
    image_path: Path
    relative_image_path: str


@dataclass(frozen=True)
class LineEventResult:
    """Validated production API response and request timing for one event."""

    event: LineScheduleEvent
    inspection_id: UUID
    model_name: str
    category: str
    is_anomaly: bool
    anomaly_score: float
    threshold: float
    elapsed_ms: float


@dataclass(frozen=True)
class LineSimulationSummary:
    """Input-source and observed-prediction totals for one line simulation."""

    requested_events: int
    successful_inspections: int
    failed_inspections: int
    normal_source_events: int
    anomaly_source_events: int
    normal_predictions: int
    anomaly_predictions: int
    unique_inspection_ids: int
    elapsed_seconds: float
    average_request_ms: float

    @property
    def observed_anomaly_ratio(self) -> float:
        if self.successful_inspections == 0:
            return 0.0
        return self.anomaly_predictions / self.successful_inspections


# ADD 2026-08-24: Official test image에서 현실적인 source ratio의 deterministic schedule을 구성한다.
def build_production_demo_schedule(
    *,
    dataset_root: Path,
    manifest_path: Path,
    category: str,
    count: int = DEFAULT_EVENT_COUNT,
    anomaly_source_ratio: float = DEFAULT_ANOMALY_SOURCE_RATIO,
) -> tuple[LineScheduleEvent, ...]:
    """Build an evenly interleaved schedule without generating or modifying images."""
    _validate_schedule_parameters(count=count, anomaly_source_ratio=anomaly_source_ratio)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"MVTec dataset root not found: {dataset_root}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"MVTec manifest not found: {manifest_path}")
    if not category.strip():
        raise ValueError("category must not be empty.")

    # Manifest path boundary를 확인한 뒤 existing integrity validator로 실제 image/mask를 검증한다.
    records = read_manifest_csv(manifest_path)
    _validate_local_manifest_paths(records, dataset_root)
    report = validate_manifest_records(records, dataset_root)
    if not report.is_valid:
        raise ValueError("Line simulator manifest validation failed:\n" + "\n".join(report.errors))

    # Official test split만 source pool로 사용하며 request에는 이 metadata를 전달하지 않는다.
    test_records = [
        record
        for record in records
        if record.category == category and record.source_split == "test" and record.split == "test"
    ]
    normal_records = [
        record
        for record in test_records
        if record.label == 0 and record.defect_type == GOOD_DIR_NAME
    ]
    anomaly_records = [record for record in test_records if record.label == 1]
    anomaly_count = _rounded_source_count(count, anomaly_source_ratio)
    normal_count = count - anomaly_count
    if normal_count and not normal_records:
        raise ValueError(f"No official test normal images found for category: {category}")
    if anomaly_count and not anomaly_records:
        raise ValueError(f"No official test anomaly images found for category: {category}")

    # Cumulative target count로 anomaly source를 균등 배치하고 각 pool은 manifest 순서로 순환한다.
    events: list[LineScheduleEvent] = []
    emitted_anomalies = 0
    normal_index = 0
    anomaly_index = 0
    for sequence in range(1, count + 1):
        target_anomalies = sequence * anomaly_count // count
        if target_anomalies > emitted_anomalies:
            source_kind: SourceKind = "anomaly"
            record = anomaly_records[anomaly_index % len(anomaly_records)]
            anomaly_index += 1
            emitted_anomalies += 1
        else:
            source_kind = "normal"
            record = normal_records[normal_index % len(normal_records)]
            normal_index += 1
        events.append(
            LineScheduleEvent(
                sequence=sequence,
                total=count,
                source_kind=source_kind,
                image_path=_resolve_local_path(dataset_root, record.image_path),
                relative_image_path=record.image_path,
            )
        )
    return tuple(events)


# ADD 2026-08-24: 실제 image를 production API에 순차 전송하고 fail-fast summary를 집계한다.
def simulate_inspection_line(
    *,
    events: Sequence[LineScheduleEvent],
    api_base_url: str,
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
    request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    transport: PredictionTransport | None = None,
    sleeper: Sleeper = time.sleep,
    clock: Clock = time.perf_counter,
    event_writer: EventWriter = print,
) -> tuple[tuple[LineEventResult, ...], LineSimulationSummary]:
    """Run response-complete then interval sequential requests without retrying failures."""
    if not events:
        raise ValueError("Line simulation requires at least one scheduled event.")
    _validate_runtime_parameters(
        interval_seconds=interval_seconds,
        request_timeout_seconds=request_timeout_seconds,
    )
    prediction_url = _prediction_url(api_base_url)
    active_transport = _post_prediction if transport is None else transport
    started_at = clock()
    results: list[LineEventResult] = []
    inspection_ids: set[UUID] = set()

    for index, event in enumerate(events):
        # Disk image를 검증·로드한 뒤 HTTP request timing boundary를 시작한다.
        upload = prepare_image_upload(event.image_path, max_upload_bytes=DEFAULT_MAX_UPLOAD_BYTES)
        request_started_at = clock()
        try:
            payload = active_transport(prediction_url, upload, request_timeout_seconds)
            prediction = validate_prediction_payload(payload)
            if prediction.inspection_id in inspection_ids:
                raise ValueError("Prediction API returned a duplicate inspection_id.")
        except Exception as exc:
            request_elapsed_ms = max(0.0, (clock() - request_started_at) * 1000)
            summary = _build_summary(
                events=events,
                results=results,
                failed_inspections=1,
                elapsed_seconds=max(0.0, clock() - started_at),
            )
            event_writer(_format_failed_event(event, request_elapsed_ms, exc))
            raise LineSimulationError(
                f"Line simulation stopped at event {event.sequence}: {exc}",
                summary,
            ) from exc

        request_elapsed_ms = max(0.0, (clock() - request_started_at) * 1000)
        inspection_ids.add(prediction.inspection_id)
        result = _event_result(event, prediction, request_elapsed_ms)
        results.append(result)
        event_writer(format_line_event(result))

        # 마지막 event를 제외하고 completed response 뒤에만 다음 trigger interval을 둔다.
        if index < len(events) - 1:
            sleeper(interval_seconds)

    summary = _build_summary(
        events=events,
        results=results,
        failed_inspections=0,
        elapsed_seconds=max(0.0, clock() - started_at),
    )
    return tuple(results), summary


# ADD 2026-08-24: Event response를 concise one-line runtime output으로 변환한다.
def format_line_event(result: LineEventResult) -> str:
    label = "ANOMALY" if result.is_anomaly else "NORMAL"
    return (
        f"[{result.event.sequence:03d}/{result.event.total:03d}] "
        f"image={result.event.relative_image_path} "
        f"input_source={result.event.source_kind.upper()} result={label} "
        f"score={result.anomaly_score:.6f} threshold={result.threshold:.6f} "
        f"inspection_id={result.inspection_id} elapsed_ms={result.elapsed_ms:.3f}"
    )


# ADD 2026-08-24: Input source와 observed prediction을 분리한 final summary를 렌더링한다.
def format_line_summary(summary: LineSimulationSummary) -> str:
    return "\n".join(
        (
            f"Requested events: {summary.requested_events}",
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
            f"Observed anomaly ratio: {summary.observed_anomaly_ratio:.1%}",
            f"Unique inspection IDs: {summary.unique_inspection_ids}",
            f"Elapsed time: {summary.elapsed_seconds:.3f} seconds",
            f"Average request time: {summary.average_request_ms:.3f} ms",
        )
    )


# ADD 2026-08-24: CLI argument contract와 production-demo defaults를 정의한다.
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send deterministic real MVTec images to the production prediction API."
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
        "--interval-seconds",
        type=_nonnegative_float,
        default=DEFAULT_INTERVAL_SECONDS,
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=_positive_float,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
    )
    return parser.parse_args()


# ADD 2026-08-24: Schedule 생성부터 sequential HTTP run과 final exit status를 조율한다.
def main() -> int:
    args = _parse_args()

    # Manifest를 검증하고 configured source ratio의 deterministic event schedule을 만든다.
    events = build_production_demo_schedule(
        dataset_root=args.dataset_root,
        manifest_path=args.manifest,
        category=args.category,
        count=args.count,
        anomaly_source_ratio=args.anomaly_source_ratio,
    )
    print(f"Production line profile: {args.profile}")
    try:
        _, summary = simulate_inspection_line(
            events=events,
            api_base_url=args.api_base_url,
            interval_seconds=args.interval_seconds,
            request_timeout_seconds=args.request_timeout_seconds,
        )
    except LineSimulationError as exc:
        print("Production line simulation: FAILED")
        print(format_line_summary(exc.summary))
        print(f"Error: {exc}")
        return 1

    print("Production line simulation: PASS")
    print(format_line_summary(summary))
    return 0


# ADD 2026-08-24: Validated API base URL을 existing prediction endpoint로 고정한다.
def _prediction_url(api_base_url: str) -> str:
    normalized = api_base_url.strip().rstrip("/")
    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("api_base_url must be an absolute HTTP(S) URL.")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("api_base_url must not embed credentials.")
    if parsed.query or parsed.fragment:
        raise ValueError("api_base_url must not include a query or fragment.")
    return f"{normalized}/v1/predictions"


# ADD 2026-08-24: Multipart field 하나에 image bytes만 담아 production endpoint로 전송한다.
def _post_prediction(
    prediction_url: str,
    upload: PreparedImageUpload,
    timeout_seconds: float,
) -> object:
    boundary = f"----smartfactory-{uuid4().hex}"
    body = _encode_multipart_body(upload, boundary=boundary)
    request = Request(
        prediction_url,
        data=body,
        method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        },
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            status_code = response.status
            response_body = response.read(MAX_RESPONSE_BYTES + 1)
    except HTTPError as exc:
        raise PredictionRequestError(f"Prediction API returned HTTP {exc.code}.") from exc
    except (TimeoutError, URLError, OSError) as exc:
        raise PredictionRequestError("Prediction API request failed or timed out.") from exc
    if status_code != 200:
        raise PredictionRequestError(f"Prediction API returned HTTP {status_code}.")
    if len(response_body) > MAX_RESPONSE_BYTES:
        raise PredictionRequestError("Prediction API response exceeded the size limit.")
    try:
        return json.loads(response_body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PredictionRequestError("Prediction API returned malformed JSON.") from exc


# ADD 2026-08-24: Header injection 없이 단일 image field의 multipart body를 구성한다.
def _encode_multipart_body(upload: PreparedImageUpload, *, boundary: str) -> bytes:
    if not boundary or any(character in boundary for character in "\r\n"):
        raise ValueError("multipart boundary must be a non-empty single-line value.")
    if any(character in upload.filename for character in '\r\n"'):
        raise ValueError("upload filename contains unsupported header characters.")
    header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{upload.filename}"\r\n'
        f"Content-Type: {upload.content_type}\r\n\r\n"
    ).encode("ascii")
    return header + upload.content + f"\r\n--{boundary}--\r\n".encode("ascii")


# ADD 2026-08-24: Manifest relative path가 dataset root를 벗어나지 않도록 검증한다.
def _resolve_local_path(dataset_root: Path, relative_path: str) -> Path:
    candidate_relative = Path(relative_path)
    if (
        candidate_relative.is_absolute()
        or not candidate_relative.parts
        or ".." in candidate_relative.parts
    ):
        raise ValueError(f"Manifest path must stay under dataset root: {relative_path}")
    root = dataset_root.resolve()
    candidate = (root / candidate_relative).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"Manifest path resolves outside dataset root: {relative_path}")
    return candidate


# ADD 2026-08-24: Full manifest validation 전에 image/mask path boundary를 먼저 확인한다.
def _validate_local_manifest_paths(records: Sequence[ManifestRecord], dataset_root: Path) -> None:
    for record in records:
        _resolve_local_path(dataset_root, record.image_path)
        if record.mask_path:
            _resolve_local_path(dataset_root, record.mask_path)


# ADD 2026-08-24: Requested ratio를 deterministic integer anomaly source count로 변환한다.
def _rounded_source_count(count: int, anomaly_source_ratio: float) -> int:
    return min(count, int(math.floor(count * anomaly_source_ratio + 0.5)))


# ADD 2026-08-24: Schedule count와 source ratio bounds를 검증한다.
def _validate_schedule_parameters(*, count: int, anomaly_source_ratio: float) -> None:
    if count <= 0:
        raise ValueError("count must be positive.")
    if not math.isfinite(anomaly_source_ratio) or not 0 <= anomaly_source_ratio <= 1:
        raise ValueError("anomaly_source_ratio must be in [0, 1].")


# ADD 2026-08-24: Trigger interval과 external request timeout bounds를 검증한다.
def _validate_runtime_parameters(
    *,
    interval_seconds: float,
    request_timeout_seconds: float,
) -> None:
    if not math.isfinite(interval_seconds) or interval_seconds < 0:
        raise ValueError("interval_seconds must be finite and non-negative.")
    if not math.isfinite(request_timeout_seconds) or request_timeout_seconds <= 0:
        raise ValueError("request_timeout_seconds must be finite and positive.")


# ADD 2026-08-24: Validated API schema를 immutable line event result로 변환한다.
def _event_result(
    event: LineScheduleEvent,
    prediction: InferenceResponse,
    elapsed_ms: float,
) -> LineEventResult:
    return LineEventResult(
        event=event,
        inspection_id=prediction.inspection_id,
        model_name=prediction.model_name,
        category=prediction.category,
        is_anomaly=prediction.is_anomaly,
        anomaly_score=prediction.anomaly_score,
        threshold=prediction.threshold,
        elapsed_ms=elapsed_ms,
    )


# ADD 2026-08-24: Completed results에서 input/observed/latency summary를 계산한다.
def _build_summary(
    *,
    events: Sequence[LineScheduleEvent],
    results: Sequence[LineEventResult],
    failed_inspections: int,
    elapsed_seconds: float,
) -> LineSimulationSummary:
    request_times = [result.elapsed_ms for result in results]
    return LineSimulationSummary(
        requested_events=len(events),
        successful_inspections=len(results),
        failed_inspections=failed_inspections,
        normal_source_events=sum(event.source_kind == "normal" for event in events),
        anomaly_source_events=sum(event.source_kind == "anomaly" for event in events),
        normal_predictions=sum(not result.is_anomaly for result in results),
        anomaly_predictions=sum(result.is_anomaly for result in results),
        unique_inspection_ids=len({result.inspection_id for result in results}),
        elapsed_seconds=elapsed_seconds,
        average_request_ms=(sum(request_times) / len(request_times) if request_times else 0.0),
    )


# ADD 2026-08-24: Failed event를 non-sensitive one-line output으로 변환한다.
def _format_failed_event(event: LineScheduleEvent, elapsed_ms: float, exc: Exception) -> str:
    return (
        f"[{event.sequence:03d}/{event.total:03d}] "
        f"image={event.relative_image_path} http=FAILED "
        f"elapsed_ms={elapsed_ms:.3f} error={type(exc).__name__}"
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _ratio(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("value must be in [0, 1]")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("value must be finite and non-negative")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
