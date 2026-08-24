"""Verify live inspection events against POST and durable REST detail responses."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from uuid import uuid4

from websockets.sync.client import ClientConnection, connect

from pipelines.simulate_inspection_line import (
    DEFAULT_API_BASE_URL,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    LineEventResult,
    LineScheduleEvent,
    _encode_multipart_body,
    build_prediction_url,
    request_line_prediction,
)
from services.api.schemas import (
    InspectionCreatedEvent,
    InspectionResponse,
)
from services.api.tooling import PreparedImageUpload

MAX_RESPONSE_BYTES = 1024 * 1024


@dataclass(frozen=True)
class InspectionWebSocketSmokeSummary:
    """Validated multi-client and post-disconnect event identities."""

    first_inspection_id: str
    second_inspection_id: str
    model_name: str
    category: str
    device: str
    first_score: float
    second_score: float


# ADD 2026-08-25: Two-client delivery와 disconnect 이후 remaining delivery를 실제 API에서 검증한다.
def smoke_inspection_websocket(
    *,
    api_base_url: str,
    image_path: Path,
    timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
) -> InspectionWebSocketSmokeSummary:
    """Run two committed POSTs while validating live and durable representations."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Smoke image not found: {image_path}")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive.")
    prediction_url = build_prediction_url(api_base_url)
    websocket_url = _inspection_websocket_url(api_base_url)
    event = LineScheduleEvent(
        sequence=1,
        total=2,
        source_kind="normal",
        image_path=image_path,
        relative_image_path=image_path.name,
    )

    # 두 client를 같은 single-process broadcaster에 연결한 뒤 첫 committed POST를 생성한다.
    first_client = connect(
        websocket_url,
        open_timeout=timeout_seconds,
        close_timeout=timeout_seconds,
        proxy=None,
    )
    second_client = connect(
        websocket_url,
        open_timeout=timeout_seconds,
        close_timeout=timeout_seconds,
        proxy=None,
    )
    try:
        first_prediction = request_line_prediction(
            event=event,
            prediction_url=prediction_url,
            request_timeout_seconds=timeout_seconds,
        )
        first_event = _receive_event(first_client, timeout_seconds=timeout_seconds)
        second_event = _receive_event(second_client, timeout_seconds=timeout_seconds)
        _validate_event_matches_prediction(first_event, first_prediction)
        if first_event != second_event:
            raise ValueError("WebSocket clients received different inspection events.")
        first_detail = _get_inspection_detail(
            api_base_url,
            inspection_id=str(first_prediction.inspection_id),
            timeout_seconds=timeout_seconds,
        )
        _validate_detail_matches_event(first_detail, first_event)

        # Client A를 닫은 뒤 두 번째 POST가 remaining client B에 계속 전달되는지 확인한다.
        first_client.close()
        second_schedule_event = LineScheduleEvent(
            sequence=2,
            total=2,
            source_kind="normal",
            image_path=image_path,
            relative_image_path=image_path.name,
        )
        second_prediction = request_line_prediction(
            event=second_schedule_event,
            prediction_url=prediction_url,
            request_timeout_seconds=timeout_seconds,
        )
        after_disconnect = _receive_event(second_client, timeout_seconds=timeout_seconds)
        _validate_event_matches_prediction(
            after_disconnect,
            second_prediction,
        )
        second_detail = _get_inspection_detail(
            api_base_url,
            inspection_id=str(second_prediction.inspection_id),
            timeout_seconds=timeout_seconds,
        )
        _validate_detail_matches_event(second_detail, after_disconnect)

        # Invalid image는 400으로 거부되고 remaining client에 event를 만들지 않아야 한다.
        _verify_invalid_image_no_event(
            second_client,
            prediction_url=prediction_url,
            timeout_seconds=timeout_seconds,
        )
    finally:
        first_client.close()
        second_client.close()

    return InspectionWebSocketSmokeSummary(
        first_inspection_id=str(first_prediction.inspection_id),
        second_inspection_id=str(second_prediction.inspection_id),
        model_name=after_disconnect.inspection.model_name,
        category=after_disconnect.inspection.category,
        device=after_disconnect.inspection.device,
        first_score=first_event.inspection.anomaly_score,
        second_score=after_disconnect.inspection.anomaly_score,
    )


# ADD 2026-08-25: HTTP(S) API base URL을 fixed inspection WebSocket endpoint로 변환한다.
def _inspection_websocket_url(api_base_url: str) -> str:
    normalized = api_base_url.strip().rstrip("/")
    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("api_base_url must be an absolute HTTP(S) URL.")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("api_base_url must not embed credentials.")
    if parsed.query or parsed.fragment:
        raise ValueError("api_base_url must not include a query or fragment.")
    websocket_scheme = "wss" if parsed.scheme == "https" else "ws"
    return f"{websocket_scheme}://{parsed.netloc}{parsed.path.rstrip('/')}/v1/ws/inspections"


# ADD 2026-08-25: One server message를 versioned inspection.created schema로 검증한다.
def _receive_event(
    connection: ClientConnection,
    *,
    timeout_seconds: float,
) -> InspectionCreatedEvent:
    message = connection.recv(timeout=timeout_seconds)
    if not isinstance(message, str):
        raise ValueError("Inspection WebSocket returned a non-text message.")
    try:
        payload = json.loads(message)
    except json.JSONDecodeError as exc:
        raise ValueError("Inspection WebSocket returned malformed JSON.") from exc
    return InspectionCreatedEvent.model_validate(payload)


# ADD 2026-08-25: Live event identity와 score를 committed POST response에 맞춰 검증한다.
def _validate_event_matches_prediction(
    event: InspectionCreatedEvent,
    prediction: LineEventResult,
) -> None:
    inspection = event.inspection
    if (
        inspection.inspection_id != prediction.inspection_id
        or inspection.model_name != prediction.model_name
        or inspection.category != prediction.category
        or inspection.is_anomaly is not prediction.is_anomaly
        or inspection.anomaly_score != prediction.anomaly_score
        or inspection.threshold != prediction.threshold
        or inspection.comparison_operator != ">"
    ):
        raise ValueError("POST response does not match the WebSocket event.")


# ADD 2026-08-25: Durable REST detail과 compact live event의 shared fields를 비교한다.
def _validate_detail_matches_event(
    detail: InspectionResponse,
    event: InspectionCreatedEvent,
) -> None:
    inspection = event.inspection
    if (
        detail.inspection_id != inspection.inspection_id
        or detail.created_at != inspection.created_at
        or detail.model_name != inspection.model_name
        or detail.category != inspection.category
        or detail.is_anomaly is not inspection.is_anomaly
        or detail.anomaly_score != inspection.anomaly_score
        or detail.threshold != inspection.threshold
        or detail.comparison_operator != inspection.comparison_operator
        or detail.device != inspection.device
    ):
        raise ValueError("REST inspection detail does not match the WebSocket event.")


# ADD 2026-08-25: Persisted inspection detail을 bounded standard-library HTTP read로 조회한다.
def _get_inspection_detail(
    api_base_url: str,
    *,
    inspection_id: str,
    timeout_seconds: float,
) -> InspectionResponse:
    url = f"{api_base_url.strip().rstrip('/')}/v1/inspections/{inspection_id}"
    try:
        with urlopen(url, timeout=timeout_seconds) as response:
            status_code = response.status
            content = response.read(MAX_RESPONSE_BYTES + 1)
    except HTTPError as exc:
        raise RuntimeError(f"Inspection detail returned HTTP {exc.code}.") from exc
    except (TimeoutError, URLError, OSError) as exc:
        raise RuntimeError("Inspection detail request failed or timed out.") from exc
    if status_code != 200:
        raise RuntimeError(f"Inspection detail returned HTTP {status_code}.")
    if len(content) > MAX_RESPONSE_BYTES:
        raise RuntimeError("Inspection detail response exceeded the size limit.")
    try:
        return InspectionResponse.model_validate_json(content)
    except ValueError as exc:
        raise ValueError("Inspection detail response did not match its schema.") from exc


# ADD 2026-08-25: Malformed image의 400 response와 bounded no-event window를 실제 API에서 확인한다.
def _verify_invalid_image_no_event(
    connection: ClientConnection,
    *,
    prediction_url: str,
    timeout_seconds: float,
) -> None:
    boundary = f"----smartfactory-ws-smoke-{uuid4().hex}"
    body = _encode_multipart_body(
        PreparedImageUpload("malformed.png", "image/png", b"not-an-image"),
        boundary=boundary,
    )
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
    except HTTPError as exc:
        status_code = exc.code
    except (TimeoutError, URLError, OSError) as exc:
        raise RuntimeError("Malformed image request failed or timed out.") from exc
    if status_code != 400:
        raise RuntimeError(f"Malformed image returned HTTP {status_code}, expected 400.")
    try:
        unexpected = connection.recv(timeout=min(timeout_seconds, 0.5))
    except TimeoutError:
        return
    raise ValueError(f"Malformed image unexpectedly produced a WebSocket event: {unexpected!r}")


# ADD 2026-08-25: Real WebSocket smoke CLI arguments를 정의한다.
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify committed inspection events with two WebSocket clients."
    )
    parser.add_argument("--api-base-url", default=DEFAULT_API_BASE_URL)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
    )
    return parser.parse_args()


# ADD 2026-08-25: Multi-client/disconnect smoke 결과를 concise terminal output으로 출력한다.
def main() -> int:
    args = _parse_args()
    summary = smoke_inspection_websocket(
        api_base_url=args.api_base_url,
        image_path=args.image,
        timeout_seconds=args.timeout_seconds,
    )
    print("Inspection WebSocket smoke: PASS")
    print(f"Model/category/device: {summary.model_name}/{summary.category}/{summary.device}")
    print(f"First inspection ID/score: {summary.first_inspection_id}/{summary.first_score}")
    print(f"Second inspection ID/score: {summary.second_inspection_id}/{summary.second_score}")
    print("Multi-client delivery: PASS")
    print("Post-disconnect remaining-client delivery: PASS")
    print("REST detail consistency: PASS")
    print("Malformed image no-event: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
