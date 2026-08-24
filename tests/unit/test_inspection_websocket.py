"""Unit contracts for inspection WebSocket events and connection management."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import cast
from uuid import UUID

from fastapi import WebSocket

from services.api.schemas import InspectionCreatedEvent, InspectionCreatedPayload
from services.api.websockets import InspectionEventBroadcaster


class _FakeWebSocket:
    """Small async WebSocket double for registration and delivery isolation."""

    # ADD 2026-08-25: Fake connection state와 optional send failure를 초기화한다.
    def __init__(self, *, fail_send: bool = False, send_delay: float = 0.0) -> None:
        self.fail_send = fail_send
        self.send_delay = send_delay
        self.accepted = False
        self.closed = False
        self.sent: list[object] = []

    # ADD 2026-08-25: Connection manager accept 호출을 기록한다.
    async def accept(self) -> None:
        self.accepted = True

    # ADD 2026-08-25: Payload를 기록하거나 broken/slow client behavior를 재현한다.
    async def send_json(self, payload: object) -> None:
        if self.send_delay:
            await asyncio.sleep(self.send_delay)
        if self.fail_send:
            raise RuntimeError("broken client")
        self.sent.append(payload)

    # ADD 2026-08-25: Lifespan close 호출을 기록한다.
    async def close(self, *, code: int) -> None:
        assert code == 1001
        self.closed = True


# ADD 2026-08-25: Device와 무관한 compact inspection.created test event를 생성한다.
def _event(*, device: str = "cpu") -> InspectionCreatedEvent:
    return InspectionCreatedEvent(
        inspection=InspectionCreatedPayload(
            inspection_id=UUID(int=1),
            model_name="patchcore",
            category="metal_nut",
            is_anomaly=True,
            anomaly_score=50.0,
            threshold=40.0,
            comparison_operator=">",
            device=device,
            created_at=datetime(2026, 8, 25, tzinfo=UTC),
        )
    )


# ADD 2026-08-25: Event schema가 compact fields와 JSON-safe timestamp만 포함하는지 검증한다.
def test_inspection_created_event_schema_is_compact_and_device_independent() -> None:
    payload = _event(device="cuda").model_dump(mode="json")

    assert payload["schema_version"] == "1"
    assert payload["type"] == "inspection.created"
    assert set(payload["inspection"]) == {
        "inspection_id",
        "model_name",
        "category",
        "is_anomaly",
        "anomaly_score",
        "threshold",
        "comparison_operator",
        "device",
        "created_at",
    }
    assert payload["inspection"]["device"] == "cuda"
    assert "model_sha256" not in payload["inspection"]
    assert "image" not in payload["inspection"]
    assert datetime.fromisoformat(payload["inspection"]["created_at"]).tzinfo is not None


# ADD 2026-08-25: Broadcast가 broken client를 격리하고 healthy clients를 유지하는지 검증한다.
def test_broadcast_isolates_broken_client_and_cleans_registration() -> None:
    broadcaster = InspectionEventBroadcaster()
    first = _FakeWebSocket()
    broken = _FakeWebSocket(fail_send=True)
    second = _FakeWebSocket()

    async def scenario() -> None:
        for connection in (first, broken, second):
            await broadcaster.connect(cast(WebSocket, connection))
        assert await broadcaster.active_connection_count() == 3

        await broadcaster.broadcast(_event())

        assert first.sent == second.sent
        assert len(first.sent) == 1
        assert broken.sent == []
        assert await broadcaster.active_connection_count() == 2
        await broadcaster.disconnect(cast(WebSocket, first))
        assert await broadcaster.active_connection_count() == 1
        await broadcaster.close_all()
        assert second.closed is True
        assert await broadcaster.active_connection_count() == 0

    asyncio.run(scenario())


# ADD 2026-08-25: Slow client send timeout이 connection cleanup으로 이어지는지 검증한다.
def test_slow_client_timeout_does_not_block_healthy_delivery() -> None:
    broadcaster = InspectionEventBroadcaster(send_timeout_seconds=0.01)
    slow = _FakeWebSocket(send_delay=0.1)
    healthy = _FakeWebSocket()

    async def scenario() -> None:
        await broadcaster.connect(cast(WebSocket, slow))
        await broadcaster.connect(cast(WebSocket, healthy))
        await broadcaster.broadcast(_event())

        assert len(healthy.sent) == 1
        assert slow.sent == []
        assert await broadcaster.active_connection_count() == 1

    asyncio.run(scenario())
