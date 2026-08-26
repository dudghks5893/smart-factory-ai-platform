"""Process-local WebSocket connections for best-effort inspection notifications."""

from __future__ import annotations

import asyncio
import logging
import math

from fastapi import WebSocket
from pydantic import BaseModel

from services.api.schemas import (
    CombinedInspectionCreatedEvent,
    InspectionCreatedEvent,
    KnownDefectCreatedEvent,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_SEND_TIMEOUT_SECONDS = 1.0


class EventBroadcaster[EventModel: BaseModel]:
    """Track one domain channel's process-local clients and isolate send failures."""

    # ADD 2026-08-25: Per-client send timeout과 instance-local connection set을 초기화한다.
    def __init__(self, *, send_timeout_seconds: float = DEFAULT_SEND_TIMEOUT_SECONDS) -> None:
        if not math.isfinite(send_timeout_seconds) or send_timeout_seconds <= 0:
            raise ValueError("send_timeout_seconds must be finite and positive.")
        self._send_timeout_seconds = send_timeout_seconds
        self._connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    # ADD 2026-08-25: WebSocket handshake 완료 뒤 active connection으로 등록한다.
    async def connect(self, websocket: WebSocket) -> None:
        """Accept and register one server-to-client notification connection."""
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)

    # ADD 2026-08-25: Disconnect를 idempotent하게 active connection set에 반영한다.
    async def disconnect(self, websocket: WebSocket) -> None:
        """Remove one connection whether the peer or server initiated closure."""
        async with self._lock:
            self._connections.discard(websocket)

    # ADD 2026-08-25: Snapshot의 모든 client에 event를 보내고 broken client만 정리한다.
    # MODIFY 2026-08-26: Domain-specific typed broadcaster가 connection policy를 재사용하게 한다.
    async def broadcast(self, event: EventModel) -> None:
        """Best-effort broadcast without allowing one failed client to stop others."""
        payload = event.model_dump(mode="json")
        async with self._lock:
            connections = tuple(self._connections)
        if not connections:
            return

        # Client별 send를 동시에 실행해 slow/broken client가 healthy delivery 순서를 막지 않게 한다.
        outcomes = await asyncio.gather(
            *(self._send(connection, payload) for connection in connections)
        )
        broken = {
            connection
            for connection, delivered in zip(connections, outcomes, strict=True)
            if not delivered
        }
        if broken:
            async with self._lock:
                self._connections.difference_update(broken)

    # ADD 2026-08-25: Lifespan shutdown에서 process-local connection을 명시적으로 닫는다.
    async def close_all(self) -> None:
        """Close and forget every connection owned by this app instance."""
        async with self._lock:
            connections = tuple(self._connections)
            self._connections.clear()
        await asyncio.gather(*(self._close(connection) for connection in connections))

    # ADD 2026-08-25: Test와 lifecycle audit용 active connection count를 lock 아래에서 반환한다.
    async def active_connection_count(self) -> int:
        """Return the number of connections registered in this process."""
        async with self._lock:
            return len(self._connections)

    # ADD 2026-08-25: 한 client send를 timeout으로 제한하고 failure를 격리한다.
    async def _send(self, websocket: WebSocket, payload: dict[str, object]) -> bool:
        try:
            await asyncio.wait_for(
                websocket.send_json(payload),
                timeout=self._send_timeout_seconds,
            )
        except Exception as exc:
            LOGGER.warning(
                "Inspection WebSocket delivery failed; removing connection",
                exc_info=exc,
            )
            return False
        return True

    # ADD 2026-08-25: Shutdown close failure가 다른 connection cleanup을 막지 않게 격리한다.
    async def _close(self, websocket: WebSocket) -> None:
        try:
            await asyncio.wait_for(
                websocket.close(code=1001),
                timeout=self._send_timeout_seconds,
            )
        except Exception as exc:
            LOGGER.debug("Inspection WebSocket was already closed", exc_info=exc)


class InspectionEventBroadcaster(EventBroadcaster[InspectionCreatedEvent]):
    """Purpose-specific channel preserving the existing inspection.created contract."""


class KnownDefectEventBroadcaster(EventBroadcaster[KnownDefectCreatedEvent]):
    """Independent channel for committed known_defect.created notifications."""


class CombinedInspectionEventBroadcaster(EventBroadcaster[CombinedInspectionCreatedEvent]):
    """Manufacturing-level channel for committed combined decisions."""
