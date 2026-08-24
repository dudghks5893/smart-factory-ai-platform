"""Pure browser-state contracts executed with the dependency-free Node runtime."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import cast

import pytest


# ADD 2026-08-25: Browser state module path를 repository root에서 결정한다.
def _state_module_uri() -> str:
    return (Path(__file__).resolve().parents[2] / "apps" / "live_monitor" / "state.js").as_uri()


# ADD 2026-08-25: npm/package 없이 ES module을 실행해 JSON-safe test observation을 반환한다.
def _run_state_contract() -> dict[str, object]:
    if shutil.which("node") is None:
        pytest.skip("Node is unavailable; browser-state contract requires an ES module runtime.")
    module_uri = json.dumps(_state_module_uri())
    script = f"""
      const state = await import({module_uri});
      const item = (index, createdAt = null) => ({{
        inspection_id: `00000000-0000-4000-8000-${{String(index).padStart(12, "0")}}`,
        model_name: "patchcore",
        category: "metal_nut",
        is_anomaly: index % 10 === 0,
        anomaly_score: 20 + index,
        threshold: 41.2,
        comparison_operator: ">",
        device: "mps",
        created_at: createdAt ?? new Date(Date.UTC(2026, 7, 25, 0, 0, index)).toISOString(),
      }});
      const history = Array.from({{length: 100}}, (_, index) => item(index));
      const buffered = [item(50), item(100), item(101)];
      const merged = state.mergeInspections(history, buffered);
      const tieTime = "2026-08-25T12:00:00.000Z";
      const tied = state.mergeInspections([item(1, tieTime), item(2, tieTime)]);
      const validEvent = {{schema_version: "1", type: "inspection.created", inspection: item(1)}};
      const malformed = {{...validEvent, schema_version: "2"}};
      console.log(JSON.stringify({{
        length: merged.length,
        ids: merged.map((value) => value.inspection_id),
        kpis: state.calculateKpis(merged),
        tiedIds: tied.map((value) => value.inspection_id),
        validEvent: state.parseInspectionEvent(validEvent)?.inspection_id,
        malformedEvent: state.parseInspectionEvent(malformed),
        malformedInspection: state.normalizeInspection({{...item(1), anomaly_score: NaN}}),
        historyLength: state.parseInspectionHistory({{items: history}}).length,
        httpUrl: state.inspectionWebSocketUrl({{protocol: "http:", host: "factory.test:8000"}}),
        httpsUrl: state.inspectionWebSocketUrl({{protocol: "https:", host: "factory.test"}}),
        delays: [0, 1, 2, 3, 4, 10].map(state.reconnectDelayMs),
      }}));
    """
    result = subprocess.run(  # noqa: S603
        ["node", "--input-type=module", "--eval", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return cast(dict[str, object], json.loads(result.stdout))


# ADD 2026-08-25: REST/buffer merge, dedupe, newest-first 100 window과 visible KPI를 검증한다.
def test_live_monitor_merge_window_order_and_kpis() -> None:
    result = _run_state_contract()

    assert result["length"] == 100
    ids = result["ids"]
    assert isinstance(ids, list)
    assert ids[0].endswith("000000000101")
    assert ids[1].endswith("000000000100")
    assert len(ids) == len(set(ids))
    assert result["kpis"] == {
        "visible": 100,
        "normal": 90,
        "anomaly": 10,
        "anomalyRatio": 0.1,
    }
    assert result["tiedIds"] == [
        "00000000-0000-4000-8000-000000000001",
        "00000000-0000-4000-8000-000000000002",
    ]


# ADD 2026-08-25: Event validation, same-origin WS URL과 bounded reconnect delay를 검증한다.
def test_live_monitor_event_and_reconnect_contract() -> None:
    result = _run_state_contract()

    assert result["validEvent"] == "00000000-0000-4000-8000-000000000001"
    assert result["malformedEvent"] is None
    assert result["malformedInspection"] is None
    assert result["historyLength"] == 100
    assert result["httpUrl"] == "ws://factory.test:8000/v1/ws/inspections"
    assert result["httpsUrl"] == "wss://factory.test/v1/ws/inspections"
    assert result["delays"] == [500, 1000, 2000, 4000, 5000, 5000]
