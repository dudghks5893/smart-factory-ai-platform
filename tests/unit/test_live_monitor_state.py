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
      const knownHistoryItem = (index, instanceCount = 0, createdAt = null) => ({{
        inspection_id: `10000000-0000-4000-8000-${{String(index).padStart(12, "0")}}`,
        created_at: createdAt ?? new Date(Date.UTC(2026, 7, 26, 0, 0, index)).toISOString(),
        model: {{name: "yolo11n-seg.pt", task: "segment", category: "metal_nut", device: "mps"}},
        image: {{width: 700, height: 700}},
        diagnostic_confidence: 0.25,
        inference_ms: 30 + index,
        instance_count: instanceCount,
      }});
      const knownEventItem = (index, instanceCount, classes, createdAt = null) => ({{
        inspection_id: `10000000-0000-4000-8000-${{String(index).padStart(12, "0")}}`,
        model_name: "yolo11n-seg.pt",
        category: "metal_nut",
        device: "mps",
        diagnostic_confidence: 0.25,
        instance_count: instanceCount,
        classes,
        created_at: createdAt ?? new Date(Date.UTC(2026, 7, 26, 0, 0, index)).toISOString(),
      }});
      const knownHistory = Array.from({{length: 100}}, (_, index) =>
        knownHistoryItem(index, index % 5 === 0 ? 2 : 0),
      );
      const duplicateKnownEvent = knownEventItem(50, 3, ["scratch", "bent", "bent"]);
      const knownBuffered = [
        duplicateKnownEvent,
        knownEventItem(100, 1, ["color"]),
        knownEventItem(101, 0, []),
      ];
      const knownMerged = state.mergeKnownDefectInspections(knownHistory, knownBuffered);
      const knownReconnected = state.mergeKnownDefectInspections(
        knownHistory,
        knownMerged,
        [duplicateKnownEvent],
      );
      const knownKpiSample = state.mergeKnownDefectInspections([
        knownEventItem(200, 0, []),
        knownEventItem(201, 3, ["bent", "scratch"]),
        knownEventItem(202, 1, ["color"]),
      ]);
      const knownValidEvent = {{
        schema_version: "1",
        type: "known_defect.created",
        inspection: duplicateKnownEvent,
      }};
      const combinedHistoryItem = (
        index,
        disposition = "PASS",
        reasonCode = "NO_ANOMALY_EVIDENCE",
        createdAt = null,
      ) => ({{
        combined_inspection_id: `20000000-0000-4000-8000-${{String(index).padStart(12, "0")}}`,
        created_at: createdAt ?? new Date(Date.UTC(2026, 7, 26, 1, 0, index)).toISOString(),
        patchcore_prediction: disposition === "PASS" ? "NORMAL" : "ANOMALY",
        known_defect_instance_count: disposition === "PASS" ? 0 : 1,
        disposition,
        reason_code: reasonCode,
        policy: {{name: "model_agreement", version: "1"}},
      }});
      const combinedEventItem = (
        index,
        disposition,
        reasonCode,
        classes,
        createdAt = null,
      ) => ({{
        combined_inspection_id: `20000000-0000-4000-8000-${{String(index).padStart(12, "0")}}`,
        created_at: createdAt ?? new Date(Date.UTC(2026, 7, 26, 1, 0, index)).toISOString(),
        patchcore_prediction: disposition === "PASS" ? "NORMAL" : "ANOMALY",
        known_defect_instance_count: classes.length,
        known_defect_classes: classes,
        disposition,
        reason_code: reasonCode,
        policy_name: "model_agreement",
        policy_version: "1",
      }});
      const combinedHistory = Array.from({{length: 100}}, (_, index) =>
        combinedHistoryItem(index),
      );
      const duplicateCombinedEvent = combinedEventItem(
        50,
        "REJECT",
        "CONFIRMED_KNOWN_DEFECT",
        ["scratch", "bent", "bent"],
      );
      const combinedBuffered = [
        duplicateCombinedEvent,
        combinedEventItem(100, "REVIEW", "UNKNOWN_ANOMALY", []),
        combinedEventItem(101, "REJECT", "CONFIRMED_KNOWN_DEFECT", ["color"]),
      ];
      const combinedMerged = state.mergeCombinedInspections(
        combinedHistory,
        combinedBuffered,
      );
      const combinedReconnected = state.mergeCombinedInspections(
        combinedHistory,
        combinedMerged,
        [duplicateCombinedEvent],
      );
      const combinedTieTime = "2026-08-26T02:00:00.000Z";
      const combinedTied = state.mergeCombinedInspections([
        combinedHistoryItem(1, "PASS", "NO_ANOMALY_EVIDENCE", combinedTieTime),
        combinedHistoryItem(2, "PASS", "NO_ANOMALY_EVIDENCE", combinedTieTime),
      ]);
      const combinedKpiSample = state.mergeCombinedInspections([
        combinedHistoryItem(200, "PASS", "NO_ANOMALY_EVIDENCE"),
        combinedEventItem(201, "REVIEW", "MODEL_DISAGREEMENT", ["bent"]),
        combinedEventItem(202, "REJECT", "CONFIRMED_KNOWN_DEFECT", ["scratch"]),
      ]);
      const combinedValidEvent = {{
        schema_version: "1",
        type: "combined_inspection.created",
        inspection: duplicateCombinedEvent,
      }};
      const normalizedCombinedDetail = state.normalizeCombinedInspection({{
        ...combinedHistoryItem(300),
        patchcore: {{inspection_id: item(1).inspection_id}},
        known_defects: {{inspection_id: knownHistoryItem(1).inspection_id}},
      }});
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
        knownLength: knownMerged.length,
        knownIds: knownMerged.map((value) => value.inspection_id),
        knownDuplicate: knownMerged.find((value) => value.inspection_id.endsWith("000000000050")),
        knownReconnectIds: knownReconnected.map((value) => value.inspection_id),
        knownKpis: state.calculateKnownDefectKpis(knownKpiSample),
        knownValidEvent: state.parseKnownDefectEvent(knownValidEvent),
        knownWrongEvent: state.parseKnownDefectEvent(validEvent),
        knownHistoryLength: state.parseKnownDefectHistory({{items: knownHistory}}).length,
        knownHttpUrl: state.knownDefectWebSocketUrl({{
          protocol: "http:",
          host: "factory.test:8000",
        }}),
        knownHttpsUrl: state.knownDefectWebSocketUrl({{protocol: "https:", host: "factory.test"}}),
        knownDetailUrl: state.knownDefectDetailUrl(duplicateKnownEvent.inspection_id),
        patchContainsKnown: merged.some((value) => value.inspection_id.startsWith("10000000")),
        combinedLength: combinedMerged.length,
        combinedIds: combinedMerged.map((value) => value.combined_inspection_id),
        combinedDuplicate: combinedMerged.find((value) =>
          value.combined_inspection_id.endsWith("000000000050"),
        ),
        combinedReconnectIds: combinedReconnected.map((value) =>
          value.combined_inspection_id,
        ),
        combinedTiedIds: combinedTied.map((value) => value.combined_inspection_id),
        combinedKpis: state.calculateCombinedKpis(combinedKpiSample),
        combinedValidEvent: state.parseCombinedInspectionEvent(combinedValidEvent),
        combinedWrongEvent: state.parseCombinedInspectionEvent(knownValidEvent),
        combinedHistoryLength: state.parseCombinedInspectionHistory({{
          items: combinedHistory,
        }}).length,
        combinedHttpUrl: state.combinedInspectionWebSocketUrl({{
          protocol: "http:",
          host: "factory.test:8000",
        }}),
        combinedHttpsUrl: state.combinedInspectionWebSocketUrl({{
          protocol: "https:",
          host: "factory.test",
        }}),
        combinedDetailUrl: state.combinedInspectionDetailUrl(
          duplicateCombinedEvent.combined_inspection_id,
        ),
        reasonLabels: [
          "NO_ANOMALY_EVIDENCE",
          "UNKNOWN_ANOMALY",
          "MODEL_DISAGREEMENT",
          "CONFIRMED_KNOWN_DEFECT",
        ].map(state.decisionReasonLabel),
        combinedDetailHasChildObjects:
          Object.hasOwn(normalizedCombinedDetail, "patchcore") ||
          Object.hasOwn(normalizedCombinedDetail, "known_defects"),
        patchContainsCombined: merged.some((value) =>
          value.inspection_id.startsWith("20000000"),
        ),
        knownContainsCombined: knownMerged.some((value) =>
          value.inspection_id.startsWith("20000000"),
        ),
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
        "00000000-0000-4000-8000-000000000002",
        "00000000-0000-4000-8000-000000000001",
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


# ADD 2026-08-26: YOLO REST/WS merge, dedupe, class enrichment와 reconnect recovery를 검증한다.
def test_known_defect_live_monitor_merge_reconnect_and_isolation() -> None:
    result = _run_state_contract()

    assert result["knownLength"] == 100
    known_ids = result["knownIds"]
    assert isinstance(known_ids, list)
    assert known_ids[0].endswith("000000000101")
    assert known_ids[1].endswith("000000000100")
    assert len(known_ids) == len(set(known_ids))
    assert result["knownReconnectIds"] == known_ids
    duplicate = result["knownDuplicate"]
    assert isinstance(duplicate, dict)
    assert duplicate["instance_count"] == 3
    assert duplicate["classes"] == ["bent", "scratch"]
    assert duplicate["inference_ms"] == 80
    assert duplicate["image_width"] == duplicate["image_height"] == 700
    assert result["patchContainsKnown"] is False


# ADD 2026-08-26: Empty/defect YOLO KPI, summary event와 same-origin route builders를 검증한다.
def test_known_defect_live_monitor_kpi_and_endpoint_contract() -> None:
    result = _run_state_contract()

    assert result["knownKpis"] == {
        "visible": 3,
        "noKnownDefect": 1,
        "knownDefect": 2,
        "totalInstances": 4,
    }
    valid_event = result["knownValidEvent"]
    assert isinstance(valid_event, dict)
    assert valid_event["classes"] == ["bent", "scratch"]
    assert result["knownWrongEvent"] is None
    assert result["knownHistoryLength"] == 100
    assert result["knownHttpUrl"] == "ws://factory.test:8000/v1/ws/known-defects"
    assert result["knownHttpsUrl"] == "wss://factory.test/v1/ws/known-defects"
    detail_url = result["knownDetailUrl"]
    assert isinstance(detail_url, str)
    assert detail_url.endswith("10000000-0000-4000-8000-000000000050")


# ADD 2026-08-26: Combined REST/event merge, dedupe, tie ordering과 child-state 격리를 검증한다.
def test_combined_live_monitor_merge_reconnect_and_isolation() -> None:
    result = _run_state_contract()

    assert result["combinedLength"] == 100
    combined_ids = result["combinedIds"]
    assert isinstance(combined_ids, list)
    assert combined_ids[0].endswith("000000000101")
    assert combined_ids[1].endswith("000000000100")
    assert len(combined_ids) == len(set(combined_ids))
    assert result["combinedReconnectIds"] == combined_ids
    assert result["combinedTiedIds"] == [
        "20000000-0000-4000-8000-000000000002",
        "20000000-0000-4000-8000-000000000001",
    ]
    duplicate = result["combinedDuplicate"]
    assert isinstance(duplicate, dict)
    assert duplicate["disposition"] == "REJECT"
    assert duplicate["known_defect_classes"] == ["bent", "scratch"]
    assert result["combinedDetailHasChildObjects"] is False
    assert result["patchContainsCombined"] is False
    assert result["knownContainsCombined"] is False


# ADD 2026-08-26: Combined disposition KPI, reason labels와 same-origin endpoints를 검증한다.
def test_combined_live_monitor_kpi_reason_and_endpoint_contract() -> None:
    result = _run_state_contract()

    assert result["combinedKpis"] == {
        "visible": 3,
        "pass": 1,
        "review": 1,
        "reject": 1,
    }
    valid_event = result["combinedValidEvent"]
    assert isinstance(valid_event, dict)
    assert valid_event["policy_name"] == "model_agreement"
    assert valid_event["policy_version"] == "1"
    assert result["combinedWrongEvent"] is None
    assert result["combinedHistoryLength"] == 100
    assert result["combinedHttpUrl"] == "ws://factory.test:8000/v1/ws/combined-inspections"
    assert result["combinedHttpsUrl"] == "wss://factory.test/v1/ws/combined-inspections"
    detail_url = result["combinedDetailUrl"]
    assert isinstance(detail_url, str)
    assert detail_url.endswith("20000000-0000-4000-8000-000000000050")
    assert result["reasonLabels"] == [
        "No anomaly evidence",
        "Unknown anomaly requires review",
        "Model disagreement requires review",
        "Confirmed known-defect evidence",
    ]
