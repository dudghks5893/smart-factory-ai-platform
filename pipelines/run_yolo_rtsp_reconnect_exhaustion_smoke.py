"""Run the canonical C6-4C RTSP reconnect-exhaustion smoke."""

from __future__ import annotations

import json
from pathlib import Path

from services.streaming.yolo_rtsp_reconnect_exhaustion_smoke import (
    DEFAULT_RTSP_RECONNECT_EXHAUSTION_CONFIG,
    load_rtsp_reconnect_exhaustion_config,
    run_rtsp_reconnect_exhaustion_smoke,
)


# ADD 2026-09-04: Canonical C6-4C exhaustion/fail-closed smoke를 repo root에서 실행한다.
def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    config = load_rtsp_reconnect_exhaustion_config(DEFAULT_RTSP_RECONNECT_EXHAUSTION_CONFIG)
    output_path = run_rtsp_reconnect_exhaustion_smoke(config, repo=repo)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    exhaustion = payload["reconnect_exhaustion"]
    observability = payload["observability"]

    print("C6-4C RTSP reconnect-exhaustion smoke: PASS")
    print("State:", payload["state"])
    print("Evidence:", output_path)
    print("Run commit:", payload["repository"]["git_commit"])
    print("Reconnect attempts:", observability["rtsp_reconnects_total"])
    print("Expected backoffs ms:", exhaustion["expected_backoff_ms"])
    print(
        "Actual backoffs ms:",
        [attempt["actual_backoff_ms"] for attempt in exhaustion["attempts"]],
    )
    print("Budget exhausted:", exhaustion["budget_exhausted"])
    print("Final state:", observability["final_state"])
    print("Stream up:", observability["rtsp_stream_up"])
    print("External camera used: false")
    print("TensorRT inference used: false")
    print("DeepStream used: false")
    print("Final test used: false")


if __name__ == "__main__":
    main()
