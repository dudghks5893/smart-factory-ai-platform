"""Run the canonical C6-4B localhost RTSP fault-injection smoke."""

from __future__ import annotations

import json
from pathlib import Path

from services.streaming.yolo_rtsp_fault_injection_smoke import (
    DEFAULT_RTSP_FAULT_INJECTION_CONFIG,
    load_rtsp_fault_injection_config,
    run_rtsp_fault_injection_smoke,
)


# ADD 2026-09-04: Canonical C6-4B fault-injection smoke를 repository root에서 실행한다.
def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    config = load_rtsp_fault_injection_config(DEFAULT_RTSP_FAULT_INJECTION_CONFIG)
    output_path = run_rtsp_fault_injection_smoke(config, repo=repo)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    print("C6-4B RTSP fault-injection smoke: PASS")
    print("State:", payload["state"])
    print("Evidence:", output_path)
    print("Run commit:", payload["repository"]["git_commit"])
    print("Fault event:", payload["fault_phase"]["detected_event"])
    print("Fault detection ms:", payload["fault_phase"]["detection_ms"])
    print("Reconnect attempts:", payload["observability"]["rtsp_reconnects_total"])
    print("Requested backoff ms:", payload["recovery_phase"]["requested_backoff_ms"])
    print("Actual backoff ms:", payload["recovery_phase"]["actual_backoff_ms"])
    print("Recovery frames:", payload["recovery_phase"]["healthy_frames_received"])
    print("Final state:", payload["observability"]["final_state"])
    print("External camera used: false")
    print("TensorRT inference used: false")
    print("DeepStream used: false")
    print("Final test used: false")


if __name__ == "__main__":
    main()
