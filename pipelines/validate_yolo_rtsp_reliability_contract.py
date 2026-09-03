"""Validate the frozen C6-4A RTSP reliability contract."""

from __future__ import annotations

from services.streaming.yolo_rtsp_reliability import (
    DEFAULT_RTSP_RELIABILITY_CONFIG,
    build_rtsp_pipeline,
    load_rtsp_reliability_config,
    reconnect_backoff_schedule_ms,
    redact_rtsp_uri,
)


# ADD 2026-09-04: Native RTSP runtime 전에 repository contract를 fail-closed로 확인한다.
def main() -> None:
    config = load_rtsp_reliability_config(DEFAULT_RTSP_RELIABILITY_CONFIG)
    sample_uri = "rtsp://camera-user:camera-password@127.0.0.1:8554/inspection"
    pipeline = build_rtsp_pipeline(config, uri=sample_uri)

    safe_uri = redact_rtsp_uri(sample_uri)
    safe_pipeline = pipeline.replace(sample_uri, safe_uri)

    if "camera-user" in safe_pipeline or "camera-password" in safe_pipeline:
        raise RuntimeError("RTSP credentials leaked into the printable pipeline.")

    print("C6-4A RTSP reliability contract: PASS")
    print(f"Contract: {config.contract_id}")
    print(f"Source env: {config.source.location_env}")
    print(f"Transport: {config.source.transport}")
    print(f"Codec: {config.source.codec}")
    print(f"Frame stale after: {config.source.frame_stale_after_ms} ms")
    print(f"Reconnect backoff: {reconnect_backoff_schedule_ms(config.reconnect)} ms")
    print(f"Backpressure: {config.backpressure.mode}")
    print(f"Health states: {', '.join(config.observability.states)}")
    print("Pipeline:", safe_pipeline)
    print("Actual RTSP used: false")
    print("TensorRT inference used: false")
    print("DeepStream used: false")
    print("Final test used: false")


if __name__ == "__main__":
    main()
