# YOLO Real-Time Streaming

## 1. Scope

C6는 C5에서 acceptance가 끝난 YOLO11n-seg TensorRT backend를 실제 영상 스트림으로 연결하는 단계다.
모델을 다시 학습하거나 C4/C5 acceptance threshold를 다시 조정하지 않는다.

현재 출발점은 C5-4에서 `TENSORRT_INT8_PARITY_ACCEPTED` 된 exact engine이다.

- C5-4 closure commit:
  `88e9b0b2440e99b6dfd2594bdc9a4947eff75187`
- TensorRT INT8 engine SHA-256:
  `4f397d59741f4efb7832087030b890a0fe059a657d074a3b07cdeb54493e8971`
- INT8 acceptance policy SHA-256:
  `938c06a099b681de9ac48d95132f423f5255ba4527f05d3f27f75d9eae5ad56c`
- C5-4 final state:
  `TENSORRT_INT8_PARITY_ACCEPTED`

## 2. C6 lifecycle

| Stage | Status |
|---|---|
| C6-1 GStreamer ingress contract | `FROZEN / CONTRACT_COMMITTED` |
| C6-2 Native GStreamer synthetic/file smoke test | `CLOSED / NATIVE_SMOKE_ACCEPTED` |
| C6-3 TensorRT INT8 streaming inference + end-to-end benchmark | `NOT STARTED` |
| C6-4 RTSP reconnect/backpressure/observability | `NOT STARTED` |
| C6-5 DeepStream GPU/NVMM integration | `NOT STARTED` |
| C6-6 Service integration and closure | `NOT STARTED` |

## 3. Why GStreamer first

DeepStream은 GStreamer 위에 구축된 NVIDIA streaming SDK다. 따라서 C6에서는 먼저 source decode,
frame contract, queue/backpressure와 appsink boundary를 독립적으로 고정한다. 이 단계가 안정화된 뒤
TensorRT inference와 DeepStream/NVMM zero-copy path를 추가한다.

C6-1은 native GStreamer 실행을 요구하지 않는다. macOS에서도 repository contract와 pipeline builder를
unit test할 수 있고, 실제 `gst-launch-1.0` runtime smoke test는 C6-2에서 별도 수행한다.

## 4. Frame contract

C6-1 appsink output:

- pixel format: `BGR`
- dtype: `uint8`
- layout: `HWC`
- contiguous: `true`
- sink name: `framesink`
- pull-based consumption (`emit-signals=false`)

C6-1에서는 CPU raw frame boundary를 명시적으로 사용한다. DeepStream 단계에서는 이 contract를 그대로
가정해 NVMM path를 억지로 흉내 내지 않고, 별도 GPU-memory contract를 정의한다.

## 5. Backpressure policy

실시간 품질검사에서 처리 속도가 입력 FPS보다 느려질 때 오래된 frame을 계속 쌓으면 throughput보다
end-to-end latency가 빠르게 악화된다. 따라서 첫 policy는 `latest_frame_wins`다.

- queue max buffers: `1`
- queue leaky: `downstream`
- appsink max buffers: `1`
- appsink drop: `true`
- appsink sync: `false`

Frame drop 자체는 오류가 아니다. C6-3/C6-4에서 captured/processed/dropped FPS와 end-to-end latency를
관측 metric으로 추가한다.

## 6. Source boundary

C6-1:

- `test`: `videotestsrc` 기반 deterministic live-like source
- `file`: local video file / `file://` URI

RTSP는 C6-4에서 reconnect, timeout, transport와 stale-frame policy를 함께 정의한다. C6-1에서 RTSP를
미리 허용하지 않는다.

## 7. Non-goals

C6-1에서는 다음을 하지 않는다.

- TensorRT engine rebuild
- INT8 quantization 재수행
- C4/C5 threshold tuning
- final-test 재사용
- RTSP production policy
- DeepStream plugin/NVMM runtime
- camera hardware certification

C6-1 local validation 결과:

- targeted unit tests: `6 passed`
- targeted Ruff: `PASS`
- targeted mypy: `PASS`
- contract CLI: `PASS`
- full repository `make check`:
  - Ruff format: `320 files already formatted`
  - Ruff check: `PASS`
  - mypy: `274 source files / PASS`
  - pytest: `622 passed, 2 skipped, 46 warnings`
- native `gst-launch-1.0`: 아직 설치하지 않음
- C6-2 native runtime smoke test: 아직 시작하지 않음

이 결과를 기준으로 C6-1 ingress contract를 clean repository commit으로 freeze한다.
실제 GStreamer executable과 plugin runtime은 C6-2에서 별도 검증한다.

## 8. C6-2 Native GStreamer smoke foundation

개발용 macOS에서 Homebrew GStreamer `1.28.6` 설치 후 ad-hoc preflight를 먼저 수행했다.
`videotestsrc`, `videoconvert`, `queue`, `appsink`, `uridecodebin`, `x264enc`, `h264parse`,
`mp4mux` plugin이 모두 발견됐고 synthetic live-like pipeline과 generated local MP4 decode가
각각 exit code `0`으로 종료됐다. 이 preflight는 runtime availability 확인용이며 canonical
repository evidence로 소급 사용하지 않는다.

Canonical C6-2 run은 이 foundation을 clean commit으로 먼저 고정한 뒤 실행한다. Runner는:

- exact repository commit과 clean working tree를 기록한다.
- required GStreamer plugin 8개를 fail-closed로 확인한다.
- 30-buffer `videotestsrc`를 C6-1 BGR/appsink boundary로 실행한다.
- temporary 320×240, 30 FPS H264/MP4 fixture를 생성한다.
- `uridecodebin`으로 fixture를 decode해 동일 BGR/appsink boundary를 검증한다.
- `outputs/streaming/yolo_gstreamer/c6_2_native_smoke/` 아래 JSON evidence를 생성한다.
- TensorRT inference, DeepStream, RTSP, final-test는 사용하지 않는다.

C6-2 evidence는 runtime smoke evidence이며 model quality 또는 C5 parity acceptance를 다시 열지 않는다.

## 9. C6-2 Canonical native smoke result

C6-2 canonical run은 foundation commit `8c34064c84fa00933f1180a42894d466a41a9cc7`의 clean working tree에서 실행했다.

Runtime:

- platform: `Darwin arm64`
- Python: `3.12.14`
- GStreamer: `1.28.6`
- `gst-launch-1.0`: `/opt/homebrew/bin/gst-launch-1.0`
- `gst-inspect-1.0`: `/opt/homebrew/bin/gst-inspect-1.0`
- required plugins 8개: 모두 `PASS`

Canonical runtime result:

- synthetic `videotestsrc` 30 buffers: exit code `0`
- synthetic frame contract: `BGR/uint8/HWC`
- backpressure: `latest_frame_wins`
- generated local MP4 fixture size: `30083 bytes`
- `uridecodebin` local-file decode: exit code `0`
- final test used: `false`
- TensorRT inference used: `false`
- DeepStream used: `false`
- state: `NATIVE_GSTREAMER_SMOKE_COMPLETED`

Evidence identity:

- canonical `smoke.json` SHA-256:
  `f4d0e4a609aa3d8cd59741a58e0ef513071b6096b9543cae34a76b488cd0563e`
- C6-2 config SHA-256:
  `784c063dd171c9f9fdb5972dfed8e1be3376f6a8c1629941dc28ebc3e1bf59db`
- external evidence archive:
  `smart-factory-ai-platform-evidence/C6/C6-2/c6_2_native_gstreamer_smoke_evidence.zip`
- external evidence archive SHA-256:
  `5bc99e50fb30b4427ae2e49d43aff87a383d62c0c86eba4009f581335f3748f1`

C6-2는 native GStreamer source/decode/appsink boundary가 실제 macOS runtime에서 동작함을 확인한 단계다.
이 결과는 TensorRT latency/quality acceptance 또는 DeepStream GPU-memory path를 의미하지 않는다.
따라서 C6-3에서 accepted C5-4 TensorRT INT8 backend를 별도 NVIDIA GPU runtime에 연결하고,
C6-5에서 DeepStream/NVMM contract를 별도로 검증한다.

C6-2 final state: `CLOSED / NATIVE_SMOKE_ACCEPTED`.
