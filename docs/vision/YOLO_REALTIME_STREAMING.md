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
| C6-3 TensorRT INT8 streaming inference + end-to-end benchmark | `CLOSED / TENSORRT_INT8_STREAMING_ACCEPTED` |
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

## 10. C6-3 Python frame adapter foundation

C6-3 시작 전에 macOS local runtime에서 Python GStreamer binding을 별도 확인했다.

- Homebrew GStreamer: `1.28.6`
- project Python: `3.12.14`
- PyGObject: `3.58.0`
- PyCairo: `1.29.1`
- streaming dependency group:
  - `numpy==2.5.2`
  - `pygobject>=3.58,<4`
- ad-hoc appsink pull:
  - format: `BGR`
  - size: `320×240`
  - bytes: `230400`
  - result: `PASS`

macOS에서는 Homebrew `libffi`가 keg-only이고 uv Python이 Homebrew GLib dylib search path를
자동 상속하지 않으므로 `scripts/run_streaming_uv.sh`가 streaming runtime/build environment를
한 곳에서 구성한다. 전역 shell profile 수정이나 Homebrew Python site-packages 재사용은 하지 않는다.

C6-3 frame adapter foundation은 `GstBuffer → NumPy` 경계를 다음처럼 고정한다.

- pixel format: `BGR`
- dtype: `uint8`
- layout: `HWC`
- output memory: owned copy
- output contiguity: `C-contiguous`
- GstVideo row stride padding: 제거 후 packed frame으로 변환

이 foundation은 아직 TensorRT engine을 load하거나 inference하지 않는다. C5-4 accepted INT8 engine
identity와 final-test seal은 그대로 유지하며, Python frame boundary가 canonical runtime validation을
통과한 뒤 NVIDIA GPU 환경에서 TensorRT streaming inference를 별도 시작한다.

## 11. C6-3 Canonical Python frame validation foundation

Canonical Python frame validation은 clean repository commit에서 real PyGObject `appsink` sample을
pull하고 `GstBuffer → NumPy` adapter의 실제 runtime result를 JSON evidence로 기록한다.

Frozen validation source:

- source: `videotestsrc`
- buffers: `1`
- pattern: `ball`
- caps: `BGR`, `320×240`
- appsink: `framesink`
- backpressure: `latest_frame_wins`
- expected NumPy boundary: `BGR/uint8/HWC/C-contiguous/owned`

Canonical runner:

```bash
./scripts/run_streaming_uv.sh   python -m pipelines.run_yolo_gstreamer_python_frame
```

Evidence output root:

`outputs/streaming/yolo_gstreamer/c6_3_python_frame/`

Canonical evidence records the exact clean Git commit, local Python/PyGObject/PyCairo/NumPy/GStreamer
runtime, GstVideo row stride, frame shape/dtype/ownership/contiguity, and frame SHA-256.

이 validation은 TensorRT engine을 load하지 않고 final test도 사용하지 않는다. Canonical frame
boundary를 먼저 고정한 뒤 NVIDIA GPU runtime에서 accepted C5-4 INT8 backend streaming inference를
별도 stage로 시작한다.

## 12. C6-3 Canonical Python frame validation result

Canonical Python frame validation은 foundation commit
`bd7a59628afa2db13bfe96d3a2c64ad67e0869a6`의 clean working tree에서 실행했다.

Runtime:

- platform: `Darwin arm64`
- Python: `3.12.14`
- NumPy: `2.5.2`
- PyGObject: `3.58.0`
- PyCairo: `1.29.1`
- GStreamer: `1.28.6`

Canonical frame result:

- source: `videotestsrc`, `pattern=ball`, `num-buffers=1`
- caps: `BGR`, `320×240`
- GstVideo stride: `960 bytes`
- appsink: `framesink`
- backpressure: `latest_frame_wins`
- NumPy shape: `[240, 320, 3]`
- dtype: `uint8`
- frame bytes: `230400`
- C-contiguous: `true`
- owned memory: `true`
- frame contract: `BGR/uint8/HWC/C-contiguous/owned`
- state: `PYTHON_GSTREAMER_FRAME_ADAPTER_COMPLETED`
- TensorRT inference used: `false`
- DeepStream used: `false`
- final test used: `false`

Evidence identity:

- canonical `validation.json` SHA-256:
  `83db8f1d40bd03ab457ded7829e576e189dbbc1d3fb1fb8384be4976e71929fc`
- C6-3 Python frame config SHA-256:
  `4c20bfc683e0e20a6bdd015ddfd8ef6d24fceab3f102e390ad55facef8320fe8`
- canonical frame SHA-256:
  `e1851d821c8e04ae3f7e07e546e50a5b055b2a0ea38be00b9b4e2deac2bc852d`
- external evidence archive:
  `smart-factory-ai-platform-evidence/C6/C6-3/c6_3_python_gstreamer_frame_evidence.zip`
- external evidence archive SHA-256:
  `c63860141627e2e0aa44a7cc897acb478352728fa22f7fd01f4ecd2ea087232a`

이 결과로 C6-3의 Python `appsink → NumPy` frame boundary는 accepted 상태로 고정한다.
다만 C6-3 전체는 아직 종료하지 않는다. accepted C5-4 TensorRT INT8 engine을 NVIDIA GPU runtime에
연결한 streaming inference와 end-to-end benchmark가 남아 있으므로 다음 상태는
`FOUNDATION / PYTHON_FRAME_ACCEPTED_TRT_PENDING`이다.

Python frame validation은 C5 model quality/parity acceptance를 다시 열지 않으며,
TensorRT streaming latency/throughput acceptance를 의미하지 않는다.

## 13. C6-3B TensorRT INT8 streaming characterization foundation

C6-3B는 accepted C5-4 TensorRT INT8 engine을 **rebuild하지 않고** exact bytes로 복원한 뒤,
accepted Python `appsink → NumPy` boundary를 NVIDIA T4 runtime에 연결하는 첫 streaming
characterization 단계다.

Frozen backend identity:

- C5 state: `TENSORRT_INT8_PARITY_ACCEPTED`
- engine SHA-256:
  `4f397d59741f4efb7832087030b890a0fe059a657d074a3b07cdeb54493e8971`
- engine metadata SHA-256:
  `d44de78cc89fea67d6b351c2ba92f76dda0242386f4b6f14e216740ca682461e`
- engine config SHA-256:
  `63eebcac04d11c9247bf7543fe18d0798758ab20cc734d2b18bfbece4eaf6b41`
- engine build commit: `7835291c8fb123eba6acfa839977f94093c2f3ac`
- engine rebuild: `forbidden`

Required TensorRT runtime:

- TensorRT: `10.13.3.9.post1`
- CUDA runtime: `12.8`
- GPU: `Tesla T4`, compute capability `7.5`
- PyTorch: `2.10.0+cu128`
- Ultralytics: `8.4.128`
- device: `cuda:0`

Streaming characterization source:

- `videotestsrc`, `pattern=ball`
- `is-live=true`, `do-timestamp=true`
- `180` source buffers
- `640×640`, `30 FPS`, `BGR`
- queue/appsink: `max-buffers=1`, latest-frame-wins
- warmup: `10` TensorRT predictions before the measured streaming window

Measured scope:

`GStreamer appsink → owned NumPy BGR frame → Ultralytics TensorRT INT8 predict`

Evidence will separately record:

- frame adapter latency
- TensorRT/Ultralytics inference latency
- combined processing latency
- source / processed / dropped frame counts and drop rate
- observed processed FPS
- mean-processing-derived capacity FPS
- exact GPU/TensorRT/CUDA/PyTorch/Ultralytics/GStreamer runtime identity

This first run is **metrics-only characterization**. Numeric streaming acceptance thresholds are deliberately
`null` until the first clean T4 run is observed. Dataset, validation split, test split, final test, and
DeepStream are not used.

C6-3B expected post-run state:

`TENSORRT_INT8_STREAMING_METRICS_COLLECTED_ACCEPTANCE_PENDING`

## 14. C6-3B TensorRT INT8 streaming characterization result

Canonical C6-3B run executed on foundation commit
`8e982aeb011f6d1d92a90ad53e0a9541cd3c441a` with the exact accepted C5-4 INT8 engine.

Result:

- state: `TENSORRT_INT8_STREAMING_METRICS_COLLECTED_ACCEPTANCE_PENDING`
- source: `180` buffers, `640×640`, `30 FPS`, `BGR`
- processed: `180`
- dropped: `0`
- drop rate: `0.0`
- observed processed FPS: `30.067046090149177`
- frame adapter mean / p95: `0.578469111115333 / 0.6632636999825081 ms`
- inference mean / p95: `10.586373727777401 / 11.002993050067289 ms`
- processing mean / p95: `11.164842838892733 / 11.591202700094527 ms`
- processing capacity from mean: `89.56686757080894 FPS`
- source frame period: `33.333333333333336 ms`
- engine rebuilt: `false`
- dataset / validation / test / final test / DeepStream used: `false`

Evidence identity:

- characterization SHA-256:
  `97a4c1b233354ed362d40499e9a8e4af1b678385a6ed63a3bb394d963eb5f627`
- config SHA-256:
  `594acd505cf9ab1bdc8fbaf4028a50a8e3f475ded1f147783e844397fa3b3f8c`
- runtime preflight SHA-256:
  `7d3a997c01e186121ccd5171400b83912c25ae5075e3a5ac1a56be632f54331a`
- run summary SHA-256:
  `f1c03e4c73a5c4dd75eb80158c5a565ace045bd1d81bdf5a1d8187e2fddad042`
- external evidence ZIP SHA-256:
  `f6a9f994c2efa7e38954a256fdfcdaef4792edfe46ba7c0580add59130677bb7`

## 15. C6-3C Prospective streaming acceptance policy v1

C6-3B was deliberately threshold-free. The following policy is defined **after** characterization and must
be committed before a fresh prospective C6-3D run. C6-3B itself is not retroactively accepted by these
thresholds.

Performance gates:

- drop rate `<= 0.01`
- processed frames `>= 179 / 180`
- observed processed FPS `>= 29.0`
- frame adapter p95 `<= 1.5 ms`
- inference mean `<= 13.0 ms`
- inference p95 `<= 15.0 ms`
- processing mean `<= 14.0 ms`
- processing p95 `<= 16.0 ms`
- processing capacity from mean `>= 70 FPS`
- processing p95 must remain below the `33.333... ms` 30 FPS source frame period

These thresholds retain explicit headroom from the first C6-3B observation while still requiring
30 FPS real-time viability. At most one of the fixed 180 source frames may be dropped under the
`0.01` drop-rate gate.

Structural gates additionally require:

- a fresh clean repository run after this policy is committed
- prospective characterization commit equals the current policy repository commit
- exact accepted C5-4 INT8 engine and T4 runtime identity
- `engine_rebuilt=false`
- dataset / validation / test / final test / DeepStream all unused

The next stage is C6-3D prospective execution. Acceptance or rejection must be based only on that
fresh run, not on the C6-3B characterization used to design this policy.

C6-3C policy identity:

- policy commit: `ec72151bc595759f2de01671b487028fb8de74e1`
- policy config SHA-256:
  `6279cbfbbdcf2a57a1c69ad69158a45f73ea7be604bb255fbefddd9e9e78cd76`
- state: `FROZEN / POLICY_COMMITTED_PROSPECTIVE_RUN_PENDING`
- prospective C6-3D execution performed at policy-freeze time: `false`

## 16. C6-3D Prospective TensorRT INT8 streaming acceptance result

C6-3D는 C6-3C policy를 먼저 commit한 뒤, docs-only closure commit
`35f2405ec66257595f8b15b56253f0afa9556324`의 clean working tree에서 fresh prospective
streaming characterization을 다시 실행했다. C6-3B metrics는 threshold 설계에만 사용했고,
C6-3D acceptance에는 재사용하지 않았다.

Frozen policy identity:

- policy freeze commit:
  `ec72151bc595759f2de01671b487028fb8de74e1`
- policy config SHA-256:
  `6279cbfbbdcf2a57a1c69ad69158a45f73ea7be604bb255fbefddd9e9e78cd76`

Prospective runtime/backend identity:

- repository commit:
  `35f2405ec66257595f8b15b56253f0afa9556324`
- TensorRT INT8 engine SHA-256:
  `4f397d59741f4efb7832087030b890a0fe059a657d074a3b07cdeb54493e8971`
- TensorRT: `10.13.3.9.post1`
- CUDA runtime: `12.8`
- PyTorch: `2.10.0+cu128`
- Ultralytics: `8.4.128`
- GPU: `Tesla T4`, compute capability `7.5`
- engine rebuilt: `false`
- dataset / validation / test / final test / DeepStream used: `false`

Fresh prospective result:

- source: `180` buffers, `640×640`, `30 FPS`, `BGR`
- processed frames: `180`
- dropped frames: `0`
- drop rate: `0.0`
- observed processed FPS: `30.07497976161947`
- frame adapter mean / p95:
  `0.8436739444530557 / 1.0077304500100581 ms`
- inference mean / p95:
  `10.90899682778349 / 11.662175400113028 ms`
- processing mean / p95:
  `11.752670772236545 / 12.534434399981365 ms`
- processing capacity from mean:
  `85.08704271392595 FPS`
- source frame period:
  `33.333333333333336 ms`
- acceptance gates: `31 / 31`
- failed gates: `0`
- state: `TENSORRT_INT8_STREAMING_ACCEPTED`

C6-3D evidence identity:

- prospective characterization SHA-256:
  `6e8b0162c287ff27c9d9d7315328ade2cc1944f24ac0e31c8f3aa67bf8b0be19`
- acceptance SHA-256:
  `23b0717b114a579290de56babc5afdd09f6e71c3873b32e1547511c6e251a35e`
- runtime preflight SHA-256:
  `7d3a997c01e186121ccd5171400b83912c25ae5075e3a5ac1a56be632f54331a`
- run summary SHA-256:
  `65b07ea2cacc9fb1b9bb48b19a9b5b8bda81f7f63e4d78630ae4b48b0f34c281`
- external evidence archive:
  `smart-factory-ai-platform-evidence/C6/C6-3/C6-3D/c6_3d_tensorrt_int8_streaming_acceptance_evidence.zip`
- external evidence archive SHA-256:
  `c519af4f861a80df735cbc66f06660aa842321a689e8eb17ae1e3203736bf679`

C6-3의 목적이었던 accepted C5-4 TensorRT INT8 backend의 GStreamer appsink streaming 연결과
30 FPS prospective latency/throughput acceptance가 완료됐다. 따라서 C6-3 lifecycle은
`CLOSED / TENSORRT_INT8_STREAMING_ACCEPTED`로 종료한다.

다음 단계 C6-4에서 실제 RTSP source를 추가하고 reconnect, timeout, stale-frame/backpressure,
stream health observability를 별도 contract와 runtime evidence로 검증한다. C6-3 acceptance는
C6-4에서 threshold를 다시 조정하기 위한 근거로 소급 변경하지 않는다.
