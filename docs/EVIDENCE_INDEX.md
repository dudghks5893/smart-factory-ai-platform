# Project Evidence Index

## 1. 목적과 사용 원칙

이 문서는 SmartFactory AI Quality Platform의 주요 검증 결과를 빠르게 찾기 위한 **navigation index**다.
새 실험, 새 metric, 새 acceptance decision을 만들지 않으며 각 lifecycle의 canonical 문서를 대체하지 않는다.

Evidence 해석 원칙:

- repository의 문서·config·commit은 contract와 provenance의 source of truth다.
- model, engine, runtime JSON/ZIP 같은 큰 binary evidence는 Git 밖
  `smart-factory-ai-platform-evidence/` namespace에 보존한다.
- external artifact는 SHA-256과 생성 당시 repository identity로 교차 검증한다.
- validation, final-test, backend parity, streaming runtime처럼 measurement boundary가 다른 결과를 합산해
  하나의 overall score로 만들지 않는다.
- C4 one-time final-test 결과는 report-only이며 candidate/threshold를 다시 선택하는 근거로 사용하지 않는다.
- C5/C6는 C4 model-quality acceptance를 다시 열지 않는다.
- production deployment, 실제 공장 camera 검증, factory certification은 완료했다고 표현하지 않는다.

## 2. Canonical evidence map

| Domain | Final state | Canonical document | 무엇을 증명하는가 |
|---|---|---|---|
| STEP 15 cross-domain benchmark | authoritative benchmark | [Final Benchmark](benchmarks/FINAL_BENCHMARK.md) | PatchCore quality/runtime, API, RAG, platform verification의 당시 승인 결과 집계 |
| C4 YOLO model quality | `CLOSED` | [YOLO Experiment Log](vision/YOLO_SEGMENTATION_EXPERIMENT_LOG.md) | validation-only candidate selection과 frozen candidate의 one-time final-test |
| C5 YOLO deployment optimization | `CLOSED` | [YOLO Deployment Optimization](vision/YOLO_DEPLOYMENT_OPTIMIZATION.md) | ONNX/FP16/INT8 deployment backend parity와 accepted TensorRT INT8 identity |
| C6 real-time streaming/service | `CLOSED / REALTIME_STREAMING_ACCEPTED` | [YOLO Real-Time Streaming](vision/YOLO_REALTIME_STREAMING.md) | GStreamer/RTSP, TensorRT, DeepStream GPU/NVMM, FastAPI/WebSocket service E2E runtime |

STEP 15 benchmark는 PatchCore/RAG/platform의 기존 cross-domain snapshot이고,
그 이후 완료된 YOLO C4/C5/C6 lifecycle을 소급 포함하는 artifact가 아니다.

## 3. C4 — YOLO11n-seg model-quality evidence

Frozen candidate:

- experiment:
  `c4_2c_yolo11n_seg_crop350_nomosaic_maskratio2_seed42`
- final-candidate manifest SHA-256:
  `2a26b1bc03a1876f828e12a625c69c76af5e8c5713e3f64be699feffe2e8aa09`
- frozen model SHA-256:
  `e3fd10cdd708d31421feacfc5d694cb638e0ea60672e08796391b33aecf67155`
- C4-3 freeze commit:
  `9c6916c74beed01875421e2faf1f8113232f2d15`
- C4-4 guarded execution commit:
  `e15fd92776a3981a1b5927ad567802d0d0a3bb54`

Key validation/final-test results:

| Metric | Result | Boundary |
|---|---:|---|
| Validation Mask mAP50-95 | `0.4623876120` | C4-2C validation-only selection evidence |
| Validation strict diagnostic Recall | `0.7391304348` | confidence 0.25 / mask IoU 0.5 diagnostic |
| Final-test Mask mAP50-95 | `0.4439883323` | frozen candidate, report-only |
| Final-test strict diagnostic Recall | `0.7894736842` | one-time final-test diagnostic |
| Final-test good-negative FP images | `0 / 14` | report-only negative diagnostic |

Boundary guards:

- candidate frozen before final test: `true`
- candidate selection changed after test: `false`
- threshold tuned on test: `false`
- C4 final state: `CLOSED`

External evidence namespace:

```text
smart-factory-ai-platform-evidence/
└── C4-4/
```

C4-4의 exact result/evidence identity와 failure/retry history는
[YOLO Experiment Log](vision/YOLO_SEGMENTATION_EXPERIMENT_LOG.md)를 authoritative source로 사용한다.

## 4. C5 — ONNX / TensorRT deployment evidence

C5는 C4 frozen model을 다시 학습하지 않고 deployment backend equivalence를 검증했다.

Key artifact identity:

- source model SHA-256:
  `e3fd10cdd708d31421feacfc5d694cb638e0ea60672e08796391b33aecf67155`
- C5-1 FP32 ONNX SHA-256:
  `f916325bb126d174de9c1fdfc24802eec11c46014f723fbf3ba3b3c1755c1490`
- accepted T4 TensorRT INT8 engine SHA-256:
  `4f397d59741f4efb7832087030b890a0fe059a657d074a3b07cdeb54493e8971`
- INT8 acceptance policy SHA-256:
  `938c06a099b681de9ac48d95132f423f5255ba4527f05d3f27f75d9eae5ad56c`
- policy commit:
  `f9369f475b955fafb3ee990e2b2de63d04aa651f`
- C5-4 closure commit:
  `88e9b0b2440e99b6dfd2594bdc9a4947eff75187`

C5-4E prospective validation result:

- validation samples: `28`
- PyTorch / TensorRT INT8 predictions: `19 / 19`
- unmatched predictions: `0 / 0`
- class agreement: `1.0`
- confidence absolute error max: `0.10978221893310547`
- box IoU min: `0.9381443298969072`
- mask IoU min: `0.9405993578308954`
- PyTorch FP32 GPU mean: `32.65692755999339 ms`
- TensorRT INT8 mean: `28.274256820004666 ms`
- same-session speedup: `1.1550056918520981x`
- frozen policy checks: `39 / 39 PASS`
- final state: `TENSORRT_INT8_PARITY_ACCEPTED`
- final-test used: `false`

Authoritative C5-4E archive:

```text
smart-factory-ai-platform-evidence/C5/C5-4E/
└── c5_4e_tensorrt_int8_prospective_acceptance_evidence.zip
```

Archive SHA-256:

`ef4a4f1123a63ffe35ed12917bf6ae829b2c4a5892f4eaac43f9732c8e3ee638`

C5의 T4 accepted engine과 C6에서 NVIDIA L4용으로 검증한 runtime plan은 **서로 다른 hardware/runtime
artifact identity**다. SHA가 다르다는 사실을 backend mismatch로 해석하지 않는다.

## 5. C6 — real-time streaming and service E2E evidence

C6 lifecycle:

- C6-1 GStreamer ingress contract: `FROZEN / CONTRACT_COMMITTED`
- C6-2 native GStreamer smoke: `CLOSED / NATIVE_SMOKE_ACCEPTED`
- C6-3 TensorRT INT8 streaming: `CLOSED / TENSORRT_INT8_STREAMING_ACCEPTED`
- C6-4 RTSP reliability: `CLOSED / RTSP_RELIABILITY_ACCEPTED`
- C6-5 DeepStream GPU/NVMM: `CLOSED / DEEPSTREAM_GPU_NVMM_SEGMENTATION_ACCEPTED`
- C6-6 service integration: `CLOSED / SERVICE_E2E_ACCEPTED`
- C6 final state: `CLOSED / REALTIME_STREAMING_ACCEPTED`

Important runtime identities:

- NVIDIA L4 / compute capability: `8.9`
- DeepStream runtime plan SHA-256:
  `97acd724809f4817ad4a95525a1bafae6294b1a7c99e04c12d451eeda878866e`
- DeepStream segmentation parser SO SHA-256:
  `5cc5f9accc465b1c8dc5b8dd59a5983db85bbd39da89dd1322a7bcd910ad2728`
- C6-6 mask semantic repair commit:
  `04f3fa805825216aa41647734894290a6f8a4635`
- C6-6/C6 closure commit:
  `778a1a270674a50dfaebdb2681890d1815e9b86c`

Corrected C6-6 GPU revalidation:

- DeepStream frames observed: `34`
- compact defect frames: `32`
- compact instances: `54`
- mask semantic instances validated: `54 / 54`
- publisher delivered events: `32`
- WebSocket events: `32`
- exact event matches: `PASS`
- bbox-local mask grid semantics: `PASS`
- source-space mask area estimate: `PASS`
- raw frame transported to Python/service: `false`
- raw mask transported to Python/service: `false`
- DeepStream worker container network used: `false`
- final-test used: `false`

Corrected evidence:

```text
smart-factory-ai-platform-evidence/C6/C6-6/
├── c6_6_service_e2e_gpu_revalidation.json
└── c6_6_service_e2e_gpu_revalidation_evidence.zip
```

- JSON SHA-256:
  `d3f1ffd2e0f1dd0a5b24b5c3a56f6f68e1bb9a34862776889ccb2218a76fe13f`
- JSON bytes: `6483`
- ZIP SHA-256:
  `2a7502878f7462c0b7740ade0a1cc00345e6ed51a50ec16a0c43fb9c5b0b36c9`
- ZIP bytes: `2016`
- VM/Mac corrected evidence identity: `PASS`

C6-6 pre-fix runtime evidence는 실행 이력으로 보존하지만, 당시 raw threshold-positive mask sample count를
source image area로 나눈 `mask.area_ratio`는 source-space mask semantics acceptance로 사용하지 않는다.
최종 mask semantics acceptance는 위 corrected revalidation evidence만 기준으로 한다.

C6-5D의 NVIDIA sample road video evidence는 **DeepStream pipeline/runtime 구조 검증용**이다.
실제 제조 불량 품질 결과나 factory/live-camera evidence로 표현하지 않는다.

## 6. STEP 15 cross-domain benchmark

[Final Benchmark](benchmarks/FINAL_BENCHMARK.md)는 다음을 하나의 reproducible evidence contract로 집계한다.

- PatchCore image-level quality
- pixel localization quality
- Tesla T4 model runtime
- FastAPI application-level benchmark
- deterministic public-demo RAG evaluation
- platform engineering verification matrix

대표 결과:

- PatchCore image AUROC / F1: `0.997556 / 0.994595`
- Pixel AUROC / F1: `0.982486 / 0.834279`
- T4 model runtime p50 / p95 / p99:
  `21.634 / 25.775 / 27.113 ms`
- FastAPI schema v1 p50 / p95 / p99:
  `44.902 / 48.703 / 53.746 ms`

이 latency들은 measurement boundary가 다르므로 서로 차감하지 않는다.
STEP 15 authoritative output은 `outputs/` 정책상 Git에 commit하지 않으며 source SHA와 repository provenance를
문서와 versioned benchmark snapshot으로 추적한다.

## 7. Evidence storage map

Repository가 참조하는 external evidence의 portable namespace는 다음과 같다.

```text
smart-factory-ai-platform-evidence/
├── C4-4/
├── C5/
│   └── C5-4E/
└── C6/
    ├── C6-5D/
    └── C6-6/
```

실제 machine absolute path는 repository contract에 넣지 않는다.
Mac/VM mirror는 동일 SHA 검증을 통과한 durable copy로만 취급한다.

## 8. Claim boundary

현재 evidence로 직접 뒷받침되는 주장:

- validation-only candidate selection과 one-time report-only YOLO final-test를 분리했다.
- frozen YOLO model을 ONNX/TensorRT로 변환하고 T4에서 INT8 parity를 prospective policy로 검증했다.
- NVIDIA L4에서 TensorRT + DeepStream GPU/NVMM segmentation runtime을 실행했다.
- compact metadata를 host publisher → HTTP 202 → FastAPI → dedicated WebSocket까지 실제 연결했다.
- corrected bbox-local mask semantics를 GPU runtime에서 재검증했다.
- PatchCore, PostgreSQL integration, MLOps/monitoring/dashboard/RAG/deployment foundation을 repository 수준에서
  구현·검증했다.

현재 evidence만으로 주장하지 않는 항목:

- 실제 공장 생산라인 또는 실제 live camera에서의 품질 성능
- production GKE/Cloud SQL 배포 완료
- factory certification 또는 production calibration 완료
- NVIDIA sample runtime을 제조 불량 detection quality evidence로 사용하는 것
- 아직 만들지 않은 portfolio demo를 이미 완료된 runtime evidence로 표현하는 것

## 9. Portfolio/demo 다음 단계

최종 portfolio demo는 C6 NVIDIA sample evidence와 분리한다.

계획된 reproducible demo boundary:

```text
demo-only original defect photos
    ↓ image-sequence-derived unannotated video
actual TensorRT + DeepStream + YOLO11n-seg runtime
    ↓
model-generated boxes / masks / compact events
```

Source image/video에 box나 mask를 미리 그리지 않는다.
가능하면 sealed final-test data를 다시 사용하지 않고 demo-only/non-final input을 사용한다.
이 demo가 실제 factory/live-camera recording이 아니라는 점도 명시한다.

## 10. 빠른 탐색 순서

1. 프로젝트 전체 구조: [README](../README.md)
2. 기존 cross-domain benchmark: [Final Benchmark](benchmarks/FINAL_BENCHMARK.md)
3. YOLO model quality: [YOLO Experiment Log](vision/YOLO_SEGMENTATION_EXPERIMENT_LOG.md)
4. YOLO deployment optimization: [YOLO Deployment Optimization](vision/YOLO_DEPLOYMENT_OPTIMIZATION.md)
5. Real-time streaming/service: [YOLO Real-Time Streaming](vision/YOLO_REALTIME_STREAMING.md)

이 index는 provenance를 찾기 위한 출발점이며, 최종 판단에는 각 canonical 문서의 exact boundary와 SHA를 사용한다.
