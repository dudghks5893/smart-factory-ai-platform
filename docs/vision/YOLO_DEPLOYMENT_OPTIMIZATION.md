# YOLO Deployment Optimization

## 1. 상태와 범위

C5는 C4에서 확정한 YOLO11n-seg candidate를 다시 학습하거나 선택하는 단계가 아니다.
동일 frozen model을 deployment runtime으로 변환하고 backend equivalence를 검증한다.

| Stage | State |
| --- | --- |
| C5-1 ONNX FP32 export | `EXECUTED / ONNX_EXPORT_COMPLETED` |
| C5-2A PyTorch ↔ ONNX characterization | `EXECUTED / CHARACTERIZATION_COMPLETED` |
| C5-2B ONNX FP32 acceptance policy v1 | `EXECUTED / PARITY_ACCEPTED` |
| C5-2 ONNX FP32 parity | `CLOSED` |
| C5-3A TensorRT FP16 engine | `EXECUTED / TENSORRT_FP16_ENGINE_BUILT` |
| C5-3B TensorRT FP16 characterization | `EXECUTED / CHARACTERIZATION_COMPLETED` |
| C5-3C TensorRT FP16 acceptance policy v1 | `EXECUTED / PARITY_ACCEPTED` |
| C5-3 TensorRT FP16 parity | `CLOSED` |
| C5-4A INT8 explicit-Q/DQ PTQ contract | `FROZEN / CONTRACT_COMMITTED` |
| C5-4B1 ModelOpt INT8 Q/DQ ONNX | `FOUNDATION / LOCAL VALIDATION PENDING` |
| C5-4B2 TensorRT INT8 engine | `NOT STARTED` |
| C5-4C INT8 validation characterization | `NOT STARTED` |
| C5-4D INT8 acceptance policy v1 | `NOT STARTED` |
| C5-4E INT8 prospective verification | `NOT STARTED` |

C5-1과 첫 C5-2 characterization은 clean detached Git commit
`643ed9386a61bd2bf0c041f92a10b809b6d52c3e`에서 Kaggle로 실행했다. 첫 parity run은
수치 허용치를 정하기 위한 characterization이며, 사전에 선언된 numeric gate를 통과한 run으로 소급해
표현하지 않는다.

이후 acceptance policy v1을 commit
`1f48a047d14f032dad41f2cd4519399adf4d6bce`에서 고정한 뒤, 기존 C5-1 exact ONNX artifact를 복원해
validation parity를 새로 실행했다. Frozen policy의 17개 acceptance check를 모두 통과해
`PARITY_ACCEPTED`가 확인됐으며 C5-2는 `CLOSED` 상태다.

C5-3A/B는 exact accepted ONNX를 사용해 Tesla T4에서 TensorRT FP16 engine build와 validation-only
characterization을 완료했다. C5-3C acceptance policy v1은 이 관측 이후 별도 repository contract로
고정했다. Policy commit의 clean state에서 C5-3B의 exact TensorRT engine을 rebuild 없이 복원하고
새 validation-only parity evidence를 생성한 뒤 frozen policy를 적용했다. 34개 acceptance check를 모두
통과해 `TENSORRT_FP16_PARITY_ACCEPTED`를 확인했으며 C5-3 TensorRT FP16 parity lifecycle은 `CLOSED`다.
INT8은 별도의 calibration dataset과 accuracy-loss budget을 정의한 뒤 후속 C5-4에서 검토한다.

## 2. Immutable C4 source

ONNX export source는 C4-3에서 validation-only로 freeze된 candidate 하나뿐이다.

| Field | Value |
| --- | --- |
| Experiment | `c4_2c_yolo11n_seg_crop350_nomosaic_maskratio2_seed42` |
| Frozen manifest | `configs/model/yolo_segmentation_final_candidate.json` |
| Frozen manifest SHA-256 | `2a26b1bc03a1876f828e12a625c69c76af5e8c5713e3f64be699feffe2e8aa09` |
| Model SHA-256 | `e3fd10cdd708d31421feacfc5d694cb638e0ea60672e08796391b33aecf67155` |
| Official package SHA-256 | `81c721ab6d34e5563e9f8907fe4c9914d50e48ef35aacfabb6f4ca745420cd76` |
| Model / task | `yolo11n-seg` / `segment` |

Export pipeline은 frozen manifest bytes, Official package 전체 hash와 package evidence를 교차 검증한다.
Fixed package entry의 `model.pt`와 `metadata.json`만 임시 공간에 복원하고 model hash를 export 전후에
재검증한다. 다른 checkpoint를 조용히 받아들이지 않는다.

C4-4 final-test 결과는 report-only evidence다. C5의 export parameter, runtime tolerance, checkpoint,
threshold 또는 candidate selection에 사용하지 않는다.

## 3. C5-1 ONNX FP32 export

Repository export config는
[`configs/export/yolo_segmentation_onnx.yaml`](../../configs/export/yolo_segmentation_onnx.yaml)이다.
이 config는 실제 ONNX metadata에 hash로 연결됐으므로 C5-2 acceptance threshold를 추가하기 위해 수정하지
않는다.

| Parameter | Value |
| --- | --- |
| format / task | ONNX / segmentation |
| precision | FP32 |
| batch / input | 1 / static `1×3×640×640` |
| opset | 18 |
| dynamic / simplify | false / false |
| graph NMS | false |
| device | CPU |

Pinned Ultralytics `8.4.128` exporter를 사용하며 repository가 model graph를 재구현하지 않는다.
Generated graph는 ONNX checker와 repository graph contract를 통과해야 한다.

### 3.1 실제 C5-1 실행 결과

Kaggle에서 exact commit `643ed9386a61bd2bf0c041f92a10b809b6d52c3e`을 detached checkout한 뒤
locked environment로 실행했다.

- State: `ONNX_EXPORT_COMPLETED`
- ONNX SHA-256: `f916325bb126d174de9c1fdfc24802eec11c46014f723fbf3ba3b3c1755c1490`
- ONNX size: `11,584,716` bytes
- Export config SHA-256: `f1c2ef5045fdd89d964b2dc79c501580c9f55c2a1d38f38f13cf4794bafd0e85`
- Input: `images`, FP32, `(1, 3, 640, 640)`
- `output0`: FP32, `(1, 39, 8400)`
- `output1`: FP32, `(1, 32, 160, 160)`
- ONNX: `1.22.0`
- PyTorch: `2.13.0+cu130`
- Ultralytics: `8.4.128`
- Python: `3.12.13`
- `test_used=false`
- `test_split_used=false`

Generated binary와 metadata는 Git에서 제외된 아래 namespace에 생성한다.

```text
artifacts/deployment/yolo_segmentation/onnx/c5_1_yolo11n_seg_fp32_static_opset18/
├── model.onnx
└── metadata.json
```

## 4. C5-2 validation-only parity

PyTorch와 ONNX 모두 pinned Ultralytics prediction surface를 사용한다. 따라서 RGB/image loading,
letterbox, tensor conversion, NMS와 source-size mask normalization을 별도 구현하지 않는다.

C4-2C와 같은 prediction normalization을 유지한다.

- initial confidence: `0.001`
- diagnostic confidence: `0.25`
- NMS IoU: `0.7`
- max detections: `300`
- `retina_masks=false`
- mask threshold: `0.5`
- source-size mask resize: OpenCV nearest-neighbor
- association: positive mask overlap 기반 greedy maximum mask IoU

이는 export-equivalence를 관측하기 위한 operating point이며 C4 inference threshold를 다시 튜닝하는 과정이
아니다.

Dataset validator는 full manifest provenance를 확인하되 `content_splits={"val"}`만 materialize한다.
Derived final-test split은 C5 optimization/equivalence input으로 열지 않는다.

### 4.1 첫 characterization 실행

첫 parity run은 numeric tolerance를 정하기 전에 실행했다. 따라서 결과 상태는 의도적으로
`METRICS_COLLECTED_ACCEPTANCE_PENDING`이다.

실행 환경:

- Repository commit: `643ed9386a61bd2bf0c041f92a10b809b6d52c3e`
- ONNX Runtime: `1.29.0`
- ONNX Runtime provider: `CPUExecutionProvider`
- PyTorch device: CPU
- Ultralytics: `8.4.128`
- PyTorch: `2.13.0+cu130`
- Python: `3.12.13`

Characterization 결과:

- validation samples: `28`
- PyTorch predictions: `19`
- ONNX predictions: `19`
- matched instances: `19`
- unmatched PyTorch: `0`
- unmatched ONNX: `0`
- class agreement rate: `1.0`
- confidence absolute error mean: `3.780189313386616e-07`
- confidence absolute error max: `2.3543834686279297e-06`
- box IoU min / mean / max: `1.0 / 1.0 / 1.0`
- mask IoU min / mean / max: `1.0 / 1.0 / 1.0`
- structural gates: passed
- `test_used=false`
- `test_split_used=false`

Postprocessed binary mask SHA도 관측된 matched prediction에서 동일했지만 mask SHA equality는 portable
acceptance requirement로 사용하지 않는다. Runtime/provider가 바뀔 때 harmless boundary-pixel 차이를 허용할
수 있도록 mask IoU를 equivalence metric으로 사용한다.

Sample ID에 원본 MVTec naming의 `test` 문자열이 포함될 수 있으나 C5 split boundary는 derived manifest의
`derived_split`을 기준으로 한다. 이번 evidence의 evaluated split은 `val`이며 final-test split을 사용했다는
뜻이 아니다.

## 5. C5-2B ONNX FP32 acceptance policy v1

Characterization 결과를 관측한 뒤 numeric tolerance를 별도 repository config로 정의한다.
Export config는 수정하지 않는다.

Policy:
[`configs/deployment/yolo_onnx_fp32_parity_acceptance.yaml`](../../configs/deployment/yolo_onnx_fp32_parity_acceptance.yaml)

Policy v1은 exact identity를 고정한다.

- Policy ID: `c5_2_yolo_onnx_fp32_parity_v1`
- Source experiment: `c4_2c_yolo11n_seg_crop350_nomosaic_maskratio2_seed42`
- Frozen manifest SHA-256: `2a26b1bc03a1876f828e12a625c69c76af5e8c5713e3f64be699feffe2e8aa09`
- Source model SHA-256: `e3fd10cdd708d31421feacfc5d694cb638e0ea60672e08796391b33aecf67155`
- Export config SHA-256: `f1c2ef5045fdd89d964b2dc79c501580c9f55c2a1d38f38f13cf4794bafd0e85`
- ONNX SHA-256: `f916325bb126d174de9c1fdfc24802eec11c46014f723fbf3ba3b3c1755c1490`

Structural requirements:

- `split=val`
- `test_used=false`
- `test_split_used=false`
- characterization structural gates passed
- backend tensor outputs finite
- PyTorch/ONNX post-backend tensor name, dtype와 shape가 대응
- prediction count 동일
- unmatched PyTorch/ONNX prediction 각각 0
- class agreement rate `1.0`

Approved FP32 export-equivalence tolerances:

- confidence absolute error max `<= 1e-4`
- box IoU min `>= 0.999`
- mask IoU min `>= 0.999`

이 기준은 model-quality threshold가 아니다. mAP, precision, recall, GT matching 또는 C4 final-test metric을
acceptance gate에 넣지 않는다.

첫 characterization의 최대 confidence error `2.3543834686279297e-06`을 그대로 threshold로 사용하지 않고
FP32 backend의 harmless numerical variation을 위한 headroom을 둔다. 반대로 regression을 숨길 만큼 느슨한
기준도 두지 않는다.

첫 characterization은 이 policy보다 먼저 실행됐으므로 사전 선언 PASS로 재분류하지 않는다.
Policy를 committed clean state로 freeze한 뒤 exact same source/model/export identity에서 새 validation parity
evidence를 생성하고 policy v1을 적용해 `PARITY_ACCEPTED` 또는 `PARITY_REJECTED`를 판정한다.

### 5.1 Prospective verification 결과와 C5-2 closure

Acceptance policy v1은 characterization 이후 repository에 먼저 commit한 뒤 prospective verification에
사용했다.

- Policy repository commit: `1f48a047d14f032dad41f2cd4519399adf4d6bce`
- Policy ID: `c5_2_yolo_onnx_fp32_parity_v1`
- Policy SHA-256: `488ad32b71adbc6b7a0f0ef8e68823a0f991977ecd32b4ff6fc6b1aa73f7ebdb`
- Original ONNX export commit: `643ed9386a61bd2bf0c041f92a10b809b6d52c3e`
- ONNX SHA-256: `f916325bb126d174de9c1fdfc24802eec11c46014f723fbf3ba3b3c1755c1490`
- ONNX metadata SHA-256: `3286861db66cb4c4f886d2fd71f8f13b749b019bd0d57249f54a025d43b11fcd`
- Runtime: PyTorch FP32 CPU ↔ ONNX Runtime FP32 `CPUExecutionProvider`
- validation samples: `28`
- PyTorch / ONNX predictions: `19 / 19`
- matched instances: `19`
- unmatched PyTorch / ONNX: `0 / 0`
- class agreement rate: `1.0`
- confidence absolute error max: `2.384185791015625e-06`
- box IoU min: `1.0`
- mask IoU min: `1.0`
- acceptance checks: `17 / 17 PASS`
- final acceptance state: `PARITY_ACCEPTED`
- `test_used=false`
- `test_split_used=false`

새 Kaggle session에서 ONNX를 다시 export한 binary는 byte-level SHA가 기존 artifact와 달랐기 때문에
prospective evidence로 대체하지 않았다. 대신 C5-1에서 보존한 exact `model.onnx`와 `metadata.json`을 복원하고
각 SHA를 검증한 뒤 새 parity run을 수행했다. 따라서 acceptance policy가 고정한 exact artifact identity를
그대로 유지한다.

Prospective acceptance evidence는 Git 밖에서 보존한다.

- External evidence archive:
  `c5_2b_prospective_acceptance_1f48a04.zip`
- Archive SHA-256:
  `320ebc695f059b56c60011ab635eec00619aad9bfd44c534d1c25ddaea23e697`

이 결과로 C5-2 ONNX FP32 parity lifecycle은 `CLOSED`다. 이후 TensorRT FP16은 별도 backend와 precision
boundary이므로 ONNX FP32 acceptance threshold를 그대로 재사용하지 않고 별도의 characterization과
acceptance contract를 정의한다.

## 6. Acceptance evaluator

`ml/deployment/yolo_onnx_parity_acceptance.py`는 저장된 parity evidence에 policy를 적용하는 pure evaluation
surface다. Model inference, dataset image open, retraining 또는 threshold tuning을 수행하지 않는다.

CLI:

```bash
uv run --locked python -m pipelines.evaluate_yolo_onnx_parity_acceptance \
  --parity-evidence /secure/path/to/parity.json
```

Evaluator는 다음을 구분해 기록한다.

- characterization evidence repository provenance
- committed acceptance policy repository provenance
- policy SHA-256
- parity evidence SHA-256
- exact model/export/ONNX identity
- structural check별 observed / expected / pass-fail
- numeric check별 observed / threshold / pass-fail
- overall `PARITY_ACCEPTED` 또는 `PARITY_REJECTED`

Acceptance output은 original `parity.json`을 수정하지 않고 별도 ignored namespace에 기록한다.

```text
outputs/deployment/yolo_segmentation/onnx_parity_acceptance/
└── <parity_id>--<policy_id>/
    └── acceptance.json
```

Malformed evidence, non-finite data, test lifecycle violation과 dirty policy repository는 fail closed한다.
Wrong frozen/model/export/ONNX identity 또는 equivalence regression은 `PARITY_REJECTED` result로 남긴다.

## 7. Test seal과 다음 단계

C5-1 export는 dataset을 요구하지 않는다. C5-2 inference parity는 validation content만 사용한다. Acceptance
evaluator는 이미 저장된 JSON만 읽으며 dataset 또는 model path를 인자로 받지 않는다.

Raw/derived dataset, checkpoint, ONNX binary, parity JSON, acceptance JSON은 Git에 추가하지 않는다.
Repository에는 config, evaluator, tests와 documentation만 유지한다.

C5-2 prospective verification은 완료됐으며 `PARITY_ACCEPTED`로 종료했다.

현재 deployment optimization 상태는 다음과 같다.

1. C5-2 ONNX FP32 parity는 `PARITY_ACCEPTED / CLOSED` 상태를 유지한다.
2. Exact ONNX와 acceptance evidence는 Git 밖의 immutable evidence archive로 보존한다.
3. C5-3 TensorRT FP16은 frozen policy 기반 prospective verification에서 `PARITY_ACCEPTED`를 확인해 `CLOSED`다.
4. C5-3 prospective acceptance evidence와 exact engine identity는 Git 밖에서 immutable evidence로 보존한다.
5. C5-4 INT8은 calibration dataset, accuracy-loss budget, latency measurement boundary를 별도로 정의한 뒤에만 시작한다.

## 8. C5-3 TensorRT FP16 foundation

C5-3는 C5-2에서 승인된 exact ONNX binary를 다시 export하거나 model-quality tuning에 사용하지 않는다.
Source of truth는 다음 artifact identity다.

- ONNX SHA-256:
  `f916325bb126d174de9c1fdfc24802eec11c46014f723fbf3ba3b3c1755c1490`
- ONNX metadata SHA-256:
  `3286861db66cb4c4f886d2fd71f8f13b749b019bd0d57249f54a025d43b11fcd`
- ONNX export config SHA-256:
  `f1c2ef5045fdd89d964b2dc79c501580c9f55c2a1d38f38f13cf4794bafd0e85`
- Original ONNX export commit:
  `643ed9386a61bd2bf0c041f92a10b809b6d52c3e`

Repository config는
[`configs/export/yolo_segmentation_tensorrt_fp16.yaml`](../../configs/export/yolo_segmentation_tensorrt_fp16.yaml)
이다. 첫 build contract는 batch `1`, image size `640`, static shape, TensorRT FP16 builder flag,
workspace `4 GiB`, CUDA device `0`으로 제한한다. Dynamic shape와 INT8은 포함하지 않는다.

Engine은 accepted ONNX를 TensorRT Python API로 parse한 뒤 static FP16 serialized engine으로 build한다.
`model.engine` 앞에는 pinned Ultralytics TensorRT backend가 frozen class names와 segmentation task를 복원할 수
있도록 length-prefixed JSON metadata header를 기록한다. Engine contract 검사에서는 이 header를 검증한 뒤
raw TensorRT payload만 deserialize한다. Generated artifact와 sidecar metadata는 Git에서 제외된 namespace에
기록한다.

```text
artifacts/deployment/yolo_segmentation/tensorrt/
└── c5_3a_yolo11n_seg_tensorrt_fp16_static/
    ├── model.engine
    └── metadata.json
```

Metadata에는 source ONNX/model/config hash, engine SHA/size, static I/O tensor contract, TensorRT/CUDA/PyTorch
version, GPU name/compute capability/memory, clean Git provenance와 test seal을 기록한다. TensorRT engine은
GPU architecture와 TensorRT/CUDA environment에 종속될 수 있으므로 이후 prospective verification은
characterization에서 고정한 runtime identity를 별도로 검증해야 한다.

C5-3B characterization은 PyTorch FP32 GPU reference와 TensorRT FP16 engine을 같은 CUDA device에서
validation split만 사용해 비교한다. C4-2C prediction normalization과 mask association은 유지하며 다음을
수집한다.

- prediction count와 unmatched count
- class agreement
- confidence absolute error
- box IoU
- mask IoU
- finite result tensor evidence
- PyTorch/TensorRT end-to-end single-image latency
- engine SHA와 runtime/GPU provenance

Latency는 validation 첫 sample을 사용해 backend별 warmup `10`회 후 `50`회를 측정한다. 측정 scope는
Ultralytics의 image load/preprocess/inference/postprocess를 포함한 single-image end-to-end path이며 독립적인
kernel-only benchmark로 해석하지 않는다.

이 단계에서는 TensorRT FP16 numeric acceptance threshold를 정의하지 않는다. Evidence state는
`TENSORRT_FP16_METRICS_COLLECTED_ACCEPTANCE_PENDING`이며 실제 GPU characterization 결과를 관측한 뒤
별도 policy를 commit하기 전에는 `PASS`를 선언하지 않는다.
## 9. C5-3B characterization과 C5-3C TensorRT FP16 acceptance policy v1

C5-3A/B GPU characterization은 clean detached repository commit
`5604219d07bf384f46f2827f4da999781832e183`에서 실행했다. C5-2에서 승인된 exact ONNX를 다시 export하지
않고 보존본을 복원해 TensorRT FP16 engine을 build했다.

### 9.1 C5-3A/B 실행 결과

- GPU: `Tesla T4`, compute capability `7.5`
- CUDA runtime: `13.0`
- TensorRT: `10.13.3.9.post1`
- PyTorch: `2.13.0+cu130`
- Ultralytics: `8.4.128`
- Source ONNX SHA-256:
  `f916325bb126d174de9c1fdfc24802eec11c46014f723fbf3ba3b3c1755c1490`
- TensorRT config SHA-256:
  `edc135932e9367f67b9179dbbd47b01da6fa07db878a7f8af73b491718b517c9`
- Engine SHA-256:
  `9bbbe5297e6cc55bcea877a79f45485ee7e1e5e6a831ad5276aedc8e3d904037`
- TensorRT metadata SHA-256:
  `d400c7fe1a09c9c53baf63b0727c5cf5f84602ca26a85d3996e2296d480e99da`
- Characterization parity SHA-256:
  `0b6eba9ca3eee24b5e3fb5f1ce09227ffced26d80477d56c355648c24235f9bf`
- External evidence archive SHA-256:
  `e6266f46b3c4aad6605873fe8a950c11abd1ea642d8039673210b587dd419bcb`
- validation samples: `28`
- PyTorch / TensorRT predictions: `19 / 19`
- matched instances: `19`
- unmatched PyTorch / TensorRT: `0 / 0`
- class agreement rate: `1.0`
- confidence absolute error max: `0.005336761474609375`
- box IoU min / mean: `0.9841219602257741 / 0.998657164643238`
- mask IoU min / mean: `0.9972451790633609 / 0.9991235966986481`
- PyTorch mean latency: `32.93056183998942 ms`
- TensorRT mean latency: `27.092740620000768 ms`
- speedup ratio: `1.2154754774302523`
- `test_used=false`
- `test_split_used=false`

이 결과의 lifecycle state는
`TENSORRT_FP16_METRICS_COLLECTED_ACCEPTANCE_PENDING`이며 characterization 자체를 PASS run으로
소급하지 않는다.

### 9.2 C5-3C policy v1

Policy:
[`configs/deployment/yolo_tensorrt_fp16_parity_acceptance.yaml`](../../configs/deployment/yolo_tensorrt_fp16_parity_acceptance.yaml)

Policy는 characterization 관측값을 그대로 threshold로 복사하지 않고 다음 headroom을 둔다.

- confidence absolute error max `<= 0.01`
- box IoU min `>= 0.98`
- mask IoU min `>= 0.995`
- TensorRT mean latency `<` PyTorch mean latency
- end-to-end speedup ratio `>= 1.05`

Structural gate는 validation `28` samples, no-test seal, prediction count match, zero unmatched,
class agreement `1.0`, nested tensor finite evidence를 요구한다. 또한 hardware-specific TensorRT engine을
다른 runtime에서 조용히 재사용하지 않도록 TensorRT/CUDA/GPU/compute-capability/PyTorch/Ultralytics
identity를 characterization 환경에 고정한다.

가장 중요한 prospective boundary는 parity evidence의 clean Git commit이 acceptance policy를 실행하는
clean Git commit과 같아야 한다는 점이다. 따라서 C5-3B에서 policy보다 먼저 생성한 characterization JSON은
policy v1의 prospective PASS evidence로 재사용할 수 없다.

Prospective verification에서는 engine을 rebuild하지 않는다. External C5-3B evidence archive에서 exact
`model.engine`과 sidecar metadata를 복원하고 engine SHA
`9bbbe5297e6cc55bcea877a79f45485ee7e1e5e6a831ad5276aedc8e3d904037`을 검증한 뒤, policy가 이미 commit된
clean repository state에서 validation parity를 새로 실행한다. 그 새 parity JSON에만 C5-3C acceptance
evaluator를 적용한다.

Acceptance evaluator는 saved parity JSON과 committed policy만 읽으며 dataset/model inference를 다시
수행하지 않는다. 최종 state는 `TENSORRT_FP16_PARITY_ACCEPTED` 또는
`TENSORRT_FP16_PARITY_REJECTED`다. Prospective verification이 끝나기 전까지 C5-3는 `CLOSED`로
표현하지 않는다.

### 9.3 C5-3C prospective verification 결과와 C5-3 closure

TensorRT FP16 acceptance policy v1은 characterization 이후 repository에 먼저 commit한 뒤
prospective verification에 사용했다.

Prospective run은 policy commit의 clean detached repository state에서 실행했다. C5-3B에서 보존한
exact TensorRT engine과 ONNX artifact를 복원했으며 engine을 다시 build하지 않았다.
Dataset은 validation split만 사용했고 final-test split은 열지 않았다.

- Policy repository commit:
  `880b2cba33013320adf966a4097b556309688864`
- Policy SHA-256:
  `4f8f81a70417e380062358a9f3888d4fe0fa236fdfbc7b04da2616356833bfd9`
- TensorRT engine SHA-256:
  `9bbbe5297e6cc55bcea877a79f45485ee7e1e5e6a831ad5276aedc8e3d904037`
- Runtime: PyTorch FP32 GPU ↔ TensorRT FP16
- GPU: `Tesla T4`, compute capability `7.5`
- Validation samples: `28`
- PyTorch / TensorRT predictions: `19 / 19`
- Matched instances: `19`
- Unmatched PyTorch / TensorRT: `0 / 0`
- Class agreement rate: `1.0`
- Confidence absolute error max: `0.005336761474609375`
- Box IoU min / mean:
  `0.9841219602257741 / 0.998657164643238`
- Mask IoU min / mean:
  `0.9972451790633609 / 0.9991235966986481`
- PyTorch mean latency: `31.130911420002576 ms`
- TensorRT FP16 mean latency: `25.844023020001714 ms`
- Speedup ratio: `1.2045690949860681`
- Acceptance checks: `34 / 34 PASS`
- Final acceptance state: `TENSORRT_FP16_PARITY_ACCEPTED`
- `engine_rebuilt=false`
- `test_used=false`
- `test_split_used=false`

Prospective evidence identities:

- Parity SHA-256:
  `d14400bbb1b71036ee3c87e307a9b830d44a1824089b08fbf7a05eb333d8549c`
- Acceptance JSON SHA-256:
  `b6499f261a5c726a7017b39f74f917a2263297bc9bf676f429fa9d05199ff651`
- External evidence archive:
  `c5_3c_tensorrt_fp16_prospective_acceptance_evidence.zip`
- External evidence archive SHA-256:
  `5ae1a16fcbecdb73634ba0dc19876232bec67178defa5fee0720a39c54a11de6`

이 결과로 C5-3 TensorRT FP16 parity lifecycle은 `CLOSED`다.

C5-4 INT8 / Quantization은 C5-3 tolerance를 재사용하지 않는다. C5-4A에서 calibration dataset,
quantization toolchain과 latency measurement boundary를 먼저 고정하고, 실제 INT8 numeric tolerance는
새 validation characterization을 관측한 뒤 별도 policy commit으로만 정의한다.

## 10. C5-4A INT8 explicit-Q/DQ PTQ foundation

C5-4는 accepted C5-1 FP32 ONNX와 closed C5-3 FP16 evidence를 source/baseline으로 사용하지만,
FP16 engine을 INT8 source로 변환하지 않는다. INT8 source graph는 exact accepted FP32 ONNX다.

TensorRT 10.x의 legacy implicit INT8 calibrator API는 deprecated 상태이고 TensorRT 11.x에서는 제거되므로,
C5-4는 NVIDIA ModelOpt로 ONNX graph에 explicit Quantize/Dequantize(Q/DQ) node를 삽입한 뒤
TensorRT engine을 build하는 경로를 사용한다.

C5-4A contract:

- Exact source ONNX SHA-256:
  `f916325bb126d174de9c1fdfc24802eec11c46014f723fbf3ba3b3c1755c1490`
- Quantizer: `nvidia-modelopt==0.46.0`
- Quantization: INT8 explicit Q/DQ PTQ
- Calibration method: `entropy`
- High-precision fallback dtype: `fp16`
- Static input: batch `1`, `640×640`
- Calibration source: dataset manifest의 `train` 84장 전체
- Calibration order: manifest sample ID ascending
- Validation calibration 사용: `false`
- Final-test calibration 사용: `false`
- INT8 quality characterization: `val` 28장만 사용
- Final-test characterization 사용: `false`
- Numeric acceptance threshold: 아직 정의하지 않음
- Benchmark: C5-3과 동일한 warmup `10`, measured `50`,
  `ultralytics_end_to_end_single_image`

Calibration은 activation range를 정하는 quantization 과정이므로 model-quality validation과 분리한다.
`val` 28장은 INT8 결과의 quality/equivalence characterization에만 사용하며 calibration data로 재사용하지 않는다.
Final-test는 C5-4 전체에서 계속 sealed 상태를 유지한다.

C5-4C characterization에서는 최소 다음 세 runtime을 같은 Tesla T4 boundary에서 비교한다.

1. PyTorch FP32 GPU reference
2. Accepted TensorRT FP16 baseline
3. Candidate TensorRT INT8

관측할 evidence는 prediction count, unmatched count, class agreement, confidence error, box/mask IoU와
end-to-end latency다. Characterization 결과를 보기 전에는 INT8 numeric PASS를 선언하지 않는다.
관측 이후 별도 C5-4D policy commit에서 tolerance와 INT8 채택에 필요한 latency benefit을 고정하고,
새 clean repository state의 C5-4E prospective verification으로 최종 `ACCEPTED` 또는 `REJECTED`를 판정한다.
### 10.1 C5-4B1 ModelOpt INT8 Q/DQ ONNX foundation

C5-4A contract commit 이후 C5-4B는 quantized ONNX 생성과 TensorRT engine build를 분리한다.

C5-4B1은 exact accepted C5-1 FP32 ONNX를 source로 사용하고 frozen derived dataset의 `train`
84장만 calibration에 사용해 NVIDIA ModelOpt explicit Q/DQ ONNX를 생성한다. `val` 28장은
calibration에 사용하지 않으며 C5-4C characterization까지 열지 않는다. final-test 역시 계속 sealed다.

Implementation boundary:

- Source: exact accepted FP32 ONNX
- Quantizer: `nvidia-modelopt==0.46.0`
- API: `modelopt.onnx.quantization.quantize`
- Quantization mode: `int8`
- Calibration method: `entropy`
- Calibration execution provider: `cpu`
- Calibration reader: one image per batch, manifest `sample_id` ascending
- Preprocess: static `640×640` letterbox, BGR→RGB, NCHW, FP32, `/255`
- High precision dtype: `fp16`
- ONNX simplify: `false`
- Required candidate structure: at least one `QuantizeLinear` and one `DequantizeLinear`
- Source/candidate external input and output names/shapes must remain identical
- Validation used: `false`
- Final-test used: `false`

C5-4B1 output은 ignored artifact namespace에 quantized ONNX와 deterministic provenance metadata로
보존한다. Metadata에는 source/config/dataset SHA-256, calibration sample ID digest, Q/DQ node count,
ModelOpt version과 clean repository commit을 기록한다.

C5-4B1에서 생성된 exact Q/DQ ONNX가 별도 evidence로 고정된 뒤에만 C5-4B2 TensorRT INT8
engine build의 source로 사용한다.
