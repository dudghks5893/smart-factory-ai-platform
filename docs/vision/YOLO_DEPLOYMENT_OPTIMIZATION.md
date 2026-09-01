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
| C5-3+ TensorRT / Quantization | `NOT STARTED` |

C5-1과 첫 C5-2 characterization은 clean detached Git commit
`643ed9386a61bd2bf0c041f92a10b809b6d52c3e`에서 Kaggle로 실행했다. 첫 parity run은
수치 허용치를 정하기 위한 characterization이며, 사전에 선언된 numeric gate를 통과한 run으로 소급해
표현하지 않는다.

이후 acceptance policy v1을 commit
`1f48a047d14f032dad41f2cd4519399adf4d6bce`에서 고정한 뒤, 기존 C5-1 exact ONNX artifact를 복원해
validation parity를 새로 실행했다. Frozen policy의 17개 acceptance check를 모두 통과해
`PARITY_ACCEPTED`가 확인됐으며 C5-2는 `CLOSED` 상태다.

TensorRT, FP16, INT8, quantization은 아직 시작하지 않았다.

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

다음 순서는 다음과 같다.

1. C5-2 ONNX FP32 parity 상태를 `CLOSED`로 유지한다.
2. Exact ONNX와 acceptance evidence는 Git 밖의 immutable evidence archive로 보존한다.
3. C5-3 TensorRT FP16 export/runtime foundation을 별도 contract로 설계한다.
4. TensorRT FP16은 먼저 characterization을 수행한 뒤 별도의 acceptance tolerance를 고정한다.
5. INT8은 calibration dataset과 accuracy-loss budget이 별도로 승인될 때만 후속 단계에서 검토한다.
