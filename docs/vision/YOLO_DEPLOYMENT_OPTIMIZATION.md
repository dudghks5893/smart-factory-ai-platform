# YOLO Deployment Optimization

## 1. 상태와 범위

C5는 C4에서 확정한 YOLO11n-seg candidate의 배포 형식과 backend equivalence를 검증하는 단계다. Model
quality 실험이나 candidate selection을 다시 열지 않는다.

| Stage | State |
| --- | --- |
| C5-1 ONNX export foundation | `IMPLEMENTED / NOT YET OFFICIALLY EXECUTED` |
| C5-2 PyTorch ↔ ONNX Runtime parity | `IMPLEMENTED / NOT YET OFFICIALLY EXECUTED` |
| C5-3+ TensorRT / Quantization | `NOT STARTED` |

이번 구현에서는 실제 model package, validation dataset 또는 generated ONNX를 실행하지 않았다. Unit test는
synthetic fixtures만 사용한다. TensorRT, FP16, INT8, quantization은 범위 밖이다.

## 2. Immutable C4 source

ONNX export source는 C4-3에서 validation-only로 freeze된 다음 identity 하나뿐이다.

| Field | Value |
| --- | --- |
| Experiment | `c4_2c_yolo11n_seg_crop350_nomosaic_maskratio2_seed42` |
| Frozen manifest | `configs/model/yolo_segmentation_final_candidate.json` |
| Frozen manifest SHA-256 | `2a26b1bc03a1876f828e12a625c69c76af5e8c5713e3f64be699feffe2e8aa09` |
| Model SHA-256 | `e3fd10cdd708d31421feacfc5d694cb638e0ea60672e08796391b33aecf67155` |
| Official package SHA-256 | `81c721ab6d34e5563e9f8907fe4c9914d50e48ef35aacfabb6f4ca745420cd76` |
| Model / task | `yolo11n-seg` / `segment` |

Export pipeline은 frozen manifest bytes, Official package 전체 hash와 package evidence를 교차 검증한다. Fixed
package entry의 `model.pt`와 `metadata.json`만 임시 공간에 복원하고 model hash를 export 전후에 재검증한다.
다른 checkpoint를 조용히 받아들이지 않는다. C4-4 final-test 결과는 report-only evidence이며 export parameter,
parity threshold 또는 candidate를 선택하는 데 사용하지 않는다.

## 3. C5-1 ONNX export contract

Repository config는 [`configs/export/yolo_segmentation_onnx.yaml`](../../configs/export/yolo_segmentation_onnx.yaml)이다.

| Parameter | Value | Rationale |
| --- | --- | --- |
| format / task | ONNX / segmentation | Segmentation output을 보존한다. |
| precision | FP32 | 첫 backend conversion에서 precision optimization을 분리한다. |
| batch / input | 1 / static `1×3×640×640` | Frozen C4-2C image size와 online request contract를 유지한다. |
| opset | 18 | Pinned Ultralytics 8.4.128의 ONNX Runtime compatibility logic이 선택 가능한 범위에 명시적으로 고정한다. |
| dynamic / simplify | false / false | 첫 evidence에서 graph rewrite와 shape variation을 배제한다. |
| graph NMS | false | 기존 Ultralytics prediction/postprocessing policy를 재사용한다. |
| device | CPU | Portable graph export이며 CUDA/TensorRT conversion이 아니다. |

Exporter는 pinned Ultralytics export surface를 얇게 호출하고 repository가 model graph를 재구현하지 않는다.
생성 graph는 ONNX checker를 통과해야 하며 `images` input, `output0`/`output1` segmentation outputs, static positive
shape와 opset을 검사한다. `metadata.json`에는 source hashes, exact export config, ONNX SHA/size, graph I/O,
Python/PyTorch/Ultralytics/ONNX versions, clean Git HEAD와 dirty flag를 기록한다. Dirty working tree에서는 Official
export를 거부한다.

Generated files는 Git에서 제외된 아래 namespace에만 쓴다.

```text
artifacts/deployment/yolo_segmentation/onnx/<export_id>/
├── model.onnx
└── metadata.json
```

Clean committed state와 exact Official package를 준비한 뒤 실행할 reproduction path는 다음과 같다. 현재 문서화된
명령은 아직 Official execution evidence가 아니다.

```bash
uv run --locked python -m pipelines.export_yolo_onnx \
  --official-package /secure/path/to/official-c4-2c-package.zip \
  --created-at 2026-09-02T12:00:00+09:00
```

## 4. C5-2 validation-only parity policy

PyTorch와 ONNX 모두 pinned Ultralytics prediction surface를 사용한다. 따라서 RGB/image loading, letterbox,
tensor conversion, NMS, source-size mask normalization을 별도 구현하지 않는다. C4-2C와 동일하게 initial
confidence `0.001`, NMS IoU `0.7`, max detections `300`, `retina_masks=false`, mask threshold `0.5`, OpenCV
nearest resize를 사용한다. Postprocessed parity instance는 C4 diagnostic confidence `0.25`를 적용한 동일
prediction pool에서 비교한다. 이는 export-equivalence 관측용 고정 operating point이며 C4 threshold를
변경하거나 다시 튜닝하지 않는다.

두 backend prediction은 class와 무관하게 positive mask overlap의 IoU가 큰 순서로 one-to-one matching한다.
Class를 matching 조건으로 숨기지 않고 각 pair의 `class_agreement`로 별도 기록한다. 다음 evidence를 수집한다.

- Frozen manifest, source model, ONNX와 export config SHA-256
- `split=val`, sample count, `test_used=false`, `test_split_used=false`
- Backend result tensor name/dtype/shape와 finite 여부
- Prediction count, class ID, confidence, box coordinates, mask shape/foreground pixels/SHA-256
- Matched count, class agreement, confidence absolute error, box IoU, mask IoU
- Unmatched PyTorch/ONNX prediction count

### 4.1 Predeclared acceptance contract

Official real-data run 전에 승인되지 않은 수치 허용치를 만들지 않는다. 현재 contract는 다음 두 층으로
분리된다.

1. Structural gates는 필수다. NaN/Inf, invalid tensor/shape, invalid class, malformed mask/box, artifact/hash/backend
   mismatch, non-validation row는 즉시 실패한다.
2. Numerical evidence는 count/class agreement/confidence error/box IoU/mask IoU 분포를 수집한다. 실제 output
   representation을 관측하고 equivalence tolerance를 별도 승인하기 전에는 `ACCEPT`를 선언하지 않으며 상태는
   `METRICS_COLLECTED_ACCEPTANCE_PENDING`이다.

이 정책은 bit-identical output을 가정하지 않으면서도 임의의 느슨한 숫자로 regression을 통과시키지 않는다.
수치 gate가 향후 승인되더라도 model-quality gate나 C4 inference threshold가 아니라 backend export equivalence
gate여야 한다.

Parity evidence는 Git에서 제외된 namespace에 저장한다.

```text
outputs/deployment/yolo_segmentation/onnx_parity/<parity_id>/parity.json
```

Official execution reproduction path는 다음과 같다. Dataset validator는 full manifest provenance를 확인하되
`content_splits={"val"}`만 typed materialization/content validation한다. Excluded test row는 lexical CSV split
gating 이후 열거나 평가하지 않는다.

```bash
uv run --locked python -m pipelines.validate_yolo_onnx_parity \
  --official-package /secure/path/to/official-c4-2c-package.zip \
  --onnx-artifact artifacts/deployment/yolo_segmentation/onnx/c5_1_yolo11n_seg_fp32_static_opset18 \
  --dataset-root /secure/path/to/supervised-derived-dataset \
  --parity-id c5_2_yolo11n_seg_fp32_static_opset18_val \
  --created-at 2026-09-02T12:30:00+09:00
```

## 5. Test seal과 다음 단계

C5-1 export는 dataset을 요구하지 않는다. C5-2는 validation content만 사용한다. Derived final-test split은 C4-4
report 이후에도 optimization/tuning input으로 재사용하지 않는다. No raw dataset, checkpoint, ONNX binary 또는
runtime evidence JSON을 Git에 추가하지 않는다.

C5-1/C5-2의 Official execution과 numeric tolerance approval이 끝나기 전 C5를 완료로 표시하지 않는다. 이후
C5-3 TensorRT, FP16/INT8 또는 quantization은 별도 설계·evidence 단계에서만 시작한다.
