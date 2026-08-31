# YOLO Fast Research Log

## 1. 연구 목적과 평가 기준

이 연구의 목적은 MVTec AD `metal_nut` 기반 YOLO11 segmentation에서 반복적으로 발생한 **Small defect 탐지/분할 실패**의 원인을 분해하고, 한 번에 하나의 가설을 검증하는 controlled experiment를 통해 개선 가능한 방향을 찾는 것이다.

연구 중에는 validation split만 사용했고 test split은 계속 봉인했다.

### 고정 환경
- Python: `3.12.13`
- PyTorch: `2.10.0+cu128`
- Ultralytics: `8.4.128`
- GPU: `Tesla T4`

### 고정 strict diagnostic
- confidence threshold: `0.25`
- class-aware greedy mask matching
- strict mask IoU threshold: `0.5`
- validation GT instances: `23`
- validation Small GT instances: `8`
- good-negative validation images: `14`

### Fast Research reference
실험 비교 기준은 clean Kaggle 환경에서 다시 재현한 아래 reference를 사용했다.

`ref_region_calibration_component_aware_x2_seed42`

조건:
- `yolo11n-seg.pt`
- `component_aware x2`
- train entries: `103`
- positive exposure: `61`
- negative exposure: `42`
- `imgsz=640`
- `mosaic=1.0`
- `mask_ratio=4`
- `scale=0.5`
- `close_mosaic=10`
- seed `42`

결과:
- Mask Precision: `0.778853`
- Mask Recall: `0.674182`
- Mask mAP50-95: `0.414314`
- TP / FP / FN: `16 / 6 / 7`
- Strict Precision: `0.727273`
- Strict Recall: `0.695652`
- F1: `0.711111`
- Small Recall: `0.250` (`2/8`)
- Multi Recall: `0.500`
- Good-negative FP rate: `0.000`

Region Coverage 기준값:
- Component Coverage@50: `0.739130`
- Small Coverage@50: `0.375` (`3/8`)
- Class-aware Union IoU: `0.698222`
- GT Coverage: `0.788857`
- Prediction Precision: `0.858699`
- Covered>=50% but strict fail: `1`

이 reference는 Fast Research 내부 비교 기준이며, official C4-2B checkpoint와 동일한 artifact라고 표현하지 않는다.

---

## 2. 전체 실험 요약

| 실험 | 핵심 변경 | Small R | Mask mAP50-95 | Strict Recall | F1 | Multi R | Neg FP | 판단 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Reference | component-aware x2 | 0.250 | 0.414 | 0.696 | 0.711 | 0.500 | 0.000 | 기준 |
| R2 | small-aware x3 | 0.250 | 0.357 | 0.522 | 0.500 | 0.429 | 0.000 | DROP |
| R3 | component x2 + scale 0.25 | 0.250 | 0.371 | 0.522 | 0.600 | 0.429 | 0.000 | DROP |
| R4 | component x2 + mosaic 0.50 | **0.375** | 0.335 | 0.522 | 0.381 | 0.500 | 0.143 | DROP |
| R5 | component x2 + close_mosaic 30 | 0.250 | 0.414 | 0.696 | 0.711 | 0.500 | 0.000 | DROP |
| R6 | component x2 + mosaic 0.75 | **0.375** | 0.371 | 0.609 | 0.560 | 0.500 | 0.000 | DROP |
| R7 | small-aware x3 + mosaic 0.75 | 0.250 | 0.413 | 0.609 | 0.667 | 0.429 | 0.000 | DROP |
| R8 | interrupted/reused invalid checkpoint | 0.000 | 0.076 | 0.174 | 0.258 | 0.214 | 0.000 | INVALID |
| R8b | component x2 + mask_ratio 2 | 0.250 | **0.444** | 0.609 | 0.596 | 0.429 | 0.000 | DROP |
| R9 | mosaic 0.75 + mask_ratio 2 | 0.125 | 0.390 | 0.522 | 0.490 | 0.429 | 0.000 | DROP |
| R10 | component x2 + mosaic 0.875 | 0.250 | 0.383 | 0.609 | 0.560 | 0.500 | 0.000 | DROP |
| R11 | R6 checkpoint audit | best Small 0.375 | gate survivor 없음 | - | - | - | 0.000 | NO_SURVIVOR |
| R12 | component x2 + crop350 + mosaic 1.0 | 0.250 | 0.369 | 0.565 | 0.448 | 0.500 | 0.000 | DROP |
| R13 | component x2 + crop350 + mosaic 0.0 | 0.250 | **0.426** | 0.609 | 0.571 | 0.429 | 0.000 | DROP / partial positive |
| R14 | component x2 + no crop + mosaic 0.0 | 0.250 | 0.278 | 0.565 | 0.520 | **0.643** | 0.000 | DROP / ablation |
| R15 | R13 + overlap_mask=False | 0.250 | **0.426** | 0.609 | 0.571 | 0.429 | 0.000 | NO_EFFECT_ABLATION |
| R16 | R13 + YOLO11s | 0.125 | 0.343 | 0.609 | 0.583 | 0.500 | 0.000 | DROP |
| R17 | crop350 + mosaic 0.0 + mask_ratio 2 | **0.375** | **0.448** | 0.652 | **0.682** | **0.571** | 0.000 | **KEEP_FOR_CONFIRMATION** |

---

## 3. 실험 흐름과 가설 변화

### R2 — Small-aware x3 oversampling

**가설**  
Small defect를 더 자주 노출시키면 작은 불량에 대한 recall이 올라갈 수 있다.

**변경**
- Small-aware 14개 sample의 학습 노출을 x3로 증가
- train entries: `112`
- 나머지는 reference 조건 유지

**결과**
- Mask mAP50-95: `0.356906`
- Strict Recall: `0.521739`
- F1: `0.500000`
- Small Recall: `0.250`
- Multi Recall: `0.428571`
- Good-negative FP: `0`

**해석**  
Small sample의 단순 반복 노출만으로는 Small Recall이 전혀 개선되지 않았다. 오히려 전체 recall과 multi-instance 성능이 악화됐다.

**다음 가설로 연결**  
단순 oversampling보다 **작은 결함의 기하학적 표현을 학습 중 얼마나 보존하는지**가 더 중요할 수 있다고 보고 augmentation 방향을 조사했다.

---

### R3 — scale 0.50 → 0.25

**가설**  
강한 scale augmentation이 작은 결함을 지나치게 축소하거나 왜곡할 수 있으므로 scale 범위를 줄이면 Small Recall이 개선될 수 있다.

**변경**
- `scale=0.5 -> 0.25`
- component-aware x2 유지

**결과**
- Mask mAP50-95: `0.371032`
- Strict Recall: `0.521739`
- F1: `0.600000`
- Small Recall: `0.250`
- Multi Recall: `0.428571`
- Good-negative FP: `0`

**해석**  
Small Recall은 그대로였고 전체 recall도 악화됐다. scale 감소만으로는 Small defect 문제를 해결하지 못했다.

**결론**  
scale 축소 방향은 종료했다.

---

### R4 — Mosaic 1.0 → 0.50

**가설**  
Mosaic이 작은 결함을 더 작게 만들거나 주변 문맥을 과도하게 섞으면서 Small defect 학습을 방해할 수 있다.

**변경**
- `mosaic=1.0 -> 0.5`
- component-aware x2 유지

**결과**
- Mask mAP50-95: `0.335064`
- Strict Recall: `0.521739`
- F1: `0.380952`
- **Small Recall: `0.375` (`3/8`)**
- Multi Recall: `0.500`
- Good-negative FP rate: `0.142857` (`2/14`)

**해석**  
처음으로 Small Recall이 `2/8 -> 3/8`로 증가했다. 따라서 Mosaic 강도와 Small defect 감도 사이에 실제 신호가 있음을 확인했다.

하지만 전체 segmentation 품질과 F1이 크게 하락했고, 정상 이미지에서도 FP가 발생했다. Small 개선만으로 받아들일 수 없는 결과였다.

**다음 가설로 연결**  
Mosaic을 완전히 강하게 유지하거나 크게 줄이는 양극단보다, **중간 수준으로 줄여 Small 신호를 보존하면서 품질 붕괴를 완화할 수 있는지** 확인하기로 했다.

---

### R5 — close_mosaic 10 → 30

**가설**  
전체 학습 동안 Mosaic 강도는 유지하되, 마지막 Mosaic-free 구간을 길게 두면 후반부에 더 자연스러운 원본 문맥으로 mask를 정제할 수 있다.

**변경**
- `close_mosaic=10 -> 30`

**결과**
- Mask mAP50-95: `0.414314`
- Strict Recall: `0.695652`
- F1: `0.711111`
- Small Recall: `0.250`
- Multi Recall: `0.500`
- Good-negative FP: `0`

**해석**  
최종 결과는 reference와 동일했고 Small Recall도 개선되지 않았다. Mosaic-free tail을 길게 만드는 것만으로는 Small defect 문제를 해결하지 못했다.

**결론**  
`close_mosaic` 조정만으로 해결하는 방향은 종료했다.

---

### R6 — Mosaic 0.75

**가설**  
R4의 `mosaic=0.5`가 보여준 Small Recall 개선 신호를 유지하면서, 너무 낮은 Mosaic로 인한 품질 붕괴와 normal FP를 줄일 수 있다.

**변경**
- `mosaic=0.75`

**결과**
- Mask mAP50-95: `0.371134`
- Strict Recall: `0.608696`
- F1: `0.560000`
- **Small Recall: `0.375`**
- Multi Recall: `0.500`
- Good-negative FP: `0`

**해석**  
R4와 동일하게 Small Recall `0.375`를 유지하면서 정상 FP는 제거했다. 다만 overall mAP와 strict metrics는 reference보다 여전히 낮았다.

**인사이트**  
Mosaic 감소가 Small defect에 영향을 준다는 방향성은 반복적으로 확인됐지만, 단순 scalar 조정만으로 전체 품질과 Small 개선을 동시에 만족시키기는 어려웠다.

---

### R7 — Small-aware x3 + Mosaic 0.75

**가설**  
R6에서 얻은 reduced-Mosaic 신호와 Small oversampling을 결합하면 Small Recall을 더 높일 수 있다.

**변경**
- small-aware x3
- `mosaic=0.75`

**결과**
- Mask mAP50-95: `0.412650`
- Strict Recall: `0.608696`
- F1: `0.666667`
- Small Recall: `0.250`
- Multi Recall: `0.428571`
- Good-negative FP: `0`

**해석**  
Small Recall이 다시 reference 수준으로 떨어졌고 multi-instance 성능도 악화됐다. oversampling은 Mosaic 감소와 결합해도 반복적으로 Small 문제를 해결하지 못했다.

**결론**  
Small oversampling branch를 종료했다.

---

### R8 — INVALID run

**발생한 문제**  
잘못된 sampling 설정으로 학습을 시작한 뒤 중단했고, 동일 experiment ID에서 기존 `best.pt`를 재사용하면서 현재 설정과 다른 checkpoint가 평가됐다.

**결론**
- R8 결과는 품질 비교에서 제외
- `INVALID`로 영구 기록

**연구 과정에서 얻은 교훈**  
실험 ID만으로 checkpoint를 재사용하면 안 되며, 재사용 시 실제 configuration fingerprint와 completed metadata를 함께 확인해야 한다.

---

### R8b — mask_ratio 4 → 2

**가설**  
Small defect는 mask downsampling 해상도에서 세부 형상이 손실될 수 있으므로 더 높은 mask supervision resolution이 segmentation 품질을 개선할 수 있다.

**변경**
- `mask_ratio=4 -> 2`
- component-aware x2
- Mosaic 기본값 유지

**결과**
- **Mask mAP50-95: `0.443992`**
- Strict Recall: `0.608696`
- F1: `0.595745`
- Small Recall: `0.250`
- Multi Recall: `0.428571`
- Good-negative FP: `0`

**해석**  
Mask mAP50-95는 reference보다 명확하게 향상됐지만 strict Small Recall은 전혀 움직이지 않았다.

**인사이트**  
`mask_ratio=2`는 **segmentation representation 품질**에는 유효한 신호지만, 작은 결함을 실제로 더 많이 보게 만드는 신호는 아니었다.

---

### R9 — Mosaic 0.75 + mask_ratio 2

**가설**  
R6의 Small Recall 신호와 R8b의 mask-quality 신호를 결합하면 두 장점을 동시에 얻을 수 있다.

**변경**
- `mosaic=0.75`
- `mask_ratio=2`

**결과**
- Mask mAP50-95: `0.389780`
- Strict Recall: `0.521739`
- F1: `0.489796`
- Small Recall: `0.125`
- Multi Recall: `0.428571`
- Good-negative FP: `0`

**해석**  
두 개의 개별적인 긍정 신호가 단순 결합에서는 오히려 강한 부정적 interaction을 보였다.

**결론**  
`mosaic=0.75 + mask_ratio=2` 조합은 종료했다. 다만 `mask_ratio=2` 자체의 mask-quality 신호는 버리지 않고, 이후 다른 문맥에서 다시 검증할 가치가 있다고 남겼다.

---

### R10 — Mosaic 0.875

**가설**  
Mosaic `0.75`와 `1.0` 사이에서 Small Recall과 overall quality의 절충점이 존재할 수 있다.

**변경**
- `mosaic=0.875`

**결과**
- Mask mAP50-95: `0.383110`
- Strict Recall: `0.608696`
- F1: `0.560000`
- Small Recall: `0.250`
- Multi Recall: `0.500`
- Good-negative FP: `0`

**해석**  
Small 개선이 사라졌고 mAP floor도 만족하지 못했다.

**결론**  
Mosaic scalar sweep을 더 세분화해서 이어가는 것은 validation overfitting 위험 대비 가치가 낮다고 판단해 종료했다.

---

### R11 — Checkpoint selection audit

**가설**  
Ultralytics가 선택하는 `best.pt`가 framework fitness 중심이기 때문에, 학습 중 다른 epoch에는 Small Recall과 overall quality를 동시에 만족하는 checkpoint가 존재할 수 있다.

**방법**
- R6 조건 재학습
- `save_period=1`
- 전체 epoch checkpoint를 동일한 predeclared gate로 audit

Gate:
- Small Recall > `0.25`
- Mask mAP50-95 >= `0.40`
- Multi Recall >= `0.50`
- Good-negative FP = `0`

**결과**  
Small Recall이 `0.375`인 epoch는 존재했지만 네 gate를 동시에 만족한 checkpoint는 하나도 없었다.

**해석**  
문제는 단순히 Ultralytics의 `best.pt` 선택 규칙 때문이 아니었다.

**결론**  
checkpoint-selection hypothesis 종료.

---

## 4. Small-centered crop과 Region Coverage 분석

### R12 — component-aware x2 + small-centered crop350 + Mosaic 1.0

**가설**  
Small defect가 원본 640 resize에서 너무 작게 보이는 것이 문제라면, Small component를 중심으로 한 deterministic crop을 추가해 상대적인 defect 크기를 키우는 방식이 도움이 될 수 있다.

**변경**
- sampling mode: `component_aware_crop`
- canonical entries: `84`
- component-aware duplicate entries: `19`
- Small-aware crop entries: `14`
- crop size: `350`
- total train entries: `117`
- positive exposure: `75`
- negative exposure: `42`
- Mosaic: `1.0`

**결과**
- Mask mAP50-95: `0.369284`
- TP / FP / FN: `13 / 22 / 10`
- Strict Recall: `0.565217`
- F1: `0.448276`
- Small Recall: `0.250`
- Multi Recall: `0.500`
- Good-negative FP: `0`

**해석**  
Small-centered crop을 추가했지만 Small Recall은 개선되지 않았고 FP가 크게 늘어 전체 품질이 악화됐다.

이 시점에서 failure preview를 보면 실제 결함 위치를 어느 정도 덮지만 strict IoU에서는 실패하는 사례가 확인되기 시작했다. 이후 strict instance matching만으로는 실패 원인을 충분히 설명하기 어렵다고 판단했다.

---

### Region Coverage Audit 도입

R12/R13 분석 과정에서 다음 현상이 반복적으로 관찰됐다.

- prediction이 실제 Small GT 위치를 상당 부분 덮음
- 하지만 여러 GT를 큰 mask 하나로 합치거나 boundary가 과대 예측됨
- 개별 GT와 single prediction IoU는 `0.5` 미만
- strict metric에서는 FN + FP로 계산

따라서 기존 strict metric을 유지하면서 secondary diagnostic으로 Region Coverage를 추가했다.

사용 지표:
- `strict_instance_gt_recall`
- `gt_component_coverage_recall_at_50`
- `small_gt_coverage_recall_at_50`
- `class_aware_union_iou`
- `class_aware_union_gt_coverage`
- `class_aware_union_pred_precision`
- `near_miss_iou_030_to_050`
- `covered50_but_strict_instance_fail`

Region metric은 strict metric을 대체하지 않으며, **miss / merged-mask / wrong-class / oversized-mask**를 구분하기 위한 진단 지표로 사용했다.

---

### R13 — crop350 + Mosaic 0.0

**가설**  
R12의 crop 자체보다, crop sample이 다시 Mosaic 안으로 들어가면서 Small-centered 효과가 희석되는 것이 문제일 수 있다. crop350은 유지하고 Mosaic만 제거해 본다.

**변경**
- R12 train view 유지
- `mosaic=1.0 -> 0.0`

**결과**
- Mask mAP50-95: `0.426415`
- Mask Recall: `0.624923`
- TP / FP / FN: `14 / 12 / 9`
- Strict Precision: `0.538462`
- Strict Recall: `0.608696`
- F1: `0.571429`
- Small Recall: `0.250`
- Multi Recall: `0.428571`
- Good-negative FP: `0`

Region Coverage:
- Component Coverage@50: `0.782609`
- **Small Coverage@50: `0.625` (`5/8`)**
- Union IoU: `0.611346`
- GT Coverage: `0.788059`
- Prediction Precision: `0.731639`
- Covered>=50% but strict fail: `4`

**해석**  
Strict Small Recall은 여전히 `2/8`이었지만 Small Coverage는 reference의 `3/8`에서 `5/8`로 증가했다.

즉 작은 결함을 더 자주 **공간적으로 인식**하기 시작했지만, 이를 개별 instance로 tight하게 분리하지 못해 strict 성공으로 전환되지 않았다.

**핵심 인사이트**
- Small spatial awareness에는 개선 신호가 존재
- 병목은 단순 detection miss뿐 아니라 merged/oversized mask와 instance separation에도 있음

R13은 strict 기준에서는 DROP이지만 이후 실험 방향을 바꾼 중요한 partial-positive 결과였다.

---

### R14 — no crop + Mosaic 0.0 ablation

**가설**  
R13의 Small Coverage 개선이 crop350 때문인지 Mosaic 제거 때문인지 분리해야 한다.

**변경**
- crop 제거
- component-aware x2만 유지
- `mosaic=0.0`

**결과**
- Mask mAP50-95: `0.278369`
- TP / FP / FN: `13 / 14 / 10`
- Strict Recall: `0.565217`
- F1: `0.520000`
- Small Recall: `0.250`
- Multi Recall: `0.642857`
- Good-negative FP: `0`

Region Coverage:
- Component Coverage@50: `0.695652`
- **Small Coverage@50: `0.625`**
- Union IoU: `0.533579`
- GT Coverage: `0.690344`
- Prediction Precision: `0.701468`
- Covered>=50% but strict fail: `3`

**비교에서 얻은 결론**

Small Coverage@50:
- Reference: `0.375`
- R12 crop + Mosaic 1.0: `0.375`
- R13 crop + Mosaic 0.0: `0.625`
- R14 no crop + Mosaic 0.0: `0.625`

따라서 Small spatial awareness 상승의 주된 요인은 crop350 자체보다 **Mosaic 제거**로 판단했다.

반면 R13과 R14를 비교하면 crop350이 있을 때 mAP, Component Coverage, Union IoU, GT Coverage가 모두 더 높았다.

**인사이트**  
- Mosaic off: Small defect를 더 자주 보게 하는 역할
- crop350: Mosaic off로 무너진 segmentation 품질을 일부 회복시키는 역할

이 결과로 다음 목표는 “더 많이 보게 하기”가 아니라 **본 Small defect를 개별 instance로 더 정확히 분리하기**로 바뀌었다.

---

### R15 — overlap_mask=False

**가설**  
R13에서 보인 merged-mask 현상이 training mask representation의 `overlap_mask=True`와 관련 있을 수 있다.

**변경**
- R13과 동일
- `overlap_mask=True -> False`

**결과**  
R13과 모든 metric이 완전히 동일했다.

추가 검증:
- model tensor key 동일
- final weights bitwise equal: `True`
- different tensors: `0`
- max abs weight diff: `0.0`
- training history excluding time: identical

**해석**  
현재 dataset/configuration에서는 `overlap_mask` 변경이 실제 optimization에 아무 영향도 주지 않았다.

**결론**
- `NO_EFFECT_ABLATION`
- overlap_mask branch 종료

---

## 5. 모델 용량 가설과 최종 결합 실험

### R16 — YOLO11n → YOLO11s capacity ablation

**가설**  
Small defect의 class/instance separation 실패가 YOLO11n의 모델 capacity 부족 때문일 수 있다.

**변경**
- R13 recipe 유지
- `yolo11n-seg.pt -> yolo11s-seg.pt`

**결과**
- Mask Precision: `0.834519`
- Mask Recall: `0.611709`
- Mask mAP50-95: `0.343211`
- TP / FP / FN: `14 / 11 / 9`
- Strict Recall: `0.608696`
- F1: `0.583333`
- **Small Recall: `0.125` (`1/8`)**
- Multi Recall: `0.500`
- Good-negative FP: `0`

Region Coverage:
- Component Coverage@50: `0.695652`
- Small Coverage@50: `0.375`
- Union IoU: `0.614282`
- GT Coverage: `0.721547`
- Prediction Precision: `0.805148`
- Covered>=50% but strict fail: `2`

Resource:
- Training time: `469.97 sec`
- Model size: `20,528,228 bytes`
- trainer GPU memory: 약 `5.84 GiB`

**해석**  
더 큰 모델은 prediction precision을 일부 높였지만 Small Recall과 Small Coverage를 모두 악화시켰고 mAP floor도 통과하지 못했다.

즉 현재 문제는 단순한 model capacity 부족으로 설명되지 않았다.

**결론**
- DROP
- capacity branch 종료
- YOLO11m으로 더 키우는 실험은 진행하지 않음

---

### R17 — crop350 + no-Mosaic + mask_ratio=2

**가설**  
서로 다른 문맥에서 확인된 두 신호를 다시 결합한다.

- R13: `crop350 + mosaic=0`에서 Small spatial awareness 개선
- R8b: `mask_ratio=2`에서 segmentation mask quality 개선

R9에서는 `mosaic=0.75 + mask_ratio=2`가 실패했지만, R13에서 완전히 다른 train-view/crop 문맥이 만들어졌으므로 `mask_ratio=2`를 다시 검증할 가치가 있다고 판단했다.

**변경**
- model: `yolo11n-seg.pt`
- `component_aware_crop`
- sampling x2
- crop size `350`
- total train entries `117`
- positive exposure `75`
- negative exposure `42`
- `mosaic=0.0`
- `mask_ratio=2`
- `scale=0.5`
- seed `42`

**학습**
- Best epoch: `79`
- Early stop: epoch `99`
- Training time: `530.02 sec`
- Model size: `6,015,012 bytes`
- trainer GPU memory: 약 `7.03 GiB`

**Framework 결과**
- Mask Precision: `0.804189`
- **Mask Recall: `0.738068`**
- Mask mAP50: `0.763544`
- **Mask mAP50-95: `0.448025`**

**Strict diagnostic**
- TP / FP / FN: `15 / 6 / 8`
- Precision: `0.714286`
- Recall: `0.652174`
- **F1: `0.681818`**
- **Small Recall: `0.375` (`3/8`)**
- Medium Recall: `0.857143`
- Large Recall: `0.750`
- **Multi Recall: `0.571429`**
- Single Recall: `0.777778`
- Good-negative FP: `0`

Per class:
- bent: TP `4`, FP `3`, FN `4`, Recall `0.500`
- color: TP `3`, FP `0`, FN `1`, Recall `0.750`
- scratch: TP `8`, FP `3`, FN `3`, Recall `0.727273`

**Region Coverage**
- Component Coverage@50: `0.739130`
- **Small Coverage@50: `0.625` (`5/8`)**
- Union IoU: `0.664664`
- GT Coverage: `0.732779`
- **Prediction Precision: `0.877307`**
- Covered>=50% but strict fail: `2`

### R17 vs Reference
- Mask mAP50-95: `0.414314 -> 0.448025` (`+0.033711`)
- Framework Mask Recall: `0.674182 -> 0.738068` (`+0.063886`)
- Strict Recall: `0.695652 -> 0.652174` (`-0.043478`)
- F1: `0.711111 -> 0.681818` (`-0.029293`)
- **Small Recall: `0.250 -> 0.375`**
- **Multi Recall: `0.500 -> 0.571429`**
- Good-negative FP: `0 -> 0`

### R17 vs R13
- Mask mAP50-95: `0.426415 -> 0.448025`
- TP / FP / FN: `14 / 12 / 9 -> 15 / 6 / 8`
- Strict Recall: `0.608696 -> 0.652174`
- F1: `0.571429 -> 0.681818`
- **Small Recall: `0.250 -> 0.375`**
- **Multi Recall: `0.428571 -> 0.571429`**
- Small Coverage@50: `0.625 -> 0.625` 유지
- Union IoU: `0.611346 -> 0.664664`
- Prediction Precision: `0.731639 -> 0.877307`
- Covered>=50% but strict fail: `4 -> 2`

### 사전 Primary gate
- Small Recall > `0.25`: **PASS**
- Mask mAP50-95 >= `0.40`: **PASS**
- Multi Recall >= `0.50`: **PASS**
- Good-negative FP = `0`: **PASS**

Secondary:
- Diagnostic Recall non-regression: FAIL
- F1 non-regression: FAIL
- Framework Mask Recall non-regression: PASS

**해석**  
R17은 Fast Research에서 처음으로 사전에 정한 Primary gate를 모두 통과했다.

특히 R13에서 얻었던 Small Coverage `5/8`을 그대로 유지하면서 Strict Small Recall이 `2/8 -> 3/8`로 증가했고, `covered50_but_strict_instance_fail`도 `4 -> 2`로 감소했다.

즉 단순히 Small 영역을 더 넓게 덮은 것이 아니라, **spatial awareness의 일부가 실제 strict instance 성공으로 전환된 첫 실험**이다. Prediction Precision도 `0.877`로 높아져 over-segmentation만으로 성능을 만든 결과로 보기 어렵다.

다만 Small GT는 8개뿐이므로 `0.25 -> 0.375`는 실제로는 1개 instance 개선이다. 따라서 결과를 과해석하지 않고 재현 확인이 필요하다.

**결론**
- **KEEP_FOR_CONFIRMATION**
- Fast Research 최초의 Primary-gate 통과 후보

---

## 6. 실험을 통해 확인된 실패 유형

Small defect 실패는 하나의 원인으로 설명되지 않았다.

### 실제 miss
같은 class prediction union 기준 coverage 자체가 거의 0인 경우. 모델이 작은 결함 위치를 실제로 보지 못한 사례다.

### merged mask
작은 GT 여러 개의 위치는 덮지만 하나의 큰 prediction으로 합쳐 예측하는 경우. Region Coverage는 높지만 개별 instance IoU가 낮아 strict에서는 실패한다.

### wrong-class overlap
공간적으로 GT 위치를 보지만 defect class를 잘못 예측하는 경우다.

### coarse / oversized mask
GT coverage는 높지만 prediction이 지나치게 넓어 Union IoU와 prediction precision이 낮아지는 경우다.

연구 후반부의 핵심 병목은 단순히 “Small defect를 못 본다”가 아니라, **본 defect를 올바른 class와 tight한 instance mask로 분리하는 문제**까지 포함하는 것으로 정리됐다.

---

## 7. 연구 흐름에서 얻은 핵심 인사이트

1. Small sample을 단순 반복하는 oversampling은 Small Recall을 개선하지 못했다.
2. scale augmentation 축소 역시 Small defect 문제에 유효하지 않았다.
3. Mosaic 감소는 여러 실험에서 Small Recall 또는 Small Coverage 변화와 반복적으로 연결됐다.
4. 그러나 Mosaic을 과도하게 낮추면 전체 segmentation 품질이나 normal-image guardrail이 무너질 수 있다.
5. `close_mosaic` tail을 늘리는 것만으로는 Small 문제를 해결하지 못했다.
6. `mask_ratio=2`는 mask mAP를 개선하는 강한 신호였지만 단독으로 Small Recall을 올리지는 못했다.
7. `mosaic=0.75 + mask_ratio=2`는 부정적 interaction을 보였으므로 좋은 단일 요인을 단순 결합한다고 항상 좋아지는 것은 아니었다.
8. checkpoint selection을 전체 epoch 수준에서 audit했지만 gate를 모두 통과한 hidden checkpoint는 없었다.
9. crop350 단독은 Small awareness를 개선하지 못했다.
10. `mosaic=0`에서 Small Coverage가 `0.375 -> 0.625`로 상승해, Mosaic 제거가 Small spatial awareness 증가의 주요 원인으로 확인됐다.
11. crop350은 Mosaic-off 상태에서 무너진 segmentation 품질을 회복하는 보완 역할을 했다.
12. `overlap_mask=False`는 현재 데이터에서는 bitwise no-op이었다.
13. YOLO11s로 capacity를 늘리는 것은 오히려 Small Recall을 `0.125`로 악화시켰으므로 capacity 부족이 핵심 원인은 아니었다.
14. R17에서 `crop350 + mosaic=0 + mask_ratio=2`를 결합했을 때 Small Coverage를 유지하면서 Strict Small Recall, Multi Recall, mask mAP, FP 수가 동시에 개선됐다.
15. R17은 최초로 Primary gate를 모두 통과했지만 Small GT 8개 중 1개 개선이므로 재현성 확인이 필요하다.
16. test split은 모든 Fast Research 동안 봉인했다.

---

## 8. 현재 연구 상태

Fast Research 단계의 현재 최우선 후보는:

`fr_r17_crop350_nomosaic_maskratio2`

판정:

**KEEP_FOR_CONFIRMATION**

현재까지의 연구 흐름은 다음과 같이 요약된다.

`Oversampling 실패 -> Scale 실패 -> Reduced Mosaic에서 Small 신호 발견 -> Mask resolution에서 품질 신호 발견 -> 단순 결합 실패 -> Checkpoint-selection 가설 종료 -> Small-centered crop 도입 -> Region Coverage로 실패 유형 분해 -> Mosaic-off가 Small awareness의 핵심 요인임을 확인 -> overlap-mask no-op 확인 -> model capacity 가설 실패 -> crop350 + no-Mosaic + mask_ratio2에서 최초 Primary gate 전체 통과`

R17은 최종 모델이라고 확정하지 않으며, 같은 recipe가 재현되는지 확인된 뒤 최종 후보 여부를 판단한다.
