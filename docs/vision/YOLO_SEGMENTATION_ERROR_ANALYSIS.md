# YOLO Segmentation Validation Error Analysis

## 1. 목적과 범위

C4-1은 고정된 YOLO11n-seg Baseline v1의 validation behavior를 설명하는 diagnostics다. 재학습, model
comparison, test 재평가, production confidence 결정 또는 threshold tuning 단계가 아니다. 분석 입력은
supervised-derived `val` 28장뿐이며 `test` row가 analysis boundary에 들어오면 fail-fast한다. 기존 test metric은
가설 생성이나 C4-2 experiment 선택에 사용하지 않는다.

실행 명령:

```bash
uv run python -m pipelines.analyze_yolo_segmentation_errors --device mps
```

기본 output은 Git-ignored `outputs/analysis/yolo_segmentation/error_analysis/`이다. 기존
`smartfactory_yolo11n_seg_metal_nut_seed42_t4` runtime bundle의 metadata/checkpoint SHA와 dataset Manifest SHA를
검증하고 model 및 metadata SHA가 실행 전후 동일한지도 확인한다. Production API response는 확장하지 않았다.

## 2. Protocol

Baseline operating point는 diagnostic confidence `0.25`다. GT polygon 하나를 derived annotation의 한 component
instance로 복원하고 prediction mask와 source resolution에서 비교한다. Matching은 class-aware greedy maximum mask
IoU이며 threshold는 common diagnostic boundary `IoU >= 0.5`다. Candidate는 mask IoU 내림차순,
GT index, prediction index 순으로 결정해 재현 가능하다.

다음 값은 Ultralytics mAP가 아니라 C4-1 instance matching diagnostics다.

- TP: 같은 class의 GT/prediction mask가 IoU 0.5 이상으로 match
- FP: match되지 않은 predicted instance
- FN: match되지 않은 GT component
- Wrong class: unmatched GT/prediction의 mask IoU가 0.5 이상이지만 class가 다름
- Low-IoU localization: matched IoU가 0.65 미만이거나 unmatched same-class overlap이 `(0, 0.5)`
- Under-segmentation: matched prediction의 GT coverage가 0.75 미만
- Over-segmentation: matched prediction precision이 0.75 미만
- Multi-component miss: manifest component count가 2 이상인 image에서 하나 이상의 GT component miss

Main taxonomy는 mutually exclusive하게 하나를 선택하고, secondary tags는 동시 failure evidence를 보존한다.
Negative에 prediction이 없는 경우는 `TRUE_NEGATIVE`, 오류 없는 positive match는 `TRUE_POSITIVE`다.

## 3. Size와 confidence policy

Size bucket은 business threshold가 아니라 validation GT instance mask-area ratio의 descriptive tertile이다.

| Bucket | Validation boundary |
|---|---:|
| Small | `area_ratio <= 0.0159476190` |
| Medium | `0.0159476190 < area_ratio <= 0.0244714286` |
| Large | `area_ratio > 0.0244714286` |

Confidence sweep `0.10/0.15/0.20/0.25/0.30/0.40/0.50`은 validation에서만 수행했다. Model을 최저 `0.10`으로
한 번 실행한 뒤 같은 prediction pool을 filtering하므로 operating point 간 model execution 차이를 만들지 않는다.
이 sweep은 C4-2 후보 생성용이며 production threshold를 선택하지 않는다.

## 4. Actual local result

2026-08-26 macOS MPS에서 runtime baseline artifact와 derived validation 28장을 분석했다.

| Environment / provenance | Value |
|---|---|
| Device | MPS |
| torch | `2.13.0` |
| Validation samples | 28 = 14 positive + 14 good negative |
| GT instances | 23 |
| Dataset Manifest SHA | `1746338c091c18e96a11399c81ea9be0d7350105c4860cfa6a4162144ddb9905` |
| Model SHA | `594003121b0e071c47d68c3e53c10f438dcec18b5b56b4e5d8831d64001192bd` |
| Artifact metadata SHA | `9f3e3878141e831a6721c5136d67057da906485b9825262bd4e0897b2879fc6b` |

Confidence 0.25의 instance diagnostics:

| TP | FP | FN | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|
| 14 | 17 | 9 | 0.451613 | 0.608696 | 0.518519 |

FP 17개는 전부 positive image의 duplicate, fragment 또는 wrong-class unmatched prediction이다. Good negative
14장에서는 FP image/instance가 모두 0이었다. 따라서 이 결과를 background false alarm 문제로 일반화하지 않는다.

### Per-class

| Class | Samples | GT instances | TP | FP | FN | Precision | Recall | Mean matched mask IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bent | 5 | 8 | 4 | 4 | 4 | 0.500000 | 0.500000 | 0.780252 |
| color | 4 | 4 | 3 | 2 | 1 | 0.600000 | 0.750000 | 0.812484 |
| scratch | 5 | 11 | 7 | 11 | 4 | 0.388889 | 0.636364 | 0.783105 |

Scratch는 가장 많은 unmatched predictions와 가장 낮은 precision을 보였다. Bent는 recall 0.50으로 네 component를
miss했다. Color는 세 class 중 recall이 가장 높지만 `metal_nut_test_color_021`에는 prediction이 없었다.

### Error taxonomy와 worst samples

Main sample taxonomy는 `TRUE_NEGATIVE 14`, `TRUE_POSITIVE 2`, `MISSED_DEFECT 6`, `WRONG_CLASS 1`,
`FALSE_POSITIVE 4`, `LOW_IOU_LOCALIZATION 1`이다. Secondary evidence는 missed 7, false-positive 9,
multi-component miss 5, low-IoU localization 6, under-segmentation 3, wrong-class 1이며 over-segmentation은 0이다.

- Highest-FN images: `metal_nut_test_bent_010`과 `metal_nut_test_scratch_006`, 각각 2 FN
- Complete miss: `metal_nut_test_color_021`, prediction 0 / FN 1
- Lowest matched mask IoU: `metal_nut_test_scratch_008`, `0.510205`
- Wrong class: `metal_nut_test_scratch_008`의 scratch component 하나가 bent로 prediction
- Positive over-prediction: `metal_nut_test_scratch_009`, TP 2 / FP 4 / FN 1
- Good-negative false positive: 없음 at confidence 0.25

Confusion-like instance table은 bent `4 correct / 4 no-prediction`, color `3 correct / 1 no-prediction`, scratch
`7 correct / 3 no-prediction / 1 bent`다.

### Size와 multi-component behavior

| Group | Instances | TP | FN | Recall | Mean matched mask IoU |
|---|---:|---:|---:|---:|---:|
| Small | 8 | 2 | 6 | 0.250000 | 0.802257 |
| Medium | 7 | 6 | 1 | 0.857143 | 0.819805 |
| Large | 8 | 6 | 2 | 0.750000 | 0.752809 |
| Single-component | 9 | 7 | 2 | 0.777778 | 0.778242 |
| Multi-component | 14 | 7 | 7 | 0.500000 | 0.798929 |

Small instances와 multi-component images에서 주된 차이는 match 이후 IoU보다 recall이다. 검출된 small/multi
component의 mean IoU는 낮지 않지만 검출되지 않은 component가 많다. Validation sample이 작으므로 통계적
일반화가 아니라 다음 controlled experiment의 근거로만 사용한다.

### Validation-only confidence sweep

| Confidence | Precision | Recall | F1 | Good FP image rate | Mean predictions/image |
|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.304348 | 0.608696 | 0.405797 | 0.142857 | 1.642857 |
| 0.15 | 0.358974 | 0.608696 | 0.451613 | 0.142857 | 1.392857 |
| 0.20 | 0.424242 | 0.608696 | 0.500000 | 0.000000 | 1.178571 |
| 0.25 | 0.451613 | 0.608696 | 0.518519 | 0.000000 | 1.107143 |
| 0.30 | 0.482759 | 0.608696 | 0.538462 | 0.000000 | 1.035714 |
| 0.40 | 0.541667 | 0.565217 | 0.553191 | 0.000000 | 0.857143 |
| 0.50 | 0.590909 | 0.565217 | 0.577778 | 0.000000 | 0.785714 |

0.10으로 낮춰도 recall이 증가하지 않았고 good FP만 발생했다. 0.40 이상에서는 color recall이 0.75에서 0.50으로
감소했다. F1 최대값만 보고 0.50을 production threshold로 선택하지 않는다. Calibration을 진행하려면 validation
objective, error cost와 operating-point protocol을 C4-2에서 먼저 정의해야 한다.

## 5. Diagnostic outputs

```text
outputs/analysis/yolo_segmentation/error_analysis/
├── sample_analysis.jsonl
├── summary.json
├── per_class.json
├── confidence_sweep.json
├── error_taxonomy.json
├── improvement_hypotheses.json
└── visualizations/                 # deterministic Top-10
```

각 sample row에는 GT/predicted instance count, expected class hit, TP/FP/FN, best mask/box IoU, confidences,
GT/predicted area ratio, component count, size bucket, main/secondary taxonomy, match detail과 confusion pair가 있다.
Visualization은 original, GT mask/bbox와 predicted mask/bbox/class/confidence를 나란히 표시한다. Output은 Git에
추가하지 않는다.

## 6. Evidence-based C4-2 candidates

우선순위 후보는 각각 독립 실험으로 비교하며 validation evidence로만 선택한다.

1. `imgsz` 상향 실험: small recall 0.25가 large 0.75보다 현저히 낮은지 재검증한다.
2. Component-preserving crop/sampling 실험: multi-component recall 0.50과 single-component 0.778 차이를 검증한다.
3. Larger segmentation model 대조: low-IoU/under-coverage와 scratch fragmentation이 개선되는지 확인한다.
4. `metal_nut_test_scratch_008` annotation/prediction audit 후 class-confusion targeted data experiment를 판단한다.
5. 별도 confidence calibration protocol: production threshold 결정이 아니라 validation cost trade-off를 비교한다.

현재 evidence가 지원하지 않는 결론:

- Confidence를 0.25 아래로 낮추면 FN이 회복된다는 가설
- Good-negative false alarm 때문에 retraining이 필요하다는 가설
- Over-segmentation이 주요 failure mode라는 가설
- 무조건 augmentation을 늘려야 한다는 결론
- 특정 confidence가 production-calibrated라는 결론
- Test metric을 근거로 C4-2 model/hyperparameter를 선택하는 행위

Baseline v1 artifact는 수정하지 않는다. C4-2가 학습을 수행한다면 새 artifact ID와 validation-only selection
record를 사용하고, test split은 experiment 선택이 끝난 뒤 최종 evaluation boundary에서만 사용한다.
첫 controlled candidate인 C4-2A의 predeclared hypothesis, fixed protocol, resource telemetry와 planned comparison은
[YOLO Segmentation Experiment Log](YOLO_SEGMENTATION_EXPERIMENT_LOG.md)에 기록한다. Actual Kaggle result는 아직
없으며 이 문서의 C4-1 evidence를 덮어쓰지 않는다.
