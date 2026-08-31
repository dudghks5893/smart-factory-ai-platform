# YOLO Segmentation Experiment Log

## 1. 목적과 적용 범위

이 문서는 YOLO segmentation model-quality evolution의 기술 source of truth다. 각 candidate는 한 번에 하나의
명시적 변수를 바꾸고, sealed `test` split이 아니라 `val` evidence로만 비교한다. `ACCEPT`는 다음 candidate로
보존할 가치가 있다는 뜻이며 runtime model 교체, production calibration 또는 factory certification을 뜻하지
않는다. 최종 candidate 선택과 derived-test 1회 평가는 C4-3의 별도 경계다.

현재 순서는 Baseline v1 → C4-1 validation error analysis → C4-2A higher resolution → C4-2B
component-aware x2 Official experiment까지 진행됐다. C4-2B는 `COMPLETED` / `PENDING`이며 final candidate로
확정되지 않았다. C4-2C crop confirmation은 구현만 준비됐고 Official run은 실행하지 않았다. C4-2D와 C4-3
final candidate selection도 아직 구현하거나 실행하지 않았다.

## 2. 공통 dataset과 validation protocol

Protocol 이름은 `MVTec AD-derived supervised segmentation feasibility split`이다. Dataset Manifest SHA-256은
`1746338c091c18e96a11399c81ea9be0d7350105c4860cfa6a4162144ddb9905`이며 class mapping은
`0=bent`, `1=color`, `2=scratch`다. `train` 84장은 parameter optimization, `val` 28장은 early stopping,
checkpoint selection, framework validation metric과 diagnostics에만 사용한다. C4-2 candidate 선택에는 derived
`test` row나 C2 final test evaluator를 사용하지 않는다.

C4-1-compatible diagnostic protocol은 confidence `0.25`, class-aware greedy maximum mask IoU matching,
`IoU >= 0.5` 및 deterministic tie-breaking이다. Size bucket은 C4-1에서 고정한 validation GT mask-area ratio
boundary를 재사용한다.

| Bucket | 고정 boundary |
|---|---:|
| Small | `area_ratio <= 0.015947619047619047` |
| Medium | `0.015947619047619047 < area_ratio <= 0.02447142857142857` |
| Large | `area_ratio > 0.02447142857142857` |

Ultralytics validation box/mask metrics와 C4-1 diagnostic instance metrics는 protocol이 다른 값이다. JSON,
table과 판단 근거에서 각각 `ultralytics`와 `diagnostic`으로 분리하며 서로 대체 가능한 metric으로 표현하지
않는다.

## 3. Baseline v1 / C4-1 starting reference

| Field | Value |
|---|---|
| Reference ID | `smartfactory_yolo11n_seg_metal_nut_seed42_t4` |
| Date | 2026-08-25; artifact metadata `created_at=2026-08-25T12:28:05.146385+00:00` |
| Git SHA | not captured |
| Status / Decision | `COMPLETED` / `REFERENCE` |
| Model initialization | `yolo11n-seg.pt` pretrained initialization |
| Training constants | `imgsz=640`, `batch=16`, `seed=42`, `epochs=100`, `patience=20`, baseline optimizer/deterministic/AMP/workers policy |
| Training progress | 80 completed epochs; configured 100 / patience 20에 대해 early stopped |
| Best epoch | 60, validation-selected `weights/best.pt` |
| Dataset Manifest SHA | `1746338c091c18e96a11399c81ea9be0d7350105c4860cfa6a4162144ddb9905` |
| Model SHA | `594003121b0e071c47d68c3e53c10f438dcec18b5b56b4e5d8831d64001192bd` |
| Metadata SHA | `9f3e3878141e831a6721c5136d67057da906485b9825262bd4e0897b2879fc6b` |
| Model artifact size | 6,011,684 bytes / 약 5.733 MiB |
| Environment evidence | Tesla T4, torch `2.13.0+cu130`, Ultralytics `8.4.128` |
| Dependency resolution | Linux x86_64 lock의 torchvision `0.28.0+cu130`; checkpoint runtime field에는 미포함 |

Evidence source는 SHA가 승인된 `model/model.pt` 내부 `train_metrics`와 `train_results`, project artifact
`model/metadata.json`, actual T4 artifact를 기록한 `YOLO_SEGMENTATION_DATASET.md`다. torchvision은 checkpoint가
직접 기록한 runtime observation이 아니라 해당 training pipeline commit의 `pyproject.toml` pin과 Linux x86_64
`uv.lock` CUDA resolution에서 확인한 dependency contract다.

Checkpoint `train_results.epoch`는 1부터 80까지이며 configured epoch 100, patience 20, best epoch 60과 함께
early stopping at epoch 80을 입증한다. `train_results.time`의 마지막 값은 `222.485`초지만 이는 Ultralytics가
checkpoint에 보존한 cumulative epoch time이다. 새 C4 telemetry가 정의하는 training start/end 전체 wall-clock과
동일한 boundary로 간주하지 않는다.

### 3.1 Ultralytics validation framework metrics

다음 값은 selected best checkpoint의 embedded `train_metrics`에서 직접 복원한 **validation** framework
metrics다. Derived-test evaluation file에서 가져온 값이 아니다.

| Metric | Box | Mask |
|---|---:|---:|
| Precision | 0.55955 | 0.55955 |
| Recall | 0.59848 | 0.59848 |
| mAP50 | 0.56157 | 0.59929 |
| mAP50-95 | 0.32398 | 0.34359 |

Per-class validation framework metrics와 model parameter count는 checkpoint/exported evidence에 없어
**not captured**다.

### 3.2 C4-1 validation diagnostics

C4-1은 이 immutable artifact를 2026-08-26 local macOS MPS에서 `val` 28장에 대해 분석했다. 따라서 다음
수치는 T4 training resource metric이 아니라 validation-only local diagnostic evidence다.

| Diagnostic metric at confidence 0.25 / mask IoU 0.5 | Value |
|---|---:|
| TP / FP / FN | 14 / 17 / 9 |
| Instance Precision / Recall / F1 | 0.451613 / 0.608696 / 0.518519 |
| Small / Medium / Large Recall | 0.250000 / 0.857143 / 0.750000 |
| Single / Multi-component Recall | 0.777778 / 0.500000 |
| Good-negative FP images | 0 / 14, rate 0.000000 |
| Good-negative FP instances | 0 |

Per-class Recall은 bent `0.500000`, color `0.750000`, scratch `0.636364`다. Overall FP 17개는 good image가
아니라 positive image의 unmatched duplicate, fragment 또는 wrong-class prediction에서 발생했다. Complete
miss는 `metal_nut_test_color_021`, worst FN evidence에는 `metal_nut_test_bent_010`과
`metal_nut_test_scratch_006`이 포함된다. `metal_nut_test_scratch_008`의 최저 matched mask IoU는
`0.510205`이며 scratch component 하나가 bent로 prediction된 wrong-class evidence가 있다. 자세한 taxonomy는
[C4-1 Error Analysis](YOLO_SEGMENTATION_ERROR_ANALYSIS.md)를 따른다.

### 3.3 Historical derived-test evidence — selection excluded

Exported `evaluation/metrics.json`은 `split="test"`, sample count 28로 명시된 historical independent
derived-test evidence다. Box Precision/Recall/mAP50/mAP50-95는 각각 `0.6739637238`, `0.6891269447`,
`0.7197372270`, `0.4435146476`이고 mask 값은 `0.8427370779`, `0.5397984928`, `0.6961360407`,
`0.4250304964`다. 이 subsection은 protocol provenance 확인용이며 C4-2A의 `quality_before`, overview table,
decision policy 또는 candidate selection field에 이 값을 넣지 않는다.

Baseline total VRAM, exact end-to-end training wall-clock duration, PyTorch peak allocated/reserved memory, sampled
device-wide VRAM, GPU utilization mean/p50/p95/max와 power는 **not captured**다. Hardware spec이나 새 telemetry
schema로 과거 값을 추정하지 않는다.

## 4. C4-2A predeclared experiment와 official identity

| Field | Value |
|---|---|
| `experiment_id` | `c4_2a_yolo11n_seg_imgsz1024_seed42` |
| Date | 2026-08-27 |
| Official Git SHA | `1353aefed744ad5c67e931b6e7dd4034c903c065` |
| Status / Decision | `REJECTED` / `REJECT` |
| Hypothesis | `imgsz=640`이 small defect component 검출·분할에 필요한 spatial information을 잃을 수 있다. |
| Target failure mode | `small_defect_miss` |
| 단일 controlled change | `training.imgsz: 640 -> 1024` |
| Dedicated config | `configs/experiments/yolo_segmentation/c4_2a_yolo11n_seg_imgsz1024_seed42.yaml` |
| Config SHA-256 | `49be69586f5aecb04665f5a9346c0e6fd5de9e5cf636c4dd6fb785c5e0dc890b` |
| Fixed model initialization | `yolo11n-seg.pt`, model family `yolo11n-seg` |
| Fixed training contract | seed 42, batch 16, epochs 100, workers 2, patience 20, optimizer `auto`, deterministic, AMP |
| Hardware | Single Tesla T4 |
| Dataset | Manifest `1746338c091c18e96a11399c81ea9be0d7350105c4860cfa6a4162144ddb9905`; train 84 / val 28 |
| Test boundary | `SEALED`, `NOT USED` |

Candidate는 Baseline fine-tuned `model.pt`에서 이어 학습하지 않고 기존 pipeline과 동일한 pretrained
`yolo11n-seg.pt` initialization에서 시작한다. Model family, seed 42, batch 16, epochs 100, patience 20,
optimizer, deterministic/AMP/workers policy, dataset row identity, Manifest, class taxonomy 및 validation protocol은
고정한다. Batch 16 CUDA OOM 시 batch 8로 자동 retry하지 않으며 실패 evidence를 남기고 실제 training error를
전파한다.

결과를 보기 전에 정한 evaluation priority는 다음과 같다.

1. Primary: Ultralytics validation mask mAP50-95, diagnostic instance Recall
2. Failure-focused: Small-defect Recall, Multi-component Recall
3. Guardrail: validation good-negative FP image rate
4. Secondary: Precision/F1, per-class Recall, unmatched/duplicate/fragment, wrong-class, localization behavior
5. Resource trade-off: peak VRAM, training duration, artifact size

실제 `recommend_experiment` implementation의 check는 다음과 같다. 현재 dedicated config에서는 다섯 check를
모두 required로 사용한다.

| Check | PASS condition | Role |
|---|---|---|
| `mask_map50_95_non_regression` | candidate `>=` baseline | Primary |
| `instance_recall_non_regression` | candidate `>=` baseline | Primary |
| `small_recall_improvement` | candidate `>` baseline | Failure-focused; strict improvement |
| `multi_component_recall_non_regression` | candidate `>=` baseline | Failure-focused |
| `good_negative_fp_guardrail` | candidate FP image rate `<=` baseline | Negative guardrail |

Required check가 모두 PASS하면 `ACCEPT`다. Mask mAP50-95, instance Recall 또는 good-negative FP guardrail 중
하나라도 실패하면 `REJECT`다. 이 세 check가 모두 통과했지만 Small Recall이나 Multi-component Recall만 실패하면
`PENDING`이다. Runner status는 각각 `ACCEPTED`, `REJECTED`, `COMPLETED`로 저장한다. 어떤 decision도 runtime
artifact를 자동 교체하지 않는다.

## 5. Experiment overview

| Experiment | Change | Val mask mAP50-95 | Diagnostic Recall | Small Recall | Multi Recall | Good FP | PyTorch peak reserved | Training time | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline v1 / C4-1 | Reference `imgsz=640` | 0.34359 | 0.608696 | 0.250000 | 0.500000 | 0/14 | not captured | not captured¹ | `REFERENCE` |
| C4-2A | `imgsz 640 -> 1024` | 0.303871 | 0.521739 | 0.125000 | 0.428571 | 0/14 | 7,551,844,352 bytes | 430.421539 sec | `REJECT` |
| C4-2B | component-aware eligible x2; Official run completed | 0.4236909445 | 0.6521739 | 0.250000 | 0.500000 | 0/14 | approximately 2.875 GB | 281.608 sec | `PENDING` |
| C4-2C | crop350 + no-Mosaic + mask ratio 2; implementation ready, Official run not run | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| C4-2D | Future slot; not implemented | — | — | — | — | — | — | — | — |
| C4-3 | Future final selection; not implemented | — | — | — | — | — | — | — | — |

¹ Baseline checkpoint에는 cumulative epoch time `222.485`초가 있으나 C4 telemetry의 exact end-to-end
wall-clock boundary가 아니므로 candidate의 `430.421539`초와 속도 비율을 계산하지 않는다.

### 5.1 Validation-only quality comparison

다음 before/after 값은 official retry package의 validation-only comparison evidence다. Section 3.2의 historical
C4-1 local diagnostic은 TP / FP / FN 14 / 17 / 9, Precision 0.451613, Recall 0.608696, F1 0.518519다. 반면
official retry가 생성한 `quality_before` baseline re-evaluation snapshot은 TP / FP / FN 14 / 16 / 9,
Precision 0.4666666667, Recall 0.6086956522, F1 0.5283018868이다. 두 run은 FP count가 1개 달라 Precision과
F1이 다르며 원인은 현재 evidence로 확정되지 않았다. Platform numeric behavior를 포함한 원인을 추측하지 않는다.

Candidate decision에는 official retry의 `quality_before` comparison snapshot을 사용한다. Section 3의 selected
checkpoint metric과 historical C4-1 local diagnostic은 그대로 보존하고 official retry snapshot으로 덮어쓰지
않는다. Historical derived-test metric도 candidate selection에 사용하지 않았다.

| Protocol | Metric | Baseline 640 | C4-2A 1024 | Result |
|---|---|---:|---:|---|
| Diagnostic | Precision | 0.4666666667 | 0.4000000000 | regressed |
| Diagnostic | Recall | 0.6086956522 | 0.5217391304 | regressed |
| Diagnostic | F1 | 0.5283018868 | 0.4528301887 | regressed |
| Ultralytics Mask | Precision | 0.5590870339 | 0.6504205887 | improved |
| Ultralytics Mask | Recall | 0.5984848485 | 0.4544197496 | regressed |
| Ultralytics Mask | mAP50 | 0.5971524426 | 0.5178000110 | regressed |
| Ultralytics Mask | mAP50-95 | 0.3435446786 | 0.3038714418 | regressed |
| Ultralytics Box | Precision | 0.5590870339 | 0.6090295004 | improved |
| Ultralytics Box | Recall | 0.5984848485 | 0.5151515152 | regressed |
| Ultralytics Box | mAP50 | 0.5619806728 | 0.5322743484 | regressed |
| Ultralytics Box | mAP50-95 | 0.3240964579 | 0.3260775877 | effectively flat / slight increase |

Candidate diagnostic confusion count는 TP 12 / FP 18 / FN 11이다. Mask Precision은 상승했지만 overall
instance Recall과 mask mAP50-95는 하락했다.

### 5.2 Failure-mode comparison

| Validation diagnostic | Baseline 640 | C4-2A 1024 | Result |
|---|---:|---:|---|
| Small Recall | 0.2500000000 | 0.1250000000 | **REGRESSED** |
| Medium Recall | 0.8571428571 | 1.0000000000 | improved |
| Large Recall | 0.7500000000 | 0.5000000000 | regressed |
| Single-component Recall | 0.7777777778 | 0.6666666667 | regressed |
| Multi-component Recall | 0.5000000000 | 0.4285714286 | **REGRESSED** |
| Good-negative FP images | 0 / 14 | 0 / 14 | unchanged; candidate rate 0.0 |

Candidate per-class diagnostic Recall은 bent 0.625, color 0.500, scratch 0.4545454545이고 Precision은 각각
0.4545454545, 0.6666666667, 0.3125다. 이 결과에서 가장 중요한 관찰은 실험의 primary target이던 Small
Recall이 0.25에서 0.125로 감소했다는 점이다.

### 5.3 Predeclared guardrail과 decision

| Predeclared check | Result |
|---|---|
| `good_negative_fp_guardrail` | **PASS** |
| `instance_recall_non_regression` | **FAIL** |
| `mask_map50_95_non_regression` | **FAIL** |
| `small_recall_improvement` | **FAIL** |
| `multi_component_recall_non_regression` | **FAIL** |

Official decision은 `REJECT`, status는 `REJECTED`다. Machine evidence의 decision reason은
`Predeclared primary or good-negative guardrail failed: mask_map50_95_non_regression,
instance_recall_non_regression, small_recall_improvement, multi_component_recall_non_regression`이다.
Good-negative guardrail 자체는 PASS했지만 나머지 네 required condition이 실패했으므로 candidate를 v2로
promote하지 않는다.

### 5.4 Training과 resource evidence

| Evidence | C4-2A official retry |
|---|---|
| GPU | Tesla T4 |
| Completed / configured epochs | 87 / 100 |
| Early stopping / best epoch | true / 67 |
| End-to-end training wall-clock | 430.421539109 sec |
| Average per completed epoch | 4.947374013 sec |
| Model size | 6,094,756 bytes / 5.8124 MiB |
| Parameters | 2,835,153 |
| `nvidia-smi` samples | 85 |
| GPU utilization mean / p50 / p95 / max | 57.0941% / 81% / 99% / 100% |
| Sampled device VRAM mean / max | 7,104.67 MiB / 7,407 MiB |
| GPU power mean / max | 55.1385 W / 99.18 W |
| PyTorch peak allocated | 6,834,240,512 bytes |
| PyTorch peak reserved | 7,551,844,352 bytes |
| Total device memory | 15,636,037,632 bytes |

Historical Baseline에는 같은 boundary의 end-to-end wall-clock, GPU utilization, device-wide VRAM, power 및
PyTorch peak telemetry가 없다. 따라서 정확한 baseline/candidate GPU usage 비교나 wall-clock speed ratio를
주장하지 않는다. Baseline의 `222.485`초는 checkpoint cumulative epoch time일 뿐 위 wall-clock과 동등하지 않다.

### 5.5 Artifact identity와 package

| Artifact | SHA-256 |
|---|---|
| Baseline model | `594003121b0e071c47d68c3e53c10f438dcec18b5b56b4e5d8831d64001192bd` |
| Baseline metadata | `9f3e3878141e831a6721c5136d67057da906485b9825262bd4e0897b2879fc6b` |
| Candidate model | `06ab9e64dc177f17bb28bfa6d86d90d02904ab0019cd550185dae2a013f8eec1` |
| Candidate metadata | `5b9a48fbc19df66f51d832e1c96fbe0bcb83b5bd3c1fba0dda5aebc4677b5bfe` |
| Official package | `e74b933473fac156c5309f24eb4d072991deeb9610ead5fa89f6e7605e6c9c25` |

Official package 이름은 `c4_2a_yolo11n_seg_imgsz1024_seed42_OFFICIAL_REJECTED.zip`이다. Package는 의도적으로
Git repository 밖에 보관하며 repository의 portable dependency로 사용하지 않는다. 내부에는 다음 evidence와
candidate artifact만 포함한다.

```text
evidence/experiment_metadata.json
evidence/training_metrics.json
evidence/validation_metrics.json
evidence/error_analysis_summary.json
evidence/resource_telemetry.json
evidence/comparison_to_baseline.json
evidence/experiment_result.json
evidence/environment.json
evidence/epoch_metrics.jsonl
evidence/visualization_manifest.json
model/model.pt
model/metadata.json
config/c4_2a_yolo11n_seg_imgsz1024_seed42.yaml
SHA256SUMS.txt
```

### 5.6 결론과 다음 hypothesis

C4-1에서 Small Recall 0.25라는 failure mode를 확인한 뒤 `imgsz` 하나만 640에서 1024로 변경하고, sealed
test를 열지 않은 validation-only controlled experiment를 수행했다. Quality, failure-mode와 GPU resource
evidence를 수집한 결과 mask Precision은 증가했지만 Small Recall은 0.125로 감소했고 overall instance Recall,
Multi-component Recall과 mask mAP50-95도 하락했다. 따라서 사전 정의한 guardrail에 따라 candidate를 거부했고
v2로 promote하지 않았다. 이 결론은 해당 dataset split, model, seed와 training protocol에 한정하며 higher
resolution이 일반적으로 성능을 악화시킨다는 주장으로 확장하지 않는다.

Commit `edc0bd4`의 첫 시도는 training과 final validation 후 epoch callback lifecycle bug로 종료되었으므로
`FAILED_AFTER_TRAINING`, `official_result_eligible=false`인 engineering history다. Runtime fix commit
`1353aefed744ad5c67e931b6e7dd4034c903c065`에서 수행한 retry만 official result이며 첫 시도의 metric이나
artifact identity를 섞지 않는다.

C4-2A 종료 시점의 다음 planned hypothesis는 **C4-2B component-preserving sampling strategy**였다.
이후 C4-2B 구현과 Official experiment가 수행됐으며, 이 historical C4-2A 결론은 후속 결과를
사전에 가정한 기록이 아니다.

### 5.7 C4-2B Official historical evidence

다음은 실제 Kaggle Official run에서 확보된
`c4_2b_yolo11n_seg_component_aware_sampling_x2_seed42`의 authoritative historical record다. Implementation과
Official run status는 모두 `COMPLETED`이고 decision은 `PENDING`이다. Candidate는 `ACCEPTED`, `CONFIRMED` 또는
final model이 아니며, derived test split은 `SEALED_NOT_USED`로 유지됐다.

| Field | Official value |
|---|---|
| Implementation | `COMPLETED` |
| Official run | `COMPLETED` |
| Status / Decision | `COMPLETED` / `PENDING` |
| Decision reason | `Failure-focused improvement is incomplete: small_recall_improvement` |
| Test used | `false`; `SEALED_NOT_USED` |

Candidate와 같은 session에서 다시 측정한 Baseline framework validation은 derived-test 결과가 아닌 validation-only
comparison reference다.

| Ultralytics Mask validation metric | Same-session Baseline | C4-2B Candidate |
|---|---:|---:|
| Precision | 0.559087 | 0.8145489791 |
| Recall | 0.598485 | 0.6267087419 |
| mAP50 | 0.597152 | 0.7112364954 |
| mAP50-95 | 0.343545 | 0.4236909445 |

| Strict candidate diagnostic at confidence 0.25 / mask IoU 0.5 | Value |
|---|---:|
| TP / FP / FN | 15 / 12 / 8 |
| Instance Precision | 0.5555556 |
| Instance Recall | 0.6521739 |
| Instance F1 | 0.6000000 |
| Small / Medium / Large Recall | 0.250 / 1.000 / 0.750 |
| Multi / Single-component Recall | 0.500 / 0.888889 |
| Good-negative FP images | 0 / 14; rate 0.0 |

| Decision check | Result |
|---|---|
| `mask_map50_95_nonregression` | **PASS** |
| `instance_recall_nonregression` | **PASS** |
| `small_recall_improvement` | **FAIL** |
| `multi_recall_nonregression` | **PASS** |
| `good_negative_guardrail` | **PASS** |

Primary non-regression과 good-negative guardrail은 통과했지만 failure-focused Small Recall이 Baseline `0.250`보다
strict하게 개선되지 않았다. 따라서 Official status는 `COMPLETED`, decision은 `PENDING`이며 candidate를 승인,
확정 또는 final promotion하지 않는다.

| Resource evidence | Official value |
|---|---:|
| Training wall time | 281.608 sec |
| Epoch logger cumulative time | 261.424 sec |
| Best epoch | 98 |
| PyTorch peak allocated | approximately 2.708 GB |
| PyTorch peak reserved | approximately 2.875 GB |
| `nvidia-smi` memory max / mean | 2,947 / 2,858.9 MiB |
| GPU utilization mean / max | 47.625% / 87% |
| GPU power mean / max | 51.01 / 90.17 W |
| Model size | 6,015,588 bytes |

| Artifact | SHA-256 |
|---|---|
| Official C4-2B model | `f14e1fde030bdc95658bb28a5de49fa1eb310f0c1957db3b0d83d162a9c76356` |
| Official C4-2B metadata | `98e5878cd1ca4277bb98cc738cfdbd1d23885a53d82130180c898562de99e48e` |
| Official C4-2B package | `0fe60bc0a000e74a6da8e17dfe8c2b6b824abe3424d334f353bcefebd6fd94f2` |

이 SHA들은 Official execution evidence의 identity다. Repository에 package binary가 포함되지 않은 것은 Official
run 미실행을 뜻하지 않는다. 다만 해당 ZIP이 local Mac에 보존됐다는 근거는 없으므로 local preservation을
주장하지 않는다. Fast Research의 clean C4-2B reference는 탐색용 별도 run이며 위 Official checkpoint의 exact
reproduction이나 artifact identity로 사용하지 않는다.

## 6. Machine-readable evidence contract

성공한 run은 Git-ignored
`outputs/experiments/yolo_segmentation/c4_2a_yolo11n_seg_imgsz1024_seed42/`에 다음 evidence를 만든다.

```text
experiment_metadata.json
training_metrics.json
epoch_metrics.jsonl
validation_metrics.json
error_analysis_summary.json
resource_telemetry.json
comparison_to_baseline.json
experiment_result.json
environment.json
visualization_manifest.json
package_metadata.json
baseline_framework_validation/
baseline_error_analysis/
candidate_framework_validation/
candidate_error_analysis/
```

Candidate artifact와 intermediate output은 ignored runtime namespace에 생성됐다. Official rejected package는
`c4_2a_yolo11n_seg_imgsz1024_seed42_OFFICIAL_REJECTED.zip` 이름으로 Git repository 밖에 보존한다. Machine
result는 `split="val"`, `test_split_used=false`, before/after quality, failure-mode/resource metrics,
Manifest/model/metadata/config SHA와 decision을 명시한다. `experiment_metadata.json`과
`comparison_to_baseline.json`에는 checkpoint-derived historical Baseline reference도 포함하되
`derived_test_metrics_used_for_selection=false`를 유지한다. Raw dataset, cache, model binary와 external ZIP은
Git에 추가하지 않는다.

## 7. GPU와 compute telemetry 정의

Training 직전 PyTorch CUDA peak counter를 reset하고 종료 또는 exception 뒤 다음 framework-owned 값을 읽는다.

- `torch.cuda.get_device_name()`과 total device memory
- `torch.cuda.max_memory_allocated()`
- `torch.cuda.max_memory_reserved()`
- UTC start/end와 training wall-clock duration

별도 lifecycle-managed thread는 5초 간격으로 `nvidia-smi`의 device-wide utilization, memory used/total 및
지원되는 경우 power draw를 sampling한다. Utilization과 memory는 sample count, mean, p50, p95, max를,
power는 supported sample의 mean/max를 기록한다. 5초 sampling은 짧은 spike를 놓칠 수 있고 device-wide
memory/utilization에는 같은 GPU의 다른 process가 포함될 수 있다.

PyTorch allocated/reserved peak는 framework allocator 관점이고 `nvidia-smi` memory는 device-wide sample이므로
같은 값으로 해석하지 않는다. `nvidia-smi` 부재, optional power 미지원 또는 malformed sample은 `null`/invalid
attempt evidence로 남기고 training을 중단하지 않는다. Sampling thread는 성공과 exception 모두에서 stop되며
실제 model-training error는 숨기지 않는다.

Training progress에는 completed epoch, early stopping 여부, validation-selected best epoch/checkpoint,
wall-clock duration, completed epoch당 평균 시간, model bytes/MiB를 기록한다. Validation 시 실제 loaded model에서
parameter count를 읽는다. Python, platform, torch, torchvision, CUDA runtime, Ultralytics, NVIDIA driver와
requested/actual device도 별도 environment evidence로 저장한다.

Per-epoch evidence는 pinned Ultralytics callback lifecycle에서 `epoch_metrics.jsonl`로 기록한다.
`epoch_time_seconds`는 project callback의 `on_train_epoch_start` 진입부터 `on_fit_epoch_end` 진입까지 측정한
**measured fit-epoch elapsed time**이다. 따라서 scheduler step, training batches, train-epoch-end 처리와 해당
epoch에서 실행된 validation, metric 저장 및 checkpoint 저장을 포함한다. Project `on_fit_epoch_end` callback 뒤의
memory clear와 early-stop broadcast/break는 포함하지 않는다. `cumulative_epoch_seconds`는 이 fit-epoch 측정값의
합이며 전체 runner의 end-to-end training wall-clock과 다른 boundary다. 제공되는 train loss/validation metric/LR와
optional CUDA reserved memory만 compact scalar로 남긴다. Historical Baseline checkpoint cumulative epoch time
`222.485`초 역시 새 end-to-end wall-clock과 직접 비교하지 않는다.

## 8. Kaggle experiment workbench

Reusable notebook [YOLO Segmentation Experiment Workbench](../../notebooks/vision/yolo_segmentation_experiment_workbench.ipynb)는
repository module을 호출하는 thin orchestration/visualization interface다. Research Mode는 임시 training override와
별도 ignored namespace를 사용하며 official evidence가 아니다. Official Mode는 committed config의 override를
거부하고 Manifest, Baseline SHA, Git provenance, CUDA와 `test_split_used=false`를 preflight에서 확인한다.

Pre-training view는 train/validation Manifest EDA, deterministic GT gallery, pinned Ultralytics의 실제 training
transform preview와 actual non-augmented letterbox를 사용한 640/1024 representation 비교를 제공한다. Post-training
view는 existing telemetry/result JSON과 C4-1-compatible validation diagnostics를 읽고 training curves, resource
summary, taxonomy별 deterministic failure gallery 및 동일 validation sample의 Baseline/Candidate 비교를 표시한다.
Generated PNG/JSONL은 ignored output에만 저장한다. Final Test Review는 C4-3 전까지 locked이며 test row를 load하지
않는다.

## 9. Hardware benchmark domain

Kaggle T4는 CUDA training duration, VRAM, utilization과 power의 experiment domain이다. 기존 local Apple
Silicon evidence는 MPS/CPU functional runtime smoke domain이며 측정 boundary도 다르다. Future production
NVIDIA serving latency는 다시 별도 domain이다. 이 문서는 서로 다른 run으로부터 MPS와 T4의 상대 성능을
결론 내리지 않는다.

## 10. Kaggle T4 execution과 reproduction contract

Actual official retry의 execution entry point는 Kaggle의
[YOLO Segmentation Experiment Workbench](../../notebooks/vision/yolo_segmentation_experiment_workbench.ipynb)다.
Workbench Section 12가 project-owned `run_yolo_segmentation_experiment` function을 호출해 official lifecycle을
수행했다. 아래 CLI는 같은 repository runner를 직접 호출하는 **equivalent reproduction path**이며 official retry가
CLI로 직접 실행됐다는 provenance를 뜻하지 않는다. CLI와 Workbench 모두 C2 final test evaluator를 호출하지 않고
같은 validation-only boundary를 사용한다.

```bash
cd /kaggle/working/smart-factory-ai-platform
uv sync --locked
uv run python - <<'PY'
import torch
import torchvision
from ultralytics import __version__ as ultralytics_version

print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("ultralytics", ultralytics_version)
print("cuda", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for C4-2A")
print("device", torch.cuda.get_device_name(0))
print("total_memory_bytes", torch.cuda.get_device_properties(0).total_memory)
PY
uv run python -m pipelines.run_yolo_segmentation_experiment \
  --experiment-config configs/experiments/yolo_segmentation/c4_2a_yolo11n_seg_imgsz1024_seed42.yaml \
  --dataset data/processed/supervised_derived/mvtec_ad/metal_nut/yolo_segmentation/v1 \
  --baseline-artifact-dir artifacts/runtime/yolo_segmentation/smartfactory_yolo11n_seg_metal_nut_seed42_t4 \
  --device cuda \
  --repository-root /kaggle/working/smart-factory-ai-platform
```

Runner는 dataset/config/baseline SHA와 CUDA를 먼저 검증하고, 동일 val protocol의 Baseline reference를 만든 뒤
candidate training과 telemetry를 시작한다. 이후 best checkpoint로 Ultralytics val metric과 C4-1 diagnostics를
계산하고 comparison, hashes와 package를 만든다. Batch 16 OOM이면 experiment를 `REJECTED` failure attempt로
기록하고 environment, T4 memory, available telemetry, error type/message와 failure point를 보존한 뒤 종료한다.
Batch를 바꾸거나 새 experiment를 자동 시작하지 않는다.

Official retry에서는 output export 후 `package_metadata.json`의 ZIP/model/metadata/config SHA와 ZIP 내부
`SHA256SUMS.txt`를 함께 검증했다. Section 5의 result와 artifact identity는 이 검증을 통과한 package를 근거로
하며, 후속 reproduction도 검증 완료 전에는 새 result로 간주하지 않는다.

## 11. C4-2C Official confirmation 준비 상태

`c4_2c_yolo11n_seg_crop350_nomosaic_maskratio2_seed42`의 repository-owned typed recipe, deterministic
component-aware duplicate/crop train view, explicit augmentation override, Fast-compatible validation prediction,
Region Coverage secondary audit와 absolute Primary confirmation gate가 구현됐다. Workbench는 같은 20-section
lifecycle에서 이 config를 preflight하고 repository runner를 호출한다.

현재 상태는 **IMPLEMENTATION READY / OFFICIAL CONFIRMATION NOT RUN**이다. 이 문서에는 Fast Research 결과를
Official 결과로 옮기지 않았으며 실제 C4-2C metric, decision 또는 model promotion을 기록하지 않는다. Derived test
split은 계속 `SEALED_NOT_USED`이고, runtime dataset YAML은 `train`, `val`, `names`만 포함한다. Primary gate를 모두
통과하더라도 `CONFIRMED_CANDIDATE`는 final model selection을 의미하지 않는다.

| C4-2C state | Value |
|---|---|
| Implementation | `READY` |
| Official run | `NOT RUN` |
| Decision | `NOT AVAILABLE` |
| Test | `SEALED_NOT_USED` |
| Official metric | none |
| Final promotion | none |
