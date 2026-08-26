# YOLO Segmentation Experiment Log

## 1. 목적과 적용 범위

이 문서는 YOLO segmentation model-quality evolution의 기술 source of truth다. 각 candidate는 한 번에 하나의
명시적 변수를 바꾸고, sealed `test` split이 아니라 `val` evidence로만 비교한다. `ACCEPT`는 다음 candidate로
보존할 가치가 있다는 뜻이며 runtime model 교체, production calibration 또는 factory certification을 뜻하지
않는다. 최종 candidate 선택과 derived-test 1회 평가는 C4-3의 별도 경계다.

현재 순서는 Baseline v1 → C4-1 validation error analysis → C4-2A higher resolution이다. C4-2B
component-preserving sampling/crop, C4-2C larger segmentation model, C4-2D validation cost-based confidence
calibration 및 C4-3 final candidate selection은 아직 실행하거나 구현하지 않았다.

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

## 4. C4-2A predeclared experiment

| Field | Value |
|---|---|
| `experiment_id` | `c4_2a_yolo11n_seg_imgsz1024_seed42` |
| Date | 2026-08-27 |
| Git SHA | 실행 시 repository HEAD와 `working_tree_dirty`를 자동 기록 |
| Status / Decision | `PLANNED` / `PENDING` |
| Hypothesis | `imgsz=640`이 small defect component 검출·분할에 필요한 spatial information을 잃을 수 있다. |
| Target failure mode | `small_defect_miss` |
| 단일 controlled change | `training.imgsz: 640 -> 1024` |
| Dedicated config | `configs/experiments/yolo_segmentation/c4_2a_yolo11n_seg_imgsz1024_seed42.yaml` |

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

`ACCEPT` recommendation에는 mask mAP50-95와 instance Recall non-regression, Small Recall improvement,
Multi-component Recall non-regression 및 good-negative FP rate non-increase가 모두 필요하다. Primary 또는
negative guardrail regression은 `REJECT`, failure-focused improvement만 불충분하면 `PENDING`이다. 어떤
decision도 runtime artifact를 자동 교체하지 않는다.

## 5. Experiment overview

| Experiment | Change | Val mask mAP50-95 | Diagnostic Recall | Small Recall | Multi Recall | Good FP | Peak VRAM | Training time | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline v1 / C4-1 | Reference `imgsz=640` | 0.34359 | 0.608696 | 0.250000 | 0.500000 | 0/14 | not captured | not captured¹ | `REFERENCE` |
| C4-2A | `imgsz 640 -> 1024` | pending | pending | pending | pending | pending | pending | pending | `PENDING` |
| C4-2B | Future slot; not implemented | — | — | — | — | — | — | — | — |
| C4-2C | Future slot; not implemented | — | — | — | — | — | — | — | — |
| C4-2D | Future slot; not implemented | — | — | — | — | — | — | — | — |
| C4-3 | Future final selection; not implemented | — | — | — | — | — | — | — | — |

¹ Checkpoint에는 cumulative epoch time `222.485`초가 있으나 C4 telemetry의 exact end-to-end wall-clock
boundary가 아니므로 resource comparison field에는 사용하지 않는다.

C4-2A actual Kaggle result가 아직 없으므로 quality, resource, environment, model/metadata/package SHA를
작성하지 않는다. Run output의 `comparison_to_baseline.json`과 `experiment_result.json`을 검토한 뒤 이 table을
갱신한다.

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

Candidate artifact는
`artifacts/models/yolo_segmentation/experiments/c4_2a_yolo11n_seg_imgsz1024_seed42/`, export ZIP은
`outputs/packages/c4_2a_yolo11n_seg_imgsz1024_seed42.zip`에 생성된다. ZIP에는 best model/metadata, config,
validation evidence, resource summary, environment와 `SHA256SUMS.txt`만 포함하며 raw dataset이나 cache를 넣지
않는다. Machine result는 `split="val"`, `test_split_used=false`, before/after quality, failure-mode/resource
metrics, Manifest/model/metadata/config SHA와 decision을 명시한다. `experiment_metadata.json`과
`comparison_to_baseline.json`에는 checkpoint-derived historical Baseline reference도 포함하되
`derived_test_metrics_used_for_selection=false`를 유지한다.

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

## 10. Kaggle T4 execution

Repository, derived dataset package와 immutable Baseline runtime bundle을 각각 기본 path에 복원한 뒤 다음 cell을
그대로 실행한다. 이 command는 C2 final test evaluator를 호출하지 않는다.

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

Actual output을 export한 뒤 `package_metadata.json`의 ZIP/model/metadata/config SHA와 ZIP 내부
`SHA256SUMS.txt`를 함께 검증한다. 그 전에는 C4-2A result를 완료 또는 개선으로 표현하지 않는다.
