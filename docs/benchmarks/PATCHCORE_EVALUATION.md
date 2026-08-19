# PatchCore Evaluation and Threshold Calibration

## 1. 범위

STEP 3-1은 STEP 2에서 저장한 raw image score와 anomaly map을 사용해 threshold를
calibration하고 test metric을 계산한다. 모델 inference를 다시 수행하거나 test score로
threshold를 이동시키지 않는다.

```text
Normal-only validation predictions
        ↓
max_normal_validation calibration
        ↓
thresholds.json
        ↓
Test predictions + ground-truth masks
        ↓
metrics.json + per_defect_metrics.json
```

## 2. Threshold policy

현재 validation split은 정상 sample만 포함하므로 F1-max calibration을 사용하지 않는다.
첫 baseline은 다음 conservative operating point를 사용한다.

- Strategy: `max_normal_validation`
- Image threshold: validation normal image score의 최댓값
- Pixel threshold: validation normal anomaly map 전체 pixel score의 최댓값
- Comparison operator: `score > threshold`

Strict `>`를 사용하므로 threshold를 만든 validation maximum 자체는 anomaly로 판정하지 않는다.
이 threshold는 첫 baseline operating point이며 final production calibration이 아니다.

## 3. Test leakage 방지

Calibration과 evaluation은 서로 다른 CLI와 함수로 분리한다.

- `pipelines.calibrate_patchcore_thresholds`: validation prediction만 읽고 `thresholds.json` 생성
- `pipelines.evaluate_patchcore`: test prediction과 기존 `thresholds.json`만 읽고 metric 생성

Calibration은 모든 prediction의 split이 `validation`이고 label이 0인지 검증한다. Evaluation은
모든 prediction의 split이 `test`인지 검증하며 threshold 계산 함수를 호출하지 않는다. 양쪽 모두
manifest, artifact metadata, `model.pt`, prediction JSONL과 anomaly map SHA-256 provenance를 확인한다.

## 4. Metrics

Image level:

- AUROC
- Fixed-threshold precision, recall, F1
- TP, TN, FP, FN
- Normal/anomaly support

Pixel level:

- AUROC
- Fixed-threshold precision, recall, F1
- TP, TN, FP, FN

AUROC는 scikit-learn의 표준 implementation을 사용하며 두 class가 모두 있어야 한다. Precision,
recall 또는 F1 denominator가 0이면 명시적으로 `0.0`을 기록한다. NaN과 Infinity input/output은
저장 전에 거부한다.

Ground-truth mask는 `MVTecManifestDataset`으로 읽고 기존 `PatchCorePreprocessor`의 resize,
nearest interpolation, center crop, binary mask contract를 그대로 사용해 anomaly map과 정렬한다.

## 5. Per-defect diagnostics

고정된 image threshold를 모든 defect에 동일하게 적용한다.

- Anomaly defect: sample count, detected count, recall
- `good`: sample count, false-positive count, false-positive rate

Per-defect 값은 diagnostics일 뿐 threshold tuning에 사용하지 않는다.

## 6. Output contracts

`thresholds.json`은 schema version, strategy/operator, image/pixel threshold, validation sample/pixel
count, artifact metadata, manifest/model/prediction SHA-256와 생성 시각을 기록한다.

`metrics.json`은 evaluation schema version, threshold artifact 정보, 전체 provenance, image/pixel
metric, sample/pixel count와 per-defect diagnostics를 기록한다. `per_defect_metrics.json`은 같은
diagnostics의 별도 inspectable view다. 기존 output directory나 threshold file을 overwrite하지 않는다.

## 7. 실행

Validation threshold calibration:

```bash
uv run python -m pipelines.calibrate_patchcore_thresholds \
  --validation-predictions outputs/predictions/patchcore/<validation-id>/predictions.jsonl \
  --validation-anomaly-maps outputs/predictions/patchcore/<validation-id>/anomaly_maps.pt \
  --artifact-dir artifacts/models/patchcore/<artifact-id> \
  --output-id <threshold-id>
```

Fixed-threshold test evaluation:

```bash
uv run python -m pipelines.evaluate_patchcore \
  --test-predictions outputs/predictions/patchcore/<test-id>/predictions.jsonl \
  --test-anomaly-maps outputs/predictions/patchcore/<test-id>/anomaly_maps.pt \
  --thresholds outputs/evaluation/patchcore/thresholds/<threshold-id>/thresholds.json \
  --artifact-dir artifacts/models/patchcore/<artifact-id> \
  --output-id <evaluation-id>
```

## 8. Kaggle baseline evaluation 결과

Kaggle Linux GPU runtime에서 동일 artifact, validation calibration과 official test split으로 실행한 결과다.
Threshold는 test metric을 보기 전에 normal-only validation에서 고정했다.

- Strategy: `max_normal_validation`
- Comparison operator: `score > threshold`
- Image threshold: `41.19657897949219`
- Pixel threshold: `40.1362419128418`

Image level:

| Metric | Value |
| --- | ---: |
| AUROC | 0.9975562072336266 |
| Precision | 1.0 |
| Recall | 0.989247311827957 |
| F1 | 0.9945945945945946 |
| TP / TN / FP / FN | 92 / 22 / 0 / 1 |

Pixel level:

| Metric | Value |
| --- | ---: |
| AUROC | 0.9824857431023005 |
| Precision | 0.8339393856698755 |
| Recall | 0.8346198271398415 |
| F1 | 0.8342794676622264 |

Per-defect image detection은 bent 25/25, color 21/22, flip 23/23, scratch 23/23이며 good false
positive는 0/22다. 이 결과는 model quality baseline이고 inference latency benchmark와는 별도 계약이다.
Tesla T4 inference 성능 결과는 `PATCHCORE_INFERENCE_BENCHMARK.md`에서 관리한다.
