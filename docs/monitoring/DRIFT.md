# PatchCore Production Drift Detection

## 1. 목적과 책임 경계

STEP 10은 PostgreSQL `inspections` history의 image-level anomaly score와 anomaly 판정 비율을
validation-normal reference와 batch 단위로 비교한다. Request 처리 path에 통계를 추가하지 않으며 실제 model,
MVTec image 또는 ground-truth label 없이도 저장된 production evidence만으로 실행한다.

Monitoring과 drift는 서로 다른 질문을 다룬다.

| 영역 | 질문 | 현재 구현 |
|---|---|---|
| Prometheus/Grafana monitoring | API가 정상이고 빠르게 동작하는가 | request/error/latency/rate/resource-adjacent metric |
| PatchCore drift | 최근 score/output population이 reference와 달라졌는가 | PostgreSQL batch window의 PSI, score shift, anomaly ratio shift |

Request score를 Prometheus label/gauge로 게시하지 않는다. Batch job과 API process의 lifecycle이 다르므로 drift
결과를 ephemeral API metric이나 MLflow experiment metric에 억지로 연결하지 않는다.

> `drift detected`는 `model performance degraded`와 같은 뜻이 아니다. Ground-truth label이 없는 production
> window에서는 accuracy, recall 또는 false-negative 증가를 직접 측정할 수 없다. Drift는 조사 신호이며 자동
> retraining 명령이 아니다.

## 2. Architecture

```text
validation-normal predictions + manifest + model/threshold artifacts
                              ↓ provenance validation
                  immutable reference.json
                              ↓
PostgreSQL inspections -- category/model/time filter
                              ↓ remaining full-lineage validation
                 score statistics + PSI + ratio shift
                              ↓ operational status policy
                    immutable drift.json
```

- `ml/drift/patchcore.py`: reference/report schema, descriptive statistics, PSI와 status policy
- `services/persistence/drift.py`: read-only production window repository
- `pipelines/prepare_patchcore_drift_reference.py`: reference preparation CLI
- `pipelines/analyze_patchcore_drift.py`: PostgreSQL batch analysis CLI

Drift domain logic은 generic utility나 FastAPI route에 두지 않는다.

## 3. Reference population

Reference는 threshold calibration에 사용된 **normal validation prediction만** 허용한다. Manifest의
`split=validation`, `source_split=train`, `label=0`, category와 sample id/metadata를 prediction JSONL과 대조한다.
공식 MVTec `test` score는 reference 또는 policy tuning에 사용하지 않는다.

Threshold, manifest, model metadata, `model.pt`, validation prediction file의 SHA-256을 기존 evaluation validator와
대조한다. 따라서 다른 calibration prediction이나 model artifact를 경로만 바꿔 전달해도 reference를 만들 수
없다.

현재 `metal_nut` validation-normal reference는 22개 sample뿐이다. 재현 가능한 demo baseline에는 사용할 수
있지만 production population의 변동성을 충분히 대표한다고 보장하지 않는다. 실제 운영에서는 동일한
known-good lineage에서 더 큰 승인 window를 확보한 뒤 별도 reference를 재설정해야 한다.

기본 저장 경로:

```text
outputs/drift/reference/patchcore/<reference-id>/reference.json
```

Reference에는 schema/reference id, model/category, model·metadata·manifest·threshold SHA, validation prediction
SHA, raw score values, count, mean/std/min/max/p50/p90/p95/p99, image threshold, reference anomaly ratio, fixed PSI
bin/epsilon과 생성 시각을 저장한다. 경로와 DB credential은 저장하지 않는다. Output directory overwrite는
허용하지 않는다.

```bash
uv run python -m pipelines.prepare_patchcore_drift_reference \
  --validation-predictions outputs/evaluation/patchcore/predictions/<run>/validation/predictions.jsonl \
  --artifact-dir artifacts/models/patchcore/<artifact-id> \
  --thresholds outputs/evaluation/patchcore/thresholds/<threshold-id>/thresholds.json \
  --manifest data/interim/manifests/mvtec_ad_metal_nut.csv \
  --reference-id metal-nut-validation-v1
```

## 4. Production window query와 lineage isolation

분석 CLI는 timezone offset이 있는 `--since`, `--until`을 필수로 받는다. Window는
`created_at >= since AND created_at < until`인 half-open UTC interval이다. Query는 reference의 category,
model name과 model SHA로 먼저 제한하며 `created_at ASC, id ASC`로 고정한다.

조회된 모든 row의 다음 lineage를 reference와 다시 대조한다.

- `model_sha256`
- `artifact_metadata_sha256`
- `manifest_sha256`
- `threshold_artifact_sha256`

같은 model SHA 안에서 threshold, manifest 또는 metadata lineage가 하나라도 섞이면 일부만 골라 분석하지 않고
전체 run을 실패시킨다. Persisted `is_anomaly`도 reference의 strict `score > image_threshold` 계약과 일치해야
한다. 다른 category/model/model SHA row는 query 대상이 아니다. DB failure는 credential이나 원본 DB exception을
report에 기록하지 않고 safe `PersistenceError`로 종료하며 output directory도 만들지 않는다.

```bash
export DATABASE_URL=postgresql+psycopg://<user>:<password>@<host>:5432/<database>

uv run python -m pipelines.analyze_patchcore_drift \
  --reference outputs/drift/reference/patchcore/metal-nut-validation-v1/reference.json \
  --since 2026-08-20T00:00:00+00:00 \
  --until 2026-08-21T00:00:00+00:00 \
  --drift-id metal-nut-20260820
```

## 5. Statistical method

### Score summary와 quantile shift

Reference/current 각각 population standard deviation과 mean, min, max, p50, p90, p95, p99를 계산한다. Report는
current minus reference의 mean, p50, p95 delta를 별도 기록한다. Quantile은 ordered sample에 대한 linear
interpolation으로 결정하므로 input row order에 의존하지 않는다.

### PSI bin과 smoothing

기본 요청 bin 수는 10이다. Reference의 10%~90% quantile을 internal edge로 고정하고 동일 값의 edge는
제거한다. Current data로 bin을 다시 맞추지 않으며 reference range 바깥 값도 첫/마지막 bin에 포함한다. Reference
score가 모두 같으면 그 값의 양쪽에 `max(abs(score), 1) × 1e-9` edge를 두어 constant baseline의 상·하향 변화를
구분한다.

각 aligned bin count에는 기본 `epsilon=1e-6`을 더하고 다음처럼 다시 정규화한다.

```text
p_i = (count_i + epsilon) / (sample_count + epsilon * number_of_bins)
PSI = Σ (current_i - reference_i) * ln(current_i / reference_i)
```

이 symmetric additive smoothing은 empty bin의 0 나눗셈과 log(0)을 방지한다. Score, edge, epsilon과 최종
metric이 non-finite이면 artifact를 만들지 않는다. Reference CLI의 `--psi-bin-count`, `--psi-epsilon`으로 baseline
통계 계약을 명시적으로 바꿀 수 있다.

### Anomaly ratio

Anomaly ratio는 동일한 stored image threshold와 strict `score > threshold`로 계산한다. Ratio 변화는
`abs(current_ratio - reference_ratio)`인 절대 차이다. Reference ratio가 0이어도 reference ratio로 나누지
않으므로 안전하다. Ratio는 output population 변화 신호이지 defect prevalence 또는 model accuracy의 확정값이
아니다.

## 6. Minimum sample과 status policy

기본 minimum current sample count는 30이다. 현재 22개 validation reference의 한계를 인정하면서도, production
판정을 소수 request에 반응시키지 않고 최소한 reference demo 크기보다 큰 window에서 시작하기 위한 보수적 운영
기본값이다. 30건 미만이면 통계를 계산할 수 있어도 status는 항상 `insufficient_data`다.

30건 이상에서는 statistics 계산과 status 정책을 분리한다.

| Signal | stable | warning | drift |
|---|---:|---:|---:|
| PSI | `< 0.10` | `>= 0.10` | `>= 0.25` |
| anomaly ratio absolute delta | `< 0.10` | `>= 0.10` | `>= 0.20` |

두 signal 중 높은 등급을 status로 사용한다. 이 값은 검증된 과학적 보편 기준이 아니라 초기 operational default다.
CLI의 `--minimum-sample-count`, `--psi-warning-threshold`, `--psi-drift-threshold`,
`--anomaly-ratio-warning-threshold`, `--anomaly-ratio-drift-threshold`로 변경할 수 있으며 warning은 drift보다
작아야 한다. Ground truth와 운영 feedback이 쌓이면 category별 sensitivity/false-positive trade-off를 다시
검증해야 한다.

## 7. Drift output schema와 persistence 선택

기본 경로는 다음과 같고 overwrite하지 않는다.

```text
outputs/drift/patchcore/<drift-id>/drift.json
```

주요 field:

- schema/drift id, model/category와 full lineage
- reference id/source/count/summary
- current `since`, `until`, half-open boundary/count/summary
- PSI/current bin counts, reference/current anomaly ratio와 absolute delta
- mean/p50/p95 delta
- 적용된 minimum/PSI/anomaly-ratio policy와 status
- timezone-aware `created_at`

이번 단계에서는 `drift_runs` table을 추가하지 않았다. Immutable JSON 하나가 full distribution statistic,
policy와 lineage를 재현 가능하게 보존하고, 현재 요구에는 Dashboard/API query가 없기 때문이다. Summary row를
DB에 중복 저장하면 migration·retention·JSON과의 consistency 책임만 늘어난다. STEP 12 Dashboard가 latest/history
query를 실제로 요구할 때 summary persistence와 index를 설계한다. Raw score distribution 전체를 DB에 복제하지
않는다.

같은 이유로 `GET /v1/drift/latest` 같은 speculative API도 추가하지 않았다.

## 8. 실행과 scheduling 범위

현재는 명시적 batch CLI만 제공한다. Celery, Airflow, Kafka, cron daemon과 automatic retraining은 없다. 향후
Kubernetes 도입 후 이 CLI를 immutable reference/output storage와 DB secret을 받는 CronJob으로 실행할 수 있다.
동일 drift id의 재실행은 새 id를 사용해야 한다.

## 9. 검증과 향후 범위

Synthetic unit/integration tests는 다음을 model/dataset 없이 검증한다.

- validation-normal reference와 provenance, non-normal/non-finite/overwrite 거부
- identical/shifted distribution, PSI, anomaly ratio와 insufficient sample
- full lineage mismatch, half-open time/category/model filter와 deterministic finite report
- DB failure의 safe error conversion
- SQLite end-to-end query/report와 Docker PostgreSQL synthetic batch smoke

현재는 image-level PatchCore anomaly score/output drift v1이다. Input image statistic, backbone feature/embedding
drift, defect-type drift와 ground-truth performance monitoring은 포함하지 않는다. 향후 embedding drift를 추가할
때는 feature extraction version, sampling/retention, dimensionality reduction, storage 비용과 privacy를 먼저
정의해야 하며 vector DB나 feature store를 선행 도입하지 않는다.
