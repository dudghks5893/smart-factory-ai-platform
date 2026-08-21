# Prometheus / Grafana Application Monitoring

## 1. 목적과 architecture

STEP 9은 FastAPI serving process의 availability, traffic, latency, prediction 결과와 persistence 상태를
관측한다. Model input/output distribution drift 판정은 수행하지 않는다.

```text
FastAPI
  ├── app-local CollectorRegistry
  ├── HTTP ASGI middleware
  └── inference/persistence domain instrumentation
             ↓ GET /metrics
Prometheus 3.12.0
             ↓ provisioned datasource
Grafana 13.1.0 — Smart Factory AI — API Overview
```

Prometheus와 Grafana는 API가 의존하는 component가 아닌 optional observer다. Monitoring container가 없거나
실패해도 PostgreSQL → migration → API startup dependency는 바뀌지 않는다.

## 2. Metric catalog

Metric 이름과 label schema는 dashboard/향후 alert가 의존하는 public operational contract로 취급한다.

| Metric | Type | Labels | 의미 |
|---|---|---|---|
| `smartfactory_http_requests_total` | Counter | `method`, `route`, `status_code` | 완료된 application request 수 |
| `smartfactory_http_request_duration_seconds` | Histogram | `method`, `route` | FastAPI application-level HTTP 처리 시간 |
| `smartfactory_predictions_total` | Counter | `category`, `result` | persistence까지 성공한 `normal`/`anomaly` prediction 수 |
| `smartfactory_inference_duration_seconds` | Histogram | `model_name`, `category`, `device` | `runtime.predict(...)` 호출 시간 |
| `smartfactory_persistence_duration_seconds` | Histogram | `operation` | inspection persistence 시도 시간 |
| `smartfactory_persistence_errors_total` | Counter | `operation` | stable persistence operation failure 수 |
| `smartfactory_model_info` | Info | model/category/device/model SHA | process-local model identity |

HTTP histogram bucket은 5 ms부터 5 s까지 두어 기존 약 44.9 ms application benchmark와 persistence 추가 후
latency를 모두 포괄한다. Persistence histogram은 1 ms부터 2.5 s까지 더 세밀하게 구성한다. Grafana는 bucket을
`histogram_quantile`로 집계해 p50/p95/p99를 계산한다.

별도 application error counter는 추가하지 않았다. HTTP 실패는 status code label로 이미 관측되며, DB stage만
operation이 명확하고 대응 가치가 있어 별도 persistence error counter를 둔다. Readiness gauge도 추가하지
않았다. Prometheus `up`이 scrape availability를 제공하고 `/metrics` scrape가 DB/model health query를 반복하지
않게 하기 위함이다. `/ready` 계약은 그대로 유지된다.

## 3. Label과 cardinality policy

- Route는 raw URL이 아니라 FastAPI route template을 사용한다.
- `/v1/inspections/<UUID>`는 `/v1/inspections/{inspection_id}` 하나로 집계한다.
- Unmatched 404 path는 모두 `unmatched`로 집계한다.
- 알려지지 않은 HTTP method token은 `OTHER`로 집계한다.
- `/metrics` scrape 자체는 API traffic/latency를 왜곡하지 않도록 HTTP metric에서 제외한다.
- `inspection_id`, filename, image SHA, score, threshold, path, exception class/message와 arbitrary user input은
  label로 사용하지 않는다.
- Model SHA는 deployment당 종류가 제한된 static `smartfactory_model_info`에서만 노출한다. Manifest/artifact/
  threshold SHA를 다른 metric에 반복하지 않는다.

Request별 anomaly score를 label이나 gauge로 게시하지 않는다. Score distribution은 STEP 10 batch CLI가
persisted inspection history를 이용해 분석하며 계약은 `docs/monitoring/DRIFT.md`에서 관리한다.

## 4. 측정 경계

### HTTP duration

ASGI request 진입부터 completed response까지 측정한다. Routing, upload read, decode, inference, persistence,
exception handler와 response serialization을 포함한다. External network RTT, proxy/TLS와 client-side 시간은
포함하지 않는다.

### Inference duration

이미 decode된 tensor를 `runtime.predict(...)`에 전달한 시점부터 반환 또는 failure까지 측정한다. PatchCore
preprocessing, device transfer, model execution, postprocessing과 strict threshold는 포함하지만 multipart read,
image decode, persistence와 HTTP serialization은 포함하지 않는다.

### Persistence duration

현재 핵심 write path인 `repository.create(...)`의 validation, insert, flush와 commit을 포함한다. Image decode와
inference는 제외한다. Detail/history read latency는 이번 단계에서 계측하지 않아 operation label을 불필요하게
늘리지 않았다.

## 5. `/metrics`와 registry lifecycle

`GET /metrics`는 Prometheus text exposition을 반환하며 business OpenAPI schema에서는 숨긴다. 각
`create_app()` 호출이 독립 `CollectorRegistry`와 collector를 생성하므로 한 test process에서 여러 app을 만들어도
`Duplicated timeseries`가 발생하지 않는다. Default global registry는 사용하지 않는다.

현재 local/internal monitoring scope에서는 endpoint authentication을 추가하지 않았다. `/metrics`에는 request
payload나 credential이 없지만 traffic/model metadata가 있으므로 production에서는 public ingress에 노출하지
말고 private network, service perimeter 또는 authenticated metrics proxy 안에 둬야 한다.

Docker runtime은 worker 1개다. Registry와 counter가 process-local이므로 worker를 늘리면 각 process의 metric을
단일 endpoint에 합치는 Prometheus multiprocess mode, lifecycle directory와 worker cleanup 설계를 먼저 추가해야
한다. 현재 구성을 그대로 multi-worker로 확장하면 집계가 불완전하다.

## 6. Prometheus

Compose는 `prom/prometheus:v3.12.0`을 사용한다. `monitoring/prometheus/prometheus.yml`은 15초 scrape/evaluation
interval과 다음 target만 가진다.

```text
job: smartfactory-api
target: api:8000
path: /metrics
```

Configuration은 read-only bind mount이고 TSDB는 `prometheus_data` named volume에 보존한다. node exporter,
postgres exporter와 cAdvisor는 이번 application monitoring 범위에 없다.

## 7. Grafana dashboard

Compose는 `grafana/grafana:13.1.0`을 사용한다. Datasource와 dashboard provider는 YAML로, dashboard JSON은
repository file로 provisioning한다. 수동 UI 설정은 필요하지 않다.

`Smart Factory AI — API Overview` dashboard panels:

1. API Request Rate
2. HTTP Error Rate
3. HTTP p50
4. HTTP p95
5. HTTP p99
6. Prediction Rate
7. Anomaly Ratio
8. Inference p95
9. Persistence p95
10. Prometheus Target Up

Rate는 `rate(...[5m])`, latency는 `histogram_quantile`, error/anomaly ratio denominator는 `clamp_min(...,
1e-9)`을 사용한다. Anomaly ratio는 service가 성공적으로 저장한 prediction 중 anomaly result 비율이지 model
quality metric이나 drift 판정이 아니다.

## 8. Docker lifecycle, credentials와 volume

```bash
make monitoring-config-check
make monitoring-up
make monitoring-down
```

- Prometheus: `http://localhost:${PROMETHEUS_PORT:-9090}`
- Grafana: `http://localhost:${GRAFANA_PORT:-3000}`
- Grafana credential: `.env`의 `GRAFANA_ADMIN_USER`, `GRAFANA_ADMIN_PASSWORD`

Repository default credential은 local example일 뿐이다. Anonymous access와 user signup은 꺼져 있다. Production은
managed secret과 private network policy를 사용해야 한다. Local host port publish도 development 편의를 위한
설정이다.

`monitoring-down`은 container를 stop하고 `prometheus_data`/`grafana_data`를 보존한다. `make docker-down`도 named
volume을 보존한다. `make docker-clean-volumes`는 PostgreSQL뿐 아니라 Prometheus와 Grafana history/dashboard DB
volume까지 삭제하는 destructive operation이다.

## 9. Monitoring과 drift 책임

Monitoring은 service health/performance를 다룬다.

- availability와 HTTP error rate
- HTTP/inference/persistence latency
- request/prediction rate
- operational anomaly ratio

STEP 10 drift는 일정 기간의 persisted PatchCore score/output distribution 변화를 별도 batch process에서
분석한다. Request별 score를 Prometheus label로 보내거나 anomaly ratio 변동만으로 drift라고 판정하지 않는다.
Input/feature/embedding drift는 현재 범위가 아니다.

## 10. 현재 한계와 향후 확장

- TestClient fake runtime으로 실제 exposition과 metric increment를 검증한다.
- Apple Silicon Docker에서 Prometheus 3.12.0 `promtool check config`와 healthy endpoint를 확인했다.
- Grafana 13.1.0 DB health, Prometheus datasource와 10-panel dashboard provisioning을 API로 확인했다.
- Startup log에 provisioning/plugin error가 없고 container/network 정리 후 named volume이 보존됨을 확인했다.
- Real model artifact가 없는 local session에서는 production API target scrape가 `up=0`인 것이 정상이다.
- 실제 trained model API scrape와 production network/security는 배포 환경에서 별도 검증해야 한다.
- 향후 근거가 쌓이면 error rate, p95 latency, readiness down, anomaly ratio 급변 alert를 추가할 수 있다.
- PostgreSQL/node/container 내부 metric은 필요성이 확인될 때 exporter를 별도 검토한다.
