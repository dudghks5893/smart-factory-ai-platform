# Smart Factory AI Operations Dashboard

## 1. 목적과 범위

STEP 12 Dashboard는 품질·운영 담당자가 최근 PatchCore prediction, model lineage와 batch drift 상태를 한 화면에서
확인하는 internal operations UI다. Public customer frontend, 검사 판정 API, PostgreSQL analytics service 또는
Grafana 대체 UI가 아니다.

```text
Browser
  ↓ localhost:8501
Streamlit Dashboard
  ├─ HTTP → FastAPI /v1/inspections → PostgreSQL
  ├─ read-only filesystem → <DRIFT_REPORT_DIR>/<drift-id>/drift.json
  └─ browser link → Grafana
```

Dashboard는 SQLAlchemy Session을 만들거나 persistence query를 복제하지 않는다. Inspection history와 detail은
기존 FastAPI 계약만 사용하며, drift report는 STEP 10 immutable artifact schema를 검증해 읽는다. FastAPI domain
logic과 threshold 판정을 Dashboard에서 다시 수행하지 않는다.

## 2. Technology와 dependency

UI는 Python-only internal tool에 맞춰 Streamlit `1.62.0`을 사용한다. Chart는 Streamlit built-in
`st.line_chart`를 사용하며 Plotly, React/Next.js, Dash, Panel 또는 Gradio를 추가하지 않았다. Dashboard dependency는
uv의 `dashboard` group으로 분리해 API production image에 Streamlit을 설치하지 않는다. Local/CI quality 환경은
`dev`와 `dashboard` default group을 함께 사용한다.

## 3. Inspection data와 filter

`InspectionApiClient`가 다음 existing endpoint만 호출한다. 각 request에는 bounded timeout을 적용한다.

- `GET /v1/inspections?limit=<1..100>&offset=0`
- `GET /v1/inspections/{inspection_id}`

API는 category와 `is_anomaly` filter도 이미 지원하므로 별도 analytics endpoint나 API 변경은 없다. Overview는 선택한
최근 20/50/100건에서 발견된 category를 hard-code 없이 filter option으로 만들고 All/Normal/Anomaly filter를
적용한다. 이는 전체 inspection population aggregate가 아니라 현재 조회·표시된 bounded sample이다.

Connection refused, timeout, 4xx/5xx, oversized payload, invalid UTF-8/JSON과 malformed response는 non-sensitive UI
오류로 변환한다. Traceback이나 API response body는 사용자 화면에 표시하지 않는다. API가 unavailable이어도
Dashboard process와 drift/Grafana 영역은 계속 동작한다.

## 4. KPI와 score chart

상단 KPI 정의:

| KPI | 정의 |
|---|---|
| Recent Inspections | 현재 filter 후 화면에 포함된 inspection 수 |
| Normal Predictions | 현재 sample의 `is_anomaly=false` 수 |
| Anomaly Predictions | 현재 sample의 `is_anomaly=true` 수 |
| Anomaly Ratio | `Anomaly Predictions / Recent Inspections`; empty이면 0 |
| Latest Drift Status | latest valid drift report의 operational status 또는 `NO REPORT` |

Anomaly Ratio는 AI prediction ratio이며 confirmed defect rate나 production ground truth가 아니다. 서로 다른 model
SHA가 섞일 수 있으므로 `Model versions in view`도 표시한다.

Score Trend는 timezone-aware `created_at`을 UTC로 정렬하고 anomaly score와 각 inspection에 저장된 threshold를
함께 그린다. 서로 다른 threshold/model lineage가 섞여도 하나의 고정 threshold line을 가정하지 않고 record별
threshold series가 변화를 그대로 보존한다. 판정 계약은 API/DB에 저장된 strict `score > threshold`다.

## 5. Recent table과 detail

기본 table은 다음만 표시한다.

- Created At (UTC), Category, Anomaly Score, Threshold, Result, Model Name, Device

UUID는 선택 label의 짧은 suffix로만 보이고, 선택 후 existing detail endpoint를 호출해 전체 Inspection ID, Model SHA,
Artifact Metadata SHA, Threshold Artifact SHA와 Manifest SHA를 표시한다. Dashboard-safe response model에는 image
SHA, image size와 content type을 보관하지 않아 presentation layer에서 실수로 노출하지 않는다.

현재 DB에는 raw product image가 없으므로 image preview, defect bounding box, segmentation overlay와 defect type을
표시하지 않는다. PatchCore image-level 결과만으로 없는 defect class를 만들어내지 않는다.

## 6. Drift integration

Configured root 아래 정확한 STEP 10 layout을 읽는다.

```text
<DRIFT_REPORT_DIR>/<drift-id>/drift.json
```

모든 candidate의 schema version, supported status, finite statistic, sample count와 timezone-aware timestamp를 먼저
검증한다. 최신 report는 filesystem mtime이 아니라 `(created_at, drift_id, path)` metadata 순서로 deterministic하게
선택한다. Candidate 하나라도 malformed이면 이전 valid report를 latest처럼 숨기지 않고 해당 relative path와 함께
명시적 error state를 표시한다.

Panel은 status, PSI, reference/current sample count, mean, p95, anomaly ratio, window start/end와 report created time을
표시한다. Status는 `stable`, `warning`, `drift`, `insufficient_data`를 구분한다. Drift report root가 없거나 report가
0개이면 `No drift report available.`을 표시하며 service fatal error가 아니다.

Drift detected는 model accuracy degradation을 증명하지 않는다. Ground truth가 없으므로 current anomaly ratio 역시
confirmed defect rate가 아니다. Dashboard는 retraining이나 model promotion을 자동 실행하지 않는다.

## 7. Dashboard와 Grafana 책임

Dashboard 책임:

- inspection history와 AI prediction 결과
- anomaly score/record별 threshold
- inspection model lineage
- STEP 10 batch drift status

Grafana 책임:

- API request rate, HTTP latency와 error rate
- inference/persistence latency
- persisted prediction metric과 service availability

Dashboard는 Prometheus metric chart를 복제하지 않고 `GRAFANA_URL`로 `Open Grafana Monitoring` link만 제공한다.
이 URL은 container-to-container 주소가 아니라 사용자 browser에서 접근 가능한 URL이어야 한다. MLflow Tracking
Server가 배포되지 않았으므로 가짜 MLflow UI link는 제공하지 않는다.

## 8. Configuration

| Environment variable | Local direct default | Compose contract |
|---|---|---|
| `DASHBOARD_API_BASE_URL` | `http://localhost:8000` | `http://api:8000` |
| `DRIFT_REPORT_DIR` | `outputs/drift/patchcore` | `/runtime/drift` |
| `DRIFT_REPORT_DIR_HOST` | 해당 없음 | `./outputs/drift/patchcore` read-only bind source |
| `GRAFANA_URL` | `http://localhost:3000` | browser-accessible URL 그대로 |
| `DASHBOARD_REQUEST_TIMEOUT_SECONDS` | `5` | `(0, 60]` seconds |
| `DASHBOARD_PORT` | `8501` | host published port |
| `DASHBOARD_ENV_LABEL` | unset | unset; optional single-line environment banner |

API URL과 Grafana URL은 absolute HTTP(S) URL이어야 하며 query/fragment를 받지 않는다. Source code에 local absolute
artifact path나 credential을 넣지 않는다.

## 9. Local Dashboard

Host에서 API와 optional Grafana가 이미 실행 중이면 다음 명령으로 Dashboard를 시작한다.

```bash
make dashboard
```

`make dashboard`가 repository root를 Python import path로 설정하므로 사용자가 `PYTHONPATH`를 직접 입력할 필요가
없다. Dashboard는 `http://127.0.0.1:${DASHBOARD_PORT:-8501}`에서 실행된다. 실제 API/model/PostgreSQL이 없으면
`Inspection API is unavailable`, zero KPI와 no-report 상태를 표시하는 것이 정상이다. Docker Dashboard command와
production API runtime은 이 local entrypoint 설정의 영향을 받지 않는다.

Manual `Refresh Data` button 또는 Streamlit rerun이 refresh boundary다. Background auto-refresh나 초 단위 polling은
없다.

### Browser-native live view

`http://127.0.0.1:8000/live/`는 production Vision API가 same-origin으로 제공하는 별도 Live Inspection
Monitor다. WebSocket notification, latest-100 KPI, latest decision과 reconnect REST recovery에 집중하며
Streamlit의 history analysis, drift, score trend를 대체하지 않는다. Streamlit은 계속 manual refresh
boundary를 유지한다. Live Monitor contract은 `PATCHCORE_API.md`에서 관리한다.

## 10. Portfolio Demo Dashboard

Model artifact, PostgreSQL 또는 Grafana 없이 populated UI를 확인하려면 두 terminal에서 다음을 실행한다.

Terminal 1 — localhost 전용 synthetic API:

```bash
make dashboard-demo-api
```

Terminal 2 — existing Dashboard와 tracked synthetic drift fixture:

```bash
make dashboard-demo
```

Dashboard URL은 `http://127.0.0.1:8501`, demo API URL은 `http://127.0.0.1:8001`이다. Port를 바꿀 때는 두
terminal에 같은 `DASHBOARD_DEMO_API_PORT`를 설정하고 Dashboard에는 `DASHBOARD_PORT`를 설정한다.

```text
Deterministic synthetic inspection fixture
  → dedicated examples.dashboard_demo FastAPI
  → existing InspectionApiClient
  → existing Streamlit Dashboard

Tracked synthetic STEP 10 drift JSON
  → existing validated drift loader
  → existing Streamlit Dashboard
```

Demo는 100건의 timezone-aware UTC inspection을 3분 간격으로 제공한다. Normal 88건은 threshold `41.2` 아래
`25.3–39.2`, anomaly 12건은 threshold 위 `43.4–56.2`에 분포한다. UUID, timestamps, scores와 lineage SHA는
항상 동일하게 생성된다. Drift fixture는 `warning`, PSI `0.17`, reference/current mean `30/35`, p95 `36/44`,
anomaly ratio `0.02/0.12`를 표시한다.

화면 상단의 `DEMO — SYNTHETIC DATA` warning banner는 `DASHBOARD_ENV_LABEL`이라는 generic opt-in 표시다. 설정하지
않으면 기존 UI에는 banner가 나타나지 않는다. 이 demo는 local portfolio visualization 전용이며 factory data,
production result, STEP 3/4 benchmark evidence 또는 실제 drift evidence가 아니다. Raw image, defect class, fake
Grafana metric, credential과 production database write는 포함하지 않는다. Public deployment 대상으로 사용하지 않는다.

## 11. Docker 실행

Compose Dashboard만 build/start할 수 있다. API artifact가 없는 환경에서도 Dashboard 자체 startup과 unavailable
state를 검증할 수 있으며 Dashboard failure는 API startup dependency가 아니다.

```bash
make dashboard-build
make dashboard-up
curl --fail http://localhost:8501/_stcore/health
make dashboard-down
```

Compose는 host drift root만 `/runtime/drift:ro`로 mount한다. `outputs/` 전체를 writable mount하거나 image에 넣지
않는다. Dashboard container는 UID/GID 10001 non-root, read-only root filesystem, writable `/tmp` tmpfs로 실행한다.
`HOME=/tmp/dashboard-home`, disabled file watcher와 disabled usage telemetry 설정으로 production container에서 source
watch와 home-directory write를 피한다.

## 12. Security와 deployment boundary

이 Dashboard는 local/internal access용이며 anonymous public internet exposure에 적합하지 않다. Production에서는
private network와 SSO/IAP 또는 authenticated reverse proxy, TLS, authorization과 audit policy를 별도로 적용해야
한다. 이번 단계는 auth system을 만들지 않는다.

STEP 11 Kubernetes/GCP manifest에는 Dashboard workload를 추가하지 않았다. 실제 GKE 배포 시 API와 Dashboard를
별도 workload로 두고 browser access, IAP, resource/probe와 read-only drift artifact delivery를 함께 설계해야 한다.
Local Docker health는 Streamlit 1.62.0의 `/_stcore/health` endpoint를 사용한다.

## 13. Verification scope

Deterministic tests는 response parsing, filter/KPI/empty/mixed sample, record별 threshold trend, safe projection,
API timeout/error/malformed response, four drift statuses, latest selection, missing/malformed report를 검증한다.
Streamlit AppTest는 title, unavailable API와 missing drift의 graceful UI를 확인한다. CI의 existing Docker job은 API
runtime과 별도로 lightweight `dashboard-runtime` target도 Linux arm64에서 build한다.

Synthetic demo tests는 100건/88 normal/12 anomaly, unique UUID, UTC ordering, score-threshold consistency, list/detail/
filter/pagination endpoint, STEP 10 drift schema, warning values, lineage 일치와 opt-in banner를 검증한다. CI에서는
Streamlit/demo server를 상시 실행하지 않고 deterministic TestClient/AppTest contract만 실행한다.

2026-08-21 local verification에서는 Streamlit 1.62.0 server를 직접 시작해 `/_stcore/health`의 `ok`와 application
HTML response를 확인했다. Apple Silicon Docker에서 최종 `dashboard-runtime` image를 실제 build/start했으며
container health `healthy`, runtime user `app`, read-only root filesystem `true`, `/runtime/drift` writable `false`를
확인했다. 같은 최종 image 안에서 API가 없는 주소와 empty drift root로 AppTest를 실행해 unavailable/no-report UI가
exception 없이 렌더링되는 것도 확인했다. 실제 PostgreSQL inspection data와 real drift artifact를 연결한 browser
E2E는 운영 artifact가 준비된 환경에서 별도로 수행해야 한다.
