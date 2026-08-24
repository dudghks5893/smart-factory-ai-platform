# PatchCore FastAPI Serving Core

## 1. 범위

STEP 4-1은 STEP 2에서 생성한 PatchCore artifact와 STEP 3 validation threshold artifact를 하나의
FastAPI process에서 제공한다. STEP 5에서는 성공 prediction을 required PostgreSQL inspection history에
기록한다. Docker, MLflow, monitoring과 anomaly-map visualization은 아직 포함하지 않는다.

```text
Multipart JPEG/PNG
        ↓
FastAPI request validation
        ↓
Process-local inference runtime
        ↓
Artifact preprocessing → device transfer → PatchCore inference
        ↓
score > validation image threshold
        ↓
Inspection transaction commit
        ↓
Inspection UUID + image-level JSON response
```

HTTP route는 model-specific inference를 구현하지 않는다. `services/inference/runtime.py`가 기존
`PatchCoreAdapter`, `PatchCorePreprocessor`와 threshold contract를 결합하고, `services/api`는 transport,
schema와 error mapping만 담당한다.

## 2. Startup lifecycle

FastAPI lifespan startup에서 다음 순서로 required dependency와 runtime을 구성한다.

1. Environment serving configuration 검증
2. Database engine/session factory 생성과 connectivity 확인
3. 요청 device 결정(`auto`: CUDA → MPS → CPU)
4. `metadata.json`과 `thresholds.json` schema 검증
5. Embedded manifest SHA와 실제 metadata/model SHA provenance 검증
6. `model.pt`를 pretrained download 없이 복원
7. Artifact preprocessing과 validation image threshold를 process-local runtime에 보관
8. DB와 runtime 준비가 끝난 뒤 readiness 활성화

Artifact directory/file, threshold, JSON/schema, provenance 또는 명시적 device가 잘못되면 startup이
실패하며 application은 ready 상태가 되지 않는다. Request마다 metadata/model을 다시 읽거나 memory bank를
재구성하지 않는다. Shutdown에서는 SQLAlchemy engine pool을 dispose한다. Table 생성은 startup이 아니라
Alembic migration에서 수행한다.

## 3. Configuration

| Environment variable | Required | Meaning |
| --- | --- | --- |
| `PATCHCORE_ARTIFACT_DIR` | Yes | `model.pt`와 `metadata.json`이 있는 artifact directory |
| `PATCHCORE_THRESHOLDS_PATH` | Yes | validation calibration으로 생성한 `thresholds.json` |
| `DATABASE_URL` | Yes | `postgresql+psycopg://` inspection database URL |
| `MODEL_DEVICE` | No | `auto`, `cpu`, `mps`, `cuda`; default `auto` |
| `MAX_UPLOAD_BYTES` | No | Image file 최대 byte 수; default 10 MiB |

Experiment hyperparameter는 serving environment에 중복하지 않는다. Backbone, layers, preprocessing과
threshold는 artifact files가 source of truth다. `.env.example`은 변수 형식만 제공하며 실제 raw artifact와
output은 Git에 추가하지 않는다.

Local 실행 예:

```bash
export PATCHCORE_ARTIFACT_DIR=artifacts/models/patchcore/<artifact-id>
export PATCHCORE_THRESHOLDS_PATH=outputs/evaluation/patchcore/thresholds/<threshold-id>/thresholds.json
export DATABASE_URL=postgresql+psycopg://<user>:<password>@localhost:5432/<database>
export MODEL_DEVICE=auto

uv run uvicorn services.api.app:app --host 127.0.0.1 --port 8000
```

## 4. Endpoint contract

### `GET /health`

Process liveness만 나타낸다. Model이 아직 ready가 아니어도 process가 request를 처리하면 `200`과
`{"status":"ok"}`를 반환할 수 있다.

### `GET /ready`

Model artifact/threshold가 검증·복원되고 required DB connection이 유효한 경우에만 `200`을 반환한다.
Runtime이 없으면 `503 model_not_ready`, DB가 unavailable이면 `503 database_not_ready`다. Ready response에는
model name, category와 선택된 device를 포함한다.

### `POST /v1/predictions`

`multipart/form-data`의 `image` field로 JPEG(`image/jpeg`) 또는 PNG(`image/png`) 하나를 받는다. Upload를
설정된 최대 크기까지만 읽고 application code에서 별도 temporary file을 생성하지 않는다. Empty,
malformed, MIME/decoded format mismatch와 unsupported media type을 inference 전에 구분해 거부한다.

Response:

```json
{
  "inspection_id": "3c9d9238-5a15-4fe1-9752-b233425663c0",
  "model_name": "patchcore",
  "category": "metal_nut",
  "is_anomaly": true,
  "anomaly_score": 45.0,
  "threshold": 41.19657897949219,
  "comparison_operator": ">"
}
```

Image 판정은 오직 저장된 validation threshold로 `anomaly_score > image_threshold`를 적용한다. 같은 score는
normal이다. Test score, request별 threshold 또는 API magic number는 사용하지 않는다.
성공 inference는 input/model provenance와 함께 commit된 후에만 response를 반환한다.

PatchCore는 defect class classifier가 아니므로 `bent`, `color`, `flip`, `scratch` 같은 defect type을
반환하지 않는다. Anomaly map도 대용량 float JSON으로 반환하지 않으며 향후 heatmap/overlay contract에서
별도로 설계한다.

## 5. Error contract

API error는 다음 envelope를 사용한다.

```json
{
  "error": {
    "code": "invalid_image",
    "message": "Uploaded content is not a valid image."
  }
}
```

주요 code는 `invalid_request`, `empty_image`, `invalid_image`, `unsupported_media_type`,
`unsupported_image_format`, `image_too_large`, `model_not_ready`, `database_not_ready`, `inference_failed`,
`persistence_unavailable`, `inspection_not_found`, `internal_error`다.
Client response에는 traceback, artifact path, model internals와 원래 exception message를 노출하지 않는다.

Inspection schema, transaction과 history API는 `INSPECTION_HISTORY.md`에서 관리한다.

## 6. Concurrency와 worker policy

한 process는 PatchCore model과 memory bank 한 copy를 공유한다. PyTorch module과 accelerator queue의 동시
접근을 검토한 결과, STEP 4-1에서는 runtime instance별 lock으로 inference를 직렬화한다. 전역 lock이나
request별 model copy는 사용하지 않으며 upload parsing과 image decoding은 lock 밖에서 수행한다. Blocking
inference는 FastAPI event loop가 아니라 threadpool에서 실행한다.

Uvicorn worker를 여러 개 실행하면 worker process마다 lifespan이 독립 실행되어 model과 memory bank도
worker마다 한 copy씩 생성된다. 따라서 worker 수는 GPU/host memory 사용량과 함께 결정해야 한다. 단일 GPU
worker의 처리량 확장은 HTTP concurrency만 보고 늘리지 않고 실제 resource benchmark로 검증한다.

## 7. Real artifact smoke

`pipelines.smoke_patchcore_api`는 실제 `model.pt`, `metadata.json`, validation
`thresholds.json`을 lifespan에서 한 번 복원한 뒤, direct `runtime.predict()`가 아니라 FastAPI HTTP
route를 통과해 검증한다. 정상/불량 image는 CLI에서 명시하며 threshold 조정에는 사용하지 않는다.

```bash
uv run python -m pipelines.smoke_patchcore_api \
  --artifact-dir artifacts/models/patchcore/<artifact-id> \
  --thresholds outputs/evaluation/patchcore/thresholds/<threshold-id>/thresholds.json \
  --normal-image <normal-image.png> \
  --anomaly-image <anomaly-image.png> \
  --device cuda
```

Smoke는 `/health`, `/ready`, normal/anomaly `/v1/predictions`의 HTTP status와 response schema를
확인한다. 두 score는 finite여야 하며 normal은 `score <= threshold`와 `is_anomaly=false`, anomaly는
`score > threshold`와 `is_anomaly=true`를 만족해야 한다. Threshold source는 test image가 아니라
normal-only validation calibration artifact다.

### Kaggle real-model smoke 결과

Tesla T4 CUDA 환경에서 실제 `model.pt`, `metadata.json`과 validation `thresholds.json`을 사용한
FastAPI lifecycle smoke가 통과했다.

| 항목 | 결과 |
| --- | --- |
| Status | PASS |
| Model / category / device | `patchcore` / `metal_nut` / `cuda` |
| Image threshold | `41.19657897949219` |
| Normal sample | `metal_nut/test/good/000.png` |
| Normal score / 판정 | `34.763465881347656` / normal |
| Anomaly sample | `metal_nut/test/bent/000.png` |
| Anomaly score / 판정 | `54.36906051635742` / anomaly |

두 sample 모두 저장된 strict `score > threshold` 계약과 일치했다. Smoke sample은 판정 확인에만
사용했으며 threshold를 변경하거나 calibration에 사용하지 않았다.

## 8. HTTP E2E benchmark

STEP 4-2 benchmark는 Starlette가 현재 우선 사용하는 `httpx2` 기반 FastAPI `TestClient`를 사용한다.
실제 uvicorn socket client/server 방식보다 process 관리와 OS network jitter가 적고 application routing,
multipart parsing, upload read, decode, tensor conversion, threadpool dispatch, preprocessing, device transfer,
PatchCore inference, strict threshold, response schema/JSON serialization과 completed ASGI response까지 같은
process에서 재현하기 쉽기 때문이다.

Primary timing boundary는 image bytes가 memory에 준비된 뒤 `client.post()` 직전부터 response가 완전히
반환된 직후까지다. 다음 항목은 포함하지 않는다.

- 매 request의 disk image read
- FastAPI lifespan의 artifact/threshold restore
- warmup request
- 외부 network/TCP round-trip

따라서 이 결과는 localhost와 유사한 in-process application HTTP E2E 기준이며, uvicorn scheduling,
socket/TLS, proxy/load balancer와 remote client network latency는 나타내지 않는다. Production-like external
HTTP latency와 concurrency는 별도 deployment benchmark에서 측정해야 한다.

기본 조건은 request당 image 1장, warmup 10회, manifest의 official test split 전체다. 현재 `metal_nut`
manifest에서는 서로 다른 115개 image가 measured request 115개가 된다. Warmup 뒤에도 같은 process,
FastAPI lifespan, runtime, GPU, model과 threshold를 재사용하며 reload하지 않는다. 별도 `--measured-count`를
주면 manifest 순서의 distinct prefix만 사용하고 근거 없는 request 반복은 만들지 않는다.

```bash
uv run python -m pipelines.benchmark_patchcore_api \
  --dataset-root data/raw/mvtec_ad \
  --manifest data/interim/manifests/mvtec_ad_metal_nut.csv \
  --artifact-dir artifacts/models/patchcore/<artifact-id> \
  --thresholds outputs/evaluation/patchcore/thresholds/<threshold-id>/thresholds.json \
  --output-id <benchmark-id> \
  --device cuda
```

결과는 overwrite 없이 `outputs/benchmarks/api/<benchmark-id>/benchmark.json`에 저장한다. p50/p95/p99,
mean, total timed seconds, requests/second, 성공/실패 request 수와 error rate 외에 runtime version, GPU,
manifest/model/metadata/threshold SHA-256, image payload size, count와 timing boundary flag를 기록한다.
Measured HTTP error response와 transport exception도 elapsed attempt와 failure count에 포함한다.

실제 model/dataset/benchmark output은 크기, 원본 dataset 배포 조건과 실행별 artifact provenance 때문에
Git에 넣지 않으며 `.gitignore` 대상이다.

### Kaggle FastAPI application benchmark 결과

Tesla T4 CUDA 환경에서 실제 PatchCore runtime으로 측정한 FastAPI in-process application-level HTTP
E2E 결과다. Production network latency로 해석하지 않는다.

| 조건 | 값 |
| --- | ---: |
| Benchmark | `patchcore_fastapi_http_e2e` |
| Transport | `in_process_asgi_testclient` |
| Request batch size | 1 |
| Warmup | 10 |
| Measured requests | 115 |
| Successful / failed | 115 / 0 |
| Error rate | 0.0 |

| Metric | 값 |
| --- | ---: |
| p50 | 44.90185999998175 ms |
| p95 | 48.70313200001419 ms |
| p99 | 53.7457109400475 ms |
| Mean | 45.39321613043957 ms |
| Total timed | 5.2202198550005505 sec |
| Throughput | 22.029723497151036 requests/sec |

Provenance:

| Artifact | SHA-256 |
| --- | --- |
| Manifest | `da81db68eadd22421ba2b284ffee85f49d41fcec47d6aadfa6bdb2cae14f285b` |
| Model | `1a2016a6b75377cc5e6bbeee33b3ed2f3a3b4d1cedb2e80236dbcd1da8c28ca9` |
| Threshold artifact | `9e885f2a3b0de29eeb3e04304d5dc9051fb1a9c6831bf820b885760ccd12fe89` |

측정에는 multipart request부터 completed ASGI response까지 포함한다. Disk image loading, artifact
restore, warmup, external network RTT와 uvicorn/socket/TLS/proxy는 포함하지 않는다.

이 실측은 inspection persistence 도입 전 STEP 4 schema v1 결과다. STEP 5 이후 현재 benchmark tooling은
같은 prediction route의 inspection insert/commit도 timing에 포함하고 schema v2
`inspection_persistence_included=true`로 기록한다. 따라서 향후 PostgreSQL 포함 결과는 이 STEP 4 수치와
동일한 latency boundary로 직접 비교하지 않는다.

## 9. Model benchmark와 HTTP benchmark 구분

STEP 3 Tesla T4 p50 21.634ms와 45.114 images/second는 disk image load, artifact restore와 warmup을 제외한
preprocessing-to-synchronization model benchmark다. 이 값은 multipart parsing, upload read, image decode,
threadpool scheduling, JSON serialization, network를 포함하지 않으므로 FastAPI HTTP end-to-end latency가
아니다.

같은 STEP 4 Kaggle session에서 비교용으로 다시 실행한 STEP 3 model benchmark는 p50
`23.024974000009024ms`, p95 `27.069118700012496ms`, p99 `29.552606360020945ms`, mean
`23.637801304351775ms`, throughput `42.30511912357507 images/sec`였다. 이는 FastAPI benchmark와 같은
session의 comparison/reference run이며 기존 공식 STEP 3 p50 21.634ms 결과를 대체하지 않는다.

같은 session의 FastAPI in-process application-level HTTP E2E p50은 `44.90185999998175ms`였다.
Application boundary가 추가되므로 model-serving-oriented latency보다 느린 것은 정상이다.

## 10. Kaggle real-model 실행 순서

아래 순서는 repository clone 이후 같은 `pyproject.toml`과 `uv.lock`로 STEP 4-2 필수 artifact를 만드는
구조다. `KAGGLE_MVTEC_SOURCE`, smoke image와 ID 값은 session input에 맞게 지정하며 Kaggle 절대 경로를
source code에 하드코딩하지 않는다.

```bash
uv sync --locked
export DATABASE_URL=postgresql+psycopg://<user>:<password>@<host>:5432/<database>

# Kaggle base environment의 wrapt 충돌이 있는 session에서만 project venv에 재설치한다.
uv pip install --python .venv/bin/python --reinstall wrapt

mkdir -p data/raw
ln -s "$KAGGLE_MVTEC_SOURCE" data/raw/mvtec_ad

uv run --no-sync python -m pipelines.prepare_mvtec_ad
uv run --no-sync python -m pipelines.train_patchcore \
  --artifact-id "$ARTIFACT_ID" --device cuda
uv run --no-sync python -m pipelines.predict_patchcore \
  --artifact-dir "artifacts/models/patchcore/$ARTIFACT_ID" \
  --output-id "$VALIDATION_ID" --split validation --device cuda
uv run --no-sync python -m pipelines.calibrate_patchcore_thresholds \
  --validation-predictions \
    "outputs/predictions/patchcore/$VALIDATION_ID/predictions.jsonl" \
  --validation-anomaly-maps \
    "outputs/predictions/patchcore/$VALIDATION_ID/anomaly_maps.pt" \
  --artifact-dir "artifacts/models/patchcore/$ARTIFACT_ID" \
  --output-id "$THRESHOLD_ID"
uv run --no-sync alembic upgrade head
uv run --no-sync python -m pipelines.smoke_patchcore_api \
  --artifact-dir "artifacts/models/patchcore/$ARTIFACT_ID" \
  --thresholds \
    "outputs/evaluation/patchcore/thresholds/$THRESHOLD_ID/thresholds.json" \
  --normal-image "$NORMAL_IMAGE" --anomaly-image "$ANOMALY_IMAGE" --device cuda
uv run --no-sync python -m pipelines.benchmark_patchcore_api \
  --artifact-dir "artifacts/models/patchcore/$ARTIFACT_ID" \
  --thresholds \
    "outputs/evaluation/patchcore/thresholds/$THRESHOLD_ID/thresholds.json" \
  --output-id "$API_BENCHMARK_ID" --device cuda
```

`wrapt` 재설치는 Kaggle preinstalled environment 충돌 대응이며 project dependency를 변경하지 않는다.
이 session-only compatibility package가 자동 exact sync에서 제거되지 않도록 이후 command는 `--no-sync`를
사용한다. Dependency 동기화 자체는 첫 `uv sync --locked`에서 완료된다.
STEP 4-2에 필요한 prediction은 threshold calibration용 validation split뿐이다. STEP 3 전체 결과를 함께
재현할 때는 test prediction, evaluation과 offline inference benchmark를 추가로 실행할 수 있지만 real-model
smoke와 HTTP benchmark의 필수 선행 단계는 아니다.

Real-model smoke는 fresh Kaggle session에서 독립적으로 실행되는 단일 cell이 아니다. Prepared manifest,
PatchCore artifact, validation predictions/anomaly maps와 calibrated thresholds가 먼저 생성되어 있어야 한다.

## 11. Local production-line simulator

`pipelines.simulate_inspection_line`은 실제 MVTec `metal_nut` test image를 일정 간격으로 production
`POST /v1/predictions`에 순차 전송하는 local operations smoke CLI다. Image 외 ground-truth label, defect type,
expected result, score 또는 threshold를 request에 넣지 않으며 실제 PatchCore runtime과 PostgreSQL persistence
결과만 사용한다.

```bash
uv run python -m pipelines.simulate_inspection_line \
  --api-base-url http://127.0.0.1:8000 \
  --dataset-root data/raw/mvtec_ad \
  --manifest data/interim/manifests/mvtec_ad_metal_nut.csv \
  --category metal_nut \
  --count 100 \
  --anomaly-source-ratio 0.1 \
  --interval-seconds 1.0
```

Default `production-demo` profile은 100 event의 input source를 normal 90, anomaly 10으로 구성하고 열 번째
event마다 anomaly source를 배치한다. Official test good image가 22장이므로 normal input은 manifest 순서로
deterministically 순환·재사용한다. 이는 prediction 비율을 강제하지 않으며 출력 summary는 input source count와
observed model prediction count를 분리한다.

각 request는 이전 response가 완료된 후 configured interval을 기다리고 다음 image를 전송한다. Queue, retry,
parallel worker와 backpressure는 포함하지 않는다. HTTP/timeout/schema failure는 retry 없이 즉시 run을 중단하고
partial summary를 출력한다. Simulator는 DB에 직접 접근하거나 drift analysis를 실행하지 않는다.

이 workload는 실제 MVTec image와 실제 model inference를 사용하지만 trigger/order/source distribution은 local
simulation이다. 실제 공장 production traffic이나 model evaluation이 아니며 accuracy, precision, recall 또는 F1을
계산하지 않는다. 생성된 inspection은 기존 Dashboard에서 manual `Refresh Data`로 조회한다.

## 12. Bounded queue production-line simulator

`pipelines.simulate_queued_inspection_line`은 B1의 validated deterministic schedule과 single-event HTTP
client를 재사용하되 capture producer와 inference request worker를 분리한다. Main-thread producer가 configured
cadence로 event를 `queue.Queue(maxsize=N)`에 넣고, non-daemon worker 하나가 schedule 순서대로 실제
`POST /v1/predictions`를 호출한다. Production API와 database schema는 변경하지 않는다.

```bash
uv run python -m pipelines.simulate_queued_inspection_line \
  --api-base-url http://127.0.0.1:8000 \
  --dataset-root data/raw/mvtec_ad \
  --manifest data/interim/manifests/mvtec_ad_metal_nut.csv \
  --category metal_nut \
  --count 100 \
  --anomaly-source-ratio 0.1 \
  --capture-interval-seconds 0.02 \
  --queue-size 8
```

두 실행 모드의 interval 경계는 다음과 같다.

- B1 sequential: `HTTP response 완료 → interval → 다음 capture`
- B2 queued: `capture → enqueue → 다음 capture deadline`; inference latency는 producer cadence에 직접
  더해지지 않지만 queue가 full이면 producer가 block된다.

Queue 기본 크기는 8이며 policy는 blocking backpressure와 no-drop이다. Drop-oldest/newest, retry, multiple
worker, parallel HTTP, Redis/Kafka/Celery는 사용하지 않는다. 정상 종료는 producer 완료, queued event 처리,
sentinel, worker join 순서다. HTTP/timeout/schema/duplicate-ID failure는 stop signal을 전달하고 thread를 join한 뒤
non-zero로 종료하므로 queue-full 상태에서도 daemon thread나 deadlock에 의존하지 않는다.

Summary는 requested/enqueued/success/failed와 input source/observed prediction을 분리하고 queue capacity와 최대
depth, producer block count/time, queue wait average/p95, HTTP average, wall-clock과 processing throughput을
출력한다. Configured capture rate와 processing throughput은 서로 다른 값이며 Mac MPS, localhost HTTP와
PostgreSQL을 포함한 결과를 STEP 3 Tesla T4 model benchmark와 직접 비교하지 않는다.

이 simulator는 실제 MVTec image와 실제 production API 경로를 사용하지만 실제 PLC/camera 또는 공장 traffic은
아니다. Queue worker는 DB에 직접 접근하지 않으며 Dashboard 갱신은 기존 manual `Refresh Data`로 확인한다.

## 13. Inspection WebSocket event stream

REST와 WebSocket은 서로 다른 책임을 가진다.

- REST `POST /v1/predictions`: inference와 durable inspection 생성
- REST `GET /v1/inspections`, `GET /v1/inspections/{inspection_id}`: initial state, history, detail와
  reconnect recovery
- WebSocket `/v1/ws/inspections`: 새 inspection의 best-effort live notification

한 prediction은 repository의 SQLAlchemy `session.commit()`이 성공해 committed `Inspection`이 반환된 뒤에만
`inspection.created` event가 만들어진다. Route는 그 event를 FastAPI background task에 등록하므로 WebSocket
send는 HTTP response body 이후 실행되고 prediction request critical path에 포함되지 않는다. Commit이 실패하거나
image/inference validation이 실패하면 background broadcast 자체를 예약하지 않는다.

```json
{
  "schema_version": "1",
  "type": "inspection.created",
  "inspection": {
    "inspection_id": "3c9d9238-5a15-4fe1-9752-b233425663c0",
    "model_name": "patchcore",
    "category": "metal_nut",
    "is_anomaly": true,
    "anomaly_score": 54.369,
    "threshold": 41.19657897949219,
    "comparison_operator": ">",
    "device": "mps",
    "created_at": "2026-08-25T00:00:00Z"
  }
}
```

Event에는 raw image, anomaly map, artifact body, provenance hash와 DB internal field를 넣지 않는다. 상세
provenance는 REST detail에서 조회한다. Client가 보내는 domain command protocol은 없으며 server route는 peer
disconnect를 감지하기 위한 receive loop만 유지한다.

Connection manager는 API process 안에서 client snapshot을 관리하고 per-client timeout으로 send를 동시에
실행한다. 한 slow/broken client는 제거되며 다른 client delivery를 중단하지 않는다. 이 live notification은
durable queue나 exactly-once delivery가 아니다. PostgreSQL만 source of truth이며 disconnect 중 놓친 event는
reconnect 후 REST history reload로 복구한다.

현재 구조는 `uvicorn --workers 1` local validation을 전제로 한다. Multi-process 또는 multi-replica에서 Pod A가
만든 event는 Pod B의 process-local client set에 전달되지 않는다. Production scale에서 cross-process delivery가
필요해지면 external pub/sub 또는 broker를 별도로 설계해야 하며 현재 단계에는 포함하지 않는다.

실제 API에 두 client를 연결해 multi-client delivery, 한 client disconnect 이후 remaining delivery, POST/REST
detail consistency를 검증할 수 있다.

```bash
uv run python -m pipelines.smoke_inspection_websocket \
  --api-base-url http://127.0.0.1:8000 \
  --image data/raw/mvtec_ad/metal_nut/test/good/000.png
```

## 14. Browser-native Live Inspection Monitor

FastAPI application image에 HTML/CSS/vanilla JavaScript asset을 함께 package하고 same-origin `/live/`에
mount한다. UI source는 `apps/live_monitor/`에 두어 transport code와 분리하며 React/Vue/Next.js,
npm build, Streamlit custom component, polling을 추가하지 않는다. Existing Streamlit Dashboard의 history,
drift와 analytical view는 그대로 유지된다.

```text
Browser /live/
  ├─ WebSocket /v1/ws/inspections: best-effort inspection.created notification
  ├─ REST /v1/inspections: initial snapshot and reconnect recovery
  └─ REST /v1/inspections/{id}: selected inspection provenance detail
                         ↓
                PostgreSQL durable source of truth
```

Initial synchronization은 WebSocket을 먼저 연결한다. Connection open 후 incoming event를 buffer하며
REST latest 100 history를 로드하고, history와 buffer를 `inspection_id`로 dedupe해 `created_at`
newest-first로 병합한 다음에만 `LIVE`로 전환한다. REST-first 간격에서 event를 놓치지 않는
contract이다. Same ID overlap/duplicate/reconnect replay는 row를 늘리지 않으며 UUID로 ordering하지
않는다.

Disconnect하면 `RECONNECTING`으로 전환하고 0.5, 1, 2, 4, maximum 5초 bounded exponential
backoff로 다시 연결한다. 매 reconnect는 동일한 WebSocket-first buffering → REST reload →
merge/dedupe → `LIVE` 순서를 반복하므로 disconnect 기간의 gap은 PostgreSQL history에서
복구한다. Browser offline은 `OFFLINE`으로 표시하고 online event에서 recovery를 재시작하며,
page unload에서 socket, fetch와 timer를 정리한다.

WebSocket URL은 current page protocol/host에서 `ws://` 또는 `wss://`로 생성하고 REST는 `/v1/...`
relative URL을 사용한다. UI는 current latest-100 window의 visible/normal/anomaly/anomaly-ratio
KPI와 latest decision, newest-first feed를 보여준다. Timestamp는 backend UTC value를 browser local timezone으로
표시한다. Result는 event의 `is_anomaly`가 source of truth이며 client에서 inference를 재실행하지
않는다. Detail dialog만 existing REST endpoint를 호출해 image/model/metadata/threshold/manifest SHA를
보여준다.

`schema_version == "1"`, `type == "inspection.created"`와 compact inspection field를 client에서 검증하고
unknown/malformed message는 무시한다. Static directory가 없으면 `/live/`만 mount하지 않아 API
boot/readiness를 막지 않으며 production Docker application stage는 asset directory를 명시적으로 copy한다.

현재 WebSocket broadcaster는 single-process best-effort delivery이므로 PostgreSQL을 내구성 계층으로 유지한다.
Production browser exposure에서는 WebSocket/REST authentication, Origin validation, authorization,
TLS/`wss://`와 multi-replica pub/sub을 별도로 설계해야 한다.

2026-08-25 local actual smoke는 PostgreSQL 17.6, Alembic head, actual PatchCore artifact, MPS FastAPI,
browser tab 2개와 B2 queued simulator를 동시에 사용했다. Initial REST sync에서 두 tab이 같은
latest 100 window(90 normal, 10 anomaly)를 복원했고, `count=100`, capture interval `0.05`, queue size
`8`의 actual image run이 100/100 success와 90/10 observed prediction으로 완료되는 동안 두 tab의
latest ID, KPI와 feed가 manual refresh 없이 같이 갱신됐다. 두 window 모두 100 unique ID만
유지했다. 별도 migrated empty PostgreSQL database로 반복한 smoke에서는 두 tab의 visible
count/KPI가 `0/0/0/0.0%` → `100/90/10/10.0%`로 증가하는 것을 확인했다.

API process를 종료하자 두 tab이 `RECONNECTING`으로 전환했고, 같은 DB/artifact로 재시작한
뒤 WebSocket-first reconnect와 REST reload를 거쳐 `LIVE`와 latest 100 unique row를 복원했다.
Full API outage 중에는 POST endpoint도 unavailable이므로 별도 inspection gap을 생성하지 않았다.
Restart 후 persisted normal inspection은 두 tab에 같은 latest ID로 반영됐다. Malformed PNG request는
`400 invalid_image`를 반환했고 DB count `336 → 336`, 두 tab latest ID/count/KPI 불변을 확인했다.
이 smoke는 기능 검증이지 latency/throughput benchmark가 아니다.
