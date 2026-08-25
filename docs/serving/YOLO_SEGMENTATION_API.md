# YOLO Segmentation FastAPI Serving

## 1. Scope

C2-4A는 C2-3의 `YoloSegmentationAdapter`를 기존 Vision API lifespan에 선택적으로 연결하고
`POST /v1/known-defects`로 known-defect instance를 제공했다. C2-4B는 이 독립 YOLO inference를
PostgreSQL parent/child row로 저장하고 REST recovery와 전용 WebSocket notification을 추가한다. Route는
project-owned runtime result만 API schema로 변환하며 Ultralytics `Results`를 직접 해석하지 않는다.

이 endpoint는 PatchCore의 anomaly score와 자동 결합하지 않으며 Live Monitor를 변경하거나
`PASS`/`REJECT`/`REVIEW` 판정을 만들지 않는다. PatchCore row와 YOLO row는 별도 durable inference
result이고 최종 결합은 C3 Decision Engine 범위다.

```text
                         ┌─ POST /v1/predictions ── PatchCore ── anomaly score
Multipart JPEG/PNG ──────┤
                         └─ POST /v1/known-defects ─ YOLO ─ commit parent + children
                                                                 │
                                                  ┌──────────────┴──────────────┐
                                                  ↓                             ↓
                                           REST history/detail       known_defect.created
```

## 2. Configuration and enablement

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `YOLO_SEGMENTATION_ENABLED` | `false` | `true`/`false`로 optional runtime 활성화 |
| `YOLO_SEGMENTATION_ARTIFACT_DIR` | 없음 | `model/model.pt`, metadata를 포함한 runtime bundle |
| `YOLO_SEGMENTATION_DEVICE` | `auto` | `auto`, `cpu`, `mps`, `cuda` |
| `YOLO_SEGMENTATION_CONFIDENCE` | `0.25` | Server-owned diagnostic operating point |

Disabled이면 YOLO artifact가 없는 기존 PatchCore 환경과 CI가 그대로 시작되고 known-defect endpoint는
`503 known_defect_model_disabled`를 반환한다. Enabled이면 artifact directory가 필수이며 config, artifact,
explicit device 또는 model restore가 잘못되면 startup을 실패시킨다. Silent disable이나 CPU fallback은
사용하지 않는다.

`0.25`는 C2-2/C2-3 관찰을 재현하는 **diagnostic confidence**다. Validation-only calibration을 거친
production/reject/quality threshold가 아니며 client request로 변경할 수 없다.

## 3. Lifecycle and readiness

FastAPI lifespan은 required database, PatchCore repository와 known-defect parent/child schema를 확인한 뒤
PatchCore와 enabled YOLO artifact를 복원한다. 성공한 process는 같은 `YoloSegmentationAdapter`를 모든
request에서 재사용하며 request마다 `YOLO(model.pt)`를 호출하지 않는다. Application startup이 migration을
실행하거나 table을 자동 생성하지 않는다.

`GET /health`는 기존 process liveness다. `GET /ready`의 response body는 기존 PatchCore identity contract를
유지한다. YOLO가 disabled이면 readiness 의미도 기존과 같다. Enabled이면 YOLO restore 성공이 aggregate
readiness의 필수 조건이며 startup failure 또는 runtime loss는 ready 상태가 아니다.
Database 또는 두 persistence schema가 query 불가능해도 ready 상태가 아니다.

## 4. Request and response

`POST /v1/known-defects`는 `multipart/form-data`의 `image` field로 JPEG 또는 PNG 한 장을 받는다.
`MAX_UPLOAD_BYTES` 제한, media type, decoded format, empty/truncated/malformed image 정책은
`POST /v1/predictions`와 공유한다.

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/known-defects \
  -F 'image=@data/raw/mvtec_ad/metal_nut/test/bent/000.png;type=image/png'
```

Response는 original image 좌표계의 bbox와 compact mask area만 포함한다.

```json
{
  "inspection_id": "ba3036b4-00f8-4a88-9a45-b7915548140b",
  "model": {
    "name": "yolo11n-seg.pt",
    "task": "segment",
    "category": "metal_nut",
    "device": "mps"
  },
  "image": {"width": 700, "height": 700},
  "diagnostic_confidence": 0.25,
  "inference_ms": 52.004792,
  "instances": [
    {
      "class_id": 0,
      "class_name": "bent",
      "confidence": 0.9574589133262634,
      "box": {
        "x_min": 248.28866577148438,
        "y_min": 33.24338912963867,
        "x_max": 392.6673278808594,
        "y_max": 132.72903442382812
      },
      "mask": {"pixel_count": 9478, "area_ratio": 0.019342857142857142}
    }
  ]
}
```

700x700 boolean mask, base64 PNG, RLE 또는 polygon은 JSON에 넣지 않는다. `pixel_count`와
`area_ratio`만 반환하며 향후 spatial payload는 별도 versioned endpoint로 설계한다. Detection이 없으면
`instances`는 빈 list다.

## 5. Persistence schema and transaction

Migration `20260826_01`은 기존 `inspections` table을 변경하지 않고 다음 table을 additive하게 만든다.

| Table | Purpose | Stored fields |
| --- | --- | --- |
| `known_defect_inspections` | YOLO inference 1회당 parent 1개 | UUID/time, model/task/category/device, diagnostic confidence, inference time, image dimensions/SHA, model/metadata/dataset hashes, instance count |
| `known_defect_instances` | Parent의 0..N compact children | UUID/FK, `instance_index`, class/confidence, bbox, mask pixel count/area ratio |

Parent와 모든 child는 request별 SQLAlchemy Session 한 transaction에서 flush 후 commit한다. Child constraint나
commit이 실패하면 parent까지 rollback하고 HTTP success와 WebSocket event를 만들지 않는다. Good image의
empty prediction도 `instance_count=0` parent는 저장하고 child는 만들지 않는다. 동일 image bytes의 반복
request는 deduplicate하지 않으며 서로 다른 inspection UUID를 가진다.

`instance_index`와 `(inspection_id, instance_index)` unique constraint로 inference order를 보존한다. Child
FK는 parent를 참조하고 `ON DELETE CASCADE`를 사용한다. History query용 parent `created_at` index와 detail
query용 child `inspection_id` index만 둔다. 현재 query requirement가 없는 class index는 추가하지 않았다.

Database에는 upload bytes, filename/source path, raw boolean mask, polygon 또는 Ultralytics result를 저장하지
않는다. `image_sha256`가 input lineage이고 model checkpoint, artifact metadata, dataset Manifest와 semantic
fingerprint SHA-256은 runtime에서 이미 검증된 provenance를 사용한다.

## 6. REST history and detail

- `GET /v1/known-defects`: parent summary를 `created_at DESC, id DESC`로 조회한다. `limit` 기본 20, 최대
  100이며 `offset`, `returned_count`, `has_more`를 제공한다.
- `GET /v1/known-defects/{inspection_id}`: parent provenance와 모든 child를 `instance_index ASC`로 반환한다.
  Unknown UUID는 `404 known_defect_inspection_not_found`다.

History는 parent의 stored `instance_count`만 조회하고 children을 hydrate하지 않으므로 N+1 query가 없다.
Detail만 parent query와 ordered child query를 같은 read Session에서 수행한다. ORM object를 직접 serialize하지
않고 project-owned domain/Pydantic schema로 변환한다.

## 7. WebSocket notification and recovery

기존 browser Live Monitor는 `/v1/ws/inspections`에서 `inspection.created`만 가정한다. Unknown event를 현재
무시하더라도 기존 channel에 새 domain을 섞지 않기 위해 C2-4B는 `/v1/ws/known-defects`를 별도로 제공한다.
두 channel은 connection snapshot, per-client timeout, broken-client 격리 정책을 generic infrastructure로
공유하지만 connection set과 event type은 목적별 instance로 분리한다.

```json
{
  "schema_version": "1",
  "type": "known_defect.created",
  "inspection": {
    "inspection_id": "ba3036b4-00f8-4a88-9a45-b7915548140b",
    "model_name": "yolo11n-seg.pt",
    "category": "metal_nut",
    "device": "mps",
    "diagnostic_confidence": 0.25,
    "instance_count": 3,
    "classes": ["bent", "scratch"],
    "created_at": "2026-08-26T00:00:00Z"
  }
}
```

Repository commit이 성공해 durable domain object를 반환한 뒤에만 event를 만들고 HTTP response 이후
background broadcast를 예약한다. Event는 compact summary이며 image, raw mask, bbox list와 provenance
hash를 포함하지 않는다. WebSocket은 single-process best-effort notification이지 durable queue나
exactly-once delivery가 아니다. PostgreSQL이 source of truth이고 reconnect 중 누락한 event는 REST
history/detail로 복구한다. C2-4C는 이 별도 channel을 소비하되 이번 단계에서는 Live Monitor UI를 변경하지
않는다.

## 8. Error contract

기존 safe error envelope를 재사용한다. 주요 code는 다음과 같다.

| HTTP | Code | Condition |
| ---: | --- | --- |
| 400 | `empty_image`, `invalid_image` | Empty 또는 decode 불가능한 upload |
| 413 | `image_too_large` | `MAX_UPLOAD_BYTES` 초과 |
| 415 | `unsupported_media_type`, `unsupported_image_format` | Media/decoded format 불일치 |
| 503 | `known_defect_model_disabled` | Optional YOLO가 disabled |
| 503 | `known_defect_model_not_ready` | Enabled runtime이 ready가 아님 |
| 503 | `persistence_unavailable` | Parent/child transaction 또는 query failure |
| 500 | `inference_failed` | Runtime inference failure |

Response에는 traceback, artifact path, accelerator detail 또는 원래 exception message를 노출하지 않는다.

## 9. Concurrency and process policy

Image read/decode는 FastAPI event loop에서 수행하고 blocking inference는 threadpool로 보낸다. C2-3 adapter의
instance lock이 같은 process의 shared Ultralytics model 접근을 직렬화하며 request별 mutable result를
공유하지 않는다. 현재 local/Compose Uvicorn contract는 `workers=1`이다.

여러 worker를 사용하면 각 process가 독립 lifespan과 model copy를 가지므로 worker마다 PatchCore memory
bank와 YOLO weight가 다시 적재된다. Worker 수는 GPU/MPS memory와 실제 concurrency benchmark 없이
늘리지 않는다.

## 10. C2-4A actual macOS MPS API smoke

Actual ignored runtime bundle과 `model.pt` SHA-256
`594003121b0e071c47d68c3e53c10f438dcec18b5b56b4e5d8831d64001192bd`를 사용했다. PatchCore는 같은
FastAPI lifespan에서 CPU로 복원하고 YOLO는 MPS, diagnostic confidence `0.25`로 실행했다. 네 request는
모두 HTTP 200이고 response device는 `mps`였다.

| Image | Instances / classes | Confidence | Mask pixels | Inference | In-process HTTP E2E |
| --- | --- | --- | --- | ---: | ---: |
| `good/000.png` | 0 | - | - | 742.323 ms cold | 770.959 ms cold |
| `bent/000.png` | 3 / bent, bent, scratch | 0.957459, 0.443082, 0.388407 | 9478, 6850, 649 | 52.005 ms | 66.095 ms |
| `color/000.png` | 1 / color | 0.955443 | 2204 | 39.801 ms | 51.811 ms |
| `scratch/000.png` | 1 / scratch | 0.662739 | 8221 | 34.623 ms | 46.163 ms |

Instance count, class, confidence, bbox와 mask pixel count는 C2-3 direct-runtime MPS observation과
일치했다. 이는 semantic smoke이지 accuracy나 production performance benchmark가 아니다. 첫 request의
cold MPS graph 준비도 포함된다.

`inference_ms`는 adapter model call 경계이고 HTTP 값은 memory에 준비된 image bytes에 대한 FastAPI
TestClient `client.post()`부터 completed ASGI response까지다. 후자는 multipart, routing, upload read,
decode, schema validation/serialization과 inference를 포함하지만 disk loading, artifact restore, external
network RTT, uvicorn socket, TLS와 proxy는 제외한다.

```bash
export DATABASE_URL=sqlite+pysqlite:////tmp/smartfactory-yolo-api.db

uv run python -m pipelines.smoke_yolo_segmentation_api \
  --patchcore-artifact-dir artifacts/runtime/patchcore/<artifact-id>/model \
  --patchcore-thresholds artifacts/runtime/patchcore/<artifact-id>/thresholds/thresholds.json \
  --yolo-artifact-dir artifacts/runtime/yolo_segmentation/<artifact-id> \
  --yolo-device mps \
  --patchcore-device cpu \
  --confidence 0.25
```

Database schema는 명령 전에 Alembic으로 준비되어 있어야 한다. Smoke summary는 ignored
`outputs/analysis/yolo_segmentation/api_smoke/`에 저장하며 Git에 추가하지 않는다.

## 11. C2-4B actual MPS + PostgreSQL smoke

기존 `smartfactory_postgres_data` volume을 삭제하거나 초기화하지 않고 PostgreSQL 17.6을 다시 기동했다.
Migration 전 revision은 `20260820_01`, 기존 PatchCore row는 336개였다. Additive migration 후 revision은
`20260826_01 (head)`이고 PatchCore row는 계속 336개였다. Smoke 직전 known-defect parent/child count는
`0/0`이었다.

Actual PatchCore CPU runtime, actual YOLO MPS runtime, PostgreSQL과 FastAPI TestClient lifespan을 함께
실행했다. 각 POST 뒤 같은 inspection ID의 `known_defect.created` event와 REST detail을 확인하고 마지막에
history로 네 ID를 모두 복구했다.

| Image | Inspection ID | Instances | Inference | In-process HTTP E2E |
| --- | --- | ---: | ---: | ---: |
| `good/000.png` | `57bea816-cf7c-4f7e-b4ff-07614ee68b18` | 0 | 791.740 ms cold | 841.953 ms cold |
| `bent/000.png` | `ba3036b4-00f8-4a88-9a45-b7915548140b` | 3 | 44.033 ms | 68.630 ms |
| `color/000.png` | `e29325bc-d8e6-4c9c-975c-df1aca3c4356` | 1 | 40.959 ms | 62.842 ms |
| `scratch/000.png` | `4f70d6e2-2264-48d5-bb9c-23aa30b38f88` | 1 | 33.996 ms | 57.307 ms |

Parent는 `0 → 4`, child는 `0 → 5`로 증가했다. Good은 parent 1개/child 0개, bent는 child 3개,
color와 scratch는 각각 child 1개다. Class/confidence/bbox/mask summary는 C2-3/C2-4A semantic observation을
유지했다. 네 DB row의 `image_sha256`가 실제 multipart bytes SHA와 일치했고 다음 runtime provenance도
모두 일치했다.

| Provenance | SHA-256 |
| --- | --- |
| Model checkpoint | `594003121b0e071c47d68c3e53c10f438dcec18b5b56b4e5d8831d64001192bd` |
| Artifact metadata | `9f3e3878141e831a6721c5136d67057da906485b9825262bd4e0897b2879fc6b` |
| Dataset Manifest | `1746338c091c18e96a11399c81ea9be0d7350105c4860cfa6a4162144ddb9905` |
| Dataset semantic fingerprint | `768125f38580864e8240dc6a242d7e29eeb1bf0c3167b098456cfec7610dddbf` |

이 결과는 local functional smoke이며 production latency/throughput benchmark가 아니다. HTTP timing에는
PostgreSQL commit과 in-process ASGI response가 포함되지만 WebSocket receive, detail/history recovery는
포함하지 않는다.

## 12. Docker packaging

`ultralytics==8.4.128`은 core project dependency여서 Docker application runtime stage에도 설치된다.
Compose는 YOLO를 default disabled로 두고 enabled runtime bundle을 read-only
`/runtime/yolo-segmentation`에 mount한다. macOS host의 Docker container는 Metal/MPS device를 사용할 수
없으므로 local container 기본 device는 CPU다. Host MPS FastAPI smoke는 loopback-bound
`POSTGRES_PORT`로 Compose PostgreSQL에 연결한다. Migration container와 application startup ordering은
기존 commit-before-serving contract를 유지하며 실제 Docker accelerator benchmark를 주장하지 않는다.

## 13. C3 Decision Engine boundary

C2-4B까지 PatchCore anomaly inference와 YOLO known-defect inference는 endpoint, table, history와 event가
독립적이다. 공통 image SHA가 같아도 자동 join하거나 최종 manufacturing decision을 만들지 않는다. C3에서
correlation identity, timing policy, missing/failed branch, confidence calibration과
`PASS`/`REJECT`/`REVIEW` 규칙을 별도 설계해야 한다.
