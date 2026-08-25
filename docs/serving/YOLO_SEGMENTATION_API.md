# YOLO Segmentation FastAPI Serving

## 1. Scope

C2-4A는 C2-3의 `YoloSegmentationAdapter`를 기존 Vision API lifespan에 선택적으로 연결하고
`POST /v1/known-defects`로 known-defect instance를 제공한다. Route는 project-owned runtime result만
API schema로 변환하며 Ultralytics `Results`를 직접 해석하지 않는다.

이 endpoint는 PatchCore의 anomaly score와 자동 결합하지 않으며 PostgreSQL 저장, WebSocket broadcast,
Live Monitor 갱신 또는 `PASS`/`REJECT`/`REVIEW` 판정을 수행하지 않는다. 이 책임은 후속 C2-4B 및
PatchCore+YOLO Decision Engine 범위다.

```text
                         ┌─ POST /v1/predictions ── PatchCore ── anomaly score
Multipart JPEG/PNG ──────┤
                         └─ POST /v1/known-defects ─ YOLO ───── known-defect instances
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

FastAPI lifespan은 required database와 PatchCore runtime을 준비한 뒤 enabled YOLO artifact를 검증하고
model을 한 번 복원한다. 성공한 process는 같은 `YoloSegmentationAdapter`를 모든 request에서 재사용하며
request마다 `YOLO(model.pt)`를 호출하지 않는다.

`GET /health`는 기존 process liveness다. `GET /ready`의 response body는 기존 PatchCore identity contract를
유지한다. YOLO가 disabled이면 readiness 의미도 기존과 같다. Enabled이면 YOLO restore 성공이 aggregate
readiness의 필수 조건이며 startup failure 또는 runtime loss는 ready 상태가 아니다.

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

## 5. Error contract

기존 safe error envelope를 재사용한다. 주요 code는 다음과 같다.

| HTTP | Code | Condition |
| ---: | --- | --- |
| 400 | `empty_image`, `invalid_image` | Empty 또는 decode 불가능한 upload |
| 413 | `image_too_large` | `MAX_UPLOAD_BYTES` 초과 |
| 415 | `unsupported_media_type`, `unsupported_image_format` | Media/decoded format 불일치 |
| 503 | `known_defect_model_disabled` | Optional YOLO가 disabled |
| 503 | `known_defect_model_not_ready` | Enabled runtime이 ready가 아님 |
| 500 | `inference_failed` | Runtime inference failure |

Response에는 traceback, artifact path, accelerator detail 또는 원래 exception message를 노출하지 않는다.

## 6. Concurrency and process policy

Image read/decode는 FastAPI event loop에서 수행하고 blocking inference는 threadpool로 보낸다. C2-3 adapter의
instance lock이 같은 process의 shared Ultralytics model 접근을 직렬화하며 request별 mutable result를
공유하지 않는다. 현재 local/Compose Uvicorn contract는 `workers=1`이다.

여러 worker를 사용하면 각 process가 독립 lifespan과 model copy를 가지므로 worker마다 PatchCore memory
bank와 YOLO weight가 다시 적재된다. Worker 수는 GPU/MPS memory와 실제 concurrency benchmark 없이
늘리지 않는다.

## 7. Actual macOS MPS API smoke

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

## 8. Docker packaging

`ultralytics==8.4.128`은 core project dependency여서 Docker application runtime stage에도 설치된다.
Compose는 YOLO를 default disabled로 두고 enabled runtime bundle을 read-only
`/runtime/yolo-segmentation`에 mount한다. macOS host의 Docker container는 Metal/MPS device를 사용할 수
없으므로 local container 기본 device는 CPU다. C2-4A는 실제 Docker accelerator benchmark를 주장하지
않는다.
