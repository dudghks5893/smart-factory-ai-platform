# PatchCore FastAPI Serving Core

## 1. 범위

STEP 4-1은 STEP 2에서 생성한 PatchCore artifact와 STEP 3 validation threshold artifact를 하나의
FastAPI process에서 제공한다. PostgreSQL, Docker, MLflow, monitoring과 anomaly-map visualization은 이
단계에 포함하지 않는다.

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
Image-level JSON response
```

HTTP route는 model-specific inference를 구현하지 않는다. `services/inference/runtime.py`가 기존
`PatchCoreAdapter`, `PatchCorePreprocessor`와 threshold contract를 결합하고, `services/api`는 transport,
schema와 error mapping만 담당한다.

## 2. Startup lifecycle

FastAPI lifespan startup에서 다음 순서로 runtime을 정확히 한 번 구성한다.

1. Environment serving configuration 검증
2. 요청 device 결정(`auto`: CUDA → MPS → CPU)
3. `metadata.json`과 `thresholds.json` schema 검증
4. Embedded manifest SHA와 실제 metadata/model SHA provenance 검증
5. `model.pt`를 pretrained download 없이 복원
6. Artifact preprocessing과 validation image threshold를 process-local runtime에 보관
7. Runtime 저장이 끝난 뒤 readiness 활성화

Artifact directory/file, threshold, JSON/schema, provenance 또는 명시적 device가 잘못되면 startup이
실패하며 application은 ready 상태가 되지 않는다. Request마다 metadata/model을 다시 읽거나 memory bank를
재구성하지 않는다. Shutdown에 별도 외부 resource가 없으므로 불필요한 cleanup hook은 두지 않는다.

## 3. Configuration

| Environment variable | Required | Meaning |
| --- | --- | --- |
| `PATCHCORE_ARTIFACT_DIR` | Yes | `model.pt`와 `metadata.json`이 있는 artifact directory |
| `PATCHCORE_THRESHOLDS_PATH` | Yes | validation calibration으로 생성한 `thresholds.json` |
| `MODEL_DEVICE` | No | `auto`, `cpu`, `mps`, `cuda`; default `auto` |
| `MAX_UPLOAD_BYTES` | No | Image file 최대 byte 수; default 10 MiB |

Experiment hyperparameter는 serving environment에 중복하지 않는다. Backbone, layers, preprocessing과
threshold는 artifact files가 source of truth다. `.env.example`은 변수 형식만 제공하며 실제 raw artifact와
output은 Git에 추가하지 않는다.

Local 실행 예:

```bash
export PATCHCORE_ARTIFACT_DIR=artifacts/models/patchcore/<artifact-id>
export PATCHCORE_THRESHOLDS_PATH=outputs/evaluation/patchcore/thresholds/<threshold-id>/thresholds.json
export MODEL_DEVICE=auto

uv run uvicorn services.api.app:app --host 127.0.0.1 --port 8000
```

## 4. Endpoint contract

### `GET /health`

Process liveness만 나타낸다. Model이 아직 ready가 아니어도 process가 request를 처리하면 `200`과
`{"status":"ok"}`를 반환할 수 있다.

### `GET /ready`

Model artifact와 threshold가 startup에서 검증·복원된 경우에만 `200`을 반환한다. Runtime이 없으면
`503 model_not_ready`다. Ready response에는 model name, category와 선택된 device를 포함한다.

### `POST /v1/predictions`

`multipart/form-data`의 `image` field로 JPEG(`image/jpeg`) 또는 PNG(`image/png`) 하나를 받는다. Upload를
설정된 최대 크기까지만 읽고 application code에서 별도 temporary file을 생성하지 않는다. Empty,
malformed, MIME/decoded format mismatch와 unsupported media type을 inference 전에 구분해 거부한다.

Response:

```json
{
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
`unsupported_image_format`, `image_too_large`, `model_not_ready`, `inference_failed`, `internal_error`다.
Client response에는 traceback, artifact path, model internals와 원래 exception message를 노출하지 않는다.

## 6. Concurrency와 worker policy

한 process는 PatchCore model과 memory bank 한 copy를 공유한다. PyTorch module과 accelerator queue의 동시
접근을 검토한 결과, STEP 4-1에서는 runtime instance별 lock으로 inference를 직렬화한다. 전역 lock이나
request별 model copy는 사용하지 않으며 upload parsing과 image decoding은 lock 밖에서 수행한다. Blocking
inference는 FastAPI event loop가 아니라 threadpool에서 실행한다.

Uvicorn worker를 여러 개 실행하면 worker process마다 lifespan이 독립 실행되어 model과 memory bank도
worker마다 한 copy씩 생성된다. 따라서 worker 수는 GPU/host memory 사용량과 함께 결정해야 한다. 단일 GPU
worker의 처리량 확장은 HTTP concurrency만 보고 늘리지 않고 실제 resource benchmark로 검증한다.

## 7. Latency boundary

STEP 3 Tesla T4 p50 21.634ms와 45.114 images/second는 disk image load, artifact restore와 warmup을 제외한
preprocessing-to-synchronization model benchmark다. 이 값은 multipart parsing, upload read, image decode,
threadpool scheduling, JSON serialization, network를 포함하지 않으므로 FastAPI HTTP end-to-end latency가
아니다. HTTP E2E latency와 concurrency benchmark는 후속 serving 단계에서 별도로 측정한다.
