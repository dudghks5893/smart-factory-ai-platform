# Combined Inspection API

## 1. Scope

`POST /v1/combined-inspections` accepts one image and returns independent PatchCore anomaly detection and YOLO
known-defect segmentation observations under one `combined_inspection_id`. Existing `POST /v1/predictions` and
`POST /v1/known-defects` contracts remain unchanged.

This endpoint is an orchestration boundary, not a manufacturing Decision Engine. PatchCore `is_anomaly` remains a
model output. The combined response does not contain `PASS`, `REJECT`, `REVIEW`, `decision` or `disposition`.
C3-2 owns any future decision policy and final-decision persistence.

## 2. Execution flow

```text
One bounded multipart upload
        ↓
One JPEG/PNG validation and RGB decode
        ↓
Tensor input + uint8 RGB array + one image SHA-256
        ↓
┌─────────────────────────┬──────────────────────────┐
│ PatchCore worker        │ YOLO worker              │
│ existing runtime lock   │ existing runtime lock    │
└─────────────────────────┴──────────────────────────┘
        ↓ both successful
One SQLAlchemy transaction: PatchCore row + YOLO parent/children + correlation row
        ↓ commit successful
Existing inspection.created and known_defect.created broadcasts
```

Blocking runtime calls execute in separate threadpool workers and are awaited together. The orchestrator has no
global model lock and no CPU/MPS/CUDA selection logic: each process-local runtime keeps its configured device and its
existing instance lock. It does not call either HTTP endpoint internally.

The multipart bytes are read once and the Pillow RGB image is decoded once. PatchCore tensor and YOLO HWC uint8 input
are derived from that decoded image. The upload SHA-256 is computed once and passed to the correlation and both child
persistence values.

## 3. Availability and errors

Combined serving requires PatchCore runtime readiness, enabled and ready YOLO runtime, database connectivity, all
three migrated persistence schemas, and valid model provenance. If YOLO is disabled, the combined endpoint returns
`503 known_defect_model_disabled`; the PatchCore-only endpoint remains usable. An enabled but missing YOLO runtime
returns `503 known_defect_model_not_ready`.

Malformed/unsupported/oversized image errors reuse the existing image contract. Either inference failure returns the
safe `500 inference_failed` envelope. Persistence failure returns `503 persistence_unavailable`. Internal exception,
artifact path, database credential and accelerator detail are not returned.

Combined HTTP success requires both inference branches and the persistence transaction to succeed. A model failure
occurs before persistence. A SQL failure rolls back both child aggregates and the correlation, so no partial durable
combined result or post-commit event is produced.

## 4. Persistence and recovery

Migration `20260826_02` adds `combined_inspections` without changing the existing child tables. One row contains:

- the externally visible combined UUID and creation time;
- unique non-null foreign keys to `inspections` and `known_defect_inspections`;
- shared image SHA-256, dimensions, byte size and media type;
- PatchCore branch wall time and combined orchestration wall time.

YOLO instances remain normalized children of `known_defect_inspections`. Raw upload bytes, raw masks and a final
manufacturing decision are not stored. Restrictive child foreign keys prevent a correlation from silently becoming
incomplete. Existing independently-created PatchCore and YOLO rows need no nullable correlation field.

`GET /v1/combined-inspections/{combined_inspection_id}` reconstructs the correlation, PatchCore row, YOLO parent and
ordered YOLO instances. Unknown IDs return `404 combined_inspection_not_found`. A combined history endpoint is not
added in C3-1.

## 5. Response contract

```json
{
  "combined_inspection_id": "da95a454-b36d-4698-a7fc-df4362be4a3e",
  "created_at": "2026-08-26T00:00:00Z",
  "image": {"width": 700, "height": 700, "sha256": "<shared-image-sha256>"},
  "patchcore": {
    "inspection_id": "<patchcore-child-uuid>",
    "model_name": "patchcore",
    "category": "metal_nut",
    "device": "cpu",
    "is_anomaly": true,
    "anomaly_score": 54.0,
    "threshold": 41.19657897949219,
    "comparison_operator": ">"
  },
  "known_defects": {
    "inspection_id": "<yolo-child-uuid>",
    "created_at": "2026-08-26T00:00:00Z",
    "model": {"name": "yolo11n-seg.pt", "task": "segment", "category": "metal_nut", "device": "mps"},
    "image": {"width": 700, "height": 700},
    "diagnostic_confidence": 0.25,
    "inference_ms": 65.0,
    "image_sha256": "<shared-image-sha256>",
    "model_sha256": "<sha256>",
    "artifact_metadata_sha256": "<sha256>",
    "dataset_manifest_sha256": "<sha256>",
    "dataset_semantic_fingerprint_sha256": "<sha256>",
    "instance_count": 0,
    "instances": []
  },
  "timings": {
    "patchcore_inference_ms": 80.0,
    "yolo_inference_ms": 65.0,
    "orchestration_ms": 82.0
  }
}
```

The PatchCore branch timing is worker wall time around `runtime.predict`. YOLO timing is the adapter's synchronized
device interval. Orchestration timing starts immediately before scheduling both workers and ends after both return.
These boundaries differ; upload read/decode, SHA computation, persistence, broadcast and response serialization are
excluded. Their sum is therefore not HTTP latency or a guaranteed serial estimate.

## 6. WebSocket compatibility

C3-1 adds no combined WebSocket. After the atomic commit, each child is published through its existing independent
channel: `/v1/ws/inspections` receives `inspection.created`, and `/v1/ws/known-defects` receives
`known_defect.created`. These remain process-local best-effort notifications; REST and PostgreSQL remain the recovery
source of truth.

## 7. Local verification boundary

The required actual smoke uses the existing PostgreSQL volume, PatchCore CPU runtime and YOLO MPS runtime with MVTec
`good`, `bent`, `color` and `scratch` samples. It verifies both child IDs, shared image SHA, persisted recovery and
model semantics. A 1–4 sample sequential-versus-parallel observation is a local smoke observation only, not a
production throughput or latency benchmark.

### C3-1 actual macOS smoke

The existing PostgreSQL 17.6 named volume was preserved. Before migration, the revision was `20260826_01` and row
counts were PatchCore `337`, known-defect parent `8`, and known-defect child `10`. Additive upgrade to
`20260826_02` preserved those counts and created an empty correlation table.

Actual PatchCore CPU, YOLO MPS, PostgreSQL and FastAPI TestClient were then used for one combined request per sample.
Every POST and correlation GET returned HTTP 200, each recovered payload equaled its POST payload, and all four
correlation/PatchCore/YOLO rows carried the same request image SHA-256.

| Sample | Combined ID | PatchCore child / score / output | YOLO child / classes |
| --- | --- | --- | --- |
| `good/000.png` | `7de68452-0a7d-4f4b-acba-a941f569bb8d` | `c5591f2d-5354-4413-af29-30ec9931d408` / `34.763465881347656` / normal | `04d9f62b-1d13-469c-8dc6-cfec093f412f` / none |
| `bent/000.png` | `805547d8-dff1-4b1c-a9a6-3f30d12fee2e` | `0e603e5f-aad6-4fff-bf87-0d935f90e643` / `54.36902618408203` / anomaly | `034fa276-9b35-4f80-9d54-b7e786174eae` / bent, bent, scratch |
| `color/000.png` | `b113e59c-5424-4dd4-b72f-b2cd829d98c5` | `dc265f08-b088-44bb-959c-a240917bc32c` / `49.58808898925781` / anomaly | `e35f96f0-5882-4d0b-8733-cb60dc720d20` / color |
| `scratch/000.png` | `928284c6-bbd6-44da-abcc-f4b15ed22fe3` | `a1d692dd-1882-4fc0-9a81-a3fca168731d` / `41.75379180908203` / anomaly | `09df55bc-ae16-4e8e-8702-9ae1301a409e` / scratch |

The PatchCore threshold was `41.19657897949219` with strict `score > threshold`. YOLO used diagnostic confidence
`0.25`; its good/bent/color/scratch cardinality and classes matched the approved actual reference. The smoke added
exactly four PatchCore rows, four YOLO parents, five YOLO instances and four correlations. A direct SQL join found
all four correlations with equal image SHA across the correlation and both child parents.

| Sample | PatchCore branch | YOLO device interval | Combined orchestration | In-process HTTP |
| --- | ---: | ---: | ---: | ---: |
| `good/000.png` | 83.610 ms | 25.986 ms | 84.227 ms | 151.416 ms |
| `bent/000.png` | 82.882 ms | 42.864 ms | 83.261 ms | 118.392 ms |
| `color/000.png` | 103.782 ms | 31.351 ms | 104.665 ms | 129.094 ms |
| `scratch/000.png` | 92.628 ms | 29.796 ms | 93.027 ms | 116.925 ms |

The in-process HTTP column spans `TestClient.post()` through completed ASGI response and additionally includes
multipart handling, decode, SHA calculation, persistence and serialization. It excludes disk image loading, runtime
restore, external network, uvicorn socket, TLS and proxy.

The same loaded runtimes were also called sequentially and through the parallel orchestrator before each HTTP request:

| Sample | Sequential direct wall | Parallel orchestration wall | Observation |
| --- | ---: | ---: | --- |
| `good/000.png` | 2005.715 ms | 250.800 ms | Faster, but dominated by first-call MPS warm-up/order |
| `bent/000.png` | 149.241 ms | 88.896 ms | Faster in this run |
| `color/000.png` | 127.296 ms | 109.489 ms | Faster in this run |
| `scratch/000.png` | 112.601 ms | 85.288 ms | Faster in this run |

This four-sample, sequential-then-parallel observation demonstrates actual overlapping execution, not a controlled
benchmark. It is order-sensitive, has no repeated trials or distribution statistics, and the first row contains cold
MPS initialization. No production performance claim is made.
