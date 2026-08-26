# Manufacturing Decision Engine v1

## 1. Status and boundary

Decision Policy v1 is an **experimental model-agreement baseline** for combined inspections. It is deterministic,
versioned, explainable, persisted and recoverable, but it is not a factory-certified or production-calibrated quality
policy. Independent `POST /v1/predictions` and `POST /v1/known-defects` requests do not create manufacturing
decisions.

The pure domain implementation under `services/decision` depends on neither FastAPI, SQLAlchemy nor WebSocket. It
accepts only normalized model evidence: PatchCore anomaly state/score/threshold and YOLO instance count/classes. It
does not accept an image, Tensor, mask, framework result, random source, clock or external mutable state.

## 2. Policy identity and truth table

Every result stores `policy_name=model_agreement` and `policy_version=1`. These constants are defined once and allow a
future calibrated policy to coexist with historical v1 decisions.

| PatchCore output | YOLO instances | Disposition | Reason code |
| --- | ---: | --- | --- |
| `NORMAL` | 0 | `PASS` | `NO_ANOMALY_EVIDENCE` |
| `ANOMALY` | 0 | `REVIEW` | `UNKNOWN_ANOMALY` |
| `NORMAL` | >0 | `REVIEW` | `MODEL_DISAGREEMENT` |
| `ANOMALY` | >0 | `REJECT` | `CONFIRMED_KNOWN_DEFECT` |

`PASS`, `REVIEW` and `REJECT` are v1 policy outputs rather than model labels. PatchCore `NORMAL`/`ANOMALY` remains a
model output. Stable reason codes carry machine-readable semantics; a separate deterministic explanation string is
returned for operators.

## 3. Calibration constraints

PatchCore uses the existing validation-derived threshold and strict `score > threshold` contract. The Decision Engine
checks this normalized state but does not calculate or adjust the threshold.

YOLO diagnostic confidence `0.25` controls which segmentation instances the runtime reports. It is not a calibrated
reject threshold and is deliberately absent from the Decision Engine input. V1 uses only whether the normalized
instance count is zero or nonzero. It adds no weighted heuristic score, decision confidence or class severity.

## 4. Evidence and persistence

Migration `20260826_03` creates `inspection_decisions` with a unique, non-null foreign key to
`combined_inspections`. The row stores:

- disposition, policy name/version and reason code;
- PatchCore anomaly state, score and threshold snapshot;
- known-defect instance-count snapshot.

This bounded snapshot makes the policy input auditable without putting the decision into one opaque JSON blob.
Ordered YOLO instances and their class names already exist in immutable child rows, so class evidence is recovered
from those rows instead of being duplicated. Human-readable reason text is derived from the versioned stable code.

Existing C3-1 correlations are backfilled during the migration using their immutable child evidence and the same v1
truth table. New requests compute the pure decision before opening the persistence work unit. PatchCore child, YOLO
parent/children, correlation and decision are then flushed and committed in one transaction. Decision insert failure
rolls back the entire request and suppresses all events.

The decision foreign key uses `ON DELETE RESTRICT`, matching the audit-oriented combined relation. One combined UUID
can have exactly one decision.

## 5. REST recovery and history

- `POST /v1/combined-inspections` returns both model results plus `decision`.
- `GET /v1/combined-inspections/{id}` reconstructs the persisted decision and class evidence.
- `GET /v1/combined-inspections` returns newest-first summaries with bounded `limit` (maximum 100), `offset`,
  `returned_count` and `has_more`.

History reads only `combined_inspections` and `inspection_decisions`. It does not hydrate child instances and therefore
does not introduce a per-row N+1 query. Detail recovery loads ordered instances only for the requested combined UUID.

## 6. WebSocket

`/v1/ws/combined-inspections` publishes `combined_inspection.created` after the atomic commit. Its compact payload
contains combined UUID/time, PatchCore output, known-defect count/unique classes, disposition, reason code and policy
identity. It excludes raw images, masks, full instances and provenance.

Combined request event scheduling order is:

1. `inspection.created`;
2. `known_defect.created`;
3. `combined_inspection.created`.

All channels remain process-local best-effort notifications. Ordering describes server scheduling after one durable
commit, not exactly-once delivery across clients or processes. PostgreSQL plus REST remains the recovery source.

The browser-native Live Monitor consumes the combined summary channel independently from both child channels. It
connects WebSocket first, buffers messages, merges PostgreSQL-backed history by combined UUID and repeats REST
recovery after reconnect. Visible-window KPI and cards display the persisted disposition, reason code and policy
identity; JavaScript neither recomputes the truth table nor correlates independent child feeds. Detail is fetched only
after operator interaction. The Combined section visibly identifies v1 as experimental and not production calibrated.

## 7. Actual local verification

The actual macOS/PostgreSQL smoke preserved the existing volume, backfilled four C3-1 correlations, and then created
four new combined decisions using PatchCore CPU and YOLO MPS. `good/000.png` produced `PASS`; `bent/000.png`,
`color/000.png` and `scratch/000.png` produced `REJECT`. POST/detail recovery, bounded history and the dedicated
combined WebSocket agreed for every new row. See `docs/serving/COMBINED_INSPECTION_API.md` for IDs and evidence.

No actual result was modified to force a `REVIEW`; both review branches are covered with policy-contract unit and
integration tests.

## 8. Future calibration boundary

Future policy work may evaluate validation-based YOLO confidence calibration, PatchCore score margins, disagreement
analysis, class-specific severity, false-accept/false-reject cost and manufacturing business rules. Such work requires
appropriate validation and operational evidence. Test-set observations must not be used to tune the current policy.
