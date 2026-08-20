# PostgreSQL Inspection History

## 1. 목적과 경계

STEP 5는 성공한 PatchCore image-level prediction을 inspection history로 영속화하고 UUID로 추적하거나
최신 이력을 조회하는 API를 제공한다. Production target은 PostgreSQL이며 SQLAlchemy 2.x ORM, psycopg 3와
Alembic을 사용한다. Route는 SQL을 직접 작성하지 않고 request별 work unit을 여는 inspection repository를
호출한다.

```text
HTTP upload/decode
        ↓
PatchCore inference + strict threshold
        ↓
Input/model provenance 구성
        ↓
Inspection repository
        ↓
SQLAlchemy Session: insert → commit
        ↓
inspection_id response
```

STEP 7에서 Docker PostgreSQL 17.6과 psycopg 3를 사용한 실제 integration을 추가했다. Local SQLite tests는
빠른 repository/application 회귀를 계속 담당하고, Docker suite는 Alembic과 PostgreSQL 고유 계약을
별도로 검증한다. Production traffic의 connection pool/concurrency는 아직 검증 범위가 아니다.

## 2. Schema

`inspections` table은 다음 값을 저장한다.

| Domain | Columns |
| --- | --- |
| Identity | UUID `id`, DB/server `created_at` |
| Prediction | model, category, anomaly flag, finite score/threshold, `>` operator |
| Input provenance | image SHA-256, byte size, content type |
| Model provenance | model/metadata/threshold/manifest SHA-256 |
| Runtime | device |

모든 field는 `NOT NULL`이다. DB에는 `comparison_operator = '>'`, positive image size와 각 SHA-256
문자열 길이 constraint를 둔다. Application은 score/threshold finite 여부, strict threshold 결과와 SHA-256
hex 형식도 insert 전에 검증한다.

조회에 필요한 index만 유지한다.

- `created_at`
- `(category, created_at)`
- `(is_anomaly, created_at)`

원본 image binary와 filename은 저장하지 않는다. Database row 크기와 개인정보/보존 책임을 불필요하게
늘리지 않고, 향후 object storage가 도입될 때 별도 lifecycle과 access policy로 설계하기 위해서다.
PatchCore가 생성하지 않는 defect type도 저장하지 않는다.

## 3. Transaction과 failure policy

정상 prediction request마다 독립 SQLAlchemy Session을 생성한다. Inference가 성공한 뒤 inspection 한 건을
flush하고 commit하며 commit된 UUID만 client에 반환한다. Insert/commit 실패는 rollback한 뒤
`503 persistence_unavailable`로 변환한다. Prediction은 성공했지만 trace가 유실된 상태를 성공 response로
숨기지 않는다.

Empty/malformed/unsupported/oversized upload, inference failure와 invalid model provenance는 insert 전에
종료하므로 inspection을 만들지 않는다. Session은 request 사이에 공유하지 않으며 model inference lock과 DB
transaction을 하나의 lock으로 결합하지 않는다.

## 4. Lifecycle과 readiness

FastAPI lifespan은 다음 순서로 fail-fast한다.

1. Serving/DB configuration 검증
2. SQLAlchemy engine과 Session factory 생성
3. `SELECT 1` database connectivity 확인
4. Inspection repository 생성과 migrated table query 확인
5. PatchCore artifact/threshold 검증과 runtime 1회 load
6. Ready 상태 전환

Shutdown에서는 engine pool을 dispose한다. Application startup은 table을 자동 생성하지 않는다.
`GET /health`는 process liveness만 유지한다. `GET /ready`는 model runtime 존재와 현재 DB connectivity를
모두 확인하며 DB 또는 migrated inspection table을 사용할 수 없으면 `503 database_not_ready`를 반환한다.

## 5. API contract

`POST /v1/predictions`는 기존 prediction field 의미를 유지하면서 `inspection_id` UUID를 추가한다.

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

조회 API:

- `GET /v1/inspections/{inspection_id}`: 존재하지 않으면 safe `404 inspection_not_found`
- `GET /v1/inspections`: optional `category`, `is_anomaly`, `limit`, `offset`

History는 `created_at DESC, id DESC`로 정렬한다. `limit` 기본값은 20, 최댓값은 100이다. Aggregate count
query 대신 `returned_count`와 limit+1 query로 계산한 `has_more`를 반환한다.

## 6. Migration workflow

Production schema source of truth는 `migrations/versions/`의 Alembic revision이다.

```bash
export DATABASE_URL=postgresql+psycopg://<user>:<password>@<host>:5432/<database>
uv run alembic upgrade head
uv run alembic current
```

초기 revision은 upgrade에서 table/constraint/index를 만들고 downgrade에서 index와 table을 역순 제거한다.
Credential은 `.env.example`의 placeholder가 아니라 deployment secret으로 제공한다.

일반 test fixture만 임시 SQLite file에 `Base.metadata.create_all()`을 사용한다. `make docker-test`는 전용
Compose project와 ephemeral PostgreSQL volume에서 migration downgrade/upgrade, psycopg connectivity,
PostgreSQL native UUID와 timezone-aware timestamp, constraint/index, commit/rollback, application readiness와
FastAPI inspection API를 실제 검증한 뒤 해당 test volume을 제거한다. 실제 model 대신 app factory에 test-only
runtime을 주입하므로 production runtime에는 fake flag가 없다.

STEP 7 local Apple Silicon 검증에서 PostgreSQL integration은 통과했다. 장시간/concurrent production pool,
실제 production credential과 managed PostgreSQL 연결은 아직 검증하지 않았다. Docker lifecycle과 실행 명령은
`docs/deployment/DOCKER.md`에서 관리한다.

Inspection history는 향후 Dashboard의 검사 이력과 Monitoring의 error/anomaly 집계 입력이 될 수 있지만,
Dashboard와 monitoring 구현 자체는 현재 범위에 포함하지 않는다.
