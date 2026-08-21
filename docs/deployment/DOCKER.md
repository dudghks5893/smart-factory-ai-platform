# Dockerized FastAPI + PostgreSQL + Alembic

## 1. 범위와 architecture

STEP 7은 FastAPI application image, 실제 PostgreSQL과 별도 Alembic migration lifecycle을 제공한다. STEP 12는
이 lifecycle을 변경하지 않고 별도 internal Dashboard image/service를 observer로 추가한다.

```text
postgres:17.6-bookworm
        ↓ service_healthy
migrate: application runtime image + alembic upgrade head
        ↓ service_completed_successfully
api: uvicorn, one worker, external PatchCore artifacts
```

Migration은 application startup에서 실행하지 않는다. Migration 실패 시 Compose dependency가 API 시작을
허용하지 않으며, API 자체도 startup에서 connectivity와 migrated table을 다시 확인한다.

MLflow Tracking Server는 이 Compose stack에 없다. STEP 6 client는 `MLFLOW_TRACKING_URI`로 향후 remote
server에 연결할 수 있지만 이번 stack은 serving persistence만 다룬다.

## 2. Dockerfile

Base/runtime tool version은 다음 tag로 고정한다.

- `python:3.12.14-slim-bookworm`
- `ghcr.io/astral-sh/uv:0.12.5`

Dependency layer는 `pyproject.toml`과 `uv.lock`을 source보다 먼저 복사하고
`uv sync --locked --no-dev --no-group dashboard --no-install-project`로 API production dependency만 설치한다.
Dashboard target은 `uv sync --locked --only-group dashboard --no-install-project`로 Streamlit group만 설치한다.
Runtime target에는 pytest/Ruff/mypy/Streamlit이 없고 test target에만 quality dependency와 `tests/`를 추가한다.

Runtime은 UID/GID 10001의 `app` non-root user로 실행한다. Source bind mount, reload와 development secret은
사용하지 않는다. 기본 command는 다음과 같다.

```text
uvicorn services.api.app:app --host 0.0.0.0 --port 8000 --workers 1
```

PatchCore memory bank가 process마다 복제되므로 worker 기본값은 1이다. Worker 증가는 실제 memory/capacity
측정 후 결정한다.

## 3. Platform dependency

하나의 universal lock에서 PyTorch source를 architecture별로 선택한다.

| Platform | PyTorch source | 용도 |
|---|---|---|
| macOS | PyPI fallback | MPS/CPU local development |
| Linux arm64/aarch64 | PyTorch official CPU index | Apple Silicon Docker integration |
| Linux x86_64 | PyTorch official cu130 index | Kaggle 및 future GPU target |

Apple Silicon Docker에서 실제 `linux/arm64`, `torch 2.13.0+cpu`, `torchvision 0.28.0+cpu`, CUDA False를
확인했다. Linux x86_64 uv resolution은 `torch 2.13.0+cu130`, `torchvision 0.28.0+cu130`과 CUDA 13
dependencies를 유지한다. Kaggle T4 real-model 실행 결과는 Docker GPU image 검증이 아니다. Future GCP GPU는
NVIDIA runtime/driver compatibility와 필요시 GPU-specific target을 별도로 검증해야 한다.

## 4. Compose services

### postgres

- Image: `postgres:17.6-bookworm`
- Readiness: `pg_isready`
- Data: `postgres_data` named volume
- Host port는 노출하지 않고 Compose network 안에서만 접근

Compose의 `change-me-local-only` default는 local development example일 뿐 production secret이 아니다.
Production에서는 deployment secret으로 `POSTGRES_PASSWORD`를 반드시 교체한다. Password에 URL reserved
character가 있으면 `DATABASE_URL`에 사용할 수 있도록 percent-encoding해야 한다.

### migrate

API와 같은 runtime image/code를 사용해 `alembic upgrade head`만 실행하는 one-shot service다. PostgreSQL
health 이후 시작하며 root filesystem은 read-only다.

### api

Migration 성공 후 시작한다. `/health`는 process liveness이고 `/ready`는 model runtime, DB connectivity와
schema readiness를 함께 확인한다. Dockerfile 기본 healthcheck는 liveness, Compose healthcheck는 readiness를
사용한다.

### test

`test` profile과 Dockerfile test target만 dev dependency를 포함한다. 실제 PostgreSQL과 migration service를
사용하지만 inference는 app factory DI로 test-only fake runtime을 전달한다. Production code/image에는
`FAKE_MODEL` 같은 우회 설정이 없다.

### prometheus / grafana

Prometheus와 Grafana는 API의 optional observer이며 API startup dependency가 아니다. Prometheus는
`api:8000/metrics`를 15초마다 scrape하고 Grafana는 provisioned Prometheus datasource/dashboard를 사용한다.
Configuration/dashboard는 read-only bind mount하고 time-series/Grafana DB는 각각 `prometheus_data`,
`grafana_data` named volume에 둔다. 세부 metric 계약은 `docs/monitoring/MONITORING.md`에서 관리한다.

### dashboard

Streamlit `dashboard-runtime` target을 사용하는 API client/observer다. API startup dependency가 아니므로 Dashboard
실패가 PostgreSQL/migration/API lifecycle을 막지 않는다. `DASHBOARD_API_BASE_URL=http://api:8000`으로 기존
inspection endpoint를 사용하고 browser link는 `GRAFANA_URL=http://localhost:3000`처럼 사용자 접근 주소를
사용한다. Host drift artifact root만 `/runtime/drift:ro`로 mount하며 non-root/read-only root filesystem과 `/tmp`
tmpfs를 유지한다. 상세 계약은 `docs/dashboard/DASHBOARD.md`에서 관리한다.

## 5. Model artifact policy

다음은 image/build context에 포함하지 않는다.

- raw MVTec dataset
- trained `model.pt`와 `metadata.json`
- `thresholds.json`, benchmark/evaluation outputs
- MLflow SQLite/artifacts
- PostgreSQL data
- `.env`와 credential

API 실행 시 검증된 model directory와 threshold file을 read-only mount한다.

```text
PATCHCORE_ARTIFACT_DIR_HOST → /runtime/model
PATCHCORE_THRESHOLDS_PATH_HOST → /runtime/thresholds/thresholds.json
```

기본 placeholder 경로에 artifact가 없으면 API startup이 실패하는 것이 정상이다. STEP 7 Docker/PostgreSQL
검증 때문에 186 MiB baseline 재학습을 강제하지 않는다. Future deployment는 object/artifact storage 또는
별도 delivery mechanism으로 같은 project-native artifact contract를 공급해야 한다.

## 6. Environment

Compose가 사용하는 주요 값:

- `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`
- `PATCHCORE_ARTIFACT_DIR_HOST`, `PATCHCORE_THRESHOLDS_PATH_HOST`
- `MODEL_DEVICE` (`cpu`가 local Docker 기본값)
- `MAX_UPLOAD_BYTES`, `API_PORT`, `IMAGE_TAG`
- `PROMETHEUS_PORT`, `GRAFANA_PORT`
- `GRAFANA_ADMIN_USER`, `GRAFANA_ADMIN_PASSWORD` (local example; production secret 교체 필수)
- `DASHBOARD_API_BASE_URL`, `DRIFT_REPORT_DIR_HOST`, `GRAFANA_URL`
- `DASHBOARD_REQUEST_TIMEOUT_SECONDS`, `DASHBOARD_PORT`

Container 내부 `DATABASE_URL`은 `postgres` service hostname과 `postgresql+psycopg` driver를 사용한다.
Credential을 image나 repository에 bake하지 않는다.

## 7. Commands와 lifecycle

```bash
make docker-build
make docker-up
make docker-down
make docker-test
make monitoring-config-check
make monitoring-up
make monitoring-down
make dashboard
make dashboard-build
make dashboard-up
make dashboard-down
```

- `docker-build`: runtime API image build
- `docker-up`: PostgreSQL → migration → API 시작. External model/threshold가 필요하다.
- `docker-down`: containers/network만 종료하며 persistent `postgres_data` volume은 유지한다.
- `docker-test`: 별도 `smartfactory-step7-test` project에서 PostgreSQL integration을 실행하고 test volume까지 제거한다.
- `monitoring-config-check`: pinned Prometheus image의 promtool로 scrape config를 검증한다.
- `monitoring-up`: API와 독립적으로 Prometheus/Grafana observer를 시작한다.
- `monitoring-down`: monitoring containers를 stop하고 named volume은 보존한다.
- `dashboard`: host에서 Streamlit Dashboard를 manual-refresh mode로 실행한다.
- `dashboard-build`: API와 분리된 minimal Dashboard target을 build한다.
- `dashboard-up`: Dashboard만 시작한다. API unavailable 상태도 UI error로 처리한다.
- `dashboard-down`: Dashboard container를 stop한다.

다음 명령은 persistent PostgreSQL, Prometheus와 Grafana volume을 모두 삭제하는 destructive cleanup이다.

```bash
make docker-clean-volumes
```

Backup/보존 필요 여부를 확인하지 않고 이 명령을 실행하면 안 된다. `docker-test`에
`COMPOSE_PROJECT_NAME=smartfactory`를 지정하면 persistent project 보호를 위해 거부한다.

## 8. Actual local verification

Apple Silicon Docker Desktop의 `linux/arm64`에서 다음을 실제 확인했다.

- Compose config validation
- Runtime/test image build from the lockfile
- Non-root runtime user
- PostgreSQL 17.6 health PASS
- Migration one-shot `upgrade head` PASS
- Dedicated ephemeral DB `downgrade base → upgrade head` PASS
- psycopg connectivity와 application readiness query
- PostgreSQL UUID, `timestamp with time zone`, constraints와 indexes
- FastAPI valid PNG inference result → insert/commit → inspection UUID
- Detail/history 조회와 timezone-aware response
- Malformed image가 row를 추가하지 않음
- Constraint failure rollback과 unavailable endpoint error
- Test containers/network/volume cleanup 후 orphan 없음

실제 trained PatchCore를 mount한 API container readiness와 HTTP inference, Linux x86_64 GPU Docker, remote
PostgreSQL, production concurrency/TLS/reverse proxy는 아직 검증하지 않았다.
