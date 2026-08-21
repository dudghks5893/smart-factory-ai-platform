# GitHub Actions CI와 CD-ready Foundation

## 1. 현재 범위

STEP 8은 Pull Request와 `main` push에 대한 CI automation을 제공한다. Quality gate, 실제 PostgreSQL
integration과 production runtime Docker target build가 모두 통과해야 한다. STEP 12부터 같은 Docker job이
internal Dashboard runtime target도 분리 build한다. Registry publication과 실제 production deployment는 아직
구현하지 않았다.

하나의 `.github/workflows/ci.yml`에서 책임을 다음처럼 분리한다.

```text
quality
   ├── postgres-integration
   ├── docker
   └── kubernetes
```

세 infra job은 `quality` 성공 후 병렬로 실행되어, 기본 regression 실패 시 DB/image/Kubernetes validation
자원을 사용하지 않는다.

## 2. Trigger와 execution policy

- 모든 Pull Request
- `main` branch push
- 수동 `workflow_dispatch`

일반 feature branch push에는 별도 run을 만들지 않는다. 같은 Pull Request의 새 commit이 들어오면 이전 run을
취소하지만, `main` push run은 취소하지 않는다. Quality/PostgreSQL job timeout은 30분, Docker build timeout은
45분이다.

## 3. Quality gate

Python `3.12.14`와 uv `0.12.5`를 명시적으로 사용하고 `uv.lock` 기반 cache를 적용한다.

```text
uv lock --check
uv sync --locked
make check
```

`make check`는 Ruff format, Ruff lint, mypy와 전체 pytest suite를 기존 순서 그대로 실행한다. 실제 PostgreSQL
환경 변수가 없는 quality job에서는 Docker/PostgreSQL 전용 test가 기존 contract대로 skip된다. 실제 model,
MVTec raw dataset과 remote MLflow server는 요구하지 않는다.

## 4. PostgreSQL integration

`postgres:17.6-bookworm` service container가 healthy 상태가 된 뒤 production과 같은
`alembic upgrade head`를 실행한다. 이어서 기존 `tests/integration/test_postgres_container.py`가 dedicated
ephemeral database에서 `downgrade base → upgrade head` migration round-trip과 다음 계약을 실제로 검증한다.

- psycopg connectivity와 application readiness
- PostgreSQL UUID, timezone-aware timestamp, indexes와 constraints
- prediction INSERT/COMMIT 및 detail/history retrieval
- malformed request의 non-insert contract
- constraint failure rollback과 unavailable database error

Workflow credential은 해당 ephemeral CI database만을 위한 literal이다. Production credential이나 secret이
아니며 외부 service에 재사용하지 않는다.

## 5. CPU-oriented CI dependency strategy

프로젝트의 universal lock은 platform marker에 따라 다음 source를 유지한다.

| Environment | PyTorch build |
|---|---|
| macOS | PyPI, MPS/CPU |
| Linux arm64/aarch64 | official CPU index |
| Linux x86_64 | official cu130 index |

uv 0.12.5의 `UV_TORCH_BACKEND`/`--torch-backend`는 `uv pip` interface에서만 동작하고 project
`uv sync --locked`의 source를 바꾸지 않는다. CI에서 pyproject를 임시 rewrite하거나 lock 밖에서 PyTorch를
재설치하지 않고, GitHub-hosted `ubuntu-24.04-arm` runner를 사용한다. 따라서 quality, PostgreSQL과 Docker
runtime build는 기존 Linux arm64 marker가 선택한 `torch 2.13.0+cpu`와 `torchvision 0.28.0+cpu`를 lock
그대로 설치한다.

이 선택은 Kaggle/future production Linux x86_64의 cu130 source, local macOS MPS와 기존 Apple Silicon
Docker contract를 변경하지 않는다. CI는 GPU image나 CUDA inference를 검증했다고 간주하지 않는다. 실제 GPU
correctness는 현재 Kaggle Tesla T4 결과로 한정하며, future production GPU image는 NVIDIA runtime/driver와
self-hosted GPU runner 정책이 정해진 뒤 별도 검증해야 한다.

## 6. Docker validation

Docker job은 먼저 `docker compose config --quiet`로 Compose contract를 검증하고, Buildx로 Dockerfile의
`runtime` target을 native `linux/arm64` CPU image로 build한다. 별도 `dashboard-runtime` target도 같은 job에서
build해 Streamlit dependency/image contract를 검증한다. Dashboard target은 dashboard group만 설치하므로 PyTorch
runtime layer를 중복 설치하지 않는다. PostgreSQL integration은 service container와 host test process를
사용하므로 Dockerfile `test` target을 중복 build하지 않는다.

BuildKit cache는 API runtime과 Dashboard target에 별도 scope를 사용하고, lock/source 변경을 실제 layer cache
key에 반영한다.
Image는 push하거나 export하지 않으며 Docker build record artifact upload도 비활성화한다. 따라서 raw dataset,
model, outputs와 MLflow artifact가 Actions artifact로 올라가지 않는다.

## 7. Permissions, secrets와 failure behavior

Workflow token 권한은 `contents: read`만 사용한다. `packages: write`, `id-token: write`, production database,
MLflow credential과 cloud secret은 없다. 각 command가 non-zero로 종료되면 해당 required check가 실패하며,
downstream job은 quality 실패 시 실행되지 않는다.

Branch protection의 required checks에는 다음 job name을 지정한다.

- `quality`
- `postgres-integration`
- `docker`
- `kubernetes`

Branch protection 자체는 repository 설정이므로 이 workflow가 변경하지 않는다.

## 8. Kubernetes validation

`kubernetes` job은 kubectl v1.34.1을 명시적으로 설치하고 base, local-cpu, gcp-gpu Kustomize profile을 모두
render한다. 선행 quality job의 Kubernetes configuration tests가 YAML schema shape, probe/resource/security,
Secret/artifact/migration/GPU scope를 검증한다. CI는 cluster context, GCP credential과 server-side apply를
사용하지 않으며 workflow 권한은 계속 `contents: read`뿐이다.

## 9. 향후 CD 확장

현재 상태는 **CI automation complete, CD-ready image/build pipeline foundation**이다. 향후 deployment target과
credential policy가 확정되면 다음 단계를 별도 workflow/job으로 추가할 수 있다.

```text
verified main commit → immutable image tag → registry publication → deployment → rollout verification
```

GHCR/GCP publication, environment approval, OIDC, Kubernetes rollout/rollback과 production secret은 아직 없다.
Standard GitHub-hosted runner에서 CUDA inference도 실행하지 않는다.

## 10. 검증 한계

Workflow YAML parsing과 repository contract test, lock consistency, local quality gate 및 STEP 7의 actual arm64
Docker/PostgreSQL integration으로 구성을 사전 검증한다. GitHub service scheduling, hosted arm64 image의 실제
dependency install/cache restore와 Actions BuildKit cache는 commit/push 후 실제 run에서만 최종 확인할 수 있다.
