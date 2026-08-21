# Kubernetes / GCP Deployment Foundation

## 1. 범위와 현재 검증 수준

STEP 11은 existing FastAPI PatchCore runtime을 Kubernetes workload로 표현하고 GCP production deployment에
필요한 경계를 정의한다. 실제 GKE cluster, Artifact Registry, Cloud Storage bucket, Cloud SQL, GPU node pool,
Load Balancer 또는 public endpoint는 생성하지 않았다.

현재 검증 수준은 다음과 같다.

- Kubernetes API workload와 migration Job Kustomize render
- CPU/GPU resource, probe, security, Secret와 external artifact mount contract
- Unit configuration tests와 GitHub Actions render job
- Local kubectl client-only static validation
- 실제 GKE/GPU/container runtime/model mount/Cloud SQL 연결은 미검증

## 2. Production architecture

```text
GitHub Actions
  ├── quality / PostgreSQL / Docker / Kubernetes render
  └── future approved CD
              ↓ immutable git-SHA image
       Artifact Registry
              ↓
GKE namespace: smartfactory
  ├── Alembic migration Job ───────────┐
  └── FastAPI Deployment → ClusterIP   │
          │                            │
          ├── /health /ready /metrics  │
          └── read-only model mount    │
                                       │
External/managed dependencies          │
  ├── Cloud SQL PostgreSQL ← DATABASE_URL Secret
  ├── Cloud Storage → model/threshold/drift artifacts
  ├── Secret Manager → credentials
  └── future external MLflow tracking/artifact store
```

Docker Compose PostgreSQL과 Prometheus/Grafana는 local development/integration observer로 유지한다. Production
PostgreSQL StatefulSet, Kubernetes MLflow server와 duplicate monitoring stack은 만들지 않는다.

## 3. Kustomize 구조

```text
infra/k8s/
├── base/
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── migration-job.yaml
│   ├── deployment.yaml
│   ├── service.yaml
│   └── kustomization.yaml
└── overlays/
    ├── local-cpu/
    │   └── kustomization.yaml
    └── gcp-gpu/
        ├── configmap-patch.yaml
        ├── deployment-patch.yaml
        └── kustomization.yaml
```

Helm과 template engine은 사용하지 않는다. Base image는 local placeholder인 `smartfactory-api:local`이며 개인
project/region을 하드코딩하지 않는다. 실제 release renderer는 Deployment와 migration Job 모두에 같은
immutable image를 적용하도록 Kustomize `images` 값을 다음 형태로 교체해야 한다.

```text
REGION-docker.pkg.dev/PROJECT_ID/REPOSITORY/smartfactory-api:<git-sha>
```

Production에서 `latest` tag를 사용하지 않는다.

## 4. API Deployment contract

- Replica 1, Dockerfile의 uvicorn worker 1 CMD를 그대로 상속한다.
- PatchCore runtime/model memory는 process와 replica마다 별도로 load된다.
- `RollingUpdate`, `maxUnavailable=0`, `maxSurge=1`, `minReadySeconds=10`
- `terminationGracePeriodSeconds=60`으로 uvicorn graceful shutdown 시간을 제공한다.
- `revisionHistoryLimit=5`, `progressDeadlineSeconds=600`
- Service account token은 mount하지 않는다.

근거 없는 `preStop` sleep은 추가하지 않았다. Pod restart 시 application lifespan이 DB와 artifact를 검증하고
PatchCore를 다시 load한 뒤 readiness가 통과한다. Application에 Kubernetes 전용 recovery state를 추가하지
않는다.

Replica 1에서 PDB는 node maintenance/eviction을 불필요하게 막을 수 있어 제외했다. HPA/KEDA도 model memory가
replica마다 복제되고 CPU/GPU capacity relationship이 아직 측정되지 않아 제외했다. Production load test 후
동시성, replica와 accelerator capacity를 함께 보정해야 한다.

## 5. Probe contract

| Probe | Endpoint | 설정 | 의미 |
|---|---|---|---|
| Startup | `/health` | 5초 후 10초 주기, 최대 30회, timeout 3초 | DB/artifact/model startup 동안 liveness restart 억제 |
| Liveness | `/health` | 20초 주기, timeout 3초, 3회 실패 | Process/application liveness |
| Readiness | `/ready` | 10초 주기, timeout 5초, 3회 실패 | Model loaded + DB reachable + migrated schema ready |

Uvicorn은 FastAPI lifespan 완료 전 정상 request를 받지 않으므로 startup probe가 최대 약 5분의 model/DB 준비
시간을 제공한다. Readiness에 model load와 DB/schema 의미를 유지하고 liveness와 섞지 않는다.

## 6. Resource baseline

CPU base:

| Resource | Request | Limit |
|---|---:|---:|
| CPU | 500m | 2 |
| Memory | 1 GiB | 3 GiB |

GPU overlay:

| Resource | Request | Limit |
|---|---:|---:|
| CPU | 1 | 4 |
| Memory | 2 GiB | 6 GiB |
| GPU | scheduler default | `nvidia.com/gpu: 1` |

186 MiB model file만으로 memory를 추정하지 않고 PyTorch, backbone, memory bank, preprocessing와 request
temporary memory를 고려한 초기 baseline이다. Production peak RSS/OOM/latency를 실제 workload로 측정한 뒤
조정해야 한다. 특정 T4/L4/A100 selector나 node affinity는 실제 region/node pool 선택 전에 넣지 않는다.

GPU Deployment의 `maxSurge=1`은 rollout 중 임시 spare GPU가 필요할 수 있다. 추가 accelerator capacity가 없으면
surge Pod가 Pending 상태가 되어 zero-downtime rollout이 불가능할 수 있다. 그 경우 capacity 확보 또는 승인된
rollout strategy 조정이 필요하다.

## 7. Security와 writable filesystem

Pod/container는 Docker runtime UID/GID 10001과 다음 contract를 유지한다.

- `runAsNonRoot: true`, `runAsUser/runAsGroup: 10001`
- `seccompProfile: RuntimeDefault`
- `allowPrivilegeEscalation: false`
- `readOnlyRootFilesystem: true`
- Linux capabilities `ALL` drop
- `automountServiceAccountToken: false`

Multipart/parser/library temporary write를 위해 `/tmp`에 memory-backed `emptyDir`를 mount한다. API는 256 MiB,
migration Job은 64 MiB size limit을 사용한다. API upload limit은 10 MiB이지만 framework/runtime temporary
overhead를 위한 여유를 둔다. Memory-backed volume 사용량은 Pod memory 소비에 포함되므로 production에서 함께
관찰해야 한다.

## 8. ConfigMap, Secret과 artifact mount

ConfigMap에는 existing `ServingSettings`의 non-secret 값만 둔다.

- `MODEL_DEVICE`
- `MAX_UPLOAD_BYTES`
- `PATCHCORE_ARTIFACT_DIR=/runtime/model`
- `PATCHCORE_THRESHOLDS_PATH=/runtime/thresholds/thresholds.json`

`DATABASE_URL`은 repository에 없는 `smartfactory-api-secrets` Secret의 `database-url` key만 참조한다. 실제
credential YAML은 commit하지 않는다. Production에서는 Google Secret Manager와 Secret Manager CSI 또는
External Secrets 도입 여부를 별도 결정한다. 이번 foundation은 operator를 설치하지 않는다.

Base는 repository가 생성하지 않는 pre-provisioned PVC `smartfactory-model-artifacts`를 external contract로
참조한다. PVC root 구조는 다음과 같아야 한다.

```text
model/
├── model.pt
└── metadata.json
thresholds/
└── thresholds.json
```

두 subpath를 `/runtime/model`, `/runtime/thresholds`에 read-only mount한다. 186 MiB model이나 threshold를
ConfigMap/Secret에 넣지 않고 application이 production artifact를 수정하지 못하게 한다. Local cluster에서는
operator가 compatible PVC를 준비해야 하며, GCP에서는 Cloud Storage CSI/FUSE 또는 별도 sync/delivery 방식을
선택해 동일 PVC/path contract를 만족시킬 수 있다. Bucket, CSI driver와 storage credential은 이번 단계에서
만들지 않았다.

## 9. Migration Job과 release 순서

`smartfactory-api-migrate` Job은 API와 같은 image 및 DATABASE_URL Secret으로 `alembic upgrade head`만 실행한다.
Application Deployment와 initContainer에서는 migration을 실행하지 않는다. `restartPolicy=Never`,
`backoffLimit=2`, `activeDeadlineSeconds=600`을 사용한다.

Base Kustomization에는 configuration, migration, API resource를 함께 두어 같은 image transformation을 받게
한다. 실제 apply는 component label selector로 다음 순서를 반드시 지킨다. Selector 없이 overlay 전체를 한 번에
apply하는 방식은 migration completion gate를 우회하므로 production 배포 절차로 사용하지 않는다.

```bash
PROFILE=infra/k8s/overlays/local-cpu

# 1. Namespace와 non-secret configuration
kubectl apply -k "${PROFILE}" -l app.kubernetes.io/component=configuration

# 2. Repository 밖에서 Secret 생성/갱신
kubectl create secret generic smartfactory-api-secrets \
  --namespace smartfactory \
  --from-literal=database-url='postgresql+psycopg://USER:PASSWORD@HOST:5432/DATABASE' \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl get secret smartfactory-api-secrets --namespace smartfactory

# 3. External artifact provider가 smartfactory-model-artifacts PVC를 Ready로 제공
kubectl get pvc smartfactory-model-artifacts --namespace smartfactory

# 4. 고정 이름 Job의 immutable field 충돌을 피하도록 delete/recreate
kubectl delete job smartfactory-api-migrate --namespace smartfactory --ignore-not-found
kubectl apply -k "${PROFILE}" -l app.kubernetes.io/component=migration
kubectl wait --for=condition=complete job/smartfactory-api-migrate \
  --namespace smartfactory --timeout=10m

# 5. Migration 성공이 확인된 뒤 API release
kubectl apply -k "${PROFILE}" -l app.kubernetes.io/component=api
kubectl rollout status deployment/smartfactory-api --namespace smartfactory --timeout=10m
```

Migration 실패 상태에서는 새 API release를 정상으로 간주하지 않는다. 고정 Job 이름은 이해하기 쉬운
delete/recreate 절차를 선택했으며 Job history가 필요해지면 CD run/git SHA가 포함된 versioned name을 검토한다.

## 10. Health 확인과 rollback

Cluster 내부 Service는 `ClusterIP`이고 port 80을 container의 named port 8000으로 전달한다. LoadBalancer,
Ingress/Gateway, TLS, Cloud Armor와 public internet exposure는 없다.

```bash
kubectl get pods,service --namespace smartfactory
kubectl rollout history deployment/smartfactory-api --namespace smartfactory
kubectl port-forward service/smartfactory-api 8000:80 --namespace smartfactory

curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/ready
curl --fail http://127.0.0.1:8000/metrics
```

Application rollback:

```bash
kubectl rollout undo deployment/smartfactory-api --namespace smartfactory
kubectl rollout status deployment/smartfactory-api --namespace smartfactory --timeout=10m
```

Deployment rollback은 DB schema를 자동 rollback하지 않는다. Migration이 backward-incompatible하면 이전
application image가 새 schema에서 동작하지 않을 수 있다. Expand/migrate/contract와 backward-compatible
migration을 우선하고, schema downgrade는 데이터 손실과 compatibility를 별도로 검토한 명시적 운영 작업으로
취급한다.

## 11. CPU와 GPU profile

`local-cpu` overlay는 architecture-neutral Kubernetes resource를 사용하며 `MODEL_DEVICE=cpu`다. Local Apple
Silicon Docker에서 검증한 Linux arm64 CPU image는 image build contract이고 manifest architecture contract가
아니다. 같은 manifest는 immutable multi-arch 또는 target architecture image로 교체할 수 있다.

`gcp-gpu` overlay는 `MODEL_DEVICE=cuda`와 GPU 1개만 요청한다. Kaggle T4의 Linux x86_64, CUDA 13,
torch/torchvision cu130 inference 성공은 GKE 검증이 아니다. 다음은 아직 실제로 확인해야 한다.

- Linux x86_64 production Docker image와 NVIDIA container runtime
- GKE accelerator scheduling/device allocation
- Driver/CUDA/PyTorch compatibility
- Mounted real model/threshold artifact와 Cloud SQL readiness
- GPU rollout spare capacity와 production latency/memory

## 12. GKE Standard와 Autopilot

Google은 일반 workload에 Autopilot을 우선 권장하고, Standard는 node infrastructure와 placement를 직접
제어해야 하는 workload에 적합하다고 설명한다. Autopilot도 Accelerator compute class로 GPU workload를
지원한다. 공식 비교는 [GKE mode 선택](https://cloud.google.com/kubernetes-engine/docs/concepts/choose-cluster-mode)과
[Autopilot GPU](https://cloud.google.com/kubernetes-engine/docs/how-to/autopilot-gpus)를 기준으로 한다.

이 프로젝트의 첫 실제 verification target은 **GKE Standard**를 추천한다. 이유는 GPU node pool, accelerator,
capacity, driver/runtime와 rollout condition을 명시적으로 관찰하고 Kaggle/Docker와 production 차이를 재현 가능한
기록으로 남기는 것이 초기 목표이기 때문이다. 운영 단순화가 더 중요한 단계에서는 Autopilot GPU profile도
비교할 가치가 있다. Cluster 생성 직전 현재 가격, 지원 accelerator/GKE version/region/quota와 Autopilot 제약을
공식 문서에서 다시 확인해야 한다.

## 13. External service와 observability 범위

- PostgreSQL: GKE 안에 StatefulSet을 만들지 않고 Cloud SQL 같은 managed service를 우선 검토한다. Private IP,
  Auth Proxy/Connector 선택은 실제 network design 때 결정한다.
- Container: Artifact Registry에 immutable API/migration image를 둔다.
- Model/threshold/drift reference/report: Cloud Storage를 권장하며 image와 역할을 분리한다.
- Secret: Secret Manager 연계를 향후 CD/security 단계에서 구성한다.
- Metrics: existing `/metrics`를 유지한다. Google Managed Service for Prometheus 또는 별도 Prometheus/Grafana는
  실제 GKE observability architecture에서 선택한다.
- MLflow: Kubernetes server를 추가하지 않고 future external tracking URI에 연결한다.
- Drift: Batch CLI를 CronJob으로 만들지 않았다. Cloud Storage output/reference persistence와 retention이 실제로
  연결된 후 schedule한다.

NetworkPolicy는 CNI/provider와 실제 egress/ingress requirement를 확인하지 않은 static foundation에서 넣지
않았다. GKE production hardening에서 API→Cloud SQL/metrics/artifact egress와 frontend ingress를 명시적으로
검증한 뒤 추가한다.

## 14. Static validation과 CI

```bash
make k8s-render
make k8s-check
```

두 명령은 base, local-cpu와 gcp-gpu overlay를 kubectl native Kustomize로 render한다. GitHub Actions의
`kubernetes` job은 pinned kubectl v1.34.1로 `make k8s-check`를 실행한다. YAML parsing과 configuration contract
tests는 선행 `quality` job의 `make check`에 포함된다. 실제 cluster나 GCP credential은 CI에 없다.

Local macOS에는 kubectl v1.34.1/Kustomize v5.7.1 client가 있지만 current context와 cluster가 없다. Render는
client-only로 검증했다. `kubectl apply --dry-run=client --validate=false`도 built-in resource discovery를 위해 API
server에 접근하려 했으므로 cluster 없는 환경에서는 완료되지 않았다. 이를 server/dry-run 검증 성공으로
표현하지 않는다. Docker Desktop Kubernetes나 kind를 자동 설치하지 않았다.

## 15. 비용과 향후 작업

GKE control plane/compute/GPU, Cloud SQL, Artifact Registry storage/egress, Cloud Storage와 Load Balancer는 비용이
발생할 수 있다. 이번 STEP에서는 resource 생성, API enablement, credential/OIDC와 billing 작업을 전혀 수행하지
않았다. 실제 배포는 사용자 승인, budget/alert/quota/region 확인 후 별도 단계로 진행한다.

향후 범위:

- Approved Workload Identity Federation 기반 Artifact Registry push와 GKE CD
- Cloud SQL connection/network 선택과 Secret Manager integration
- Cloud Storage artifact delivery/retention 및 drift CronJob
- GKE cluster-side apply/server validation과 real model CPU/GPU smoke
- Load test 기반 replicas/HPA/resource/PDB/rollout calibration
- Gateway/Ingress, TLS, Cloud Armor와 tested NetworkPolicy
