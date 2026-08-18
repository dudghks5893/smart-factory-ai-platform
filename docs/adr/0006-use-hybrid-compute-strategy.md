# ADR-0006: Local macOS + Kaggle GPU + GCP Production Hybrid Compute 전략

- 상태: 승인(Approved)
- 결정일: 2026-08-18

## 1. Context

개발 환경은 macOS이며 Vision AI 실험에서는 GPU가 필요할 수 있다.

모든 실험을 GCP GPU에서 실행하면 비용이 증가하고,
Kaggle만 사용하면 Production Infrastructure와 재현 가능한 최종 Cloud Benchmark를
충분히 검증하기 어렵다.

개발, 실험, Production의 목적이 서로 다르므로 각 환경의 역할을 분리할 필요가 있다.

## 2. Decision

Compute 환경을 다음 세 역할로 나눈다.

### Local macOS

사용 목적:

- Source Code 개발
- Unit / Integration Test
- Dataset Pipeline 개발
- CPU/MPS Smoke Test
- Local Docker
- Local PostgreSQL
- Local MLflow

### Kaggle GPU

사용 목적:

- 초기 Vision AI GPU 실험
- PatchCore 실험
- 후보 모델 비교
- 비용을 최소화한 반복 실험

Kaggle Notebook 자체를 Production Code Source로 사용하지 않는다.

### GCP

사용 목적:

- 최종 재현 가능 Benchmark
- GPU Inference Benchmark
- Production-like Deployment
- GKE / Kubernetes
- Autoscaling
- Rolling Update
- Rollback
- Monitoring
- 최종 시스템 검증

## 3. Reason

- 무료 또는 저비용 GPU 자원을 초기 실험에 활용할 수 있다.
- GCP GPU 사용 시간을 최종 검증에 집중할 수 있다.
- Local 개발 생산성을 유지할 수 있다.
- Cloud Production 환경을 실제로 검증할 수 있다.
- 동일한 Source Code를 서로 다른 실행환경에서 검증할 수 있다.

## 4. Environment Independence 원칙

환경별로 별도의 Source Code를 만들지 않는다.

다음 요소를 이용해 환경 차이를 처리한다.

- Configuration
- Environment Variable
- Dependency Lock
- Container Image
- Artifact Storage

다음과 같은 구조는 피한다.

```text
local_code.py
kaggle_code.py
gcp_code.py
```

가능한 한 동일한 Core Code를 재사용한다.

## 5. Alternatives

### GCP Only

환경 통일에는 유리하지만 반복 실험 비용이 증가한다.

### Kaggle / Colab Only

GPU 실험에는 유리하지만 Production Deployment와 Cloud 운영 환경을 충분히 검증하기 어렵다.

### Local Only

Apple Silicon을 활용할 수 있으나 CUDA 기반 최종 Benchmark와 Kubernetes Production 환경을 검증하기 어렵다.

## 6. Consequences

### 장점

- 비용 효율성이 높다.
- Experiment와 Production 역할을 분리할 수 있다.
- Environment Portability를 검증할 수 있다.

### 단점

- 환경별 Hardware 차이로 결과가 달라질 수 있다.
- Kaggle과 GCP 사이 Artifact 전달 방식이 필요하다.

최종 Benchmark에는 항상 Hardware와 실행환경 Metadata를 함께 기록한다.
