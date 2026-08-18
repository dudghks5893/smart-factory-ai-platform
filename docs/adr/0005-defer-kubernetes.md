# ADR-0005: Kubernetes 도입을 후반 단계로 연기

- 상태: 승인(Approved)
- 결정일: 2026-08-18

## 1. Context

최종 프로젝트에는 Kubernetes Deployment, Autoscaling, Rolling Update,
Rollback을 포함할 계획이다.

하지만 모델, API, Database, Monitoring이 완성되기 전에 Kubernetes부터 구축하면
Application 문제와 Infrastructure 문제를 동시에 디버깅해야 하며
프로젝트 복잡도가 빠르게 증가한다.

## 2. Decision

Kubernetes는 초기 개발 단계에서 사용하지 않는다.

개발 순서는 다음 원칙을 따른다.

```text
Application 기능
    ↓
Local 실행
    ↓
Test
    ↓
Docker
    ↓
Monitoring
    ↓
Kubernetes
```

초기 Application과 ML Component는 Local 환경에서 직접 실행하거나
필요한 단계에서 Docker Compose를 사용한다.

Kubernetes는 공식 구현 순서의 후반 단계에서 도입한다.

## 3. Reason

- 기능 문제와 Infrastructure 문제를 분리해서 해결할 수 있다.
- 초기 개발 속도를 유지할 수 있다.
- 불필요한 YAML 및 Cluster 운영 작업을 줄일 수 있다.
- Kubernetes가 해결해야 할 실제 요구사항을 먼저 확보할 수 있다.
- 최종적으로 Kubernetes 도입 이유를 면접에서 명확하게 설명할 수 있다.

## 4. Alternatives

### 프로젝트 시작부터 Kubernetes 사용

Production 환경과 유사하다는 장점이 있지만,
현재 규모에서는 개발 복잡도와 비용이 지나치게 증가한다.

### Kubernetes 미사용

프로젝트 구현은 가능하지만 최종 목표인
Deployment Lifecycle, HPA, Rolling Update, Rollback을 검증할 수 없다.

## 5. Consequences

### 장점

- Application Layer를 먼저 안정화할 수 있다.
- Kubernetes 학습이 단순 기술 시연으로 끝나는 것을 방지한다.
- Container 단계와 Orchestration 단계를 분리해서 설명할 수 있다.

### 단점

- 후반 단계에서 Kubernetes에 맞는 일부 설정 변경이 필요할 수 있다.
- Local과 Cluster 환경 차이를 별도로 검증해야 한다.

Kubernetes 도입 시점에는 명확한 Deployment와 Scaling 요구사항을 함께 정의한다.
