# ADR-0001: Monorepo 구조 사용

- 상태: 승인(Approved)
- 결정일: 2026-08-18

## 1. Context

SmartFactory AI Quality Platform은 Vision AI, FastAPI, RAG, Monitoring,
Infrastructure, Dashboard 등 서로 다른 영역을 하나의 시스템에서 구현한다.

초기 단계부터 Repository를 여러 개로 분리하면 각 서비스 간 버전 관리,
공통 설정, 테스트, CI/CD 구성의 복잡도가 불필요하게 증가할 수 있다.

현재 프로젝트 규모에서는 전체 시스템 변경을 하나의 Repository에서 추적하면서
내부 영역의 책임을 명확히 분리하는 방식이 적합하다.

## 2. Decision

전체 프로젝트를 `smart-factory-ai-platform` 하나의 Monorepo로 관리한다.

주요 영역은 다음과 같이 분리한다.

```text
apps/
services/
ml/
pipelines/
configs/
manuals/
monitoring/
infra/
tests/
docs/
```

Repository는 하나로 유지하되 각 디렉터리의 책임과 의존성은 명확하게 구분한다.

## 3. Reason

- 전체 시스템의 변경 이력을 하나의 Git History로 관리할 수 있다.
- API, Model, Infrastructure 변경을 하나의 Pull Request에서 함께 검토할 수 있다.
- 공통 CI/CD와 품질 기준을 적용하기 쉽다.
- 서비스 간 호환 변경을 한 번에 추적하기 쉽다.
- 초기 개발 단계에서 불필요한 Repository 운영 복잡도를 줄일 수 있다.

## 4. Alternatives

### Multi-repository

`api`, `inference`, `rag`, `infra` 등을 각각 별도 Repository로 관리하는 방법을 검토할 수 있다.

하지만 현재 프로젝트 규모에서는 다음 단점이 더 크다고 판단한다.

- Repository 간 버전 동기화 필요
- CI/CD 관리 포인트 증가
- 공통 개발환경 관리 복잡도 증가
- 서비스 간 변경 추적 복잡도 증가

## 5. Consequences

### 장점

- 개발 및 탐색이 단순해진다.
- 시스템 전체의 변경 추적이 쉽다.
- 공통 Tooling을 적용하기 쉽다.

### 단점

- 프로젝트 규모가 커지면 Repository가 복잡해질 수 있다.
- 서비스별 독립 Release가 필요해질 경우 추가 구조가 필요할 수 있다.

향후 팀 규모나 배포 요구사항이 크게 증가하면 Multi-repository 전환을 다시 검토할 수 있다.
