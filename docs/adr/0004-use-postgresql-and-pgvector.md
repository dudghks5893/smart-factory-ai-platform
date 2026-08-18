# ADR-0004: PostgreSQL 및 pgvector 사용

- 상태: 승인(Approved)
- 결정일: 2026-08-18

## 1. Context

프로젝트에서는 제조 검사 이력과 Application Data를 저장할 Relational Database가 필요하다.

후반 RAG 단계에서는 제조 매뉴얼/SOP의 Embedding Vector를 저장하고
Similarity Search를 수행할 Vector Storage도 필요하다.

초기부터 Relational Database와 Vector Database를 각각 별도 제품으로 분리하면
운영 및 Infrastructure 복잡도가 증가한다.

## 2. Decision

기본 Database로 PostgreSQL을 사용한다.

RAG Vector Search는 초기에는 PostgreSQL Extension인 `pgvector`를 사용한다.

초기 저장 대상은 다음과 같다.

```text
PostgreSQL
├── 검사 이력
├── Prediction Metadata
├── Model Version Reference
├── Application Data
└── RAG Vector Data
```

## 3. Reason

- 제조 검사 이력은 명확한 Relational Data 구조를 가진다.
- PostgreSQL은 Production 환경에서 널리 사용되는 Database다.
- pgvector를 이용하면 별도 Vector Database 없이 RAG를 시작할 수 있다.
- Transactional Data와 Vector Metadata를 함께 관리하기 쉽다.
- Local Docker와 GCP 환경에서 동일한 Database 기술을 사용할 수 있다.
- 초기 Infrastructure 복잡도를 줄일 수 있다.

## 4. Alternatives

### 별도 Vector Database

예: Qdrant, Weaviate, Milvus 등.

대규모 Vector Search나 Vector-specific Feature가 필요해질 경우 장점이 있을 수 있다.

하지만 초기 RAG 규모에서는 운영 Component를 하나 더 추가하는 비용이 크다고 판단한다.

### 다른 Relational Database

MySQL 등도 가능하지만 pgvector 활용성과 프로젝트 확장성을 고려해 PostgreSQL을 선택한다.

## 5. Consequences

### 장점

- Database Architecture가 단순하다.
- Local과 Cloud 환경을 통일하기 쉽다.
- RAG Prototype을 빠르게 구축할 수 있다.

### 단점

- Vector 규모가 매우 커지면 별도 Vector Database보다 성능상 불리할 수 있다.
- Vector Search 요구사항이 복잡해질 경우 pgvector만으로 부족할 수 있다.

실제 Benchmark에서 pgvector가 병목이 되는 경우 별도 Vector Database 도입을 재검토한다.
