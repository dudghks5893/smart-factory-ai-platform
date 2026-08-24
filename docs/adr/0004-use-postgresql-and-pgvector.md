# ADR-0004: PostgreSQL 사용과 pgvector 검토

- 상태: 부분 대체(Amended — PostgreSQL 승인, pgvector 도입 보류)
- 결정일: 2026-08-18
- 수정일: 2026-08-22

## 1. Context

프로젝트에서는 제조 검사 이력과 Application Data를 저장할 Relational Database가 필요하다.

후반 RAG 단계에서는 제조 매뉴얼/SOP의 Embedding Vector를 저장하고
Similarity Search를 수행할 Vector Storage도 필요하다.

초기부터 Relational Database와 Vector Database를 각각 별도 제품으로 분리하면
운영 및 Infrastructure 복잡도가 증가한다.

## 2. Original decision

기본 Database로 PostgreSQL을 사용한다.

초기 계획에서는 RAG Vector Search에 PostgreSQL Extension인 `pgvector`를 사용하는 방안을 선택했다.

## 3. Amendment

STEP 13 구현 시점에 demo SOP corpus 규모와 운영 복잡도를 다시 평가한 결과, inspection history에는 PostgreSQL을
사용하고 RAG에는 immutable normalized NumPy matrix와 exact cosine retrieval을 사용하도록 결정을 수정했다.
`pgvector`는 corpus latency/memory가 실제 한계를 넘는 경우 retrieval abstraction 뒤에서 검토하는 future candidate다.

현재 저장 경계는 다음과 같다.

```text
PostgreSQL
├── 검사 이력
├── Prediction Metadata
├── Model Version Reference
└── Application Data

Immutable RAG artifact
├── Document/chunk metadata
└── Normalized float32 embedding matrix
```

## 4. Reason

- 제조 검사 이력은 명확한 Relational Data 구조를 가진다.
- PostgreSQL은 Production 환경에서 널리 사용되는 Database다.
- 현재 작은 corpus는 exact search가 충분하며 별도 extension/schema lifecycle이 필요하지 않다.
- Inspection transactional data와 RAG artifact lifecycle을 분리할 수 있다.
- Local Docker와 GCP 환경에서 동일한 Database 기술을 사용할 수 있다.
- 초기 Infrastructure 복잡도를 줄일 수 있다.

## 5. Alternatives

### 별도 Vector Database

예: Qdrant, Weaviate, Milvus 등.

대규모 Vector Search나 Vector-specific Feature가 필요해질 경우 장점이 있을 수 있다.

하지만 초기 RAG 규모에서는 운영 Component를 하나 더 추가하는 비용이 크다고 판단한다.

### 다른 Relational Database

MySQL 등도 가능하지만 inspection workload, SQLAlchemy/Alembic support와 managed Cloud SQL target을 고려해
PostgreSQL을 선택한다.

## 6. Consequences

### 장점

- Database Architecture가 단순하다.
- Local과 Cloud 환경을 통일하기 쉽다.
- 작은 demo RAG artifact를 재현하고 검증하기 쉽다.

### 단점

- In-memory exact search는 corpus 규모가 커지면 latency와 memory 측면에서 불리할 수 있다.
- PostgreSQL과 RAG index의 backup/delivery lifecycle을 별도로 관리해야 한다.

실제 corpus benchmark에서 exact retrieval이 한계를 넘는 경우 pgvector 또는 별도 Vector Database를 재검토한다.
