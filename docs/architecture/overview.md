# SmartFactory AI Quality Platform — Architecture Overview

## 1. 전체 Architecture

SmartFactory AI Quality Platform은 크게 다음 네 영역으로 구분한다.

1. Machine Learning
2. Application Service
3. Platform / Infrastructure
4. Quality Knowledge RAG

전체 Repository는 Monorepo로 관리하지만,
각 영역의 책임과 의존성은 명확하게 분리한다.

---

## 2. High-Level Architecture

```text
Manufacturing Image
        │
        ▼
   Data Pipeline
        │
        ▼
     Vision AI
(PatchCore Baseline)
        │
        ├───────────────┐
        ▼               ▼
   Evaluation         MLflow
        │               │
        └───────┬───────┘
                ▼
        Inference Layer
                │
                ▼
             FastAPI
                │
        ┌───────┴───────┐
        ▼               ▼
   PostgreSQL         Metrics
검사 이력 저장            │
                        ▼
                   Prometheus
                        │
                        ▼
                     Grafana
```

Production 환경은 후반 단계에서 Kubernetes 위에 구성한다.

```text
GitHub
   │
   ▼
CI/CD
   │
   ▼
Docker Image
   │
   ▼
GCP / Kubernetes
   │
   ├── Deployment
   ├── Autoscaling
   ├── Rolling Update
   └── Rollback
```

---

## 3. Machine Learning 영역

`ml/` 영역은 다음 책임을 가진다.

- Dataset Loading
- Dataset Validation
- Preprocessing
- Model Training / Construction
- Anomaly Detection Experiment
- Model Evaluation
- Benchmark 생성

API 또는 Dashboard의 Business Logic을 포함하지 않는다.

---

## 4. Inference 영역

Inference Layer는 모델 구현과 API 사이에 안정적인 Interface를 제공한다.

개념적인 구조는 다음과 같다.

```text
API
 │
 ▼
Inference Interface
 │
 ├── PatchCore Adapter
 │
 └── Future Model Adapter
```

API Layer가 PatchCore 내부 구현에 직접 의존하지 않도록 한다.

이를 통해 향후 모델을 교체하더라도 HTTP API Layer를
대규모로 수정하지 않도록 한다.

---

## 5. API 영역

`services/api`는 다음 책임을 가진다.

- HTTP Endpoint
- Request Validation
- Response Schema
- API Level Exception Handling
- 검사 Workflow Orchestration

Model 내부 구현은 API Layer에 포함하지 않는다.

---

## 6. Persistence 영역

PostgreSQL은 초기에는 다음 데이터를 저장한다.

- 검사 이력
- Prediction Metadata
- Model Version Reference
- Application 운영 데이터

STEP 13의 작은 demo SOP corpus는 PostgreSQL/pgvector에 넣지 않고 immutable normalized NumPy matrix와 exact cosine
retrieval을 사용한다. Corpus latency/memory가 실제 한계를 넘을 때 retrieval abstraction 뒤에서 pgvector 또는
다른 vector backend를 검토한다. Inspection history와 민감할 수 있는 RAG query log를 한 table에 섞지 않는다.

---

## 7. RAG Architecture

RAG 시스템은 특정 Model Provider에 종속되지 않도록 구성한다.

```text
QualityRAGService
        │
        ├── Retriever
        ├── EmbeddingProvider
        └── AnswerGenerator
```

예시는 다음과 같다.

```text
LLMProvider
   ├── OpenAI
   ├── Local LLM
   └── Other API Provider

EmbeddingProvider
   ├── OpenAI Embedding
   └── Local Embedding Model

Retriever
   ├── Exact cosine / immutable NumPy index (current)
   └── PostgreSQL / pgvector or other backend (future)
```

Provider별 SDK 사용 코드는 Provider Adapter 내부에 격리한다.

---

## 8. Compute Architecture

### Local macOS

주 개발환경으로 사용한다.

### Kaggle GPU

Vision AI의 임시 GPU 실험환경으로 사용한다.

### GCP

최종 Benchmark 및 Production 유사 Infrastructure 환경으로 사용한다.

환경별로 별도의 Source Code Branch를 만드는 방식은 사용하지 않는다.

동일한 Source Code를 사용하고,
환경 차이는 Configuration과 Environment Variable로 처리한다.

---

## 9. Configuration 원칙

Configuration과 Source Code를 분리한다.

Configuration 예시는 다음과 같다.

- Dataset Path
- Model Parameter
- Serving Configuration
- Monitoring Threshold
- Environment별 Service URL

Secret은 Git에 Commit하지 않는다.

---

## 10. Architecture 발전 원칙

Architecture는 단순한 구조에서 시작한다.

실제 운영 또는 Engineering Requirement가 발생했을 때만
Component를 추가로 분리한다.

예를 들어 초기에는 FastAPI Process 내부에서 Inference가 실행될 수 있다.

향후 다음 요구가 발생한다면 별도의 Inference Service로 분리할 수 있다.

- Independent Scaling
- GPU Resource Isolation
- Resource Scheduling
- Deployment Independence

즉 Microservice 자체를 목표로 하지 않고,
필요성이 발생했을 때 분리한다.
