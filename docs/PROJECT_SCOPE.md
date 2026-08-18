# SmartFactory AI Quality Platform — 프로젝트 범위

## 1. 프로젝트 목적

SmartFactory AI Quality Platform은 실제 제조 품질검사 환경을 가정하여
AI 모델의 개발부터 배포와 운영까지 전체 생명주기를 구현하는
Production AI 시스템 프로젝트이다.

단순한 AI 모델 데모나 Notebook 실험으로 끝내지 않고 다음 과정을
하나의 시스템으로 연결하는 것을 목표로 한다.

- 제조 이미지 수집 및 입력
- 데이터 검증 및 전처리
- Vision AI 기반 이상/불량 탐지
- 모델 평가 및 Benchmark
- 실험 및 모델 버전 관리
- Model Serving
- 검사 이력 저장
- Container 기반 실행환경 구축
- CI/CD
- 운영 Monitoring
- Data Drift Detection
- Kubernetes 기반 배포 및 운영
- 제조 품질 Dashboard
- 제조 매뉴얼/SOP 기반 RAG
- RAG 성능 평가

최종적으로 AI 모델이 높은 성능을 내는 것뿐 아니라
실제 서비스 환경에서 배포되고, 관찰되고, 관리되고, 교체될 수 있는
시스템을 구현하는 것을 목표로 한다.

---

## 2. 전체 시스템 흐름

```text
제조 이미지
    ↓
Data Pipeline
    ↓
Vision AI 이상 탐지
    ↓
Model Evaluation
    ↓
MLflow Experiment / Model Registry
    ↓
FastAPI Model Serving
    ↓
PostgreSQL 검사 이력
    ↓
Docker
    ↓
GitHub Actions CI/CD
    ↓
Prometheus / Grafana Monitoring
    ↓
Data Drift Detection
    ↓
Kubernetes Deployment
    ↓
Autoscaling / Rolling Update / Rollback
    ↓
제조 Dashboard
    ↓
제조 SOP / Manual RAG
    ↓
RAG Evaluation
```

---

## 3. 주요 기술 방향

초기 기술 스택은 다음과 같다.

- Python
- PyTorch
- OpenCV
- MVTec AD
- PatchCore 계열 Anomaly Detection
- FastAPI
- PostgreSQL
- pgvector
- MLflow
- Docker
- Docker Compose
- GitHub Actions
- Prometheus
- Grafana
- Kubernetes
- GCP
- pytest

RAG 영역은 특정 LLM Provider에 종속되지 않도록 설계한다.

사용 가능한 후보는 다음과 같다.

- OpenAI API
- Local LLM
- Local Embedding Model
- 기타 외부 Model API
- Reranker Model

Application Layer가 특정 LLM SDK에 직접 종속되지 않도록 한다.

---

## 4. Compute 전략

개발과 실행환경은 목적에 따라 분리한다.

### Local macOS

다음 작업에 사용한다.

- Application 개발
- Unit Test
- Integration Test
- Dataset Pipeline 개발
- CPU/MPS Smoke Test
- Docker 개발
- Local PostgreSQL
- Local MLflow

### Kaggle GPU

다음 작업에 사용한다.

- 초기 Vision AI GPU 실험
- PatchCore 실험
- 후보 모델 비교
- 비용을 최소화한 반복 실험

Kaggle은 실험 환경이며 Production 환경으로 사용하지 않는다.

### GCP

다음 작업에 사용한다.

- 최종 재현성 Benchmark
- GPU Inference Benchmark
- Production 유사 환경 배포
- Kubernetes
- Autoscaling
- Rolling Update
- Rollback
- Monitoring
- 최종 시스템 검증

---

## 5. 초기 Machine Learning 범위

초기 산업용 Anomaly Detection Dataset으로 MVTec AD를 사용한다.

첫 Baseline Model은 PatchCore 계열을 우선 검토한다.

PatchCore를 선택하는 이유는 제조 이미지의 이상 탐지 문제에 적합하고,
Production AI 시스템 구축과 자연스럽게 연결할 수 있기 때문이다.

Model Serving Layer는 특정 PatchCore 구현에 직접 의존하지 않도록 구성한다.

향후 비교 가치가 있을 경우 다음 모델을 추가 검토할 수 있다.

- PaDiM
- EfficientAD
- FastFlow
- Custom PyTorch Model

기술 스택을 늘리기 위한 목적으로 모델을 추가하지 않는다.

---

## 6. RAG 범위

RAG 시스템은 제조 매뉴얼 및 SOP 문서를 기반으로
품질 관련 질문에 답변하는 Quality Assistant를 구현하는 것을 목표로 한다.

다음 영역을 분리해서 설계한다.

- Document Ingestion
- Chunking
- Embedding
- Vector Storage
- Retrieval
- Reranking
- Generation
- Citation
- Evaluation

초기 구현에서 OpenAI 서비스를 사용할 수 있으나
Architecture 자체가 OpenAI에 종속되어서는 안 된다.

향후 Local LLM이나 Local Embedding Model로 교체할 수 있어야 한다.

---

## 7. 초기 개발 범위에서 제외하는 기술

다음 기술은 단순히 기술 스택을 늘리기 위한 목적으로 사용하지 않는다.

- Kafka
- Airflow
- Kubeflow
- KServe
- Distributed Training
- Multi-region Deployment
- 대규모 Streaming Architecture
- Multi-tenant SaaS Architecture
- 복잡한 Authentication System
- 실제 PLC 연동
- 실제 MES 연동

향후 명확한 시스템 요구사항이나 측정 가능한 필요성이 발생할 경우에만 도입한다.

---

## 8. 개발 원칙

모든 주요 기술 및 Architecture 선택에는 이유를 기록한다.

프로젝트에서는 다음을 우선한다.

- Reproducibility
- 측정 가능한 성능
- Maintainability
- Testability
- Observability
- 명확한 Service Boundary
- 불필요한 Infrastructure 복잡도 최소화
- 실제 서비스 수준의 Error Handling

Infrastructure 복잡도는 해당 Infrastructure를 필요로 하는 기능이
완성된 이후 단계적으로 높인다.

---

## 9. 시스템 설명 가능성

최종 시스템에서는 다음 질문에 명확한 기술적 근거로 답할 수 있어야 한다.

- 왜 이 기술을 선택했는가?
- 어떤 대안을 검토했는가?
- 모델 성능은 어떻게 평가했는가?
- Inference Latency는 어떻게 측정했는가?
- 장애 상황은 어떻게 처리하는가?
- Rollback은 어떻게 수행하는가?
- Data Drift는 어떻게 감지하는가?
- RAG 성능은 어떻게 평가하는가?
- Cloud 비용은 어떻게 관리하는가?

주요 설계 결정과 측정 방법은 코드와 문서에서 추적 가능해야 한다.
