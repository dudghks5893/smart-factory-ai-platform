# SmartFactory AI Quality Platform

제조 이미지 기반 이상 탐지 모델을 개발하는 데서 끝나지 않고,
데이터 파이프라인부터 모델 평가, Serving, MLOps, Monitoring,
Kubernetes 운영, 제조 매뉴얼 기반 RAG까지 연결하는
Production AI 포트폴리오 프로젝트입니다.

> 현재 상태: **STEP 0 — 프로젝트 기준 및 Repository 구축**

---

## 1. 프로젝트 목표

실제 제조 품질검사 환경을 가정하여 다음 전체 흐름을 구현합니다.

```text
제조 이미지
    ↓
Data Pipeline
    ↓
Vision AI 이상/불량 탐지
    ↓
Model Evaluation / Benchmark
    ↓
MLflow Experiment / Model Registry
    ↓
FastAPI Model Serving
    ↓
PostgreSQL 검사 이력
    ↓
Docker / Docker Compose
    ↓
GitHub Actions CI/CD
    ↓
Prometheus / Grafana Monitoring
    ↓
Data Drift Detection
    ↓
Kubernetes
    ↓
Autoscaling / Rolling Update / Rollback
    ↓
제조 Dashboard
    ↓
제조 Manual / SOP 기반 RAG
    ↓
RAG Evaluation
```

단순 Notebook 데모가 아니라 모델이 실제 서비스 환경에서
배포되고, 관찰되고, 버전 관리되고, 교체될 수 있는 시스템을 목표로 합니다.

---

## 2. 주요 기술 방향

### Vision / ML

- Python
- PyTorch
- OpenCV
- MVTec AD
- PatchCore 계열 Anomaly Detection

### Backend / Data

- FastAPI
- PostgreSQL
- pgvector

### MLOps / Platform

- MLflow
- Docker
- Docker Compose
- GitHub Actions
- Prometheus
- Grafana
- Kubernetes
- GCP

### RAG

- 제조 Manual / SOP
- Retrieval
- Citation
- Faithfulness Evaluation
- OpenAI 또는 Local Model을 포함한 Provider-Agnostic Architecture

### Quality

- pytest
- Ruff
- mypy

---

## 3. Compute 전략

환경은 목적에 따라 분리합니다.

| 환경 | 주요 목적 |
|---|---|
| Local macOS | 개발, 테스트, Dataset Pipeline, Docker, Local PostgreSQL/MLflow |
| Kaggle GPU | 초기 Vision AI GPU 실험, PatchCore 실험, 후보 모델 비교 |
| GCP | 최종 Benchmark, Production-like Deployment, Kubernetes, Monitoring |

Kaggle은 실험 환경으로 사용하고,
최종 성능 검증 및 Production 환경은 GCP에서 구성합니다.

환경별 별도 Source Code를 만들지 않고
동일한 코드와 Environment-specific Configuration을 사용하는 것을 원칙으로 합니다.

---

## 4. Repository 구조

```text
smart-factory-ai-platform/
├── apps/
│   └── dashboard/
├── services/
│   ├── api/
│   ├── inference/
│   └── rag/
├── ml/
│   ├── datasets/
│   ├── training/
│   └── evaluation/
├── pipelines/
├── configs/
│   ├── data/
│   ├── model/
│   ├── serving/
│   └── monitoring/
├── manuals/
├── monitoring/
│   ├── prometheus/
│   └── grafana/
├── infra/
│   ├── docker/
│   ├── kubernetes/
│   └── cloud/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── docs/
│   ├── architecture/
│   ├── adr/
│   ├── benchmarks/
│   └── runbooks/
├── scripts/
└── .github/
    └── workflows/
```

---

## 5. 개발환경

기본 Python Version:

```text
Python 3.12.x
```

Python Project 및 Dependency 관리는 `uv`를 사용합니다.

### 환경 동기화

```bash
uv sync
```

### 전체 품질 검사

```bash
make check
```

현재 `make check`는 다음 순서로 실행됩니다.

```text
Ruff Format Check
        ↓
Ruff Lint
        ↓
mypy Type Check
        ↓
pytest
```

### 자동 Format

```bash
make format
```

---

## 6. 평가 지표

프로젝트 전반에서 다음 지표를 기록합니다.

### Vision AI

- AUROC
- Precision
- Recall
- F1 Score
- Image-level / Pixel-level Metrics

### Serving

- p50 Latency
- p95 Latency
- p99 Latency
- Throughput
- API Error Rate

### Resource

- Model Size
- Memory Bank Size
- CPU Memory
- GPU Memory

### Deployment

- Deployment 성공 여부
- Health Check
- Rolling Update
- Rollback

### Drift

- Input / Feature / Embedding / Anomaly Score Distribution 기반 Drift

### RAG

- Recall@K
- Citation Accuracy
- Faithfulness
- Latency
- 필요 시 Cost

세부 측정 원칙은
`docs/benchmarks/METRICS_CONTRACT.md`에서 관리합니다.

---

## 7. Architecture Decision Record

주요 설계 선택은 `docs/adr/`에 ADR 형태로 기록합니다.

현재 결정된 내용은 다음과 같습니다.

1. Monorepo 사용
2. Python 3.12 + uv 사용
3. PatchCore 계열을 초기 Baseline으로 사용
4. PostgreSQL + pgvector 사용
5. Kubernetes는 후반 단계에서 도입
6. Local macOS + Kaggle GPU + GCP Production Hybrid Compute
7. Provider-Agnostic RAG Architecture

---

## 8. Data / Artifact 정책

다음 항목은 Git에 직접 저장하지 않습니다.

- MVTec AD 원본 Dataset
- Model Checkpoint
- Model Artifact
- MLflow Artifact
- Secret
- 대용량 Output

실제 Dataset은 Local의 `data/` 아래에서 관리하고,
Dataset 관련 Source Code는 `ml/datasets/`에서 관리합니다.

자세한 정책은 `docs/DATA_ARTIFACT_POLICY.md`를 참고합니다.

---

## 9. 공식 구현 순서

- [x] STEP 0. 프로젝트 기준 및 Repository 구축
- [ ] STEP 1. Dataset / Data Pipeline
- [ ] STEP 2. Vision AI Baseline
- [ ] STEP 3. 모델 평가 및 Benchmark
- [ ] STEP 4. FastAPI Model Serving
- [ ] STEP 5. PostgreSQL 검사 이력
- [ ] STEP 6. MLflow Experiment / Model Registry
- [ ] STEP 7. Docker
- [ ] STEP 8. CI/CD
- [ ] STEP 9. Prometheus / Grafana Monitoring
- [ ] STEP 10. Data Drift Detection
- [ ] STEP 11. Kubernetes
- [ ] STEP 12. 제조 Dashboard
- [ ] STEP 13. 제조 Manual / SOP RAG
- [ ] STEP 14. RAG Evaluation
- [ ] STEP 15. 최종 성능 / Latency / 비용 Benchmark
- [ ] STEP 16. README / Architecture / 취업 Portfolio 완성

---

## 10. 현재 진행 상태

STEP 0에서는 다음 기반을 구축했습니다.

- Monorepo Directory
- Python 3.12
- uv 기반 Virtual Environment / Dependency 관리
- Ruff
- mypy
- pytest
- Makefile 기반 Quality Gate
- Project Scope
- Architecture Overview
- Data / Artifact Policy
- Metrics Contract
- ADR 0001~0007

다음 단계는 **STEP 1 — MVTec AD Dataset / Data Pipeline 구축**입니다.

---

## 11. 상세 문서

- `docs/PROJECT_SCOPE.md`
- `docs/architecture/overview.md`
- `docs/DATA_ARTIFACT_POLICY.md`
- `docs/benchmarks/METRICS_CONTRACT.md`
- `docs/adr/`
