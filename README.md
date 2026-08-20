# SmartFactory AI Quality Platform

제조 이미지 기반 이상 탐지부터 데이터 파이프라인, 모델 평가, Serving,
MLOps, Monitoring, Kubernetes 운영, 제조 매뉴얼 기반 RAG까지 연결하는
Production AI 시스템 프로젝트입니다.

> 현재 상태: **STEP 4 — FastAPI Model Serving 완료**

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

단순한 모델 실험이 아니라 모델이 실제 서비스 환경에서
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
- Provider-Agnostic Architecture

### Quality

- pytest
- Ruff
- mypy

---

## 3. Compute 전략

| 환경 | 주요 목적 |
|---|---|
| Local macOS | 개발, 테스트, Dataset Pipeline, Docker, Local PostgreSQL/MLflow |
| Kaggle GPU | 초기 Vision AI GPU 실험, PatchCore 실험, 후보 모델 비교 |
| GCP | 최종 Benchmark, Production-like Deployment, Kubernetes, Monitoring |

환경별 별도 Source Code를 만들지 않고 동일한 코드와
Environment-specific Configuration을 사용하는 것을 원칙으로 합니다.

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
│   ├── data/
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

## 6. Dataset / Data Pipeline

초기 Dataset으로 MVTec AD를 사용하며 첫 Category는 `metal_nut`입니다.

현재 Pipeline은 다음을 지원합니다.

```text
Raw MVTec AD
    ↓
Dataset Structure / Image Validation
    ↓
Deterministic Train / Validation Split
    ↓
Manifest 생성
    ↓
Manifest Integrity Validation
    ↓
PyTorch Dataset
    ↓
DataLoader
```

`metal_nut` 기준 현재 Split은 다음과 같습니다.

| Split | Samples |
|---|---:|
| Train | 198 |
| Validation | 22 |
| Test Good | 22 |
| Test Anomaly | 93 |
| Total | 335 |

원본 MVTec AD의 `test`는 분할하거나 Threshold 튜닝에 사용하지 않습니다.

Dataset Pipeline 세부 내용은 `docs/data/MVTEC_AD_PIPELINE.md`에서 관리합니다.

PatchCore 전처리, memory bank artifact 및 raw prediction pipeline은
`docs/vision/PATCHCORE_BASELINE.md`에서 관리합니다.

STEP 3 PatchCore evaluation과 Tesla T4 inference benchmark까지 완료했습니다.

| Baseline | Result |
|---|---:|
| Image AUROC / F1 | 0.997556 / 0.994595 |
| Pixel AUROC / F1 | 0.982486 / 0.834279 |
| Image FP / FN | 0 / 1 |
| T4 inference p50 / p95 / p99 | 21.634 / 25.775 / 27.113 ms |
| T4 offline model throughput | 45.114 images/second |
| T4 FastAPI in-process HTTP p50 | 44.902 ms |
| T4 FastAPI throughput / success | 22.030 requests/second / 115 of 115 |

Offline throughput은 batch size 1에서 disk I/O, artifact restore와 warmup을 제외한 값입니다. 56초의
training wall time과 CLI 전체 runtime은 inference latency가 아닙니다. FastAPI 결과는 disk image loading,
artifact restore, warmup, external network RTT와 uvicorn/socket/TLS/proxy를 제외한 in-process
application-level HTTP E2E 기준입니다. 세부 계약과 실측값은
`docs/benchmarks/PATCHCORE_EVALUATION.md`와
`docs/benchmarks/PATCHCORE_INFERENCE_BENCHMARK.md`에서 관리합니다.

FastAPI serving core와 real-artifact smoke/HTTP benchmark tooling의 lifecycle, endpoint, 측정 계약은
`docs/serving/PATCHCORE_API.md`에서 관리합니다.

---

## 7. 평가 지표

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

세부 측정 원칙은 `docs/benchmarks/METRICS_CONTRACT.md`에서 관리합니다.

---

## 8. Architecture Decision Record

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

## 9. Data / Artifact 정책

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

## 10. 공식 구현 순서

- [x] STEP 0. 프로젝트 기준 및 Repository 구축
- [x] STEP 1. Dataset / Data Pipeline
- [x] STEP 2. Vision AI Baseline
- [x] STEP 3. 모델 평가 및 Benchmark
- [x] STEP 4. FastAPI Model Serving (Real-model smoke 및 in-process HTTP E2E 실측 완료)
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
- [ ] STEP 16. README / Architecture 최종 정리

---

## 11. 상세 문서

- `docs/CODING_CONVENTIONS.md`
- `docs/PROJECT_SCOPE.md`
- `docs/architecture/overview.md`
- `docs/DATA_ARTIFACT_POLICY.md`
- `docs/benchmarks/METRICS_CONTRACT.md`
- `docs/serving/PATCHCORE_API.md`
- `docs/data/MVTEC_AD_PIPELINE.md`
- `docs/adr/`
