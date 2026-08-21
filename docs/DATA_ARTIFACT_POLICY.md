# Data 및 Artifact 관리 정책

## 1. 목적

이 문서는 SmartFactory AI Quality Platform에서
어떤 파일을 Git으로 관리하고 어떤 파일을 Repository 외부에서
관리할지 정의한다.

목표는 대용량 파일이나 생성 Artifact를 Git에 저장하지 않으면서도
프로젝트의 재현성을 유지하는 것이다.

---

## 2. Source Code

Source Code는 Git으로 관리한다.

예시는 다음과 같다.

- Dataset Loader
- Preprocessing Logic
- Model Implementation
- Evaluation Code
- API Code
- RAG Code
- Infrastructure Definition
- Test Code

---

## 3. Dataset 정책

Raw Dataset은 Git Repository에 Commit하지 않는다.

Local Dataset은 다음 구조를 사용한다.

```text
data/
├── raw/
├── interim/
└── processed/
```

`data/` 전체는 `.gitignore` 대상으로 관리한다.

반면 `ml/datasets/`는 실제 Dataset 저장공간이 아니다.

다음 Source Code가 위치한다.

- Dataset Download
- Dataset Validation
- Dataset Loading
- Preprocessing
- Dataset Split
- Dataset Metadata

따라서 `ml/datasets/`는 Git으로 관리한다.

---

## 4. MVTec AD 정책

초기 산업용 Anomaly Detection Dataset으로 MVTec AD를 사용한다.

Repository에는 다음 내용만 포함한다.

- Download 방법
- Dataset Metadata
- Validation Logic
- Preprocessing Logic
- 필요한 경우 Manifest
- License 정보

MVTec AD 원본 Dataset 자체는 Repository에서 재배포하지 않는다.

---

## 5. Model Artifact 정책

생성된 Model Artifact는 Git에 직접 Commit하지 않는다.

예시는 다음과 같다.

```text
models/
checkpoints/
artifacts/
outputs/
```

해당 경로는 `.gitignore` 대상으로 관리한다.

향후 Model Artifact는 필요에 따라 다음 시스템을 이용한다.

- MLflow Artifact Store
- GCP Cloud Storage
- 기타 별도로 선택한 Artifact Storage

---

## 6. Experiment Metadata

Experiment Configuration 및 작은 Benchmark 결과는
재현성에 도움이 되는 경우 Git에 저장할 수 있다.

대용량 Experiment Artifact는 Git에 저장하지 않는다.

---

## 7. MLflow

MLflow에서는 다음 정보를 관리한다.

- Experiment
- Run
- Parameter
- Metric
- Dataset / Model / Threshold / Evaluation / Benchmark lineage artifact copy

Project-native `artifacts/`, `outputs/` 계약이 source of truth이며 MLflow는 검증된 결과를 추적하는
계층이다. Local 개발은 SQLite metadata backend와 local artifact store를 사용한다. Remote tracking
server와 Model Registry 운영은 후속 deployment 단계에서 결정한다.

---

## 8. Secret 관리

Secret은 Git Repository에 Commit하지 않는다.

예시는 다음과 같다.

- API Key
- Database Password
- Cloud Credential
- Private Service Token

Local Secret은 `.env`에 저장한다.

Repository에는 `.env.example`만 Commit한다.

## 8.1 RAG manual 및 index

Repository에는 `PROJECT DEMO SOP — NOT AN ACTUAL FACTORY PROCEDURE`로 명시된 작은 공개 demo manual만 저장한다.
실제 회사 SOP/manual은 source repository에 Commit하지 않으며 access-controlled storage와 delivery policy를
사용한다.

Generated RAG index의 `metadata.json`, `chunks.jsonl`, `embeddings.npy`는
`artifacts/rag/manuals/<index-id>/` 아래 artifact이며 Git에 저장하지 않는다. Index에는 corpus-relative source
path, source SHA와 provider/model identity만 기록하고 API key나 local absolute path를 기록하지 않는다.

Versioned public RAG evaluation dataset은 `configs/evaluation/`에 Commit한다. Generated evaluation summary와 case
evidence는 `outputs/evaluation/rag/<evaluation-id>/` 아래 immutable artifact이며 Git에 저장하지 않는다. Dataset
SHA, index metadata/corpus/chunking/embedding lineage와 output hash를 함께 기록하고 credential은 저장하지 않는다.

---

## 9. 향후 Cloud Artifact 구조

예상 구조는 다음과 같다.

```text
GitHub
   │
   ├── Source Code
   ├── Configuration
   ├── Architecture Document
   └── Benchmark Summary

GCP Cloud Storage
   │
   ├── Dataset Artifact
   └── Model Artifact

MLflow
   │
   ├── Experiment Metadata
   └── Model Metadata

PostgreSQL
   │
   ├── Inspection History
   ├── Application Data
   └── Vector Data
```

향후 시스템 요구사항에 따라 저장 구조는 변경될 수 있다.
