# SmartFactory AI Quality Platform — Metrics Contract

## 1. 목적

이 문서는 SmartFactory AI Quality Platform에서
공통으로 사용할 평가 지표와 측정 원칙을 정의한다.

모델 최적화 이후 유리한 지표만 선택하는 것을 방지하기 위해
주요 평가 기준을 모델 개발 전에 정의한다.

---

# Vision AI

## 2. Anomaly Detection 평가 지표

가능한 경우 다음 지표를 기록한다.

- AUROC
- Precision
- Recall
- F1 Score

Model Output에 따라 다음 두 수준의 평가를 구분한다.

- Image-level
- Pixel-level

---

## 3. Threshold 결정 원칙

Precision, Recall, F1 Score를 계산하려면
Anomaly Threshold가 필요하다.

Test Set을 사용하여 Threshold를 최적화하지 않는다.

기본 평가 흐름은 다음과 같다.

```text
Training Data
      │
      ▼
Model Construction
      │
      ▼
Validation Data
      │
      ▼
Threshold Selection
      │
      ▼
Test Data
      │
      ▼
Final Evaluation
```

최종 Benchmark에는 Threshold 결정 방법도 함께 기록한다.

---

## 4. AUROC

AUROC는 Threshold에 독립적인
Anomaly 구분 성능 평가 지표로 사용한다.

가능한 경우 다음을 별도로 기록한다.

- Image-level AUROC
- Pixel-level AUROC

---

# Serving

## 5. Inference Latency

Latency는 평균값만 기록하지 않는다.

다음 Percentile을 기록한다.

- p50
- p95
- p99

Latency는 두 종류로 구분한다.

### Model Latency

다음 범위를 포함한다.

```text
Preprocessing
+
Model Inference
+
Postprocessing
```

### API Latency

전체 Request 처리 시간을 측정한다.

```text
Request Handling
+
Image Decoding
+
Preprocessing
+
Inference
+
Postprocessing
+
필요한 Persistence
+
Response Generation
```

---

## 6. Throughput

상황에 따라 다음 단위를 사용한다.

- images/sec
- requests/sec

Throughput 결과에는 반드시 다음 조건을 함께 기록한다.

- Hardware
- Batch Size
- Concurrency
- Model Version
- Input Resolution

---

# Resource

## 7. Model Size

PatchCore 계열 모델에서는 단순 Neural Network Weight 크기만
Model Size로 보지 않는다.

다음 항목을 별도로 기록할 수 있다.

- Backbone Size
- Memory Bank Size
- Total Deployable Artifact Size

---

## 8. Memory

가능한 경우 다음을 기록한다.

- Process RSS Memory
- Peak CPU Memory
- Peak GPU Memory

Memory 결과에는 실행 Hardware와 Environment를 함께 기록한다.

---

# API Reliability

## 9. API Error Rate

API 운영 상태 평가를 위해 Error Rate를 기록한다.

최소한 다음 요청을 구분할 수 있어야 한다.

- Successful Request
- Failed Request

향후 Monitoring 단계에서는 필요에 따라 다음 기준으로 구분한다.

- HTTP Status Family
- Endpoint
- Error Type
- Model Serving Failure

---

# Deployment

## 10. Deployment

새 Application 또는 Model Version 배포 시 다음을 확인한다.

- 정상적으로 Ready 상태가 되었는가
- Health Check를 통과했는가
- 실제 Request를 정상 처리하는가

---

## 11. Rollback

Rollback은 문서로만 설명하지 않고 실제로 검증한다.

최종 프로젝트에서는 다음 내용을 기록한다.

- Deployment Version
- Rollback Target
- Rollback 성공 여부
- Service Recovery 결과

---

# Drift

## 12. Data Drift

안정적인 Model 및 Serving Pipeline이 완성된 이후
Data Drift Monitoring을 구현한다.

어떤 Representation을 Monitoring하는지에 따라
적절한 Drift Metric을 선택한다.

후보는 다음과 같다.

- Input Statistics
- Embedding Distribution
- Anomaly Score Distribution
- Feature Distribution

Monitoring 대상이 결정되기 전에 특정 Drift Library를
먼저 선택하지 않는다.

---

# RAG

## 13. Retrieval Recall@K

Answerable case별 unique expected evidence 중 threshold 적용 후 top-K 결과에 포함된 비율을 계산한다.

- `document_recall_at_k`: expected document ID 기준
- `chunk_recall_at_k`: index-lineage에 종속된 exact expected chunk ID 기준

Multi-evidence case는 fractional recall을 허용하고 answerable case의 macro average를 보고한다. Unanswerable case는
Recall denominator에서 제외한다. `mean_reciprocal_rank`는 첫 expected chunk raw rank의 reciprocal macro average다.

사용한 K 값과 Evaluation Dataset을 함께 기록한다.

---

## 14. Citation Accuracy

STEP 13 structural allow-list validation과 source correctness를 구분한다.

- `citation_precision`: unique cited chunk 중 exact expected chunk의 비율
- `citation_recall`: expected chunk 중 실제 citation으로 사용된 비율

Answerable case별로 계산한 뒤 macro average하며 no citation은 두 metric 모두 0이다.

---

## 15. Faithfulness

생성된 답변 claim이 marker로 연결된 Retrieved Evidence에서 지원되는지 평가한다. STEP 14 deterministic baseline은
answer의 non-empty line을 claim으로 보고 normalized claim text가 cited chunk에 직접 포함되는지 계산한다.

`faithfulness = supported cited claims / all answer claims`

Faithfulness는 correctness와 다르다. 별도 `reference_fact_recall`은 versioned dataset의 required term coverage를
correctness diagnostic으로 측정한다.

Evaluation 방법은 다음 중 하나가 될 수 있다.

- Deterministic Evaluation
- LLM-based Evaluation
- Hybrid Evaluation

사용한 평가 방식과 Evaluator Model을 반드시 기록한다.

Unanswerable case는 insufficient context, generator 미호출과 empty citation을 모두 만족할 때
`unanswerable_abstention_accuracy`의 correct case로 계산한다.

---

# Benchmark Reproducibility

## 16. 필수 Benchmark Metadata

최종 Benchmark 결과에는 최소한 다음 정보를 함께 기록한다.

```text
Timestamp
Git Commit
Model Version
Dataset Version
Hardware
CPU
GPU
RAM
OS
Python Version
PyTorch Version
CUDA Version
Input Resolution
Batch Size
Concurrency
```

환경에 존재하지 않는 항목은 N/A로 기록할 수 있다.

---

## 17. Benchmark 원칙

실행환경 정보가 없는 성능 수치는
완전한 Benchmark로 간주하지 않는다.

예를 들어 다음 정보만으로는 충분하지 않다.

```text
p95 latency = 42 ms
```

대신 다음과 같은 Context가 함께 있어야 한다.

```text
Model: PatchCore-v3
Hardware: NVIDIA L4
Input Resolution: 256x256
Batch Size: 1
Concurrency: 4

p50: ...
p95: ...
p99: ...
Throughput: ...
GPU Memory: ...
```

이 원칙은 프로젝트 전체 Benchmark에 동일하게 적용한다.
