# Final Benchmark and Performance Evidence

## 1. 목적과 범위

STEP 15는 새로운 실험이나 성능 최적화가 아니라 기존의 승인된 Vision/API benchmark와 실제 STEP 14 RAG
evaluation을 하나의 재현 가능한 evidence contract로 집계한다. Final artifact는 다음 section을 독립적으로 보존한다.

- Vision image-level quality
- Pixel localization quality
- T4 model runtime performance
- FastAPI application performance
- Demo/deterministic RAG quality
- Platform engineering verification matrix

서로 다른 환경, 시점과 measurement boundary의 수치를 합산하지 않으며 overall score를 만들지 않는다. 모델 재학습,
threshold 재보정, retrieval tuning, DB query 변경, container 최적화와 비용 추정은 수행하지 않았다.

## 2. Source와 집계 계약

공식 STEP 3/4 raw output은 `outputs/` 정책에 따라 Git에 보존되지 않았다. 따라서 당시 승인된 실제 결과 문서를
source로 하는 작은 versioned historical approved evidence snapshot 세 개를 `configs/benchmarks/official/`에 둔다.
이 snapshot은 당시 raw benchmark artifact 자체가 아니며 새 측정값도 아니다. 원문 값, 환경, boundary, lineage와
알려진 한계를 그대로 보존하며 Builder source code에는 metric 숫자를 넣지 않는다.

| Input | 역할 | Source policy |
|---|---|---|
| `vision_quality_step3.json` | Official image/pixel quality | STEP 3 승인 결과 compact snapshot |
| `model_runtime_step3_t4.json` | Official T4 model runtime | STEP 3 승인 결과 compact snapshot |
| `api_http_step4_v1_t4.json` | Official FastAPI schema v1 | STEP 4 승인 결과 compact snapshot |
| STEP 14 `evaluation.json`, `cases.jsonl` | Actual RAG quality/evidence | Existing immutable output artifact |
| `platform_verification.json` | Engineering verification state | Versioned factual matrix |
| API schema v2 artifact | Persistence-inclusive latency | Optional; 이번 집계에는 없음 |

필수 source 부재, NaN/Inf, schema/label 오류는 fail-fast한다. Vision source의 `category`, manifest SHA, model SHA와
적용 가능한 threshold artifact SHA가 다르면 집계를 거부한다. Optional API v2는 현재 API benchmark schema v2,
실제 inspection persistence 포함 표시와 네 provenance SHA를 모두 만족해야 한다. Final output은 기존 directory를
overwrite하지 않고 모든 source file SHA-256을 기록한다.

## 3. 한눈에 보는 결과

| Area | Metric | Result | Environment / boundary |
|---|---|---:|---|
| Vision image | AUROC / F1 | 0.997556 / 0.994595 | Kaggle T4, fixed validation threshold |
| Vision image | FP / FN | 0 / 1 | Official MVTec `metal_nut` test |
| Pixel localization | AUROC / F1 | 0.982486 / 0.834279 | Kaggle T4, pixel threshold 적용 |
| Model runtime | p50 / p95 / p99 | 21.634 / 25.775 / 27.113 ms | T4 inference timing boundary |
| Model runtime | Throughput | 45.114 images/s | Batch 1, disk I/O 제외 |
| FastAPI schema v1 | p50 / p95 / p99 | 44.902 / 48.703 / 53.746 ms | T4, in-process ASGI, persistence 제외 |
| FastAPI schema v1 | Throughput / error | 22.030 req/s / 0.0 | 115 measured requests |
| RAG retrieval | Document Recall@1/@3/@5 | 0.5625 / 0.8750 / 1.0000 | 9-case public demo |
| RAG retrieval | Chunk Recall@1/@3/@5 | 0.4375 / 0.7500 / 1.0000 | Deterministic exact retrieval |
| RAG evidence | Citation precision / fact recall | 0.25625 / 0.25000 | Extractive evaluation baseline |

위 표의 latency끼리 차감하거나 품질 metric과 합산해서는 안 된다.

## 4. Environment matrix

| Section | Environment | Device | Measurement boundary |
|---|---|---|---|
| Vision image/pixel quality | Kaggle Linux, Python 3.12.13, torch 2.13.0+cu130 | Tesla T4 / CUDA 13.0 | 저장된 official test prediction을 normal-only validation threshold로 평가 |
| Model runtime | Kaggle Linux x86_64, 동일 PyTorch runtime | Tesla T4 / CUDA 13.0 | Disk load 이후 preprocessing부터 synchronized prediction materialization까지 |
| API schema v1 | Kaggle Linux, in-process ASGI TestClient | Tesla T4 / CUDA 13.0 | Multipart request부터 completed ASGI response까지, persistence 제외 |
| RAG | Local deterministic public demo evaluation | CPU/NumPy exact retrieval | Existing immutable index의 9개 case quality evaluation |
| Platform verification | CI, local macOS/Docker/PostgreSQL/SQLite, Kaggle 등 | 항목별 상이 | 기능/구성 contract 검증이며 latency benchmark가 아님 |
| API schema v2 | `not_available` | 측정하지 않음 | Real PostgreSQL + production PatchCore GPU evidence가 필요 |

모든 결과가 하나의 production environment에서 측정된 것은 아니다.

## 5. Vision quality

MVTec AD `metal_nut`의 official test 115장에 normal-only validation에서 사전 고정한 threshold를 적용했다.

| Image metric | Result |
|---|---:|
| AUROC | 0.9975562072336266 |
| Precision | 1.0 |
| Recall | 0.989247311827957 |
| F1 | 0.9945945945945946 |
| TP / TN / FP / FN | 92 / 22 / 0 / 1 |

Per-defect detection은 bent 25/25, color 21/22, flip 23/23, scratch 23/23이고 good false positive는
0/22다. Image threshold는 `41.19657897949219`, comparison은 strict `score > threshold`다.

## 6. Pixel localization quality

| Pixel metric | Result |
|---|---:|
| AUROC | 0.9824857431023005 |
| Precision | 0.8339393856698755 |
| Recall | 0.8346198271398415 |
| F1 | 0.8342794676622264 |

Pixel threshold는 `40.1362419128418`이다. 이 결과는 image-level anomaly 판정과 별도 section이다.

## 7. Official T4 model runtime

조건은 batch size 1, warmup 10 batch, measured image/batch 각각 115, `num_workers=0`이다. Timing은 image
batch disk loading 후 `preprocessing → device transfer → PatchCore inference → prediction materialization →
accelerator synchronization`을 포함한다. Disk image read, artifact restore, warmup, threshold와 output persistence는
제외한다.

| Metric | Result |
|---|---:|
| p50 | 21.6343939998751 ms |
| p95 | 25.774736599942116 ms |
| p99 | 27.113390779900328 ms |
| Mean | 22.16589099130824 ms |
| Total timed | 2.5490774640004474 s |
| Throughput | 45.11436063599353 images/s |
| CUDA peak allocated / reserved | 294.603515625 / 382.0 MiB |
| Model size | 195,058,659 bytes / 186.02243328094482 MiB |

56초 training wall time과 benchmark CLI 전체 runtime은 이 inference latency가 아니다.

## 8. API application performance

### API Benchmark — schema v1 / pre-persistence

실제 STEP 4 결과는 FastAPI **in-process application-level HTTP E2E benchmark**다. `multipart request →
routing → upload read → image decode → tensor conversion → preprocessing → device transfer → PatchCore inference →
strict threshold → response validation/serialization → completed ASGI response`를 포함한다.

| Metric | Result |
|---|---:|
| p50 / p95 / p99 | 44.90185999998175 / 48.70313200001419 / 53.7457109400475 ms |
| Mean / total | 45.39321613043957 ms / 5.2202198550005505 s |
| Throughput | 22.029723497151036 requests/s |
| Successful / failed / error rate | 115 / 0 / 0.0 |

Disk image loading, artifact restore, warmup, inspection persistence, external network RTT와
uvicorn/socket/TLS/proxy는 제외한다. 따라서 production network latency 또는 현재 persistence-inclusive API latency가
아니다.

### API Benchmark — schema v2 / persistence-inclusive

이번 STEP에서는 `not_available`이다. 현재 tooling은 inference 뒤 PostgreSQL `INSERT/COMMIT`을 포함하는 schema v2를
지원하지만, Tesla T4 real model과 실제 PostgreSQL을 함께 사용한 comparable artifact가 없다. Mac CPU, SQLite 또는
fake repository 결과로 대체하지 않았다. PostgreSQL migration/CRUD integration은 별도의 기능 검증으로는 완료됐다.

## 9. Demo / deterministic RAG benchmark

Actual STEP 14 artifact의 dataset SHA는
`b2c7d988c1ca39ea4d5d20bb418050c3ed38fc63f3e1dc61914e2821e545b6c6`, index ID는
`step14-demo-eval-v1`, index metadata SHA는
`eb31849bd379797689c87c83ff3a8b5be3f6554d9f7e207023adc2e1ee80fa99`다. Evaluator는
`deterministic-extractive-support-v1`이며 external judge를 사용하지 않았다.

| Metric | Result |
|---|---:|
| Document Recall@1 / @3 / @5 | 0.5625 / 0.875 / 1.0 |
| Chunk Recall@1 / @3 / @5 | 0.4375 / 0.75 / 1.0 |
| MRR | 0.6875 |
| Citation Precision / Recall | 0.25625000000000003 / 1.0 |
| Faithfulness | 1.0 |
| Reference Fact Recall | 0.25 |
| Unanswerable Abstention Accuracy | 1.0 |
| Answerability Accuracy | 1.0 |

Top-5 retrieval coverage는 좋지만 extractive baseline이 통과한 context를 과도하게 인용해 citation precision이 낮다.
Reference fact recall도 0.25로 answer completeness가 제한적이다. Faithfulness 1.0은 추출형 문장이 citation source에
lexically 포함된다는 의미이지 production answer correctness를 뜻하지 않는다. 향후 private held-out corpus와 실제
production embedding/generation provider에서 개선·재평가해야 한다.

## 10. Platform verification matrix

| Area | Status | Evidence environment |
|---|---|---|
| Data split leakage check | verified | Deterministic CI/unit tests |
| Model artifact integrity | verified | Local/Kaggle artifact loaders |
| FastAPI real-model smoke | verified | Kaggle Tesla T4 |
| PostgreSQL real integration | verified | Local Docker PostgreSQL |
| MLflow round-trip | verified | Local SQLite tracking backend |
| Docker build | verified | macOS arm64 Docker |
| GitHub Actions | verified | GitHub-hosted CI |
| Prometheus | verified | Local application/Compose |
| Grafana | verified | Local Compose provisioning |
| Drift detection | verified | Deterministic offline pipeline |
| Kubernetes render | verified | Local `kubectl kustomize` |
| Dashboard | partially_verified | Local Streamlit/Docker; production IAP pending |
| RAG API | partially_verified | Local deterministic Docker; production provider pending |
| RAG evaluation | verified | Deterministic public demo |
| Actual GKE GPU deployment | pending | GCP |
| Production Cloud SQL | pending | GCP |
| Production LLM provider | pending | External provider |
| Private SOP benchmark | pending | Controlled production corpus |
| Persistence-inclusive T4 API benchmark | pending | GPU + real PostgreSQL |
| Real-model API and Prometheus production scrape | pending | Production Kubernetes |
| Dashboard production authentication | pending | GKE/IAP |

상태는 factual verification일 뿐 subjective maturity score가 아니다. 근거 문구의 machine-readable source는
`configs/benchmarks/platform_verification.json`이다.

## 11. Final artifact와 실행

실제 생성 경로:

```text
outputs/benchmarks/final/step15-final-authoritative-v1/benchmark.json
```

실행 명령:

```bash
uv run python -m pipelines.build_final_benchmark \
  --rag-evaluation-dir outputs/evaluation/rag/step14-demo-eval-v2 \
  --benchmark-id step15-final-authoritative-v1
```

Artifact schema version은 2다. `benchmark_id`, timezone-aware `created_at`, repository provenance, source
filename/SHA, 독립 section, environment matrix와 limitations를 저장한다. `outputs/`는 Git ignore 대상이므로 실제
artifact를 Commit하지 않는다.

Repository provenance는 CLI 입력으로 SHA를 받지 않고 build 시작 시 Git에서 자동 조회한다.

```json
{
  "repository": {
    "git_commit": "5c59e9ebb992231b39d4f7c0e20879d97a4d89dd",
    "working_tree_dirty": false
  }
}
```

`git_commit`은 build 시 checkout된 HEAD다. `working_tree_dirty=true`이면 이 SHA는 working tree의 기준 commit일
뿐이며 builder/evidence/docs/test 변경 전체를 재현하는 commit이 아니다. 현재
`step15-final-authoritative-v1`은 STEP 15 commit 직후 clean working tree에서 생성되어 `working_tree_dirty=false`를
기록한 authoritative STEP 15 artifact다. 이 artifact는 STEP 16 문서 변경 전 STEP 15 repository state를 가리킨다.

### Clean committed authoritative artifact 생성

1. STEP 15 builder, evidence, tests와 문서를 하나의 commit으로 확정한다.
2. `git status --porcelain --untracked-files=normal` 출력이 비어 있는지 확인한다.
3. 위 CLI를 새 immutable benchmark ID로 다시 실행한다. SHA override는 사용하지 않는다.
4. 생성된 `repository.git_commit`이 STEP 15 commit과 같고 `working_tree_dirty=false`인지 loader로 검증한다.

이 절차로 만든 artifact만 STEP 15 코드와 versioned evidence를 clean committed state에서 재현하는 authoritative
artifact다. Artifact 생성 후 생기는 ignored `outputs/` 파일은 Git dirty 판정에 포함되지 않는다.

실제 source SHA-256:

| Source | SHA-256 |
|---|---|
| Vision quality snapshot | `2901cc8b0d8dba7d7f1f40cd4574af140cce864d8e2bc9fd134f0c5d0d253915` |
| Model runtime snapshot | `ddeaabbd9cae7c64442e00d94e5a15d2c858dca5518d8e03dad395a88329127c` |
| API schema v1 snapshot | `52cb1de649bf5229f8f8a283479ae963bd517d60f9089456076339b7f76b2cd6` |
| RAG evaluation | `9271e458d2f12c302820fcdacf567a84240eb9cfb33061451600ab2c86d82af1` |
| RAG cases | `275d8414c5b8f2979e3a0707e40014b31e4f0c5e7feef75adc90dbcd726ed804` |
| Platform verification | `cbeae49a00cfcea73429d6d06178d63c76ab98b8213d8308ff52df1bb67eb84f` |

Vision lineage는 category `metal_nut`, manifest
`da81db68eadd22421ba2b284ffee85f49d41fcec47d6aadfa6bdb2cae14f285b`, model
`1a2016a6b75377cc5e6bbeee33b3ed2f3a3b4d1cedb2e80236dbcd1da8c28ca9`, threshold artifact
`9e885f2a3b0de29eeb3e04304d5dc9051fb1a9c6831bf820b885760ccd12fe89`다. 당시 문서에 artifact
metadata SHA가 보존되지 않아 snapshot에는 `null`로 남기며 추정하지 않는다.

## 12. Limitations와 다음 검증

- Historical approved STEP 3/4 snapshot은 승인 문서 기반 compact evidence이며 당시 raw artifact 자체가 아니다.
- Authoritative artifact는 STEP 15 clean commit을 재현하며 이후 STEP 16 문서 변경은 포함하지 않는다.
- API schema v1은 persistence와 external network를 포함하지 않는다.
- 실제 GKE GPU, Cloud SQL, production auth와 production Prometheus scrape는 실행하지 않았다.
- RAG는 fictional public demo와 deterministic evaluation provider 결과다.
- Production embedding/generation/judge, private SOP, latency와 cost benchmark는 아직 없다.
- 비용 수치는 실제 billable provider/deployment evidence가 없어 산출하거나 추정하지 않았다.

후속 단계는 새로운 evidence를 같은 이름으로 덮어쓰지 말고 새 benchmark ID/schema로 생성해야 한다. 특히 API v2는
real PostgreSQL과 production-class GPU를 함께 사용하고, RAG provider benchmark는 provider/model/version, corpus
lineage, latency boundary와 실제 비용 source를 명시해야 한다.
