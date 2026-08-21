# RAG Evaluation

## 1. 목적과 경계

STEP 14는 public demo SOP에 대해 retrieval Recall@K, citation precision/recall, deterministic faithfulness와
abstention을 offline pipeline으로 평가한다. Online RAG API에는 evaluation logic을 추가하지 않으며 evaluation 실행
중 index를 재생성하지 않는다.

```text
Versioned QA dataset + existing immutable RAG index
                  ↓
Raw exact-cosine ranking + operational threshold
                  ↓
Deterministic extractive answer/citations
                  ↓
Per-case evidence + aggregate metrics
                  ↓
Immutable evaluation.json + cases.jsonl
```

이 결과는 작은 fictional demo corpus와 evaluation-only provider의 재현성 검증이다. 실제 factory/private SOP,
production embedding/LLM 품질 또는 production latency를 나타내지 않는다.

## 2. Evaluation dataset

Dataset은 `configs/evaluation/rag_demo.jsonl`이며 case ID 순으로 정렬된 9개 record다.

- answerable 8, unanswerable 1
- direct retrieval
- paraphrase
- multi-chunk evidence
- multi-document evidence
- Korean question / English manual cross-language
- manual에 없는 conveyor motor torque 질문

각 record는 `case_id`, question, expected document/chunk IDs, reference facts/required terms, answerable과 language를
가진다. Expected chunk ID는 chunking 변경에 민감하므로 evaluation artifact의 index ID, metadata SHA, corpus SHA,
chunking config와 함께 해석해야 한다. Question/case ID/reference answer는 retrieval 또는 generation code에 전달되지
않으며 evaluator만 ground truth로 읽는다.

Dataset SHA-256:

```text
b2c7d988c1ca39ea4d5d20bb418050c3ed38fc63f3e1dc61914e2821e545b6c6
```

## 3. Retrieval metric contract

Document Recall@K는 한 answerable case의 unique expected document 중 threshold 적용 후 top-K 결과에 포함된 비율이다.
Chunk Recall@K도 동일하게 exact expected chunk ID로 계산한다. Multi-evidence case는 일부만 찾으면 fractional recall을
받는다. 최종 값은 8개 answerable case의 case-level macro average다. Unanswerable case는 Recall denominator에서
제외한다.

Mean Reciprocal Rank는 각 answerable case에서 첫 expected chunk의 raw ranking reciprocal rank를 macro average한다.
Per-case artifact에는 raw top score, expected chunk rank, threshold 적용 후 top-K rank/score가 남는다.

Minimum retrieval score 0.2는 evaluation configuration이며 자동 tuning하지 않는다. Artifact는 answerable/unanswerable
raw top-score 분포를 recommendation evidence로만 보존한다.

## 4. Citation metric contract

STEP 13이 marker allow-list와 structural validity를 이미 강제하므로 STEP 14는 source correctness를 측정한다.

- `citation_precision`: unique cited chunk 중 case의 exact expected chunk인 비율
- `citation_recall`: case의 expected chunk 중 실제 citation으로 사용된 비율

두 값은 answerable case별로 계산한 뒤 macro average한다. Marker가 존재한다는 사실과 해당 source가 expected
evidence라는 사실을 구분한다. 중요한 claim의 marker 누락은 deterministic faithfulness에서 unsupported claim으로
처리한다.

## 5. Faithfulness와 correctness

Faithfulness는 “answer claim이 cited retrieved context에서 지원되는가”이며 정답 여부와 다르다. Default baseline은
retrieved chunk마다 한 source sentence를 그대로 추출하고 controlled citation을 붙인다. Evaluator는 answer의 각
non-empty line을 claim 하나로 보고 marker가 연결한 chunk에 normalized claim text가 직접 포함되는지 검사한다.

`faithfulness = supported cited claims / all answer claims`

이는 deterministic lexical support baseline이며 paraphrase entailment나 복잡한 의미 모순을 판정하지 않는다.
`reference_fact_recall`은 required term coverage를 계산하는 별도 correctness diagnostic이다. 잘못 검색된 source를
정확히 복사한 답변은 faithful할 수 있지만 correct하지 않을 수 있다.

External LLM-as-a-judge는 구현하거나 실행하지 않았다. 향후 도입 시 deterministic core를 대체하지 않고 judge
provider/model/version/prompt를 artifact에 추가해야 한다.

## 6. Abstention

Unanswerable case는 다음을 모두 만족할 때 correct다.

- threshold 이상 retrieved evidence 없음
- `insufficient_context`
- generator 미호출
- citation 없음

`unanswerable_abstention_accuracy`는 unanswerable case macro accuracy다. `answerability_accuracy`는 answerable case의
answered 상태와 unanswerable case의 올바른 abstention을 함께 측정한다.

## 7. Index와 artifact lineage

Actual demo run:

| Field | Value |
|---|---|
| Index ID | `step14-demo-eval-v1` |
| Index metadata SHA | `eb31849bd379797689c87c83ff3a8b5be3f6554d9f7e207023adc2e1ee80fa99` |
| Documents / chunks | 3 / 8 |
| Chunking | max 1200 characters, paragraph overlap 1 |
| Embedding | `evaluation-demo-semantic/evaluation-demo-semantic-v1` |
| Dimension / normalization | 12 / L2 |
| Dataset SHA | `b2c7d988c1ca39ea4d5d20bb418050c3ed38fc63f3e1dc61914e2821e545b6c6` |

`evaluation.json`은 schema/evaluation ID/time, dataset SHA/count, complete index lineage, top-K/threshold,
generator/evaluator identity, aggregate metrics, score analysis와 `cases.jsonl` SHA를 가진다. `cases.jsonl`은 question,
expected evidence, raw score/rank, retrieved evidence, answer/status, citations, generator-called flag와 per-case metric을
보존한다. NaN/Inf, count/hash mismatch, corrupt record와 overwrite를 거부한다.

Output은 `outputs/evaluation/rag/<evaluation-id>/`에 생성되며 Git에 추가하지 않는다.

## 8. 실행

Evaluation용 index build는 evaluation 실행과 분리한다.

```bash
uv run python -m pipelines.build_demo_rag_evaluation_index \
  --index-id step14-demo-eval-v1
```

Existing index만 명시해 평가한다.

```bash
uv run python -m pipelines.evaluate_rag \
  --index-dir artifacts/rag/manuals/step14-demo-eval-v1 \
  --evaluation-dataset configs/evaluation/rag_demo.jsonl \
  --evaluation-id step14-demo-eval-v2 \
  --top-k 1 3 5 \
  --min-score 0.2
```

## 9. Actual demo result

| Metric | Result |
|---|---:|
| Document Recall@1 | 0.562500 |
| Document Recall@3 | 0.875000 |
| Document Recall@5 | 1.000000 |
| Chunk Recall@1 | 0.437500 |
| Chunk Recall@3 | 0.750000 |
| Chunk Recall@5 | 1.000000 |
| Mean Reciprocal Rank | 0.687500 |
| Citation Precision | 0.256250 |
| Citation Recall | 1.000000 |
| Faithfulness | 1.000000 |
| Reference Fact Recall | 0.250000 |
| Unanswerable Abstention Accuracy | 1.000000 |
| Answerability Accuracy | 1.000000 |

Answerable raw top score는 min 0.5, mean 0.674855, max 0.866025였고 unanswerable case는 0.0이었다. 이 작은
dataset만 보고 production threshold를 변경하지 않았으며 0.2를 유지했다.

Citation Recall 1.0과 낮은 Precision 0.25625는 baseline generator가 threshold를 통과한 top-5 context를 모두
인용하기 때문이다. Faithfulness 1.0도 extractive behavior의 결과이며 높은 correctness를 의미하지 않는다.

## 10. One-time correction과 limitations

최초 run의 faithfulness는 0.891667이었다. TXT source 내부 줄바꿈이 claim separator로 해석되어 citation 없는 line이
생기는 deterministic generator serialization bug를 발견했다. Source sentence의 whitespace만 정규화한 뒤 동일
index/dataset/config로 한 번 재실행해 faithfulness가 1.0이 됐다. Retrieval, threshold, dataset과 expected evidence는
수정하지 않았고 Recall/citation 결과도 변하지 않았다.

남은 한계:

- 9-case public demo set이라 통계적 일반화가 불가능하다.
- Evaluation-only concept embedding은 production semantic model이 아니다.
- Extractive generator는 간결성이나 answer synthesis를 평가하지 않는다.
- Lexical faithfulness는 paraphrase entailment, contradiction과 claim importance를 판정하지 않는다.
- External production embedding/generation과 LLM judge는 미검증이다.
- STEP 15에서 provider별 품질/latency/cost와 더 큰 held-out corpus 평가가 필요하다.
