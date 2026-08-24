# ADR-0007: Provider-Agnostic RAG Architecture 사용

- 상태: 승인(Approved)
- 결정일: 2026-08-18

## 1. Context

프로젝트 후반에는 제조 매뉴얼/SOP 기반 RAG Quality Assistant를 구현한다.

초기 구현에서는 OpenAI API를 사용할 가능성이 높지만,
향후 다음과 같은 대안을 실험할 수 있다.

- Local LLM
- Local Embedding Model
- 다른 외부 LLM API
- Reranker Model

RAG Business Logic이 특정 Provider SDK에 직접 결합되면
Model 교체와 비교 실험이 어려워진다.

## 2. Decision

RAG Architecture는 특정 Provider에 종속되지 않도록 설계한다.

핵심 Application Service는 다음 Interface에 의존한다.

```text
QualityRAGService
        │
        ├── EmbeddingProvider
        ├── Retriever
        └── AnswerGenerator
```

Provider-specific SDK 코드는 Adapter 내부에 격리한다.

예시는 다음과 같다.

```text
AnswerGenerator
├── OpenAICompatibleAnswerGenerator
└── DeterministicEvaluationGenerator

EmbeddingProvider
├── OpenAICompatibleEmbeddingProvider
└── DeterministicEvaluationEmbeddingProvider

Retriever
└── ExactCosineRetriever (current immutable NumPy index)
```

현재 corpus에는 Vector DB나 reranker를 도입하지 않는다. `pgvector`, ANN backend와 reranker는 measured corpus
scale 또는 품질 요구가 생길 때 existing retrieval boundary 뒤에서 검토한다.

## 3. Reason

- OpenAI와 Local Model을 동일한 Evaluation Pipeline에서 비교할 수 있다.
- 특정 Vendor 종속성을 줄일 수 있다.
- RAG Business Logic과 Model API Integration을 분리할 수 있다.
- 비용, Latency, 품질 비교 실험이 쉬워진다.
- 향후 Model 교체 시 수정 범위를 줄일 수 있다.

## 4. Alternatives

### OpenAI SDK 직접 사용

초기 구현은 빠르지만 Application Logic 전체가 특정 Provider에 결합될 수 있다.

### RAG Framework에 전체 구조 의존

LangChain, LlamaIndex 등의 Framework를 사용할 수 있으나
핵심 Domain Logic까지 Framework에 강하게 종속시키지 않는다.

필요한 기능이 있을 경우 Adapter 또는 제한된 영역에서 사용할 수 있다.

## 5. Evaluation 원칙

Provider가 변경되어도 동일한 RAG Evaluation 기준을 적용할 수 있어야 한다.

주요 지표는 다음과 같다.

- Retrieval Recall@K
- Citation Accuracy
- Faithfulness
- Latency
- 필요 시 Cost

Provider 비교 시 Prompt, Evaluation Dataset, Retrieval 조건 등
비교 조건을 가능한 한 통제한다.

## 6. Consequences

### 장점

- Model 및 Provider 교체가 쉬워진다.
- OpenAI와 Local Model 비교가 가능하다.
- Architecture 설명력이 높아진다.

### 단점

- 초기 구현 코드가 직접 SDK를 사용하는 방식보다 약간 증가한다.
- Provider마다 지원 기능이 다르기 때문에 공통 Interface 설계가 필요하다.

Interface는 실제 구현 요구사항이 확인된 이후 최소 기능부터 정의한다.
처음부터 과도하게 추상화하지 않는다.
