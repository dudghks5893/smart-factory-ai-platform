# ADR-0003: 초기 Vision AI Baseline으로 PatchCore 계열 사용

- 상태: 승인(Approved)
- 결정일: 2026-08-18

## 1. Context

프로젝트의 핵심 Vision AI 문제는 제조 이미지에서 정상과 이상을 구분하고
가능한 경우 이상 위치까지 시각화하는 Industrial Anomaly Detection이다.

MVTec AD를 기반으로 초기 Baseline을 구축해야 하며,
모델 자체 연구보다 Production AI System 전체를 단계적으로 구현하는 것이 우선이다.

따라서 초기부터 복잡한 Custom Model을 개발하기보다
검증된 Anomaly Detection Baseline을 먼저 확보할 필요가 있다.

## 2. Decision

초기 Vision AI Baseline으로 PatchCore 계열을 우선 사용한다.

초기 구현에서는 검증된 Library Implementation을 활용할 수 있으며,
Serving Layer가 특정 구현체에 직접 종속되지 않도록 Adapter/Interface 경계를 둔다.

개념적인 구조는 다음과 같다.

```text
API
  ↓
Inference Interface
  ↓
PatchCore Adapter
  ↓
PatchCore Implementation
```

## 3. Reason

- Industrial Anomaly Detection 문제와 직접적으로 맞닿아 있다.
- MVTec AD 기반 Benchmark와 비교하기 용이하다.
- Pretrained Feature를 활용해 초기 Baseline을 빠르게 확보할 수 있다.
- Full Supervised Defect Dataset이 없는 제조 환경을 다루기에 적합하다.
- 모델 개발에만 프로젝트 전체 시간이 소모되는 것을 방지할 수 있다.
- Memory Bank, Latency, Artifact Size 등 Production 관점의 평가 포인트가 존재한다.

## 4. Alternatives

향후 비교 대상으로 다음 모델을 검토할 수 있다.

- PaDiM
- EfficientAD
- FastFlow
- Custom PyTorch Model

단, 기술 스택을 늘리기 위한 목적으로 추가하지 않는다.

다음과 같은 명확한 목적이 있을 때만 추가한다.

- 정확도 개선
- Inference Latency 개선
- Memory 사용량 감소
- Artifact Size 감소
- 운영 복잡도 감소

## 5. Consequences

### 장점

- 초기 End-to-End Pipeline 구축 속도가 빨라진다.
- 산업용 Anomaly Detection 요구사항과 잘 맞는다.
- 모델 교체 전후를 수치로 비교할 수 있다.

### 단점

- PatchCore Memory Bank가 Serving Resource에 영향을 줄 수 있다.
- 특정 Dataset에서는 다른 모델이 더 적합할 수 있다.
- Library 내부 구현을 그대로 사용할 경우 내부 동작 이해가 부족해질 수 있다.

따라서 구현 과정에서 PatchCore의 동작 원리와 Serving 특성을 별도로 검증하고 기록한다.
