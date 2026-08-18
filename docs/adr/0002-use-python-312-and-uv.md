# ADR-0002: Python 3.12 및 uv 기반 개발환경 사용

- 상태: 승인(Approved)
- 결정일: 2026-08-18

## 1. Context

프로젝트는 PyTorch, OpenCV, Anomaly Detection Library, FastAPI, MLflow,
PostgreSQL Client 등 다양한 Python 생태계 라이브러리를 함께 사용하게 된다.

최신 Python Version만을 우선하면 일부 ML/CV Library와의 호환성 문제가 발생할 수 있으며,
Local macOS, Kaggle, GCP 환경에서 동일한 개발환경을 재현해야 한다.

Dependency와 Virtual Environment를 일관된 방식으로 관리할 도구도 필요하다.

## 2. Decision

프로젝트의 기본 Python Version을 `Python 3.12.x`로 사용한다.

Python Version, Virtual Environment, Dependency 및 Lock File 관리는 `uv`를 사용한다.

주요 파일은 다음과 같다.

```text
.python-version
pyproject.toml
uv.lock
.venv/
```

`.venv/`는 Local Project 전용 Virtual Environment이며 Git에는 포함하지 않는다.

## 3. Reason

- PyTorch 및 주요 ML/CV 생태계와의 호환성을 우선할 수 있다.
- Local macOS와 Cloud 환경 사이의 재현성을 높일 수 있다.
- `uv` 하나로 Python Project Dependency와 Virtual Environment를 관리할 수 있다.
- `uv.lock`을 통해 Dependency Version을 재현할 수 있다.
- 기존 `pip + requirements.txt + venv` 조합보다 프로젝트 관리 포인트를 줄일 수 있다.

## 4. Alternatives

### 최신 Python Version 사용

최신 기능을 사용할 수 있으나 ML Library 호환성이 늦게 따라올 가능성이 있다.

### pip + venv

표준적인 방식이지만 Dependency Lock과 Project Metadata를 별도로 관리해야 한다.

### Poetry

충분히 사용 가능한 대안이지만 이번 프로젝트에서는 더 단순하고 빠른 `uv`를 선택한다.

## 5. Consequences

### 장점

- 프로젝트별 독립된 Python 환경을 유지할 수 있다.
- Dependency 재현성이 높아진다.
- 설치 및 환경 동기화 명령을 단순화할 수 있다.

### 단점

- `uv`를 모르는 개발자는 별도 Tool 학습이 필요하다.
- 향후 일부 Library가 Python 3.12 지원을 종료하면 Version 변경 검토가 필요하다.

Python Version 변경이 필요할 경우 호환성 검증 후 별도 ADR로 기록한다.
