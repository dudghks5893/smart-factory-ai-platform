# Coding Conventions

## 1. 목적

이 문서는 프로젝트 Python 코드의 변경 이력, 실행 흐름, 책임 분리, 검증,
오류 처리, 테스트 및 dependency 관리 기준을 정의한다. Correctness를 가장 먼저
보호하면서 가독성, 유지보수성, 테스트 가능성, 메모리 효율성을 순서대로 고려한다.

## 2. Revision Comment

프로젝트가 직접 작성한 모든 function, method, test function 바로 위에는 생성 이력을 둔다.
Decorator가 있으면 comment를 decorator 위에 작성한다.

신규 함수:

```python
# ADD YYYY-MM-DD: 함수의 목적 또는 생성 이유를 설명한다.
def function_name() -> None: ...
```

의미 있는 변경이 있는 함수:

```python
# ADD YYYY-MM-DD: 최초 생성 목적을 유지한다.
# MODIFY YYYY-MM-DD: 직전 변경 요약 → 현재 변경 내용과 이유를 설명한다.
def function_name() -> None: ...
```

Revision comment는 최대 두 줄로 유지한다. ADD 날짜는 최초 생성 시점을 유지하고,
MODIFY는 가장 최근 변경 흐름만 기록한다. 기존 docstring은 API 계약 설명을 위해 유지한다.

## 3. 실행 흐름 Comment

Data loading, validation, preprocessing, feature extraction, artifact 접근, inference,
prediction 저장과 같은 주요 단계에는 날짜 없는 일반 comment를 사용한다.

연속된 호출이 하나의 작업 단계이면 각 줄에 반복하지 않고 block 앞에 한 번만 작성한다.
코드 자체로 명확한 대입, 길이 계산, 단순 변환에는 comment를 추가하지 않는다.

## 4. Function과 Method

- Function은 하나의 명확한 책임과 typed input/output 계약을 가진다.
- 비용이 큰 외부 접근이나 model 연산 전에 검증 가능한 입력을 먼저 확인한다.
- 중첩을 줄일 수 있으면 early return을 사용한다.
- Public API에는 parameter, return, exception을 설명하는 docstring을 유지한다.
- Runtime 설정을 함수 내부 literal로 고정하지 않고 configuration을 전달한다.

## 5. Reusable Helper

공통 helper는 실제 caller가 두 곳 이상이고 특정 domain에 종속되지 않을 때만 분리한다.
이름만 보고 역할을 알 수 있어야 하며 side effect는 없거나 명확해야 한다.

SHA-256처럼 범용적인 로직은 `shared/`에 둘 수 있다. MVTec mask 규칙이나 PatchCore
memory bank 구성처럼 domain knowledge를 포함한 로직은 해당 module에 유지한다.
거대한 `utils.py` 또는 의미 없는 helper 계층은 만들지 않는다.

## 6. Constants와 Configuration

- Dataset split name, artifact filename 같은 domain invariant는 해당 domain constants로 관리한다.
- Resize size, batch size, random seed, coreset ratio, device처럼 실행마다 변경 가능한 값은 YAML
  configuration으로 관리한다.
- Secret은 environment variable 또는 별도 secret store를 사용한다.
- 함수 내부에서 한 번만 사용되고 의미가 분명한 literal은 무조건 constant로 승격하지 않는다.

## 7. Validation과 Error Handling

Validation은 가능한 한 비용이 큰 작업보다 먼저 수행한다. Pipeline은 config, device,
input artifact, manifest schema/hash, split, category, label, output destination을 확인한 뒤
feature extraction 또는 inference를 시작한다.

Built-in exception이 의미를 정확히 전달하면 다음 형식을 유지한다.

- `ValueError`: 값 또는 domain invariant 위반
- `TypeError`: runtime type 계약 위반
- `FileNotFoundError`: 필수 file/artifact 부재
- `FileExistsError`: overwrite가 금지된 destination 존재
- `RuntimeError`: model lifecycle 또는 실행 상태 위반

여러 domain과 외부 interface가 공통 error code를 실제로 요구할 때만 custom exception과
error code를 도입한다. HTTP handler는 API 구현 단계에서 결정한다.

## 8. OOP와 Pure Function

Configuration과 lifecycle state를 공유하는 model adapter, preprocessor, artifact metadata에는
class를 사용한다. Hashing, validation, deterministic conversion처럼 state가 필요 없는 작은
연산은 pure function으로 유지한다.

Composition을 우선하며 구현 요구가 없는 inheritance hierarchy나 manager class는 만들지 않는다.

## 9. Readability와 Memory Efficiency

- 의미가 드러나는 변수명과 작은 block comment를 사용한다.
- 동일 expression과 불필요한 tensor copy를 피한다.
- Dataset은 lazy loading을 유지한다.
- Inference는 `torch.inference_mode()`를 사용한다.
- Accelerator prediction은 필요한 output만 batch 단위로 CPU에 이동한다.
- Artifact에는 Python model object가 아닌 CPU tensor state_dict와 JSON metadata만 저장한다.
- Memory 최적화를 위해 이해하기 어려운 control flow를 도입하지 않는다.

## 10. Testing

- Unit test는 validation, deterministic behavior, boundary condition과 error path를 검증한다.
- Integration test는 pipeline과 artifact round-trip처럼 component 경계의 계약을 검증한다.
- Test comment는 Arrange/Act/Assert 구분이나 특수 시나리오 이해에 도움이 될 때만 작성한다.
- Test에서 network download와 대규모 dataset 실행에 의존하지 않는다.
- 변경 후 Ruff, mypy, pytest 및 `make check`를 모두 실행한다.

## 11. Dependency 관리

- Python dependency는 `uv add`, 제거는 `uv remove`를 사용한다.
- `pyproject.toml`과 `uv.lock`을 함께 관리한다.
- 새 dependency는 기존 표준 library나 설치 dependency로 해결할 수 없는 명확한 요구가 있을 때만
  추가한다.
- Production code와 test에서 사용하는 dependency version은 lockfile로 재현한다.
