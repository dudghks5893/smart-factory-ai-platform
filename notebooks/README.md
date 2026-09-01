# Experiment Notebooks

이 디렉터리의 notebook은 repository-owned pipeline을 호출하는 연구·시각화용 interface다.
Dataset parsing, augmentation, training, evaluation, matching, telemetry, artifact serialization의
source of truth는 `ml/`과 `pipelines/`에 유지하며 notebook cell에 동일 로직을 복제하지 않는다.

Tracked notebook은 output을 비운 상태로 유지한다. 생성된 PNG, JSON/JSONL, model, dataset cache,
package는 `.gitignore`가 적용되는 `outputs/`, `artifacts/`, `data/` 아래에 저장하며 notebook에
base64 형태로 embed하지 않는다.

## YOLO Segmentation Experiment Workbench

`vision/yolo_segmentation_experiment_workbench.ipynb`는 두 가지 mode를 제공한다.

- `research`
  - 임시 `imgsz`, `batch`, `epochs`, `patience` 값을 별도 research namespace에서 탐색한다.
  - 이 결과는 official experiment나 Experiment Log를 갱신하지 않는다.

- `official`
  - committed experiment config만 사용하며 notebook override를 허용하지 않는다.
  - Manifest, Baseline artifact, Git provenance, validation-only policy를 preflight에서 검증한다.
  - Notebook kernel은 display/controller 역할만 담당한다.
  - 실제 augmentation/representation preview와 training은 `uv run --locked python` subprocess에서 실행해
    lock refresh와 repository `.venv` 외부의 system package 개입을 방지한다.
  - `RUN_OFFICIAL_TRAINING=True`를 명시하기 전에는 학습을 시작하지 않는다.

C4-2A/C4-2B candidate 평가 당시에는 train/validation만 EDA·preview·selection에 사용했고,
derived test는 candidate selection에서 제외했다.

C4-3에서는 validation-only evidence를 기준으로 최종 candidate와 artifact v2를 freeze했다.
이후 C4-4의 별도 execution surface에서 derived test를 report-only final evaluation으로 수행했다.

## C4-4 Final Test Notebook

`vision/yolo_final_test_evaluation.ipynb`는 C4-4 전용 one-time final-test execution surface다.

Training Workbench와 분리해 다음 항목을 preflight에서 먼저 검증한다.

- clean Git state
- frozen manifest
- Official package
- committed config
- Dataset Manifest bytes
- dedicated output namespace

기본값은 `RUN_FINAL_TEST=False`이며,
CLI `--confirm-final-test`가 명시된 operator cell에서만 test resolver를 연다.

Actual final test는 reviewed/pushed commit
`e15fd92776a3981a1b5927ad567802d0d0a3bb54`에서 완료됐으며 최종 상태는
`FINAL_TEST_COMPLETED`다.

Candidate selection과 protocol tuning은 이 notebook의 범위가 아니었다.
Final-test 결과를 근거로 threshold 변경, checkpoint 재선택, augmentation 변경,
hyperparameter tuning, candidate reselection도 수행하지 않았다.

Final-test output, model, dataset, result JSON, evidence ZIP은 Git에 추가하지 않는다.

Notebook의 fail-closed 기본값과 reviewed SHA exact-match contract는
historical reproduction safety를 위해 그대로 유지한다.
