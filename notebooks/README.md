# Experiment Notebooks

이 디렉터리의 notebook은 repository-owned pipeline을 호출하는 연구·시각화 interface다. Dataset parsing,
augmentation, training, evaluation, matching, telemetry와 artifact serialization의 source of truth는 `ml/`과
`pipelines/`에 유지한다. Notebook cell에 이를 복제하지 않는다.

Tracked notebook은 output을 비운 상태로 유지한다. 생성된 PNG, JSON/JSONL, model, dataset cache와 package는
`.gitignore`가 적용되는 `outputs/`, `artifacts/`, `data/` 아래에 저장하며 notebook에 base64로 embed하지 않는다.

`vision/yolo_segmentation_experiment_workbench.ipynb`는 두 mode를 제공한다.

- `research`: 임시 `imgsz`, `batch`, `epochs`, `patience`를 별도 research namespace에서 탐색한다. 이 결과는
  official experiment나 Experiment Log를 갱신하지 않는다.
- `official`: committed experiment config만 사용하며 notebook override를 거부한다. Manifest, Baseline artifact,
  Git provenance와 validation-only policy를 preflight에서 확인한다. `RUN_OFFICIAL_TRAINING=True`를 명시하기
  전에는 학습을 시작하지 않는다.

현재 C4-2A에서는 train/validation만 EDA·preview·selection에 사용한다. Derived test는 C4-3에서 validation으로
최종 후보를 선택하고 artifact v2를 freeze한 뒤 한 번 평가하기 전까지 `SEALED / NOT USED`다.
