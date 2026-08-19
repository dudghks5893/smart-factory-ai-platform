# PatchCore Baseline Pipeline

## 1. 범위

STEP 2-2는 기존 MVTec manifest와 `MVTecManifestDataset`을 유지하면서
PatchCore 전처리, memory bank 생성, artifact 저장/복원, raw prediction 출력을 구현한다.

```text
Manifest train split
        ↓
MVTecManifestDataset / DataLoader
        ↓
PatchCorePreprocessor
        ↓
Anomalib PatchcoreModel
        ↓
Coreset memory bank
        ↓
Portable model artifact
```

Anomalib의 MVTec DataModule이나 threshold 최적화는 이 단계에서 사용하지 않는다.

## 2. 설정

기준 설정은 `configs/model/patchcore_baseline.yaml`에서 관리한다.

- Backbone: `wide_resnet50_2`
- Feature layers: `layer2`, `layer3`
- Coreset sampling ratio: `0.1`
- Neighbors: `9`
- Resize: `256x256`
- Center crop: `224x224`
- Image normalization: ImageNet mean/std
- Training random seed: `42`
- Device: `auto`

`auto`는 CUDA, MPS, CPU 순서로 선택한다. 명시적으로 요청한 CUDA 또는 MPS가
없으면 다른 device로 fallback하지 않고 오류를 반환한다.

## 3. 재현성

PatchCore 학습 시작 전에 Python, NumPy, PyTorch RNG를 동일 seed로 초기화한다.
CUDA 환경에서는 모든 CUDA RNG도 초기화하고 cuDNN deterministic 설정을 적용한다.

재현성 목표는 동일 실행환경, dependency lock, configuration, manifest에서 동일한
artifact construction을 반복하는 것이다. CPU, MPS, CUDA처럼 서로 다른 backend에서
bitwise identical artifact를 보장하지 않는다.

## 4. 전처리

Dataset layer는 원본 RGB/mask tensor 로딩만 담당한다. PatchCore 전처리는
`ml/training/preprocessing.py`에서 별도로 수행한다.

Image:

```text
Resize 256x256 → CenterCrop 224x224 → ImageNet Normalize
```

Mask:

```text
Resize 256x256 (nearest) → CenterCrop 224x224 → binary tensor
```

Mask에는 ImageNet normalization을 적용하지 않는다.

## 5. Artifact 계약

```text
artifacts/models/patchcore/<artifact-id>/
├── model.pt
└── metadata.json
```

`model.pt`에는 Python model object가 아니라 CPU tensor로 변환한 `state_dict`만 저장한다.
Backbone weights와 PatchCore memory bank가 모두 포함된다.

복원할 때는 저장 당시 `pretrained=true`였더라도 `pre_trained=False`로 모델을 생성한 뒤
`torch.load(..., weights_only=True)`로 읽은 state_dict를 엄격하게 로드한다. 따라서 완성된
artifact의 inference는 외부 pretrained weight 다운로드에 의존하지 않는다.

## 6. 실행

Artifact 생성:

```bash
uv run python -m pipelines.train_patchcore \
  --artifact-id <artifact-id> \
  --device auto
```

Raw prediction 생성:

```bash
uv run python -m pipelines.predict_patchcore \
  --artifact-dir artifacts/models/patchcore/<artifact-id> \
  --output-id <output-id> \
  --split test \
  --device auto
```

Prediction output은 sample metadata와 raw anomaly score를 담은 `predictions.jsonl`,
lossless tensor anomaly map을 담은 `anomaly_maps.pt`로 구성한다. Threshold, calibration,
F1 최적화는 수행하지 않는다.

## 7. Local 검증 범위

Local STEP 2-2에서는 unit test, integration test, 작은 CPU/MPS smoke test만 수행한다.
공식 `metal_nut` train 198장의 full memory bank 생성은 실행하지 않는다. 기준 설정의
full baseline은 STEP 2-3 Kaggle CUDA 환경에서 수행한다.
