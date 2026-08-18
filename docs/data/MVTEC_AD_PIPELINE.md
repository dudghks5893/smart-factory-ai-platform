# MVTec AD Data Pipeline

## 1. 목적

이 문서는 SmartFactory AI Quality Platform에서 사용하는
MVTec AD Dataset의 Local 배치 구조, 검증 절차, Split 정책,
Manifest 생성 방식 및 PyTorch Dataset 연결 방식을 정의한다.

---

## 2. Dataset

초기 Dataset은 MVTec AD를 사용한다.

첫 개발 및 검증 Category는 `metal_nut`이다.

원본 Dataset은 Git Repository에 포함하지 않으며 다음 위치에서 관리한다.

```text
data/raw/mvtec_ad/
```

현재 `metal_nut` 원본 구성은 다음과 같다.

| 구분 | 개수 |
|---|---:|
| Official Train Good | 220 |
| Official Test Good | 22 |
| Official Test Anomaly | 93 |
| Ground-truth Mask | 93 |

Defect Type:

- bent: 25
- color: 22
- flip: 23
- scratch: 23

모든 `metal_nut` 이미지는 현재 `700x700` 해상도이다.

---

## 3. Local Data 구조

```text
data/
├── raw/
│   └── mvtec_ad/
├── interim/
│   └── manifests/
└── processed/
```

### raw

원본 Dataset을 저장한다.

원본 파일은 수정, Rename, Resize, Overwrite하지 않는다.

### interim

원본에서 재생성 가능한 중간 Metadata 및 Manifest를 저장한다.

### processed

향후 실제 Preprocessing Cache가 필요할 경우에만 사용한다.

기본 정책은 Raw Image를 Runtime에 읽어 Transform하는 Lazy Processing이다.

---

## 4. Dataset Validation

Dataset을 Model에 전달하기 전에 다음 항목을 검증한다.

- Category Directory 존재 여부
- `train/good` 존재 여부
- `test/good` 존재 여부
- Test Defect Directory와 Ground Truth Directory 대응
- Anomaly Image와 Mask 1:1 대응
- PNG 파일 Decode 가능 여부
- Image Width / Height 유효성
- 예상하지 않은 Mask 존재 여부

실행:

```bash
uv run python -m pipelines.validate_mvtec_ad \
  --dataset-root data/raw/mvtec_ad \
  --category metal_nut
```

현재 `metal_nut` 검증 결과:

```text
Dataset validation: PASS
Train good: 220
Test good: 22
Test anomaly: 93
Ground-truth masks: 93
Corrupted files: 0
Missing masks: 0
Unexpected masks: 0
```

---

## 5. Train / Validation / Test 정책

MVTec AD의 Official Test Set은 그대로 Test로 유지한다.

Official `train/good`만 내부적으로 Train과 Validation으로 분리한다.

현재 설정:

```yaml
split:
  validation_ratio: 0.1
  random_seed: 42
```

결과:

```text
Official train/good: 220
        │
        ├── Train: 198
        └── Validation: 22

Official test:
        ├── Test Good: 22
        └── Test Anomaly: 93
```

Test Dataset은 Threshold Selection에 사용하지 않는다.

Validation Split은 고정 Seed를 사용하여 동일한 입력과 설정에서
동일한 결과가 생성되도록 한다.

---

## 6. Manifest

이미지를 별도 Train/Validation Directory로 복사하지 않는다.

대신 CSV Manifest에서 내부 Split을 관리한다.

생성 위치:

```text
data/interim/manifests/mvtec_ad_metal_nut.csv
```

주요 Column:

```text
sample_id
category
source_split
split
defect_type
label
image_path
mask_path
width
height
```

`source_split`과 `split`을 구분하여 MVTec 원본 Split과
내부 Train/Validation Split을 모두 추적한다.

예:

```text
source_split = train
split        = validation
```

---

## 7. Manifest Integrity Validation

Manifest 생성 후 다음 항목을 다시 검증한다.

- `sample_id` 중복
- `image_path` 중복
- 허용되지 않은 Split
- Official Test Sample의 Split 변경
- `defect_type`과 Label 불일치
- Image Path 존재 여부
- Mask Path 존재 여부
- Manifest Image Size와 실제 Image Size 일치
- Mask Size와 Image Size 일치

검증에 실패하면 Manifest Artifact를 정상 결과로 취급하지 않는다.

---

## 8. Pipeline 실행

Dataset Validation부터 Manifest와 Summary 생성까지 한 번에 실행한다.

```bash
uv run python -m pipelines.prepare_mvtec_ad
```

현재 `metal_nut` 결과:

```text
Train: 198
Validation: 22
Test good: 22
Test anomaly: 93
Manifest rows: 335
Image sizes: {'700x700': 335}
```

Summary JSON:

```text
data/interim/manifests/mvtec_ad_metal_nut_summary.json
```

---

## 9. PyTorch Dataset / DataLoader

Manifest 기반 `MVTecManifestDataset`을 사용한다.

현재 Dataset Layer의 책임은 다음으로 제한한다.

- Manifest Split Filtering
- Raw Image Loading
- RGB Tensor 변환
- `[0, 1]` 범위 `float32` 변환
- Ground Truth Mask Loading
- Normal Sample용 Zero Mask 생성

PatchCore 전용 Resize, Normalize, Feature Extractor 입력 규칙은
Vision AI Baseline 단계에서 모델 요구사항과 함께 결정한다.

Train Smoke Test:

```bash
uv run python -m pipelines.smoke_test_mvtec_dataloader \
  --split train \
  --batch-size 4
```

현재 결과:

```text
Dataset samples: 198
Image batch shape: (4, 3, 700, 700)
Mask batch shape: (4, 1, 700, 700)
Image dtype: torch.float32
Mask dtype: torch.float32
```

Test Split은 115개 Sample을 포함한다.

---

## 10. 품질 검증

전체 품질 검사는 다음 명령으로 수행한다.

```bash
make check
```

현재 검증 범위:

- Ruff Format
- Ruff Lint
- mypy
- pytest
- Dataset Validation Unit Test
- Split Reproducibility Test
- Manifest Test
- Manifest Integrity Test
- PyTorch Dataset/DataLoader Test
- Pipeline Integration Test

---

## 11. 변경 원칙

다음 조건이 변경될 경우 Config 또는 별도 설계 결정을 통해 추적한다.

- Dataset Category
- Validation Ratio
- Random Seed
- Manifest Schema
- Image Preprocessing
- Model Input Resolution
- Dataset Version

Raw Dataset은 변경하지 않는다.
