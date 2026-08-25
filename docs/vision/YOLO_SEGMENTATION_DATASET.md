# MVTec AD-Derived YOLO Segmentation Dataset

## 1. Scope and benchmark isolation

C2-1은 MVTec AD `metal_nut`의 binary ground-truth mask에서 YOLO segmentation polygon을
생성하는 dataset-only 단계다. 학습, pretrained weight, Ultralytics dependency와 serving 변경은
포함하지 않는다.

Source of truth는 기존
`data/interim/manifests/mvtec_ad_metal_nut.csv`이며 SHA-256은
`da81db68eadd22421ba2b284ffee85f49d41fcec47d6aadfa6bdb2cae14f285b`이다. 이 Manifest와
기존 PatchCore official evaluation contract는 수정하지 않았다.

Positive 70장은 원래 MVTec AD official test anomaly다. 따라서 이 derived dataset의 후속 metric은
**MVTec AD-derived supervised segmentation feasibility split**로만 표현한다. Official MVTec AD
benchmark 또는 leakage-free PatchCore combined metric으로 표현하면 안 된다.

## 2. Task taxonomy

Class mapping의 source of truth는
`configs/data/mvtec_ad_metal_nut_yolo_segmentation.yaml`이다.

| Class ID | Name | Source samples |
| ---: | --- | ---: |
| 0 | `bent` | 25 |
| 1 | `color` | 22 |
| 2 | `scratch` | 23 |

`flip`은 whole-object pose/orientation 상태이므로 segmentation positive에서 제외한다. 실제 anomaly인
`flip`을 background negative로도 사용하지 않는다.

## 3. Good negative policy

Manifest의 real `good` image 242장 가운데 positive 수와 동일한 70장을 seed `42`의 SHA-256 rank로
선택했다. Derived split별 positive:negative를 정확히 1:1로 유지한다. 선택된 good source의 기존
Manifest split은 train 58장, validation 6장, test 6장이며 모든 source image/hash lineage를 보존한다.

Negative image도 원본 bytes를 self-contained package에 복사하며 대응 label file은 존재하지만
길이가 0 byte다. `flip`이나 unlabeled anomaly를 negative로 대체하지 않는다.

## 4. Deterministic supervised split

Nominal ratio는 train/validation/test `60/20/20`이다. Class별 validation/test 수를 독립적으로
반올림하고 remainder를 train에 할당했다. 이 정책은 22–25장뿐인 각 class에서 validation과 test를
각각 4–5장 확보하면서 전체 split을 정확히 42/14/14 positive로 만든다.

| Derived split | Bent | Color | Scratch | Positive | Good negative | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 15 | 14 | 13 | 42 | 42 | 84 |
| val | 5 | 4 | 5 | 14 | 14 | 28 |
| test | 5 | 4 | 5 | 14 | 14 | 28 |
| Total | 25 | 22 | 23 | 70 | 70 | 140 |

Assignment는 class/sample ID/seed를 SHA-256으로 rank하므로 input ordering과 process hash seed의 영향을
받지 않는다. Image 하나와 그 mask/component/polygon은 같은 split에 머물고 source image는 split을
가로질러 재사용되지 않는다.

## 5. Polygon conversion

Binary PNG mask는 계속 lossless source of truth다. Exporter는 OpenCV `RETR_CCOMP`로 contour hierarchy를
읽고 8-connected external component를 각각 한 polygon으로 보존한다. 모든 좌표는 source
`700x700` pixel coordinate를 width/height로 나눈 `[0, 1]` 값이다. Image resize나 recompression은
하지 않는다.

`CHAIN_APPROX_SIMPLE`의 collinear chain compression만 사용한다. Geometric epsilon은 `0.0`이며
`approxPolyDP` 같은 tolerance simplification을 적용하지 않는다. 실제 70개 label은 polygon 109개,
vertex 25,028개, 총 550,834 bytes다. Sample label의 median은 5,755 bytes, maximum은 scratch sample의
23,016 bytes로 feasibility package에서 허용 가능한 크기다. 최소 3개 unique vertex와 non-zero area를
검증하며 degenerate contour는 거부한다.

## 6. Hole topology

실제 70개 mask 가운데 4개 sample에 총 13개의 작은 hole contour가 존재했다.

| Sample | Defect | Holes | Filled pixels introduced by YOLO round-trip |
| --- | --- | ---: | ---: |
| `metal_nut_test_bent_010` | bent | 10 | 16 |
| `metal_nut_test_color_008` | color | 1 | 1 |
| `metal_nut_test_scratch_002` | scratch | 1 | 1 |
| `metal_nut_test_scratch_008` | scratch | 1 | 1 |

YOLO segmentation polygon union은 subtraction hole을 표현하지 못한다. Exporter는 이를 숨기지 않고
`hole_count`, source mask SHA와 round-trip precision/IoU를 manifest 및 metadata에 기록한다. Hole을
채우면서 source positive pixel은 모두 유지되므로 recall은 1.0이고 precision/IoU에만 제한된 손실이
발생한다. Binary source mask를 삭제하거나 polygon으로 대체하지 않는다.

## 7. Round-trip fidelity

모든 positive sample에 `mask -> normalized polygon label -> rasterized mask` round-trip을 수행했다.

| Scope | IoU min | IoU p05 | IoU median | IoU mean | Precision min | Recall min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall | 0.998870 | 0.999990 | 1.000000 | 0.999983 | 0.998870 | 1.000000 |
| Bent | 0.998870 | 1.000000 | 1.000000 | 0.999955 | 0.998870 | 1.000000 |
| Color | 0.999956 | 1.000000 | 1.000000 | 0.999998 | 0.999956 | 1.000000 |
| Scratch | 0.999971 | 0.999984 | 1.000000 | 0.999998 | 0.999971 | 1.000000 |

Worst sample은 `metal_nut_test_bent_010`이며 IoU/precision `0.998869737214`, recall `1.0`이다.
다음 세 sample은 `color_008` 0.999955682, `scratch_008` 0.999971076,
`scratch_002` 0.999982567이다.

Actual distribution을 확인한 뒤 conversion-integrity gate를 sample IoU/precision `>=0.998`, recall
`>=0.999`, defect p05 IoU `>=0.999`로 정했다. 이는 model quality threshold가 아니라 현재 YOLO
representation에서 hole loss 또는 향후 exporter regression을 silent pass시키지 않기 위한 gate다.
현재 dataset은 모든 gate를 통과한다.

## 8. Output and lineage contract

Local derived workspace는 다음 self-contained 구조다.

```text
data/processed/supervised_derived/mvtec_ad/metal_nut/yolo_segmentation/v1/
├── images/{train,val,test}/
├── labels/{train,val,test}/
├── dataset.yaml
├── manifest.csv
└── metadata.json
```

`dataset.yaml`은 absolute local path 없이 `path: .`, `images/train`, `images/val`, `images/test`와
class mapping만 기록한다. C2-2에서는 package root를 current dataset root로 해석해야 한다.

Derived manifest는 dataset/task/version, source Manifest SHA, existing `source_split`과
`source_manifest_split`, source image/mask path와 SHA, copied image/label path, class, derived split,
component/polygon/hole/vertex count 및 sample fidelity를 기록한다. 현재 derived Manifest SHA-256은
`1746338c091c18e96a11399c81ea9be0d7350105c4860cfa6a4162144ddb9905`다.

Metadata의 `semantic_fingerprint_sha256`은 timestamp를 제외한 source SHA, derived Manifest SHA,
seed, split/negative/polygon/gate 정책과 count를 canonical JSON으로 hash한다. `created_at_utc`는 실행
관찰값일 뿐 semantic identity가 아니다.

## 9. Validation and visualization

Exporter는 image readability/dimensions/byte hash, source Manifest fields/hashes, mask hash, label schema,
coordinate/class bounds, component/polygon count, hole lineage, empty negative label, split uniqueness,
flip exclusion, count, portable YAML와 fidelity gate를 export 후 다시 검증한다.

```bash
uv run python -m pipelines.export_yolo_segmentation_dataset --validate-only
```

Representative original/GT/polygon overlay/reconstruction montage는
`outputs/analysis/mvtec_ad/metal_nut/yolo_segmentation/v1/` 아래에 class별로 생성된다. Good montage는
GT/label/reconstruction이 모두 empty임을 보여준다. 이 visualization은 Git 대상이 아니다.

## 10. Kaggle package and C2-2 contract

Kaggle package는 `outputs/packages/mvtec_ad_metal_nut_yolo_seg_v1.zip`이다. 사용된 image/label과
`dataset.yaml`, manifest, metadata만 포함하며 전체 raw MVTec tree나 source mask를 중복 포함하지
않는다. File 283개, uncompressed content 약 70.3 MB이며 package SHA-256은
`3cd656e53a59044eb3d008a8b18c35bb16d9b6f0fdb597387a81ab359368852e`다.

C2-2는 ZIP을 Kaggle에 올린 뒤 다음을 먼저 확인해야 한다.

1. Package SHA와 derived Manifest SHA를 확인한다.
2. `--validate-only`와 동등한 dataset validation을 수행한다.
3. 이 split을 feasibility protocol로 명명하고 official MVTec test metric과 분리한다.
4. Ultralytics dependency/version, pretrained weight와 training config를 C2-2에서 별도로 고정한다.
5. Evaluation은 derived test 28장에만 수행하고 original PatchCore 결과와 합산하지 않는다.

## 11. Data scarcity and MVTec AD 2

Positive는 70장뿐이고 class별 test는 4–5장이다. Augmentation은 새로운 physical defect나 capture
diversity를 만들지 못한다. C2-2 결과는 production-ready segmenter가 아니라 small-data feasibility
baseline으로 해석하고 per-sample error를 함께 검토해야 한다.

Polygon core는 boolean mask와 normalized annotation만 다루며 MVTec directory layout을 알지 않는다.
향후 MVTec AD 2 adapter는 dataset/version/capture condition/domain partition lineage를 추가할 수 있다.
AD 2의 illumination/domain-shift sample은 training에 자동 흡수하지 않고 external robustness protocol로
보존한다.
