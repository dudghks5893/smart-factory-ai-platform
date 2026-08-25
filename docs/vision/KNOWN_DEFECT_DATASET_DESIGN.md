# Known-Defect Dataset Feasibility and Design

## 1. Objective

C1은 MVTec AD `metal_nut` official test anomaly image와 ground-truth mask를 분석해 known-defect
subsystem의 task formulation을 결정한다. PatchCore는 unknown anomaly detection을 계속 담당하고,
supervised subsystem은 known defect type/location을 보완하는 장기 구조다.

C1은 data feasibility, annotation representation과 split policy만 다룬다. YOLO training, dependency/weight,
supervised split file, PatchCore threshold, API, database, WebSocket와 Dashboard는 변경하지 않았다.

## 2. Source data and methodology

Source of truth는 기존 `data/interim/manifests/mvtec_ad_metal_nut.csv`이며 SHA-256은
`da81db68eadd22421ba2b284ffee85f49d41fcec47d6aadfa6bdb2cae14f285b`이다. Manifest는
수정하지 않고 다음 조건의 record만 분석했다.

- `category == metal_nut`
- `source_split == test`
- `split == test`
- `label == 1`
- non-empty ground-truth `mask_path`

모든 image/mask는 `700x700`이며 mask value는 `0/255`다. Mask를 sample 단위로 lazy load하고
8-connectivity로 component를 추출했다. Bounding box의 `x_max`, `y_max`는 inclusive pixel index이며
YOLO `xywh`는 pixel edge에서 normalized한다. Union box와 component box를 둘 다 보고, multiple
component를 무조건 하나의 box로 합치지 않았다.

Actual analysis command:

```bash
uv run python -m pipelines.analyze_known_defect_feasibility
```

Analysis-only output은 Git-ignore된 다음 경로에 생성된다.

```text
outputs/analysis/mvtec_ad/metal_nut/known_defect_feasibility/
├── sample_metrics.csv
├── component_metrics.csv
├── defect_summary.json
└── visualizations/
    ├── bent_examples.png
    ├── color_examples.png
    ├── flip_examples.png
    └── scratch_examples.png
```

Representative example은 random/cherry-pick이 아니라 mask area range, maximum fragmentation과 lowest box fill을
deterministic하게 포함한 defect별 6장이다. 각 row에 original, mask overlay와 component box를
함께 표시한다.

## 3. Benchmark isolation

이 93장은 PatchCore official MVTec AD test anomaly sample이다. C1에서는 mask geometry를 분석하는
것만 허용한다. 후속 supervised experiment가 이 sample을 train/validation에 사용하면 해당
model은 MVTec AD official anomaly benchmark에 대해 학습한 것이다. 따라서 그 결과를 official MVTec AD
test metric, 기존 PatchCore metric 또는 leakage-free combined-system metric으로 부르면 안 된다.

```text
MVTec AD original source
        ├─ PatchCore official protocol (existing, unchanged)
        └─ supervised known-defect derivative (new protocol and split lineage)
```

Derived manifest는 별도 ID/path/SHA를 가져야 하며 기존
`data/interim/manifests/mvtec_ad_metal_nut.csv`를 overwrite하지 않는다.

## 4. Ground-truth mask analysis

전체 93 sample의 mask 93개에서 132 component를 확인했다.

| Defect | Samples / masks | Components | Multi-component samples | Edge-touch samples |
| --- | ---: | ---: | ---: | ---: |
| `bent` | 25 / 25 | 35 | 7 (28.0%) | 1 (4.0%) |
| `color` | 22 / 22 | 30 | 3 (13.6%) | 0 |
| `flip` | 23 / 23 | 23 | 0 | 0 |
| `scratch` | 23 / 23 | 44 | 14 (60.9%) | 0 |
| Total | 93 / 93 | 132 | 24 | 1 |

Positive mask area와 union bounding box는 image area 대비 비율이다.

| Defect | Positive area median / mean / range | Union bbox median / mean / range | Mask-to-union-box fill median / mean / range |
| --- | --- | --- | --- |
| `bent` | 1.873% / 2.036% / 0.706–4.699% | 3.725% / 6.025% / 1.113–32.468% | 55.658% / 52.626% / 9.133–69.628% |
| `color` | 2.090% / 2.358% / 0.436–6.569% | 3.680% / 5.336% / 0.617–14.678% | 48.738% / 48.223% / 16.946–73.265% |
| `flip` | 48.173% / 48.218% / 47.847–48.725% | 78.829% / 77.670% / 67.473–80.230% | 61.250% / 62.174% / 60.177–71.480% |
| `scratch` | 5.537% / 5.899% / 1.898–12.301% | 20.002% / 23.251% / 4.867–51.703% | 30.000% / 31.716% / 9.361–69.828% |

Component box로 보면 `bent`의 median bbox area/fill은 2.628%/63.595%, `color`는
2.614%/57.673%, `scratch`는 5.070%/44.784%다. Union box의 fragmentation 손실은 component
box로 줄일 수 있지만, scratch의 thin/irregular shape 정보는 여전히 box에서 손실된다.

Centroid는 `bent` x=0.134–0.885, y=0.072–0.852, `color` x=0.107–0.832,
y=0.215–0.856, `scratch` x=0.234–0.801, y=0.234–0.795로 상대적으로 다양하다.
`flip`은 x=0.501–0.508, y=0.493–0.503으로 거의 완전히 image 중앙에 고정된다.

## 5. Defect-by-defect findings

### `bent`

Mask는 주로 nut 외곽 tab의 국소 deformation/broken geometry에 집중된다. Median positive area는
1.873%로 작고, sample의 72%는 single component이며 component box median fill이 63.595%라
box가 결함 위치를 상대적으로 잘 설명한다. 다만 28%는 multiple component이고 union box maximum이
32.468%, minimum fill이 9.133%라 멀리 떨어진 deformation을 하나의 box로 합치면 과도한
background을 포함한다.

Detection은 component-wise box라면 자연스럽고, segmentation은 불규칙한 변형 경계를 더 잘
보존한다. 전체 object geometry 변화의 성격도 있지만 actual mask와 image에서 defect는 특정
tab 부위로 localize되므로 `flip`과 같은 global-state problem으로 분류하지 않는다.

### `color`

Color defect는 dark spot, paint/marker-like region 또는 변색 patch로 보이며 median positive area는
2.090%다. 19/22 sample은 single component이지만 한 sample은 7 component를 포함한다. Component
box median fill은 57.673%로 box baseline이 가능하지만 다중 spot의 경우 각 component box와
mask boundary가 핵심이다. Color region의 shape과 area가 다양하고 fine-grained pixel extent가
불량 설명에 유용하므로 segmentation이 detection보다 자연스럽다.

### `flip`

23/23 mask가 single component이지만 이는 local defect가 아니라 제품 전체 silhouette를 표시한다.
Positive area는 매 sample에서 47.847–48.725%, box는 image의 67.473–80.230%를 차지하고
aspect ratio는 0.989–1.005다. Centroid도 image center에 거의 고정된다.

따라서 mask-derived box는 "어디가 뒤집혔는지"를 설명하지 못하고 단지 object detector와
같은 전체 nut box를 만든다. Segmentation은 silhouette extraction으로는 가능하지만 flip semantics를
localize하지 못한다. `flip` known defect는 image-level front/back classification, pose/orientation/state
classification으로 다루는 것이 맞다.

### `scratch`

Scratch는 thin line, broad abrasion band과 여러 separated scratch region이 혼재한다. Sample의 60.9%가
multiple component이고 component total은 44개다. Largest component가 mask에서 차지하는 비율의
median은 84.245%, minimum은 31.921%이다. Union bbox median area는 20.002%이지만 fill은
30.000%에 불과하고 maximum bbox는 image의 51.703%를 차지한다.

Component box는 union box보다 나지만 thin/diagonal/irregular pixel extent를 보존하지 못한다. Detection
box는 coarse affected region으로는 사용할 수 있으나 primary annotation으로는 segmentation mask가
명확하게 더 적합하다.

## 6. Task suitability matrix

Suitability는 mask geometry와 representative image의 semantic meaning을 함께 판단한 qualitative result이다.
이 grade는 model accuracy 측정값이 아니다.

| Defect | Detection | Segmentation | Image-level | Recommended task | Evidence |
| --- | --- | --- | --- | --- | --- |
| `bent` | HIGH | HIGH | MEDIUM | Segmentation + optional component boxes | Local tab deformation, component box fill median 63.6%; 28% multi-component |
| `color` | MEDIUM | HIGH | MEDIUM | Segmentation | Local irregular color regions; one 7-component sample; union fill median 48.7% |
| `flip` | LOW | LOW | HIGH | Image-level pose/orientation classification | Whole-object mask 48.2%, bbox 77.7%, center-fixed silhouette |
| `scratch` | LOW | HIGH | MEDIUM | Segmentation | 60.9% multi-component, union fill median 30.0%, thin/irregular regions |

`flip` segmentation LOW는 mask quality가 나쁘다는 뜻이 아니라, whole-object mask가 defect location을
설명하지 못한다는 task-semantic 판단이다. `scratch` detection LOW도 component box exporter를
만들 수 없다는 뜻이 아니라 mask 대비 information loss가 크다는 뜻이다.

## 7. Class taxonomy recommendation

Folder name 네 개를 하나의 detection class list로 확정하지 않는다. C1의 recommendation은
known local defect와 global state를 분리하는 것이다.

```text
Known local defect annotations
  ├─ bent    -> segmentation; component box can be derived
  ├─ color   -> segmentation; component box optional
  └─ scratch -> segmentation primary

Global state
  └─ flip    -> image-level pose/orientation classification
```

즉 후보 A(`bent/color/flip/scratch` 모두 detection class)는 기각한다. Detection subset만 사용하는
B는 `bent`에 적합하고 `color`에는 optional baseline이다. 실제 추천은 C/D의 혼합인
local segmentation과 global-state classification 분리다.

## 8. Data scarcity

Sample은 defect당 22–25장에 불과하다. Component count 35/30/23/44는 box/mask instance의
수이지 독립 image/object diversity가 아니다. 같은 product category, resolution, background와 capture
setup에서 수집된 소규모 dataset이므로 이 자료만으로 production-ready YOLO를 주장할 수 없다.

- `bent/color/scratch` centroid는 다양하지만 defect morphology와 product diversity는 제한적이다.
- `flip`은 center-fixed whole object이므로 spatial diversity가 거의 없다.
- Augmentation은 rotation, photometric과 limited geometric variation을 늘릴 수 있지만 새로운 physical
  defect mode를 만들지 못한다.
- Cross-category expansion은 class semantics가 정렬될 때만 가능하다.
- Additional approved industrial data와 MVTec AD 2를 external robustness/shift evidence로 추가해야 한다.

따라서 후속 model 실험은 feasibility baseline으로 표현하고 class별 confusion에 대한 넓은
confidence interval와 sample-level error inspection을 함께 보고해야 한다.

## 9. Proposed supervised-derived split

C2에서는 다음 policy를 versioned configuration으로 확정한다. C1에서 actual split file은 만들지
않았다.

1. Original manifest를 read-only source lineage로 사용한다.
2. `dataset_name`, `source_manifest_sha256`, `source_split`, source image/mask path를 보존한다.
3. Fixed seed(`42` 후보)와 versioned algorithm으로 defect-stratified train/validation/test를 만든다.
4. `sample_id`/image path 하나는 정확히 한 derived split에만 속하고 mask/component/box는 그
   image split을 따른다. Component를 서로 다른 split으로 나누지 않는다.
5. Class별 sample/component count, spatial distribution과 mask/bbox lineage를 summary에 기록한다.
6. Normal negative image를 추가할 경우 anomaly와 같은 uniqueness/lineage rule를 적용한다.
7. Output path와 protocol name에 `supervised_derived`를 포함하고 `official MVTec AD split`로 표현하지
   않는다.

22–25 sample/class에 single holdout만 적용하면 validation/test가 class당 수 장이 된다. C2에서
derived train/validation/test artifact를 생성하더라도, model-selection evidence는 repeated stratified
cross-validation 또는 fold sensitivity로 보완할 것을 추천한다. 단 fold를 가로지르는 image reuse를
허용하라는 뜻은 아니며, 각 fold 내 leakage는 없어야 한다.

## 10. Normalized annotation design

Model/exporter가 MVTec directory layout을 알지 않도록 adapter output을 다음 개념으로 normalization한다.

```text
DefectAnnotation
  dataset_name
  dataset_version (when source provides it)
  category
  sample_id
  source_image_reference
  image_width / image_height
  source_split / source_protocol
  defect_type
  binary_mask_reference
  components[]
    component_index
    positive_pixel_count
    bounding_box (inclusive source-pixel coordinates)
    centroid
    edge_touch
```

`dataset_name/category/sample_id/source path` combination이 lineage identity다. `component_index`는 stable scan order를
따르지만 semantic instance ID로 과장하지 않는다. MVTec의 한 binary mask에서 disconnected component가
같은 physical defect인지 다른 defect인지는 dataset이 명시하지 않기 때문이다.

Current source의 `sample_metrics.csv`/`component_metrics.csv`는 이 normalization에 필요한 geometry와
lineage를 검증하는 analysis artifact이지, 아직 training annotation/export contract가 아니다.

### Future detection exporter

Detection으로 승인된 defect/component만 source-pixel box를 normalized YOLO `xywh`로 변환한다.
Default는 component-wise box이며 union box는 explicit policy로만 허용한다. Export 전에 connectivity,
minimum component policy, class taxonomy와 disconnected-component semantics를 versioned config에 고정한다.
C1 CSV의 YOLO coordinate는 coordinate feasibility 검증용이지 class label을 확정한 것이 아니다.

### Future segmentation exporter

Binary mask는 lossless source of truth로 보존한다. YOLO segmentation polygon을 만들 때에는 contour
simplification tolerance, hole, self-intersection, multiple polygon과 thin-region loss를 검증해야 한다.
Polygon이 scratch detail을 손실하면 raster mask를 primary representation으로 유지하는 exporter/model을
고려한다. C1에서 full exporter는 구현하지 않았다.

## 11. MVTec AD 2 compatibility

Annotation core는 binary mask array에서 component/box를 추출하며 MVTec path layout을 알지 않는다.
MVTec AD-specific adapter만 기존 manifest record와 source path를 normalized metric에 연결한다. 후속
MVTec AD 2 adapter는 같은 annotation core를 사용하되 실제 source가 제공하는 경우에만
capture condition/domain partition을 추가한다. C1 output에 fake lighting/capture field를 넣지 않았다.

MVTec AD 2의 unseen lighting, illumination shift, false-positive과 drift analysis split은 supervised
training data에 흡수하지 않고 external robustness domain으로 보존해야 한다. Dataset name,
version, capture condition과 domain partition을 lineage field로 넣을 수 있는 구조를 유지한다.

## 12. C2 recommendation

**Option B: YOLO Segmentation first**를 제한된 feasibility baseline으로 추천한다.

- `bent/color/scratch`만 local segmentation taxonomy에 포함한다.
- Richest annotation인 mask를 먼저 보존하면 `bent/color`의 component box baseline을 나중에 파생할
  수 있다.
- `flip`은 segmentation/detection class에서 제외하고 별도 image-level pose/orientation task로
  유지한다.
- Data scarcity 때문에 C2의 우선 목표는 production accuracy 주장이 아니라 leakage-isolated
  derived manifest, mask/polygon export validation과 small-data baseline이어야 한다.

Option A detection-first는 scratch/flip semantics을 훼손한다. Option C detection+segmentation을 동시에
시작하면 70개 local-defect image에서 experiment surface만 늘어난다. Option D의 PatchCore 중심
운영 판단은 production에서 여전히 유효하지만, annotation feasibility는 local defect segmentation
baseline을 시도할 충분한 근거를 보였다.

## 13. Verification boundary

Synthetic unit/integration fixture는 mask loading, empty/malformed/missing mask, 8-connected component, bbox,
YOLO coordinate bounds, edge touch, path escape, deterministic order, defect aggregation과 CSV/JSON/PNG artifact
round-trip만 검증한다. 이 fixture는 위 task recommendation에 사용하지 않았다. 결론은
local raw dataset 93장에서 생성한 actual metric과 visualization을 기준으로 한다.

## 14. C2-1 implementation

C1 recommendation에 따라 `bent/color/scratch`만 포함한 supervised-derived YOLO segmentation dataset과
real-good negative split을 구현했다. 실제 split, polygon topology, round-trip fidelity와 Kaggle package
contract는 [YOLO_SEGMENTATION_DATASET.md](YOLO_SEGMENTATION_DATASET.md)에 기록한다. `flip`은 C2-1
segmentation/negative dataset에 포함하지 않았다.
