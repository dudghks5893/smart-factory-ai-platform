# PatchCore Inference Benchmark

## 1. 목적과 범위

STEP 3-3은 저장된 PatchCore artifact와 실제 MVTec AD test image로 offline
serving-oriented inference 성능을 재현한다. 별도 Kaggle 전용 Python source를 사용하지 않으며
macOS에서는 MPS/CPU, Linux GPU에서는 uv environment marker로 선택된 PyTorch cu130 runtime과
동일한 core pipeline을 실행한다. `device=auto`는 실행 시점에 CUDA, MPS, CPU 순서로 accelerator를
선택한다.

이 benchmark는 label, ground-truth mask, calibration threshold를 사용하지 않는다. 따라서 품질 metric을
재계산하거나 threshold를 적용하지 않고 model runtime 특성만 측정한다.

## 2. 시간 지표 구분

- Training wall time: embedding 수집과 coreset memory bank 생성 및 artifact 저장까지 걸린 전체 시간
- CLI runtime: artifact/manifest 검증, model restore, image disk I/O, warmup, 측정, JSON 저장을 모두 포함한 시간
- Inference latency: image batch가 disk에서 로드된 직후부터 preprocessing, device transfer,
  PatchCore inference, prediction materialization과 accelerator synchronization이 끝날 때까지의 시간

`benchmark.json`의 p50, p95, p99, mean과 throughput에는 마지막 inference latency만 사용한다. 따라서
disk image read, artifact restore, pretrained weight download, warmup과 output 저장 시간은 포함하지 않는다.
Artifact는 benchmark 시작 전에 한 번만 복원하며 반복 구간에서 다시 로드하지 않는다.

## 3. 측정 정책

- 기본 batch size: `1`
- 기본 warmup: test split의 첫 `10`개 batch
- 기본 measured set: manifest에 기록된 test split 전체(현재 metal_nut은 115 images)
- Sample order: manifest order, `shuffle=False`
- Autograd: `torch.inference_mode()`
- Threshold: 적용하지 않음
- Percentile: 정렬된 관측치 사이의 linear interpolation
- Throughput: measured image count / measured batch latency 합계

Warmup 후 같은 test DataLoader를 처음부터 다시 순회하므로 warmup image도 measured set에서 빠지지 않는다.
Warmup latency는 percentile, mean, throughput과 CUDA peak memory에서 제외한다. CUDA는 각 측정 시작 전과
prediction 후 `torch.cuda.synchronize()`를 호출해 비동기 kernel이 timer 밖으로 빠져나가지 않게 한다.
MPS도 대응하는 synchronization을 수행하고 CPU는 no-op이다.

Batch size가 1이면 각 latency 관측치가 image latency다. Batch size가 1보다 크면 p50/p95/p99/mean은
batch latency이고 throughput은 계속 images/second이다. 서로 다른 batch size 결과는 같은 조건끼리 비교한다.

## 4. CUDA memory

CUDA에서는 warmup과 동기화가 끝난 뒤 allocator peak 통계를 reset하고 measured inference 동안 다음 값을
수집한다.

- peak allocated bytes/MiB
- peak reserved bytes/MiB

Artifact는 reset 전에 이미 복원되어 있으므로 peak 값은 resident model/memory bank와 measured inference
allocation을 함께 반영한다. CPU와 MPS에서는 `supported=false`와 `null` 값을 기록하며 CUDA API 부재로
실패하지 않는다.

## 5. 실행과 output

```bash
uv run python -m pipelines.benchmark_patchcore \
  --dataset-root data/raw/mvtec_ad \
  --manifest data/interim/manifests/mvtec_ad_metal_nut.csv \
  --artifact-dir artifacts/models/patchcore/<artifact-id> \
  --output-id <benchmark-id> \
  --device auto
```

필요할 때만 `--batch-size`, `--warmup-count`, `--measured-count`, `--num-workers`를 명시한다. 기본 output은
`outputs/benchmarks/patchcore/<benchmark-id>/benchmark.json`이다. 기존 output directory 또는 JSON을
overwrite하지 않는다.

`benchmark.json`은 schema version, category/device, Python/PyTorch/torchvision/anomalib version,
manifest/artifact metadata/model SHA-256, backbone/layers/preprocessing, batch/warmup/measured sample 및 batch
count, latency definition, percentile method, p50/p95/p99/mean, throughput, `model.pt` 크기, CUDA peak memory와
생성 시각을 기록한다. 실제 latency 값은 위 pipeline을 대상 runtime에서 실행했을 때만 생성하며 문서에
추정값을 기입하지 않는다.
