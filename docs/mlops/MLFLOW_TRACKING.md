# MLflow Experiment Tracking & PatchCore Model Lineage

## 1. 범위

STEP 6은 서로 다른 시점과 환경에서 생성된 PatchCore 결과를 하나의 canonical MLflow run으로
backfill한다. Training, threshold calibration, evaluation 또는 benchmark 알고리즘에는 MLflow 호출을
추가하지 않는다.

```text
project-native artifacts
  config + manifest + model + threshold + metrics + benchmarks
                              ↓
             pipelines.track_patchcore_run
                              ↓
          PatchCore lineage validation/flattening
                              ↓
               project MLflow adapter
                              ↓
          MLflow backend + artifact store
```

`artifacts/`와 `outputs/`의 기존 파일이 source of truth다. MLflow artifact는 해당 파일의 추적용
copy이며 MLflow-specific 모델 포맷으로 원본 계약을 교체하지 않는다.

## 2. Local backend

외부 server가 없는 local 개발 기본값은 다음과 같다.

```text
MLFLOW_TRACKING_URI=sqlite:///outputs/mlflow/mlflow.db
MLFLOW_EXPERIMENT_NAME=smartfactory-patchcore
```

SQLite는 experiment, run, parameter, metric과 tag metadata를 저장한다. MLflow의 legacy file metadata
store가 maintenance mode이므로 relational backend인 SQLite를 선택했다. Local SQLite URI에서는 별도
설정이 없을 때 `outputs/mlflow/artifacts/`를 local artifact store로 사용한다. 두 경로는 모두
`.gitignore`의 `/outputs/` 정책 안에 있다.

Remote 환경은 같은 source code에서 `MLFLOW_TRACKING_URI`를 HTTPS tracking server URI로 바꾼다.
Remote artifact store와 credential은 server/deployment 단계에서 설정하며 repository와 pointer 파일에
secret을 저장하지 않는다. `MLFLOW_ARTIFACT_ROOT` 또는 `--artifact-location`은 새 experiment의 artifact
location을 명시해야 할 때만 사용한다.

## 3. Experiment와 run identity

기본 experiment는 `smartfactory-patchcore`다. CLI/environment로 다른 이름을 지정할 수 있다. Run name을
생략하면 검증된 artifact metadata에서 `<category>_baseline_seed<seed>` 형식으로 만든다.

Training부터 benchmark까지 장시간 active run을 유지하지 않는다. 각 stage가 생성한 기존 artifact를
모은 뒤 backfill pipeline 한 번이 단일 run을 만들고 종료한다. 성공하면 다음 immutable pointer를 쓴다.

```text
outputs/mlflow/patchcore/<tracking-id>/tracking.json
```

Pointer에는 experiment ID/name, run ID/name, artifact ID와 model/metadata/manifest/threshold SHA-256만
들어간다. Tracking URI와 credential은 기록하지 않는다. 같은 `<tracking-id>`의 overwrite는 거부하므로
실수로 동일 명령을 다시 실행해 새 run을 만드는 것을 project side에서 막는다. MLflow tag의 model SHA와
manifest SHA로 backend에서도 같은 lineage를 검색할 수 있다. 완전한 global deduplication은 범위 밖이다.

## 4. CLI

최소 lineage는 config, manifest와 model artifact로 구성한다.

```bash
uv run python -m pipelines.track_patchcore_run \
  --config configs/model/patchcore_baseline.yaml \
  --manifest data/interim/manifests/mvtec_ad_metal_nut.csv \
  --artifact-dir artifacts/models/patchcore/<artifact-id> \
  --tracking-id <tracking-id>
```

Required:

- `--config`
- `--manifest`
- `--artifact-dir` (`model.pt`, `metadata.json`)
- `--tracking-id`

Optional stage input:

- `--manifest-summary`
- `--thresholds`
- `--metrics`
- `--per-defect-metrics`
- `--model-benchmark`
- `--api-benchmark`

Optional tracking setting:

- `--tracking-uri` 또는 `MLFLOW_TRACKING_URI`
- `--experiment-name` 또는 `MLFLOW_EXPERIMENT_NAME`
- `--run-name` 또는 `MLFLOW_RUN_NAME`
- `--artifact-location` 또는 `MLFLOW_ARTIFACT_ROOT`
- `--output-root`

Evaluation metrics는 thresholds가 있어야 하고 API benchmark도 threshold provenance를 검증할 수 있어야
하므로 thresholds가 필요하다. Optional 결과가 없으면 해당 metric과 artifact를 만들거나 추정하지 않는다.
이 명령은 training, evaluation 또는 새 benchmark를 실행하지 않는다.

## 5. 기록 계약

### Parameters

- model name, category, implementation, backbone, layers
- pretrained training flag, resize/crop size, coreset ratio, neighbors, seed
- train sample count와 model file size bytes/artifact ID
- manifest row/train/validation/test-normal/test-anomaly counts
- available stage의 threshold/evaluation/benchmark counts와 조건

Nested config는 MLflow parameter에 그대로 넣지 않고 stable scalar/string으로 flatten한다.

### Tags

- `lineage.model_sha256`
- `lineage.artifact_metadata_sha256`
- `lineage.manifest_sha256`
- `lineage.threshold_artifact_sha256` (threshold가 있을 때)
- `lineage.artifact_id`
- model benchmark accelerator와 Python/PyTorch/torchvision/anomalib/CUDA runtime
- `api.benchmark_schema_version`
- `api.inspection_persistence_included`

STEP 4 공식 API benchmark schema v1은 persistence 이전 측정이므로
`api.inspection_persistence_included=false`로 기록한다. STEP 5 이후 schema v2는 JSON의 명시적 값을
사용한다.

### Metrics

- `threshold.image`, `threshold.pixel`
- `image.auroc`, `image.precision`, `image.recall`, `image.f1`, `image.tp/tn/fp/fn`
- `pixel.auroc`, `pixel.precision`, `pixel.recall`, `pixel.f1`, `pixel.tp/tn/fp/fn`
- `defect.<type>.recall`, `defect.good.false_positive_rate`
- `benchmark.model.p50_ms/p95_ms/p99_ms/mean_ms`
- model throughput, size와 available CUDA peak allocated/reserved MiB
- `api.http.p50_ms/p95_ms/p99_ms/mean_ms`, request throughput와 error rate

수치는 JSON input에서 읽는다. 문서에 기록된 STEP 3/4 공식 수치를 source code에 하드코딩하지 않는다.

### Artifacts

Allowlist로 다음 파일만 기록한다.

- relevant YAML config
- manifest CSV와 optional summary JSON
- project-native `model.pt`, `metadata.json`
- optional `thresholds.json`, `metrics.json`, `per_defect_metrics.json`
- optional model/API `benchmark.json`

Raw MVTec image, 전체 test image, `anomaly_maps.pt`, `.env`와 secret은 기록하지 않는다. Raw dataset은
크고 재배포/중복 비용이 있으며 manifest hash와 row metadata만으로 이 lineage 단계의 dataset identity를
고정할 수 있기 때문이다.

## 6. Provenance validation

MLflow 연결 전에 다음을 fail-fast 검증한다.

1. Config의 model/preprocessing/seed가 `metadata.json`과 일치한다.
2. Manifest SHA와 train count/category가 model metadata와 일치한다.
3. Optional manifest summary count가 manifest CSV와 일치한다.
4. Threshold의 manifest/model/metadata hash가 실제 파일과 일치한다.
5. Evaluation의 manifest/model/metadata/threshold hash가 canonical lineage와 일치한다.
6. Model benchmark의 manifest/model/metadata hash가 canonical lineage와 일치한다.
7. API benchmark의 manifest/model/metadata/threshold hash가 canonical lineage와 일치한다.

MLflow logging 중 오류가 발생하면 성공으로 숨기지 않는다. Run이 이미 생성되었다면 `FAILED`로 종료하고
caller에 오류를 반환하며, 성공 pointer는 만들지 않는다.

## 7. MLflow와 PostgreSQL의 책임

| System | 책임 |
|---|---|
| PostgreSQL inspection history | 실제 serving 요청별 prediction, input/model provenance, 생성 시각과 검사 이력 |
| MLflow tracking | 모델 개발 단계의 config, dataset lineage, model, threshold, 평가/benchmark metric과 artifact |

MLflow는 생산 검사 이력 database를 대체하지 않고 PostgreSQL은 experiment tracker를 대체하지 않는다.

## 8. Registry와 검증 범위

현재 구현은 Experiment Tracking과 project-native artifact lineage까지다. PatchCore memory bank 복원 계약을
억지로 MLflow pyfunc/model flavor로 감싸지 않으며 Model Registry version 등록, alias promotion, remote
artifact store와 production tracking server는 구현하지 않았다.

Temporary SQLite backend에서 실제 experiment/run 생성, params/metrics/tags/artifact logging, run 재조회,
pointer와 overwrite/실패 정책을 자동 테스트한다. 실제 remote MLflow server, concurrency, authentication,
remote artifact credential과 Registry 운영은 Docker/deployment 단계에서 별도 검증해야 한다.
