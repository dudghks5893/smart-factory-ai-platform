# Portfolio Demo / Presentation Layer

## 1. 목적과 경계

이 문서는 SmartFactory AI Quality Platform의 **portfolio-facing demo surface**를 정리한다.
C4 model-quality, C5 backend parity, C6 streaming/service acceptance나 STEP 15 benchmark를 새로 만들거나
대체하지 않는다.

Portfolio demo는 기존에 검증한 runtime/service contract를 사람이 확인하기 쉬운 화면으로 연결한 presentation
layer다. 실제 factory production line, live factory camera, production GKE/Cloud SQL, private SOP 또는
production LLM verification으로 해석하지 않는다.

## 2. Demo chapter

발표/녹화에서는 다음 순서를 기본으로 사용한다.

1. **Combined Inspection**
   - `apps/demo_web/`
   - 한 장의 JPEG/PNG를 `POST /v1/combined-inspections`에 전달한다.
   - PatchCore + YOLO observation, Decision Policy 결과와 persisted recent history를 표시한다.
   - image inspection 결과는 durable PostgreSQL history를 가진다.

2. **Live DeepStream**
   - GStreamer / DeepStream / TensorRT INT8 runtime의 compact streaming observation을 사용한다.
   - raw frame/raw mask를 Python/FastAPI streaming payload로 전달하지 않는다.

3. **Live Monitor**
   - `apps/live_monitor/`
   - browser-local demo video를 `<video>` object URL로만 열고, WebSocket으로 수신한 bbox/class/confidence
     metadata를 Canvas overlay로 표시한다.
   - demo video file 자체를 backend 또는 PostgreSQL에 업로드하지 않는다.
   - streaming observation은 non-persisted browser-memory path다.

4. **Operations Dashboard**
   - `apps/dashboard/`
   - inspection KPI/history와 immutable drift report를 표시한다.
   - SQL credential을 갖지 않고 Vision API를 통해 inspection history를 조회한다.

5. **Synthetic Drift**
   - deterministic fixture 기반 UI/logic demonstration이다.
   - 실제 production drift observation이나 ground-truth accuracy degradation evidence가 아니다.

6. **Grafana**
   - Prometheus가 수집한 API/service telemetry를 표시한다.
   - inspection business history UI나 model-quality benchmark를 대체하지 않는다.

## 3. GCP L4 portfolio runtime

Repository에는 `compose.gcp-l4.yaml`과 `configs/deployment/gcp-l4.env.example`이 있다.

이 profile은 single-VM NVIDIA L4 portfolio runtime을 위해 다음 경계를 추가한다.

- API의 PatchCore/YOLO device를 CUDA로 설정
- API container에 NVIDIA GPU device reservation
- API / Prometheus / Grafana / Dashboard host port를 loopback으로 제한
- restored immutable runtime artifact path를 external environment에서 주입

이 single-VM Compose runtime은 `infra/k8s/overlays/gcp-gpu`의 **production target foundation과 별개**다.
GKE, Cloud SQL, public ingress/TLS, HA, HPA/CD를 완료했다고 주장하지 않는다.

검증에 사용한 NVIDIA L4 VM은 evidence와 runtime artifact를 보존한 뒤 비용 관리를 위해 삭제했다.
현재 repository는 always-on cloud deployment를 주장하지 않는다. 재구성이 필요할 때는 portable configuration과
repository 밖의 durable artifact/snapshot lifecycle을 기준으로 복원한다.

## 4. Reproducibility / evidence boundary

Portfolio UI screenshot이나 recording은 canonical benchmark evidence가 아니다.

Canonical evidence는 계속 다음 문서를 따른다.

- `docs/EVIDENCE_INDEX.md`
- `docs/vision/YOLO_SEGMENTATION_EXPERIMENT_LOG.md`
- `docs/vision/YOLO_DEPLOYMENT_OPTIMIZATION.md`
- `docs/vision/YOLO_REALTIME_STREAMING.md`
- `docs/rag/RAG_EVALUATION.md`
- `docs/benchmarks/FINAL_BENCHMARK.md`

Demo에서 사용하는 model/engine/runtime 값은 이미 frozen/accepted artifact identity를 가리키되, 영상 편집이나
UI 재생을 이유로 model-quality acceptance, threshold 또는 final-test selection을 다시 열지 않는다.

## 5. Current implementation identity

Portfolio-facing 기능을 추가한 주요 repository commit:

- Combined Inspection Web UI: `25a8f16b4cbc62e5b4f3af04ca06c0b268c1d069`
- GCP L4 single-VM Compose profile: `ba8ed65810b3ed4b717b5247af6e98bcb5ad03f1`
- Live Monitor browser-local video inference overlay: `01a1672966ab335f9be5776e93d2c10383cb73d5`

이 commit들은 presentation/demo surface의 구현 identity이며 C4/C5/C6 canonical acceptance SHA를 대체하지 않는다.
