"""Run raw PatchCore predictions from a portable model artifact."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ml.datasets.constants import MVTEC_SPLITS
from ml.datasets.dataset import MVTecManifestDataset
from ml.training.batches import require_batch_tensor
from ml.training.config import PatchCoreBaselineConfig, load_patchcore_config
from ml.training.device import SUPPORTED_DEVICES, resolve_device
from ml.training.patchcore import PatchCoreAdapter, read_artifact_metadata
from ml.training.preprocessing import PatchCorePreprocessor
from shared.hashing import sha256_file

PREDICTIONS_FILENAME = "predictions.jsonl"
ANOMALY_MAPS_FILENAME = "anomaly_maps.pt"


@dataclass(frozen=True)
class RawPredictionRecord:
    """Threshold-free prediction metadata for one manifest sample."""

    sample_id: str
    category: str
    defect_type: str
    label: int
    split: str
    raw_anomaly_score: float
    anomaly_map_key: str
    anomaly_map_file: str = ANOMALY_MAPS_FILENAME


@dataclass(frozen=True)
class PredictionOutputSummary:
    """Summary of persisted raw predictions."""

    output_dir: Path
    sample_count: int
    predictions_path: Path
    anomaly_maps_path: Path
    device: str


# ADD 2026-08-19: Persist raw anomaly scores and lossless anomaly maps for a manifest split.
# MODIFY 2026-08-19: 모델 선복원 → artifact와 manifest를 검증한 뒤 모델을 복원한다.
def predict_patchcore(
    *,
    config: PatchCoreBaselineConfig,
    dataset_root: Path,
    manifest_path: Path,
    artifact_dir: Path,
    output_dir: Path,
    split: str,
    requested_device: str | None = None,
) -> PredictionOutputSummary:
    """Persist raw anomaly scores and lossless anomaly maps for a manifest split."""
    if output_dir.exists():
        raise FileExistsError(f"Prediction output directory already exists: {output_dir}")

    # 모델 복원 전에 device, artifact layout, metadata와 manifest hash를 검증한다.
    device = resolve_device(requested_device or config.training.device)
    artifact_metadata = read_artifact_metadata(artifact_dir)
    prediction_manifest_sha256 = sha256_file(manifest_path)
    if prediction_manifest_sha256 != artifact_metadata.manifest_sha256:
        raise ValueError(
            "Prediction manifest SHA-256 does not match the loaded PatchCore artifact."
        )
    # 선택한 manifest split을 inference 대상으로 로드하고 artifact category와 대조한다.
    dataset = MVTecManifestDataset(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        split=split,
    )
    if any(record.category != artifact_metadata.category for record in dataset.records):
        raise ValueError(
            "Prediction manifest category does not match the loaded PatchCore artifact."
        )

    # 검증된 metadata로 외부 weight 다운로드 없이 PatchCore artifact를 복원한다.
    adapter, _ = PatchCoreAdapter.load_artifact(
        artifact_dir,
        device,
        metadata=artifact_metadata,
    )

    # Artifact preprocessing 계약과 결정적 sample 순서로 inference runtime을 구성한다.
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=device.type == "cuda",
    )
    preprocessor = PatchCorePreprocessor(artifact_metadata.preprocessing)
    records: list[RawPredictionRecord] = []
    anomaly_maps: dict[str, Tensor] = {}

    # Threshold 적용 없이 batch별 raw score와 anomaly map을 CPU output으로 수집한다.
    for batch in loader:
        images = require_batch_tensor(batch, "image")
        labels = require_batch_tensor(batch, "label")
        sample_ids = _batch_strings(batch, "sample_id")
        categories = _batch_strings(batch, "category")
        defect_types = _batch_strings(batch, "defect_type")
        prediction = adapter.predict(images, preprocessor)

        if prediction.scores.shape[0] != len(sample_ids):
            raise RuntimeError("PatchCore score batch size does not match sample metadata.")

        for index, sample_id in enumerate(sample_ids):
            if sample_id in anomaly_maps:
                raise ValueError(f"Duplicate sample_id in prediction output: {sample_id}")
            anomaly_maps[sample_id] = prediction.anomaly_maps[index].clone()
            records.append(
                RawPredictionRecord(
                    sample_id=sample_id,
                    category=categories[index],
                    defect_type=defect_types[index],
                    label=int(labels[index].item()),
                    split=split,
                    raw_anomaly_score=float(prediction.scores[index].item()),
                    anomaly_map_key=sample_id,
                )
            )

    if len(records) != len(dataset):
        raise RuntimeError(f"Produced {len(records)} predictions; expected {len(dataset)}.")

    # STEP 3에서 재사용할 JSONL metadata와 lossless tensor map을 저장한다.
    output_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = output_dir / PREDICTIONS_FILENAME
    predictions_path.write_text(
        "".join(json.dumps(asdict(record), ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    anomaly_maps_path = output_dir / ANOMALY_MAPS_FILENAME
    torch.save(anomaly_maps, anomaly_maps_path)

    return PredictionOutputSummary(
        output_dir=output_dir,
        sample_count=len(records),
        predictions_path=predictions_path,
        anomaly_maps_path=anomaly_maps_path,
        device=str(device),
    )


# ADD 2026-08-19: Collated batch의 string sequence field를 검증해 반환한다.
def _batch_strings(batch: dict[str, object], field: str) -> list[str]:
    value = batch.get(field)
    if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"Batch field '{field}' must be a sequence of strings.")
    return list(value)


# ADD 2026-08-19: CLI 입력 인자를 정의하고 파싱한다.
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run threshold-free PatchCore predictions.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/model/patchcore_baseline.yaml"),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/raw/mvtec_ad"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/interim/manifests/mvtec_ad_metal_nut.csv"),
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-id", required=True)
    parser.add_argument("--split", choices=MVTEC_SPLITS, default="test")
    parser.add_argument("--device", choices=SUPPORTED_DEVICES)
    return parser.parse_args()


# ADD 2026-08-19: CLI 작업 흐름을 조정하고 종료 코드를 반환한다.
def main() -> int:
    # CLI config와 output destination을 결정한다.
    args = _parse_args()
    config = load_patchcore_config(args.config)
    output_dir = config.output.prediction_root / args.output_id
    # 선택한 manifest split의 raw PatchCore prediction을 생성한다.
    summary = predict_patchcore(
        config=config,
        dataset_root=args.dataset_root,
        manifest_path=args.manifest,
        artifact_dir=args.artifact_dir,
        output_dir=output_dir,
        split=args.split,
        requested_device=args.device,
    )

    print("PatchCore prediction: PASS")
    print(f"Device: {summary.device}")
    print(f"Split: {args.split}")
    print(f"Samples: {summary.sample_count}")
    print(f"Predictions: {summary.predictions_path}")
    print(f"Anomaly maps: {summary.anomaly_maps_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
