"""Build a PatchCore memory-bank artifact from the manifest train split."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader

from ml.datasets.dataset import MVTecManifestDataset
from ml.datasets.manifest_validation import validate_manifest_records
from ml.training.config import PatchCoreBaselineConfig, load_patchcore_config
from ml.training.device import SUPPORTED_DEVICES, resolve_device
from ml.training.patchcore import PatchCoreAdapter, PatchCoreArtifactMetadata
from ml.training.preprocessing import PatchCorePreprocessor


@dataclass(frozen=True)
class PatchCoreTrainingResult:
    """Summary of a completed PatchCore memory-bank construction run."""

    artifact_dir: Path
    metadata: PatchCoreArtifactMetadata
    memory_bank_shape: tuple[int, ...]
    device: str


# ADD 2026-08-19: Construct and persist PatchCore from normal train records only.
# MODIFY 2026-08-19: 기본 split 검사 → manifest와 artifact 경로를 fail-fast 검증한다.
def train_patchcore(
    *,
    config: PatchCoreBaselineConfig,
    dataset_root: Path,
    manifest_path: Path,
    category: str,
    artifact_dir: Path,
    requested_device: str | None = None,
) -> PatchCoreTrainingResult:
    """Construct and persist PatchCore from normal train records only."""
    # Config 기반 device를 먼저 확정하고 입력/출력 파일 조건을 fail-fast 검증한다.
    device = resolve_device(requested_device or config.training.device)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Training manifest not found: {manifest_path}")
    if artifact_dir.exists():
        raise FileExistsError(f"Artifact directory already exists: {artifact_dir}")

    # Manifest의 train split만 lazy dataset으로 로드한다.
    dataset = MVTecManifestDataset(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        split="train",
    )

    # 비용이 큰 feature extraction 전에 category, label, path integrity를 검증한다.
    if any(record.category != category for record in dataset.records):
        raise ValueError(f"Train manifest records do not all match category: {category}")
    if any(record.label != 0 for record in dataset.records):
        raise ValueError("PatchCore train split must contain only normal label 0 records.")

    manifest_report = validate_manifest_records(dataset.records, dataset_root)
    if not manifest_report.is_valid:
        raise ValueError(
            "PatchCore train manifest validation failed:\n" + "\n".join(manifest_report.errors)
        )

    # 결정적 순서를 유지하는 DataLoader와 PatchCore runtime component를 구성한다.
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=device.type == "cuda",
    )
    preprocessor = PatchCorePreprocessor(config.preprocessing)
    adapter = PatchCoreAdapter.for_training(config, device)

    # Normal embedding을 수집하고 coreset memory bank를 생성한다.
    train_sample_count = adapter.fit(loader, preprocessor)
    if train_sample_count != len(dataset):
        raise RuntimeError(
            f"PatchCore processed {train_sample_count} samples; expected {len(dataset)}."
        )

    # 완성된 backbone/memory bank state_dict와 provenance metadata를 저장한다.
    metadata = adapter.save_artifact(
        artifact_dir=artifact_dir,
        category=category,
        train_sample_count=train_sample_count,
        manifest_path=manifest_path,
        random_seed=config.training.random_seed,
    )
    return PatchCoreTrainingResult(
        artifact_dir=artifact_dir,
        metadata=metadata,
        memory_bank_shape=tuple(adapter.model.memory_bank.shape),
        device=str(device),
    )


# ADD 2026-08-19: CLI 입력 인자를 정의하고 파싱한다.
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a PatchCore baseline artifact.")
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
    parser.add_argument("--category", default="metal_nut")
    parser.add_argument(
        "--artifact-id",
        required=True,
        help="Directory name created below output.artifact_root.",
    )
    parser.add_argument("--device", choices=SUPPORTED_DEVICES)
    return parser.parse_args()


# ADD 2026-08-19: CLI 작업 흐름을 조정하고 종료 코드를 반환한다.
def main() -> int:
    # CLI config를 로드하고 artifact destination을 결정한다.
    args = _parse_args()
    config = load_patchcore_config(args.config)
    artifact_dir = config.output.artifact_root / args.artifact_id
    # Manifest train split에서 PatchCore artifact를 생성한다.
    result = train_patchcore(
        config=config,
        dataset_root=args.dataset_root,
        manifest_path=args.manifest,
        category=args.category,
        artifact_dir=artifact_dir,
        requested_device=args.device,
    )

    print("PatchCore training: PASS")
    print(f"Device: {result.device}")
    print(f"Train samples: {result.metadata.train_sample_count}")
    print(f"Memory bank shape: {result.memory_bank_shape}")
    print(f"Artifact: {result.artifact_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
