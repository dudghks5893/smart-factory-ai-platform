"""Smoke test the PyTorch DataLoader using a generated MVTec AD manifest."""

import argparse
from pathlib import Path

from torch import Tensor
from torch.utils.data import DataLoader

from ml.datasets.dataset import MVTecManifestDataset


# ADD 2026-08-18: CLI 입력 인자를 정의하고 파싱한다.
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test the MVTec DataLoader.")
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
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="train",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


# ADD 2026-08-18: CLI 작업 흐름을 조정하고 종료 코드를 반환한다.
def main() -> int:
    # Manifest split과 DataLoader를 구성해 첫 batch를 lazy loading한다.
    args = _parse_args()
    dataset = MVTecManifestDataset(
        dataset_root=args.dataset_root,
        manifest_path=args.manifest,
        split=args.split,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Collation 결과의 image/mask tensor 계약을 확인한다.
    batch = next(iter(loader))
    images = batch["image"]
    masks = batch["mask"]

    if not isinstance(images, Tensor) or not isinstance(masks, Tensor):
        raise TypeError("Expected collated image and mask tensors.")

    print("MVTec DataLoader smoke test: PASS")
    print(f"Split: {args.split}")
    print(f"Dataset samples: {len(dataset)}")
    print(f"Batch size: {images.shape[0]}")
    print(f"Image batch shape: {tuple(images.shape)}")
    print(f"Mask batch shape: {tuple(masks.shape)}")
    print(f"Image dtype: {images.dtype}")
    print(f"Mask dtype: {masks.dtype}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
