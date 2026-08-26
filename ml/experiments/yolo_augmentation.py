"""Truthful Ultralytics training-transform previews for the YOLO workbench."""

from __future__ import annotations

import os
import random
import shutil
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

from ml.datasets.yolo_segmentation_manifest import DerivedManifestRecord
from ml.training.yolo_segmentation import (
    YoloSegmentationBaselineConfig,
    build_ultralytics_training_overrides,
)
from shared.hashing import sha256_file


@dataclass(frozen=True)
class TransformPreview:
    """One image/mask result produced by the pinned actual dataset transform path."""

    sample_id: str
    split: str
    variant: int
    imgsz: int
    image_rgb: NDArray[np.uint8]
    component_masks: tuple[NDArray[np.bool_], ...]
    class_ids: tuple[int, ...]
    transform_names: tuple[str, ...]

    # ADD 2026-08-27: Preview image/mask/class alignment과 finite geometry를 검증한다.
    def validate(self) -> None:
        if self.split not in {"train", "val"} or self.variant < 0 or self.imgsz <= 0:
            raise ValueError("Transform preview identity is invalid.")
        if (
            self.image_rgb.dtype != np.uint8
            or self.image_rgb.ndim != 3
            or self.image_rgb.shape[2] != 3
        ):
            raise ValueError("Transform preview image must be RGB uint8.")
        if len(self.component_masks) != len(self.class_ids):
            raise ValueError("Transform preview masks and classes are not aligned.")
        for mask in self.component_masks:
            if mask.dtype != np.bool_ or mask.shape != self.image_rgb.shape[:2] or not mask.any():
                raise ValueError("Transform preview contains an invalid component mask.")
        if any(class_id < 0 for class_id in self.class_ids):
            raise ValueError("Transform preview contains an invalid class ID.")


@dataclass(frozen=True)
class RepresentationPreview:
    """Non-augmented actual letterbox representation and per-component pixel evidence."""

    sample_id: str
    imgsz: int
    image_rgb: NDArray[np.uint8]
    component_mask_pixels: tuple[int, ...]
    represented_input_pixels: tuple[int, ...]
    transform_names: tuple[str, ...]


type DatasetBuilder = Callable[[YoloSegmentationBaselineConfig, Path, str], Any]


# ADD 2026-08-27: Preview RNG를 deterministic하게 격리하고 caller state를 복원한다.
@contextmanager
def isolated_preview_seed(seed: int) -> Iterator[None]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)


# ADD 2026-08-27: Source를 수정하지 않는 ignored mirror에서 Ultralytics cache/preview를 격리한다.
def prepare_preview_dataset(
    *,
    dataset_root: Path,
    preview_root: Path,
    records: list[DerivedManifestRecord],
    split: str,
) -> Path:
    if split not in {"train", "val"}:
        raise ValueError("Preview dataset accepts only train or validation.")
    selected = sorted(
        (record for record in records if record.derived_split == split),
        key=lambda record: record.sample_id,
    )
    if not selected:
        raise ValueError(f"Preview dataset contains no {split} records.")
    image_dir = preview_root / "images" / split
    label_dir = preview_root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    for record in selected:
        source_image = dataset_root / record.image_path
        source_label = dataset_root / record.label_path
        target_image = image_dir / source_image.name
        target_label = label_dir / source_label.name
        if not target_image.exists():
            try:
                target_image.symlink_to(source_image.resolve())
            except OSError:
                shutil.copy2(source_image, target_image)
        if not target_label.exists():
            shutil.copy2(source_label, target_label)
        if sha256_file(target_image) != sha256_file(source_image):
            raise RuntimeError("Preview image mirror changed source bytes.")
        if sha256_file(target_label) != sha256_file(source_label):
            raise RuntimeError("Preview label mirror changed source bytes.")
    return preview_root


# ADD 2026-08-27: Pinned trainer와 같은 build_yolo_dataset/build_transforms path를 구성한다.
def build_actual_ultralytics_dataset(
    config: YoloSegmentationBaselineConfig,
    preview_root: Path,
    split: str,
) -> Any:
    if split not in {"train", "val"}:
        raise ValueError("Actual transform preview rejects sealed test data.")
    os.environ.setdefault(
        "YOLO_CONFIG_DIR",
        str((preview_root / ".ultralytics-config").resolve()),
    )
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics.utils import DEFAULT_CFG

    overrides = build_ultralytics_training_overrides(config)
    overrides["mode"] = "train" if split == "train" else "val"
    cfg = get_cfg(DEFAULT_CFG, overrides=overrides)
    data = {
        "names": config.dataset_contract.classes,
        "nc": len(config.dataset_contract.classes),
        "channels": 3,
    }
    return build_yolo_dataset(
        cast(Any, cfg),
        str((preview_root / "images" / split).resolve()),
        config.training.batch,
        data,
        mode="train" if split == "train" else "val",
        rect=split == "val",
        stride=32,
    )


# ADD 2026-08-27: Nested Ultralytics Compose를 readable version-specific transform names로 펼친다.
def _transform_names(transform: object) -> tuple[str, ...]:
    children = getattr(transform, "transforms", None)
    if not isinstance(children, list):
        return (type(transform).__name__,)
    names: list[str] = []
    for child in children:
        names.extend(_transform_names(child))
    return tuple(names)


# ADD 2026-08-27: Formatted overlap/non-overlap mask tensor를 aligned component masks로 복원한다.
def _component_masks(
    raw_masks: object,
    *,
    image_shape: tuple[int, int],
    class_count: int,
) -> tuple[NDArray[np.bool_], ...]:
    tensor = torch.as_tensor(raw_masks).detach().cpu().numpy()
    if tensor.ndim != 3:
        raise ValueError("Ultralytics preview masks must have N,H,W dimensions.")
    low_resolution: list[NDArray[np.bool_]] = []
    if tensor.shape[0] == 1 and class_count > 1:
        low_resolution = [
            np.asarray(tensor[0] == index, dtype=np.bool_) for index in range(1, class_count + 1)
        ]
    else:
        low_resolution = [np.asarray(mask > 0, dtype=np.bool_) for mask in tensor]
    masks: list[NDArray[np.bool_]] = []
    for mask in low_resolution:
        if not mask.any():
            continue
        aligned = cv2.resize(
            mask.astype(np.uint8),
            (image_shape[1], image_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.bool_)
        aligned.setflags(write=False)
        masks.append(aligned)
    return tuple(masks)


# ADD 2026-08-27: One actual dataset sample을 immutable RGB/mask/class preview contract로 변환한다.
def _normalize_preview_sample(
    sample: dict[str, Any],
    *,
    sample_id: str,
    split: str,
    variant: int,
    imgsz: int,
    transform_names: tuple[str, ...],
    valid_class_ids: set[int],
) -> TransformPreview:
    raw_image = torch.as_tensor(sample["img"]).detach().cpu().numpy()
    if raw_image.ndim != 3:
        raise ValueError("Ultralytics preview image must have C,H,W dimensions.")
    image = np.ascontiguousarray(raw_image.transpose(1, 2, 0), dtype=np.uint8)
    raw_classes = torch.as_tensor(sample["cls"]).detach().cpu().numpy().reshape(-1)
    if not np.isfinite(raw_classes).all():
        raise ValueError("Ultralytics preview classes contain non-finite values.")
    raw_class_values = raw_classes.tolist()
    class_ids = tuple(int(value) for value in raw_class_values)
    if any(
        float(class_id) != value
        for class_id, value in zip(class_ids, raw_class_values, strict=True)
    ):
        raise ValueError("Ultralytics preview classes must be integer-valued.")
    if any(class_id not in valid_class_ids for class_id in class_ids):
        raise ValueError("Ultralytics preview contains an unsupported class ID.")
    masks = _component_masks(
        sample["masks"],
        image_shape=image.shape[:2],
        class_count=len(class_ids),
    )
    if len(masks) != len(class_ids):
        raise ValueError("Ultralytics formatted masks/classes are not one-to-one.")
    preview = TransformPreview(
        sample_id=sample_id,
        split=split,
        variant=variant,
        imgsz=imgsz,
        image_rgb=image,
        component_masks=masks,
        class_ids=class_ids,
        transform_names=transform_names,
    )
    preview.validate()
    if split == "train" and image.shape[:2] != (imgsz, imgsz):
        raise ValueError("Training transform preview has invalid image geometry.")
    return preview


# ADD 2026-08-27: Stable train sample에 actual stochastic augmentation variants를 생성한다.
def preview_actual_training_augmentations(
    *,
    config: YoloSegmentationBaselineConfig,
    preview_root: Path,
    sample_ids: list[str],
    variants: int = 3,
    dataset_builder: DatasetBuilder = build_actual_ultralytics_dataset,
) -> list[TransformPreview]:
    if variants <= 0 or not sample_ids:
        raise ValueError("Augmentation preview requires samples and positive variant count.")
    dataset = dataset_builder(config, preview_root, "train")
    paths = [Path(path) for path in cast(list[str], dataset.im_files)]
    index_by_sample = {path.stem: index for index, path in enumerate(paths)}
    transforms = _transform_names(dataset.transforms)
    if not transforms or "Format" not in transforms:
        raise RuntimeError("Pinned Ultralytics actual transform path could not be verified.")
    previews: list[TransformPreview] = []
    for sample_id in sample_ids:
        if sample_id not in index_by_sample:
            raise ValueError(f"Augmentation preview sample is missing: {sample_id}")
        for variant in range(1, variants + 1):
            with isolated_preview_seed(config.training.seed + variant):
                sample = dataset[index_by_sample[sample_id]]
            previews.append(
                _normalize_preview_sample(
                    sample,
                    sample_id=sample_id,
                    split="train",
                    variant=variant,
                    imgsz=config.training.imgsz,
                    transform_names=transforms,
                    valid_class_ids=set(config.dataset_contract.classes),
                )
            )
    return previews


# ADD 2026-08-27: Actual non-augmented letterbox/mask-grid에서 640/1024 pixel evidence를 만든다.
def preview_actual_representations(
    *,
    configs: tuple[YoloSegmentationBaselineConfig, ...],
    preview_root: Path,
    sample_id: str,
    dataset_builder: DatasetBuilder = build_actual_ultralytics_dataset,
) -> list[RepresentationPreview]:
    previews: list[RepresentationPreview] = []
    for config in configs:
        dataset = dataset_builder(config, preview_root, "val")
        paths = [Path(path) for path in cast(list[str], dataset.im_files)]
        index_by_sample = {path.stem: index for index, path in enumerate(paths)}
        if sample_id not in index_by_sample:
            raise ValueError(f"Representation preview sample is missing: {sample_id}")
        sample = dataset[index_by_sample[sample_id]]
        normalized = _normalize_preview_sample(
            sample,
            sample_id=sample_id,
            split="val",
            variant=0,
            imgsz=config.training.imgsz,
            transform_names=_transform_names(dataset.transforms),
            valid_class_ids=set(config.dataset_contract.classes),
        )
        raw_masks = torch.as_tensor(sample["masks"]).detach().cpu().numpy()
        class_count = len(normalized.class_ids)
        if raw_masks.shape[0] == 1 and class_count > 1:
            low_masks = [raw_masks[0] == index for index in range(1, class_count + 1)]
        else:
            low_masks = [mask > 0 for mask in raw_masks]
        previews.append(
            RepresentationPreview(
                sample_id=sample_id,
                imgsz=config.training.imgsz,
                image_rgb=normalized.image_rgb,
                component_mask_pixels=tuple(int(np.count_nonzero(mask)) for mask in low_masks),
                represented_input_pixels=tuple(
                    int(np.count_nonzero(mask)) for mask in normalized.component_masks
                ),
                transform_names=normalized.transform_names,
            )
        )
    return previews
