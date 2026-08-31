"""Fast-compatible prediction normalization for C4-2C validation confirmation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from ml.evaluation.yolo_segmentation_error_analysis import PredictedInstance, mask_box


class ConfirmationModel(Protocol):
    """Minimal already-loaded Ultralytics prediction surface."""

    def predict(self, **kwargs: object) -> Sequence[object]: ...


def _array(value: object, *, field: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"C4-2C prediction is missing {field}.")
    detached = getattr(value, "detach", None)
    materialized = detached() if callable(detached) else value
    cpu = getattr(materialized, "cpu", None)
    materialized = cpu() if callable(cpu) else materialized
    numpy = getattr(materialized, "numpy", None)
    result = np.asarray(numpy() if callable(numpy) else materialized)
    if not np.issubdtype(result.dtype, np.number) or not np.isfinite(result).all():
        raise ValueError(f"C4-2C prediction {field} must be finite numeric data.")
    return result


# ADD 2026-08-31: Fast R17 predict args와 nearest source-size mask contract를 재현한다.
def predict_c4_2c_instances(
    *,
    model: ConfirmationModel,
    source_image_path: Path,
    image_width: int,
    image_height: int,
    imgsz: int,
    device: str,
    valid_class_ids: set[int],
) -> tuple[PredictedInstance, ...]:
    if not source_image_path.is_file() or not valid_class_ids:
        raise FileNotFoundError(f"C4-2C validation source is missing: {source_image_path}")
    raw = list(
        model.predict(
            source=str(source_image_path),
            conf=0.001,
            iou=0.7,
            max_det=300,
            retina_masks=False,
            imgsz=imgsz,
            device=device,
            save=False,
            stream=False,
            verbose=False,
        )
    )
    if len(raw) != 1:
        raise ValueError("C4-2C prediction must return one result per source image.")
    boxes = getattr(raw[0], "boxes", None)
    if boxes is None:
        raise ValueError("C4-2C prediction is missing boxes.")
    classes = _array(getattr(boxes, "cls", None), field="boxes.cls").reshape(-1)
    confidences = _array(getattr(boxes, "conf", None), field="boxes.conf").reshape(-1)
    masks = getattr(raw[0], "masks", None)
    if len(classes) == 0:
        return ()
    if masks is None:
        raise ValueError("C4-2C segmentation prediction is missing masks.")
    mask_values = _array(getattr(masks, "data", None), field="masks.data")
    if mask_values.ndim != 3 or mask_values.shape[0] != len(classes):
        raise ValueError("C4-2C prediction box/mask counts are not aligned.")
    if len(confidences) != len(classes):
        raise ValueError("C4-2C prediction class/confidence counts are not aligned.")

    normalized: list[PredictedInstance] = []
    for index, raw_class in enumerate(classes):
        class_value = float(raw_class)
        if not class_value.is_integer() or int(class_value) not in valid_class_ids:
            raise ValueError("C4-2C prediction contains an invalid class ID.")
        mask = np.asarray(mask_values[index] > 0.5, dtype=np.uint8)
        if mask.shape != (image_height, image_width):
            mask = cv2.resize(
                mask,
                (image_width, image_height),
                interpolation=cv2.INTER_NEAREST,
            )
        mask_bool = np.asarray(mask > 0, dtype=np.bool_)
        if not mask_bool.any():
            continue
        normalized.append(
            PredictedInstance(
                class_id=int(class_value),
                confidence=float(confidences[index]),
                mask=mask_bool,
                box_xyxy=mask_box(mask_bool),
            )
        )
    return tuple(normalized)
