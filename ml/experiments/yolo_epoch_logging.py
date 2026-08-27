"""Compact fit-epoch evidence logging for project-owned YOLO training runs."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch


@dataclass(frozen=True)
class EpochMetricRecord:
    """One completed fit epoch timed from train-epoch start to fit-epoch end."""

    epoch: int
    total_epochs: int
    epoch_time_seconds: float
    cumulative_epoch_seconds: float
    train_box_loss: float | None
    train_seg_loss: float | None
    train_cls_loss: float | None
    train_dfl_loss: float | None
    val_box_precision: float | None
    val_box_recall: float | None
    val_box_map50: float | None
    val_box_map50_95: float | None
    val_mask_precision: float | None
    val_mask_recall: float | None
    val_mask_map50: float | None
    val_mask_map50_95: float | None
    learning_rate_pg0: float | None
    cuda_memory_reserved_bytes: int | None

    # ADD 2026-08-27: Epoch scalar evidence를 strict JSON-compatible mapping으로 변환한다.
    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        json.dumps(payload, allow_nan=False)
        return payload


METRIC_KEYS = {
    "train_box_loss": "train/box_loss",
    "train_seg_loss": "train/seg_loss",
    "train_cls_loss": "train/cls_loss",
    "train_dfl_loss": "train/dfl_loss",
    "val_box_precision": "metrics/precision(B)",
    "val_box_recall": "metrics/recall(B)",
    "val_box_map50": "metrics/mAP50(B)",
    "val_box_map50_95": "metrics/mAP50-95(B)",
    "val_mask_precision": "metrics/precision(M)",
    "val_mask_recall": "metrics/recall(M)",
    "val_mask_map50": "metrics/mAP50(M)",
    "val_mask_map50_95": "metrics/mAP50-95(M)",
    "learning_rate_pg0": "lr/pg0",
}


# ADD 2026-08-27: Framework scalar를 finite float로 읽고 unavailable value는 null로 보존한다.
def _optional_scalar(metrics: Mapping[str, object], key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    if not isinstance(value, int | float):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


class EpochMetricsLogger:
    """Lifecycle-aware JSONL and concise console logger with an injectable clock."""

    # ADD 2026-08-27: Logger를 초기화한다. → MODIFY 2026-08-28: Lifecycle state를 추가한다.
    def __init__(
        self,
        *,
        output_path: Path,
        total_epochs: int,
        clock: Callable[[], float] = perf_counter,
        console_writer: Callable[[str], None] = print,
    ) -> None:
        if total_epochs <= 0:
            raise ValueError("Epoch logger total_epochs must be positive.")
        if output_path.exists():
            raise FileExistsError(f"Epoch log already exists: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path = output_path
        self.total_epochs = total_epochs
        self._clock = clock
        self._console_writer = console_writer
        self._epoch_started_at: float | None = None
        self._active_epoch: int | None = None
        self._last_completed_epoch: int | None = None
        self._final_eval_callback_consumed = False
        self._cumulative_seconds = 0.0
        self._closed = False

    # ADD 2026-08-27: Timing을 연다. → MODIFY 2026-08-28: Final-eval 후 start를 막는다.
    def start_epoch(self, epoch: int) -> None:
        if self._closed or self._active_epoch is not None:
            raise RuntimeError("Epoch logger is closed or already timing an epoch.")
        if self._final_eval_callback_consumed:
            raise RuntimeError("Epoch logger already consumed its post-training callback.")
        if not 1 <= epoch <= self.total_epochs:
            raise ValueError("Epoch number is outside the configured training range.")
        self._active_epoch = epoch
        self._epoch_started_at = self._clock()

    # ADD 2026-08-27: Epoch evidence를 쓴다. → MODIFY 2026-08-28: 완료 epoch를 추적한다.
    def finish_epoch(
        self,
        *,
        epoch: int,
        metrics: Mapping[str, object],
        cuda_memory_reserved_bytes: int | None = None,
    ) -> EpochMetricRecord:
        if self._closed or self._active_epoch != epoch or self._epoch_started_at is None:
            raise RuntimeError("Epoch finish does not match an active timing boundary.")
        duration = self._clock() - self._epoch_started_at
        if not math.isfinite(duration) or duration < 0.0:
            raise ValueError("Measured fit-epoch elapsed time must be finite and non-negative.")
        self._cumulative_seconds += duration
        values = {
            field: _optional_scalar(metrics, source_key)
            for field, source_key in METRIC_KEYS.items()
        }
        record = EpochMetricRecord(
            epoch=epoch,
            total_epochs=self.total_epochs,
            epoch_time_seconds=duration,
            cumulative_epoch_seconds=self._cumulative_seconds,
            cuda_memory_reserved_bytes=cuda_memory_reserved_bytes,
            **values,
        )
        with self.output_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record.to_json_dict(), sort_keys=True, allow_nan=False) + "\n")
        self._console_writer(
            "[C4 EXPERIMENT] "
            f"Epoch {epoch:03d}/{self.total_epochs} "
            f"epoch_time_seconds={duration:.3f} "
            f"cumulative_epoch_seconds={self._cumulative_seconds:.3f} "
            f"train_seg_loss={values['train_seg_loss']} "
            f"val_mask_map50_95={values['val_mask_map50_95']}"
        )
        self._active_epoch = None
        self._epoch_started_at = None
        self._last_completed_epoch = epoch
        return record

    # ADD 2026-08-27: Training exception에서도 incomplete timing state를 폐기한다.
    def close(self) -> None:
        self._active_epoch = None
        self._epoch_started_at = None
        self._closed = True

    # ADD 2026-08-27: Ultralytics on_train_epoch_start callback을 stable logger API에 연결한다.
    def on_train_epoch_start(self, trainer: Any) -> None:
        self.start_epoch(int(trainer.epoch) + 1)

    # ADD 2026-08-27: Fit-end를 기록한다. → MODIFY 2026-08-28: Final-eval 1회만 무시한다.
    def on_fit_epoch_end(self, trainer: Any) -> None:
        if self._closed:
            raise RuntimeError("Epoch logger received a callback after close.")
        callback_epoch = int(trainer.epoch) + 1
        if self._active_epoch is None:
            expected_final_eval_epoch = (
                None if self._last_completed_epoch is None else self._last_completed_epoch + 1
            )
            if (
                not self._final_eval_callback_consumed
                and callback_epoch == expected_final_eval_epoch
            ):
                self._final_eval_callback_consumed = True
                return
            raise RuntimeError("Fit-end callback does not match an active or final-eval boundary.")
        loss_metrics = trainer.label_loss_items(trainer.tloss, prefix="train")
        metrics = {**loss_metrics, **dict(trainer.metrics), **dict(trainer.lr)}
        reserved: int | None = None
        device = getattr(trainer, "device", None)
        if torch.cuda.is_available() and getattr(device, "type", None) == "cuda":
            reserved = int(torch.cuda.memory_reserved(device))
        self.finish_epoch(
            epoch=callback_epoch,
            metrics=metrics,
            cuda_memory_reserved_bytes=reserved,
        )
