"""Tests for compact project-owned YOLO epoch evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from ml.experiments.yolo_epoch_logging import EpochMetricsLogger


class FakeClock:
    """Deterministic monotonic clock fixture."""

    # ADD 2026-08-27: Test-provided timestamps를 순서대로 반환한다.
    def __init__(self, values: list[float]) -> None:
        self._values = iter(values)

    # ADD 2026-08-27: Real sleep 없이 timing boundary를 전진시킨다.
    def __call__(self) -> float:
        return next(self._values)


# ADD 2026-08-27: Duration/cumulative/optional scalar와 stable JSONL fields를 검증한다.
def test_epoch_metrics_logger_serializes_compact_evidence(tmp_path: Path) -> None:
    output = tmp_path / "epoch_metrics.jsonl"
    messages: list[str] = []
    logger = EpochMetricsLogger(
        output_path=output,
        total_epochs=2,
        clock=FakeClock([10.0, 12.5, 20.0, 24.0]),
        console_writer=messages.append,
    )
    logger.start_epoch(1)
    first = logger.finish_epoch(
        epoch=1,
        metrics={"train/seg_loss": 0.5, "metrics/mAP50-95(M)": 0.25},
    )
    logger.start_epoch(2)
    second = logger.finish_epoch(epoch=2, metrics={"train/box_loss": 0.2})
    logger.close()

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert first.epoch_time_seconds == 2.5
    assert second.cumulative_epoch_seconds == 6.5
    assert rows[0]["train_seg_loss"] == 0.5
    assert rows[0]["train_dfl_loss"] is None
    assert rows[1]["val_mask_map50_95"] is None
    assert messages[0].startswith("[C4 EXPERIMENT] Epoch 001/2")


# ADD 2026-08-27: Exception cleanup 뒤 incomplete epoch를 완료로 기록하지 않는지 검증한다.
def test_epoch_metrics_logger_cleanup_discards_active_epoch(tmp_path: Path) -> None:
    logger = EpochMetricsLogger(
        output_path=tmp_path / "epoch_metrics.jsonl",
        total_epochs=1,
        clock=FakeClock([1.0]),
    )
    logger.start_epoch(1)
    logger.close()
    with pytest.raises(RuntimeError, match="active"):
        logger.finish_epoch(epoch=1, metrics={})


# ADD 2026-08-27: Ultralytics-shaped callback event에서 loss/validation/LR를 추출한다.
def test_epoch_metrics_logger_callback_adapter(tmp_path: Path) -> None:
    class FakeTrainer:
        epoch = 0
        tloss = torch.tensor([0.4, 0.5, 0.6])
        metrics = {"metrics/mAP50-95(M)": 0.3}
        lr = {"lr/pg0": 0.001}
        device = torch.device("cpu")

        # ADD 2026-08-27: Pinned trainer의 loss-label mapping shape를 모사한다.
        def label_loss_items(
            self,
            loss_items: torch.Tensor,
            *,
            prefix: str,
        ) -> dict[str, float]:
            assert prefix == "train"
            return {
                "train/box_loss": float(loss_items[0]),
                "train/seg_loss": float(loss_items[1]),
                "train/cls_loss": float(loss_items[2]),
            }

    output = tmp_path / "callback.jsonl"
    logger = EpochMetricsLogger(
        output_path=output,
        total_epochs=1,
        clock=FakeClock([2.0, 5.0]),
        console_writer=lambda _: None,
    )
    trainer = FakeTrainer()
    logger.on_train_epoch_start(trainer)
    logger.on_fit_epoch_end(trainer)
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["epoch_time_seconds"] == 3.0
    assert row["train_seg_loss"] == pytest.approx(0.5)
    assert row["val_mask_map50_95"] == 0.3
    assert row["learning_rate_pg0"] == 0.001
