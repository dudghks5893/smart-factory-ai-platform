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


class FakeTrainer:
    """Minimal Ultralytics-shaped trainer callback fixture."""

    # ADD 2026-08-27: Callback epoch/scalar/device contract를 재사용 가능한 fixture로 만든다.
    def __init__(self, *, epoch: int = 0) -> None:
        self.epoch = epoch
        self.tloss = torch.tensor([0.4, 0.5, 0.6])
        self.metrics = {"metrics/mAP50-95(M)": 0.3}
        self.lr = {"lr/pg0": 0.001}
        self.device = torch.device("cpu")

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


# ADD 2026-08-27: Cleanup을 검증한다. → MODIFY 2026-08-28: Incomplete JSONL 부재를 확인한다.
def test_epoch_metrics_logger_cleanup_discards_active_epoch(tmp_path: Path) -> None:
    output = tmp_path / "epoch_metrics.jsonl"
    logger = EpochMetricsLogger(
        output_path=output,
        total_epochs=1,
        clock=FakeClock([1.0]),
    )
    logger.start_epoch(1)
    logger.close()
    with pytest.raises(RuntimeError, match="active"):
        logger.finish_epoch(epoch=1, metrics={})
    assert not output.exists()


# ADD 2026-08-27: Callback을 검증한다. → MODIFY 2026-08-28: Shared trainer fixture를 쓴다.
def test_epoch_metrics_logger_callback_adapter(tmp_path: Path) -> None:
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


# ADD 2026-08-27: Expected final_eval은 한 번만 무시하고 duplicate record를 막는지 검증한다.
def test_epoch_metrics_logger_ignores_final_eval_without_active_epoch(tmp_path: Path) -> None:
    output = tmp_path / "final-eval.jsonl"
    logger = EpochMetricsLogger(
        output_path=output,
        total_epochs=1,
        clock=FakeClock([2.0, 5.0]),
        console_writer=lambda _: None,
    )
    trainer = FakeTrainer()
    logger.on_train_epoch_start(trainer)
    logger.on_fit_epoch_end(trainer)

    # Ultralytics final_eval은 epoch를 임시 증가시킨 뒤 fit-end callback만 재호출한다.
    trainer.epoch += 1
    logger.on_fit_epoch_end(trainer)
    with pytest.raises(RuntimeError, match="active or final-eval boundary"):
        logger.on_fit_epoch_end(trainer)

    rows = output.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["epoch"] == 1


# ADD 2026-08-27: Completed epoch가 없는 start-less fit-end callback을 거부한다.
def test_epoch_metrics_logger_rejects_fit_end_without_start(tmp_path: Path) -> None:
    logger = EpochMetricsLogger(
        output_path=tmp_path / "start-less.jsonl",
        total_epochs=1,
        clock=FakeClock([]),
    )
    with pytest.raises(RuntimeError, match="active or final-eval boundary"):
        logger.on_fit_epoch_end(FakeTrainer())


# ADD 2026-08-27: Completed epoch와 같은 번호의 duplicate fit-end callback을 거부한다.
def test_epoch_metrics_logger_rejects_completed_epoch_duplicate(tmp_path: Path) -> None:
    logger = EpochMetricsLogger(
        output_path=tmp_path / "completed-duplicate.jsonl",
        total_epochs=1,
        clock=FakeClock([1.0, 2.0]),
        console_writer=lambda _: None,
    )
    trainer = FakeTrainer()
    logger.on_train_epoch_start(trainer)
    logger.on_fit_epoch_end(trainer)
    with pytest.raises(RuntimeError, match="active or final-eval boundary"):
        logger.on_fit_epoch_end(trainer)


# ADD 2026-08-27: Active timing과 다른 epoch의 fit-end callback은 계속 fail-fast하는지 검증한다.
def test_epoch_metrics_logger_rejects_mismatched_active_epoch(tmp_path: Path) -> None:
    logger = EpochMetricsLogger(
        output_path=tmp_path / "mismatch.jsonl",
        total_epochs=2,
        clock=FakeClock([1.0]),
    )
    trainer = FakeTrainer()
    logger.on_train_epoch_start(trainer)
    trainer.epoch = 1
    with pytest.raises(RuntimeError, match="active timing boundary"):
        logger.on_fit_epoch_end(trainer)


# ADD 2026-08-27: Completed JSONL은 보존하고 close가 partial epoch/final callback을 막는지 검증한다.
def test_epoch_metrics_logger_close_preserves_completed_rows(tmp_path: Path) -> None:
    output = tmp_path / "partial.jsonl"
    logger = EpochMetricsLogger(
        output_path=output,
        total_epochs=2,
        clock=FakeClock([1.0, 2.0, 3.0]),
        console_writer=lambda _: None,
    )
    trainer = FakeTrainer()
    logger.on_train_epoch_start(trainer)
    logger.on_fit_epoch_end(trainer)
    trainer.epoch = 1
    logger.on_train_epoch_start(trainer)
    logger.close()

    with pytest.raises(RuntimeError, match="after close"):
        logger.on_fit_epoch_end(trainer)
    assert len(output.read_text(encoding="utf-8").splitlines()) == 1
