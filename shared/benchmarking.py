"""Reusable finite latency distribution calculations for benchmark domains."""

import math
from dataclasses import asdict, dataclass

import torch

LINEAR_PERCENTILE_METHOD = "linear interpolation between adjacent ordered observations"


@dataclass(frozen=True)
class LatencyDistribution:
    """Positive finite latency percentiles, mean, and summed duration."""

    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    total_timed_seconds: float

    # ADD 2026-08-20: Latency distribution을 stable JSON mapping으로 변환한다.
    def to_json_dict(self) -> dict[str, float]:
        return asdict(self)


# ADD 2026-08-20: Linear percentile와 mean/total을 positive finite latency에서 계산한다.
def summarize_latency_distribution(
    latencies_ms: tuple[float, ...] | list[float],
) -> LatencyDistribution:
    """Summarize latencies with linear interpolation on a CPU float64 tensor."""
    if not latencies_ms:
        raise ValueError("At least one measured latency is required.")
    if any(not math.isfinite(value) or value <= 0.0 for value in latencies_ms):
        raise ValueError("Latency values must be finite and positive.")

    values = torch.tensor(latencies_ms, dtype=torch.float64)
    percentiles = torch.quantile(
        values,
        torch.tensor([0.50, 0.95, 0.99], dtype=torch.float64),
        interpolation="linear",
    )
    return LatencyDistribution(
        p50_ms=float(percentiles[0].item()),
        p95_ms=float(percentiles[1].item()),
        p99_ms=float(percentiles[2].item()),
        mean_ms=float(values.mean().item()),
        total_timed_seconds=float(values.sum().item()) / 1000.0,
    )
