"""Latency statistics. Pure numpy so the deployed app can import it."""
from __future__ import annotations

import numpy as np


def summarise(latencies_ms, budget_ms: float = 1000.0, wall_seconds: float | None = None) -> dict[str, float]:
    t = np.asarray(latencies_ms, float)
    if t.size == 0:
        return {k: float("nan") for k in
                ("median_ms", "p95_ms", "p99_ms", "mean_ms", "std_ms", "deadline_miss_rate", "throughput_ips")}
    return {
        "median_ms": float(np.median(t)),
        "p95_ms": float(np.percentile(t, 95)),
        "p99_ms": float(np.percentile(t, 99)),
        "mean_ms": float(t.mean()),
        "std_ms": float(t.std(ddof=1)) if t.size > 1 else 0.0,
        "deadline_miss_rate": float(np.mean(t > budget_ms)),
        # sequential single-image throughput: images completed per second of wall time
        "throughput_ips": float(t.size / wall_seconds) if wall_seconds else float(1000.0 / t.mean()),
    }
