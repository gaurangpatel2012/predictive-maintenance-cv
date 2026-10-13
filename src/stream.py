"""Sliding-window defect-rate aggregation and the window-length sweep (Exp 5)."""
from __future__ import annotations

from collections import deque
from typing import Sequence

import numpy as np


class DefectRateMonitor:
    """Running defect rate over the last ``window`` decisions.

    The indication is raised when the window is full and the rate strictly
    exceeds ``rate_threshold``; it clears when the rate falls back to or below it.
    """

    def __init__(self, window: int, rate_threshold: float) -> None:
        self.window = int(window)
        self.rate_threshold = float(rate_threshold)
        self.buffer: deque[int] = deque(maxlen=self.window)
        self._sum = 0

    def update(self, decision: int) -> tuple[float, bool]:
        if len(self.buffer) == self.window:
            self._sum -= self.buffer[0]
        self.buffer.append(int(decision))
        self._sum += int(decision)
        rate = self._sum / len(self.buffer)
        return rate, len(self.buffer) == self.window and rate > self.rate_threshold

    @property
    def rate(self) -> float:
        return self._sum / len(self.buffer) if self.buffer else 0.0


def step_change_sequence(pred_normal: Sequence[int], pred_anomalous: Sequence[int], length: int,
                         change_index: int, p_before: float, p_after: float,
                         rng: np.random.Generator) -> np.ndarray:
    """Decisions for a synthetic line whose defect prevalence steps up at ``change_index``.

    Each position draws a true label from the prevailing prevalence, then the
    model's real decision on a randomly chosen evaluation image of that label, so
    the model's measured false-positive and miss rates propagate into the signal.
    """
    prevalence = np.where(np.arange(length) < change_index, p_before, p_after)
    labels = rng.random(length) < prevalence
    pn, pa = np.asarray(pred_normal, int), np.asarray(pred_anomalous, int)
    out = np.where(labels, pa[rng.integers(0, len(pa), length)], pn[rng.integers(0, len(pn), length)])
    return out.astype(int)


def evaluate_monitor(decisions: np.ndarray, window: int, rate_threshold: float,
                     change_index: int) -> dict[str, float]:
    mon = DefectRateMonitor(window, rate_threshold)
    active = np.zeros(len(decisions), dtype=bool)
    for i, d in enumerate(decisions):
        active[i] = mon.update(int(d))[1]
    pre = active[window - 1 : change_index]  # full windows entirely before the change
    fired = np.flatnonzero(active[change_index:])
    rising = np.flatnonzero(active & ~np.r_[False, active[:-1]])
    return {
        "detection_delay_images": float(fired[0]) if fired.size else float("nan"),
        "false_alarm_rate": float(pre.mean()) if pre.size else float("nan"),
        "n_indications": float(rising.size),
    }


def window_sweep(pred_normal: Sequence[int], pred_anomalous: Sequence[int], windows: Sequence[int],
                 thresholds: Sequence[float], length: int, change_index: int, p_before: float,
                 p_after: float, trials: int, seed: int) -> list[dict]:
    """Mean over ``trials`` synthetic sequences for every (window, threshold) pair.

    Detection delay is averaged over the trials in which the indication fired
    after the change; ``n_trials_detected`` says how many that was.
    """
    rng = np.random.default_rng(seed)
    seqs = [step_change_sequence(pred_normal, pred_anomalous, length, change_index, p_before, p_after, rng)
            for _ in range(trials)]
    rows = []
    for w in windows:
        for thr in thresholds:
            res = [evaluate_monitor(s, int(w), float(thr), change_index) for s in seqs]
            delays = np.array([r["detection_delay_images"] for r in res])
            detected = np.isfinite(delays)
            rows.append({
                "window_W": int(w), "rate_threshold": float(thr),
                "detection_delay_images": float(delays[detected].mean()) if detected.any() else float("nan"),
                "false_alarm_rate": float(np.nanmean([r["false_alarm_rate"] for r in res])),
                "n_indications": float(np.mean([r["n_indications"] for r in res])),
                "n_trials_detected": int(detected.sum()), "n_trials": int(trials),
            })
    return rows
