"""Threshold selection, threshold drift and calibration error."""
from __future__ import annotations

import numpy as np


def select_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """F1-maximising threshold (predict anomalous iff score >= t). Returns (threshold, f1).

    Candidates are the observed scores; ties in F1 go to the highest threshold,
    i.e. the most conservative one. Returns (nan, nan) if there are no anomalies.
    """
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    n_pos = int(labels.sum())
    if n_pos == 0 or len(scores) == 0:
        return float("nan"), float("nan")
    order = np.argsort(-scores, kind="mergesort")
    s, y = scores[order], labels[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    # Only the last index of each run of equal scores is a valid cut.
    last = np.r_[s[1:] != s[:-1], True]
    tp, fp, s = tp[last], fp[last], s[last]
    f1 = 2 * tp / (tp + fp + n_pos)
    best = np.flatnonzero(f1 == f1.max())[0]  # first = highest threshold among ties
    return float(s[best]), float(f1[best])


def threshold_drift(optimal: float, fixed: float, nominal_sd: float) -> tuple[float, float]:
    """(absolute, normalised by the nominal score standard deviation)."""
    drift = optimal - fixed
    norm = drift / nominal_sd if nominal_sd and np.isfinite(nominal_sd) and nominal_sd > 0 else float("nan")
    return float(drift), float(norm)


def minmax_normalise(scores: np.ndarray, p01: float, p99: float) -> np.ndarray:
    """Map unbounded scores (CAE, PatchCore) to [0, 1] with nominal-validation percentiles 1 and 99."""
    span = p99 - p01 if p99 > p01 else 1.0
    return np.clip((np.asarray(scores, float) - p01) / span, 0.0, 1.0)


def ece(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error, top-label form (Guo et al. 2017), equal-width bins.

    ``probs`` is P(anomalous); the predicted class is probs >= 0.5 and its
    confidence max(p, 1 - p).
    """
    p, y = np.asarray(probs, float), np.asarray(labels, int)
    if len(p) == 0:
        return float("nan")
    pred = (p >= 0.5).astype(int)
    conf = np.where(pred == 1, p, 1 - p)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.5, 1.0, bins + 1)  # top-label confidence lies in [0.5, 1]
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)
