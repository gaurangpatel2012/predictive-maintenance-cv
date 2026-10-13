"""AU-PRO: area under the per-region-overlap curve up to an FPR limit, normalised.

Each connected ground-truth region carries equal weight regardless of its size.
The curve is computed exactly by sorting all pixels by anomaly score once
(same construction as the MVTec reference evaluation), rather than by sampling
a fixed number of thresholds.
"""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np


def pro_curve(maps: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """maps: (N, H, W) float, masks: (N, H, W) {0,1}. Returns (fpr, pro), both increasing."""
    maps = np.asarray(maps, np.float32)
    masks = np.asarray(masks) > 0
    weights = np.zeros(maps.shape, dtype=np.float64)
    n_regions = 0
    region_weights = []
    for i in range(len(masks)):
        if not masks[i].any():
            continue
        n_lab, lab = cv2.connectedComponents(masks[i].astype(np.uint8), connectivity=8)
        for r in range(1, n_lab):
            region = lab == r
            region_weights.append((i, region, 1.0 / region.sum()))
            n_regions += 1
    if n_regions == 0:
        raise ValueError("AU-PRO needs at least one anomalous region")
    for i, region, w in region_weights:
        weights[i][region] = w / n_regions
    neg = ~masks
    n_neg = int(neg.sum())
    if n_neg == 0:
        raise ValueError("AU-PRO needs at least one normal pixel")
    weights[neg] = -1.0 / n_neg  # sign marks normal pixels; magnitude is their FPR weight

    flat_s = maps.ravel()
    flat_w = weights.ravel()
    order = np.argsort(-flat_s, kind="mergesort")
    s, w = flat_s[order], flat_w[order]
    fpr = np.cumsum(np.where(w < 0, -w, 0.0))
    pro = np.cumsum(np.where(w > 0, w, 0.0))
    keep = np.r_[s[1:] != s[:-1], True]  # one point per distinct threshold
    fpr, pro = np.r_[0.0, fpr[keep]], np.r_[0.0, pro[keep]]
    return np.clip(fpr, 0, 1), np.clip(pro, 0, 1)


def au_pro(maps: np.ndarray, masks: np.ndarray, limits: Sequence[float] = (0.05, 0.30)) -> dict[float, float]:
    fpr, pro = pro_curve(maps, masks)
    out = {}
    for limit in limits:
        # interpolate the curve at exactly the limit, then integrate and normalise
        pro_at = float(np.interp(limit, fpr, pro))
        m = fpr <= limit
        x = np.r_[fpr[m], limit]
        y = np.r_[pro[m], pro_at]
        out[float(limit)] = float(np.trapezoid(y, x) / limit)
    return out
