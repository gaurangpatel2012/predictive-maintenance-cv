from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def at_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    """Precision, recall, F1 and error rates when predicting anomalous iff score >= threshold."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    pred = scores >= threshold
    tp = int(np.sum(pred & (labels == 1)))
    fp = int(np.sum(pred & (labels == 0)))
    fn = int(np.sum(~pred & (labels == 1)))
    tn = int(np.sum(~pred & (labels == 0)))
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "fpr": fp / (fp + tn) if fp + tn else float("nan"),
        "fnr": fn / (fn + tp) if fn + tp else float("nan"),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, int)
    if len(np.unique(labels)) < 2:
        return float("nan")  # undefined with a single class
    return float(roc_auc_score(labels, np.asarray(scores, float)))
