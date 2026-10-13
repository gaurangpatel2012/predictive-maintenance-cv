"""Batch scoring and the per-condition metric bundle shared by the evaluation scripts."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

from src.metrics.localisation import au_pro
from src.metrics.predictive import at_threshold, auroc
from src.metrics.threshold import ece, minmax_normalise, select_threshold, threshold_drift
from src.models.train_utils import loader

log = logging.getLogger(__name__)


@dataclass
class Scored:
    scores: np.ndarray
    labels: np.ndarray
    filenames: list[str]
    maps: np.ndarray | None = None  # (N, 384, 384) float16
    masks: np.ndarray | None = None  # (N, 384, 384) uint8


@torch.no_grad()
def score_dataset(model, ds, cfg, with_maps: bool = False) -> Scored:
    dl = loader(ds, int(cfg.train.batch_size), int(cfg.runtime.num_workers))
    scores, labels, names, maps, masks = [], [], [], [], []
    for x, y, m, f in dl:
        if with_maps:
            s, a = model.predict(x)
            if a is not None:
                maps.append(a.float().cpu().numpy().astype(np.float16))
                masks.append(m.numpy().astype(np.uint8))
        else:
            s = model.score(x)
        scores.append(s.float().cpu().numpy())
        labels.append(np.asarray(y))
        names.extend(f)
    return Scored(
        np.concatenate(scores) if scores else np.zeros(0), np.concatenate(labels) if labels else np.zeros(0, int),
        names, np.concatenate(maps) if maps else None, np.concatenate(masks) if masks else None,
    )


def calibrate(scored: Scored) -> dict:
    """Fixed operating point from the nominal calibration set (re-split val, nominal lighting)."""
    thr, f1 = select_threshold(scored.scores, scored.labels)
    if not np.isfinite(thr):
        raise RuntimeError("calibration set has no anomalous images - cannot select a threshold")
    return {
        "threshold": thr, "calib_f1": f1, "calib_auroc": auroc(scored.scores, scored.labels),
        "nominal_score_sd": float(np.std(scored.scores, ddof=1)) if len(scored.scores) > 1 else float("nan"),
        "score_p01": float(np.percentile(scored.scores, 1)),
        "score_p99": float(np.percentile(scored.scores, 99)),
        "n_calib": int(len(scored.scores)), "n_calib_anomalous": int(scored.labels.sum()),
    }


def probabilities(scores: np.ndarray, model_name: str, calibration: dict) -> np.ndarray:
    """Scores as probabilities for ECE. Classifiers already emit P(anomalous); CAE and
    PatchCore are min-max normalised with nominal-validation percentiles 1 and 99."""
    if model_name in ("cae", "patchcore"):
        return minmax_normalise(scores, calibration["score_p01"], calibration["score_p99"])
    return np.asarray(scores, float)


def condition_metrics(scored: Scored, model_name: str, calibration: dict, cfg) -> dict:
    fixed = float(calibration["threshold"])
    m = at_threshold(scored.scores, scored.labels, fixed)
    optimal, _ = select_threshold(scored.scores, scored.labels)
    drift, drift_norm = (threshold_drift(optimal, fixed, calibration["nominal_score_sd"])
                         if np.isfinite(optimal) else (float("nan"), float("nan")))
    limits = [float(l) for l in cfg.evaluation.aupro_fpr_limits]
    aupro = {l: float("nan") for l in limits}
    if scored.maps is not None and scored.masks is not None and scored.masks.any():
        aupro = au_pro(scored.maps.astype(np.float32), scored.masks, limits)
    return {
        "n_images": int(len(scored.scores)), "n_anomalous": int(scored.labels.sum()),
        "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
        "auroc": auroc(scored.scores, scored.labels),
        "aupro_005": aupro.get(0.05, float("nan")), "aupro_030": aupro.get(0.30, float("nan")),
        "fpr_at_fixed": m["fpr"], "fnr_at_fixed": m["fnr"],
        "threshold_fixed": fixed, "threshold_optimal": optimal,
        "threshold_drift": drift, "threshold_drift_norm": drift_norm,
        "ece": ece(probabilities(scored.scores, model_name, calibration), scored.labels,
                   int(cfg.evaluation.ece_bins)),
    }
