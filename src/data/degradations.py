"""Evaluation-time image degradations, applied after resize at the working resolution.

All functions take and return uint8 RGB arrays of shape (H, W, 3).
"""
from __future__ import annotations

import math

import cv2
import numpy as np

KINDS = ("gaussian_blur", "gaussian_noise", "jpeg", "downscale")


def apply_degradation(img: np.ndarray, kind: str, level: float, seed: int | None = None) -> np.ndarray:
    if kind == "gaussian_blur":
        sigma = float(level)
        k = 2 * math.ceil(3 * sigma) + 1
        return cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma)
    if kind == "gaussian_noise":
        # Noise is seeded per image content so repeated evaluations are identical.
        rng = np.random.default_rng(seed if seed is not None else _content_seed(img))
        noisy = img.astype(np.float32) + rng.normal(0.0, float(level), size=img.shape)
        return np.clip(np.rint(noisy), 0, 255).astype(np.uint8)
    if kind == "jpeg":
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(level)])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        return cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if kind == "downscale":
        h, w = img.shape[:2]
        small = (max(1, round(w / float(level))), max(1, round(h / float(level))))
        low = cv2.resize(img, small, interpolation=cv2.INTER_LINEAR)
        return cv2.resize(low, (w, h), interpolation=cv2.INTER_LINEAR)
    raise ValueError(f"unknown degradation '{kind}'")


def _content_seed(img: np.ndarray) -> int:
    return int(np.frombuffer(img[::17, ::17].tobytes(), dtype=np.uint8).astype(np.uint64).sum() % 2**32)
