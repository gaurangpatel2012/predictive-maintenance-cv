"""Instrumented single-image inference path, shared by Exp 4/5, the API and the dashboard.

Timer boundaries (device synchronised before every timestamp):
  t_pre   encoded bytes -> decode + validate -> resize 384 -> normalise -> tensor on device
  t_infer model forward: score and, where available, anomaly map
  t_post  score to host, threshold decision, anomaly map to host memory
  t_total t_pre + t_infer + t_post
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from src.data.dataset import decode_rgb, resize_rgb, to_tensor
from src.runtime import synchronize


@dataclass
class PredictionResult:
    score: float
    decision: int  # 1 = anomalous
    anomaly_map: np.ndarray | None
    t_pre_ms: float
    t_infer_ms: float
    t_post_ms: float

    @property
    def t_total_ms(self) -> float:
        return self.t_pre_ms + self.t_infer_ms + self.t_post_ms


class InferencePipeline:
    def __init__(self, model, device: torch.device, threshold: float, image_size: int,
                 mean, std, return_map: bool = True) -> None:
        self.model = model
        self.device = device
        self.threshold = float(threshold)
        self.image_size = int(image_size)
        self.mean, self.std = list(mean), list(std)
        self.return_map = return_map

    @classmethod
    def from_model(cls, model, device: torch.device, calibration: dict, **kwargs) -> "InferencePipeline":
        meta = model.meta
        return cls(model, device, calibration["threshold"], meta["image_size"], meta["mean"], meta["std"], **kwargs)

    @torch.no_grad()
    def __call__(self, image_bytes: bytes) -> PredictionResult:
        sync = lambda: synchronize(self.device)  # noqa: E731
        sync()
        t0 = time.perf_counter()
        img = decode_rgb(image_bytes)
        img = resize_rgb(img, self.image_size)
        x = to_tensor(img, self.mean, self.std).unsqueeze(0).to(self.device)
        sync()
        t1 = time.perf_counter()
        score, amap = self.model.predict(x)
        sync()
        t2 = time.perf_counter()
        s = float(score[0].item())
        decision = int(s >= self.threshold)
        amap_np = amap[0].float().cpu().numpy() if (amap is not None and self.return_map) else None
        sync()
        t3 = time.perf_counter()
        return PredictionResult(s, decision, amap_np, (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3)
