"""Common model interface and helpers shared by the five approaches."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class AnomalyModel(Protocol):
    name: str

    def fit(self, train_ds, val_ds, cfg) -> dict: ...  # returns training log
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path, cfg=None, device: torch.device | None = None) -> "AnomalyModel": ...
    def score(self, x: Tensor) -> Tensor: ...  # (B,) higher = more anomalous
    def anomaly_map(self, x: Tensor) -> Tensor | None: ...  # (B,384,384) or None
    def predict(self, x: Tensor) -> tuple[Tensor, Tensor | None]: ...  # both from one forward pass
    def size_mb(self) -> float: ...


def module_size_mb(*modules: nn.Module) -> float:
    """Parameter + buffer bytes in MiB, i.e. the in-memory size of the weights."""
    total = 0
    for m in modules:
        total += sum(p.numel() * p.element_size() for p in m.parameters())
        total += sum(b.numel() * b.element_size() for b in m.buffers())
    return total / 2**20


def gaussian_kernel1d(sigma: float, device: torch.device) -> Tensor:
    radius = max(1, math.ceil(4 * sigma))
    xs = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    k = torch.exp(-(xs**2) / (2 * sigma**2))
    return k / k.sum()


def gaussian_smooth(maps: Tensor, sigma: float) -> Tensor:
    """Separable Gaussian smoothing of (B, H, W) maps with reflect padding."""
    if sigma <= 0:
        return maps
    k = gaussian_kernel1d(sigma, maps.device)
    r = (k.numel() - 1) // 2
    x = maps.unsqueeze(1).float()
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))
    return x.squeeze(1)


def denormalise(x: Tensor, mean, std) -> Tensor:
    m = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    s = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x * s + m).clamp_(0, 1)


def preprocess_meta(cfg) -> dict:
    """Stored inside every model.pt so the app can rebuild inputs without the YAML config."""
    return {
        "image_size": int(cfg.image_size),
        "mean": list(cfg.normalise_mean),
        "std": list(cfg.normalise_std),
    }
