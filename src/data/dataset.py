"""Image loading, the fixed preprocessing order, and the torch Dataset.

Order of operations, identical for every model:
  1. load RGB  2. resize to 384 x 384 bilinear (aspect ratio not preserved)
  3. degradation (evaluation only)  4. augmentation (training only)
  5. to tensor + ImageNet normalisation.

This module deliberately avoids pandas and the YAML config loader so the
deployed app can import the preprocessing functions with its slim requirements.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.degradations import apply_degradation

if TYPE_CHECKING:
    from src.config import Config

log = logging.getLogger(__name__)


def read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read image {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def decode_rgb(data: bytes) -> np.ndarray:
    """Decode an encoded image (PNG/JPEG bytes) and validate it."""
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
    if img is None:
        raise ValueError("input is not a decodable image")
    if img.ndim != 3 or img.shape[2] != 3 or min(img.shape[:2]) < 32:
        raise ValueError(f"unexpected image shape {img.shape}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_rgb(img: np.ndarray, size: int) -> np.ndarray:
    if img.shape[0] == size and img.shape[1] == size:
        return img
    return np.asarray(Image.fromarray(img).resize((size, size), Image.BILINEAR))


def read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"cannot read mask {path}")
    return (mask > 0).astype(np.uint8)


def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    if mask.shape == (size, size):
        return mask
    return np.asarray(Image.fromarray(mask).resize((size, size), Image.NEAREST))


def to_tensor(img: np.ndarray, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div_(255.0)
    m = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    s = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    return (x - m) / s


def augment(img: np.ndarray, hflip: float, vflip: float, rotate_degrees: float) -> np.ndarray:
    """Flips and small rotations only; never blur, noise, compression or rescaling."""
    if random.random() < hflip:
        img = img[:, ::-1]
    if random.random() < vflip:
        img = img[::-1, :]
    if rotate_degrees > 0:
        angle = random.uniform(-rotate_degrees, rotate_degrees)
        h, w = img.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(
            np.ascontiguousarray(img), m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
        )
    return np.ascontiguousarray(img)


def resolve_source(cfg: "Config", category: str, relpath: str) -> tuple[Path, bool]:
    """Return (path, is_cached). Cached files are already at the working resolution."""
    if cfg.cache.enabled:
        cached = cfg.cache_dir / category / relpath
        if cached.exists():
            return cached, True
    return cfg.data_root / category / relpath, False


def load_working_image(cfg: "Config", category: str, relpath: str) -> np.ndarray:
    path, cached = resolve_source(cfg, category, relpath)
    img = read_rgb(path)
    return img if cached else resize_rgb(img, cfg.image_size)


def load_working_mask(cfg: "Config", category: str, relpath: str) -> np.ndarray:
    path, cached = resolve_source(cfg, category, relpath)
    mask = read_mask(path)
    return mask if cached else resize_mask(mask, cfg.image_size)


class InspectionDataset(Dataset):
    """Returns (tensor, label, mask, filename).

    ``mask`` is the ground-truth mask at 384 x 384 for anomalous images and an
    all-zero mask for normal images (normal images genuinely have no defect
    pixels, which is what AU-PRO needs), so batches collate without a custom
    collate function.
    """

    def __init__(
        self,
        cfg: "Config",
        category: str,
        filenames: Sequence[str],
        labels: Sequence[int],
        mask_paths: Sequence[str | None] | None = None,
        train: bool = False,
        degradation: tuple[str, float] | None = None,
    ) -> None:
        self.cfg = cfg
        self.category = category
        self.filenames = list(filenames)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.mask_paths = list(mask_paths) if mask_paths is not None else [None] * len(self.filenames)
        self.train = train
        self.degradation = degradation
        self.size = int(cfg.image_size)
        self.mean = list(cfg.normalise_mean)
        self.std = list(cfg.normalise_std)
        aug = cfg.train.augment
        self.aug = (float(aug.hflip), float(aug.vflip), float(aug.rotate_degrees))

    @classmethod
    def from_table(cls, cfg: "Config", category: str, table, **kwargs) -> "InspectionDataset":
        masks = [m if isinstance(m, str) and m else None for m in table["mask_path"].tolist()]
        return cls(cfg, category, table["filename"].tolist(), table["label"].tolist(), masks, **kwargs)

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, i: int):
        img = load_working_image(self.cfg, self.category, self.filenames[i])
        if self.degradation is not None:
            img = apply_degradation(img, *self.degradation)
        if self.train:
            img = augment(img, *self.aug)
        x = to_tensor(img, self.mean, self.std)
        mp = self.mask_paths[i]
        if mp:
            mask = torch.from_numpy(load_working_mask(self.cfg, self.category, mp).astype(np.uint8))
        else:
            mask = torch.zeros(self.size, self.size, dtype=torch.uint8)
        return x, int(self.labels[i]), mask, self.filenames[i]
