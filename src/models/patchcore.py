"""PatchCore, implemented directly (no anomalib) so timing and footprint stay under our control."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import timm
import torch
import torch.nn.functional as F
from torch import Tensor

from src.models.base import gaussian_smooth, module_size_mb
from src.models.train_utils import loader

log = logging.getLogger(__name__)


class PatchCoreModel:
    name = "patchcore"

    def __init__(self, device: torch.device, backbone: str = "resnet50", layers=(2, 3),
                 neighbourhood: int = 3, coreset_rate: float = 0.01, projection_dim: int = 128,
                 map_sigma: float = 4.0, image_size: int = 384, pretrained: bool = True,
                 seed: int = 0, meta: dict | None = None) -> None:
        self.device = device
        self.backbone_name = backbone
        self.layers = tuple(int(l) for l in layers)
        self.neighbourhood = int(neighbourhood)
        self.coreset_rate = float(coreset_rate)
        self.projection_dim = int(projection_dim)
        self.map_sigma = float(map_sigma)
        self.image_size = int(image_size)
        self.seed = int(seed)
        self.meta = meta or {}
        self.backbone = timm.create_model(backbone, pretrained=pretrained, features_only=True,
                                          out_indices=self.layers).to(device).eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.bank: Tensor | None = None  # (K, C), float32 on device for search
        self.grid: tuple[int, int] | None = None

    # ------------------------------------------------------------------ features
    @torch.no_grad()
    def _embed(self, x: Tensor) -> Tensor:
        """(B, 3, 384, 384) -> (B, P, C) locally aware patch descriptors, P = 24 * 24."""
        feats = self.backbone(x.to(self.device))
        # Plan §8.6: common grid of 24 x 24 (576 patches), i.e. the resolution of the
        # deeper layer; the shallower map is bilinearly resized onto it.
        target = feats[-1].shape[-2:]
        feats = [f if f.shape[-2:] == target else
                 F.interpolate(f, size=target, mode="bilinear", align_corners=False) for f in feats]
        emb = torch.cat(feats, dim=1)
        k = self.neighbourhood
        emb = F.avg_pool2d(emb, kernel_size=k, stride=1, padding=k // 2)
        b, c, h, w = emb.shape
        self.grid = (h, w)
        return emb.permute(0, 2, 3, 1).reshape(b, h * w, c)

    # ------------------------------------------------------------------ memory bank
    def fit(self, train_ds, val_ds, cfg) -> dict:
        """Single forward pass over defect-free images, then greedy k-center coreset."""
        t0 = time.perf_counter()
        dl = loader(train_ds, int(cfg.train.batch_size), int(cfg.runtime.num_workers))
        gen = torch.Generator().manual_seed(self.seed)
        proj = None
        full, projected = [], []
        for x, _, _, _ in dl:
            emb = self._embed(x)  # (B, P, C)
            emb = emb.reshape(-1, emb.shape[-1])
            if proj is None:
                # Johnson-Lindenstrauss random projection, used only for coreset selection.
                proj = torch.randn(emb.shape[-1], self.projection_dim, generator=gen)
                proj = (proj / self.projection_dim**0.5).to(self.device)
            projected.append((emb @ proj).float())
            full.append(emb.to(torch.float16).cpu())
        full_t = torch.cat(full)
        proj_t = torch.cat(projected)
        t_extract = time.perf_counter() - t0
        n_select = max(1, int(round(len(full_t) * self.coreset_rate)))
        idx = greedy_kcenter(proj_t, n_select, gen)
        self.bank = full_t[idx.cpu()].to(self.device).float()  # unprojected descriptors
        del full, projected, proj_t
        seconds = time.perf_counter() - t0
        info = {"construction_seconds": seconds, "feature_seconds": t_extract,
                "n_patches": int(len(full_t)), "bank_size": int(n_select),
                "descriptor_dim": int(full_t.shape[1]), "bank_mb_fp16": self.bank_mb()}
        log.info("patchcore bank %d/%d patches (%.2f MiB fp16) built in %.1fs",
                 n_select, len(full_t), info["bank_mb_fp16"], seconds)
        return {"history": [], "construction": info}

    def bank_mb(self) -> float:
        return 0.0 if self.bank is None else self.bank.numel() * 2 / 2**20

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def _patch_distances(self, x: Tensor) -> Tensor:
        if self.bank is None:
            raise RuntimeError("memory bank is empty - fit() or load() first")
        emb = self._embed(x).float()  # (B, P, C)
        d = torch.cdist(emb, self.bank.unsqueeze(0).expand(emb.shape[0], -1, -1))
        return d.min(dim=2).values  # (B, P) nearest-neighbour distance (n_neighbours = 1)

    def _to_map(self, dist: Tensor) -> Tensor:
        h, w = self.grid
        m = dist.view(-1, 1, h, w)
        m = F.interpolate(m, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return gaussian_smooth(m.squeeze(1), self.map_sigma)

    def score(self, x: Tensor) -> Tensor:
        return self._patch_distances(x).max(dim=1).values

    def anomaly_map(self, x: Tensor) -> Tensor:
        return self._to_map(self._patch_distances(x))

    def predict(self, x: Tensor) -> tuple[Tensor, Tensor]:
        d = self._patch_distances(x)
        return d.max(dim=1).values, self._to_map(d)

    def size_mb(self) -> float:
        """Frozen backbone (fp32) plus the fp16 memory bank as stored on disk."""
        return module_size_mb(self.backbone) + self.bank_mb()

    # ------------------------------------------------------------------ persistence
    def save(self, path: Path) -> None:
        # The backbone is stored too, so loading never needs a network download.
        torch.save({
            "model": self.name, "meta": self.meta, "bank": self.bank.half().cpu(), "grid": self.grid,
            "backbone": self.backbone_name, "layers": self.layers, "neighbourhood": self.neighbourhood,
            "coreset_rate": self.coreset_rate, "projection_dim": self.projection_dim,
            "map_sigma": self.map_sigma, "image_size": self.image_size, "seed": self.seed,
            "backbone_state": {k: v.cpu() for k, v in self.backbone.state_dict().items()},
        }, path)

    @classmethod
    def load(cls, path: Path, cfg=None, device: torch.device | None = None) -> "PatchCoreModel":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        device = device or torch.device("cpu")
        obj = cls(device, ckpt["backbone"], ckpt["layers"], ckpt["neighbourhood"], ckpt["coreset_rate"],
                  ckpt["projection_dim"], ckpt["map_sigma"], ckpt["image_size"], pretrained=False,
                  seed=ckpt["seed"], meta=ckpt.get("meta"))
        obj.backbone.load_state_dict(ckpt["backbone_state"])
        obj.bank = ckpt["bank"].to(device).float()
        obj.grid = tuple(ckpt["grid"])
        return obj


@torch.no_grad()
def greedy_kcenter(points: Tensor, n: int, generator: torch.Generator) -> Tensor:
    """Greedy k-center selection: repeatedly add the point farthest from the chosen set."""
    n_points = points.shape[0]
    n = min(n, n_points)
    start = int(torch.randint(n_points, (1,), generator=generator))
    selected = torch.empty(n, dtype=torch.long, device=points.device)
    selected[0] = start
    min_d = ((points - points[start]) ** 2).sum(1)
    for i in range(1, n):
        nxt = torch.argmax(min_d)
        selected[i] = nxt
        min_d = torch.minimum(min_d, ((points - points[nxt]) ** 2).sum(1))
    return selected
