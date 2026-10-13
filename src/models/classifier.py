"""Supervised two-class classifiers: resnet50 | efficientnet_b0 | vit."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import WeightedRandomSampler

from src.models.base import module_size_mb
from src.models.train_utils import (
    EarlyStopping, EpochTimer, autocast, loader, make_optimiser,
)

log = logging.getLogger(__name__)

TIMM_NAMES = {
    "resnet50": "resnet50",
    "efficientnet_b0": "efficientnet_b0",
    "vit": "vit_base_patch16_384",  # native 384 input, matches the working resolution
}


class ClassifierModel:
    def __init__(self, name: str, device: torch.device, pretrained: bool = True,
                 meta: dict | None = None) -> None:
        self.name = name
        self.device = device
        self.meta = meta or {}
        self.net = timm.create_model(TIMM_NAMES[name], pretrained=pretrained, num_classes=2).to(device)

    # ------------------------------------------------------------------ training
    def fit(self, train_ds, val_ds, cfg) -> dict:
        from src.metrics.threshold import select_threshold  # training-only dependency

        tc = cfg.train
        nw = int(cfg.runtime.num_workers)
        labels = train_ds.labels
        counts = np.bincount(labels, minlength=2).astype(float)
        if (counts == 0).any():
            raise RuntimeError(f"training set lacks a class: counts={counts.tolist()}")
        weights = torch.as_tensor(1.0 / counts[labels], dtype=torch.double)
        g = torch.Generator().manual_seed(int(torch.initial_seed() % 2**31))
        sampler = WeightedRandomSampler(weights, num_samples=len(labels), replacement=True, generator=g)
        train_dl = loader(train_ds, int(tc.batch_size), nw, sampler=sampler)
        val_dl = loader(val_ds, int(tc.batch_size), nw)

        epochs = int(tc.max_epochs)
        # Gradient accumulation keeps the effective batch at batch_size * accumulation_steps
        # when memory forces smaller micro-batches (README, decision D15).
        accum = max(1, int(tc.get("accumulation_steps", 1)))
        steps_per_epoch = -(-len(train_dl) // accum)
        opt, sched = make_optimiser(
            self.net.parameters(), float(tc.lr), float(tc.weight_decay),
            epochs, int(tc.warmup_epochs), steps_per_epoch,
        )
        use_amp = bool(tc.amp) and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        stopper = EarlyStopping(int(tc.early_stopping_patience))
        history = []
        for epoch in range(epochs):
            with EpochTimer() as timer:
                self.net.train()
                total, n = 0.0, 0
                opt.zero_grad(set_to_none=True)
                n_batches = len(train_dl)
                for b, (x, y, _, _) in enumerate(train_dl):
                    x, y = x.to(self.device), torch.as_tensor(y, device=self.device)
                    with autocast(self.device, use_amp):
                        loss = F.cross_entropy(self.net(x), y)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"non-finite training loss at epoch {epoch}")
                    group = min(accum, n_batches - (b // accum) * accum)  # last group may be short
                    scaler.scale(loss / group).backward()
                    if (b + 1) % accum == 0 or b + 1 == n_batches:
                        scaler.step(opt)
                        scaler.update()
                        sched.step()
                        opt.zero_grad(set_to_none=True)
                    total += loss.item() * len(y)
                    n += len(y)
                val_loss, scores, ys = self._evaluate(val_dl)
                val_f1 = select_threshold(scores, ys)[1]
            lr = opt.param_groups[0]["lr"]
            history.append({
                "epoch": epoch, "train_loss": total / max(n, 1), "val_loss": val_loss,
                "val_f1": val_f1, "lr": lr, "epoch_seconds": timer.seconds,
            })
            log.info("%s epoch %d train_loss=%.4f val_loss=%.4f val_f1=%.4f (%.1fs)",
                     self.name, epoch, total / max(n, 1), val_loss, val_f1, timer.seconds)
            if stopper.step(epoch, val_f1, val_loss, self.net):
                log.info("early stopping at epoch %d (best %d)", epoch, stopper.best_epoch)
                break
        stopper.restore(self.net)
        return {"history": history, "best_epoch": stopper.best_epoch,
                "epochs_run": len(history), "stopped_early": len(history) < epochs}

    @torch.no_grad()
    def _evaluate(self, dl) -> tuple[float, np.ndarray, np.ndarray]:
        self.net.eval()
        losses, scores, ys = [], [], []
        for x, y, _, _ in dl:
            logits = self.net(x.to(self.device)).float()
            y_t = torch.as_tensor(y, device=self.device)
            losses.append(F.cross_entropy(logits, y_t, reduction="sum").item())
            scores.append(logits.softmax(1)[:, 1].cpu().numpy())
            ys.append(np.asarray(y))
        n = sum(len(a) for a in ys)
        return sum(losses) / max(n, 1), np.concatenate(scores), np.concatenate(ys)

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def score(self, x: Tensor) -> Tensor:
        self.net.eval()
        return self.net(x.to(self.device)).float().softmax(1)[:, 1]

    def anomaly_map(self, x: Tensor) -> None:
        return None  # image-level only; no CAM substitute (stated finding)

    def predict(self, x: Tensor) -> tuple[Tensor, None]:
        return self.score(x), None

    def size_mb(self) -> float:
        return module_size_mb(self.net)

    # ------------------------------------------------------------------ persistence
    def save(self, path: Path) -> None:
        torch.save({"model": self.name, "meta": self.meta,
                    "state_dict": {k: v.cpu() for k, v in self.net.state_dict().items()}}, path)

    @classmethod
    def load(cls, path: Path, cfg=None, device: torch.device | None = None) -> "ClassifierModel":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        device = device or torch.device("cpu")
        obj = cls(ckpt["model"], device, pretrained=False, meta=ckpt.get("meta"))
        obj.net.load_state_dict(ckpt["state_dict"])
        obj.net.eval()
        return obj
