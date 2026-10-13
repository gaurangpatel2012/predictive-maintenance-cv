"""Training-loop pieces shared by the classifiers and the autoencoder."""
from __future__ import annotations

import contextlib
import math
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from src.seeding import worker_init_fn


def warmup_cosine(total_steps: int, warmup_steps: int):
    """LambdaLR factor: linear warm-up, then cosine decay to zero."""

    def factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return factor


def make_optimiser(params, lr: float, weight_decay: float, epochs: int, warmup_epochs: int, steps_per_epoch: int):
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, warmup_cosine(epochs * steps_per_epoch, warmup_epochs * steps_per_epoch)
    )
    return opt, sched


def autocast(device: torch.device, enabled: bool):
    """Mixed precision on CUDA only; MPS/CPU autocast is not used so results stay comparable."""
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def loader(ds: Dataset, batch_size: int, num_workers: int, shuffle: bool = False,
           sampler: Sampler | None = None, seed: int = 0) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle and sampler is None, sampler=sampler,
        num_workers=num_workers, worker_init_fn=worker_init_fn, generator=g,
        persistent_workers=num_workers > 0, drop_last=False,
    )


class EarlyStopping:
    """Keeps the best weights; higher ``primary`` is better, ties broken by lower ``tiebreak``."""

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best: tuple[float, float] | None = None
        self.best_epoch = -1
        self.best_state: dict | None = None
        self.bad_epochs = 0

    def step(self, epoch: int, primary: float, tiebreak: float, model: nn.Module) -> bool:
        """Returns True when training should stop."""
        key = (primary, -tiebreak)
        if self.best is None or key > self.best:
            self.best = key
            self.best_epoch = epoch
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def restore(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


class EpochTimer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0
        return False


def nan_to_none(v: float) -> float | None:
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else v
