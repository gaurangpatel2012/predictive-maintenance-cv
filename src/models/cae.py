"""Convolutional autoencoder trained on defect-free images only."""
from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.models.base import denormalise, gaussian_smooth, module_size_mb
from src.models.train_utils import EarlyStopping, EpochTimer, autocast, loader, make_optimiser

log = logging.getLogger(__name__)

ENCODER_CHANNELS = (32, 64, 128)  # followed by latent_channels; 384 -> 192 -> 96 -> 48 -> 24


class ConvAutoencoder(nn.Module):
    """4 stride-2 down blocks and 4 stride-2 up blocks (Conv -> BatchNorm -> ReLU).

    The last up block ends in a sigmoid instead of BatchNorm/ReLU so the
    reconstruction lies in [0, 1], the space the L1 and SSIM terms are computed in.
    """

    def __init__(self, latent_channels: int) -> None:
        super().__init__()
        enc_ch = (3, *ENCODER_CHANNELS, latent_channels)
        enc = []
        for cin, cout in zip(enc_ch[:-1], enc_ch[1:]):
            enc += [nn.Conv2d(cin, cout, 4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
        self.encoder = nn.Sequential(*enc)
        dec_ch = (latent_channels, *reversed(ENCODER_CHANNELS))
        dec = []
        for cin, cout in zip(dec_ch[:-1], dec_ch[1:]):
            dec += [nn.ConvTranspose2d(cin, cout, 4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
        dec += [nn.ConvTranspose2d(dec_ch[-1], 3, 4, stride=2, padding=1), nn.Sigmoid()]
        self.decoder = nn.Sequential(*dec)

    def forward(self, x01: Tensor) -> Tensor:
        return self.decoder(self.encoder(x01))


def ssim(a: Tensor, b: Tensor, sigma: float = 1.5, c1: float = 0.01**2, c2: float = 0.03**2) -> Tensor:
    """Mean SSIM over the batch for images in [0, 1], Gaussian window (11 taps at sigma 1.5)."""
    xs = torch.arange(-5, 6, device=a.device, dtype=torch.float32)
    k1 = torch.exp(-(xs**2) / (2 * sigma**2))
    k1 = k1 / k1.sum()
    c = a.shape[1]
    kh = k1.view(1, 1, 1, -1).repeat(c, 1, 1, 1)
    kv = k1.view(1, 1, -1, 1).repeat(c, 1, 1, 1)

    def blur(t: Tensor) -> Tensor:
        return F.conv2d(F.conv2d(t, kh, groups=c), kv, groups=c)

    mu_a, mu_b = blur(a), blur(b)
    s_aa = blur(a * a) - mu_a**2
    s_bb = blur(b * b) - mu_b**2
    s_ab = blur(a * b) - mu_a * mu_b
    num = (2 * mu_a * mu_b + c1) * (2 * s_ab + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (s_aa + s_bb + c2)
    return (num / den).mean()


class CAEModel:
    name = "cae"

    def __init__(self, device: torch.device, latent_channels: int, mean, std,
                 map_sigma: float = 4.0, meta: dict | None = None, image_score: str = "mean") -> None:
        if image_score not in ("max", "mean"):
            raise ValueError(f"cae image_score must be 'max' or 'mean', got '{image_score}'")
        self.device = device
        self.image_score = image_score
        self.latent_channels = latent_channels
        self.mean, self.std = list(mean), list(std)
        self.map_sigma = map_sigma
        self.meta = meta or {}
        self.net = ConvAutoencoder(latent_channels).to(device)

    def _loss(self, x01: Tensor, recon: Tensor, w_l1: float, w_ssim: float) -> Tensor:
        return w_l1 * F.l1_loss(recon, x01) + w_ssim * (1.0 - ssim(recon, x01))

    def fit(self, train_ds, val_ds, cfg) -> dict:
        cc, tc = cfg.cae, cfg.train
        nw = int(cfg.runtime.num_workers)
        w_l1, w_ssim = float(cc.loss_weights.l1), float(cc.loss_weights.ssim)
        train_dl = loader(train_ds, int(tc.batch_size), nw, shuffle=True)
        val_dl = loader(val_ds, int(tc.batch_size), nw)
        epochs = int(cc.max_epochs)
        opt, sched = make_optimiser(self.net.parameters(), float(cc.lr), float(tc.weight_decay),
                                    epochs, int(tc.warmup_epochs), len(train_dl))
        use_amp = bool(tc.amp) and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        # Early stopping on validation reconstruction loss: official_val contains no
        # anomalies, so val_f1 is undefined for this model (README, decision D2).
        stopper = EarlyStopping(int(tc.early_stopping_patience))
        history = []
        for epoch in range(epochs):
            with EpochTimer() as timer:
                self.net.train()
                total, n = 0.0, 0
                for x, _, _, _ in train_dl:
                    x01 = denormalise(x.to(self.device), self.mean, self.std)
                    opt.zero_grad(set_to_none=True)
                    with autocast(self.device, use_amp):
                        recon = self.net(x01)
                    loss = self._loss(x01, recon.float(), w_l1, w_ssim)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"non-finite CAE loss at epoch {epoch}")
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                    sched.step()
                    total += loss.item() * len(x)
                    n += len(x)
                val_loss = self._val_loss(val_dl, w_l1, w_ssim)
            history.append({"epoch": epoch, "train_loss": total / max(n, 1), "val_loss": val_loss,
                            "val_f1": float("nan"), "lr": opt.param_groups[0]["lr"],
                            "epoch_seconds": timer.seconds})
            log.info("cae epoch %d train_loss=%.4f val_loss=%.4f (%.1fs)",
                     epoch, total / max(n, 1), val_loss, timer.seconds)
            if stopper.step(epoch, -val_loss, 0.0, self.net):
                log.info("early stopping at epoch %d (best %d)", epoch, stopper.best_epoch)
                break
        stopper.restore(self.net)
        return {"history": history, "best_epoch": stopper.best_epoch,
                "epochs_run": len(history), "stopped_early": len(history) < epochs}

    @torch.no_grad()
    def _val_loss(self, dl, w_l1: float, w_ssim: float) -> float:
        self.net.eval()
        total, n = 0.0, 0
        for x, _, _, _ in dl:
            x01 = denormalise(x.to(self.device), self.mean, self.std)
            total += self._loss(x01, self.net(x01), w_l1, w_ssim).item() * len(x)
            n += len(x)
        return total / max(n, 1)

    @torch.no_grad()
    def _error(self, x: Tensor) -> Tensor:
        self.net.eval()
        x01 = denormalise(x.to(self.device), self.mean, self.std)
        return (self.net(x01) - x01).abs().mean(1)  # (B, H, W) per-pixel L1

    def _image_score(self, err: Tensor, amap: Tensor) -> Tensor:
        # 'mean' (plan §8.5, default): mean per-pixel L1. 'max' (peak of the smoothed map)
        # was tried on sheet_metal and did no better; see README, decision D14.
        return amap.flatten(1).max(1).values if self.image_score == "max" else err.mean((1, 2))

    def score(self, x: Tensor) -> Tensor:
        return self.predict(x)[0]

    def anomaly_map(self, x: Tensor) -> Tensor:
        return gaussian_smooth(self._error(x), self.map_sigma)

    def predict(self, x: Tensor) -> tuple[Tensor, Tensor]:
        err = self._error(x)
        amap = gaussian_smooth(err, self.map_sigma)
        return self._image_score(err, amap), amap

    def size_mb(self) -> float:
        return module_size_mb(self.net)

    def save(self, path: Path) -> None:
        torch.save({"model": self.name, "meta": self.meta, "latent_channels": self.latent_channels,
                    "mean": self.mean, "std": self.std, "map_sigma": self.map_sigma,
                    "image_score": self.image_score,
                    "state_dict": {k: v.cpu() for k, v in self.net.state_dict().items()}}, path)

    @classmethod
    def load(cls, path: Path, cfg=None, device: torch.device | None = None) -> "CAEModel":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(device or torch.device("cpu"), ckpt["latent_channels"], ckpt["mean"], ckpt["std"],
                  ckpt["map_sigma"], ckpt.get("meta"), ckpt.get("image_score", "mean"))
        obj.net.load_state_dict(ckpt["state_dict"])
        obj.net.eval()
        return obj
