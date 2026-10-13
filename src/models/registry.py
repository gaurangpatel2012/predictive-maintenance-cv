"""build_model / load_model by name, plus artefact locations."""
from __future__ import annotations

import json
from pathlib import Path

import torch

MODEL_NAMES = ("resnet50", "efficientnet_b0", "vit", "cae", "patchcore")
CLASSIFIERS = ("resnet50", "efficientnet_b0", "vit")
LOCALISING = ("cae", "patchcore")
DISPLAY_NAMES = {
    "resnet50": "ResNet-50", "efficientnet_b0": "EfficientNet-B0", "vit": "Vision Transformer",
    "cae": "Convolutional Autoencoder", "patchcore": "PatchCore",
}


def build_model(name: str, cfg, device: torch.device, seed: int = 0):
    from src.models.base import preprocess_meta

    meta = preprocess_meta(cfg)
    pretrained = bool(cfg.runtime.pretrained)
    if name in CLASSIFIERS:
        from src.models.classifier import ClassifierModel

        return ClassifierModel(name, device, pretrained=pretrained, meta=meta)
    if name == "cae":
        from src.models.cae import CAEModel

        c = cfg.cae
        return CAEModel(device, int(c.latent_channels), cfg.normalise_mean, cfg.normalise_std,
                        float(c.map_sigma), meta=meta, image_score=str(c.get("image_score", "mean")))
    if name == "patchcore":
        from src.models.patchcore import PatchCoreModel

        p = cfg.patchcore
        return PatchCoreModel(device, p.backbone, p.layers, p.neighbourhood, p.coreset_rate,
                              p.projection_dim, p.map_sigma, int(cfg.image_size), pretrained, seed, meta)
    raise ValueError(f"unknown model '{name}'")


def load_model(name: str, path: Path, device: torch.device):
    if name in CLASSIFIERS:
        from src.models.classifier import ClassifierModel as cls
    elif name == "cae":
        from src.models.cae import CAEModel as cls
    elif name == "patchcore":
        from src.models.patchcore import PatchCoreModel as cls
    else:
        raise ValueError(f"unknown model '{name}'")
    return cls.load(Path(path), device=device)


def artifact_dir(cfg, category: str, model: str, seed: int) -> Path:
    return cfg.path("artifacts") / category / model / f"seed{seed}"


def load_calibration(directory: Path) -> dict:
    path = Path(directory) / "calibration.json"
    if not path.exists():
        raise SystemExit(f"{path} missing - train the model with scripts/02_train.py first")
    return json.loads(path.read_text())
