"""Loading deployed models for the API and the dashboard (slim dependencies only).

Deployed layout, copied from artifacts/ by the README's export step:
  <weights_dir>/<category>/<model>/model.pt
  <weights_dir>/<category>/<model>/calibration.json

The calibration files are small and live in the repository. The weights do not
(the ViT alone is 328 MB); if a model.pt is missing it is downloaded once from
<base>/<category>__<model>.pt, where <base> is $PMCV_WEIGHTS_URL or the first line
of <weights_dir>/weights_url.txt (a GitHub release download URL).
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

import torch

from src.models.registry import MODEL_NAMES, load_model
from src.pipeline import InferencePipeline

REPO_ROOT = Path(__file__).resolve().parents[1]


def weights_dir() -> Path:
    return Path(os.environ.get("PMCV_WEIGHTS_DIR", REPO_ROOT / "app" / "weights"))


def weights_base_url(root: Path | None = None) -> str | None:
    url = os.environ.get("PMCV_WEIGHTS_URL")
    f = (root or weights_dir()) / "weights_url.txt"
    if not url and f.exists():
        url = f.read_text().strip().splitlines()[0].strip() if f.read_text().strip() else None
    return url.rstrip("/") if url else None


def ensure_weights(category: str, model: str, root: Path | None = None) -> Path:
    """Path to model.pt, downloading it from the release URL on first use."""
    root = root or weights_dir()
    dest = root / category / model / "model.pt"
    if dest.exists():
        return dest
    base = weights_base_url(root)
    if not base:
        raise FileNotFoundError(f"{dest} is missing and no weights URL is configured")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"{base}/{category}__{model}.pt", timeout=600) as r, \
            tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
        shutil.copyfileobj(r, tmp)
    Path(tmp.name).replace(dest)  # atomic: a half-downloaded file is never loaded
    return dest


def available(root: Path | None = None) -> dict[str, list[str]]:
    """{category: [model, ...]} for every deployed model (weights present or downloadable)."""
    root = root or weights_dir()
    out: dict[str, list[str]] = {}
    if not root.is_dir():
        return out
    remote = weights_base_url(root) is not None
    for cat in sorted(p for p in root.iterdir() if p.is_dir()):
        models = [m for m in MODEL_NAMES
                  if (cat / m / "calibration.json").exists() and (remote or (cat / m / "model.pt").exists())]
        if models:
            out[cat.name] = models
    return out


def load_pipeline(category: str, model: str, root: Path | None = None,
                  device: torch.device | None = None) -> InferencePipeline:
    d = (root or weights_dir()) / category / model
    device = device or torch.device("cpu")
    m = load_model(model, ensure_weights(category, model, root), device)
    calibration = json.loads((d / "calibration.json").read_text())
    return InferencePipeline.from_model(m, device, calibration)
