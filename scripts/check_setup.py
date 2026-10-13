"""Check that a copied project is complete before running the queue.

  python scripts/check_setup.py

Verifies the Python packages, the GPU, every dataset folder (image counts), the
384 px cache, the split files and the already-trained models. Prints OK or
MISSING for each item and exits non-zero if anything required is missing.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "mvtec_ad_2"
EXPECTED = {
    "sheet_metal": {"train/good": 137, "validation/good": 19, "test_public/good": 24,
                    "test_public/bad": 90, "test_public/ground_truth/bad": 90},
    "can": {"train/good": 412, "validation/good": 46, "test_public/good": 72,
            "test_public/bad": 90, "test_public/ground_truth/bad": 90},
}
problems: list[str] = []


def line(ok: bool, text: str, required: bool = True) -> None:
    print(f"  [{'OK' if ok else ('MISSING' if required else 'note')}] {text}")
    if not ok and required:
        problems.append(text)


def count_png(d: Path) -> int:
    return sum(1 for p in d.glob("*.png")) if d.is_dir() else 0


print("Python packages")
for mod in ("torch", "timm", "cv2", "numpy", "pandas", "sklearn", "yaml", "matplotlib", "psutil"):
    try:
        __import__(mod)
        line(True, mod)
    except ImportError:
        line(False, f"{mod} (run: pip install -r requirements.txt)")

print("GPU")
try:
    import torch

    if torch.cuda.is_available():
        line(True, f"CUDA GPU: {torch.cuda.get_device_name(0)} (torch {torch.__version__})")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        line(True, "Apple GPU (MPS)")
    else:
        line(False, "no GPU visible to torch - install the CUDA build of torch (see HOW_TO_RUN.md)",
             required=False)
except ImportError:
    pass

print("Dataset (data/mvtec_ad_2)")
for cat, dirs in EXPECTED.items():
    for sub, n in dirs.items():
        found = count_png(DATA / cat / sub)
        line(found == n, f"{cat}/{sub}: {found} of {n} images")

print("Cache (data/cache_384)")
for cat in EXPECTED:
    m = ROOT / "data" / "cache_384" / cat / "manifest.csv"
    line(m.exists(), f"{cat} cache manifest", required=False)

print("Splits and lighting labels (results/)")
for name in ("lighting_assignment.csv", "splits_sheet_metal.csv", "splits_can.csv"):
    line((ROOT / "results" / name).exists(), name)

print("Already-trained models (artifacts/)")
for cat in EXPECTED:
    for model in ("patchcore", "cae", "resnet50", "efficientnet_b0", "vit"):
        n = sum((ROOT / "artifacts" / cat / model / f"seed{s}" / "calibration.json").exists() for s in range(3))
        line(True, f"{cat}/{model}: {n} of 3 seeds trained" + ("" if n == 3 else " (queue will train the rest)"),
             required=False)

print()
if problems:
    print(f"NOT READY - {len(problems)} problem(s) above. Fix them before running the queue.")
    sys.exit(1)
print("READY - start the queue with: run_windows.bat   (or: python scripts/run_queue.py)")
