"""Post-hoc diagnostic: does the convolutional autoencoder copy its input (identity mapping)?

Compares the reconstruction error inside the ground-truth defect regions with that of
normal pixels, and the pixel-level AUROC of the error map, on the evaluation portion
(all images, normal and defective) for every seed of both categories. It informed no
change to the model; the values are diagnostic only.
"""
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config  # noqa: E402
from src.data.dataset import InspectionDataset  # noqa: E402
from src.data.splits import load_splits, partitions  # noqa: E402
from src.evaluation import score_dataset  # noqa: E402
from src.models.registry import load_model  # noqa: E402
from src.runtime import resolve_device  # noqa: E402

cfg = load_config("configs/default.yaml")
dev = resolve_device("auto")
rows = []
for cat in ("sheet_metal", "can"):
    ds = InspectionDataset.from_table(cfg, cat, partitions(load_splits(cat, cfg), "cae")["eval"])
    for seed in (0, 1, 2):
        m = load_model("cae", f"artifacts/{cat}/cae/seed{seed}/model.pt", dev)
        s = score_dataset(m, ds, cfg, with_maps=True)
        mp, mk = s.maps.astype(np.float32).ravel(), s.masks.ravel() > 0
        row = dict(category=cat, seed=seed, n_images=len(ds), device=str(dev),
                   pixel_auroc=round(float(roc_auc_score(mk, mp)), 4),
                   err_defect=round(float(mp[mk].mean()), 5), err_normal=round(float(mp[~mk].mean()), 5),
                   ratio=round(float(mp[mk].mean() / mp[~mk].mean()), 3))
        rows.append(row)
        print(row, flush=True)
        del m
out = Path("results/cae_identity_check.csv")
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("wrote", out)
