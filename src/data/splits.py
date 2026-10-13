"""Re-split of test_public into train/val/eval, stratified jointly on (label, lighting).

The split is generated once, written to results/splits_<category>.csv and read by
every later script. ``filename`` and ``mask_path`` are paths relative to the
category directory (e.g. ``test_public/bad/000_regular.png``) so they resolve
against both the original dataset and the 384 px cache.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config

log = logging.getLogger(__name__)

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
OFFICIAL_DIRS = {"official_train": "train/good", "official_val": "validation/good"}
TEST_DIRS = {0: "test_public/good", 1: "test_public/bad"}
MASK_DIR = "test_public/ground_truth/bad"
SPLIT_COLUMNS = ["filename", "split_role", "label", "lighting", "mask_path"]


def list_images(cfg: Config, category: str, rel_dir: str) -> list[str]:
    """Relative paths of images in ``rel_dir``; reads the cache if the originals are gone."""
    for root in (cfg.data_root / category, cfg.cache_dir / category):
        d = root / rel_dir
        if d.is_dir():
            return sorted(
                f"{rel_dir}/{p.name}" for p in d.iterdir()
                if p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith(".")
            )
    return []


def mask_relpath(relpath: str) -> str:
    stem = Path(relpath).stem
    return f"{MASK_DIR}/{stem}_mask.png"


def mask_exists(cfg: Config, category: str, rel: str) -> bool:
    return (cfg.data_root / category / rel).exists() or (cfg.cache_dir / category / rel).exists()


def splits_path(cfg: Config, category: str) -> Path:
    return cfg.path("results") / f"splits_{category}.csv"


def lighting_path(cfg: Config) -> Path:
    return cfg.path("results") / "lighting_assignment.csv"


def load_lighting(cfg: Config, category: str) -> dict[str, str]:
    path = lighting_path(cfg)
    if not path.exists():
        raise SystemExit(f"{path} missing - run scripts/00_verify_data.py first (Milestone 1)")
    df = pd.read_csv(path)
    df = df[df["category"] == category]
    return {f"{s}/{f}": lg for s, f, lg in zip(df["split"], df["filename"], df["lighting"])}


def _allocate(n: int, ratios: dict[str, float]) -> dict[str, int]:
    """Largest-remainder allocation of n items to roles in the configured ratios."""
    names = list(ratios)
    total = sum(ratios.values())
    raw = [n * ratios[k] / total for k in names]
    counts = [int(np.floor(r)) for r in raw]
    order = sorted(range(len(names)), key=lambda i: (-(raw[i] - counts[i]), i))
    for i in order[: n - sum(counts)]:
        counts[i] += 1
    return dict(zip(names, counts))


def make_splits(category: str, cfg: Config) -> pd.DataFrame:
    lighting = load_lighting(cfg, category)
    rows = []
    for label, rel_dir in TEST_DIRS.items():
        for rel in list_images(cfg, category, rel_dir):
            if rel not in lighting:
                raise SystemExit(f"{rel} has no lighting assignment; re-run 00_verify_data.py")
            mask = ""
            if label == 1:
                mask = mask_relpath(rel)
                if not mask_exists(cfg, category, mask):
                    log.warning("no ground-truth mask for %s", rel)
                    mask = ""
            rows.append({"filename": rel, "label": label, "lighting": lighting[rel], "mask_path": mask})
    if not rows:
        raise SystemExit(f"no test_public images found for '{category}'")
    df = pd.DataFrame(rows).sort_values("filename").reset_index(drop=True)

    ratios = dict(cfg.split.ratios.to_dict())
    rng = np.random.default_rng(int(cfg.split.seed))
    df["split_role"] = ""
    if cfg.split.get("group_by") == "part":
        # MVTec AD 2 photographs each test part under six lightings (000_regular.png,
        # 000_overexposed.png, ...). All variants of a part go to the same portion, so no
        # defect is seen in training under one lighting and evaluated under another.
        # Groups are stratified by label; lighting is balanced by construction.
        df["part"] = df["label"].astype(str) + ":" + df["filename"].map(part_id)
        for label, group in df.groupby("label", sort=True):
            parts = np.array(sorted(group["part"].unique()))
            rng.shuffle(parts)
            counts = _allocate(len(parts), ratios)
            start = 0
            for role, c in counts.items():
                df.loc[df["part"].isin(parts[start : start + c]), "split_role"] = role
                start += c
            log.info("label %d: %d parts -> %s", label, len(parts), counts)
        return df[SPLIT_COLUMNS]
    for key, group in df.groupby(list(cfg.split.stratify_by), sort=True):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        counts = _allocate(len(idx), ratios)
        start = 0
        for role, c in counts.items():
            df.loc[idx[start : start + c], "split_role"] = role
            start += c
        log.info("stratum %s: n=%d -> %s", key, len(idx), counts)
    return df[SPLIT_COLUMNS]


def part_id(relpath: str) -> str:
    """Physical part identifier: the leading index of the filename ('003_shift_1.png' -> '003')."""
    m = re.match(r"^(\d+)", Path(relpath).name)
    if not m:
        raise ValueError(f"cannot derive a part id from '{relpath}'")
    return m.group(1)


def load_splits(category: str, cfg: Config, include_official: bool = True) -> pd.DataFrame:
    path = splits_path(cfg, category)
    if not path.exists():
        raise SystemExit(f"{path} missing - run scripts/01_make_splits.py --category {category}")
    df = pd.read_csv(path, keep_default_na=False)
    df["label"] = df["label"].astype(int)
    if include_official:
        extra = [
            {"filename": rel, "split_role": role, "label": 0, "lighting": "nominal", "mask_path": ""}
            for role, rel_dir in OFFICIAL_DIRS.items()
            for rel in list_images(cfg, category, rel_dir)
        ]
        df = pd.concat([df, pd.DataFrame(extra, columns=SPLIT_COLUMNS)], ignore_index=True)
    return df


def subset(df: pd.DataFrame, roles: list[str], lighting: str | None = None) -> pd.DataFrame:
    out = df[df["split_role"].isin(roles)]
    if lighting is not None:
        out = out[out["lighting"] == lighting]
    return out.reset_index(drop=True)


def partitions(df: pd.DataFrame, model: str) -> dict[str, pd.DataFrame]:
    """Which model sees what (plan §7.1), plus the threshold calibration set.

    ``calib`` is the nominal-lighting part of the re-split ``val`` portion. It is the
    only nominal validation data that contains anomalies, so the F1-maximising
    threshold is selected on it for all five models (see README, decision D1).
    """
    one_class = model in ("cae", "patchcore")
    return {
        "train": subset(df, ["official_train"] if one_class else ["official_train", "train"]),
        "val": subset(df, ["official_val"] if one_class else ["official_val", "val"]),
        "calib": subset(df, ["val"], lighting="nominal"),
        "eval_nominal": subset(df, ["eval"], lighting="nominal"),
        "eval_shifted": subset(df, ["eval"], lighting="shifted"),
        "eval": subset(df, ["eval"]),
    }
