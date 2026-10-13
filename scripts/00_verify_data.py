"""Milestone 1: verify the MVTec AD 2 download and assign a lighting condition to every test image.

  python scripts/00_verify_data.py

1. Checks the directory structure and prints real file counts (private test
   splits are reported once as out of scope; they are never downloaded).
2. Prints 20 example filenames from test_public/good and test_public/bad verbatim.
3. Probes luminance (mean, SD, p05, p95 at 1/8 scale) for train/good and
   test_public/{good,bad} -> results/lighting_probe.csv.
4. Prints a histogram summary of mean luminance per split.
5. Assigns lighting in {nominal, shifted} -> results/lighting_assignment.csv:
   - filename path: if filenames carry a condition token (e.g. 003_overexposed.png)
     and test_public has tokens absent from train, nominal = token seen in train;
   - luminance path otherwise: nominal iff |mean_lum - train mean| <= k * train SD.
   The rule actually used is written to results/lighting_rule.json.

Stop after this script and report the outcome before continuing (plan §4).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.config import base_parser, setup  # noqa: E402
from src.data.splits import IMAGE_SUFFIXES  # noqa: E402

EXPECTED = {  # plan §3 - verified, not assumed
    "can": {"train/good": 412, "validation/good": 46, "test_public/good": 72, "test_public/bad": 90},
    "sheet_metal": {"train/good": 137, "validation/good": 19, "test_public/good": 24, "test_public/bad": 90},
}
IN_SCOPE = ["train/good", "validation/good", "test_public/good", "test_public/bad",
            "test_public/ground_truth/bad"]
OUT_OF_SCOPE = ["test_private", "test_private_mixed"]
PROBE_SPLITS = ["train/good", "test_public/good", "test_public/bad"]


def images(d: Path) -> list[Path]:
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith("."))


def luminance_stats(path: Path) -> tuple[float, float, float, float]:
    img = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_8)
    if img is None:
        raise FileNotFoundError(path)
    y = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)  # ITU-R BT.601 luma
    return float(y.mean()), float(y.std()), float(np.percentile(y, 5)), float(np.percentile(y, 95))


def token(filename: str) -> str:
    """Condition token of a filename: '003_overexposed.png' -> 'overexposed', '003.png' -> ''."""
    return re.sub(r"^\d+[_-]?", "", Path(filename).stem)


def text_histogram(values: np.ndarray, lo: float, hi: float, bins: int = 12, width: int = 40) -> str:
    counts, edges = np.histogram(values, bins=bins, range=(lo, hi))
    peak = max(counts.max(), 1)
    return "\n".join(f"    {edges[i]:6.1f}-{edges[i + 1]:6.1f} | {'#' * int(round(width * c / peak)):<{width}} {c}"
                     for i, c in enumerate(counts))


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--method", choices=["auto", "filename", "luminance"], default="auto",
                        help="lighting assignment rule (default: filename if tokens allow it)")
    args = parser.parse_args()
    cfg = setup(args)
    results = cfg.path("results")
    results.mkdir(parents=True, exist_ok=True)
    k_sd = float(cfg.lighting.sd_multiplier)

    probe_rows, assign_rows, rules = [], [], {}
    for category in cfg.categories:
        root = cfg.data_root / category
        print(f"\n=== {category}  ({root})")
        if not root.is_dir():
            print("  category directory not found - download it first; skipping")
            continue
        for sub in IN_SCOPE:
            n = len(images(root / sub))
            exp = EXPECTED.get(category, {}).get(sub)
            flag = "" if exp is None else ("  ok" if n == exp else f"  MISMATCH (plan expects {exp})")
            print(f"  {sub:32s} {n:5d}{flag}")
        for sub in OUT_OF_SCOPE:
            state = "present (ignored, out of scope)" if (root / sub).exists() else "not present (out of scope)"
            print(f"  {sub:32s} {state}")

        for sub in ("test_public/good", "test_public/bad"):
            names = [p.name for p in images(root / sub)][:20]
            print(f"\n  first {len(names)} filenames in {sub}:")
            for name in names:
                print(f"    {name}")

        cat_rows = []
        for sub in PROBE_SPLITS:
            for p in tqdm(images(root / sub), desc=f"{category}/{sub}", leave=False):
                mean, sd, p05, p95 = luminance_stats(p)
                cat_rows.append({"category": category, "split": sub, "filename": p.name,
                                 "mean_lum": mean, "std_lum": sd, "p05": p05, "p95": p95})
        if not cat_rows:
            continue
        probe = pd.DataFrame(cat_rows)
        probe_rows.extend(cat_rows)

        lo, hi = float(probe["mean_lum"].min()), float(probe["mean_lum"].max()) + 1e-6
        print(f"\n  mean luminance by split (0-255):")
        for sub, g in probe.groupby("split", sort=False):
            v = g["mean_lum"].to_numpy()
            print(f"  {sub}: n={len(v)} mean={v.mean():.1f} sd={v.std(ddof=1) if len(v) > 1 else 0:.1f} "
                  f"min={v.min():.1f} max={v.max():.1f}")
            print(text_histogram(v, lo, hi))

        # ------------------------------------------------------------ lighting assignment
        train = probe[probe["split"] == "train/good"]
        test = probe[probe["split"] != "train/good"].copy()
        train_tokens = set(train["filename"].map(token)) - {""}
        test_tokens = set(test["filename"].map(token)) - {""}
        print(f"\n  filename tokens  train: {sorted(train_tokens) or '-'}  test_public: {sorted(test_tokens) or '-'}")
        usable = bool(train_tokens) and bool(test_tokens - train_tokens)
        method = args.method if args.method != "auto" else ("filename" if usable else "luminance")
        if method == "filename" and not usable:
            raise SystemExit("--method filename requested but filenames do not distinguish lighting conditions")
        if method == "filename":
            test["lighting"] = np.where(test["filename"].map(token).isin(train_tokens), "nominal", "shifted")
            rule = {"method": "filename", "nominal_tokens": sorted(train_tokens),
                    "shifted_tokens": sorted(test_tokens - train_tokens)}
        else:
            mu, sd = float(train["mean_lum"].mean()), float(train["mean_lum"].std(ddof=1))
            low, high = mu - k_sd * sd, mu + k_sd * sd
            test["lighting"] = np.where(test["mean_lum"].between(low, high), "nominal", "shifted")
            rule = {"method": "luminance", "train_mean": mu, "train_sd": sd, "k_sd": k_sd,
                    "nominal_range": [low, high]}
        counts = test.groupby(["split", "lighting"]).size().unstack(fill_value=0)
        rule["counts"] = {s: {k: int(v) for k, v in row.items()} for s, row in counts.iterrows()}
        rules[category] = rule
        print(f"\n  lighting assignment: {method} path")
        print("  " + counts.to_string().replace("\n", "\n  "))
        assign_rows.extend(test[["category", "split", "filename", "lighting", "mean_lum"]].to_dict("records"))

    if probe_rows:
        pd.DataFrame(probe_rows).to_csv(results / "lighting_probe.csv", index=False)
    if assign_rows:
        pd.DataFrame(assign_rows).to_csv(results / "lighting_assignment.csv", index=False)
        (results / "lighting_rule.json").write_text(json.dumps(rules, indent=2))
        print(f"\nwrote {results / 'lighting_probe.csv'}, {results / 'lighting_assignment.csv'}, "
              f"{results / 'lighting_rule.json'}")
    else:
        print("\nno images found - nothing written")


if __name__ == "__main__":
    main()
