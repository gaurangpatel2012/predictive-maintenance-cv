"""Experiments 2 + 3 (RQ2): reliability under degradation and under shifted lighting.

  python scripts/04_eval_reliability.py --model patchcore --category can [--seed 0]

On the re-split eval portion, always at the fixed nominal threshold:
  baseline     condition=none, level=0, nominal lighting, untouched
  degradation  4 families x 5 severities, applied at 384 px to nominal-lighting images
  lighting     condition=shifted_lighting, level=0, shifted-lighting images, untouched
With --scope all (or both), the same grid is also run on the whole eval portion, both
lightings together, so the degradation curves rest on every evaluation image:
  baseline_all     condition=none, level=0, whole eval portion, untouched
  degradation_all  4 families x 5 severities on the whole eval portion
Each degraded condition is compared with the same images undegraded (README, D18).
Rows in results/exp2_reliability.csv are keyed on
(category, model, seed, condition_type, condition, level).
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import add_model_args, base_parser, seeds_for, setup  # noqa: E402
from src.data.dataset import InspectionDataset  # noqa: E402
from src.data.splits import load_splits, partitions  # noqa: E402
from src.evaluation import condition_metrics, score_dataset  # noqa: E402
from src.models.registry import LOCALISING, artifact_dir, load_calibration, load_model  # noqa: E402
from src.results_io import replace_rows  # noqa: E402
from src.runtime import empty_cache, resolve_device  # noqa: E402
from src.seeding import seed_everything  # noqa: E402

COLUMNS = ["category", "model", "seed", "condition_type", "condition", "level",
           "n_images", "precision", "recall", "f1", "auroc", "aupro_005",
           "fpr_at_fixed", "fnr_at_fixed", "threshold_fixed", "threshold_optimal",
           "threshold_drift", "threshold_drift_norm", "ece"]
KEY = ["category", "model", "seed", "condition_type", "condition", "level"]


def main() -> None:
    parser = base_parser(__doc__)
    add_model_args(parser)
    parser.add_argument("--device", default=None)
    parser.add_argument("--scope", choices=["nominal", "all", "both"], default="both",
                        help="degradation grid on nominal-lighting images, the whole eval portion, or both")
    args = parser.parse_args()
    cfg = setup(args)
    device = resolve_device(args.device or cfg.runtime.device)
    parts = partitions(load_splits(args.category, cfg), args.model)
    nominal, shifted = parts["eval_nominal"], parts["eval_shifted"]
    with_maps = args.model in LOCALISING

    conditions: list[tuple[str, str, float, object]] = []
    grids = {"nominal": [("", nominal)], "all": [("_all", parts["eval"])],
             "both": [("", nominal), ("_all", parts["eval"])]}[args.scope]
    for suffix, table in grids:
        conditions.append((f"baseline{suffix}", "none", 0.0, table))
        for kind, spec in cfg.degradations.data.items():
            conditions += [(f"degradation{suffix}", kind, float(level), table) for level in spec["levels"]]
    if len(shifted):
        conditions.append(("lighting", "shifted_lighting", 0.0, shifted))
    else:
        print("no shifted-lighting images in the eval portion - lighting condition skipped")

    for seed in seeds_for(args, cfg):
        seed_everything(seed)
        d = artifact_dir(cfg, args.category, args.model, seed)
        calibration = load_calibration(d)
        model = load_model(args.model, d / "model.pt", device)
        rows = []
        for ctype, cond, level, table in conditions:
            degradation = (cond, level) if ctype.startswith("degradation") else None
            ds = InspectionDataset.from_table(cfg, args.category, table, degradation=degradation)
            m = condition_metrics(score_dataset(model, ds, cfg, with_maps=with_maps), args.model, calibration, cfg)
            rows.append({"category": args.category, "model": args.model, "seed": seed,
                         "condition_type": ctype, "condition": cond, "level": level, **m})
            print(f"seed {seed} {cond:18s} {level:5g}: F1={m['f1']:.3f} AUROC={m['auroc']:.3f} "
                  f"FPR={m['fpr_at_fixed']:.3f} FNR={m['fnr_at_fixed']:.3f} drift={m['threshold_drift_norm']:+.2f}sd")
        replace_rows(cfg.path("results") / "exp2_reliability.csv", rows, scope=KEY, columns=COLUMNS)
        del model
        gc.collect()
        empty_cache(device)


if __name__ == "__main__":
    main()
