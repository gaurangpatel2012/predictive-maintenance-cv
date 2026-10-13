"""Experiment 1 (RQ1): predictive performance under nominal lighting.

  python scripts/03_eval_nominal.py --model patchcore --category can [--seed 0]

Evaluates on the re-split eval portion restricted to lighting == nominal, at the
fixed threshold from calibration.json. Every seed in train.seeds is evaluated
unless --seed is given, loading one model at a time. Rows in
results/exp1_predictive.csv are keyed on (category, model, seed) and replaced on re-run.
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

COLUMNS = ["category", "model", "seed", "n_images", "n_anomalous", "threshold",
           "precision", "recall", "f1", "auroc", "aupro_005", "aupro_030"]


def main() -> None:
    parser = base_parser(__doc__)
    add_model_args(parser)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    cfg = setup(args)
    device = resolve_device(args.device or cfg.runtime.device)
    eval_table = partitions(load_splits(args.category, cfg), args.model)["eval_nominal"]
    ds = InspectionDataset.from_table(cfg, args.category, eval_table)

    for seed in seeds_for(args, cfg):
        seed_everything(seed)
        d = artifact_dir(cfg, args.category, args.model, seed)
        calibration = load_calibration(d)
        model = load_model(args.model, d / "model.pt", device)
        scored = score_dataset(model, ds, cfg, with_maps=args.model in LOCALISING)
        m = condition_metrics(scored, args.model, calibration, cfg)
        row = {"category": args.category, "model": args.model, "seed": seed,
               "threshold": calibration["threshold"], **m}
        replace_rows(cfg.path("results") / "exp1_predictive.csv", [row],
                     scope=["category", "model", "seed"], columns=COLUMNS)
        print(f"seed {seed}: n={m['n_images']} F1={m['f1']:.3f} AUROC={m['auroc']:.3f} "
              f"AU-PRO@0.05={m['aupro_005']:.3f}")
        del model, scored
        gc.collect()
        empty_cache(device)


if __name__ == "__main__":
    main()
