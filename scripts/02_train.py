"""Train one model on one category with one seed.

  python scripts/02_train.py --model resnet50 --category can --seed 0

Writes artifacts/<category>/<model>/seed<seed>/{model.pt, calibration.json, train_meta.json}
and replaces this run's rows in results/training_log.csv. PatchCore has no
gradient training; its memory-bank construction time goes to
results/patchcore_construction.csv and is never averaged with epoch times.

After training, the fixed decision threshold is selected once (F1-max on the
nominal-lighting part of the re-split val portion) and stored in
calibration.json; every evaluation script reuses it.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import MODEL_NAMES, base_parser, setup  # noqa: E402
from src.data.dataset import InspectionDataset  # noqa: E402
from src.data.splits import load_splits, partitions  # noqa: E402
from src.evaluation import calibrate, score_dataset  # noqa: E402
from src.models.registry import artifact_dir, build_model, load_model  # noqa: E402
from src.results_io import replace_rows  # noqa: E402
from src.runtime import resolve_device  # noqa: E402
from src.seeding import seed_everything  # noqa: E402

log = logging.getLogger("train")
LOG_COLUMNS = ["category", "model", "seed", "epoch", "train_loss", "val_loss", "val_f1", "lr", "epoch_seconds"]
PC_COLUMNS = ["category", "model", "seed", "construction_seconds", "feature_seconds", "n_patches",
              "bank_size", "descriptor_dim", "bank_mb_fp16", "coreset_rate", "device"]


def write_calibration(model, cfg, args, parts, out: Path) -> dict:
    calib_ds = InspectionDataset.from_table(cfg, args.category, parts["calib"])
    calibration = calibrate(score_dataset(model, calib_ds, cfg))
    calibration.update({"model": args.model, "category": args.category, "seed": args.seed,
                        "calibration_set": "re-split val, nominal lighting"})
    if args.model == "cae":
        calibration["image_score"] = model.image_score
    (out / "calibration.json").write_text(json.dumps(calibration, indent=2))
    log.info("threshold %.6g (calib F1 %.3f, AUROC %.3f, n=%d)", calibration["threshold"],
             calibration["calib_f1"], calibration["calib_auroc"], calibration["n_calib"])
    if args.model == "cae" and calibration["calib_auroc"] < 0.6:
        log.warning("CAE calibration AUROC %.3f on n=%d images. Check pixel-level localisation on the "
                    "full test set before concluding it learned an identity mapping.",
                    calibration["calib_auroc"], calibration["n_calib"])
    return calibration


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--category", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default=None, help="override runtime.device")
    parser.add_argument("--calibrate-only", action="store_true",
                        help="skip training: reload the saved model and re-select its threshold "
                             "(use after changing how scores are computed)")
    args = parser.parse_args()
    cfg = setup(args)
    device = resolve_device(args.device or cfg.runtime.device)
    seed_everything(args.seed)
    log.info("training %s on %s, seed %d, device %s", args.model, args.category, args.seed, device)

    parts = partitions(load_splits(args.category, cfg), args.model)
    out = artifact_dir(cfg, args.category, args.model, args.seed)
    if args.calibrate_only:
        model = load_model(args.model, out / "model.pt", device)
        write_calibration(model, cfg, args, parts, out)
        return
    augment = args.model != "patchcore"  # the memory bank is built from un-augmented images
    train_ds = InspectionDataset.from_table(cfg, args.category, parts["train"], train=augment)
    val_ds = InspectionDataset.from_table(cfg, args.category, parts["val"])
    log.info("train n=%d (anomalous %d), val n=%d (anomalous %d)", len(train_ds), int(train_ds.labels.sum()),
             len(val_ds), int(val_ds.labels.sum()))

    model = build_model(args.model, cfg, device, seed=args.seed)
    result = model.fit(train_ds, val_ds, cfg)

    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "model.pt")
    write_calibration(model, cfg, args, parts, out)

    meta = {k: v for k, v in result.items() if k != "history"}
    meta.update({"size_mb": model.size_mb(), "device": str(device), "n_train": len(train_ds),
                 "n_val": len(val_ds), "config": cfg.to_dict()})
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2, default=str))

    key = {"category": args.category, "model": args.model, "seed": args.seed}
    results = cfg.path("results")
    if result["history"]:
        replace_rows(results / "training_log.csv", [{**key, **h} for h in result["history"]],
                     scope=["category", "model", "seed"], columns=LOG_COLUMNS)
    if "construction" in result:
        row = {**key, **result["construction"], "coreset_rate": float(cfg.patchcore.coreset_rate),
               "device": str(device)}
        replace_rows(results / "patchcore_construction.csv", [row],
                     scope=["category", "model", "seed"], columns=PC_COLUMNS)
    print(f"saved {out / 'model.pt'} ({model.size_mb():.1f} MiB)")


if __name__ == "__main__":
    main()
