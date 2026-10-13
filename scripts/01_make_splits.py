"""Re-split test_public 50:20:30 into train/val/eval, stratified jointly on (label, lighting).

  python scripts/01_make_splits.py --category can

Written once to results/splits_<category>.csv with split.seed and reused by every
script. An existing file is never regenerated unless --force is given.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import base_parser, setup  # noqa: E402
from src.data.splits import make_splits, splits_path  # noqa: E402


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--category", required=True)
    parser.add_argument("--force", action="store_true", help="overwrite an existing split file")
    args = parser.parse_args()
    cfg = setup(args)
    out = splits_path(cfg, args.category)
    if out.exists() and not args.force:
        print(f"{out} already exists - reusing it (never regenerate with a different seed)")
        return
    df = make_splits(args.category, cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} images, seed {cfg.split.seed})")
    table = df.groupby(["split_role", "lighting", "label"]).size().unstack(fill_value=0)
    table.columns = [f"label={c}" for c in table.columns]
    print(table.to_string())


if __name__ == "__main__":
    main()
