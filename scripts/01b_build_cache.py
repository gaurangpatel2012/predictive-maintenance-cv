"""Build the 384 px image cache once; every later script reads it.

  python scripts/01b_build_cache.py --category can

Images are resized bilinearly and masks with nearest neighbour, saved as PNG
under data/cache_384/<category>/ with the original relative paths, plus a
manifest.csv (relpath, split, label, orig_w, orig_h). Degradations are applied
after resize at 384 px, so cached and uncached runs are identical.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.config import base_parser, setup  # noqa: E402
from src.data.dataset import read_mask, read_rgb, resize_mask, resize_rgb  # noqa: E402
from src.data.splits import IMAGE_SUFFIXES, MASK_DIR  # noqa: E402

SOURCES = {"train/good": 0, "validation/good": 0, "test_public/good": 0, "test_public/bad": 1}
EXPECTED = {"can": 620, "sheet_metal": 270}  # plan §3: train + validation + test_public


def files(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
                  and not p.name.startswith(".")) if d.is_dir() else []


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--category", required=True)
    args = parser.parse_args()
    cfg = setup(args)
    src_root = cfg.data_root / args.category
    dst_root = cfg.cache_dir / args.category
    if not src_root.is_dir():
        raise SystemExit(f"{src_root} not found")
    size = int(cfg.image_size)
    rebuild = bool(cfg.cache.rebuild)

    rows, n_masks, n_src = [], 0, 0
    for split, label in SOURCES.items():
        for p in tqdm(files(src_root / split), desc=split, leave=False):
            n_src += 1
            rel = f"{split}/{p.name}"
            out = (dst_root / rel).with_suffix(".png")
            img = read_rgb(p)
            h, w = img.shape[:2]
            if rebuild or not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out), cv2.cvtColor(resize_rgb(img, size), cv2.COLOR_RGB2BGR))
            rows.append({"relpath": rel, "split": split, "label": label, "orig_w": w, "orig_h": h})
    for p in tqdm(files(src_root / MASK_DIR), desc="masks", leave=False):
        out = dst_root / MASK_DIR / (p.stem + ".png")
        if rebuild or not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), resize_mask(read_mask(p), size) * 255)
        n_masks += 1

    manifest = pd.DataFrame(rows)
    dst_root.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(dst_root / "manifest.csv", index=False)
    print(f"cached {len(manifest)} images and {n_masks} masks under {dst_root}")
    print(manifest.groupby("split").size().to_string())
    if len(manifest) != n_src:
        raise SystemExit(f"manifest has {len(manifest)} rows but {n_src} source images were found")
    exp = EXPECTED.get(args.category)
    if exp is not None and len(manifest) != exp:
        print(f"WARNING: plan §3 expects {exp} images for {args.category}, found {len(manifest)} - report this")
    elif exp is not None:
        print(f"row count matches plan §3 ({exp})")


if __name__ == "__main__":
    main()
