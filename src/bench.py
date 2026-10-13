"""Helpers shared by the latency benchmark and the stream simulation."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.data.dataset import resolve_source

log = logging.getLogger(__name__)


def load_encoded_images(cfg, category: str, table: pd.DataFrame) -> tuple[list[bytes], list[int], list[str], str]:
    """Encoded image bytes held in memory, so disk I/O never enters the timed path.

    Native-resolution originals are used when present, so t_pre includes the real
    decode and resize cost of a camera frame. If only the 384 px cache exists, the
    cached files are used and ``input_source`` says so - those timings understate
    preprocessing and must be reported as such.
    """
    data, labels, names, sources = [], [], [], set()
    for rel, label in zip(table["filename"], table["label"]):
        original = cfg.data_root / category / rel
        path = original if original.exists() else resolve_source(cfg, category, rel)[0]
        sources.add("original" if path == original else "cache_384")
        data.append(Path(path).read_bytes())
        labels.append(int(label))
        names.append(rel)
    source = "+".join(sorted(sources))
    if "cache_384" in sources:
        log.warning("timing on cached 384 px inputs (originals missing): preprocessing is understated")
    return data, labels, names, source
