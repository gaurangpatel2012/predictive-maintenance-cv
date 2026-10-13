"""Idempotent, crash-safe writes to results/*.csv.

Re-running a configuration replaces its rows instead of duplicating them. The
file is written to a temporary sibling and atomically renamed, so a run that
dies midway leaves the previous file intact.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


def _norm(value: object) -> str:
    """Normalise a key value so 2, 2.0 and '2' compare equal."""
    if value is None:
        return ""
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(f):
        return "nan"
    return f"{f:.10g}"


def _keys(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    return df[list(cols)].apply(lambda r: tuple(_norm(v) for v in r), axis=1)


def replace_rows(
    path: Path,
    rows: pd.DataFrame | Iterable[dict],
    scope: Sequence[str],
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Drop existing rows whose ``scope`` columns match any new row, then append.

    ``scope`` is the identity of a run: e.g. (category, model, seed) for Exp 1, or
    (category, model) for raw latency samples, where a re-run replaces every sample.
    """
    new = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    if columns is not None:
        missing = [c for c in columns if c not in new.columns]
        for c in missing:
            new[c] = float("nan")
        new = new[list(columns)]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        old = pd.read_csv(path)
        if scope and len(old) and len(new) and all(c in old.columns for c in scope):
            old = old[~_keys(old, scope).isin(set(_keys(new, scope)))]
        out = pd.concat([old, new], ignore_index=True) if len(old) else new
        if columns is not None:
            out = out[[c for c in columns if c in out.columns]]
    else:
        out = new
    tmp = path.with_suffix(path.suffix + ".tmp")
    out.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return out


def append_rows(path: Path, rows: pd.DataFrame | Iterable[dict], columns: Sequence[str]) -> None:
    """Plain append for logs whose rows are unique by construction (e.g. timestamped)."""
    replace_rows(path, rows, scope=[], columns=columns)
