#!/usr/bin/env bash
# Full experimental matrix, one model in memory at a time.
# Usage: bash scripts/run_all.sh [category ...]      (default: sheet_metal can)
# Milestone 1 (00_verify_data.py) must have been run and reviewed first.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
CATEGORIES=${*:-"sheet_metal can"}
MODELS="patchcore cae resnet50 efficientnet_b0 vit"
SEEDS="0 1 2"

for cat in $CATEGORIES; do
  $PY scripts/01b_build_cache.py --category "$cat"
  $PY scripts/01_make_splits.py --category "$cat"
  for model in $MODELS; do
    for seed in $SEEDS; do
      $PY scripts/02_train.py --model "$model" --category "$cat" --seed "$seed"
    done
    $PY scripts/03_eval_nominal.py --model "$model" --category "$cat"
    $PY scripts/04_eval_reliability.py --model "$model" --category "$cat"
    $PY scripts/05_bench_latency.py --model "$model" --category "$cat"
    $PY scripts/06_stream.py --model "$model" --category "$cat"
  done
done
$PY scripts/08_make_figures.py
