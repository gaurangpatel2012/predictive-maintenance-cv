"""Experiment 4 (RQ3): latency and computational profile. Local machine only.

  python scripts/05_bench_latency.py --model vit --category can [--seed 0]

Batch size 1 (the line presents parts one at a time). Per repeat: 50 warm-up
images discarded, then 500 timed single-image inferences cycling the eval
portion; 3 repeats. Inputs are encoded native-resolution frames held in memory,
so t_pre covers decode + validation + resize + normalise + host-to-device copy.

  results/exp4_latency_raw.csv      one row per timed inference
  results/exp4_latency_summary.csv  one row per (category, model)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bench import load_encoded_images  # noqa: E402
from src.config import add_model_args, base_parser, setup  # noqa: E402
from src.data.splits import load_splits, partitions  # noqa: E402
from src.metrics.timing import summarise  # noqa: E402
from src.models.registry import artifact_dir, load_calibration, load_model  # noqa: E402
from src.pipeline import InferencePipeline  # noqa: E402
from src.results_io import replace_rows  # noqa: E402
from src.runtime import (device_name, peak_device_memory_mb, peak_rss_mb,  # noqa: E402
                         reset_peak_device_memory, resolve_device)
from src.seeding import seed_everything  # noqa: E402

RAW = ["category", "model", "repeat", "idx", "t_pre_ms", "t_infer_ms", "t_post_ms", "t_total_ms"]
SUMMARY = ["category", "model", "median_ms", "p95_ms", "p99_ms", "mean_ms", "std_ms",
           "throughput_ips", "deadline_miss_rate", "model_size_mb", "peak_mem_mb", "device", "device_name",
           "t_pre_median_ms", "t_infer_median_ms", "t_post_median_ms", "input_source", "peak_mem_kind"]


def main() -> None:
    parser = base_parser(__doc__)
    add_model_args(parser)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    cfg = setup(args)
    seed = args.seed if args.seed is not None else int(cfg.train.seeds[0])
    seed_everything(seed)
    device = resolve_device(args.device or cfg.runtime.device)
    rt = cfg.realtime
    budget = float(rt.latency_budget_ms)

    table = partitions(load_splits(args.category, cfg), args.model)["eval"]
    images, _, _, source = load_encoded_images(cfg, args.category, table)
    d = artifact_dir(cfg, args.category, args.model, seed)
    model = load_model(args.model, d / "model.pt", device)
    pipe = InferencePipeline.from_model(model, device, load_calibration(d))
    reset_peak_device_memory(device)

    raw, walls = [], []
    n_timed, n_warm = int(rt.latency_images), int(rt.warmup_images)
    for rep in range(int(rt.repeats)):
        for i in range(n_warm):
            pipe(images[i % len(images)])
        t0 = time.perf_counter()
        for i in range(n_timed):
            r = pipe(images[i % len(images)])
            raw.append({"category": args.category, "model": args.model, "repeat": rep, "idx": i,
                        "t_pre_ms": r.t_pre_ms, "t_infer_ms": r.t_infer_ms, "t_post_ms": r.t_post_ms,
                        "t_total_ms": r.t_total_ms})
        walls.append(time.perf_counter() - t0)
        print(f"repeat {rep}: {n_timed} images in {walls[-1]:.1f}s")

    totals = [r["t_total_ms"] for r in raw]
    s = summarise(totals, budget, wall_seconds=sum(walls))
    dev_peak = peak_device_memory_mb(device)
    median = lambda k: float(sorted(r[k] for r in raw)[len(raw) // 2])  # noqa: E731
    summary = {
        "category": args.category, "model": args.model, **s,
        "model_size_mb": model.size_mb(),
        "peak_mem_mb": dev_peak if device.type == "cuda" else peak_rss_mb(),
        "peak_mem_kind": "cuda_max_allocated" if device.type == "cuda" else "process_peak_rss",
        "device": device.type, "device_name": device_name(device),
        "t_pre_median_ms": median("t_pre_ms"), "t_infer_median_ms": median("t_infer_ms"),
        "t_post_median_ms": median("t_post_ms"), "input_source": source,
    }
    results = cfg.path("results")
    replace_rows(results / "exp4_latency_raw.csv", raw, scope=["category", "model"], columns=RAW)
    replace_rows(results / "exp4_latency_summary.csv", [summary], scope=["category", "model"], columns=SUMMARY)
    print(f"median {s['median_ms']:.1f} ms, p95 {s['p95_ms']:.1f} ms, p99 {s['p99_ms']:.1f} ms, "
          f"miss rate {s['deadline_miss_rate']:.4f}, {s['throughput_ips']:.2f} img/s on {summary['device_name']}")


if __name__ == "__main__":
    main()
