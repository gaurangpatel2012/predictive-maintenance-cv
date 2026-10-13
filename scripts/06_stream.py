"""Experiment 5 (RQ3): real-time stream simulation and the sliding-window sweep.

  python scripts/06_stream.py --model patchcore --category can [--seed 0]

Presents realtime.sequence_length images (500) sequentially at 1 Hz (cycling the eval portion in a fixed
order), after 50 warm-up images that are processed back-to-back and discarded;
realtime.stream_repeats repeats (2). Arrival times follow a fixed schedule, so if a result is late the
next image waits in the queue and its e2e latency includes that wait - the
behaviour of a firm real-time station that cannot skip parts.

  results/exp5_stream.csv        one row per image per repeat
  results/exp5_window_sweep.csv  window length x rate threshold, synthetic step change

A full run takes sequence_length / arrival_rate_hz seconds per repeat
(about 8 minutes at the defaults).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.bench import load_encoded_images  # noqa: E402
from src.config import add_model_args, base_parser, setup  # noqa: E402
from src.data.splits import load_splits, partitions  # noqa: E402
from src.metrics.timing import summarise  # noqa: E402
from src.models.registry import artifact_dir, load_calibration, load_model  # noqa: E402
from src.pipeline import InferencePipeline  # noqa: E402
from src.results_io import replace_rows  # noqa: E402
from src.runtime import resolve_device, rss_mb  # noqa: E402
from src.seeding import seed_everything  # noqa: E402
from src.stream import window_sweep  # noqa: E402

STREAM = ["category", "model", "repeat", "idx", "arrival_ts", "result_ts", "e2e_ms", "pred", "label",
          "rss_mb", "filename", "score"]
SWEEP = ["category", "model", "window_W", "rate_threshold", "detection_delay_images", "false_alarm_rate",
         "n_indications", "n_trials_detected", "n_trials"]


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
    length, period = int(rt.sequence_length), 1.0 / float(rt.arrival_rate_hz)

    table = partitions(load_splits(args.category, cfg), args.model)["eval"]
    images, labels, names, source = load_encoded_images(cfg, args.category, table)
    d = artifact_dir(cfg, args.category, args.model, seed)
    model = load_model(args.model, d / "model.pt", device)
    pipe = InferencePipeline.from_model(model, device, load_calibration(d), return_map=True)

    rows, decisions = [], []
    for rep in range(int(rt.get("stream_repeats", rt.repeats))):
        for i in range(int(rt.warmup_images)):
            pipe(images[i % len(images)])
        start = time.perf_counter() + 0.05
        rep_preds = []
        for i in range(length):
            j = i % len(images)
            arrival = start + i * period
            wait = arrival - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            r = pipe(images[j])
            done = time.perf_counter()
            rep_preds.append(r.decision)
            rows.append({"category": args.category, "model": args.model, "repeat": rep, "idx": i,
                         "arrival_ts": arrival - start, "result_ts": done - start,
                         "e2e_ms": (done - arrival) * 1e3, "pred": r.decision, "label": labels[j],
                         "rss_mb": rss_mb(), "filename": names[j], "score": r.score})
        decisions.append(np.asarray(rep_preds))
        e2e = [r["e2e_ms"] for r in rows if r["repeat"] == rep]
        s = summarise(e2e, float(rt.latency_budget_ms))
        print(f"repeat {rep}: median {s['median_ms']:.1f} ms, p99 {s['p99_ms']:.1f} ms, "
              f"miss rate {s['deadline_miss_rate']:.4f}")

    identical = all(np.array_equal(decisions[0], x) for x in decisions[1:])
    print(f"decisions identical across {len(decisions)} repeats: {identical} (input source: {source})")

    # Window sweep from the model's real decisions on each eval image (first repeat).
    per_image = {names[i % len(images)]: int(decisions[0][i]) for i in range(min(length, len(images)))}
    pred_n = [per_image[n] for n, y in zip(names, labels) if y == 0 and n in per_image]
    pred_a = [per_image[n] for n, y in zip(names, labels) if y == 1 and n in per_image]
    sc = rt.step_change
    if pred_n and pred_a:
        sweep = window_sweep(pred_n, pred_a, list(rt.window_lengths), list(rt.defect_rate_thresholds),
                             int(sc.sequence_length), int(sc.change_index), float(sc.prevalence_before),
                             float(sc.prevalence_after), int(sc.trials), int(sc.seed))
        sweep = [{"category": args.category, "model": args.model, **r} for r in sweep]
        replace_rows(cfg.path("results") / "exp5_window_sweep.csv", sweep,
                     scope=["category", "model"], columns=SWEEP)
    else:
        print("eval portion lacks normal or anomalous images - window sweep skipped")
    replace_rows(cfg.path("results") / "exp5_stream.csv", rows, scope=["category", "model"], columns=STREAM)


if __name__ == "__main__":
    main()
