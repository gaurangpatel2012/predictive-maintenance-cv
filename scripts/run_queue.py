"""Run the whole experimental matrix on one machine, one job at a time (any OS).

  python scripts/run_queue.py                 # everything still missing
  python scripts/run_queue.py --only train    # just the training stage
  python scripts/run_queue.py --dry-run       # list what would run

Stages, in order: train -> eval (Exp 1, Exp 2+3) -> timing (Exp 4, Exp 5) -> figures.
Timing runs last so nothing else competes for the machine. Each step logs to
logs/<step>.log; finished steps are listed in logs/done.txt and skipped on re-run,
and a model whose calibration.json exists is never retrained, so the queue can be
restarted after any interruption. Exp 6 (online service) is not part of the queue.

Keep the machine awake and plugged in: on Windows set the power plan to never
sleep while plugged in; on macOS run it under `caffeinate -dims`.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ["sheet_metal", "can"]
MODELS = ["patchcore", "cae", "resnet50", "efficientnet_b0", "vit"]
SEEDS = [0, 1, 2]
# Micro-batch x accumulation = effective batch 16 in every profile (README, decision D15).
# "laptop" fits an 8 GB machine; "colab" uses a 16 GB GPU and two data-loading workers.
PROFILES = {
    "laptop": {
        "vit": ["--set", "train.batch_size=4", "--set", "train.accumulation_steps=4"],
        "resnet50": ["--set", "train.batch_size=8", "--set", "train.accumulation_steps=2"],
        "efficientnet_b0": ["--set", "train.batch_size=8", "--set", "train.accumulation_steps=2"],
    },
    "colab": {
        "vit": ["--set", "train.batch_size=8", "--set", "train.accumulation_steps=2"],
        "*": ["--set", "runtime.num_workers=2"],
    },
}
MEM_ARGS: dict[str, list[str]] = {}


def mem_args(model: str) -> list[str]:
    return [*MEM_ARGS.get(model, []), *MEM_ARGS.get("*", [])]


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def build_steps(stream_categories: list[str]) -> list[tuple[str, str, list[str]]]:
    """(stage, name, argv) in execution order."""
    py = sys.executable
    steps = []
    for cat in CATEGORIES:
        for m in MODELS:
            for s in SEEDS:
                steps.append(("train", f"train_{m}_{cat}_seed{s}",
                              [py, "scripts/02_train.py", "--model", m, "--category", cat, "--seed", str(s),
                               *mem_args(m)]))
    for cat in CATEGORIES:
        for m in MODELS:
            steps.append(("eval", f"exp1_{m}_{cat}",
                          [py, "scripts/03_eval_nominal.py", "--model", m, "--category", cat, *mem_args(m)]))
            steps.append(("eval", f"exp2_{m}_{cat}",
                          [py, "scripts/04_eval_reliability.py", "--model", m, "--category", cat,
                           *mem_args(m)]))
    for cat in CATEGORIES:
        for m in MODELS:
            steps.append(("timing", f"exp4_{m}_{cat}", [py, "scripts/05_bench_latency.py", "--model", m, "--category", cat]))
            if cat in stream_categories:
                steps.append(("timing", f"exp5_{m}_{cat}",
                              [py, "scripts/06_stream.py", "--model", m, "--category", cat]))
    steps.append(("figures", "figures", [py, "scripts/08_make_figures.py"]))
    return steps


def already_trained(name: str) -> bool:
    # train_<model>_<category>_seed<s>; model and category names may contain underscores
    for m in MODELS:
        for cat in CATEGORIES:
            prefix = f"train_{m}_{cat}_seed"
            if name.startswith(prefix):
                seed = name[len(prefix):]
                return (ROOT / "artifacts" / cat / m / f"seed{seed}" / "calibration.json").exists()
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=["train", "eval", "timing", "figures"],
                        help="run only these stages")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="laptop",
                        help="batch/worker settings: laptop (8 GB) or colab (16 GB GPU)")
    parser.add_argument("--stream-categories", nargs="+", choices=CATEGORIES, default=CATEGORIES,
                        help="categories that get the Exp 5 stream (default: all)")
    args = parser.parse_args()
    MEM_ARGS.update(PROFILES[args.profile])

    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    done_file = logs / "done.txt"
    done = set(done_file.read_text().split()) if done_file.exists() else set()
    failed = []
    for stage, name, argv in build_steps(args.stream_categories):
        if args.only and stage not in args.only:
            continue
        if name in done or (stage == "train" and already_trained(name)):
            print(f"skip   {name}")
            continue
        if args.dry_run:
            print(f"would  {name}: {' '.join(argv[1:])}")
            continue
        print(f"[{now()}] start  {name}", flush=True)
        with open(logs / f"{name}.log", "w", encoding="utf-8") as fh:
            code = subprocess.run(argv, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
        if code == 0:
            with open(done_file, "a", encoding="utf-8") as fh:
                fh.write(name + "\n")
            print(f"[{now()}] done   {name}", flush=True)
        else:
            failed.append(name)
            print(f"[{now()}] FAILED {name} (exit {code}, see logs/{name}.log)", flush=True)
    print("queue finished" + (f"; failed: {', '.join(failed)}" if failed else ""))


if __name__ == "__main__":
    main()
