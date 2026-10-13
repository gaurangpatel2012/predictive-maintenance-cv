"""Experiment 6 (RQ4): online service reliability, run against the deployed FastAPI endpoint.

  # cold start: first request after >= 13 h without traffic (hibernation after 12 h)
  python scripts/07_load_test.py --base-url https://<host> --image app/demo_images/x.png --cold-start
  # steady state: concurrency 1/2/4/8, 200 requests each
  python scripts/07_load_test.py --base-url https://<host> --image app/demo_images/x.png
  # model-switching cost: first request per model after switching
  python scripts/07_load_test.py --base-url https://<host> --image app/demo_images/x.png --switch-models

The idle period is not observed by this script; state it with --idle-hours so it
lands in the notes column. Rows are appended to results/exp6_service.csv.
"""
from __future__ import annotations

import asyncio
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402
import numpy as np  # noqa: E402

from src.config import base_parser, setup  # noqa: E402
from src.results_io import append_rows  # noqa: E402

COLUMNS = ["timestamp", "concurrency", "n_requests", "n_failed", "availability",
           "median_ms", "p95_ms", "throughput_rps", "cold_start_ms", "notes"]


async def request(client: httpx.AsyncClient, url: str, image: bytes, name: str, model: str,
                  category: str) -> tuple[bool, float, dict | str]:
    t0 = time.perf_counter()
    try:
        r = await client.post(url, files={"file": (name, image, "image/png")},
                              data={"model": model, "category": category})
        ok = r.status_code == 200
        body = r.json() if ok else f"HTTP {r.status_code}"
    except httpx.HTTPError as exc:
        ok, body = False, type(exc).__name__
    return ok, (time.perf_counter() - t0) * 1e3, body


async def level(client, url, image, name, model, category, concurrency: int, n: int) -> dict:
    sem = asyncio.Semaphore(concurrency)

    async def one():
        async with sem:
            return await request(client, url, image, name, model, category)

    t0 = time.perf_counter()
    res = await asyncio.gather(*(one() for _ in range(n)))
    wall = time.perf_counter() - t0
    ok = np.array([r[0] for r in res])
    ms = np.array([r[1] for r in res])[ok]
    errors = sorted({r[2] for r in res if not r[0]})
    return {"concurrency": concurrency, "n_requests": n, "n_failed": int((~ok).sum()),
            "availability": float(ok.mean()), "median_ms": float(np.median(ms)) if ms.size else float("nan"),
            "p95_ms": float(np.percentile(ms, 95)) if ms.size else float("nan"),
            "throughput_rps": float(ok.sum() / wall), "cold_start_ms": float("nan"),
            "notes": ("errors: " + ", ".join(errors)) if errors else ""}


async def run(args, cfg) -> list[dict]:
    url = args.base_url.rstrip("/") + "/predict"
    image = Path(args.image).read_bytes()
    name = Path(args.image).name
    sc = cfg.service
    now = lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")  # noqa: E731
    rows = []
    async with httpx.AsyncClient(timeout=float(sc.request_timeout_s)) as client:
        if args.cold_start:
            ok, ms, body = await request(client, url, image, name, args.model, args.category)
            note = f"cold start after {args.idle_hours} h idle (stated)" if args.idle_hours else "cold start"
            if isinstance(body, dict):
                note += f"; model_load_ms={body.get('model_load_ms', float('nan')):.0f}"
            rows.append({"timestamp": now(), "concurrency": 1, "n_requests": 1, "n_failed": int(not ok),
                         "availability": float(ok), "median_ms": ms, "p95_ms": ms, "throughput_rps": float("nan"),
                         "cold_start_ms": ms if ok else float("nan"),
                         "notes": note if ok else f"{note}; failed: {body}"})
            print(f"cold start: {ms:.0f} ms ok={ok}")
            return rows
        if args.switch_models:
            for m in args.switch_models.split(","):
                ok, ms, body = await request(client, url, image, name, m, args.category)
                load = body.get("model_load_ms", float("nan")) if isinstance(body, dict) else float("nan")
                rows.append({"timestamp": now(), "concurrency": 1, "n_requests": 1, "n_failed": int(not ok),
                             "availability": float(ok), "median_ms": ms, "p95_ms": ms,
                             "throughput_rps": float("nan"), "cold_start_ms": float("nan"),
                             "notes": f"switch:{m}; model_load_ms={load:.0f}"})
                print(f"switch to {m}: {ms:.0f} ms (server load {load:.0f} ms) ok={ok}")
            return rows
        await request(client, url, image, name, args.model, args.category)  # make the model resident
        for c in sc.concurrency_levels:
            r = await level(client, url, image, name, args.model, args.category, int(c),
                            int(sc.requests_per_level))
            r["timestamp"] = now()
            r["notes"] = (f"model={args.model}; " + r["notes"]).rstrip("; ")
            rows.append(r)
            print(f"concurrency {c}: availability {r['availability']:.3f}, median {r['median_ms']:.0f} ms, "
                  f"p95 {r['p95_ms']:.0f} ms, {r['throughput_rps']:.2f} req/s")
    return rows


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--image", required=True, help="demonstration image to send")
    parser.add_argument("--model", default="patchcore")
    parser.add_argument("--category", default="sheet_metal")
    parser.add_argument("--cold-start", action="store_true", help="send a single first request and record it")
    parser.add_argument("--idle-hours", type=float, default=None, help="stated idle time before --cold-start")
    parser.add_argument("--switch-models", default=None,
                        help="comma-separated models to request in turn, e.g. resnet50,vit,cae,patchcore")
    parser.add_argument("--output", default="exp6_service.csv",
                        help="results file; keep local-server runs apart from the hosted deployment")
    args = parser.parse_args()
    cfg = setup(args)
    rows = asyncio.run(run(args, cfg))
    append_rows(cfg.path("results") / args.output, rows, COLUMNS)


if __name__ == "__main__":
    main()
