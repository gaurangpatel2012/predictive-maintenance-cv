"""FastAPI request-response endpoint used for service-level measurement (Exp 6).

  uvicorn app.api:app --host 0.0.0.0 --port 8000

The dashboard talks over a persistent WebSocket that load-testing tools cannot
drive, so the service figures describe this inference path, not the UI.
One model is resident at a time; requesting another releases the previous one,
and the response reports the load time that switch cost.

  POST /predict   multipart: file=<image>, model=<name>[, category=<name>]
                  -> {score, decision, latency_ms, model_load_ms, ...}
  GET  /health    -> {"status": "ok"}
  GET  /models    -> deployed categories and models
"""
from __future__ import annotations

import gc
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402

from src.serving import available, load_pipeline  # noqa: E402

DEFAULT_CATEGORY = os.environ.get("PMCV_CATEGORY", "sheet_metal")


class ModelSlot:
    """Holds exactly one loaded pipeline; switching releases the previous one first."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.key: tuple[str, str] | None = None
        self.pipeline = None

    def get(self, category: str, model: str):
        with self.lock:
            if self.key == (category, model):
                return self.pipeline, 0.0
            self.pipeline, self.key = None, None
            gc.collect()
            t0 = time.perf_counter()
            self.pipeline = load_pipeline(category, model)
            self.key = (category, model)
            return self.pipeline, (time.perf_counter() - t0) * 1e3


app = FastAPI(title="Predictive maintenance inspection API")
slot = ModelSlot()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/models")
def models() -> dict:
    return {"models": available(), "resident": slot.key}


@app.post("/predict")
def predict(file: UploadFile = File(...), model: str = Form(...), category: str = Form(DEFAULT_CATEGORY)) -> dict:
    t0 = time.perf_counter()
    if model not in available().get(category, []):
        raise HTTPException(404, f"model '{model}' for category '{category}' is not deployed")
    data = file.file.read()
    pipe, load_ms = slot.get(category, model)
    try:
        r = pipe(data)
    except ValueError as exc:  # input validation failure is a service-level outcome, not a crash
        raise HTTPException(422, str(exc)) from exc
    return {
        "model": model, "category": category, "score": r.score, "decision": int(r.decision),
        "threshold": pipe.threshold, "latency_ms": r.t_total_ms, "t_pre_ms": r.t_pre_ms,
        "t_infer_ms": r.t_infer_ms, "t_post_ms": r.t_post_ms, "model_load_ms": load_ms,
        "server_ms": (time.perf_counter() - t0) * 1e3,
    }
