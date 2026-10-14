"""Inspection dashboard: upload -> model -> decision, score, anomaly map, response time,
and the running defect-rate indication over recent predictions.

  streamlit run app/streamlit_app.py

Weights are loaded lazily and one model is resident at a time
(st.cache_resource with max_entries=1 evicts the previous model on switch).
"""
from __future__ import annotations

import gc
import os
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

from src.models.registry import DISPLAY_NAMES  # noqa: E402
from src.runtime import rss_mb  # noqa: E402
from src.serving import available, load_pipeline  # noqa: E402

DEMO_DIR = Path(os.environ.get("PMCV_DEMO_DIR", Path(__file__).resolve().parent / "demo_images"))
BUFFER = 100  # predictions kept in the session
WINDOW = 25  # sliding-window length for the defect-rate indication
RATE_LIMIT = 0.3  # indication when the windowed defect rate exceeds this

st.set_page_config(page_title="Inspection Station", page_icon="🔍", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 1.2rem; max-width: 1400px;}
  .lamp {border-radius: 6px; padding: 14px 18px; font-weight: 700; font-size: 1.5rem;
         letter-spacing: .08em; text-align: center; color: #fff;}
  .lamp.ok {background: #1f7a3a;} .lamp.ng {background: #b3261e;} .lamp.idle {background: #555;}
  .lamp.warn {background: #b26a00;}
  .panel-title {font-size: .78rem; text-transform: uppercase; letter-spacing: .12em; color: #8a8f98;
                margin-bottom: .2rem;}
  div[data-testid="stMetricValue"] {font-variant-numeric: tabular-nums;}
</style>""", unsafe_allow_html=True)


@st.cache_resource(max_entries=1, show_spinner=False)
def get_pipeline(category: str, model: str):
    gc.collect()
    t0 = time.perf_counter()
    pipe = load_pipeline(category, model)
    return pipe, (time.perf_counter() - t0) * 1e3


def overlay(image_rgb: np.ndarray, amap: np.ndarray, threshold: float | None = None) -> np.ndarray:
    # map back onto the input's own shape, so the overlay matches the image the operator sees
    h, w = image_rgb.shape[:2]
    scale = min(1.0, 1600 / max(h, w))
    size = (max(1, int(w * scale)), max(1, int(h * scale)))
    base = cv2.resize(image_rgb, size, interpolation=cv2.INTER_AREA)
    amap = cv2.resize(amap.astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
    lo, hi = float(np.percentile(amap, 1)), float(np.percentile(amap, 99.5))
    norm = np.clip((amap - lo) / (hi - lo if hi > lo else 1.0), 0, 1)
    heat = cv2.cvtColor(cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
    return (0.55 * base + 0.45 * heat).astype(np.uint8)


def lamp(text: str, kind: str) -> None:
    st.markdown(f'<div class="lamp {kind}">{text}</div>', unsafe_allow_html=True)


deployed = available()
st.session_state.setdefault("history", deque(maxlen=BUFFER))

with st.sidebar:
    st.markdown("### Inspection station")
    if not deployed:
        st.error("No deployed models found in app/weights/. See README → Deployment.")
        st.stop()
    category = st.selectbox("Product line", list(deployed), format_func=lambda c: c.replace("_", " ").title())
    options = deployed[category]
    model = st.selectbox("Model", options, index=options.index("patchcore") if "patchcore" in options else 0,
                         format_func=lambda m: DISPLAY_NAMES.get(m, m))
    st.caption("One model is resident at a time; switching reloads weights and the load time is shown.")
    if st.button("Reset defect-rate buffer"):
        st.session_state["history"].clear()

st.markdown("## Visual inspection · predictive maintenance support")

col_in, col_out = st.columns([1.1, 1])
with col_in:
    st.markdown('<div class="panel-title">Part image</div>', unsafe_allow_html=True)
    upload = st.file_uploader("Upload an inspection image", type=["png", "jpg", "jpeg", "bmp"],
                              label_visibility="collapsed")
    demos = sorted(p for p in DEMO_DIR.glob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    demo = None
    if demos and upload is None:
        demo = st.selectbox("…or pick a demonstration image", ["—"] + [p.name for p in demos])
    data = upload.getvalue() if upload is not None else (
        (DEMO_DIR / demo).read_bytes() if demo and demo != "—" else None)

# load (and on first use download) the model only once there is an image to inspect
pipe = load_ms = None
first_use = False
if data is not None:
    with st.spinner(f"Loading {DISPLAY_NAMES.get(model, model)} (first use downloads the weights)…"):
        t_call = time.perf_counter()
        pipe, load_ms = get_pipeline(category, model)
        # the cache returns the load time of the original load; report it only when this run loaded
        fresh_load = (time.perf_counter() - t_call) * 1e3 > 0.5 * load_ms
    first_use = st.session_state.get("resident") != (category, model) and fresh_load
    st.session_state["resident"] = (category, model)

with col_out:
    st.markdown('<div class="panel-title">Decision</div>', unsafe_allow_html=True)
    if data is None:
        lamp("AWAITING PART", "idle")
    else:
        try:
            result = pipe(data)
        except ValueError as exc:
            lamp("INVALID INPUT", "warn")
            st.error(str(exc))
            st.stop()
        st.session_state["history"].append(result.decision)
        lamp("REJECT · DEFECT" if result.decision else "PASS", "ng" if result.decision else "ok")
        m1, m2, m3 = st.columns(3)
        m1.metric("Anomaly score", f"{result.score:.4g}", help=f"threshold {pipe.threshold:.4g}")
        m2.metric("Response time", f"{result.t_total_ms:.0f} ms",
                  help=f"pre {result.t_pre_ms:.0f} · infer {result.t_infer_ms:.0f} · post {result.t_post_ms:.0f} ms")
        m3.metric("Budget (1000 ms)", "met" if result.t_total_ms <= 1000 else "missed")
        rel = "≥" if result.decision else "<"
        st.markdown(f"Score **{result.score:.4g}** {rel} threshold **{pipe.threshold:.4g}** → "
                    f"**{'defect' if result.decision else 'normal part'}**")
        if first_use and load_ms:
            st.caption(f"Model loaded in {load_ms:.0f} ms (lazy load on switch).")

if data is not None:
    img = cv2.cvtColor(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    # wide line-scan images read better full width, the map directly under the input
    st.markdown('<div class="panel-title">Input image</div>', unsafe_allow_html=True)
    st.image(img, use_container_width=True)
    st.markdown('<div class="panel-title">Anomaly map · bright = least like the normal parts seen in training (scaled within this image)</div>',
                unsafe_allow_html=True)
    if result.anomaly_map is not None:
        st.image(overlay(img, result.anomaly_map), use_container_width=True)
    else:
        st.info("This model is an image-level classifier and produces no anomaly map.")

st.markdown("---")
st.markdown('<div class="panel-title">Process monitor · sliding-window defect rate</div>', unsafe_allow_html=True)
hist = list(st.session_state["history"])
window = hist[-WINDOW:]
rate = sum(window) / len(window) if window else 0.0
k1, k2, k3, k4 = st.columns(4)
k1.metric("Parts in session", len(hist))
k2.metric(f"Defect rate (last {len(window)})", f"{rate:.0%}")
k3.metric("Resident memory", f"{rss_mb():.0f} MiB")
with k4:
    if len(window) < WINDOW:
        lamp(f"FILLING {len(window)}/{WINDOW}", "idle")
    elif rate > RATE_LIMIT:
        lamp("MAINTENANCE CHECK", "ng")
    else:
        lamp("PROCESS NORMAL", "ok")
if hist:
    st.bar_chart(np.array(hist[-BUFFER:], dtype=float), height=120)
