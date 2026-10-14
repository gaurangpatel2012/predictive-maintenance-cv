"""Render every thesis figure and the Chapter 5 summary tables from results/*.csv.

  python scripts/08_make_figures.py                    # everything whose inputs exist
  python scripts/08_make_figures.py --only fig5_3 tables

Figures (PNG 300 dpi + PDF) go to figures/; the first category in the config gets the
plain name, others a _<category> suffix. Every figure except fig5_5 is drawn from the
results files alone; fig5_5 loads the CAE and PatchCore models to render anomaly maps.

  degradation_examples  one evaluation image under every degradation level
  fig5_1      F1 and AUROC per model: every seed, mean and SD     <- exp1_predictive.csv
  fig5_2      latency distribution and median decomposition        <- exp4_latency_*.csv
  fig5_3      F1 and false-positive rate vs severity, per family   <- exp2_reliability.csv
  fig5_scores every image's score against the nominal threshold    <- exp5_stream.csv
  fig5_light  false-positive and miss rate, nominal vs shifted     <- exp2_reliability.csv
  fig5_4      latency and memory over the 1 Hz stream              <- exp5_stream.csv
  fig5_window defect-rate alarm: false-alarm rate per window/level <- exp5_window_sweep.csv
  fig5_5      image, ground truth, CAE and PatchCore maps          (loads models)
  fig5_6      service availability and latency vs concurrency     <- exp6_service.csv
  tables      results/table5_{1,2,3}_*.csv, mean +- SD strings

Style: one categorical colour per model in a fixed order (validated for colour-vision
deficiency), backed by a distinct marker and dash pattern per model so every figure
also reads in greyscale; text in ink colours, never series colours.
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from src.config import base_parser, setup  # noqa: E402
from src.data.dataset import InspectionDataset, load_working_image, load_working_mask  # noqa: E402
from src.data.degradations import apply_degradation  # noqa: E402
from src.data.splits import load_splits, partitions  # noqa: E402
from src.models.registry import DISPLAY_NAMES, artifact_dir, load_calibration, load_model  # noqa: E402

# ---------------------------------------------------------------------------- style
COLOR = {"patchcore": "#2a78d6", "cae": "#eb6834", "resnet50": "#1baf7a",
         "efficientnet_b0": "#eda100", "vit": "#e87ba4"}
MARKER = {"patchcore": "o", "cae": "s", "resnet50": "^", "efficientnet_b0": "D", "vit": "v"}
DASH = {"patchcore": "-", "cae": (0, (4, 1.6)), "resnet50": (0, (6, 1.5, 1.2, 1.5)),
        "efficientnet_b0": (0, (1.2, 1.4)), "vit": (0, (8, 2))}
SHORT = {"patchcore": "PatchCore", "cae": "CAE", "resnet50": "ResNet-50",
         "efficientnet_b0": "EfficientNet-B0", "vit": "ViT"}
PARADIGM = {"patchcore": "one-class", "cae": "one-class", "resnet50": "supervised",
            "efficientnet_b0": "supervised", "vit": "supervised"}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#ffffff"
FAMILY_TITLES = {"gaussian_blur": "Gaussian blur (σ, px)", "gaussian_noise": "Gaussian noise (σ, 0–255)",
                 "jpeg": "JPEG compression (quality)", "downscale": "Downscaling (factor)"}
SHORT_FAMILY = {"gaussian_blur": "Blur  σ", "gaussian_noise": "Noise  σ", "jpeg": "JPEG  quality",
                "downscale": "Downscale  ×"}
CAT_TITLE = {"sheet_metal": "Sheet metal", "can": "Can"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 7.5, "axes.titlesize": 7.8, "axes.titleweight": "bold", "axes.labelsize": 7.3,
    "legend.fontsize": 7, "xtick.labelsize": 6.9, "ytick.labelsize": 6.9,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK2, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 2, "ytick.major.pad": 2,
    "axes.labelpad": 2.5, "axes.titlepad": 4, "legend.handletextpad": 0.5, "legend.columnspacing": 1.2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE, "legend.frameon": False, "lines.solid_capstyle": "round",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
})
FULL_W = 6.3  # inches: the thesis text width (15 cm)


def model_handles(models, line: bool = True) -> list[Line2D]:
    return [Line2D([], [], color=COLOR[m], marker=MARKER[m], linestyle=DASH[m] if line else "none",
                   linewidth=1.6, markersize=4.6, markeredgecolor=SURFACE, markeredgewidth=0.6, label=SHORT[m])
            for m in models]


def save(fig, figures: Path, name: str) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(figures / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print(f"  wrote {figures / name}.{{png,pdf}}")


def suffix(cfg, cat: str) -> str:
    return "" if cat == cfg.categories[0] else f"_{cat}"


def read(cfg, name: str) -> pd.DataFrame | None:
    p = cfg.path("results") / name
    if not p.exists():
        print(f"  skip: {p.name} not found")
        return None
    return pd.read_csv(p)


ORDER = ["patchcore", "cae", "resnet50", "efficientnet_b0", "vit"]


def ordered(models) -> list[str]:
    return [m for m in ORDER if m in set(models)]


def categories(cfg, df: pd.DataFrame) -> list[str]:
    return [c for c in cfg.categories if c in set(df["category"])]


def pm(mean: float, sd: float, digits: int = 3) -> str:
    if not np.isfinite(mean):
        return "--"
    return f"{mean:.{digits}f} ± {sd:.{digits}f}" if np.isfinite(sd) else f"{mean:.{digits}f}"


def model_axis(ax, models: list[str]) -> np.ndarray:
    """Horizontal model axis, one-class models on top, a gap before the supervised group."""
    y = np.array([i + (0.6 if PARADIGM[m] == "supervised" else 0) for i, m in enumerate(models)])
    ax.set_yticks(y, [SHORT[m] for m in models])
    ax.set_ylim(y.max() + 0.6, y.min() - 0.6)
    ax.grid(axis="y", visible=False)
    return y


def degradation_rows(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """(baseline rows, degradation rows, label) - the whole-eval-portion grid when present."""
    if (d["condition_type"] == "degradation_all").any():
        return (d[d["condition_type"] == "baseline_all"], d[d["condition_type"] == "degradation_all"],
                "whole evaluation portion")
    return d[d["condition_type"] == "baseline"], d[d["condition_type"] == "degradation"], "nominal lighting"


# ---------------------------------------------------------------------------- figures
def degradation_examples(cfg, figures: Path) -> None:
    """Two rows of two families: original + five severities each, as a compact strip."""
    cat = cfg.categories[0]
    try:
        table = partitions(load_splits(cat, cfg), "patchcore")["eval_nominal"]
    except SystemExit as exc:
        print(f"  skip: {exc}")
        return
    rows = table[table["label"] == 1]
    rel = (rows if len(rows) else table)["filename"].iloc[0]
    img = load_working_image(cfg, cat, rel)
    h = img.shape[0]
    crop = slice(h // 4, h - h // 4)  # central half of the height: wide strips, detail stays visible
    fams = list(cfg.degradations.data.items())
    n_lv = max(len(s["levels"]) for _, s in fams)
    sym = {"sigma": "σ ", "quality": "q ", "factor": ""}
    fig, axes = plt.subplots(len(fams), n_lv + 1, figsize=(FULL_W, 2.2),
                             gridspec_kw={"wspace": 0.03, "hspace": 0.05})
    for r, (kind, spec) in enumerate(fams):
        for c in range(n_lv + 1):
            ax = axes[r, c]
            shown = img if c == 0 else apply_degradation(img, kind, float(spec["levels"][c - 1]))
            ax.imshow(shown[crop], aspect="auto")
            ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if c == 0:
                ax.set_ylabel(SHORT_FAMILY[kind].split()[0], fontsize=7.0, labelpad=3)
            lab = "original" if c == 0 else (f"{sym[spec['param']]}{spec['levels'][c - 1]:g}"
                                             + ("×" if spec["param"] == "factor" else ""))
            ax.text(0.04, 0.9, lab, transform=ax.transAxes, ha="left", va="top", fontsize=6.3, color=SURFACE,
                    bbox=dict(boxstyle="round,pad=0.18", facecolor=INK, alpha=0.62, linewidth=0))
    save(fig, figures, "degradation_examples")


def _dot_panel(ax, d: pd.DataFrame, models: list[str], metric: str, y: np.ndarray) -> None:
    for yi, m in zip(y, models):
        v = d.loc[d["model"] == m, metric].dropna().to_numpy()
        if not len(v):
            continue
        mu, sd = v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)
        ax.plot([max(0, mu - sd), min(1, mu + sd)], [yi, yi], color=COLOR[m], linewidth=2.4, alpha=0.4)
        ax.scatter(v, yi + np.linspace(-0.17, 0.17, len(v)), s=11, facecolor=SURFACE, edgecolor=COLOR[m],
                   linewidth=0.8, zorder=3)
        ax.scatter([mu], [yi], s=38, marker=MARKER[m], color=COLOR[m], edgecolor=SURFACE, linewidth=0.6, zorder=4)
    ax.set_xlim(-0.04, 1.04)
    ax.set_xticks([0, 0.5, 1], ["0", ".5", "1"])


def fig5_1(cfg, figures: Path) -> None:
    """One row: F1 and AUROC for each category; every seed, mean and +-1 SD."""
    df = read(cfg, "exp1_predictive.csv")
    if df is None:
        return
    cats = categories(cfg, df)
    fig, axes = plt.subplots(1, 2 * len(cats), figsize=(FULL_W, 1.85), sharey=True,
                             gridspec_kw={"wspace": 0.12})
    for k, cat in enumerate(cats):
        d = df[df["category"] == cat]
        models = ordered(d["model"].unique())
        n, n_anom = int(d["n_images"].iloc[0]), int(d["n_anomalous"].iloc[0])
        for j, (metric, label) in enumerate([("f1", "F1"), ("auroc", "AUROC")]):
            ax = axes[2 * k + j]
            y = model_axis(ax, models)
            if metric == "auroc":
                ax.axvline(0.5, color=MUTED, linestyle=(0, (3, 2)), linewidth=0.8)
            _dot_panel(ax, d, models, metric, y)
            ax.set_title(f"{label}", loc="left", fontsize=7.2)
            ax.set_xlabel(label, fontsize=6.7)
        axes[2 * k].annotate(f"{CAT_TITLE.get(cat, cat)} · {n} images ({n - n_anom} normal)",
                             xy=(0, 1), xycoords="axes fraction", xytext=(0, 15), textcoords="offset points",
                             fontsize=6.7, color=INK2)
    fig.legend(handles=[Line2D([], [], marker="o", linestyle="none", markerfacecolor=SURFACE, markeredgecolor=INK2,
                               markersize=4, label="single seed"),
                        Line2D([], [], marker="o", linestyle="none", color=INK2, markersize=5.5, label="mean of 3 seeds"),
                        Line2D([], [], color=INK2, alpha=0.4, linewidth=2.4, label="± 1 SD"),
                        Line2D([], [], color=MUTED, linestyle=(0, (3, 2)), label="AUROC chance level")],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0), fontsize=6.4)
    fig.subplots_adjust(bottom=0.33, top=0.8)
    save(fig, figures, "fig5_1_performance")


def fig5_2(cfg, figures: Path) -> None:
    """Left: distribution of end-to-end latency. Right: median split into pre / infer / post."""
    raw, summ = read(cfg, "exp4_latency_raw.csv"), read(cfg, "exp4_latency_summary.csv")
    if raw is None:
        return
    budget = float(cfg.realtime.latency_budget_ms)
    for cat in categories(cfg, raw):
        d = raw[raw["category"] == cat]
        models = ordered(d["model"].unique())
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL_W, 1.8), sharey=True,
                                     gridspec_kw={"width_ratios": [1, 1.05], "wspace": 0.1})
        y = model_axis(a1, models)
        for yi, m in zip(y, models):
            t = d.loc[d["model"] == m, "t_total_ms"].to_numpy()
            p1, p25, p50, p75, p99 = np.percentile(t, [1, 25, 50, 75, 99])
            a1.plot([p1, p99], [yi, yi], color=COLOR[m], linewidth=1.1)
            a1.add_patch(plt.Rectangle((p25, yi - 0.24), p75 - p25, 0.48, facecolor=COLOR[m], alpha=0.28,
                                       edgecolor=COLOR[m], linewidth=0.8))
            a1.plot([p50, p50], [yi - 0.24, yi + 0.24], color=INK, linewidth=1.3)
            a1.scatter([np.max(t)], [yi], s=12, color=COLOR[m], marker="|", linewidth=1)
        hi = float(d["t_total_ms"].max()) * 1.08
        a1.set_xlim(0, hi)
        a1.set_xticks([t for t in range(0, int(hi), 50)])
        a1.set_xlabel("end-to-end latency (ms)")
        a1.set_title("Distribution per image", loc="left", fontsize=7.2)
        comp = d.groupby("model")[["t_pre_ms", "t_infer_ms", "t_post_ms"]].median()
        for yi, m in zip(y, models):
            pre, inf, post = comp.loc[m]
            a2.barh(yi, pre, height=0.52, color="#c9c8c3", edgecolor=SURFACE, linewidth=1)
            a2.barh(yi, inf, left=pre, height=0.52, color=COLOR[m], edgecolor=SURFACE, linewidth=1)
            a2.barh(yi, post, left=pre + inf, height=0.52, color=INK2, edgecolor=SURFACE, linewidth=1)
            tot = pre + inf + post
            a2.text(tot + 3, yi, f"{tot:.0f} ms · {pre / tot:.0%} pre", va="center", fontsize=6.2, color=INK)
        a2.set_xlim(0, comp.sum(axis=1).max() * 1.62)
        a2.set_xlabel("median time per stage (ms)")
        a2.set_title("Where the median time goes", loc="left", fontsize=7.2)
        a2.grid(axis="y", visible=False)
        miss = (d["t_total_ms"] > budget).mean()
        fig.legend(handles=[Patch(facecolor=MUTED, alpha=0.35, edgecolor=MUTED, label="25th–75th pct."),
                            Line2D([], [], color=INK, label="median"),
                            Line2D([], [], color=MUTED, label="1st–99th pct. (| = max)"),
                            Patch(color="#c9c8c3", label="preprocessing"),
                            Patch(color=MUTED, label="inference (model colour)"),
                            Patch(color=INK2, label="post-processing")],
                   loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=6, fontsize=6.0, handlelength=1.1,
                   columnspacing=0.8, handletextpad=0.4)
        dev = summ.loc[summ["category"] == cat, "device_name"].iloc[0].split(" /")[0] if summ is not None else ""
        fig.suptitle(f"{CAT_TITLE.get(cat, cat)} · {dev} · batch size 1 · 3 × 500 images per model · "
                     f"{miss:.0%} over the {budget:.0f} ms budget (off scale)", x=0.01, ha="left", fontsize=6.6,
                     color=INK2, y=1.03)
        fig.subplots_adjust(bottom=0.34)
        save(fig, figures, f"fig5_2_latency{suffix(cfg, cat)}")


def fig5_3(cfg, figures: Path) -> None:
    """Rows: F1, false-positive rate. Columns: degradation families. Line per model, band = SD."""
    df = read(cfg, "exp2_reliability.csv")
    if df is None:
        return
    for cat in categories(cfg, df):
        base, deg, label = degradation_rows(df[df["category"] == cat])
        models = ordered(deg["model"].unique())
        fams = list(cfg.degradations.data.items())
        fig, axes = plt.subplots(2, len(fams), figsize=(FULL_W, 2.75), sharey="row", sharex="col",
                                 gridspec_kw={"hspace": 0.16, "wspace": 0.07})
        n = int(base["n_images"].iloc[0]) if len(base) else 0
        for c, (kind, spec) in enumerate(fams):
            levels = [float(v) for v in spec["levels"]]
            xs = np.arange(len(levels) + 1)
            for r, (metric, ylab) in enumerate([("f1", "F1"), ("fpr_at_fixed", "false-pos. rate")]):
                ax = axes[r, c]
                ax.axvspan(-0.35, 0.35, color=GRID, alpha=0.6, linewidth=0)
                for m in models:
                    parts = [base[base["model"] == m].assign(level=0.0), deg[(deg["condition"] == kind) & (deg["model"] == m)]]
                    g = pd.concat(parts).groupby("level")[metric].agg(["mean", "std"]).reindex([0.0] + levels)
                    sd = g["std"].fillna(0).to_numpy()
                    ax.fill_between(xs, (g["mean"] - sd).clip(0, 1), (g["mean"] + sd).clip(0, 1),
                                    color=COLOR[m], alpha=0.09, linewidth=0)
                    ax.plot(xs, g["mean"], color=COLOR[m], linestyle=DASH[m], linewidth=1.3, marker=MARKER[m],
                            markersize=3.4, markeredgecolor=SURFACE, markeredgewidth=0.4)
                ax.set_xticks(xs, ["0"] + [f"{v:g}" for v in levels], fontsize=6.2)
                ax.set_ylim(-0.04, 1.04)
                ax.set_yticks([0, 0.5, 1], ["0", ".5", "1"])
                if c == 0:
                    ax.set_ylabel(ylab, fontsize=7.0)
                if r == 0:
                    ax.set_title(SHORT_FAMILY[kind], fontsize=7.2, loc="left")
        fig.legend(handles=model_handles(models), loc="lower center", ncol=len(models), bbox_to_anchor=(0.5, 0.0),
                   handlelength=2.4, fontsize=6.6)
        fig.suptitle(f"{CAT_TITLE.get(cat, cat)} · {label}, {n} images · fixed nominal threshold · "
                     f"mean of 3 seeds, band ±1 SD · grey = undegraded",
                     x=0.01, ha="left", fontsize=6.4, color=INK2, y=0.995)
        fig.subplots_adjust(bottom=0.2, top=0.88)
        print(f"  {cat}: {label}")
        save(fig, figures, f"fig5_3_degradation{suffix(cfg, cat)}")


def fig5_scores(cfg, figures: Path) -> None:
    """One row: every evaluation image's anomaly score against the stored nominal threshold (seed 0)."""
    st = read(cfg, "exp5_stream.csv")
    if st is None:
        return
    for cat in categories(cfg, st):
        splits = load_splits(cat, cfg, include_official=False).set_index("filename")
        d = st[(st["category"] == cat) & (st["repeat"] == 0)].drop_duplicates(["model", "filename"])
        d = d.join(splits[["lighting"]], on="filename")
        models = ordered(d["model"].unique())
        fig, axes = plt.subplots(1, len(models), figsize=(FULL_W, 1.9), gridspec_kw={"wspace": 0.42})
        groups = [("nominal", 0), ("shifted", 0), ("nominal", 1), ("shifted", 1)]
        rng = np.random.default_rng(0)
        for ax, m in zip(np.atleast_1d(axes), models):
            thr = load_calibration(artifact_dir(cfg, cat, m, int(cfg.train.seeds[0])))["threshold"]
            dm = d[d["model"] == m]
            for i, (light, lab) in enumerate(groups):
                v = dm[(dm["lighting"] == light) & (dm["label"] == lab)]["score"].to_numpy()
                x = i + rng.uniform(-0.18, 0.18, len(v))
                if lab == 0:
                    ax.scatter(x, v, s=12, facecolor=SURFACE, edgecolor=INK2, linewidth=0.8, zorder=3)
                else:
                    ax.scatter(x, v, s=12, color=COLOR[m], edgecolor=SURFACE, linewidth=0.3, zorder=3)
                if len(v):
                    ax.text(i, 1.02, f"{(v >= thr).mean() * 100:.0f}", transform=ax.get_xaxis_transform(),
                            ha="center", fontsize=6.0, color=INK2)
            ax.axhline(thr, color=INK, linewidth=0.9, linestyle=(0, (4, 2)))
            ax.axvline(1.5, color=GRID, linewidth=0.8)
            ax.set_xticks(range(4), ["N", "S", "N", "S"], fontsize=6.0)
            ax.set_xlim(-0.6, 3.6)
            ax.grid(axis="x", visible=False)
            ax.locator_params(axis="y", nbins=4)
            ax.tick_params(axis="y", labelsize=6.6, pad=1.5)
            ax.set_title(SHORT[m], fontsize=7.0, pad=12)
            ax.text(0.27, -0.22, "normal", transform=ax.transAxes, ha="center", fontsize=6.0, color=INK2)
            ax.text(0.77, -0.22, "defect", transform=ax.transAxes, ha="center", fontsize=6.0, color=INK2)
        np.atleast_1d(axes)[0].set_ylabel("anomaly score", fontsize=6.7)
        fig.legend(handles=[Line2D([], [], marker="o", linestyle="none", markerfacecolor=SURFACE, markeredgecolor=INK2,
                                   markersize=4.5, label="normal part"),
                            Line2D([], [], marker="o", linestyle="none", color=MUTED, markersize=4.5,
                                   label="defective part (model colour)"),
                            Line2D([], [], color=INK, linestyle=(0, (4, 2)), label="threshold fixed on nominal data")],
                   loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0), fontsize=6.4)
        fig.suptitle(f"{CAT_TITLE.get(cat, cat)} · evaluation portion, seed 0 · N nominal, S shifted lighting · "
                     "number above each column = % flagged as defective", x=0.01, ha="left", fontsize=6.4,
                     color=INK2, y=1.04)
        fig.subplots_adjust(bottom=0.33, top=0.8)
        save(fig, figures, f"fig5_scores{suffix(cfg, cat)}")


def fig5_light(cfg, figures: Path) -> None:
    """One row of dumbbells: false-positive and miss rate, nominal -> shifted, per category."""
    df = read(cfg, "exp2_reliability.csv")
    if df is None:
        return
    cats = categories(cfg, df)
    fig, axes = plt.subplots(1, 2 * len(cats), figsize=(FULL_W, 1.85), sharey=True, gridspec_kw={"wspace": 0.12})
    for k, cat in enumerate(cats):
        d = df[df["category"] == cat]
        nom, sh = d[d["condition_type"] == "baseline"], d[d["condition_type"] == "lighting"]
        models = ordered(sh["model"].unique())
        for j, (metric, label) in enumerate([("fpr_at_fixed", "false alarms"), ("fnr_at_fixed", "missed defects")]):
            ax = axes[2 * k + j]
            y = model_axis(ax, models)
            for yi, m in zip(y, models):
                a = nom.loc[nom["model"] == m, metric].mean()
                b = sh.loc[sh["model"] == m, metric].mean()
                ax.annotate("", xy=(b, yi), xytext=(a, yi),
                            arrowprops=dict(arrowstyle="-|>", color=COLOR[m], lw=1.3, shrinkA=2.5, shrinkB=2.5,
                                            mutation_scale=7))
                ax.scatter([a], [yi], s=24, facecolor=SURFACE, edgecolor=COLOR[m], linewidth=1.1, zorder=4)
                ax.scatter([b], [yi], s=30, marker=MARKER[m], color=COLOR[m], edgecolor=SURFACE, linewidth=0.5, zorder=5)
            ax.set_xlim(-0.05, 1.05)
            ax.set_xticks([0, 0.5, 1], ["0", ".5", "1"])
            ax.set_title(label, loc="left", fontsize=7.2)
            ax.set_xlabel("FPR" if metric == "fpr_at_fixed" else "FNR", fontsize=6.7)
        n_nom, n_sh = int(nom["n_images"].iloc[0]), int(sh["n_images"].iloc[0])
        axes[2 * k].annotate(f"{CAT_TITLE.get(cat, cat)} · {n_nom} nominal → {n_sh} shifted images",
                             xy=(0, 1), xycoords="axes fraction", xytext=(0, 15), textcoords="offset points",
                             fontsize=6.7, color=INK2)
    fig.legend(handles=[Line2D([], [], marker="o", linestyle="none", markerfacecolor=SURFACE, markeredgecolor=INK2,
                               markersize=5, label="nominal lighting"),
                        Line2D([], [], marker="o", linestyle="none", color=INK2, markersize=5, label="shifted lighting"),
                        Line2D([], [], color=INK2, label="rate at the fixed nominal threshold, mean of 3 seeds")],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0), fontsize=6.4)
    fig.subplots_adjust(bottom=0.33, top=0.8)
    save(fig, figures, "fig5_light")


def fig5_4(cfg, figures: Path) -> None:
    """Left: latency over the 1 Hz stream (rolling median). Right: change in resident memory."""
    df = read(cfg, "exp5_stream.csv")
    if df is None:
        return
    budget = float(cfg.realtime.latency_budget_ms)
    for cat in categories(cfg, df):
        d = df[df["category"] == cat].copy()
        n_per = int(d["idx"].max()) + 1
        d["t"] = d["repeat"] * n_per + d["idx"]
        models = ordered(d["model"].unique())
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL_W, 1.85), gridspec_kw={"width_ratios": [1.7, 1], "wspace": 0.28})
        for m in models:
            dm = d[d["model"] == m].sort_values("t")
            a1.scatter(dm["t"], dm["e2e_ms"], s=0.8, color=COLOR[m], alpha=0.18, linewidth=0, rasterized=True)
            a1.plot(dm["t"], dm["e2e_ms"].rolling(25, center=True, min_periods=5).median(), color=COLOR[m],
                    linestyle=DASH[m], linewidth=1.3)
            ref = dm["rss_mb"].iloc[5:15].median()
            tail = dm.iloc[5:]
            a2.plot(tail["t"], tail["rss_mb"] - ref, color=COLOR[m], linestyle=DASH[m], linewidth=1.2)
        for a in (a1, a2):
            for k in range(1, int(d["repeat"].max()) + 1):
                a.axvline(k * n_per, color=MUTED, linewidth=0.7, linestyle=(0, (2, 2)))
            a.set_xlabel("image index (2 × 500)", fontsize=6.7)
        mx = d["e2e_ms"].max()
        n_miss = int((d["e2e_ms"] > budget).sum())
        a1.set_ylim(0, mx * 1.12)
        a1.set_ylabel("latency (ms)", fontsize=6.7)
        a1.set_title("End-to-end latency, rolling median of 25", loc="left", fontsize=7.2)
        a2.set_ylabel("Δ memory (MiB)", fontsize=6.7)
        a2.set_title("Resident memory vs images 5–15", loc="left", fontsize=7.2)
        a2.axhline(0, color=INK2, linewidth=0.5)
        fig.legend(handles=model_handles(models), ncol=len(models), loc="lower center", bbox_to_anchor=(0.5, 0.0),
                   handlelength=2.4, fontsize=6.6)
        fig.suptitle(f"{CAT_TITLE.get(cat, cat)} · 1 image/s · dots = single images · max {mx:.0f} ms · "
                     f"{n_miss} of {len(d):,} over the {budget:.0f} ms budget (off scale)",
                     x=0.01, ha="left", fontsize=6.6, color=INK2, y=1.02)
        fig.subplots_adjust(bottom=0.34)
        save(fig, figures, f"fig5_4_stream{suffix(cfg, cat)}")


def fig5_window(cfg, figures: Path) -> None:
    """Heatmaps of the false-alarm rate of the defect-rate indication, per model, in one row."""
    df = read(cfg, "exp5_window_sweep.csv")
    if df is None:
        return
    for cat in categories(cfg, df):
        d = df[df["category"] == cat]
        models = ordered(d["model"].unique())
        fig, axes = plt.subplots(1, len(models), figsize=(FULL_W, 1.45), gridspec_kw={"wspace": 0.08})
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blues", ["#f2f6fc", "#9cc0ee", "#2a78d6", "#123f7a"])
        for k, (ax, m) in enumerate(zip(np.atleast_1d(axes), models)):
            p = d[d["model"] == m].pivot(index="window_W", columns="rate_threshold", values="false_alarm_rate")
            ax.imshow(p.to_numpy(), cmap=cmap, vmin=0, vmax=1, aspect="auto")
            for i in range(p.shape[0]):
                for j in range(p.shape[1]):
                    v = p.iat[i, j]
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.0,
                            color=SURFACE if v > 0.55 else INK)
            ax.set_xticks(range(p.shape[1]), [f"{c:g}" for c in p.columns], fontsize=6.0)
            ax.set_yticks(range(p.shape[0]), [str(i) for i in p.index] if k == 0 else [], fontsize=6.0)
            ax.tick_params(length=0)
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title(SHORT[m], fontsize=7.0, pad=3)
            ax.set_xlabel("alarm level", fontsize=6.0, labelpad=1)
        np.atleast_1d(axes)[0].set_ylabel("window W", fontsize=6.2)
        delay = d["detection_delay_images"]
        fig.suptitle(f"{CAT_TITLE.get(cat, cat)} · false-alarm rate of the defect-rate indication before a prevalence "
                     f"step 0.1 → 0.6 · detection delay {delay.min():.0f}–{delay.max():.0f} images",
                     x=0.01, ha="left", fontsize=6.4, color=INK2, y=1.08)
        save(fig, figures, f"fig5_window{suffix(cfg, cat)}")


def fig5_5(cfg, figures: Path, device_pref: str) -> None:
    from src.evaluation import score_dataset
    from src.runtime import empty_cache, resolve_device

    device = resolve_device(device_pref)
    cat = cfg.categories[0]
    seed = int(cfg.train.seeds[0])
    try:
        table = partitions(load_splits(cat, cfg), "patchcore")["eval"]
        dirs = {m: artifact_dir(cfg, cat, m, seed) for m in ("patchcore", "cae")}
        calib = load_calibration(dirs["patchcore"])
    except SystemExit as exc:
        print(f"  skip: {exc}")
        return
    ds = InspectionDataset.from_table(cfg, cat, table)
    maps = {}
    for m in ("patchcore", "cae"):  # one model in memory at a time
        model = load_model(m, dirs[m] / "model.pt", device)
        maps[m] = score_dataset(model, ds, cfg, with_maps=True)
        del model
        gc.collect()
        empty_cache(device)
    pc = maps["patchcore"]
    pred = pc.scores >= calib["threshold"]
    picks = []
    for title, mask, pick in [("true positive", pred & (pc.labels == 1), np.argmax),
                              ("false positive", pred & (pc.labels == 0), np.argmax),
                              ("missed defect", ~pred & (pc.labels == 1), np.argmin)]:
        idx = np.flatnonzero(mask)
        if idx.size:
            picks.append((title, idx[pick(pc.scores[idx])]))
        else:
            print(f"  note: PatchCore has no {title} on the evaluation portion - row omitted")
    if not picks:
        return
    # one row: per case, the image with its ground-truth outline, then the two anomaly maps
    fig = plt.figure(figsize=(FULL_W, 1.32))
    outer = fig.add_gridspec(1, len(picks), left=0.005, right=0.995, top=0.775, bottom=0.11, wspace=0.07)
    for k, (title, i) in enumerate(picks):
        rel = pc.filenames[i]
        img = load_working_image(cfg, cat, rel)
        mp = table.loc[table["filename"] == rel, "mask_path"].iloc[0]
        gt = load_working_mask(cfg, cat, mp) if mp else np.zeros(img.shape[:2], np.uint8)
        light = table.loc[table["filename"] == rel, "lighting"].iloc[0]
        inner = outer[0, k].subgridspec(1, 3, wspace=0.03)
        panels = [("image", None), ("CAE", maps["cae"].maps[i]), ("PatchCore", pc.maps[i])]
        for c, (name, arr) in enumerate(panels):
            ax = fig.add_subplot(inner[0, c])
            ax.imshow(img)
            if arr is not None:
                arr = arr.astype(np.float32)
                lo, hi = np.percentile(arr, [1, 99.7])
                ax.imshow(np.clip((arr - lo) / max(hi - lo, 1e-9), 0, 1), cmap="inferno", alpha=0.62)
            if gt.any():
                ax.contour(gt, levels=[0.5], colors=["#00e5ff"], linewidths=0.6)
            ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title(name, fontsize=6.3, fontweight="normal", color=INK2, pad=2)
            if c == 1:
                ax.text(0.5, 1.22, f"{title} ({light} light, score {pc.scores[i]:.2f})", transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=6.8, fontweight="bold")
    fig.text(0.005, 0.015, f"Cyan: ground-truth defect. Maps scaled per image (1st–99.7th pct.). PatchCore decisions, "
             f"seed {seed}, threshold {calib['threshold']:.2f}; {cat.replace('_', ' ')}.", fontsize=6.0, color=INK2)
    save(fig, figures, "fig5_5_examples")


def fig5_6(cfg, figures: Path) -> None:
    df = read(cfg, "exp6_service.csv")
    if df is None:
        return
    d = df[df["notes"].fillna("").str.startswith("model=")].copy()
    if d.empty:
        print("  skip: no concurrency rows in exp6_service.csv")
        return
    g = d.groupby("concurrency")[["availability", "median_ms", "p95_ms"]].mean()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL_W, 2.4), gridspec_kw={"wspace": 0.3})
    a1.plot(g.index, g["median_ms"], color=COLOR["patchcore"], marker="o", label="median")
    a1.plot(g.index, g["p95_ms"], color=COLOR["cae"], marker="s", linestyle=(0, (4, 1.6)), label="95th percentile")
    a1.set_xlabel("concurrent clients"), a1.set_ylabel("response time (ms)")
    a1.set_xticks(g.index), a1.legend()
    a2.bar(g.index.astype(str), g["availability"], color=COLOR["patchcore"], width=0.55)
    a2.set_ylim(0, 1.05), a2.set_xlabel("concurrent clients"), a2.set_ylabel("availability")
    save(fig, figures, "fig5_6_service")


# ---------------------------------------------------------------------------- tables
def tables(cfg) -> None:
    res = cfg.path("results")
    e1, e2 = read(cfg, "exp1_predictive.csv"), read(cfg, "exp2_reliability.csv")
    e4, e5 = read(cfg, "exp4_latency_summary.csv"), read(cfg, "exp5_stream.csv")
    budget = float(cfg.realtime.latency_budget_ms)
    if e1 is not None:
        rows = []
        for (cat, m), g in e1.groupby(["category", "model"]):
            row = {"category": cat, "model": DISPLAY_NAMES[m], "n_seeds": len(g)}
            for c in ("precision", "recall", "f1", "auroc", "aupro_005", "aupro_030", "threshold"):
                row[c] = pm(g[c].mean(), g[c].std(), 4 if c == "threshold" else 3)
            rows.append(row)
        pd.DataFrame(rows).to_csv(res / "table5_1_predictive.csv", index=False)
        print(f"  wrote {res / 'table5_1_predictive.csv'}")
    if e4 is not None:
        t = e4[["category", "model", "median_ms", "p95_ms", "p99_ms", "throughput_ips", "model_size_mb",
                "peak_mem_mb", "device_name"]].copy()
        t["model"] = t["model"].map(DISPLAY_NAMES)
        t.round(1).to_csv(res / "table5_2_operational.csv", index=False)
        print(f"  wrote {res / 'table5_2_operational.csv'}")
    if e1 is not None and e2 is not None:
        rows = []
        for (cat, m), g1 in e1.groupby(["category", "model"]):
            g2 = e2[(e2["category"] == cat) & (e2["model"] == m)]
            base_rows, deg_rows, _ = degradation_rows(g2)
            deg = deg_rows.groupby(["condition", "level"])["f1"].agg(["mean", "std"])
            lit = g2[g2["condition_type"] == "lighting"]
            row = {"category": cat, "model": DISPLAY_NAMES[m], "f1_nominal": pm(g1["f1"].mean(), g1["f1"].std())}
            # clean reference on the same images as the degradation grid
            row["f1_clean_grid"] = pm(base_rows["f1"].mean(), base_rows["f1"].std()) if len(base_rows) else "--"
            if len(deg):
                worst = deg["mean"].idxmin()
                row["f1_worst_degradation"] = pm(deg.loc[worst, "mean"], deg.loc[worst, "std"])
                row["worst_condition"] = f"{worst[0]}@{worst[1]:g}"
            row["f1_lighting"] = pm(lit["f1"].mean(), lit["f1"].std()) if len(lit) else "--"
            row["drift_norm_lighting"] = pm(lit["threshold_drift_norm"].mean(),
                                            lit["threshold_drift_norm"].std(), 2) if len(lit) else "--"
            dn = deg_rows.groupby(["condition", "level"])["threshold_drift_norm"].mean()
            row["max_abs_drift_norm_degradation"] = f"{dn.abs().max():.2f}" if len(dn) else "--"
            if e4 is not None:
                s4 = e4[(e4["category"] == cat) & (e4["model"] == m)]
                row["p95_ms"] = f"{s4['p95_ms'].iloc[0]:.1f}" if len(s4) else "--"
                row["size_mb"] = f"{s4['model_size_mb'].iloc[0]:.1f}" if len(s4) else "--"
            if e5 is not None:
                s5 = e5[(e5["category"] == cat) & (e5["model"] == m)]
                row["stream_miss_rate"] = f"{(s5['e2e_ms'] > budget).mean():.4f}" if len(s5) else "--"
            rows.append(row)
        pd.DataFrame(rows).to_csv(res / "table5_3_reliability.csv", index=False)
        print(f"  wrote {res / 'table5_3_reliability.csv'}")


def fig4_preprocessing(cfg, figures: Path) -> None:
    """Before -> after: one raw test image per category beside its preprocessed version."""
    from matplotlib.patches import FancyArrowPatch
    from src.data.dataset import read_rgb, resize_rgb

    size = int(cfg.image_size)
    picks = []
    for cat in cfg.categories:
        try:
            t = load_splits(cat, cfg)
        except SystemExit as exc:
            print(f"  skip: {exc}")
            return
        t = t[(t["label"] == 1) & (t["lighting"] == "nominal")]
        picks.append((cat, t["filename"].iloc[0]))
    h, arrow_w, gap_y, top, bottom = 1.12, 1.25, 0.27, 0.17, 0.03  # inches
    fig_h = top + len(picks) * h + (len(picks) - 1) * gap_y + bottom
    fig = plt.figure(figsize=(FULL_W, fig_h))
    for r, (cat, rel) in enumerate(picks):
        raw = read_rgb(cfg.data_root / cat / rel)
        work = resize_rgb(raw, size)
        H, W = raw.shape[:2]
        raw_w = min(h * W / H, FULL_W - arrow_w - h - 0.05)
        y = fig_h - top - h - r * (h + gap_y)
        x_after = FULL_W - h - 0.02
        boxes = [(raw, 0.02, raw_w, f"{cat.replace('_', ' ').capitalize()} · BEFORE: raw data, {W:,} × {H:,} px"),
                 (work, x_after, h, f"AFTER: preprocessed, {size} × {size} px")]
        for img, x, w, title in boxes:
            ax = fig.add_axes([x / FULL_W, y / fig_h, w / FULL_W, h / fig_h])
            ax.imshow(img, aspect="auto")
            ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
            for sp in ax.spines.values():
                sp.set_color(GRID)
            ax.set_title(title, fontsize=6.8, loc="left", pad=2.5, color=INK, fontweight="normal")
        x0, x1 = (0.02 + raw_w + 0.08) / FULL_W, (x_after - 0.08) / FULL_W
        ym = (y + h / 2) / fig_h
        fig.patches.append(FancyArrowPatch((x0, ym), (x1, ym), transform=fig.transFigure, arrowstyle="-|>",
                                           mutation_scale=9, linewidth=1.1, color=INK2))
        fig.text((x0 + x1) / 2, ym + 0.035, "preprocessing", ha="center", va="bottom", fontsize=6.6,
                 color=INK, fontweight="bold")
        fig.text((x0 + x1) / 2, ym - 0.035, f"1  read as RGB colour\n2  resize to {size} × {size}\n"
                 "3  normalise pixel values", ha="center", va="top", fontsize=6.0, color=INK2, linespacing=1.25)
    save(fig, figures, "fig4_preprocessing")


FIGURES = ["fig4_preprocessing", "degradation_examples", "fig5_1", "fig5_2", "fig5_3", "fig5_scores", "fig5_light", "fig5_4",
           "fig5_window", "fig5_5", "fig5_6"]


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--only", nargs="+", choices=FIGURES + ["tables"], default=FIGURES + ["tables"])
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    cfg = setup(args)
    figures = cfg.path("figures")
    for name in args.only:
        print(name)
        if name == "tables":
            tables(cfg)
        elif name == "fig5_5":
            fig5_5(cfg, figures, args.device or cfg.runtime.device)
        else:
            globals()[name](cfg, figures)


if __name__ == "__main__":
    main()
