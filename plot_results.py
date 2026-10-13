#!/usr/bin/env python
"""Figures from a mobeval leaderboard.csv: models against each other and against the baselines.

    python plot_results.py leaderboard.csv --out figures            # PNG, 200 dpi
    python plot_results.py leaderboard.csv --out figures --format pdf --against reference

Writes, numbered in reading order:

  01_overview_heatmap         every headline result at once: one row per task, one column per model,
                              colour = improvement over the baseline (blue = better, red = worse)
  02_best_model_vs_baseline   per task, the best model against the best baseline, with both values
  03_recovery_by_mask_ratio   recovery error as the masked share grows (models and baselines as lines)
  10_<family>...              one figure per task family: a panel per task, a row per model, the
                              value with its 95% interval, baselines in grey, best baseline as a line

Nothing is specific to one run: models, tasks and baselines are read from the file (baselines are
the rows whose model starts with "baseline:"). Edit HEADLINE / DETAIL below to change which metrics
are drawn, and MODEL_ORDER to fix the order and colour of the models.

"Improvement over the baseline" is the relative gain in the good direction, so that every task can
share one colour scale: 0 = same as the baseline, +0.2 = 20% better, -1 = twice the baseline's
error (or a score of zero). For errors it is (baseline - model) / baseline, for scores
(model - baseline) / baseline.
--against best (default) compares with the STRONGEST baseline of each task, computed here;
--against reference uses the leaderboard's own `skill` column instead (the pipeline's reference
baseline and its skill formulas, see EVALUATION_PROTOCOL.md).
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator

# --------------------------------------------------------------------------- configuration
# Colour follows the model, not its rank: a model keeps its colour in every figure. Models not
# listed here are appended in the order they appear in the file.
MODEL_ORDER = ["UniTraj-zeroshot", "UniTraj-finetuned", "TransferTraj", "TrajGPT", "CLIPMobility", "OmniTraj"]
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

# One metric per family for the two summary figures.
HEADLINE = {"recovery": ["ade_m"], "location": ["acc@1"], "continuous": ["crps_min"], "anomaly": ["roc_auc"],
            "generation": ["jsd"], "classification": ["macro_f1"], "identity": ["user_acc@1"],
            "retrieval": ["mrr"]}
# Metrics drawn in each family's detail figure ("jsd" matches jsd:radius_of_gyration, ...).
DETAIL = {"recovery": ["ade_m"], "location": ["acc@1", "acc@5", "acc@10", "median_dist_err_m"],
          "continuous": ["crps_min", "mae_min"], "anomaly": ["roc_auc"], "generation": ["jsd"],
          "classification": ["macro_f1", "accuracy"], "identity": ["user_acc@1", "user_macro_f1"],
          "retrieval": ["mrr", "hr@1", "cr@1"], "efficiency": ["n_parameters", "latency_ms_per_sample"]}
FAMILY_ORDER = ["recovery", "location", "continuous", "classification", "identity", "anomaly", "retrieval",
                "generation", "efficiency"]
FAMILY_TITLE = {"recovery": "Trajectory recovery", "location": "Next-location prediction",
                "continuous": "Travel time and stay duration", "classification": "Travel-mode classification",
                "identity": "User identification", "anomaly": "Anomaly detection", "retrieval": "Trajectory retrieval",
                "generation": "Generation: distance from the real distributions", "efficiency": "Size and speed"}
# Not competitors: the real-vs-real floor is the best any generator can reach, not a baseline to beat.
NOT_A_BASELINE = {"real_noise_floor"}

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
BASELINE_GREY = "#6f6e69"
BLUE, RED, NEUTRAL = "#2a78d6", "#e34948", "#f0efec"
DIVERGING = LinearSegmentedColormap.from_list("skill", [RED, NEUTRAL, BLUE])

METRIC_NAME = {"ade_m": "mean error on hidden points", "acc@1": "Acc@1", "acc@5": "Acc@5", "acc@10": "Acc@10",
               "median_dist_err_m": "median distance error", "crps_min": "CRPS", "mae_min": "MAE",
               "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "macro_f1": "macro-F1", "accuracy": "accuracy",
               "user_acc@1": "user Acc@1", "user_macro_f1": "user macro-F1", "mrr": "MRR", "hr@1": "HR@1",
               "cr@1": "CR@1", "jsd": "JSD", "n_parameters": "parameters", "latency_ms_per_sample": "latency"}
PROTOCOL_NAME = {"native": "native", "linear_probe": "probe", "embedding_knn": "embedding k-NN",
                 "reconstruction": "reconstruction", "rollout": "rollout", "embedding": "embedding"}


# --------------------------------------------------------------------------- data
def load(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"family", "task", "metric", "protocol", "model", "value", "higher_is_better", "unit"}
    if need - set(df.columns):
        raise SystemExit(f"{path}: not a mobeval leaderboard (missing columns {sorted(need - set(df.columns))})")
    for c in ("ci_low", "ci_high", "skill", "baseline"):
        if c not in df.columns:
            df[c] = np.nan
    df = df[np.isfinite(df.value)].copy()
    df["is_baseline"] = df.model.str.startswith("baseline:")
    df["name"] = df.model.str.replace("baseline:", "", regex=False)
    # native and probe results of one task share a panel: 'next_location/linear_probe' -> 'next_location'
    df["stem"] = df.task.str.replace("/linear_probe", "", regex=False)
    df["metric_base"] = df.metric.str.split(":").str[0]
    df["higher_is_better"] = df.higher_is_better.astype(str).str.lower().eq("true")
    return df


def model_order(df) -> list:
    seen = list(dict.fromkeys(df.loc[~df.is_baseline, "name"]))
    return [m for m in MODEL_ORDER if m in seen] + [m for m in seen if m not in MODEL_ORDER]


def model_colors(df) -> dict:
    known = MODEL_ORDER + [m for m in model_order(df) if m not in MODEL_ORDER]
    if len(known) > len(PALETTE):
        print(f"warning: {len(known)} models but {len(PALETTE)} colours; the last models share a colour")
    return {m: PALETTE[min(i, len(PALETTE) - 1)] for i, m in enumerate(known)}


def pretty_task(stem: str) -> str:
    s = stem
    m = re.fullmatch(r"recovery/(random|block)@([\d.]+)", s)
    if m:
        return f"{m.group(1)} mask, {float(m.group(2)):.0%} hidden"
    m = re.fullmatch(r"recovery/last:(\d+)", s)
    if m:
        return f"predict the last {m.group(1)} points"
    m = re.fullmatch(r"recovery/keep_every:(\d+)", s)
    if m:
        return f"keep every {m.group(1)}th point"
    if s == "next_location":
        return "next location"
    if s.startswith("continuous/"):
        target, *rest = s[len("continuous/"):].split("|")
        given = [r[len("given:"):].replace("+", " + ") for r in rest if r.startswith("given:")]
        return target.replace("_", " ") + (f", given {given[0]}" if given else "")
    if s.startswith("efficiency"):
        return s.split("/", 1)[1].replace("/", " ").replace("_", " ") if "/" in s else "model size"
    for prefix in ("anomaly/", "generation/", "retrieval/", "mode/"):
        if s.startswith(prefix):
            return s[len(prefix):].replace("_", " ").replace(":", ": ")
    return s.replace("_", " ")


def fmt(v: float, unit: str) -> str:
    if v is None or not np.isfinite(v):
        return "–"
    if unit == "m":
        return f"{v / 1000:,.1f} km" if abs(v) >= 10_000 else f"{v:,.0f} m"
    if unit == "min":
        return f"{v:,.1f} min" if abs(v) < 100 else f"{v:,.0f} min"
    if unit == "fraction":
        return f"{100 * v:.1f}%"
    if unit == "count":
        return f"{v / 1e6:.1f}M" if v >= 1e6 else f"{v:,.0f}"
    if unit == "ms":
        return f"{v:.2g} ms" if v < 10 else f"{v:,.0f} ms"
    return f"{v:.3f}" if abs(v) < 10 else f"{v:,.1f}"


def improvement(v, b, higher_is_better):
    """Relative gain of value v over baseline b in the good direction: 0 = same, +0.2 = 20% better,
    -1 = twice the baseline's error (or a score of zero)."""
    if b is None or not np.isfinite(b) or not np.isfinite(v) or b <= 0:
        return np.nan
    return (v - b) / b if higher_is_better else (b - v) / b


def best_baseline(g: pd.DataFrame):
    """The strongest competing baseline among the rows of one (stem, metric): a Series, or None."""
    b = g[g.is_baseline & ~g.name.isin(NOT_A_BASELINE)]
    if b.empty:
        return None
    return b.loc[b.value.idxmax() if b.higher_is_better.iloc[0] else b.value.idxmin()]


def wanted(df, family, table) -> pd.DataFrame:
    return df[(df.family == family) & df.metric_base.isin(table.get(family, []))]


def headline_rows(df: pd.DataFrame, against: str) -> pd.DataFrame:
    """One row per (task, protocol, model) for the headline metrics, with its improvement score."""
    out = []
    for fam in FAMILY_ORDER:
        sub = wanted(df, fam, HEADLINE)
        for (stem, metric), g in sub.groupby(["stem", "metric"], sort=False):
            bb = best_baseline(g)
            for _, r in g[~g.is_baseline].iterrows():
                if against == "reference":
                    imp, bname = r.skill, r.baseline
                    bval = g.loc[g.is_baseline & g.name.eq(bname), "value"]
                    bval = float(bval.iloc[0]) if len(bval) else np.nan
                    blo = bhi = np.nan
                elif bb is None:
                    continue
                else:
                    imp = improvement(r.value, bb.value, r.higher_is_better)
                    bname, bval, blo, bhi = bb["name"], bb.value, bb.ci_low, bb.ci_high
                out.append(dict(family=fam, stem=stem, metric=metric, protocol=r.protocol, model=r["name"],
                                value=r.value, ci_low=r.ci_low, ci_high=r.ci_high, unit=r.unit,
                                higher_is_better=r.higher_is_better, improvement=imp, baseline=bname,
                                baseline_value=bval, baseline_lo=blo, baseline_hi=bhi))
    h = pd.DataFrame(out)
    if h.empty:
        return h
    # a row of the summary figures = one task under one protocol
    multi = h.groupby("stem").protocol.transform("nunique") > 1
    plain = h.protocol.eq("native") & ~multi
    h["row"] = [pretty_task(s) + ("" if p else f"  ({PROTOCOL_NAME.get(pr, pr)})")
                for s, pr, p in zip(h.stem, h.protocol, plain)]
    h["row_key"] = h.family + "|" + h.stem + "|" + h.metric + "|" + h.protocol
    # reading order: families, then tasks (recovery: random, block, then the fixed schemes), native first
    h["_f"] = h.family.map(FAMILY_ORDER.index)
    h["_t"] = h.stem.map(_task_rank(h.stem))
    h["_p"] = (~h.protocol.isin(["native", "embedding_knn", "embedding"])).astype(int)
    return h.sort_values(["_f", "_t", "_p"], kind="stable").drop(columns=["_f", "_t", "_p"])


def _task_rank(stems) -> dict:
    first = {s: i for i, s in enumerate(dict.fromkeys(stems))}
    pref = ["recovery/random", "recovery/block", "recovery/last", "recovery/keep_every"]
    return {s: (next((i for i, p in enumerate(pref) if s.startswith(p)), len(pref)), first[s]) for s in first}


# --------------------------------------------------------------------------- style
def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK,
        "xtick.labelsize": 8, "ytick.labelsize": 8.5, "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": GRID, "grid.linewidth": 0.7, "legend.frameon": False, "legend.fontsize": 8.5,
    })


def header(fig, title, subtitle, top=0.99):
    fig.text(0.01, top, title, ha="left", va="top", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, top - 0.30 / fig.get_figheight(), subtitle, ha="left", va="top", fontsize=9, color=INK2)


def save(fig, out: Path, name: str, formats):
    for f in formats:
        fig.savefig(out / f"{name}.{f}", dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"  {name}")


# --------------------------------------------------------------------------- 01 overview heatmap
def fig_heatmap(h, models, out, formats, against):
    rows = list(dict.fromkeys(h.row_key))
    labels = h.drop_duplicates("row_key").set_index("row_key")
    M = np.full((len(rows), len(models)), np.nan)
    for _, r in h.iterrows():
        M[rows.index(r.row_key), models.index(r.model)] = r.improvement
    fig_h = 1.5 + 0.30 * len(rows)
    # laid out in inches, so it holds for any number of models and any label length
    left_in = 1.0 + 0.085 * max(len(labels.loc[k, "row"]) for k in rows)
    axes_w = 1.25 * len(models)
    fig_w = max(left_in + axes_w + 0.3, 10.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.subplots_adjust(top=1 - 1.45 / fig_h, bottom=0.6 / fig_h, left=left_in / fig_w,
                        right=(left_in + axes_w) / fig_w)
    ax.pcolormesh(np.ma.masked_invalid(np.clip(M, -1, 1)), cmap=DIVERGING, vmin=-1, vmax=1,
                  edgecolors=SURFACE, linewidth=2)
    best = np.nanmax(np.where(np.isfinite(M), M, -np.inf), axis=1)
    for i in range(len(rows)):
        for j in range(len(models)):
            v = M[i, j]
            if not np.isfinite(v):
                ax.text(j + 0.5, i + 0.5, "·", ha="center", va="center", color=AXIS, fontsize=9)
                continue
            ax.text(j + 0.5, i + 0.5, f"{v:+.2f}" if abs(v) < 10 else f"{v:+.0f}", ha="center", va="center",
                    fontsize=8, color=SURFACE if abs(v) > 0.6 else INK,
                    fontweight="bold" if v == best[i] else "normal")
    ax.set_xlim(0, len(models)); ax.set_ylim(len(rows), 0)
    ax.set_xticks(np.arange(len(models)) + 0.5, models, fontsize=8.5)
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(len(rows)) + 0.5, [labels.loc[k, "row"] for k in rows])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # family blocks: a rule between them and the family name with its metric on the left
    fams = [labels.loc[k, "family"] for k in rows]
    start = 0
    for i in range(1, len(rows) + 1):
        if i == len(rows) or fams[i] != fams[start]:
            if i < len(rows):
                ax.axhline(i, color=INK2, linewidth=0.8, xmin=-(left_in - 0.1) / axes_w, clip_on=False)
            m = METRIC_NAME.get(HEADLINE[fams[start]][0], HEADLINE[fams[start]][0])
            ax.text(-(left_in - 0.4) / axes_w, (start + i) / 2, f"{fams[start].upper()}\n{m}", transform=ax.get_yaxis_transform(),
                    ha="center", va="center", fontsize=7.5, color=MUTED, rotation=90, linespacing=1.4)
            start = i
    cax = fig.add_axes([left_in / fig_w, 0.3 / fig_h, min(axes_w, 3.2) / fig_w, 0.10 / fig_h])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=DIVERGING, norm=plt.Normalize(-1, 1)), cax=cax,
                      orientation="horizontal", ticks=[-1, 0, 1])
    cb.ax.set_xticklabels(["≤ −1 worse", "0 = baseline", "≥ +1 better"], fontsize=7.5)
    cb.outline.set_visible(False)
    what = ("Relative gain over the best baseline of each task (0 = same, +0.2 = 20% better, −1 = twice the error "
            "or worse)." if against == "best" else
            "Skill against the pipeline's reference baseline, from the leaderboard (0 = same as the baseline, "
            "1 = perfect).")
    header(fig, "Where each model beats the baselines",
           what + "\nBold = best model in the row;  · = task not available for that model.")
    save(fig, out, "01_overview_heatmap", formats)


# --------------------------------------------------------------------------- 02 best model vs baseline
def fig_best(h, out, formats, against):
    best = h.loc[h.groupby("row_key", sort=False).improvement.idxmax().dropna()]
    best = best.set_index("row_key").loc[[k for k in dict.fromkeys(h.row_key) if k in set(best.row_key)]]
    n = len(best)
    fig_h = 1.3 + 0.32 * n
    fig, ax = plt.subplots(figsize=(11.5, fig_h))
    fig.subplots_adjust(top=1 - 0.95 / fig_h, bottom=0.5 / fig_h, left=0.27, right=0.57)
    y = np.arange(n)
    # overlapping intervals: the difference is within the uncertainty, drawn pale
    hib = best.higher_is_better.to_numpy()
    lo, hi, blo, bhi = (best[c].to_numpy(float) for c in ("ci_low", "ci_high", "baseline_lo", "baseline_hi"))
    clear = np.where(best.improvement.to_numpy() > 0, np.where(hib, lo > bhi, hi < blo),
                     np.where(hib, hi < blo, lo > bhi))
    clear = np.where(np.isfinite(lo) & np.isfinite(blo), clear, True)
    imp = best.improvement.to_numpy(float)
    ax.barh(y, np.clip(imp, -1, 1), height=0.62, color=[BLUE if v > 0 else RED for v in imp],
            alpha=None, edgecolor="none")
    for bar, c in zip(ax.patches, clear):
        bar.set_alpha(1.0 if c else 0.35)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_xlim(-1.08, 1.0); ax.set_ylim(n - 0.4, -0.6)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1], ["≤ −1", "−0.5", "0", "+0.5", "+1"])
    ax.set_yticks(y, best.row)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x"); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(("relative gain over the baseline" if against == "best" else "skill against the baseline")
                  + "   (← baseline wins · model wins →)")
    T = ax.get_yaxis_transform()
    ax.text(1.04, -0.75, "best model", transform=T, fontsize=8, color=MUTED, va="bottom")
    ax.text(1.62, -0.75, "baseline", transform=T, fontsize=8, color=MUTED, va="bottom")
    for i, (_, r) in enumerate(best.iterrows()):
        if r.improvement < -1:
            ax.text(-0.98, i, f"{r.improvement:+.1f}" if r.improvement > -100 else "≪ −1", ha="left", va="center",
                    fontsize=7.5, color=SURFACE, fontweight="bold")
        ax.text(1.04, i, f"{r.model}  {fmt(r.value, r.unit)}", transform=T, va="center", fontsize=8.5,
                fontweight="bold" if r.improvement > 0 else "normal", color=INK)
        ax.text(1.62, i, f"{r.baseline}  {fmt(r.baseline_value, r.unit)}", transform=T, va="center", fontsize=8.5,
                fontweight="bold" if r.improvement <= 0 else "normal", color=INK2)
    fams = best.family.to_list()
    for i in range(1, n):
        if fams[i] != fams[i - 1]:
            ax.axhline(i - 0.5, color=AXIS, linewidth=0.8, xmin=-0.85, xmax=2.3, clip_on=False)
    wins, clear_wins = int((imp > 0).sum()), int(((imp > 0) & clear).sum())
    which = "best" if against == "best" else "reference"
    header(fig, f"A model is ahead of the {which} baseline in {wins} of {n} comparisons ({clear_wins} clearly)",
           "Best model of each task against the baseline, on the headline metric of the family. "
           "Pale bars: the two 95% intervals overlap, so the difference is not clear.")
    save(fig, out, "02_best_model_vs_baseline", formats)


# --------------------------------------------------------------------------- 03 recovery by ratio
def _spread(ys, min_gap):
    """Move label positions apart (in axis units) until none is closer than min_gap; keeps the order."""
    order = np.argsort(ys)
    out = np.array(ys, float)
    for _ in range(200):
        moved = False
        for a, b in zip(order[:-1], order[1:]):
            d = out[b] - out[a]
            if d < min_gap:
                out[a] -= (min_gap - d) / 2; out[b] += (min_gap - d) / 2; moved = True
        if not moved:
            break
    return out


def fig_recovery_ratio(df, models, colors, out, formats, metric="ade_m"):
    r = df[(df.family == "recovery") & (df.metric == metric)].copy()
    ex = r.task.str.extract(r"recovery/(random|block)@([\d.]+)$")
    r["kind"], r["ratio"] = ex[0], pd.to_numeric(ex[1])
    r = r.dropna(subset=["kind"])
    kinds = [k for k in ("random", "block") if (r.kind == k).any()]
    if not kinds or r.ratio.nunique() < 2:
        return
    fig, axes = plt.subplots(1, len(kinds), figsize=(5.6 * len(kinds), 4.6), sharey=True, squeeze=False)
    fig.subplots_adjust(top=0.80, bottom=0.12, left=0.07, right=0.84, wspace=0.62)
    bl_markers = ["s", "^", "D", "v", "P"]
    baselines = list(dict.fromkeys(r.loc[r.is_baseline, "name"]))
    for ax, kind in zip(axes[0], kinds):
        g = r[r.kind == kind]
        ends = []
        for name in baselines + models:
            s = g[g.name == name].sort_values("ratio")
            if s.empty:
                continue
            is_b = name in baselines
            c = BASELINE_GREY if is_b else colors[name]
            ax.fill_between(s.ratio, s.ci_low, s.ci_high, color=c, alpha=0.12, linewidth=0)
            ax.plot(s.ratio, s.value, color=c, linewidth=1.4 if is_b else 2, linestyle=(0, (4, 2)) if is_b else "-",
                    marker=bl_markers[baselines.index(name) % 5] if is_b else "o", markersize=5 if is_b else 6,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=2 if is_b else 3)
            ends.append((name, float(s.value.iloc[-1]), c, is_b))
        ax.set_yscale("log")
        ax.set_xticks(sorted(g.ratio.unique()), [f"{x:.0%}" for x in sorted(g.ratio.unique())])
        ax.set_xlabel("share of points hidden")
        ax.grid(axis="y", which="major"); ax.set_axisbelow(True)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:g} km" if v >= 1000 else f"{v:g} m")
        ax.set_title(f"{kind} mask" + ("  (scattered points)" if kind == "random" else "  (one contiguous gap)"))
        # direct labels at the right end, spread apart in log space
        ypos = _spread(np.log10([e[1] for e in ends]), 0.085 * np.diff(np.log10(ax.get_ylim()))[0])
        x_end = g.ratio.max()
        for (name, v, c, is_b), yp in zip(ends, ypos):
            ax.annotate(f"{name}" if not is_b else f"{name} (baseline)", (x_end, v), xytext=(x_end + 0.035, 10 ** yp),
                        textcoords="data", fontsize=8, color=INK2 if is_b else INK, va="center",
                        fontweight="normal" if is_b else "bold",
                        arrowprops=dict(arrowstyle="-", color=c, linewidth=0.8, shrinkA=0, shrinkB=3))
        ax.set_xlim(g.ratio.min() - 0.03, x_end + 0.03)
    axes[0][0].set_ylabel("mean error on hidden points (log scale)")
    handles = [Line2D([], [], color=INK2, linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE, label="model"),
               Line2D([], [], color=BASELINE_GREY, linewidth=1.4, linestyle=(0, (4, 2)), marker="s", markersize=5,
                      markeredgecolor=SURFACE, label="baseline")]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.84, 0.90), ncol=2)
    header(fig, "Recovery error as more of the trajectory is hidden",
           "Lower is better. A model is useful where its line is below the dashed baselines. "
           "Bands: 95% bootstrap interval.")
    save(fig, out, "03_recovery_by_mask_ratio", formats)


# --------------------------------------------------------------------------- 10+ family detail
def fig_family(df, family, models, colors, out, formats, index):
    sub = wanted(df, family, DETAIL)
    if sub.empty:
        return
    # baselines are repeated under every protocol of a task with the same value: keep one
    sub = pd.concat([sub[~sub.is_baseline], sub[sub.is_baseline].drop_duplicates(["stem", "metric", "name"])])
    one_task = sub.stem.nunique() == 1
    panels = list(dict.fromkeys(zip(sub.stem, sub.metric)))
    rank = _task_rank(sub.stem)
    panels.sort(key=lambda p: (rank[p[0]], DETAIL[family].index(p[1].split(":")[0])))
    several_metrics = len({m.split(":")[0] for _, m in panels}) > 1
    # rows, shared by every panel: models (per protocol), then baselines
    mrows = sub[~sub.is_baseline]
    several = mrows.protocol.nunique() > 1
    prot_order = list(dict.fromkeys(mrows.protocol))
    row_keys = [(m, p) for m in models for p in prot_order if ((mrows.name == m) & (mrows.protocol == p)).any()]
    brows = list(dict.fromkeys(sub.loc[sub.is_baseline, "name"]))
    labels = [m + (f"  ·  {PROTOCOL_NAME.get(p, p)}" if several else "") for m, p in row_keys] + brows
    n_rows = len(labels)
    ypos = {k: i for i, k in enumerate(row_keys)}
    ypos.update({("baseline", b): len(row_keys) + 0.6 + i for i, b in enumerate(brows)})

    # one scale type per metric in a figure: log wherever any of its panels spans a wide range
    def _wide(g):
        v = np.concatenate([g.value, g.ci_low.dropna(), g.ci_high.dropna()])
        return bool(g.unit.iloc[0] not in ("fraction", "auc", "rho") and (v > 0).all() and v.max() / v.min() > 8)
    log_metric = {mb: any(_wide(g) for _, g in sm.groupby(["stem", "metric"]))
                  for mb, sm in sub.groupby("metric_base")}

    ncols = min(len(panels), 4 if n_rows <= 9 else 3)
    nrows = math.ceil(len(panels) / ncols)
    pw, ph = 3.0, 0.9 + 0.27 * (n_rows + 1)
    left_in = 0.5 + 0.075 * max(len(l) for l in labels)
    fig_w, fig_h = left_in + pw * ncols + 0.45 * (ncols - 1) + 0.3, 1.15 + ph * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    fig.subplots_adjust(left=left_in / fig_w, right=1 - 0.3 / fig_w, top=1 - 1.25 / fig_h, bottom=0.55 / fig_h,
                        wspace=0.45 * ncols / (pw * ncols) * 1.0, hspace=0.95 / (ph - 0.9 + 0.3))
    for k, ax in enumerate(axes.ravel()):
        if k >= len(panels):
            ax.set_visible(False)
            continue
        stem, metric = panels[k]
        g = sub[(sub.stem == stem) & (sub.metric == metric)]
        unit, hib = g.unit.iloc[0], bool(g.higher_is_better.iloc[0])
        vals = np.concatenate([g.value, g.ci_low.dropna(), g.ci_high.dropna()])
        log = log_metric[metric.split(":")[0]] and (vals > 0).all()
        bb = best_baseline(g)
        if bb is not None:
            ax.axvline(bb.value, color=BASELINE_GREY, linewidth=1, linestyle=(0, (4, 2)), zorder=1)
        if unit == "auc":
            ax.axvline(0.5, color=AXIS, linewidth=0.8, zorder=1)
        best_model = None
        for _, r in g.iterrows():
            key = ("baseline", r["name"]) if r.is_baseline else (r["name"], r.protocol)
            if key not in ypos:
                continue
            yy = ypos[key]
            c = BASELINE_GREY if r.is_baseline else colors[r["name"]]
            if np.isfinite(r.ci_low) and np.isfinite(r.ci_high):
                ax.plot([r.ci_low, r.ci_high], [yy, yy], color=c, linewidth=1.6, solid_capstyle="round", zorder=2)
            hollow = r.protocol == "linear_probe" and not r.is_baseline
            ax.plot(r.value, yy, marker="D" if r.is_baseline else "o", markersize=6 if r.is_baseline else 8,
                    markerfacecolor=SURFACE if hollow else c, markeredgecolor=c if hollow else SURFACE,
                    markeredgewidth=1.8 if hollow else 1.2, linestyle="none", zorder=3, clip_on=False)
            if not r.is_baseline and (best_model is None or (r.value > best_model.value) == hib):
                best_model = r
        ax.set_ylim(max(ypos.values()) + 0.7, -0.7)
        if log:
            ax.set_xscale("log")
        ax.margins(x=0.12)
        if not log and unit in ("fraction", "auc") and vals.min() >= 0:
            ax.set_xlim(left=-0.04 * ax.get_xlim()[1])      # room for a marker sitting at zero
        if log:                                             # a short log axis needs 2 and 5 labelled too
            lo_, hi_ = ax.get_xlim()
            ax.xaxis.set_major_locator(LogLocator(10, subs=(1, 2, 5) if np.log10(hi_ / lo_) < 1.3 else (1,)))
        # one direct label per panel: the best model's value
        if best_model is not None:
            ax.annotate(fmt(best_model.value, unit), (best_model.value, ypos[(best_model["name"], best_model.protocol)]),
                        xytext=(9, 0), textcoords="offset points", ha="left", va="center", fontsize=7.5, color=INK2,
                        bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1, alpha=0.85), zorder=4)
        ax.set_yticks([ypos[k2] for k2 in row_keys] + [ypos[("baseline", b)] for b in brows],
                      labels if k % ncols == 0 else [""] * n_rows)
        for t, lab in zip(ax.get_yticklabels(), labels):
            t.set_color(INK2 if lab in brows else INK)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x"); ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        if brows and row_keys:
            ax.axhline(len(row_keys) - 0.2, color=GRID, linewidth=0.8)
        if unit == "fraction":
            ax.xaxis.set_major_formatter(lambda v, _: f"{100 * v:g}%")
        elif unit == "m":
            ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1000:g} km" if v >= 1000 else f"{v:g} m")
        elif unit == "count":
            ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1e6:g}M")
        elif log:
            ax.xaxis.set_major_formatter(lambda v, _: f"{v:g}")
        ax.tick_params(axis="x", which="minor", labelbottom=False)
        mname = METRIC_NAME.get(metric.split(":")[0], metric)
        ax.set_title(mname if one_task else pretty_task(stem) + (f"  ·  {mname}" if several_metrics else ""))
        better = "higher is better →" if hib else "← lower is better"
        u = {"min": "minutes", "ms": "ms per sample", "auc": "", "fraction": "", "m": "", "count": ""}.get(unit, unit)
        ax.set_xlabel("  ·  ".join(x for x in ([] if one_task or several_metrics else [mname]) + [u, better + (", log scale" if log else "")] if x),
                      fontsize=7.5, color=MUTED)
    note = ["◆ grey = baseline, dashed line = best baseline"] if brows else []
    if (mrows.protocol == "linear_probe").any():
        note.append("○ hollow = linear probe on the frozen embedding")
    if (sub.unit == "auc").any():
        note.append("solid line = chance (0.5)")
    if "real_noise_floor" in brows:
        note.append("real_noise_floor = real data against real data, the best reachable")
    note.append("bars = 95% interval; the number is the best model's value")
    title = FAMILY_TITLE.get(family, family)
    if one_task and pretty_task(panels[0][0]).lower() not in title.lower().replace("-", " "):
        title += f": {pretty_task(panels[0][0])}"
    header(fig, title, ";  ".join(note[:3]) + "." + ("\n" + ";  ".join(note[3:]) + "." if len(note) > 3 else ""))
    save(fig, out, f"{index:02d}_{family}", formats)


# --------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("csv", help="leaderboard.csv written by `mobeval evaluate`")
    ap.add_argument("--out", default="figures", help="output directory (default: figures)")
    ap.add_argument("--format", default="png", help="comma-separated: png, pdf, svg (default: png)")
    ap.add_argument("--against", choices=["best", "reference"], default="best",
                    help="summary figures compare with the best baseline of each task (default) or with the "
                         "leaderboard's reference baseline (its `skill` column)")
    ap.add_argument("--models", nargs="*", help="only these models (default: all in the file)")
    a = ap.parse_args(argv)

    df = load(a.csv)
    if a.models:
        df = df[df.is_baseline | df.name.isin(a.models)]
    models, colors = model_order(df), model_colors(df)
    if not models:
        raise SystemExit("no model rows in the leaderboard")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    formats = [f.strip() for f in a.format.split(",") if f.strip()]
    style()
    print(f"{len(df)} rows, {len(models)} models: {', '.join(models)}\nwriting to {out}/")

    h = headline_rows(df, a.against)
    if not h.empty and h.improvement.notna().any():
        h = h[h.improvement.notna()]
        fig_heatmap(h, models, out, formats, a.against)
        fig_best(h, out, formats, a.against)
    fig_recovery_ratio(df, models, colors, out, formats)
    for i, fam in enumerate([f for f in FAMILY_ORDER if f in set(df.family)], start=10):
        fig_family(df, fam, models, colors, out, formats, i)


if __name__ == "__main__":
    main()
