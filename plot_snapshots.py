#!/usr/bin/env python
"""Figures from a `mobeval snapshots` run: which training epoch does best, task by task.

    python plot_snapshots.py <run folder>                          # PNG next to nothing: ./figures_snapshots
    python plot_snapshots.py <run folder> --out figs --format png,pdf --models TrajGPT

The run folder is the one `mobeval snapshots` wrote (jobs/evaluate_snapshots.pbs prints it), with
leaderboard.csv, snapshots.csv and training_history.csv. Writes:

  40_snapshots_<model>     one figure per model: a panel per task with the result of every snapshot
                           against its training epoch (with its 95% interval), the weights the
                           training kept (star), the best epoch (ring), the best baseline (dashed);
                           first panel: the training and validation losses, snapshot epochs marked
  41_snapshot_heatmap      every model's snapshots side by side: a row per task, a column per epoch,
                           coloured from the worst epoch of the row (light) to the best (dark)
  42_snapshot_ranking      per model, each epoch's average position over the task families (top =
                           best), next to the same epochs ordered by validation loss alone
  best_snapshots.csv       per model and task: best epoch and value, the kept weights' value

Colours, fonts and task names come from plot_results.py, so these figures match the others.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_results import (BASELINE_GREY, BLUE, FAMILY_ORDER, INK, INK2, METRIC_NAME, MODEL_ORDER, MUTED, PALETTE,  # noqa: E402
                          PROTOCOL_NAME, _task_rank, fmt, header, pretty_task, save, style)

from mobeval import snapshots as S  # noqa: E402

SEQUENTIAL = LinearSegmentedColormap.from_list("position", ["#eef3fa", BLUE])
SHORT = {"ade_m": "mean error", "acc@1": "Acc@1", "crps_min": "CRPS", "roc_auc": "ROC-AUC", "jsd": "JSD",
         "mrr": "MRR", "macro_f1": "macro-F1", "user_acc@1": "user Acc@1"}


def subtitle_lines(text: str, fig_w: float) -> list:
    return textwrap.wrap(text, max(50, int(fig_w * 13.5)))


def wrapped_header(fig, title: str, lines: list):
    """plot_results.header with the subtitle already wrapped to the figure's width."""
    header(fig, title, "\n".join(lines))


def header_height(lines: list) -> float:
    """Inches taken by the title and a subtitle of that many lines."""
    return 0.55 + 0.16 * len(lines)


LABEL_BOX = dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85)


def _free_spot(ax, xs, ys, y0, band=0.16):
    """(side, above): where to put the label of the horizontal line y = y0 so that it hides the
    fewest of the points (xs, ys): its left or right end, just above or just below the line."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    ok = np.isfinite(xs) & np.isfinite(ys)
    if not ok.any():
        return "right", True
    (x0, x1), (y_lo, y_hi) = ax.get_xlim(), ax.get_ylim()
    xn, yn = (xs[ok] - x0) / (x1 - x0), (ys[ok] - y_lo) / (y_hi - y_lo)
    t = (y0 - y_lo) / (y_hi - y_lo)
    spots = []
    for side, sx in (("right", xn > 0.45), ("left", xn < 0.55)):
        for above, sy in ((True, (yn >= t) & (yn < t + band)), (False, (yn <= t) & (yn > t - band))):
            room = (t + 0.08 < 1.0) if above else (t - 0.08 > 0.0)
            spots.append((int(np.sum(sx & sy)) + (0 if room else 100), side != "right", not above, side, above))
    return min(spots)[3:]


def _place_label(ax, x, y, text, curves, marks=(), gap=9, **kw):
    """Annotate the point (x, y) where the label stays inside the axes and hides the least of
    `curves` (a list of (xs, ys), sampled along their segments so that a line crossing the label
    counts, not only its markers) and of `marks` (single points). The candidates are above or below
    the point - to its left, centred on it or to its right - first close to it, then a little
    farther; among equally clear ones the closest wins. Call it once the layout is final: it
    measures in display coordinates."""
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    ann = ax.annotate(text, (x, y), textcoords="offset points", xytext=(gap, gap), ha="left", va="bottom", **kw)
    k = fig.dpi / 72.0                                        # pixels per point
    ext = ann.get_window_extent(renderer)
    w, h = ext.width + 4 * k, ext.height + 4 * k
    px, py = ax.transData.transform((x, y))
    frame = ax.get_window_extent(renderer)

    def display(xs, ys, dense):
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        ok = np.isfinite(xs) & np.isfinite(ys)
        xs, ys = xs[ok], ys[ok]
        if dense and len(xs) > 1:
            t = np.linspace(0.0, 1.0, 30)
            xs = np.concatenate([a + t * (b - a) for a, b in zip(xs[:-1], xs[1:])])
            ys = np.concatenate([a + t * (b - a) for a, b in zip(ys[:-1], ys[1:])])
        return ax.transData.transform(np.c_[xs, ys]) if len(xs) else np.empty((0, 2))

    line_pts = [display(xs, ys, True) for xs, ys in curves]
    mark_pts = [display(xs, ys, False) for xs, ys in marks]
    sides = ("right", "center", "left") if px > (frame.x0 + frame.x1) / 2 else ("left", "center", "right")
    options = []
    for far in (0.0, 0.7):
        dist = gap * k + far * h                              # pixels between the point and the label
        for up in (True, False):
            for ha in sides:                                  # ha="right": the label ends left of the point
                x0 = {"right": px - gap * k - w, "center": px - w / 2, "left": px + gap * k}[ha]
                y0 = py + dist if up else py - dist - h

                def covered(p):
                    return int(np.sum((p[:, 0] >= x0) & (p[:, 0] <= x0 + w) & (p[:, 1] >= y0) & (p[:, 1] <= y0 + h)))
                inside = frame.x0 <= x0 and x0 + w <= frame.x1 and frame.y0 <= y0 and y0 + h <= frame.y1
                hidden = sum(covered(p) for p in line_pts) + 30 * sum(covered(p) for p in mark_pts)
                dx = {"right": -gap, "center": 0, "left": gap}[ha]
                options.append((not inside, hidden, len(options), dx, dist / k if up else -dist / k, ha, up))
    dx, dy, ha, up = min(options)[3:]
    ann.xyann = (dx, dy)
    ann.set_ha(ha)
    ann.set_va("bottom" if up else "top")
    return ann


# --------------------------------------------------------------------------- data
def load_run(run: Path):
    run = Path(run).expanduser()
    if run.is_file():
        run = run.parent
    need = ["leaderboard.csv", "snapshots.csv"]
    missing = [f for f in need if not (run / f).exists()]
    if missing:
        raise SystemExit(f"{run}: not a `mobeval snapshots` run folder (missing {', '.join(missing)})")
    lb = pd.read_csv(run / "leaderboard.csv")
    snaps = pd.read_csv(run / "snapshots.csv")
    if "kept" not in snaps:
        snaps["kept"] = False
    snaps["kept"] = snaps.kept.astype(str).str.lower().eq("true")
    hist = pd.read_csv(run / "training_history.csv") if (run / "training_history.csv").exists() else pd.DataFrame()
    train = pd.read_csv(run / "leaderboard_train.csv") if (run / "leaderboard_train.csv").exists() else None
    split = "val"
    if (run / "run_info.json").exists():
        try:
            split = json.loads((run / "run_info.json").read_text()).get("eval_split", "val")
        except (OSError, json.JSONDecodeError):
            pass
    return lb, snaps, hist, train, split


def values(lb: pd.DataFrame, snaps: pd.DataFrame) -> pd.DataFrame:
    """Primary-metric rows of the evaluated models, with their base model, epoch and kept flag."""
    d = S.primary_rows(lb)
    info = snaps.set_index("model")
    d = d[d.model.isin(info.index)].copy()
    d["base"] = d.model.map(info.base)
    d["epoch"] = d.model.map(info.epoch)
    d["kind"] = d.model.map(info.kind)
    d["kept"] = d.model.map(info.kept).astype(bool)
    return d


def row_key(r) -> str:
    return f"{r.task}|{r.protocol}|{r.metric}"


def row_label(task, protocol, metric) -> str:
    stem = task.replace("/linear_probe", "")
    label = pretty_task(stem)
    if protocol not in ("native",):
        label += f" ({PROTOCOL_NAME.get(protocol, protocol)})"
    if ":" in metric and not stem.startswith("generation/"):
        label += f" · {metric.split(':', 1)[1].replace('_', ' ')}"
    return label


def ordered_rows(summary: pd.DataFrame) -> pd.DataFrame:
    """Reading order of plot_results: families, then tasks, native before probe."""
    s = summary.copy()
    s["stem"] = s.task.str.replace("/linear_probe", "", regex=False)
    s["_f"] = s.family.map(lambda f: FAMILY_ORDER.index(f) if f in FAMILY_ORDER else len(FAMILY_ORDER))
    s["_t"] = s.stem.map(_task_rank(s.stem))
    s["_p"] = (~s.protocol.isin(["native", "embedding_knn", "embedding"])).astype(int)
    return s.sort_values(["base", "_f", "_t", "_p", "metric"], kind="stable").drop(columns=["_f", "_t", "_p"])


def base_colors(bases) -> dict:
    known = MODEL_ORDER + [b for b in bases if b not in MODEL_ORDER]
    return {b: PALETTE[min(known.index(b), len(PALETTE) - 1)] for b in bases}


def unit_formatter(unit):
    return FuncFormatter(lambda v, _: fmt(v, unit))


# --------------------------------------------------------------------------- 40 per-model curves
def _loss_panel(ax, h, snaps_b, color):
    ep = h.epoch.to_numpy(float)
    drawn = False
    for col, kw, label in (("train_loss", dict(color=MUTED, lw=1.0, ls=":"), "train, during the epoch"),
                           ("train_eval_loss", dict(color=color, lw=1.3, ls="--"), "train, like validation"),
                           ("val_loss", dict(color=color, lw=1.9), "validation")):
        if col in h and h[col].notna().any():
            ax.plot(ep, h[col].to_numpy(float), label=label, **kw)
            drawn = True
    if not drawn:
        ax.axis("off")
        return
    if "val_loss" in h and h.val_loss.notna().any():
        at = h.set_index("epoch").val_loss
        se = [e for e in snaps_b.epoch.dropna().astype(int) if e in at.index]
        ax.plot(se, at.loc[se].to_numpy(float), "o", ms=3.6, color=color, zorder=3)
        kept = snaps_b[snaps_b.kept & snaps_b.epoch.notna()]
        if len(kept) and int(kept.epoch.iloc[0]) in at.index:
            ke = int(kept.epoch.iloc[0])
            ax.scatter([ke], [at.loc[ke]], marker="*", s=120, color=color, edgecolor="white", linewidth=0.8, zorder=4)
    vals = pd.concat([h[c] for c in ("train_loss", "train_eval_loss", "val_loss") if c in h]).dropna()
    vals = vals[np.isfinite(vals)]
    if len(vals) and vals.min() > 0 and vals.max() / vals.min() > 30:
        from matplotlib.ticker import LogLocator, NullFormatter
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_title("training log: loss per epoch", fontsize=9)
    ax.set_ylabel("loss", fontsize=8)
    ax.legend(fontsize=6.8, loc="best")


def _task_panel(ax, g, srow, train_g, color, unit):
    snap = g[g.kind.eq(S.SNAPSHOT)].sort_values("epoch")
    ep = snap.epoch.to_numpy(float)
    if snap.ci_low.notna().any():
        ax.fill_between(ep, snap.ci_low.to_numpy(float), snap.ci_high.to_numpy(float), color=color, alpha=0.14, lw=0)
    ax.plot(ep, snap.value.to_numpy(float), color=color, lw=1.8, marker="o", ms=3.6, zorder=3)
    if train_g is not None and len(train_g):
        t = train_g.sort_values("epoch")
        ax.plot(t.epoch.to_numpy(float), t.value.to_numpy(float), color=color, lw=1.0, ls=":", zorder=2)
    kept = g[g.kept & g.epoch.notna()]
    if len(kept):
        ax.scatter(kept.epoch, kept.value, marker="*", s=120, color=color, edgecolor="white", linewidth=0.8, zorder=5)
    best_label_y = None
    if srow.best_epoch is not None and pd.notna(srow.best_epoch):
        ax.scatter([srow.best_epoch], [srow.best_value], s=170, facecolor="none", edgecolor=INK, linewidth=1.3,
                   zorder=6)
        y_lo, y_hi = ax.get_ylim()
        near_top = y_hi > y_lo and (srow.best_value - y_lo) / (y_hi - y_lo) > 0.82
        best_label_y = srow.best_value + (-0.1 if near_top else 0.1) * (y_hi - y_lo)
        ax.annotate(f"best: {int(srow.best_epoch)}", (srow.best_epoch, srow.best_value), textcoords="offset points",
                    xytext=(0, -10 if near_top else 9), ha="center", va="top" if near_top else "bottom",
                    fontsize=7, color=INK2, zorder=9, bbox=LABEL_BOX)
    # the points a label must not hide: the results, the training-data line, the best ring and
    # the "best: N" text above it
    xs = list(g.epoch) + (list(train_g.epoch) if train_g is not None else [])
    ys = list(g.value) + (list(train_g.value) if train_g is not None else [])
    if best_label_y is not None:
        xs += [srow.best_epoch] * 6
        ys += [srow.best_value] * 3 + [best_label_y] * 3
    # the best baseline: a line when it is near the results, a note at the edge when it is far
    # off (drawing it would flatten the curve that is the point of the panel)
    lo = np.nanmin(np.r_[ys, g.ci_low.dropna()]) if len(ys) else np.nan
    hi = np.nanmax(np.r_[ys, g.ci_high.dropna()]) if len(ys) else np.nan
    bv = srow.baseline_value
    if srow.baseline and np.isfinite(bv) and np.isfinite(lo):
        span = max(hi - lo, abs(hi) * 0.05, 1e-12)
        label = f"best baseline ({srow.baseline.replace('_', ' ')}): {fmt(bv, unit)}"
        if lo - 0.6 * span <= bv <= hi + 0.6 * span:
            ax.axhline(bv, color=BASELINE_GREY, lw=1.0, ls="--", zorder=1)
            side, up = _free_spot(ax, xs, ys, bv)
            ax.annotate(label, xy=(1 if side == "right" else 0, bv), xycoords=("axes fraction", "data"),
                        xytext=(-2 if side == "right" else 2, 2 if up else -2), textcoords="offset points",
                        ha=side, va="bottom" if up else "top", fontsize=6.5, color=BASELINE_GREY, bbox=LABEL_BOX,
                        zorder=8)
        else:
            above = bv > hi
            y_edge = ax.get_ylim()[1] if above else ax.get_ylim()[0]
            side, _ = _free_spot(ax, xs, ys, y_edge, band=0.3)
            ax.text(0.99 if side == "right" else 0.01, 0.98 if above else 0.02,
                    f"{label} ({'above' if above else 'below'}, off the scale)", transform=ax.transAxes, ha=side,
                    va="top" if above else "bottom", fontsize=6.5, color=BASELINE_GREY, bbox=LABEL_BOX, zorder=8)
    ax.yaxis.set_major_formatter(unit_formatter(unit))
    mb = srow.metric.split(":")[0]
    ax.set_ylabel(f"{SHORT.get(mb, METRIC_NAME.get(mb, mb))}, {'higher' if srow.higher_is_better else 'lower'} "
                  f"is better", fontsize=7.5)


def fig_model(base, vals, summary, snaps, hist, train_vals, split, color, out, formats):
    rows = summary[summary.base.eq(base)]
    h = hist[hist.base.eq(base)].sort_values("epoch") if len(hist) and "base" in hist else pd.DataFrame()
    snaps_b = snaps[snaps.base.eq(base)]
    panels = (["loss"] if len(h) else []) + list(rows.itertuples())
    ncols = min(4, len(panels))
    nrows = math.ceil(len(panels) / ncols)
    kept = snaps_b[snaps_b.kept]
    ke = int(kept.epoch.iloc[0]) if len(kept) and pd.notna(kept.epoch.iloc[0]) else None
    split_name = "validation" if split == "val" else "TEST"
    fig_w = max(3.75 * ncols, 9.0)
    lines = subtitle_lines(
        f"Every snapshot scored on the {split_name} split, task by task. "
        + (f"The training kept epoch {ke} (lowest validation loss). " if ke is not None else "")
        + "Where the ring and the star differ, the validation loss did not pick the epoch that does best on "
          "that task.", fig_w)
    top_in = header_height(lines) + 0.32                    # header, then the legend row
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, 2.75 * nrows + top_in), squeeze=False)
    epochs = snaps_b.epoch.dropna().astype(float)
    pad = max((epochs.max() - epochs.min()) * 0.06, 0.5) if len(epochs) else 0.5
    for ax, p in zip(axes.flat, panels):
        if isinstance(p, str):
            _loss_panel(ax, h, snaps_b, color)
        else:
            g = vals[vals.base.eq(base) & vals.task.eq(p.task) & vals.protocol.eq(p.protocol) & vals.metric.eq(p.metric)]
            tg = None
            if train_vals is not None:
                tg = train_vals[train_vals.base.eq(base) & train_vals.task.eq(p.task) & train_vals.protocol.eq(p.protocol)
                                & train_vals.metric.eq(p.metric) & train_vals.kind.eq(S.SNAPSHOT)]
            _task_panel(ax, g, p, tg, color, p.unit)
            ax.set_title(row_label(p.task, p.protocol, p.metric), fontsize=8.8)
        if len(epochs):
            ax.set_xlim(epochs.min() - pad, epochs.max() + pad)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7.5)
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    for ax in axes[-1]:
        if ax.axison:
            ax.set_xlabel("training epoch", fontsize=8)
    handles = [Line2D([], [], color=color, lw=1.8, marker="o", ms=3.6),
               Line2D([], [], color=color, lw=0, marker="*", ms=11, markeredgecolor="white"),
               Line2D([], [], color=INK, lw=0, marker="o", ms=10, markerfacecolor="none"),
               Line2D([], [], color=BASELINE_GREY, lw=1.0, ls="--")]
    labels = ["snapshot (weights at that epoch, with its 95% interval)", "weights the training kept",
              "best epoch of the task", "best baseline"]
    if train_vals is not None and len(train_vals[train_vals.base.eq(base)]):
        handles.append(Line2D([], [], color=color, lw=1.0, ls=":"))
        labels.append("same snapshot on training data")
    fig.legend(handles, labels, loc="upper right", ncol=min(len(handles), 3 if fig_w < 12 else 5), fontsize=7.5,
               bbox_to_anchor=(0.995, 1 - (header_height(lines) - 0.12) / fig.get_figheight()))
    wrapped_header(fig, f"{base}: which epoch does best?", lines)
    fig.tight_layout(rect=(0, 0, 1, 1 - (top_in + (0.2 if fig_w < 12 and len(handles) > 3 else 0))
                           / fig.get_figheight()))
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in base)
    save(fig, out, f"40_snapshots_{safe}", formats)


# --------------------------------------------------------------------------- 41 heatmap
def fig_heatmap(vals, summary, snaps, split, out, formats):
    bases = list(dict.fromkeys(summary.base))
    blocks = []
    for b in bases:
        rows = summary[summary.base.eq(b)]
        cols = snaps[snaps.base.eq(b)].sort_values(["epoch", "kind"], na_position="last")
        blocks.append((b, rows, cols))
    label_w = 0.075 * max(len(row_label(r.task, r.protocol, r.metric)) for r in summary.itertuples()) + 0.6
    widest = max(len(c) for _, _, c in blocks)
    cell_w = 0.62 if widest <= 14 else 0.38
    fig_w = max(label_w + cell_w * widest + 0.6, 8.5)
    split_name = "validation" if split == "val" else "test"
    lines = subtitle_lines(
        f"Scored on the {split_name} split. In each row, colour runs from that task's worst epoch (light) to its "
        f"best (dark); the number is the value. Thick border: best; thin border: inside the best one's 95% "
        f"interval. ★ = the weights the training kept.", fig_w)
    heights = [0.3 * len(r) + 0.85 for _, r, _ in blocks]
    fig_h = sum(heights) + header_height(lines) + 0.3
    fig = plt.figure(figsize=(fig_w, fig_h))
    top = 1 - (header_height(lines) + 0.05) / fig_h
    for (b, rows, cols), hgt in zip(blocks, heights):
        ax = fig.add_axes((label_w / fig_w, top - (hgt - 0.5) / fig_h, cell_w * len(cols) / fig_w, (hgt - 0.85) / fig_h))
        top -= hgt / fig_h
        M = np.full((len(rows), len(cols)), np.nan)
        V = np.full_like(M, np.nan)
        for i, r in enumerate(rows.itertuples()):
            g = vals[vals.base.eq(b) & vals.task.eq(r.task) & vals.protocol.eq(r.protocol) & vals.metric.eq(r.metric)]
            v = g.set_index("model").value
            row = np.array([v.get(m, np.nan) for m in cols.model], float)
            V[i] = row
            if np.isfinite(row).sum() >= 1:
                lo, hi = np.nanmin(row), np.nanmax(row)
                if hi > lo:                                   # 1 = this row's best epoch, 0 = its worst
                    good = (row - lo) / (hi - lo)
                    M[i] = good if r.higher_is_better else 1 - good
                else:
                    M[i] = np.where(np.isfinite(row), 0.5, np.nan)
        ax.pcolormesh(np.ma.masked_invalid(M), cmap=SEQUENTIAL, vmin=0, vmax=1, edgecolors="white", linewidth=1.5)
        ax.set_xlim(0, len(cols))
        ax.set_ylim(len(rows), 0)
        show_text = len(cols) <= 14
        for i, r in enumerate(rows.itertuples()):
            tied = {int(x) for x in str(r.tied_epochs).split()} if isinstance(r.tied_epochs, str) and r.tied_epochs else set()
            for j, c in enumerate(cols.itertuples()):
                if not np.isfinite(V[i, j]):
                    continue
                is_best = c.model == r.best_model
                if is_best or (pd.notna(c.epoch) and int(c.epoch) in tied):
                    ax.add_patch(Rectangle((j + 0.06, i + 0.08), 0.88, 0.84, fill=False, edgecolor=INK,
                                           linewidth=1.8 if is_best else 0.7, zorder=3))
                if show_text:
                    ax.text(j + 0.5, i + 0.5, fmt(V[i, j], r.unit), ha="center", va="center", fontsize=6.2,
                            color="white" if M[i, j] > 0.62 else INK, zorder=4)
        ax.set_yticks(np.arange(len(rows)) + 0.5)
        ax.set_yticklabels([row_label(r.task, r.protocol, r.metric) for r in rows.itertuples()], fontsize=7.5)
        ax.set_xticks(np.arange(len(cols)) + 0.5)
        xl = [("★" if k else "") + (str(int(e)) if pd.notna(e) else "?") + ("\nckpt" if kind == S.CHECKPOINT else "")
              for e, k, kind in zip(cols.epoch, cols.kept, cols.kind)]
        ax.set_xticklabels(xl, fontsize=7 if len(cols) <= 14 else 6, rotation=0 if len(cols) <= 20 else 90)
        ax.tick_params(length=0)
        for side in ("left", "bottom", "top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_title(f"{b}  ·  training epoch of each snapshot", fontsize=9.5, loc="left")
    wrapped_header(fig, "Snapshots side by side", lines)
    save(fig, out, "41_snapshot_heatmap", formats)


# --------------------------------------------------------------------------- 42 overall ranking
def fig_ranking(overall, snaps, split, colors, out, formats):
    bases = list(dict.fromkeys(overall.base))
    ncols = min(3, len(bases))
    nrows = math.ceil(len(bases) / ncols)
    split_name = "validation" if split == "val" else "test"
    fig_w = max(4.6 * ncols, 8.0)
    lines = subtitle_lines(
        f"Position of each snapshot among its model's evaluated epochs ({split_name} split; top = best on every "
        f"task), averaged within each task family and then over the families. Dashed: the same epochs ordered by "
        f"validation loss alone - where the two lines disagree, the loss did not pick the epoch that does best on "
        f"the tasks.", fig_w)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, 3.2 * nrows + header_height(lines) + 0.4), squeeze=False)
    labels = []                                               # placed once the layout is final
    for ax, b in zip(axes.flat, bases):
        g = overall[overall.base.eq(b)].sort_values("epoch")
        c = colors[b]
        snap = g[g.kind.eq(S.SNAPSHOT)]
        ax.plot(snap.epoch, snap.position, color=c, lw=1.9, marker="o", ms=4, zorder=3)
        curves = [(snap.epoch, snap.position)]
        if g.val_loss_position.notna().sum() > 1:
            vv = g[g.val_loss_position.notna()]
            ax.plot(vv.epoch, vv.val_loss_position, color=BASELINE_GREY, lw=1.1, ls="--", marker="o", ms=2.5,
                    zorder=2)
            curves.append((vv.epoch, vv.val_loss_position))
        kept = g[g.kept & g.epoch.notna()]
        if len(kept):
            ax.scatter(kept.epoch, kept.position, marker="*", s=140, color=c, edgecolor="white", linewidth=0.8, zorder=5)
        best = S.best_overall(overall, b)
        if best is not None and pd.notna(best.epoch):
            ax.scatter([best.epoch], [best.position], s=190, facecolor="none", edgecolor=INK, linewidth=1.3, zorder=6)
            labels.append((ax, best, curves, [(kept.epoch, kept.position)]))
        ax.set_ylim(1.08, -0.08)                              # top = best
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_yticklabels(["best", "", "middle", "", "worst"])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        ax.set_title(b, fontsize=9.5)
        ax.set_xlabel("training epoch", fontsize=8)
        ax.grid(True)
        ax.set_axisbelow(True)
    for ax in list(axes.flat)[len(bases):]:
        ax.axis("off")
    handles = [Line2D([], [], color=INK2, lw=1.9, marker="o", ms=4),
               Line2D([], [], color=BASELINE_GREY, lw=1.1, ls="--", marker="o", ms=2.5),
               Line2D([], [], color=INK2, lw=0, marker="*", ms=11, markeredgecolor="white"),
               Line2D([], [], color=INK, lw=0, marker="o", ms=10, markerfacecolor="none")]
    fig.legend(handles, ["average over the task families", "by validation loss alone", "weights the training kept",
                         "best overall"], loc="upper right", ncol=4, fontsize=7.5,
               bbox_to_anchor=(0.995, 1 - (header_height(lines) - 0.12) / fig.get_figheight()))
    wrapped_header(fig, "Which epoch is best overall", lines)
    fig.tight_layout(rect=(0, 0, 1, 1 - (header_height(lines) + 0.3) / fig.get_figheight()))
    for ax, best, curves, marks in labels:
        _place_label(ax, best.epoch, best.position,
                     f"best overall: {int(best.epoch)}\n(best or tied on {int(best.best_or_tied_on)}/{int(best.tasks)})",
                     curves, marks, fontsize=7, color=INK2, zorder=7, bbox=LABEL_BOX)
    save(fig, out, "42_snapshot_ranking", formats)


# --------------------------------------------------------------------------- main
def print_table(summary, overall):
    for b, g in summary.groupby("base", sort=False):
        best = S.best_overall(overall, b)
        head = f"\n{b}: best overall epoch {int(best.epoch)} ({best.model})" if best is not None else f"\n{b}"
        print(head)
        for r in g.itertuples():
            kept = "" if not np.isfinite(r.kept_value) else f"   kept: {fmt(r.kept_value, r.unit)}"
            print(f"  {row_label(r.task, r.protocol, r.metric)[:48]:48s} best epoch {str(r.best_epoch):>6s}"
                  f"  {fmt(r.best_value, r.unit):>10s}{kept}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", help="run folder written by `mobeval snapshots` (or its leaderboard.csv)")
    ap.add_argument("--out", default="figures_snapshots")
    ap.add_argument("--format", default="png", help="comma-separated: png,pdf,svg")
    ap.add_argument("--models", nargs="*", help="only these models")
    ap.add_argument("--families", nargs="*", help="only these task families (recovery location continuous "
                                                   "anomaly retrieval generation identity classification)")
    a = ap.parse_args(argv)
    lb, snaps, hist, train, split = load_run(Path(a.run))
    if a.models:
        snaps = snaps[snaps.base.isin(a.models)]
        if snaps.empty:
            raise SystemExit(f"none of {a.models} in this run")
    if a.families:
        lb = lb[lb.family.isin(a.families)]
    summary, overall = S.summarize(lb, snaps)
    if summary.empty:
        raise SystemExit("no primary-metric results for the snapshots in this run")
    summary = ordered_rows(summary)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    formats = [f.strip() for f in a.format.split(",") if f.strip()]
    style()
    vals = values(lb, snaps)
    train_vals = values(train, snaps) if train is not None else None
    colors = base_colors(list(dict.fromkeys(summary.base)))
    print(f"figures -> {out}/")
    for b in colors:
        fig_model(b, vals, summary, snaps, hist, train_vals, split, colors[b], out, formats)
    fig_heatmap(vals, summary, snaps, split, out, formats)
    fig_ranking(overall, snaps, split, colors, out, formats)
    summary.drop(columns=["stem"], errors="ignore").to_csv(out / "best_snapshots.csv", index=False)
    print_table(summary, overall)
    if split != "val":
        print("\nnote: scored on the TEST split - the best epochs' test scores are optimistic")
    return 0


if __name__ == "__main__":
    sys.exit(main())
