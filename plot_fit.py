#!/usr/bin/env python
"""Figures for "did the models learn their training data, and does it carry over to the test data?"

    python plot_fit.py <run folder or train_vs_test.csv> --out figures
    python plot_fit.py <run folder> --checkpoints ~/results/checkpoints --out figures --format png,pdf
    python plot_fit.py --checkpoints ~/results/checkpoints --out figures     # curves only, also mid-training

Reads train_vs_test.csv, written by an evaluation with `eval.train_eval: true`, and optionally the
checkpoints' training logs (<name>.losses.csv, every epoch) or .json files for the training curves.
Writes:

  20_fit_train_vs_test   one panel per model, one row per task: skill on the TRAIN sample (open dot)
                         and on the TEST sample (filled dot), joined by a line coloured by the reading
  21_fit_scatter         every (model, task) at once: skill on train (x) against skill on test (y);
                         on the diagonal = the model does as well on new data as on its training data
  22_training_curves     one panel per checkpoint: training and validation loss per epoch, best epoch;
                         with the training log (<checkpoint>.losses.csv), also the training loss
                         measured like the validation loss - the curve to compare validation with
  23_loss_terms          one panel per term of each model's loss (e.g. TrajGPT: location, travel
                         time, duration; OmniTraj: each contrasted pair)
  24_learning_rate_and_gradients   learning rate and gradient norm before clipping, per epoch

Skill is the pipeline's score against the task's reference baseline, computed on the same samples,
so train and test are comparable even though the samples differ: 0 = as good as the baseline,
1 = perfect (errors: share of the baseline's error removed; scores: share of the gap to perfect
closed). The readings are the report's:
  consistent        similar skill on both
  not generalising  beats the baseline on train but not on test, or keeps less than half its skill
  not fitting       no better than the baseline even on its own training data
For probe results (linear_probe) the probe is itself fitted on training data, so part of a gap there
is the probe's own fit rather than the model's.

Needs plot_results.py in the same folder (shared colours and style).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from plot_results import (AXIS, FAMILY_ORDER, GRID, INK, INK2, METRIC_NAME, MODEL_ORDER, MUTED, PALETTE,
                          _task_rank, header, pretty_task, save, style)

# The report's headline metric per task (mobeval.report.FIT_METRICS) and its "not fitting" threshold.
FIT_METRICS = {"ade_m", "acc@1", "crps_min", "macro_f1", "roc_auc", "mrr", "user_acc@1"}
FIT_MIN_SKILL = 0.02
READING_COLOR = {"consistent": "#2a78d6", "not generalising": "#e34948", "not fitting": "#eda100", "": MUTED}
READING_ORDER = ["consistent", "not generalising", "not fitting"]
FAMILY_MARKER = {"recovery": "o", "location": "s", "continuous": "^", "retrieval": "D", "classification": "P",
                 "identity": "X", "anomaly": "v", "generation": "*"}
LO, HI = -1.0, 1.0                     # skill axis; values beyond are drawn at the edge as arrowheads


# --------------------------------------------------------------------------- data
def load_fit(path: Path, models=None, all_variants: bool = False) -> pd.DataFrame:
    path = Path(path).expanduser()
    if path.is_dir():
        path = path / "train_vs_test.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found: run the evaluation with `eval.train_eval: true` (the run folder "
                         f"then holds train_vs_test.csv)")
    d = pd.read_csv(path)
    d = d[d.metric.str.split(":").str[0].isin(FIT_METRICS)].copy()
    if not all_variants:                                     # conditional variants of the same task
        d = d[~d.task.str.contains(r"\|given:", regex=True)]
    if models:
        d = d[d.model.isin(models)]
    d = d[np.isfinite(d.skill_train) & np.isfinite(d.skill_test)]
    d["reading"] = d.reading.fillna("")
    d["stem"] = d.task.str.replace("/linear_probe", "", regex=False)
    d["probe"] = d.protocol.eq("linear_probe")
    d["label"] = [f"{pretty_task(s)} · {METRIC_NAME.get(m.split(':')[0], m)}" + (" · probe" if p else "")
                  for s, m, p in zip(d.stem, d.metric, d.probe)]
    fam = {f: i for i, f in enumerate(FAMILY_ORDER)}
    rank = _task_rank(d.stem)
    d["order"] = [(fam.get(f, 99), rank[s], p) for f, s, p in zip(d.family, d.stem, d.probe)]
    return d.sort_values("order", kind="stable").reset_index(drop=True)


def colors_for(models) -> dict:
    known = MODEL_ORDER + [m for m in models if m not in MODEL_ORDER]
    return {m: PALETTE[min(i, len(PALETTE) - 1)] for i, m in enumerate(known)}


def ordered_models(names) -> list:
    seen = list(dict.fromkeys(names))
    return [m for m in MODEL_ORDER if m in seen] + [m for m in seen if m not in MODEL_ORDER]


def _clip(v):
    return float(np.clip(v, LO, HI)), ("<" if v < LO else ">" if v > HI else "o")


# --------------------------------------------------------------------------- 20 dumbbell per model
def fig_dumbbell(d: pd.DataFrame, out: Path, formats):
    models = ordered_models(d.model)
    ncols = 1 if len(models) == 1 else 2
    nrows = math.ceil(len(models) / ncols)
    per = [len(d[d.model == m]) for m in models]
    row_h = 0.24
    height = max(per) * row_h + 1.0
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, height * nrows + 1.1), squeeze=False)
    for ax, m in zip(axes.flat, models):
        g = d[d.model == m].reset_index(drop=True)
        ys = np.arange(len(g))[::-1]
        ax.axvspan(LO, FIT_MIN_SKILL, color="#fbf3dc", zorder=0, lw=0)
        ax.axvline(0, color=AXIS, lw=1.0, zorder=1)
        for y, r in zip(ys, g.itertuples()):
            (xt, mt), (xe, me) = _clip(r.skill_train), _clip(r.skill_test)
            c = READING_COLOR.get(r.reading, MUTED)
            ax.plot([xt, xe], [y, y], color=c, lw=2.0, alpha=0.75, zorder=2, solid_capstyle="round")
            ax.scatter([xt], [y], s=34, marker=mt, facecolor="white", edgecolor=c, linewidth=1.6, zorder=3)
            ax.scatter([xe], [y], s=34, marker=me, facecolor=c, edgecolor=c, linewidth=1.2, zorder=4)
        ax.set_yticks(ys)
        ax.set_yticklabels(g.label)
        ax.set_ylim(-0.7, len(g) - 0.3)
        ax.set_xlim(LO - 0.05, HI + 0.05)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.set_title(m)
        ax.set_xlabel("skill against the baseline (0 = baseline, 1 = perfect)")
        counts = g.reading.value_counts()
        ax.text(1.0, 1.01, "  ".join(f"{counts.get(k, 0)} {k}" for k in READING_ORDER), transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7.5, color=INK2)
    for ax in list(axes.flat)[len(models):]:
        ax.axis("off")
    handles = [Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=INK2, label="train sample"),
               Line2D([], [], marker="o", ls="", markerfacecolor=INK2, markeredgecolor=INK2, label="test sample")]
    handles += [Line2D([], [], color=READING_COLOR[k], lw=2.5, label=k) for k in READING_ORDER]
    handles += [Patch(facecolor="#fbf3dc", label=f"skill ≤ {FIT_MIN_SKILL:g}: no better than the baseline")]
    fig.legend(handles=handles, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.01), fontsize=8)
    header(fig, "Training data vs test data",
           "Skill on a sample of each model's training data (open) and on the test data (filled), against the "
           "task's baseline on the same samples. Values beyond ±1 are drawn at the edge (◄ ►).")
    fig.tight_layout(rect=(0, 0.04, 1, 1 - 0.75 / fig.get_figheight()))
    save(fig, out, "20_fit_train_vs_test", formats)


# --------------------------------------------------------------------------- 21 scatter
def fig_scatter(d: pd.DataFrame, out: Path, formats):
    models = ordered_models(d.model)
    colors = colors_for(models)
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    xs = np.linspace(FIT_MIN_SKILL, HI, 50)
    ax.fill_between([LO, FIT_MIN_SKILL], LO, HI, color="#fbf3dc", lw=0, zorder=0)
    ax.fill_between(xs, LO, np.maximum(0.5 * xs, FIT_MIN_SKILL), color="#fbe4e3", lw=0, zorder=0)
    ax.plot([LO, HI], [LO, HI], color=INK2, lw=1, ls="--", zorder=1)
    ax.axhline(0, color=AXIS, lw=0.9, zorder=1)
    ax.axvline(0, color=AXIS, lw=0.9, zorder=1)
    for r in d.itertuples():
        x, y = float(np.clip(r.skill_train, LO, HI)), float(np.clip(r.skill_test, LO, HI))
        mk = FAMILY_MARKER.get(r.family, "o")
        ax.scatter([x], [y], s=46, marker=mk, color=colors[r.model], alpha=0.85,
                   facecolor="white" if r.probe else colors[r.model], linewidth=1.4, zorder=3)
    ax.set_xlim(LO - 0.05, HI + 0.05)
    ax.set_ylim(LO - 0.05, HI + 0.05)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlabel("skill on the TRAIN sample")
    ax.set_ylabel("skill on the TEST data")
    ax.text(0.97, 0.93, "same skill on\nnew data", transform=ax.transAxes, ha="right", fontsize=7.5, color=INK2)
    ax.text(0.97, 0.08, "learned on train,\ndoes not carry over", transform=ax.transAxes, ha="right",
            fontsize=7.5, color="#b0302e")
    ax.text(0.03, 0.97, "not even fitting\nthe training data", transform=ax.transAxes, ha="left", va="top",
            fontsize=7.5, color="#9a6b00")
    mh = [Line2D([], [], marker="o", ls="", color=colors[m], label=m) for m in models]
    fams = [f for f in FAMILY_ORDER if f in set(d.family)]
    fh = [Line2D([], [], marker=FAMILY_MARKER.get(f, "o"), ls="", color=INK2, label=f) for f in fams]
    fh += [Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=INK2, label="probe result")]
    leg = ax.legend(handles=mh, loc="upper left", bbox_to_anchor=(1.02, 1.0), title="model", fontsize=8)
    ax.add_artist(leg)
    ax.legend(handles=fh, loc="upper left", bbox_to_anchor=(1.02, 0.55), title="task", fontsize=8)
    header(fig, "Does what was learned carry over to new data?",
           "One point per model and task; values beyond ±1 drawn at the edge.", top=1.0)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.6 / fig.get_figheight()))
    save(fig, out, "21_fit_scatter", formats)


# --------------------------------------------------------------------------- 22-24 training curves
def _read_loss_log(f: Path) -> list:
    """<checkpoint>.losses.csv -> history entries (empty cells dropped)."""
    try:
        t = pd.read_csv(f)
    except (OSError, ValueError):
        return []
    rows = []
    for r in t.to_dict("records"):
        rows.append({k: v for k, v in r.items() if not (isinstance(v, float) and math.isnan(v))})
    return [e for e in rows if e.get("train_loss") is not None and e.get("epoch") is not None]


def load_histories(paths) -> list:
    """(label, history, complete) per checkpoint under `paths` (folders, .json or .losses.csv files).

    A checkpoint's <name>.losses.csv is written after EVERY epoch, its .json only when validation
    improves (and at the end): the CSV wins when it has more epochs (a running or killed job)."""
    found = {}
    for p in map(lambda s: Path(s).expanduser(), paths):
        files = sorted(p.glob("*.json")) + sorted(p.glob("*.losses.csv")) if p.is_dir() else [p]
        for f in files:
            if f.name.endswith(".losses.csv"):
                stem, h, complete = f.name[:-len(".losses.csv")], _read_loss_log(f), None
            else:
                try:
                    s = json.loads(f.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                stem, h, complete = f.stem, [e for e in (s.get("history") or []) if e.get("train_loss") is not None], \
                    s.get("complete", True)
            if not h:
                continue
            where = f.parent.parent.name if f.parent.name == "checkpoints" else f.parent.name
            key = (stem, str(f.parent.resolve()))
            old = found.get(key)
            if old is None:
                found[key] = [stem, where, h, complete]
            else:
                if len(h) > len(old[2]):
                    old[2] = h
                if complete is not None:
                    old[3] = complete
    found = [(n, w, h, (False if c is None else c)) for n, w, h, c in found.values()]
    if not found:
        return []
    names = [x[0] for x in found]
    dup = {n for n in names if names.count(n) > 1}
    return [(f"{n} ({w})" if n in dup else n, h, c) for n, w, h, c in found]


def _series(h, key):
    return np.array([float(e[key]) if key in e and e[key] is not None else np.nan for e in h], float)


def _terms(h) -> list:
    return list(dict.fromkeys(k.split(":", 1)[1] for e in h for k in e
                              if ":" in k and k.split(":", 1)[0] in ("train_loss", "train_eval_loss", "val_loss")))


def _term_title(term: str) -> str:
    """travel_time -> travel time; OmniTraj pairs: trajectory-topology_road -> trajectory vs topology+road."""
    if "-" in term:
        return " vs ".join(x.replace("_", "+") for x in term.split("-", 1))
    return term.replace("_", " ")


def _model_of(label):
    return label.split(" (")[0]


def _sorted_hist(hist):
    names = ordered_models([_model_of(h[0]) for h in hist])
    return sorted(hist, key=lambda x: (names.index(_model_of(x[0])), x[0])), colors_for(names)


def _log_if_wide(ax, *arrays):
    v = np.concatenate([a[np.isfinite(a)] for a in arrays]) if arrays else np.array([])
    if len(v) and v.min() > 0 and v.max() / v.min() > 30:
        ax.set_yscale("log")


def _loss_lines(ax, ep, tr, te, va, c, best_dot=True):
    ax.plot(ep, tr, color=MUTED, lw=1.1, ls=":", label="train, during the epoch")
    if np.isfinite(te).any():
        ax.plot(ep, te, color=c, lw=1.4, ls="--", label="train, measured like validation")
    ax.plot(ep, va, color=c, lw=2.0, label="validation")
    if best_dot and np.isfinite(va).any():
        b = int(np.nanargmin(va))
        ax.axvline(ep[b], color=c, lw=0.9, ls=":")
        ax.scatter([ep[b]], [va[b]], s=26, color=c, zorder=3, edgecolor="white", linewidth=1)
    _log_if_wide(ax, tr, te, va)
    ax.grid(True)
    ax.set_axisbelow(True)


def fig_curves(hist, out: Path, formats):
    hist, colors = _sorted_hist(hist)
    ncols = min(3, len(hist))
    nrows = math.ceil(len(hist) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.1 * nrows + 1.1), squeeze=False)
    for ax, (label, h, complete) in zip(axes.flat, hist):
        c = colors[_model_of(label)]
        ep = _series(h, "epoch")
        tr, te, va = _series(h, "train_loss"), _series(h, "train_eval_loss"), _series(h, "val_loss")
        _loss_lines(ax, ep, tr, te, va, c)
        ax.set_title(label)
        note = f"{int(ep[-1])} epochs"
        if np.isfinite(va).any():
            b = int(np.nanargmin(va))
            note += f", best {int(ep[b])}"
            if np.isfinite(te[b]) and te[b]:
                note += f", gap there {(va[b] - te[b]) / abs(te[b]):+.0%}"
        note += f" · {'finished' if complete else 'in progress / stopped'}"
        ax.text(0.0, 1.11, note, transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5, color=INK2)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for ax in list(axes.flat)[len(hist):]:
        ax.axis("off")
    handles = [Line2D([], [], color=MUTED, lw=1.1, ls=":"), Line2D([], [], color=INK2, lw=1.4, ls="--"),
               Line2D([], [], color=INK2, lw=2.0)]
    fig.legend(handles, ["train, averaged during the epoch (dropout on, weights moving)",
                         "train, measured like validation (fixed sample, eval mode)", "validation"],
               loc="upper right", ncol=3, fontsize=7.5, bbox_to_anchor=(0.995, 1 - 0.62 / fig.get_figheight()))
    header(fig, "Training curves", "Dot = best epoch, the weights kept in the checkpoint. Compare validation "
                                   "with the dashed line: the gap between them (above each panel, at the best "
                                   "epoch) is what the model does not carry over to unseen trips.")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.95 / fig.get_figheight()))
    save(fig, out, "22_training_curves", formats)


def fig_terms(hist, out: Path, formats, per_row: int = 4):
    """One panel per term of each model's loss (TrajGPT: location, travel time, duration; ...)."""
    hist, colors = _sorted_hist(hist)
    blocks = [(label, h, _terms(h)) for label, h, _ in hist]
    blocks = [b for b in blocks if b[2]]
    if not blocks:
        return False
    rows = [(label, h, terms[i:i + per_row]) for label, h, terms in blocks for i in range(0, len(terms), per_row)]
    ncols = min(per_row, max(len(r[2]) for r in rows))
    fig, axes = plt.subplots(len(rows), ncols, figsize=(3.4 * ncols, 2.55 * len(rows) + 1.2), squeeze=False)
    for (label, h, terms), axrow in zip(rows, axes):
        c = colors[_model_of(label)]
        ep = _series(h, "epoch")
        for ax, term in zip(axrow, terms):
            _loss_lines(ax, ep, _series(h, f"train_loss:{term}"), _series(h, f"train_eval_loss:{term}"),
                        _series(h, f"val_loss:{term}"), c, best_dot=False)
            va = _series(h, "val_loss")
            if np.isfinite(va).any():                     # the epoch kept, chosen on the TOTAL validation loss
                ax.axvline(ep[int(np.nanargmin(va))], color=c, lw=0.9, ls=":")
            ax.set_title(f"{label} · {_term_title(term)}", fontsize=9)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.tick_params(labelsize=7.5)
        for ax in axrow[len(terms):]:
            ax.axis("off")
    for ax in axes[-1]:
        if ax.axison:
            ax.set_xlabel("epoch", fontsize=8.5)
    handles = [Line2D([], [], color=MUTED, lw=1.1, ls=":"), Line2D([], [], color=INK2, lw=1.4, ls="--"),
               Line2D([], [], color=INK2, lw=2.0)]
    fig.legend(handles, ["train, during the epoch", "train, measured like validation", "validation"],
               loc="upper right", ncol=3, fontsize=7.5, bbox_to_anchor=(0.995, 1 - 0.62 / fig.get_figheight()))
    header(fig, "The terms of each loss", "Each term as it enters the model's loss (before any weight); dotted "
                                          "line = the epoch kept. A term that stays flat is a part of the task "
                                          "the model is not learning.")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.95 / fig.get_figheight()))
    save(fig, out, "23_loss_terms", formats)
    return True


def fig_optimisation(hist, out: Path, formats):
    """Learning rate (top) and gradient norm before clipping (bottom), per model: two panels, one scale each."""
    hist, colors = _sorted_hist(hist)
    hist = [x for x in hist if np.isfinite(_series(x[1], "lr")).any() or np.isfinite(_series(x[1], "grad_norm")).any()]
    if not hist:
        return False
    n = len(hist)
    fig, axes = plt.subplots(2, n, figsize=(max(3.3 * n, 6.5), 5.4), squeeze=False, sharex="col")
    for j, (label, h, _) in enumerate(hist):
        c = colors[_model_of(label)]
        ep, lr = _series(h, "epoch"), _series(h, "lr")
        gn, gmax = _series(h, "grad_norm"), _series(h, "grad_norm_max")
        a, b = axes[0, j], axes[1, j]
        a.plot(ep, lr, color=c, lw=2.0)
        if np.isfinite(lr).any() and np.nanmin(lr) > 0:
            a.set_yscale("log")
        a.set_title(label, fontsize=9.5)
        b.plot(ep, gn, color=c, lw=2.0, label="mean over the steps")
        if np.isfinite(gmax).any():
            b.plot(ep, gmax, color=c, lw=1.0, ls="--", label="largest step")
        _log_if_wide(b, gn, gmax)
        for ax in (a, b):
            ax.grid(True)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=7.5)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        b.set_xlabel("epoch")
        if j == 0:
            a.set_ylabel("learning rate")
            b.set_ylabel("gradient norm")
    handles = [Line2D([], [], color=INK2, lw=2.0), Line2D([], [], color=INK2, lw=1.0, ls="--")]
    fig.legend(handles, ["gradient norm, mean over the epoch's steps", "largest step"], loc="upper right",
               ncol=2, fontsize=7.5, bbox_to_anchor=(0.995, 1 - 0.62 / fig.get_figheight()))
    header(fig, "Learning rate and gradient size", "Learning rate used in each epoch, and the size of the "
                                                   "gradient before clipping. A loss that stalls with a tiny "
                                                   "learning rate or vanishing gradients is an optimisation problem, "
                                                   "not a data one.")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.95 / fig.get_figheight()))
    save(fig, out, "24_learning_rate_and_gradients", formats)
    return True


# --------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", nargs="?", help="run folder holding train_vs_test.csv, or the csv itself")
    ap.add_argument("--checkpoints", nargs="*", default=[],
                    help="checkpoint folders (or their .json / .losses.csv files) for the training curves")
    ap.add_argument("--out", default="figures", help="output directory (default: figures)")
    ap.add_argument("--format", default="png", help="comma-separated: png, pdf, svg (default: png)")
    ap.add_argument("--models", nargs="*", help="only these models")
    ap.add_argument("--all-variants", action="store_true",
                    help="also draw the conditional variants of a task (e.g. duration given location)")
    a = ap.parse_args(argv)
    if not a.run and not a.checkpoints:
        ap.error("give a run folder (train_vs_test.csv) and/or --checkpoints")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    formats = [f.strip() for f in a.format.split(",") if f.strip()]
    style()
    print(f"writing to {out}/")
    if a.run:
        d = load_fit(a.run, a.models, a.all_variants)
        if d.empty:
            print("  train_vs_test.csv has no headline metric scored on both splits")
        else:
            fig_dumbbell(d, out, formats)
            fig_scatter(d, out, formats)
            summary = d.groupby(["model", "reading"]).size().unstack(fill_value=0)
            print(summary.reindex(columns=[c for c in READING_ORDER if c in summary.columns]).to_string())
    if a.checkpoints:
        hist = load_histories(a.checkpoints)
        if a.models:
            hist = [h for h in hist if h[0].split(" (")[0] in a.models]
        if hist:
            fig_curves(hist, out, formats)
            if not fig_terms(hist, out, formats):
                print("  no loss terms recorded (checkpoints trained before the training log): 23 skipped")
            if not fig_optimisation(hist, out, formats):
                print("  no learning rate / gradient norm recorded: 24 skipped")
        else:
            print("  no checkpoint histories found")


if __name__ == "__main__":
    main()
