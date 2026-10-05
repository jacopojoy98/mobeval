"""Leaderboards and a Markdown report.

Aggregation rules:
  * values are averaged over eval seeds AND run tags (training seeds); the spread
    across them is reported as +/- std, separately from the bootstrap CI;
  * cross-family summaries only average UNIT-FREE skill scores of type
    'ratio' / 'bounded' / 'floor' - never raw values, never nats differences;
  * efficiency is shown as a Pareto front, not collapsed into one scalar.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from .metrics.registry import HEADLINE, get_spec


def md_table(df: pd.DataFrame, index: bool = True) -> str:
    """Markdown table via tabulate when available, otherwise a plain pipe table."""
    try:
        return df.to_markdown(index=index)
    except ImportError:
        d = df.reset_index() if index else df
        cols = [str(c) for c in d.columns]
        rows = [[("" if pd.isna(v) else f"{v:.4g}" if isinstance(v, float) else str(v)) for v in r]
                for r in d.itertuples(index=False)]
        w = [max(len(c), *(len(r[i]) for r in rows)) if rows else len(c) for i, c in enumerate(cols)]
        line = lambda vals: "| " + " | ".join(v.ljust(w[i]) for i, v in enumerate(vals)) + " |"
        return "\n".join([line(cols), "|" + "|".join("-" * (x + 2) for x in w) + "|"] + [line(r) for r in rows])


def _agg(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["family", "task", "metric", "protocol", "model"], dropna=False)
    out = g.agg(value=("value", "mean"), std=("value", "std"), ci_low=("ci_low", "mean"),
                ci_high=("ci_high", "mean"), skill=("skill", "mean"), skill_ci_low=("skill_ci_low", "mean"),
                skill_ci_high=("skill_ci_high", "mean"), n_runs=("value", "size"), n=("n", "max"),
                baseline=("baseline", "first"), higher_is_better=("higher_is_better", "first"),
                unit=("unit", "first"), **({"paper_match": ("paper_match", "first")}
                                            if "paper_match" in df.columns else {})).reset_index()
    if "paper_match" not in out.columns:
        out["paper_match"] = ""
    out["paper_match"] = out["paper_match"].fillna("")
    return out


def leaderboard(store_or_df, family: Optional[str] = None) -> pd.DataFrame:
    df = store_or_df.to_frame() if hasattr(store_or_df, "to_frame") else store_or_df
    a = _agg(df)
    if family:
        a = a[a.family == family]
    return a


def _fmt(v, sd=None, unit=""):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    av = abs(v)
    s = f"{v:.3g}" if av < 1000 else f"{v:,.0f}"
    if sd is not None and np.isfinite(sd) and sd > 0:
        s += f" ±{sd:.2g}" if abs(sd) < 1000 else f" ±{sd:,.0f}"
    return s


def wide_table(a: pd.DataFrame, show_skill: bool = True) -> pd.DataFrame:
    cells = []
    for _, r in a.iterrows():
        txt = _fmt(r.value, r["std"])
        if show_skill and pd.notna(r.skill):
            txt += f" [{100 * r.skill:+.0f}%]" if get_spec(r.metric).skill != "difference" else f" [Δ{r.skill:+.2f}]"
        arrow = "↑" if r.higher_is_better else "↓"
        txt += PAPER_MARK.get(r.get("paper_match", ""), "")
        cells.append((f"{r.task} · {r.metric} {arrow} ({r.unit})", r.protocol, r.model, txt))
    t = pd.DataFrame(cells, columns=["metric", "protocol", "model", "cell"])
    return t.pivot_table(index=["metric", "protocol"], columns="model", values="cell", aggfunc="first").fillna("n/a")


def family_summary(df: pd.DataFrame, clip: float = 1.0) -> pd.DataFrame:
    """Median over HEADLINE metrics of unit-free skill, per family and model
    (baselines excluded). Skills are clipped to [-clip, clip] so one exploding
    ratio (baseline near perfect) cannot dominate; raw skills stay in the tables."""
    d = df[~df.model.str.startswith("baseline:") & df.skill.notna()].copy()
    d = d[d.metric.map(lambda m: get_spec(m).skill in ("ratio", "bounded", "floor") and m.split(":")[0] in HEADLINE)]
    if d.empty:
        return pd.DataFrame()
    d["skill"] = d.skill.clip(-clip, clip)
    per_metric = d.groupby(["model", "family", "task", "metric"]).skill.mean().reset_index()
    return per_metric.groupby(["model", "family"]).skill.median().unstack("family")


def pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    """Models not dominated on (mean skill ↑, parameters ↓, mean latency ↓)."""
    fam = family_summary(df)
    if fam.empty:
        return pd.DataFrame()
    q = pd.DataFrame({"mean_skill": fam.mean(axis=1)})
    eff = df[df.family == "efficiency"]
    q["n_parameters"] = eff[eff.metric == "n_parameters"].groupby("model").value.mean()
    q["latency_ms"] = eff[eff.metric == "latency_ms_per_sample"].groupby("model").value.mean()
    q = q.reset_index()
    def dominated(i):
        a = q.iloc[i]
        for j in range(len(q)):
            if j == i:
                continue
            b = q.iloc[j]
            ge = [b.mean_skill >= a.mean_skill]
            for c in ("n_parameters", "latency_ms"):
                if pd.notna(a[c]) and pd.notna(b[c]):
                    ge.append(b[c] <= a[c])
            gt = b.mean_skill > a.mean_skill or any(pd.notna(a[c]) and pd.notna(b[c]) and b[c] < a[c]
                                                    for c in ("n_parameters", "latency_ms"))
            if all(ge) and gt:
                return True
        return False
    q["pareto_optimal"] = [not dominated(i) for i in range(len(q))]
    return q.sort_values("mean_skill", ascending=False)


# ★ = the metric the model's paper reported, computed the same way; ☆ = the same quantity under a
# different protocol (see paper_metrics.py and the "Paper metrics" section of the report).
PAPER_MARK = {"exact": " ★", "near": " ☆"}

TITLES = {"retrieval": "Trajectory retrieval (from embeddings)",
          "identity": "User identification (from frozen embeddings)",
          "anomaly": "Anomaly detection (injected anomalies)"}

# Printed under the section heading. These two families are the easiest in the whole report to
# over-read, so the caveat travels with the numbers instead of living only in the docs.
NOTES = {
    "retrieval": "Each query's answer is its own window among `retrieval_db_size` test windows. "
                 "**odd_even** (every embedding model): query = odd-indexed points, database = even-indexed "
                 "points. **cross_modal** / **condition** (OmniTraj): query = the window's topology, road "
                 "segments or regions. Compare against **hausdorff**, a strong non-learned baseline here: odd "
                 "and even points of one trip are metres apart, so a model must beat plain geometry to add "
                 "anything.",
    "identity": "Closed-set re-identification with a linear probe on frozen embeddings. Read every "
                "model against **mean_location**, not against chance: individual mobility is largely "
                "home and work location, so an embedding that merely records position re-identifies "
                "users well without having learned anything about behaviour.",
    "anomaly": "Unsupervised: scorers are fitted on normal training data and the labels are used only "
               "to score. `teleport`, `speed` and `noise` change step lengths and are caught by any "
               "speed check - **max_step** is usually near 1.0 on them, so beating chance there means "
               "little. `detour` and `loop` preserve every step length exactly, so speed and distance "
               "statistics carry no signal at all (max_step ~0.54); they do insert a sharp turn, which "
               "**kinematic_knn** picks up (~0.60-0.64). Compare against kinematic_knn, not against "
               "0.5. A value clearly below 0.5 is also informative: it means the model finds the "
               "corrupted windows *more* predictable than real ones.",
}


# Train-vs-test reading: a skill this close to zero is "no better than the baseline" (bootstrap noise
# on a few thousand samples is of this order). The report table shows one metric per family.
FIT_MIN_SKILL = 0.02
FIT_METRICS = {"ade_m", "acc@1", "crps_min", "macro_f1", "roc_auc", "mrr", "user_acc@1"}


def train_vs_test(test_df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (task, metric, protocol, model) scored on both splits: value and skill on the
    train sample and on the test sample, and a reading of the pair.

      not fitting       no better than the baseline on its own training data (skill <= FIT_MIN_SKILL)
      not generalising  beats the baseline on train, but not on test - or loses over half its skill
      consistent        similar on both
    Skill is against the task's reference baseline on the same samples, so it is comparable across
    splits even when the raw values are not (train and test samples differ in difficulty)."""
    keys = ["family", "task", "metric", "protocol", "model"]
    a, b = _agg(train_df), _agg(test_df)
    m = a.merge(b, on=keys, suffixes=("_train", "_test"))
    m = m[~m.model.str.startswith("baseline:")]
    if m.empty:
        return m

    def reading(r):
        st, se = r.skill_train, r.skill_test
        if pd.isna(st) or pd.isna(se):
            return ""
        if st <= FIT_MIN_SKILL:
            return "not fitting"
        if se <= FIT_MIN_SKILL or se < 0.5 * st:
            return "not generalising"
        return "consistent"
    m["reading"] = m.apply(reading, axis=1)
    cols = keys + ["value_train", "value_test", "skill_train", "skill_test", "baseline_train", "unit_train",
                   "higher_is_better_train", "n_train", "n_test", "reading"]
    return m[cols].rename(columns={"baseline_train": "baseline", "unit_train": "unit",
                                   "higher_is_better_train": "higher_is_better"}).reset_index(drop=True)


def training_curves(histories: dict) -> pd.DataFrame:
    """histories: model name -> list of {epoch, train_loss, val_loss} (a checkpoint's history).
    One row per model: how far training went and whether either loss moved."""
    rows = []
    for name, h in histories.items():
        h = [e for e in (h or []) if e.get("train_loss") is not None]
        if not h:
            continue
        best = min(h, key=lambda e: e.get("val_loss", float("inf")))
        t0, t1, v0 = h[0]["train_loss"], h[-1]["train_loss"], h[0].get("val_loss")
        rows.append({"model": name, "epochs_run": h[-1]["epoch"], "best_epoch": best["epoch"],
                     "train_loss_first": t0, "train_loss_last": t1,
                     "train_loss_change": (t1 - t0) / abs(t0) if t0 else float("nan"),
                     "val_loss_first": v0, "val_loss_best": best.get("val_loss"),
                     "val_loss_last": h[-1].get("val_loss"),
                     "val_loss_change": (best["val_loss"] - v0) / abs(v0) if v0 else float("nan")})
    return pd.DataFrame(rows)


def fit_section(cmp: pd.DataFrame, curves: Optional[pd.DataFrame] = None) -> str:
    """Markdown for the report: headline metrics on train vs test, and the training curves."""
    out = ["## Fit on the training data", "",
           "The same tasks scored on a sample of the TRAIN split (`eval.train_eval`). Skill is against the "
           "task's reference baseline on the same samples. **not fitting**: no better than the baseline on its "
           "own training data. **not generalising**: better than the baseline on train but not on test, or "
           "less than half the train skill left on test. One metric per task; all of them are in "
           "train_vs_test.csv. Baselines fitted on train are also scored in-sample "
           "here, so a train skill near zero is a strong sign that the model has not learned the task.", ""]
    if cmp is not None and len(cmp):
        h = cmp[cmp.metric.map(lambda m: m.split(":")[0] in FIT_METRICS)].copy()
        h = h if len(h) else cmp.copy()
        t = pd.DataFrame({"task": h.task, "metric": h.metric, "model": h.model, "protocol": h.protocol,
                          "train": [_fmt(v, unit=u) for v, u in zip(h.value_train, h.unit)],
                          "test": [_fmt(v, unit=u) for v, u in zip(h.value_test, h.unit)],
                          "skill train": h.skill_train.map(lambda s: "" if pd.isna(s) else f"{s:+.2f}"),
                          "skill test": h.skill_test.map(lambda s: "" if pd.isna(s) else f"{s:+.2f}"),
                          "reading": h.reading})
        out += [md_table(t, index=False), ""]
        counts = h.reading[h.reading != ""].value_counts()
        if len(counts):
            out += ["Rows: " + ", ".join(f"{n} {k}" for k, n in counts.items()) + ".", ""]
    if curves is not None and len(curves):
        c = curves.copy()
        for col in ("train_loss_change", "val_loss_change"):
            c[col] = c[col].map(lambda v: "" if pd.isna(v) else f"{v:+.0%}")
        for col in ("train_loss_first", "train_loss_last", "val_loss_first", "val_loss_best", "val_loss_last"):
            c[col] = c[col].map(lambda v: "" if v is None or pd.isna(v) else f"{v:.4g}")
        out += ["### Training curves (from the checkpoints)", "",
                "`best_epoch` is the epoch whose weights were kept. A train loss that barely moves means the "
                "optimisation is not working; a train loss that keeps falling while the validation loss stopped "
                "early (best_epoch far below epochs_run) is early stopping doing its job on a model that had "
                "started to overfit.", "", md_table(c, index=False), ""]
    return "\n".join(out)


def markdown_report(store, ctx=None, title: str = "Mobility foundation model evaluation") -> str:
    df = store.to_frame()
    out: List[str] = [f"# {title}", ""]
    if ctx is not None:
        out += ["## Setup", "```", ctx.summary(), f"eval_seeds={list(ctx.cfg.eval_seeds)}  n_boot={ctx.cfg.n_boot}",
                "```", ""]
    out += ["Cells: mean over eval seeds / run tags (± std across them). Brackets: skill vs. the "
            "task's primary baseline on identical samples ([+x%] = share of baseline error removed or gap to "
            "perfect closed; [Δ] = nats/sample better than baseline). 95% bootstrap CIs are in results.jsonl.", ""]
    fs = family_summary(df)
    if not fs.empty:
        out += ["## Summary: median headline skill per task family (%, clipped to ±100)", "", md_table((fs * 100).round(1)), ""]
    pf = pareto_front(df)
    if not pf.empty and pf[["n_parameters", "latency_ms"]].notna().any().any():
        out += ["## Efficiency (Pareto front)", "", md_table(pf, index=False), ""]
    a = _agg(df)
    # Ordered so related families read together; anything new in the registry is appended
    # rather than silently dropped from the report.
    order = ["recovery", "location", "continuous", "classification", "retrieval", "identity", "anomaly",
             "generation", "efficiency"]
    families = order + sorted(set(a.family.dropna()) - set(order))
    for fam in families:
        sub = a[a.family == fam]
        if sub.empty:
            continue
        out += [f"## {TITLES.get(fam, fam.capitalize())}", ""]
        if fam in NOTES:
            out += [NOTES[fam], ""]
        out += [md_table(wide_table(sub, fam != "efficiency")), ""]
    if "paper_match" in df.columns and (df.paper_match.fillna("") != "").any():
        from .paper_metrics import describe
        out += ["## Paper metrics", "",
                "★ marks a cell whose metric is the one the model's authors reported, computed with the "
                "same formula and protocol; ☆ the same quantity under a protocol that differs as noted "
                "below. A mark is about the METRIC, not the data: the number is comparable with the "
                "paper's only on the paper's dataset and preprocessing (`recipe: paper`).", "",
                describe(), ""]
    flagged = df[df["flags"].map(len) > 0]
    if len(flagged):
        out += ["## Sanity flags", ""]
        for _, r in flagged.drop_duplicates(["model", "task", "metric"]).iterrows():
            out.append(f"- **{r.model}** {r.task} · {r.metric} = {r.value:.4g}: {'; '.join(r['flags'])}")
        out.append("")
    if ctx is not None and (ctx.skipped or ctx.errors):
        out += ["## Skipped / failed", ""]
        out += [f"- {m} · {t}: skipped ({why})" for m, t, why in ctx.skipped]
        out += [f"- {m} · {t}: **error** {e}" for m, t, e, _ in ctx.errors]
        out.append("")
    return "\n".join(out)
