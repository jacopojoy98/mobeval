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


def _agg(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["family", "task", "metric", "protocol", "model"], dropna=False)
    out = g.agg(value=("value", "mean"), std=("value", "std"), ci_low=("ci_low", "mean"),
                ci_high=("ci_high", "mean"), skill=("skill", "mean"), skill_ci_low=("skill_ci_low", "mean"),
                skill_ci_high=("skill_ci_high", "mean"), n_runs=("value", "size"), n=("n", "max"),
                baseline=("baseline", "first"), higher_is_better=("higher_is_better", "first"),
                unit=("unit", "first")).reset_index()
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
    q = pd.DataFrame({"mean_skill": fam.mean(1)})
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
        out += ["## Summary: median headline skill per task family (%, clipped to ±100)", "", (fs * 100).round(1).to_markdown(), ""]
    pf = pareto_front(df)
    if not pf.empty and pf[["n_parameters", "latency_ms"]].notna().any().any():
        out += ["## Efficiency (Pareto front)", "", pf.to_markdown(index=False), ""]
    a = _agg(df)
    for fam in ["recovery", "location", "continuous", "classification", "generation", "efficiency"]:
        sub = a[a.family == fam]
        if sub.empty:
            continue
        out += [f"## {fam.capitalize()}", "", wide_table(sub, fam != "efficiency").to_markdown(), ""]
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
