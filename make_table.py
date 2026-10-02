#!/usr/bin/env python3
"""Build LaTeX result tables from a mobeval leaderboard.csv.

    python make_table.py leaderboard.csv --out results.tex

Design choices, so the table says what the numbers mean:
  * the protocol is part of the row label, never hidden - a linear-probe number must not be
    read as a native capability;
  * baselines are in the same table as the models, not in prose, because every skill score is
    relative to them and several of them win;
  * 95% bootstrap CIs are printed where available, since many differences here are not
    significant;
  * the best MODEL is bold and the best value overall (models and baselines together) is
    underlined, so "the best model" and "better than every baseline" stay distinguishable;
  * a star marks a cell whose metric is the one that model's own paper reported, computed the
    same way ($^\star$), and a dagger the same quantity under a different protocol ($^\dagger$).
    The marks come from the leaderboard's `paper_match` column (mobeval >= this version); for an
    older leaderboard pass --model-types so they can be looked up.
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

# (task, metric) -> (column header, unit note, number format)
SECTIONS = [
    ("Trajectory recovery and prediction (mean error on hidden points, m)", [
        ("recovery/random@0.5", "ade_m", r"Random 50\%", "{:,.0f}"),
        ("recovery/block@0.5", "ade_m", r"Block 50\%", "{:,.0f}"),
        ("recovery/block@0.75", "ade_m", r"Block 75\%", "{:,.0f}"),
        ("recovery/keep_every:8", "ade_m", "Every 8th kept", "{:,.0f}"),
        ("recovery/last:5", "ade_m", "Last 5", "{:,.0f}"),
    ]),
    ("Next-location prediction", [
        ("next_location", "acc@1", "Acc@1", "{:.4f}"),
        ("next_location", "acc@5", "Acc@5", "{:.4f}"),
        ("next_location", "acc@10", "Acc@10", "{:.4f}"),
        ("next_location", "acc@20", "Acc@20", "{:.4f}"),
        ("next_location", "median_dist_err_m", "Median err.", "{:,.0f}"),
    ]),
    # CRPS in minutes: the mean absolute distance between the predicted DISTRIBUTION and the
    # observed value, so it rewards a well-placed and well-spread forecast, not just a good
    # point estimate. For a point forecast it reduces exactly to MAE, which is what makes the
    # models and the point baselines comparable on one number.
    ("Travel time and stay duration (CRPS, minutes)", [
        ("continuous/travel_time|<=4h", "crps_min", "Travel time", "{:.2f}"),
        ("continuous/duration", "crps_min", "Stay duration", "{:.1f}"),
    ]),
    # The same targets told where the next visit is (and, for duration, when it starts): TrajGPT's
    # teacher-forced evaluation, and close to TransferTraj's origin-destination travel time.
    # P(+-t) is the forecast's probability mass within t minutes of the truth.
    ("Travel time and stay duration given the next visit", [
        ("continuous/travel_time|given:location|<=4h", "mae_min", "Travel MAE", "{:.2f}"),
        ("continuous/travel_time|given:location|<=4h", "mape", "Travel MAPE", "{:.3f}"),
        ("continuous/travel_time|given:location|<=4h", "p_within_10min", r"Travel P$\pm$10", "{:.3f}"),
        ("continuous/duration|given:location+arrival", "p_within_10min", r"Stay P$\pm$10", "{:.3f}"),
        ("continuous/duration|given:location+arrival", "crps_min", "Stay CRPS", "{:.1f}"),
    ]),
    ("Anomaly detection (ROC-AUC)", [
        ("anomaly/teleport", "roc_auc", "Teleport", "{:.3f}"),
        ("anomaly/detour", "roc_auc", "Detour", "{:.3f}"),
        ("anomaly/loop", "roc_auc", "Loop", "{:.3f}"),
    ]),
    # Distributional realism of generated trajectories: Jensen-Shannon divergence between the
    # generated and the real distribution of each mobility statistic, on bins fixed from the
    # real data. Only models declaring `generation` appear.
    ("Generative quality (Jensen--Shannon divergence, bits)", [
        ("generation/radius_of_gyration", "jsd:radius_of_gyration", "Radius of gyr.", "{:.3f}"),
        ("generation/jump_length", "jsd:jump_length", "Jump length", "{:.3f}"),
        ("generation/stay_duration", "jsd:stay_duration", "Stay duration", "{:.3f}"),
        ("generation/daily_locations", "jsd:daily_locations", "Daily locations", "{:.3f}"),
        ("generation/memorisation", "copy_rate", "Copy rate", "{:.3f}"),
    ]),
]

# Metrics whose optimum is not an extreme, so "best" is meaningless: copy_rate should MATCH the
# real-data rate (~0.05), and 0.000 means the generator produces nothing like the training data
# at all - which is not a success.
NO_BEST = {"copy_rate"}

PRETTY = {"UniTraj-zeroshot": "UniTraj (zero-shot)", "UniTraj-finetuned": "UniTraj (fine-tuned)",
          "CLIPMobility": "CLIP-Mobility", "TransferTraj": "TransferTraj", "TrajGPT": "TrajGPT"}
BASE_PRETTY = {"baseline:linear_interp": "Linear interpolation", "baseline:last_observed": "Last observed",
               "baseline:user_frequent": "User frequency", "baseline:markov1": "Markov-1",
               "baseline:global_popular": "Global popularity", "baseline:train_marginal": "Train marginal",
               "baseline:kinematic_knn": "Kinematic $k$-NN", "baseline:max_step": "Max step length",
               "baseline:uniform_bbox": "Uniform bbox", "baseline:seed_only": "Seed only (no generation)",
               "baseline:real_noise_floor": "Real noise floor",
               "baseline:constant_velocity": "Constant velocity"}
# Superscripts for metrics the model's paper reported (see mobeval/paper_metrics.py).
PAPER_MARK = {"exact": r"$^{\star}$", "near": r"$^{\dagger}$"}
PROTO = {"native": "native", "linear_probe": "probe", "embedding_knn": "emb.\\ $k$-NN",
         "reconstruction": "recon.", "rollout": "rollout"}


def esc(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def cell(row, fmt, show_ci):
    if row is None or not np.isfinite(row["value"]):
        return "--"
    s = fmt.format(row["value"])
    if show_ci and np.isfinite(row.get("ci_low", np.nan)) and np.isfinite(row.get("ci_high", np.nan)):
        s += r"\,\scriptsize{[" + fmt.format(row["ci_low"]) + ", " + fmt.format(row["ci_high"]) + "]}"
    return s


def build(df: pd.DataFrame, show_ci: bool = False) -> str:
    # Known models first, in PRETTY's order, then anything else this run produced. A model must
    # never vanish from the table because it was renamed in the config.
    present = [m for m in df.model.unique() if not str(m).startswith("baseline:")]
    models = [m for m in PRETTY if m in present] + sorted(m for m in present if m not in PRETTY)
    # Direction comes from the whole file, not from the section's slice: a metric can be present
    # for one task and absent for another in the same section (a model that skipped a task, a
    # statistic with no staypoints), and looking it up locally then raises on an empty frame.
    direction = df.dropna(subset=["higher_is_better"]).groupby("metric").higher_is_better.first().to_dict()
    width = 2 + max(len(c) for _, c in SECTIONS)        # every row must span the whole tabular
    pad = lambda cells: " & ".join(cells + [""] * (width - len(cells))) + r" \\"
    out = []
    for title, all_cols in SECTIONS:
        # which rows exist for this section
        sub = df[df.apply(lambda r: any(r.task == t and r.metric == m for t, m, _, _ in all_cols), axis=1)]
        if sub.empty:
            continue
        # drop columns this run produced nothing for, rather than printing an empty column
        have = set(zip(sub.task, sub.metric))
        cols = [c for c in all_cols if (c[0], c[1]) in have]
        if not cols:
            continue
        protocols = [p for p in ("native", "linear_probe", "rollout", "embedding_knn", "reconstruction")
                     if p in set(sub.protocol.dropna())]
        rows, values = [], {}
        for model in models:
            for proto in protocols:
                key = (model, proto)
                vals = []
                for task, metric, _, fmt in cols:
                    r = sub[(sub.model == model) & (sub.protocol == proto) &
                            (sub.task == task) & (sub.metric == metric)]
                    vals.append(None if r.empty else r.iloc[0])
                if all(v is None for v in vals):
                    continue
                rows.append(key)
                values[key] = vals
        known = list(BASE_PRETTY)
        for bname in known + sorted(m for m in sub.model.unique()
                                    if str(m).startswith("baseline:") and m not in BASE_PRETTY):
            vals = []
            for task, metric, _, fmt in cols:
                r = sub[(sub.model == bname) & (sub.task == task) & (sub.metric == metric)]
                vals.append(None if r.empty else r.iloc[0])
            if any(v is not None for v in vals):
                rows.append((bname, None))
                values[(bname, None)] = vals
        if not rows:
            continue

        # best model and best overall, per column, respecting metric direction
        n_cols = len(cols)
        best_model, best_any = [None] * n_cols, [None] * n_cols
        for j, (task, metric, _, _) in enumerate(cols):
            hib = bool(direction.get(metric, False))
            pick = max if hib else min
            if metric in NO_BEST:
                continue
            cand_m = [(values[k][j]["value"], k) for k in rows
                      if k[1] is not None and values[k][j] is not None and np.isfinite(values[k][j]["value"])]
            cand_a = [(values[k][j]["value"], k) for k in rows
                      if values[k][j] is not None and np.isfinite(values[k][j]["value"])]
            if cand_m:
                best_model[j] = pick(cand_m)[1]
            if cand_a:
                best_any[j] = pick(cand_a)[1]

        arrows = []
        for task, metric, hdr, _ in cols:
            hib = bool(direction.get(metric, False))
            arrows.append(f"{hdr} $\\{'uparrow' if hib else 'downarrow'}$")
        # Sample size. Generation statistics are counted over different units (trajectories for
        # radius of gyration, staypoint pairs for jump length), so a single max would misstate
        # most of the row; show the range whenever it actually varies.
        ns = sub.loc[sub.n > 0, "n"].dropna()
        if ns.empty:
            n_note = ""
        elif ns.max() > 1.1 * ns.min():
            n_note = r" \textnormal{\small($n=%s$--$%s$)}" % (f"{int(ns.min()):,}", f"{int(ns.max()):,}")
        else:
            n_note = r" \textnormal{\small($n=%s$)}" % f"{int(ns.max()):,}"
        out.append(r"\multicolumn{%d}{@{}l}{\textit{%s}%s} \\[2pt]" % (width, title, n_note))
        out.append(pad(["Model", "Protocol"] + arrows))
        out.append(r"\midrule")
        last_base = False
        for k in rows:
            model, proto = k
            is_base = proto is None
            if is_base and not last_base:
                out.append(r"\addlinespace")
            last_base = is_base
            name = BASE_PRETTY.get(model, model) if is_base else PRETTY.get(model, model)
            label = ("\\textit{%s}" % esc(name)) if is_base else esc(name)
            pl = "--" if is_base else PROTO.get(proto, proto)
            cells = []
            for j, (task, metric, _, fmt) in enumerate(cols):
                c = cell(values[k][j], fmt, show_ci)
                if c != "--":
                    if best_any[j] == k:
                        c = r"\underline{" + c + "}"
                    if best_model[j] == k:
                        c = r"\textbf{" + c + "}"
                    if not is_base:
                        pm = values[k][j].get("paper_match", "")
                        c += PAPER_MARK.get(pm if isinstance(pm, str) else "", "")
                cells.append(c)
            out.append(pad([label, pl] + cells))
        out.append(r"\addlinespace[6pt]")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default="results.tex")
    ap.add_argument("--ci", action="store_true", help="include 95%% bootstrap intervals")
    ap.add_argument("--model-types", default="",
                    help="only for leaderboards without a paper_match column: name=type pairs, e.g. "
                         "'UniTraj-zeroshot=unitraj,TrajGPT=trajgpt,TransferTraj=transfertraj'")
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    if "paper_match" not in df.columns:
        df["paper_match"] = ""
        types = dict(kv.split("=", 1) for kv in a.model_types.split(",") if "=" in kv)
        if types:
            from mobeval.paper_metrics import lookup      # needs mobeval importable
            def level(r):
                hit = lookup(types.get(r.model, ""), r.task, r.metric, r.protocol)
                return hit[0] if hit else ""
            df["paper_match"] = df.apply(level, axis=1)
            print("paper marks looked up without the run's config: configuration-dependent exact "
                  "matches are shown as near")
        else:
            print("leaderboard has no paper_match column and --model-types was not given: no paper marks")
    df["paper_match"] = df["paper_match"].fillna("")
    # The protocol is already a column, so drop it from the task name: "next_location/linear_probe"
    # and "next_location" are the same task measured two ways, and the table pairs them by protocol.
    df["task"] = df["task"].str.replace("/linear_probe", "", regex=False)
    n_cols = max(len(c) for _, c in SECTIONS)
    cov = df[df.metric == "probe_coverage"].value
    body = build(df, a.ci)
    tex = r"""% Generated by make_table.py from mobeval's leaderboard.csv
\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{5pt}
\caption{Mobility foundation models on the Italian vehicle GPS panel, all evaluated on the
same splits, the same samples and the same baselines. \emph{Protocol} distinguishes a model's
own head (\emph{native}) from a linear probe on its frozen embedding (\emph{probe}), which is
the only way to place encoders without a prediction head on the same axis; probe numbers are
not native capabilities. \textbf{Bold} marks the best model, \underline{underline} the best
value including baselines. Note how often a baseline wins. CRPS is the distance between a
predicted \emph{distribution} and the observed value, equal to MAE for a point forecast;
JSD compares generated and real distributions of a mobility statistic on bins fixed from the
real data. Copy rate is not marked best/worst: it should match the real-data rate
($\approx$0.05), and 0.000 means the generator resembles the training data in nothing at
all. $^{\star}$: the metric the model's own paper reported, computed with the same formula and
protocol; $^{\dagger}$: the same quantity under a protocol that differs from the paper's (see
MODELS.md). The marks concern the metric, not the data: none of these numbers are on the papers'
datasets.COVERAGE}
\label{tab:mobeval-main}
\begin{tabular}{ll""" + "r" * n_cols + r"""}
\toprule
""" + body + r"""\bottomrule
\end{tabular}
\end{table}
"""
    note = ("" if cov.empty else
            r" The location probe predicts only the most-visited training cells, which reach "
            r"%.1f\%% of test targets; that share is its accuracy ceiling, not 100\%%."
            % (100 * cov.iloc[0]))
    tex = tex.replace("COVERAGE", note)
    with open(a.out, "w") as f:
        f.write(tex)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
