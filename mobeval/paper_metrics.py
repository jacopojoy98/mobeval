"""Which of mobeval's (task, metric) pairs are the ones each model's authors reported.

Two levels, so a highlighted number is never over-read:

  exact  same metric, same formula, same evaluation protocol as the paper. Given the paper's data
         and preprocessing (`recipe: paper` and the matching dataset loader) the number is
         directly comparable with the paper's table.
  near   the same quantity, but mobeval's protocol differs in a way that moves the number even on
         the paper's data. `note` says how.

Some pairs are exact only under a configuration (UniTraj's recovery error needs 200-point windows at
3 s with maskable endpoints); otherwise they fall back to near, with the reason. Most pairs are near
by construction: where the original evaluation leaks targets or reads test data, an honest
evaluation cannot reproduce it, and saying "exact" would invite a comparison that does not hold.

CLIP-Mobility has no publication, so it has no entries.
OmniTraj's own evaluation is retrieval (Tables 2-3); its downstream uses are not evaluated with metrics.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from typing import Callable, Optional, Sequence, Tuple


@dataclass(frozen=True)
class PaperMetric:
    model_type: str                  # adapter.model_type
    task: str                        # regex, full match, on the task name without '/linear_probe'
    metrics: Tuple[str, ...]
    level: str                       # 'exact' | 'near'
    paper_name: str                  # what the paper calls it
    source: str                      # where in the paper
    note: str = ""                   # why only near (or what exact still leaves out)
    protocols: Tuple[str, ...] = ("native",)
    exact_if: Optional[Callable] = None      # ctx -> bool; upgrades 'near' to 'exact'
    exact_if_note: str = ""


def _sampled_at(ctx, seconds: float, tol: float = 0.5) -> bool:
    w = getattr(ctx, "windows", {}).get("test") if ctx is not None else None
    if w is None or not len(w):
        return False
    return abs(float(np.median(np.diff(w.t, axis=1))) - seconds) <= tol


def _unitraj_exact(ctx) -> bool:
    cfg = ctx.cfg
    return cfg.window_length == 200 and not cfg.recovery_keep_endpoints and _sampled_at(ctx, 3.0)


PAPER_METRICS: Sequence[PaperMetric] = (
    # ---------------------------------------------------------------- UniTraj
    PaperMetric("unitraj", r"recovery/random@0\.5", ("ade_m", "rmse_m"), "near",
                "Trajectory recovery MAE / RMSE (m), 50% random masking",
                "UniTraj paper, recovery tables (WorldTrace, zero-shot and fine-tuned)",
                "the paper masks 50% of 200-point trajectories resampled at 3 s and may mask the "
                "endpoints; mobeval's default windows are shorter and keep the endpoints. The UniTraj "
                "repository has no evaluation code: 'same formula' is the paper's definition (mean / "
                "root-mean-square distance over the masked points)",
                exact_if=_unitraj_exact,
                exact_if_note="window_length: 200, recovery_keep_endpoints: false, test data at 3 s"),
    PaperMetric("unitraj", r"recovery/last:5", ("ade_m", "rmse_m"), "near",
                "Trajectory prediction MAE / RMSE (m), last 5 points",
                "UniTraj paper, prediction tables",
                "the paper predicts the last 5 points of 200-point trajectories at 3 s",
                exact_if=lambda ctx: ctx.cfg.window_length == 200 and _sampled_at(ctx, 3.0),
                exact_if_note="window_length: 200, test data at 3 s"),
    PaperMetric("unitraj", r"mode_classification", ("accuracy",), "near",
                "Classification accuracy", "UniTraj paper, classification table",
                "the paper fine-tunes the whole encoder; mobeval trains a head on the frozen one",
                protocols=("native", "linear_probe")),
    # ---------------------------------------------------------------- TransferTraj
    PaperMetric("transfertraj", r"recovery/keep_every:8", ("ade_m", "rmse_m"), "near",
                "Trajectory recovery (TRec) MAE / RMSE (m), every 8th point kept",
                "TransferTraj paper, recovery results (mu = 8 epsilon)",
                "the paper recovers whole trips of 5-120 points (mobeval: fixed-length windows) with a "
                "model fine-tuned on this task (train one with recipe: paper, objective: trec)"),
    PaperMetric("transfertraj", r"recovery/last:5", ("ade_m", "rmse_m"), "near",
                "Trajectory prediction MAE / RMSE (m), last 5 points",
                "TransferTraj paper, prediction results",
                "the paper predicts the end of whole trips of 5-120 points (mobeval: fixed-length "
                "windows) with a model fine-tuned on this task (recipe: paper, objective: tp)"),
    PaperMetric("transfertraj", r"continuous/travel_time\|given:location(\|.*)?", ("mae_min", "rmse_min", "mape"),
                "near", "OD travel-time estimation MAE / RMSE (min), MAPE",
                "TransferTraj paper, travel-time results",
                "the paper fine-tunes the model on origin, destination and departure time of a trip; "
                "mobeval probes the frozen embedding of the preceding visits plus the destination",
                protocols=("linear_probe",)),
    # ---------------------------------------------------------------- OmniTraj
    PaperMetric("omnitraj", r"retrieval/cross_modal:(topology|region|road|region\+topology|road\+topology|region\+road\+topology)",
                ("mean_rank", "mrr", "hr@1", "hr@10"), "near",
                "Trajectory retrieval MR / MRR / HR@1 / HR@10 ('OmniTraj' = topology queries; "
                "'OmniTraj (reg)', '(road)', ... the other rows)", "OmniTraj paper, Table 2",
                "same metrics and query/database roles, but the paper retrieves among the 20,000 whole test "
                "trips of one city (1.1M training trips); mobeval among retrieval_db_size test windows",
                protocols=("native",)),
    PaperMetric("omnitraj", r"retrieval/condition:(road|region)", ("cr@1", "cr@5"), "near",
                "Condition-based retrieval CR@1 / CR@5 (road, region)", "OmniTraj paper, Table 3 and Eq. 14",
                "the paper's database is the 20,000 test trips; mobeval's retrieval_db_size test windows",
                protocols=("native",)),
    # ---------------------------------------------------------------- TrajGPT
    PaperMetric("trajgpt", r"next_location", ("acc@1", "acc@5", "acc@10", "acc@20"), "near",
                "Next-visit region Acc@k", "TrajGPT paper, next-visit prediction table",
                "the paper ranks H3 resolution-7 regions (mobeval's shared grid unless grid_backend: h3, "
                "grid_h3_resolution: 7); it also scores users with fewer than 128 visits through left "
                "padding, which mobeval's fixed-length visit contexts cannot, and its instances overlap "
                "across the chronological split"),
    PaperMetric("trajgpt", r"continuous/travel_time\|given:location(\|.*)?",
                ("p_within_5min", "p_within_10min", "p_within_20min"), "near",
                "Arrival time P(+-5/10/20 min)", "TrajGPT paper, next-visit prediction table",
                "same formula (truncated, renormalised mass within t), but the released code's travel "
                "head reads the target's own arrival time, so the paper's numbers include that leak. "
                "mobeval conditions on the next visit's location only, as the paper's factorisation "
                "intends, and truncates at the train 99th percentile of valid gaps (the original: all "
                "splits, all gaps)"),
    PaperMetric("trajgpt", r"continuous/duration\|given:location\+arrival",
                ("p_within_5min", "p_within_10min", "p_within_20min"), "near",
                "Departure time P(+-5/10/20 min)", "TrajGPT paper, next-visit prediction table",
                "same formula, but the released code's duration head reads the target's own departure "
                "time, so the paper's numbers include that leak; mobeval conditions on the next visit's "
                "location and arrival (the paper's factorisation)"),
)


def _base_task(task: str) -> str:
    return task.replace("/linear_probe", "")


def lookup(model_type: str, task: str, metric: str, protocol: str, ctx=None) -> Optional[Tuple[str, PaperMetric]]:
    """-> (level, entry) when this result is one of the paper's metrics for that model. `ctx` (the
    EvalContext) lets configuration-dependent entries check whether they are exact; without it they
    are reported as near."""
    t = _base_task(task)
    for e in PAPER_METRICS:
        if e.model_type != model_type or metric not in e.metrics or protocol not in e.protocols:
            continue
        if not re.fullmatch(e.task, t):
            continue
        level = e.level
        if level == "near" and e.exact_if is not None and ctx is not None and hasattr(ctx, "cfg"):
            try:
                if e.exact_if(ctx):
                    level = "exact"
            except Exception:                                   # noqa: BLE001 - a check never fails a run
                pass
        return level, e
    return None


def level_for(adapter, task: str, metric: str, protocol: str, ctx=None) -> str:
    hit = lookup(getattr(adapter, "model_type", ""), task, metric, protocol, ctx)
    return hit[0] if hit else ""


def describe(model_types: Sequence[str] = ("unitraj", "trajgpt", "transfertraj", "omnitraj")) -> str:
    """Markdown list of every paper metric and what separates it from the paper's protocol."""
    lines = []
    for mt in model_types:
        entries = [e for e in PAPER_METRICS if e.model_type == mt]
        if not entries:
            continue
        lines.append(f"**{mt}**")
        for e in entries:
            shown = e.task.replace(r"(\|.*)?", "").replace("\\", "")
            s = f"- `{shown}` · {', '.join(e.metrics)} — {e.paper_name} ({e.level}"
            s += f"; exact with {e.exact_if_note}" if e.exact_if is not None else ""
            s += f"). {e.note}." if e.note else ")."
            lines.append(s)
        lines.append("")
    return "\n".join(lines)
