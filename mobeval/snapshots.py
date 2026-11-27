"""Which epoch is best? Evaluate the weights a training kept every N epochs (train.save_every).

    python -m mobeval snapshots --config exp.yaml --models TrajGPT OmniTraj \\
        [--every 100] [--max-snapshots 10] [--split val|test] [--tasks recovery next_location ...]
    (on the cluster: jobs/evaluate_snapshots.pbs)

Every snapshot <results>/snapshots/<model>/epoch_0100.pt is evaluated as a model of its own, named
<model>-ep0100, next to the checkpoint the training kept (named <model>: the weights of its lowest
validation loss). One model is loaded at a time, so the number of snapshots is not limited by GPU
memory. Besides the usual leaderboard.csv and report.md, the run folder holds

    snapshots.csv          one row per evaluated model: base model, epoch, file, whether these are the
                           weights the training kept, and the losses at that epoch
    training_history.csv   each base model's training log, epoch by epoch (for the loss curves)
    best_snapshots.csv     per model and task: the best epoch, its value, the kept checkpoint's value
    snapshot_ranking.csv   per evaluated model: its average position among its model's epochs

and report.md opens with the best epoch of each model. `python plot_snapshots.py <run folder>`
draws the figures.

The snapshots are scored on the VALIDATION split by default: picking the epoch with the best test
score and then reporting that score would overstate it. Evaluate the chosen snapshot on the test
split afterwards, as a model of its own (report.md prints the config entry).
"""
from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger("mobeval.snapshots")

SNAPSHOT_RE = re.compile(r"^epoch_(\d+)\.pt$")
# The metric each family is judged on (the same as plot_results.HEADLINE). Efficiency is left out:
# every snapshot of a model has the same size and speed.
PRIMARY = {"recovery": "ade_m", "location": "acc@1", "continuous": "crps_min", "anomaly": "roc_auc",
           "generation": "jsd", "classification": "macro_f1", "identity": "user_acc@1", "retrieval": "mrr"}
NOT_A_BASELINE = {"real_noise_floor"}           # the best any generator can reach, not a competitor
CHECKPOINT, SNAPSHOT = "checkpoint", "snapshot"     # kind of evaluated file; `kept` marks the kept weights
DEFAULT_MAX = 10


# ------------------------------------------------------------------ which files, which epochs
def snapshot_name(base: str, epoch: int) -> str:
    return f"{base}-ep{int(epoch):04d}"


def find_snapshots(model: str, dirs: Iterable) -> Dict[int, Path]:
    """epoch -> snapshot file of `model`, from <dir>/<model>/epoch_NNNN.pt in every dir.

    The same epoch can be in two places (the copy mirrored while the training ran and the one
    staged out at its end): the newer file wins, since a resumed training rewrites the epochs it
    goes through again."""
    found: Dict[int, Path] = {}
    for d in dirs:
        folder = Path(d) / model
        if not folder.is_dir():
            continue
        for f in folder.iterdir():
            m = SNAPSHOT_RE.match(f.name)
            if not m or not f.is_file():
                continue
            e = int(m.group(1))
            if e not in found or f.stat().st_mtime > found[e].stat().st_mtime:
                found[e] = f
    return dict(sorted(found.items()))


def choose(epochs: Iterable[int], every: Optional[int] = None, max_n: Optional[int] = None) -> List[int]:
    """The epochs to evaluate: those divisible by `every`, then at most `max_n` of them, evenly
    spread from the first to the last (both always kept). 0 or None: no limit."""
    eps = sorted(int(e) for e in epochs)
    if every:
        eps = [e for e in eps if e % int(every) == 0]
    if max_n and len(eps) > int(max_n):
        if int(max_n) == 1:
            return [eps[-1]]
        idx = np.unique(np.round(np.linspace(0, len(eps) - 1, int(max_n))).astype(int))
        eps = [eps[i] for i in idx]
    return eps


def read_history(checkpoint) -> List[dict]:
    """A checkpoint's training history: its <name>.losses.csv (every epoch) or its .json (rewritten
    only when validation improved, so possibly shorter) - whichever has more epochs."""
    p = Path(checkpoint)
    best: List[dict] = []
    log_csv = p.with_name(p.stem + ".losses.csv")
    if log_csv.exists():
        try:
            best = [{k: v for k, v in r.items() if not (isinstance(v, float) and math.isnan(v))}
                    for r in pd.read_csv(log_csv).to_dict("records")]
        except (OSError, ValueError):
            best = []
    side = p.with_suffix(".json")
    if side.exists():
        try:
            h = json.loads(side.read_text()).get("history") or []
        except (OSError, json.JSONDecodeError):
            h = []
        if len(h) > len(best):
            best = h
    return [e for e in best if e.get("epoch") is not None]


def kept_epoch(history: List[dict]) -> Optional[int]:
    """The epoch whose weights the training kept: the lowest validation loss (the first, on ties,
    as the training loop only takes strict improvements)."""
    h = [e for e in history if e.get("val_loss") is not None and np.isfinite(float(e["val_loss"]))]
    return int(min(h, key=lambda e: float(e["val_loss"]))["epoch"]) if h else None


def _at(history: List[dict], epoch: Optional[int]) -> dict:
    for e in history:
        if epoch is not None and int(e["epoch"]) == int(epoch):
            return e
    return {}


def snapshot_spec(base: dict, epoch: int, path) -> dict:
    """A config entry that loads one snapshot exactly as the base model is loaded."""
    s = {"name": snapshot_name(base["name"], epoch), "type": base["type"], "checkpoint": str(path),
         "adapter": dict(base.get("adapter") or {})}
    if "run_tag" in base:
        s["run_tag"] = base["run_tag"]
    return s


def plan(cfg: dict, names: Optional[Sequence[str]], dirs: Sequence, every: Optional[int] = None,
         max_n: Optional[int] = DEFAULT_MAX, trainable=None, checkpoint_for=None
         ) -> Tuple[List[dict], pd.DataFrame, Dict[str, List[dict]]]:
    """-> (model specs to evaluate, one row per spec, each base model's training history).

    Per base model: the chosen snapshots in epoch order, plus the checkpoint the training kept. The
    checkpoint holds the weights of one epoch, so when that epoch is among the chosen snapshots it
    is not evaluated twice: that snapshot is marked as the kept one. Models without snapshots are
    skipped, or an error when they were asked for."""
    from .registry import TRAIN_METHOD, checkpoint_path
    trainable = set(TRAIN_METHOD) if trainable is None else set(trainable)
    checkpoint_for = checkpoint_for or (lambda s: checkpoint_path(s, cfg["output_dir"]))
    models = cfg["models"]
    if names:
        unknown = set(names) - {m["name"] for m in models}
        if unknown:
            raise SystemExit(f"unknown model(s): {sorted(unknown)}")
        models = [m for m in models if m["name"] in names]
    taken = {m["name"] for m in cfg["models"]}
    specs, rows, histories, without = [], [], {}, []
    for base in models:
        if base["type"] not in trainable or "train" not in base:
            if names:
                log.warning(f"{base['name']}: not trained by mobeval, so it has no snapshots")
            continue
        found = find_snapshots(base["name"], dirs)
        chosen = choose(found, every, max_n)
        if not chosen:
            without.append(base["name"])
            continue
        ck = Path(checkpoint_for(base))
        kept_hist = read_history(ck) if ck.exists() else []
        newest = read_history(found[max(found)])
        histories[base["name"]] = max([kept_hist, newest], key=len)
        ke = kept_epoch(kept_hist)
        if not ck.exists():
            log.warning(f"{base['name']}: no checkpoint at {ck}; evaluating its snapshots only")
        elif ke is None or ke not in chosen:            # its epoch is not among the snapshots: score it too
            e = _at(kept_hist, ke)
            specs.append(base)
            rows.append(dict(model=base["name"], base=base["name"], kind=CHECKPOINT, kept=True, epoch=ke,
                             checkpoint=str(ck), train_loss=e.get("train_loss"),
                             train_eval_loss=e.get("train_eval_loss"), val_loss=e.get("val_loss")))
        for ep in chosen:
            s = snapshot_spec(base, ep, found[ep])
            if s["name"] in taken:
                raise SystemExit(f"{s['name']}: a model of that name is already in the config")
            e = _at(read_history(found[ep]), ep)
            specs.append(s)
            rows.append(dict(model=s["name"], base=base["name"], kind=SNAPSHOT,
                             kept=bool(ck.exists() and ke == ep), epoch=ep, checkpoint=s["checkpoint"],
                             train_loss=e.get("train_loss"), train_eval_loss=e.get("train_eval_loss"),
                             val_loss=e.get("val_loss")))
        log.info(f"{base['name']}: {len(chosen)} of {len(found)} snapshots (epochs {chosen[0]}..{chosen[-1]}); "
                 + ("no kept checkpoint" if not ck.exists() else
                    f"the training kept epoch {ke}" + (" (one of the snapshots)" if ke in chosen
                                                        else ", scored as well")))
    where = ", ".join(str(Path(d)) for d in dirs)
    if names and without:
        raise SystemExit(f"no snapshots for {without} in {where}. Train with `save_every: N` in the model's "
                         f"`train:` section, or point --snapshot-dir at the folder that holds <model>/epoch_*.pt"
                         + (f" (--every {every} may also have excluded them)" if every else ""))
    if not specs:
        raise SystemExit(f"no snapshots found in {where} (train with `save_every: N` in `train:`)")
    return specs, pd.DataFrame(rows), histories


def history_frame(histories: Dict[str, List[dict]]) -> pd.DataFrame:
    rows = [{"base": b, **e} for b, h in histories.items() for e in h]
    if not rows:
        return pd.DataFrame(columns=["base", "epoch"])
    t = pd.DataFrame(rows)
    first = [c for c in ("base", "epoch", "train_loss", "train_eval_loss", "val_loss", "lr") if c in t]
    return t[first + [c for c in t.columns if c not in first]]


def check_resume(run_dir, meta: dict, resuming: bool):
    """A resumed sweep must score on the same split as the run it continues."""
    f = Path(run_dir) / "snapshots_meta.json"
    if resuming and f.exists():
        old = json.loads(f.read_text())
        if old.get("split") != meta["split"]:
            raise SystemExit(f"--resume: that run scored the snapshots on the {old.get('split')} split, not on "
                             f"{meta['split']}. Resume with --split {old.get('split')}, or start a new run.")
    elif resuming and (Path(run_dir) / "results.jsonl").exists():
        raise SystemExit(f"--resume: {run_dir} is not a snapshot run; start a new one (without --resume)")
    f.write_text(json.dumps(meta, indent=2))


def release(adapter, contexts):
    """Forget what the contexts cached for this adapter (its embeddings), and free its GPU memory,
    so a sweep over many snapshots uses as much memory as one model."""
    key = f"{adapter.name}@{getattr(adapter, 'run_tag', None)}"
    for c in contexts:
        if c is None:
            continue
        for k in [k for k in list(c.cache) if isinstance(k, tuple) and key in k]:
            del c.cache[k]
    try:
        import torch
        net = getattr(adapter, "net", None)
        if net is not None and hasattr(net, "to"):
            net.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:                                   # noqa: BLE001 - only a memory courtesy
        pass


# ------------------------------------------------------------------ which epoch is best
def improvement(v, b, higher_is_better: bool) -> float:
    """Relative gain of v over b in the good direction (as plot_results): +0.1 = 10% better."""
    if b is None or v is None or not np.isfinite(b) or not np.isfinite(v) or b <= 0:
        return float("nan")
    return (v - b) / b if higher_is_better else (b - v) / b


def primary_rows(lb: pd.DataFrame) -> pd.DataFrame:
    """The leaderboard rows of each family's primary metric (all its qualifiers, e.g. every
    generation statistic), with the columns the summary needs."""
    d = lb.copy()
    d["metric_base"] = d.metric.astype(str).str.split(":").str[0]
    d = d[d.family.map(PRIMARY).eq(d.metric_base)]
    d = d[np.isfinite(pd.to_numeric(d.value, errors="coerce"))].copy()
    d["value"] = d.value.astype(float)
    for c in ("ci_low", "ci_high"):
        d[c] = pd.to_numeric(d[c], errors="coerce") if c in d else np.nan
    d["is_baseline"] = d.model.astype(str).str.startswith("baseline:")
    d["stem"] = d.task.astype(str).str.replace("/linear_probe", "", regex=False)
    d["higher_is_better"] = d.higher_is_better.astype(str).str.lower().eq("true")
    return d


def summarize(lb: pd.DataFrame, snaps: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """-> (one row per base model and task: its best epoch, the values of the best, kept and last
    weights, the best baseline; one row per evaluated model: its average position among its base
    model's evaluated weights, 0 = best on every task, 1 = worst).

    The average is taken per task family first, then over the families, so each family counts
    once: six anomaly rows do not outvote one retrieval row."""
    d = primary_rows(lb)
    info = snaps.set_index("model")
    bl = d[d.is_baseline & ~d.model.str.replace("baseline:", "", regex=False).isin(NOT_A_BASELINE)]
    m = d[d.model.isin(info.index)].copy()
    if m.empty:
        return pd.DataFrame(), pd.DataFrame()
    m["base"] = m.model.map(info.base)
    m["epoch"] = m.model.map(info.epoch)
    m["kind"] = m.model.map(info.kind)
    m["kept"] = m.model.map(info.kept).fillna(False).astype(bool)
    out, pos = [], []
    for (base, task, protocol, metric), g in m.groupby(["base", "task", "protocol", "metric"], sort=False):
        hib = bool(g.higher_is_better.iloc[0])
        g = g.sort_values(["epoch", "kind"])
        best = g.loc[g.value.idxmax() if hib else g.value.idxmin()]
        # not told apart from the best at this sample size: inside its 95% interval
        has_ci = bool(np.isfinite(best.ci_low) and np.isfinite(best.ci_high))
        tied_models = {mod for v, mod in zip(g.value, g.model)
                       if mod != best.model and has_ci and best.ci_low <= v <= best.ci_high}
        tied = sorted({int(e) for e, mod in zip(g.epoch, g.model) if mod in tied_models and pd.notna(e)})
        kept = g[g.kept]
        snap = g[g.kind.eq(SNAPSHOT)]
        bb = bl[bl.stem.eq(g.stem.iloc[0]) & bl.metric.eq(metric)]
        bbest = bb.loc[bb.value.idxmax() if hib else bb.value.idxmin()] if len(bb) else None
        kv = float(kept.value.iloc[0]) if len(kept) else float("nan")
        out.append(dict(
            base=base, family=g.family.iloc[0], task=task, protocol=protocol, metric=metric,
            unit=g.unit.iloc[0] if "unit" in g else "", higher_is_better=hib, n_evaluated=len(g),
            best_model=best.model, best_is_kept=bool(best.kept),
            best_epoch=int(best.epoch) if pd.notna(best.epoch) else None,
            best_value=float(best.value), best_ci_low=best.ci_low, best_ci_high=best.ci_high,
            tied_epochs=" ".join(map(str, tied)),
            kept_epoch=int(kept.epoch.iloc[0]) if len(kept) and pd.notna(kept.epoch.iloc[0]) else None,
            kept_value=kv,
            last_epoch=int(snap.epoch.iloc[-1]) if len(snap) else None,
            last_value=float(snap.value.iloc[-1]) if len(snap) else float("nan"),
            baseline=bbest.model.replace("baseline:", "") if bbest is not None else "",
            baseline_value=float(bbest.value) if bbest is not None else float("nan"),
            gain_vs_kept=improvement(float(best.value), kv, hib),
            gain_vs_baseline=improvement(float(best.value), float(bbest.value), hib) if bbest is not None
            else float("nan")))
        rank = g.value.rank(ascending=not hib, method="average")
        n = len(g)
        for mod, rk in zip(g.model, rank):
            pos.append(dict(base=base, model=mod, family=g.family.iloc[0],
                            position=(rk - 1) / (n - 1) if n > 1 else 0.0,
                            best=bool(mod == best.model), tied=bool(mod == best.model or mod in tied_models)))
    summary = pd.DataFrame(out)
    p = pd.DataFrame(pos)
    fam = p.groupby(["base", "model", "family"], sort=False).position.mean().reset_index()
    overall = (p.groupby(["base", "model"], sort=False)
               .agg(tasks=("position", "size"), best_on=("best", "sum"), best_or_tied_on=("tied", "sum"))
               .reset_index())
    overall["position"] = overall.set_index(["base", "model"]).index.map(
        fam.groupby(["base", "model"]).position.mean())
    overall["families"] = overall.set_index(["base", "model"]).index.map(fam.groupby(["base", "model"]).size())
    overall["epoch"] = overall.model.map(info.epoch)
    overall["kind"] = overall.model.map(info.kind)
    overall["kept"] = overall.model.map(info.kept).fillna(False).astype(bool)
    overall["val_loss"] = pd.to_numeric(overall.model.map(info.val_loss), errors="coerce")
    # the same weights ordered by validation loss alone, on the same 0..1 scale
    overall["val_loss_position"] = np.nan
    for base, g in overall.groupby("base"):
        ok = g.val_loss.notna()
        if ok.sum() > 1:
            r = g.loc[ok, "val_loss"].rank(method="average")
            overall.loc[r.index, "val_loss_position"] = (r - 1) / (ok.sum() - 1)
    overall = overall.sort_values(["base", "epoch", "kind"], kind="stable").reset_index(drop=True)
    return summary, overall


def best_overall(overall: pd.DataFrame, base: str) -> Optional[pd.Series]:
    g = overall[overall.base.eq(base)]
    if g.empty:
        return None
    # lowest average position; on a tie, the earlier epoch (less training for the same result)
    return g.sort_values(["position", "epoch"], kind="stable").iloc[0]


# ------------------------------------------------------------------ report section
def _fmt(v, unit: str = "") -> str:
    if v is None or not np.isfinite(v):
        return "-"
    if unit == "m":
        return f"{v / 1000:,.2f} km" if abs(v) >= 10_000 else f"{v:,.0f} m"
    if unit == "min":
        return f"{v:,.1f} min"
    if unit == "fraction":
        return f"{100 * v:.1f}%"
    return f"{v:.3f}" if abs(v) < 10 else f"{v:,.1f}"


def pretty_task(task: str, protocol: str = "native") -> str:
    s = task.replace("/linear_probe", "")
    m = re.fullmatch(r"recovery/(random|block)@([\d.]+)", s)
    if m:
        s = f"recovery, {m.group(1)} mask {float(m.group(2)):.0%}"
    elif s.startswith("recovery/last:"):
        s = f"recovery, last {s.split(':')[1]} points"
    elif s.startswith("recovery/keep_every:"):
        s = f"recovery, keep every {s.split(':')[1]}th point"
    elif s.startswith("continuous/"):
        target, *rest = s[len("continuous/"):].split("|")
        given = [r[len("given:"):].replace("+", " + ") for r in rest if r.startswith("given:")]
        s = target.replace("_", " ") + (f", given {given[0]}" if given else "")
    else:
        s = s.replace("_", " ").replace("/", ": ", 1)
    return s if protocol in ("native", "", None) else f"{s} ({protocol.replace('_', ' ')})"


def markdown(summary: pd.DataFrame, overall: pd.DataFrame, snaps: pd.DataFrame, split: str,
             bases: Optional[Dict[str, dict]] = None) -> str:
    split_name = "validation" if split == "val" else "test"
    lines = ["## Which epoch is best", "",
             f"Each snapshot (`<model>-epNNNN`, the weights at that epoch) and the weights the training kept "
             f"(its lowest validation loss; `<model>` when that epoch is not a snapshot) scored on the "
             f"**{split_name} split**, on each family's "
             f"primary metric. *Within its interval*: other epochs whose value lies inside the best one's 95% "
             f"interval, i.e. not told apart from it at this sample size. *Overall*: the lowest average "
             f"position among these weights (0 = best on every task), averaged within each task family "
             f"first so that every family counts once.", ""]
    if split != "val":
        lines += ["> **Chosen on the test split.** The best epoch's test scores are optimistic: the epoch was "
                  "picked because it did well on these very samples. For the numbers you report, choose on the "
                  "validation split (`--split val`, the default) and evaluate that epoch on test.", ""]
    if summary.empty:
        return "\n".join(lines + ["(no results)"])
    for base, g in summary.groupby("base", sort=False):
        kept = snaps[snaps.base.eq(base) & snaps.kept.astype(bool)]
        ke = kept.epoch.iloc[0] if len(kept) else None
        b = best_overall(overall, base)
        lines.append(f"### {base}")
        lines.append("")
        if b is not None:
            what = f"`{b.model}`" + (", the weights the training kept" if b.kept else "")
            lines.append(f"Best overall: **epoch {int(b.epoch) if pd.notna(b.epoch) else '?'}** ({what}): average "
                         f"position {b.position:.2f} among {len(overall[overall.base.eq(base)])} evaluated weights, "
                         f"best or within the best's interval on {int(b.best_or_tied_on)}/{int(b.tasks)} tasks."
                         + (f" The training kept epoch {int(ke)}." if ke is not None and pd.notna(ke) else ""))
            lines.append("")
        kcol = f"kept (ep {int(ke)})" if ke is not None and pd.notna(ke) else "kept"
        lines.append(f"| task | metric | best epoch | value | within its interval | {kcol} | last snapshot | best baseline |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for _, r in g.iterrows():
            arrow = "↑" if r.higher_is_better else "↓"
            best_ep = f"{r.best_epoch} (kept)" if r.best_is_kept else str(r.best_epoch)
            last = f"{_fmt(r.last_value, r.unit)} (ep {r.last_epoch})" if r.last_epoch is not None else "-"
            bl = f"{_fmt(r.baseline_value, r.unit)} ({r.baseline})" if r.baseline else "-"
            lines.append(f"| {pretty_task(r.task, r.protocol)} | {r.metric} {arrow} | {best_ep} | "
                         f"{_fmt(r.best_value, r.unit)} | {r.tied_epochs or '-'} | {_fmt(r.kept_value, r.unit)} | "
                         f"{last} | {bl} |")
        lines.append("")
        if b is not None and b.kind == SNAPSHOT and not b.kept:
            row = snaps[snaps.model.eq(b.model)].iloc[0]
            spec = (bases or {}).get(base, {})
            adapter = json.dumps(spec.get("adapter") or {}).replace('"', "")
            lines += ["To evaluate it on the test split, add it to the config as a model of its own:", "",
                      "```yaml",
                      f"  - name: {b.model}",
                      f"    type: {spec.get('type', '<the type of ' + base + '>')}",
                      f"    checkpoint: {row.get('checkpoint', '<snapshots/' + base + '/epoch_NNNN.pt>')}",
                      f"    adapter: {adapter}",
                      "```", ""]
    return "\n".join(lines)
