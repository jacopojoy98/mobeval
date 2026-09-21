"""Evaluation tasks. Each task: builds canonical inputs from the context,
calls ONE adapter capability, scores model AND baselines on identical samples,
and emits ResultRecords with bootstrap CIs and paired skill-score CIs."""
from __future__ import annotations

import logging
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from scipy import sparse

from . import baselines as B
from .adapters.base import (CONTINUOUS, EMBEDDING, GENERATION, MODE_CLASSIFICATION, NEXT_LOCATION, RECOVERY,
                            ContinuousPrediction, MobilityModelAdapter, TargetGuard)
from .context import EvalContext
from .data import MobilityDataset, TrajectoryBatch, detect_staypoints, make_mask
from .geo import haversine_m
from .metrics import generative as G
from .metrics.classification import classification_metrics, normalise_probs, ranking_metrics
from .metrics.probabilistic import continuous_metrics, pit_ks
from .metrics.reconstruction import recovery_metrics
from .results import ResultRecord
from .stats import bootstrap, evaluate_with_ci, skill_score

log = logging.getLogger("mobeval")

Scores = Tuple[Dict[str, np.ndarray], Dict[str, Callable]]   # (per-sample, set-level)


def _key(a: MobilityModelAdapter) -> str:
    return f"{a.name}@{a.run_tag}"


class Task:
    name = "task"
    capability = ""

    def applicable(self, adapter: MobilityModelAdapter) -> bool:
        return self.capability in adapter.capabilities

    def run(self, adapter: MobilityModelAdapter, ctx: EvalContext) -> List[ResultRecord]:
        raise NotImplementedError

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def emit(ctx, adapter, task, n, model: Scores, baselines: Dict[str, Scores], preferred: Sequence[str],
             seed: int, protocol: str = "native") -> List[ResultRecord]:
        recs, cfg = [], ctx.cfg
        per, sets = model
        for metric in list(per) + list(sets):
            if metric.startswith("_"):
                continue
            bname = next((b for b in preferred if metric in baselines.get(b, ({}, {}))[0]
                          or metric in baselines.get(b, ({}, {}))[1]), None)
            kw = {}
            if bname:
                bper, bset = baselines[bname]
                kw = {"baseline_vals": bper.get(metric), "baseline_set_fn": bset.get(metric)}
            if metric in sets:
                res = evaluate_with_ci(metric, n, set_fn=sets[metric], n_boot=cfg.n_boot, seed=seed, **kw)
            else:
                res = evaluate_with_ci(metric, per[metric], n_boot=cfg.n_boot, seed=seed, **kw)
            recs.append(ResultRecord(model=adapter.name, run_tag=adapter.run_tag, task=task, metric=metric, n=n,
                                     protocol=protocol, baseline=bname, eval_seed=seed, dataset=ctx.dataset_name,
                                     **res))
        # baselines themselves (once per task/seed/protocol)
        for bname, (bper, bset) in baselines.items():
            key = (task, seed, protocol, bname)
            if key in ctx.emitted_baselines:
                continue
            ctx.emitted_baselines.add(key)
            for metric in list(bper) + list(bset):
                if metric.startswith("_"):
                    continue
                if metric in bset:
                    res = evaluate_with_ci(metric, n, set_fn=bset[metric], n_boot=cfg.n_boot, seed=seed)
                else:
                    res = evaluate_with_ci(metric, bper[metric], n_boot=cfg.n_boot, seed=seed)
                recs.append(ResultRecord(model=f"baseline:{bname}", run_tag="-", task=task, metric=metric, n=n,
                                         protocol=protocol, eval_seed=seed, dataset=ctx.dataset_name, **res))
        return recs


# =========================================================================== #
class RecoveryTask(Task):
    name, capability = "recovery", RECOVERY

    def run(self, adapter, ctx):
        batch, cfg, recs = ctx.windows["test"], ctx.cfg, []
        for kind in cfg.recovery_kinds:
            for ratio in cfg.recovery_ratios:
                for seed in cfg.eval_seeds:
                    mask = make_mask(len(batch), batch.length, ratio, kind, seed)
                    hidden = TargetGuard.hide_masked(batch, mask)
                    with ctx.timed(_key(adapter), self.capability, len(batch)):
                        plat, plon = adapter.reconstruct(hidden, mask)
                    score = lambda la, lo: (recovery_metrics(la, lo, batch.lat, batch.lon, mask, ctx.grid,
                                                             cfg.recovery_dtw), {})
                    ck = ("recovery_baselines", kind, ratio, seed)
                    if ck not in ctx.cache:
                        ctx.cache[ck] = {"linear_interp": score(*B.linear_interpolation(hidden, mask)),
                                         "last_observed": score(*B.last_observed(hidden, mask))}
                    recs += self.emit(ctx, adapter, f"recovery/{kind}@{ratio:g}", len(batch), score(plat, plon),
                                      ctx.cache[ck], ["linear_interp"], seed)
        return recs


# =========================================================================== #
class NextLocationTask(Task):
    name, capability = "next_location", NEXT_LOCATION

    @staticmethod
    def _grid_probs(pred, grid, n):
        if pred.grid_scores is not None:
            p = normalise_probs(pred.grid_scores)
            return p, np.column_stack(grid.centroid(p.argmax(1))), True
        if pred.token_scores is not None:
            pt = normalise_probs(pred.token_scores)
            tl = np.asarray(pred.token_latlon, float)
            cells = grid.cell_of(tl[:, 0], tl[:, 1])
            M = sparse.csr_matrix((np.ones(len(cells)), (np.arange(len(cells)), cells)),
                                  shape=(len(cells), grid.n_cells))
            p = np.asarray((M.T @ pt.T).T)                        # (N, C): token mass summed per cell
            return p, tl[pt.argmax(1)], True
        ll = np.asarray(pred.latlon, float)
        p = np.zeros((n, grid.n_cells))
        p[np.arange(n), grid.cell_of(ll[:, 0], ll[:, 1])] = 1.0
        return p, ll, False

    @staticmethod
    def _score(p, top1_latlon, v, ranked: bool) -> Scores:
        rm = ranking_metrics(p, v.tgt_cell)
        if not ranked:                                  # point predictor: top-k / NLL undefined
            rm = {"acc@1": rm["acc@1"]}
        d = haversine_m(top1_latlon[:, 0], top1_latlon[:, 1], v.tgt_lat, v.tgt_lon)
        rm.update({"dist_err_m": d, "median_dist_err_m": d, "acc_1km": (d <= 1000).astype(float)})
        return rm, {}

    def run(self, adapter, ctx):
        v, grid, recs = ctx.visits["test"], ctx.grid, []
        seed = ctx.cfg.eval_seeds[0]                   # deterministic task: one pass
        adapter.prepare(self.name, ctx.visits.get("train"), ctx.visits.get("val"))
        with ctx.timed(_key(adapter), self.capability, len(v)):
            pred = adapter.predict_location(TargetGuard.hide_visits(v), grid)
        pred.check()
        p, top1, ranked = self._grid_probs(pred, grid, len(v))
        if "loc_baselines" not in ctx.cache:
            lb = B.LocationBaselines(ctx.visits["train"], grid.n_cells)
            ctx.cache["loc_baselines"] = {}
            for bn, fn in [("markov1", lb.markov1), ("user_frequent", lb.user_frequent),
                           ("global_popular", lb.global_popular)]:
                bp = fn(v)
                ctx.cache["loc_baselines"][bn] = self._score(bp, np.column_stack(grid.centroid(bp.argmax(1))), v, True)
        recs += self.emit(ctx, adapter, "next_location", len(v), self._score(p, top1, v, ranked),
                          ctx.cache["loc_baselines"], ["markov1", "user_frequent"], seed)
        return recs


# =========================================================================== #
class ContinuousValueTask(Task):
    capability = CONTINUOUS

    def __init__(self, target: str):
        assert target in ("travel_time", "duration")
        self.target, self.name = target, f"continuous/{target}"

    def _valid(self, y_min, cfg):
        ok = np.isfinite(y_min) & (y_min > 0)
        if self.target == "travel_time" and cfg.travel_time_max_h is not None:
            ok &= y_min <= cfg.travel_time_max_h * 60
        return ok

    def _y(self, v):
        return (v.tgt_travel_time_s if self.target == "travel_time" else v.tgt_duration_s) / 60.0

    @staticmethod
    def _to_minutes(pred: ContinuousPrediction):
        return (None if pred.point is None else np.asarray(pred.point, float) / 60.0,
                None if pred.mixture is None else pred.mixture.rescale(1 / 60.0),
                None if pred.samples is None else np.asarray(pred.samples, float) / 60.0)

    @staticmethod
    def _with_pit(scores: Dict[str, np.ndarray]) -> Scores:
        sets = {}
        if "_pit" in scores:
            pit = scores["_pit"]
            sets["pit_ks"] = lambda i: pit_ks(pit[i])
        return scores, sets

    def run(self, adapter, ctx):
        cfg, v = ctx.cfg, ctx.visits["test"]
        reveal = tuple(cfg.continuous_reveal.get(self.target, ()))
        keep = self._valid(self._y(v), cfg)
        seed = cfg.eval_seeds[0]
        adapter.prepare(self.name, ctx.visits.get("train"), ctx.visits.get("val"))
        with ctx.timed(_key(adapter), f"{self.capability}/{self.target}", len(v)):
            pred = adapter.predict_continuous(TargetGuard.hide_visits(v, reveal), self.target)
        point, mix, samples = self._to_minutes(pred)
        y = self._y(v)
        sigma = None
        if mix is None and samples is None:            # point model: sigma from VALIDATION residuals
            vv = ctx.visits["val"]
            vp = adapter.predict_continuous(TargetGuard.hide_visits(vv, reveal), self.target).point
            okv = self._valid(self._y(vv), cfg)
            sigma = float(np.std((self._y(vv) - np.asarray(vp) / 60.0)[okv])) or 1.0
        sel = lambda a: None if a is None else a[keep]
        if mix is not None:
            from .metrics.probabilistic import Mixture
            mix = Mixture(mix.weights[keep], mix.means[keep], mix.stds[keep], mix.space)
        model = self._with_pit(continuous_metrics(y[keep], sel(point), mix, sel(samples), sigma))
        ck = ("cont_baselines", self.target, reveal)
        if ck not in ctx.cache:
            ytr = self._y(ctx.visits["train"])
            cb = B.ContinuousBaselines(ytr[self._valid(ytr, cfg)], seed=seed)
            n = int(keep.sum())
            prob = continuous_metrics(y[keep], None, cb.mixture(n))
            pt = continuous_metrics(y[keep], cb.point(n))
            prob["mae_min"], prob["rmse_min"] = pt["mae_min"], pt["rmse_min"]   # median is the MAE-optimal point
            ctx.cache[ck] = {"train_marginal": self._with_pit(prob)}
        task = self.name + (f"|given:{'+'.join(reveal)}" if reveal else "")
        if self.target == "travel_time" and cfg.travel_time_max_h is not None:
            task += f"|<={cfg.travel_time_max_h:g}h"
        return self.emit(ctx, adapter, task, int(keep.sum()), model, ctx.cache[ck], ["train_marginal"], seed)


# =========================================================================== #
class ModeClassificationTask(Task):
    name, capability = "mode_classification", MODE_CLASSIFICATION

    def applicable(self, adapter):
        return bool({MODE_CLASSIFICATION, EMBEDDING} & adapter.capabilities)

    @staticmethod
    def _labelled(b: TrajectoryBatch, classes=None):
        ok = np.array([m is not None for m in b.mode]) if b.mode is not None else np.zeros(len(b), bool)
        b = b.take(np.where(ok)[0])
        classes = classes or sorted(set(b.mode))
        keep = np.isin(b.mode, classes)
        b = b.take(np.where(keep)[0])
        y = np.array([classes.index(m) for m in b.mode])
        return b, y, classes

    def run(self, adapter, ctx):
        cfg, recs = ctx.cfg, []
        if ctx.windows["train"].mode is None:
            ctx.skipped.append((_key(adapter), self.name, "dataset has no mode labels"))
            return recs
        train, ytr, classes = self._labelled(ctx.windows["train"])
        test, yte, _ = self._labelled(ctx.windows["test"], classes)
        val, _, _ = self._labelled(ctx.windows["val"], classes) if "val" in ctx.windows else (None, None, None)
        K = len(classes)
        hide = lambda b: TrajectoryBatch(b.lat, b.lon, b.t, b.user_id, b.traj_id, None)
        for frac in cfg.label_fractions:
            for seed in (cfg.eval_seeds if frac < 1 else cfg.eval_seeds[:1]):
                idx = self._stratified(ytr, frac, seed)
                tr, ysub = train.take(idx), ytr[idx]
                ck = ("mode_baselines", frac, seed)
                if ck not in ctx.cache:
                    ctx.cache[ck] = {
                        "handcrafted_gbdt": classification_metrics(B.handcrafted_classifier(tr, ysub, test, seed), yte),
                        "majority": classification_metrics(B.majority_class_probs(ysub, len(test), K), yte)}
                for protocol in cfg.mode_protocols:
                    if protocol == "native" and MODE_CLASSIFICATION in adapter.capabilities:
                        adapter.prepare(self.name, tr, val, protocol="native", label_fraction=frac,
                                        labels=ysub, classes=classes, seed=seed)
                        with ctx.timed(_key(adapter), self.capability, len(test)):
                            probs = adapter.classify_mode(hide(test), classes)
                    elif protocol == "linear_probe" and EMBEDDING in adapter.capabilities:
                        ek = ("emb", _key(adapter))
                        if ek not in ctx.cache:
                            with ctx.timed(_key(adapter), EMBEDDING, len(test)):
                                ctx.cache[ek] = (adapter.embed(hide(train)), adapter.embed(hide(test)))
                        etr, ete = ctx.cache[ek]
                        probs = B.linear_probe(etr[idx], ysub, ete, K, seed)
                    else:
                        continue
                    recs += self.emit(ctx, adapter, f"mode/{protocol}@{frac:g}", len(test),
                                      classification_metrics(probs, yte), ctx.cache[ck],
                                      ["handcrafted_gbdt", "majority"], seed, protocol)
        return recs

    @staticmethod
    def _stratified(y, frac, seed):
        if frac >= 1:
            return np.arange(len(y))
        rng = np.random.default_rng(seed)
        out = [rng.choice(np.where(y == c)[0], max(1, int(round(frac * np.sum(y == c)))), replace=False)
               for c in np.unique(y)]
        return np.sort(np.concatenate(out))


# =========================================================================== #
class GenerationTask(Task):
    name, capability = "generation", GENERATION

    def _stats(self, ctx, ds, generated: bool = False):
        # generators emit dwell points (e.g. arrival + departure per visit), so generated data always uses
        # point-based detection per trajectory; real data uses the configured method
        sp = (detect_staypoints(ds, ctx.cfg.staypoint_dist_m, ctx.cfg.staypoint_time_s, by_trajectory=True)
              if generated else ctx.detect_staypoints(ds))
        return G.trajectory_stats(ds.points, sp, ctx.grid)

    def run(self, adapter, ctx):
        cfg, recs = ctx.cfg, []
        test = ctx.splits["test"]
        n = min(test.points.traj_id.nunique(), cfg.generation_max_trajectories)
        if "gen_real" not in ctx.cache:
            ctx.cache["gen_real"] = self._stats(ctx, test)
            # real-vs-real noise floor: n REAL train trajectories, same sample size as the generator's output
            tr = ctx.splits["train"].points
            ids = np.random.default_rng(cfg.split_seed).choice(tr.traj_id.unique(), n, replace=False)
            ctx.cache["gen_floor"] = self._stats(ctx, MobilityDataset(tr[tr.traj_id.isin(ids)], "floor"))
        real, floor = ctx.cache["gen_real"], ctx.cache["gen_floor"]
        for seed in cfg.eval_seeds:
            adapter.reference_staypoints = ctx.staypoints["train"]
            with ctx.timed(_key(adapter), self.capability, n):
                gen_ds = adapter.generate(ctx.splits["train"], n, seed)
            gen = self._stats(ctx, gen_ds, generated=True)
            lk = ("gen_lower", seed)
            if lk not in ctx.cache:
                ctx.cache[lk] = self._stats(ctx, B.uniform_bbox_generator(ctx.splits["train"], n, seed), generated=True)
            lower = ctx.cache[lk]
            for stat in [s for s in real if not s.startswith("_")]:
                r = real[stat].to_numpy(float)
                bins = G.make_bins(r, stat)
                g = gen.get(stat, None)
                g = np.array([]) if g is None else g.to_numpy(float)
                fv = G.compare_stat(r, floor[stat].to_numpy(float), bins) if stat in floor else {}
                lv = G.compare_stat(r, lower[stat].to_numpy(float), bins) if stat in lower else {}
                mv = G.compare_stat(r, g, bins)
                for metric in ("jsd", "w1"):
                    stat_fn = lambda i, metric=metric: G.compare_stat(r, g[i], bins)[metric] if len(g) else np.nan
                    lo, hi = bootstrap(stat_fn, max(len(g), 1), min(cfg.n_boot, 200), seed=seed)
                    recs.append(ResultRecord(
                        model=adapter.name, run_tag=adapter.run_tag, task=f"generation/{stat}",
                        metric=f"{metric}:{stat}", value=mv[metric], ci_low=lo, ci_high=hi, n=len(g),
                        baseline="uniform_bbox", baseline_value=lv.get(metric),
                        skill=skill_score(metric, mv[metric], lv.get(metric), fv.get(metric)),
                        eval_seed=seed, dataset=ctx.dataset_name))
                    fk = ("gen_ref_emitted", stat, metric)
                    if fk not in ctx.emitted_baselines and seed == cfg.eval_seeds[0]:
                        ctx.emitted_baselines.add(fk)
                        for nm, val in (("real_noise_floor", fv), ("uniform_bbox", lv)):
                            if metric in val:
                                recs.append(ResultRecord(model=f"baseline:{nm}", run_tag="-",
                                                         task=f"generation/{stat}", metric=f"{metric}:{stat}",
                                                         value=val[metric], eval_seed=seed, dataset=ctx.dataset_name))
            if "_cells" in real and "_cells" in gen:
                v = G.cell_jsd(real["_cells"].to_numpy(), gen["_cells"].to_numpy(), ctx.grid.n_cells)
                b = G.cell_jsd(real["_cells"].to_numpy(), lower["_cells"].to_numpy(), ctx.grid.n_cells)
                f = G.cell_jsd(real["_cells"].to_numpy(), floor["_cells"].to_numpy(), ctx.grid.n_cells)
                recs.append(ResultRecord(model=adapter.name, run_tag=adapter.run_tag, task="generation/visited_cells",
                                         metric="jsd:visited_cells", value=v, baseline="uniform_bbox",
                                         baseline_value=b, skill=skill_score("jsd", v, b, f), eval_seed=seed,
                                         dataset=ctx.dataset_name))
            # memorisation check: distributional metrics cannot tell a copier from a good model
            trp = ctx.splits["train"].points
            if "gen_nn_ref" not in ctx.cache:
                ctx.cache["gen_nn_ref"] = np.percentile(G.nearest_train_distance(test.points, trp, seed=cfg.split_seed), 5)
            nn = G.nearest_train_distance(gen_ds.points, trp, seed=cfg.split_seed)
            thr = ctx.cache["gen_nn_ref"]
            for metric, vals in (("copy_rate", (nn < thr).astype(float)), ("nn_train_dist_m", nn)):
                res = evaluate_with_ci(metric, vals, n_boot=min(cfg.n_boot, 200), seed=seed)
                recs.append(ResultRecord(model=adapter.name, run_tag=adapter.run_tag, task="generation/memorisation",
                                         metric=metric, n=len(nn), eval_seed=seed, dataset=ctx.dataset_name, **res))
            rho = G.paired_spearman(real["radius_of_gyration"], gen["radius_of_gyration"])
            if rho is not None:
                recs.append(ResultRecord(model=adapter.name, run_tag=adapter.run_tag, task="generation/radius_of_gyration",
                                         metric="spearman_paired:radius_of_gyration", value=rho, eval_seed=seed,
                                         dataset=ctx.dataset_name))
        return recs


# =========================================================================== #
class EfficiencyTask(Task):
    """Run LAST: reads latencies accumulated by the other tasks."""
    name, capability = "efficiency", ""

    def applicable(self, adapter):
        return True

    def run(self, adapter, ctx):
        recs = []
        npar = adapter.num_parameters()
        if npar is not None:
            recs.append(ResultRecord(model=adapter.name, run_tag=adapter.run_tag, task="efficiency",
                                     metric="n_parameters", value=npar, dataset=ctx.dataset_name))
        for cap, (sec, n) in ctx.latency[_key(adapter)].items():
            if n:
                recs.append(ResultRecord(model=adapter.name, run_tag=adapter.run_tag, task=f"efficiency/{cap}",
                                         metric="latency_ms_per_sample", value=1000 * sec / n, n=n,
                                         dataset=ctx.dataset_name))
        # Latency is timed while a task runs, so a task whose results were kept from an earlier
        # attempt has no timing in this process. That is a real gap, not a silent one: say so.
        kept = sorted(t for (m, rt, t) in getattr(ctx, "resumed_units", None) or ()
                      if (m, rt) == (adapter.name, adapter.run_tag))
        if kept:
            log.warning(f"{adapter.name}: no latency measured for {', '.join(kept)} - those results were "
                        f"kept from an earlier run, so those tasks were not timed here. Re-run without "
                        f"--resume for complete efficiency numbers.")
            for r in recs:
                r.flags.append(f"resumed run: {len(kept)} task(s) not timed in this process")
        return recs


def default_tasks(cfg) -> List[Task]:
    return ([RecoveryTask(), NextLocationTask()] + [ContinuousValueTask(t) for t in cfg.continuous_targets]
            + [ModeClassificationTask(), GenerationTask(), EfficiencyTask()])
