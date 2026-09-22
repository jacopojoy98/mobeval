"""Evaluation tasks. Each task: builds canonical inputs from the context,
calls ONE adapter capability, scores model AND baselines on identical samples,
and emits ResultRecords with bootstrap CIs and paired skill-score CIs."""
from __future__ import annotations

import logging
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

from . import baselines as B
from .adapters.base import (CONTINUOUS, EMBEDDING, GENERATION, MODE_CLASSIFICATION, NEXT_LOCATION, RECOVERY,
                            ContinuousPrediction, MobilityModelAdapter, TargetGuard)
from .context import EvalContext
from .data import MobilityDataset, TrajectoryBatch, VisitBatch, detect_staypoints, make_mask
from .geo import haversine_m
from .metrics import generative as G
from .metrics.classification import classification_metrics, macro_f1, normalise_probs, ranking_metrics
from .metrics.detection import detection_metrics
from .metrics.probabilistic import continuous_metrics, pit_ks
from .metrics.reconstruction import recovery_metrics
from .results import ResultRecord
from .stats import bootstrap, evaluate_with_ci, skill_score

log = logging.getLogger("mobeval")

Scores = Tuple[Dict[str, np.ndarray], Dict[str, Callable]]   # (per-sample, set-level)


def _key(a: MobilityModelAdapter) -> str:
    return f"{a.name}@{a.run_tag}"


def visits_as_windows(v: VisitBatch, tag: str = "") -> TrajectoryBatch:
    """The visit context seen as a short trajectory, so ANY model with an embedding can encode it.

    This is what lets a masked-reconstruction encoder be probed on next location and travel
    time: it never had a head for those, but it can encode the C context visits as C points and
    a linear probe can be fitted on that. Target fields are not touched here - callers pass an
    already-hidden VisitBatch, so nothing about the target can reach the embedding.

    `tag` names the split. Adapters cache embeddings on (traj_id, first time, last time), so
    ids restarting at 0 in every split could collide on coarsely quantised data - hourly bins
    and a repeating weekly schedule are enough - and a test context would then be handed a
    training sample's embedding.
    """
    ids = np.array([f"vctx:{tag}:{i}" for i in range(len(v))], dtype=object)
    return TrajectoryBatch(v.ctx_lat, v.ctx_lon, v.ctx_t_arrive, v.user_id, ids, None)


def _context_embeddings(adapter, ctx, reveal=()):
    """Embed the visit context of every split once per adapter, and cache it.

    Deliberately embeds all three splits regardless of what the caller needs right now: the
    cache is keyed on the adapter, so a caller that only asked for train+test would otherwise
    poison it for the next one, and the continuous probe would silently lose the validation
    split it fits its predictive spread on.
    """
    key = ("visit_emb", _key(adapter), reveal)
    if key not in ctx.cache:
        out = {}
        for name in ("train", "val", "test"):
            v = ctx.visits.get(name)
            if v is None or not len(v):
                continue
            w = visits_as_windows(TargetGuard.hide_visits(v, reveal), tag=f"{name}:{reveal}")
            with ctx.timed(_key(adapter), EMBEDDING, len(v)):
                out[name] = adapter.embed(w)
        ctx.cache[key] = out
    return ctx.cache[key]


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

    def applicable(self, adapter):
        return bool({NEXT_LOCATION, EMBEDDING} & adapter.capabilities)

    def _probe_score(self, adapter, ctx, v, grid) -> Scores:
        """Linear probe over the most-visited training cells, on the frozen visit-context
        embedding. Metrics come out on the same shared grid as the native protocol."""
        from .metrics.classification import candidate_ranking_metrics
        from .probes import location_probe
        cfg = ctx.cfg
        emb = _context_embeddings(adapter, ctx)
        tr_v = ctx.visits["train"]
        probs, cand, counts = location_probe(emb["train"], tr_v.tgt_cell, emb["test"], grid.n_cells,
                                             top_k=cfg.probe_top_k, seed=cfg.eval_seeds[0],
                                             device=str(getattr(adapter, "device", "auto")),
                                             max_train=cfg.probe_max_train)
        rm = candidate_ranking_metrics(probs, cand, v.tgt_cell, counts)
        # Reported, not just logged: acc@1 cannot exceed this, so a reader needs it next to the
        # number rather than buried in a job log they may never see.
        rm["probe_coverage"] = rm.pop("_coverage")
        cover = float(rm["probe_coverage"].mean())
        top1 = np.column_stack(grid.centroid(cand[probs.argmax(1)]))
        d = haversine_m(top1[:, 0], top1[:, 1], v.tgt_lat, v.tgt_lon)
        rm.update({"dist_err_m": d, "median_dist_err_m": d, "acc_1km": (d <= 1000).astype(float)})
        log.info(f"[{_key(adapter)}] next_location/linear_probe: {len(cand)} candidate cells cover "
                 f"{cover:.1%} of test targets - that is this protocol's accuracy ceiling")
        return rm, {}

    def run(self, adapter, ctx):
        v, grid, recs = ctx.visits["test"], ctx.grid, []
        seed = ctx.cfg.eval_seeds[0]                   # deterministic task: one pass
        if "loc_baselines" not in ctx.cache:
            lb = B.LocationBaselines(ctx.visits["train"], grid.n_cells)
            ctx.cache["loc_baselines"] = {}
            for bn, fn in [("markov1", lb.markov1), ("user_frequent", lb.user_frequent),
                           ("global_popular", lb.global_popular)]:
                bp = fn(v)
                ctx.cache["loc_baselines"][bn] = self._score(bp, np.column_stack(grid.centroid(bp.argmax(1))), v, True)
        for protocol in ctx.cfg.location_protocols:
            if protocol == "native" and NEXT_LOCATION in adapter.capabilities:
                adapter.prepare(self.name, ctx.visits.get("train"), ctx.visits.get("val"))
                with ctx.timed(_key(adapter), self.capability, len(v)):
                    pred = adapter.predict_location(TargetGuard.hide_visits(v), grid)
                pred.check()
                p, top1, ranked = self._grid_probs(pred, grid, len(v))
                scores = self._score(p, top1, v, ranked)
            elif protocol == "linear_probe" and EMBEDDING in adapter.capabilities:
                scores = self._probe_score(adapter, ctx, v, grid)
            else:
                continue
            task = "next_location" if protocol == "native" else f"next_location/{protocol}"
            recs += self.emit(ctx, adapter, task, len(v), scores, ctx.cache["loc_baselines"],
                              ["markov1", "user_frequent"], seed, protocol)
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

    def applicable(self, adapter):
        return bool({CONTINUOUS, EMBEDDING} & adapter.capabilities)

    @staticmethod
    def _revealed_features(v, reveal):
        """The teacher-forced fields, as extra probe inputs.

        `visits_as_windows` only ever reads the CONTEXT, so `reveal` could not reach the probe
        through the embedding. Without this the probe would be an unconditional predictor
        reported next to a native head that was told the answer's location, under a task name
        claiming both were given it. Appending the revealed fields makes the two comparable.
        """
        cols = []
        if "location" in reveal:
            cols += [v.tgt_lat, v.tgt_lon]
        if "arrival" in reveal:
            t = np.asarray(v.tgt_t_arrive, float)
            cols += [np.sin(2 * np.pi * (t % 86400) / 86400), np.cos(2 * np.pi * (t % 86400) / 86400)]
        return np.column_stack(cols) if cols else None

    def _probe(self, adapter, ctx, reveal):
        """Ridge on the frozen visit-context embedding -> a log-normal predictive distribution.
        The spread is fitted on VALIDATION residuals, never on the test data being scored."""
        from .probes import continuous_probe
        emb = _context_embeddings(adapter, ctx, reveal)
        feats = {}
        for name, e in emb.items():
            extra = self._revealed_features(ctx.visits[name], reveal)
            feats[name] = e if extra is None else np.column_stack([e, extra])
        ytr = self._y(ctx.visits["train"])
        ok = self._valid(ytr, ctx.cfg)
        ev, yv = None, None
        if "val" in feats:
            yval = self._y(ctx.visits["val"])
            okv = self._valid(yval, ctx.cfg)
            ev, yv = feats["val"][okv], yval[okv]
        return continuous_probe(feats["train"][ok], ytr[ok], feats["test"], ev, yv)

    def run(self, adapter, ctx):
        cfg, v, recs = ctx.cfg, ctx.visits["test"], []
        reveal = tuple(cfg.continuous_reveal.get(self.target, ()))
        keep = self._valid(self._y(v), cfg)
        seed, y = cfg.eval_seeds[0], self._y(v)
        ck = ("cont_baselines", self.target, reveal)
        if ck not in ctx.cache:
            ytr = self._y(ctx.visits["train"])
            cb = B.ContinuousBaselines(ytr[self._valid(ytr, cfg)], seed=seed)
            n = int(keep.sum())
            prob = continuous_metrics(y[keep], None, cb.mixture(n))
            pt = continuous_metrics(y[keep], cb.point(n))
            prob["mae_min"], prob["rmse_min"] = pt["mae_min"], pt["rmse_min"]   # median is the MAE-optimal point
            ctx.cache[ck] = {"train_marginal": self._with_pit(prob)}
        for protocol in cfg.continuous_protocols:
            sigma, samples = None, None
            if protocol == "native" and CONTINUOUS in adapter.capabilities:
                adapter.prepare(self.name, ctx.visits.get("train"), ctx.visits.get("val"))
                with ctx.timed(_key(adapter), f"{self.capability}/{self.target}", len(v)):
                    pred = adapter.predict_continuous(TargetGuard.hide_visits(v, reveal), self.target)
                point, mix, samples = self._to_minutes(pred)
                vv = ctx.visits.get("val")
                if mix is None and samples is None and vv is not None and len(vv):
                    # point model: sigma from VALIDATION residuals. `.get` because a split can
                    # legitimately yield no visit sequences, and a KeyError here would take down
                    # the whole continuous task rather than just the predictive spread.
                    vp = adapter.predict_continuous(TargetGuard.hide_visits(vv, reveal), self.target).point
                    okv = self._valid(self._y(vv), cfg)
                    resid = (self._y(vv) - np.asarray(vp) / 60.0)[okv]
                    s = float(np.std(resid)) if resid.size > 1 else 0.0
                    sigma = s if np.isfinite(s) and s > 0 else 1.0
            elif protocol == "linear_probe" and EMBEDDING in adapter.capabilities:
                point, mix = self._probe(adapter, ctx, reveal)
            else:
                continue
            sel = lambda a: None if a is None else a[keep]
            m = mix
            if m is not None:
                from .metrics.probabilistic import Mixture
                m = Mixture(m.weights[keep], m.means[keep], m.stds[keep], m.space)
            model = self._with_pit(continuous_metrics(y[keep], sel(point), m, sel(samples), sigma))
            task = self.name + (f"/{protocol}" if protocol != "native" else "")
            task += f"|given:{'+'.join(reveal)}" if reveal else ""
            if self.target == "travel_time" and cfg.travel_time_max_h is not None:
                task += f"|<={cfg.travel_time_max_h:g}h"
            recs += self.emit(ctx, adapter, task, int(keep.sum()), model, ctx.cache[ck],
                              ["train_marginal"], seed, protocol)
        return recs


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
            # `_cells` exists only when staypoints were detected, which is not guaranteed for the
            # BASELINES either: the uniform-bbox generator scatters points at random, so on data
            # where stays are inferred from trip gaps it can produce none at all. Guarding only
            # `real` and `gen` made that a KeyError that killed the whole generation task.
            if "_cells" in real and "_cells" in gen:
                cells = lambda d: d["_cells"].to_numpy() if "_cells" in d else None
                v = G.cell_jsd(real["_cells"].to_numpy(), gen["_cells"].to_numpy(), ctx.grid.n_cells)
                b = None if cells(lower) is None else G.cell_jsd(real["_cells"].to_numpy(), cells(lower),
                                                                 ctx.grid.n_cells)
                f = None if cells(floor) is None else G.cell_jsd(real["_cells"].to_numpy(), cells(floor),
                                                                 ctx.grid.n_cells)
                if b is None:
                    log.warning(f"{adapter.name}: no staypoints detected in the uniform-bbox baseline's output, "
                                f"so visited_cells has no baseline to compare against (skill omitted)")
                recs.append(ResultRecord(model=adapter.name, run_tag=adapter.run_tag, task="generation/visited_cells",
                                         metric="jsd:visited_cells", value=v,
                                         baseline="uniform_bbox" if b is not None else None,
                                         baseline_value=b,
                                         skill=None if b is None else skill_score("jsd", v, b, f),
                                         eval_seed=seed, dataset=ctx.dataset_name))
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
class UserIdentificationTask(Task):
    """Can a linear model recover WHO produced a window from the frozen embedding?

    A standard representation-learning probe, and for mobility it needs a control. Individual
    identity is largely home and work location, so an embedding that merely records absolute
    position will re-identify users very well while having learned nothing about behaviour.
    The `mean_location` baseline is exactly that embedding - latitude/longitude statistics and
    nothing else - so a model only demonstrates that it captures individual mobility style if
    it clears that line, not if it merely beats chance.

    The task is closed-set: the same users appear in train and test, and the probe chooses
    among them. Open-set re-identification is a different (harder) problem.
    """
    name, capability = "user_identification", EMBEDDING

    def applicable(self, adapter):
        return EMBEDDING in adapter.capabilities

    @staticmethod
    def _cohort(ctx):
        """Users with enough windows in BOTH splits, capped at cfg.user_id_max_users.

        Chosen once and cached, so every model is scored on exactly the same users and the
        same windows - otherwise the number would depend on which model ran first.
        """
        if "user_cohort" in ctx.cache:
            return ctx.cache["user_cohort"]
        cfg = ctx.cfg
        tr, te = ctx.windows.get("train"), ctx.windows.get("test")
        if tr is None or te is None:
            ctx.cache["user_cohort"] = None
            return None
        ctr = pd.Series(tr.user_id).value_counts()
        cte = pd.Series(te.user_id).value_counts()
        ok = [u for u in ctr.index
              if ctr.get(u, 0) >= cfg.user_id_min_windows and cte.get(u, 0) >= cfg.user_id_min_windows]
        ok = sorted(ok, key=lambda u: -min(ctr[u], cte[u]))[:cfg.user_id_max_users]
        if len(ok) < 2:
            ctx.cache["user_cohort"] = None
            return None
        users = {u: i for i, u in enumerate(sorted(ok, key=str))}
        itr = np.flatnonzero(np.isin(tr.user_id, list(users)))
        ite = np.flatnonzero(np.isin(te.user_id, list(users)))
        ytr = np.array([users[u] for u in tr.user_id[itr]])
        yte = np.array([users[u] for u in te.user_id[ite]])
        ctx.cache["user_cohort"] = (itr, ite, ytr, yte, len(users))
        log.info(f"user identification: {len(users)} users, {len(itr)} train / {len(ite)} test windows "
                 f"(chance accuracy {1 / len(users):.3%})")
        return ctx.cache["user_cohort"]

    @staticmethod
    def _scores(probs, y, K) -> Scores:
        """Ranking + calibration metrics under `user_`-prefixed names, so they stay out of the
        location and classification families in every summary."""
        rm = ranking_metrics(normalise_probs(probs), y)
        per = {"user_acc@1": rm["acc@1"], "user_nll": rm["loc_nll"]}
        # With few users a top-k metric is 1.0 for every model by construction and says nothing,
        # so it is omitted rather than printed as a fake perfect score.
        if K > 5:
            per["user_acc@5"] = rm["acc@5"]
        if K > 20:
            per["user_mrr@20"] = rm["mrr@20"]
        pred = probs.argmax(1)
        return per, {"user_macro_f1": lambda i: macro_f1(y[i], pred[i], K)}

    def run(self, adapter, ctx):
        cohort = self._cohort(ctx)
        if cohort is None:
            ctx.skipped.append((_key(adapter), self.name,
                                f"fewer than 2 users have >= {ctx.cfg.user_id_min_windows} windows in "
                                f"both the train and test splits"))
            return []
        itr, ite, ytr, yte, K = cohort
        tr, te = ctx.windows["train"].take(itr), ctx.windows["test"].take(ite)
        seed = ctx.cfg.eval_seeds[0]
        if "user_baselines" not in ctx.cache:
            ctx.cache["user_baselines"] = {
                # like-for-like with the model's own linear probe; this is the one skill is measured against
                "mean_location": self._scores(B.mean_location_classifier(tr, ytr, te, K, seed), yte, K),
                # upper bound on what position alone can explain, whatever the decoder
                "mean_location_gbdt": self._scores(
                    B.mean_location_classifier(tr, ytr, te, K, seed, nonlinear=True), yte, K),
                "majority": self._scores(B.majority_class_probs(ytr, len(te), K), yte, K)}
        ek = ("user_emb", _key(adapter))
        if ek not in ctx.cache:
            with ctx.timed(_key(adapter), EMBEDDING, len(te)):
                ctx.cache[ek] = (adapter.embed(tr), adapter.embed(te))
        etr, ete = ctx.cache[ek]
        probs = B.linear_probe(etr, ytr, ete, K, seed)
        return self.emit(ctx, adapter, f"user_identification@{K}users", len(te),
                         self._scores(probs, yte, K), ctx.cache["user_baselines"],
                         ["mean_location", "mean_location_gbdt", "majority"], seed, "linear_probe")


# =========================================================================== #
class AnomalyDetectionTask(Task):
    """Score every test window for how unusual it is, having seen only normal training data.

    There are no anomaly labels in a raw GPS panel, so anomalies are injected (see
    `mobeval.anomalies`) and each kind is reported separately. That separation is the point:
    `teleport` is caught by any speed check, while `detour` and `loop` preserve every step
    length exactly, so a kinematic detector is at chance on them by construction and only a
    model that has learned what plausible movement looks like can do better.

    Two protocols, both unsupervised - the labels are used to score, never to fit:
      embedding_knn   distance to the nearest training embeddings
      reconstruction  error when asked to fill in masked points
    """
    name, capability = "anomaly_detection", EMBEDDING

    def applicable(self, adapter):
        return bool({EMBEDDING, RECOVERY} & adapter.capabilities)

    def _injected(self, ctx, kind, seed):
        """Corrupt the test windows once per (kind, seed) so every model sees identical data."""
        k = ("anomaly", kind, seed)
        if k not in ctx.cache:
            from . import anomalies
            ctx.cache[k] = anomalies.inject(ctx.windows["test"], kind, ctx.cfg.anomaly_rate, seed)
        return ctx.cache[k]

    def _baselines(self, ctx, aset, seed):
        k = ("anomaly_baselines", aset.kind, seed)
        if k not in ctx.cache:
            y = aset.is_anomalous
            ctx.cache[k] = {
                "kinematic_knn": detection_metrics(y, B.kinematic_knn_score(ctx.windows["train"], aset.batch,
                                                                           ctx.cfg.anomaly_knn_k)),
                "max_step": detection_metrics(y, B.max_step_score(aset.batch))}
        return ctx.cache[k]

    def _embedding_score(self, adapter, ctx, aset):
        from .metrics.detection import knn_distance
        tk = ("anomaly_train_emb", _key(adapter))
        if tk not in ctx.cache:
            ctx.cache[tk] = adapter.embed(ctx.windows["train"])
        with ctx.timed(_key(adapter), EMBEDDING, len(aset.batch)):
            ete = adapter.embed(aset.batch)
        return knn_distance(ctx.cache[tk], ete, k=ctx.cfg.anomaly_knn_k)

    def _reconstruction_score(self, adapter, ctx, aset, seed):
        """How badly the model fills in masked points. An anomalous window is one the model
        cannot predict, so its reconstruction error should be larger."""
        b = aset.batch
        mask = make_mask(len(b), b.length, ctx.cfg.anomaly_mask_ratio, "random", seed)
        with ctx.timed(_key(adapter), RECOVERY, len(b)):
            plat, plon = adapter.reconstruct(TargetGuard.hide_masked(b, mask), mask)
        err = haversine_m(np.asarray(plat), np.asarray(plon), b.lat, b.lon)
        n_masked = mask.sum(1)
        total = np.where(mask, err, 0.0).sum(1)
        # A row with nothing masked has no evidence either way; NaN keeps it out of the
        # ranking rather than giving it a fabricated score of 0 (which would rank it "normal").
        return np.where(n_masked > 0, total / np.maximum(n_masked, 1), np.nan)

    def run(self, adapter, ctx):
        cfg, recs = ctx.cfg, []
        if "test" not in ctx.windows or len(ctx.windows["test"]) < 20:
            ctx.skipped.append((_key(adapter), self.name, "too few test windows to inject anomalies into"))
            return recs
        for kind in cfg.anomaly_kinds:
            for seed in cfg.eval_seeds:
                aset = self._injected(ctx, kind, seed)
                base = self._baselines(ctx, aset, seed)
                for protocol in cfg.anomaly_protocols:
                    if protocol == "embedding_knn" and EMBEDDING in adapter.capabilities:
                        score = self._embedding_score(adapter, ctx, aset)
                    elif protocol == "reconstruction" and RECOVERY in adapter.capabilities:
                        score = self._reconstruction_score(adapter, ctx, aset, seed)
                    else:
                        continue
                    recs += self.emit(ctx, adapter, f"anomaly/{kind}", len(aset),
                                      detection_metrics(aset.is_anomalous, score), base,
                                      ["kinematic_knn", "max_step"], seed, protocol)
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
            + [ModeClassificationTask(), GenerationTask(), UserIdentificationTask(),
               AnomalyDetectionTask(), EfficiencyTask()])
