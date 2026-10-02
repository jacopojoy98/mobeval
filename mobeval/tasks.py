"""Evaluation tasks. Each task: builds canonical inputs from the context,
calls ONE adapter capability, scores model AND baselines on identical samples,
and emits ResultRecords with bootstrap CIs and paired skill-score CIs."""
from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

from . import baselines as B
from . import progress
from .adapters.base import (CONTINUOUS, CROSS_MODAL, EMBEDDING, GENERATION, MODE_CLASSIFICATION, NEXT_LOCATION, RECOVERY,
                            ContinuousPrediction, MobilityModelAdapter, TargetGuard)
from .context import EvalContext
from .paper_metrics import level_for as paper_level
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


def _no_visits(ctx, adapter, task) -> bool:
    """Visit tasks need train and test visit sequences; say why they are missing instead of failing."""
    missing = [k for k in ("train", "test") if not len(ctx.visits.get(k, ()) or ())]
    if missing:
        key = (_key(adapter), task, "no visit sequences in " + " and ".join(missing) +
               " (too few staypoints for eval.visit_context?)")
        if key not in ctx.skipped:
            ctx.skipped.append(key)
        return True
    return False


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
                                     paper_match=paper_level(adapter, task, metric, protocol, ctx), **res))
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


    @staticmethod
    def try_protocol(ctx, adapter, task_name, protocol, fn):
        """Run one protocol's scoring, recording a failure instead of propagating it.

        A model that cannot run one protocol must not lose the others. CLIPMobility's native
        location head needs a ~450k-way output layer on a 500 m grid over a region (~20 GB once
        its optimiser state and logits are counted); when that raised, the whole next_location
        task failed for that model, and its linear-probe result - which costs nothing and would
        have worked - was discarded with it.
        """
        try:
            return fn()
        except progress.Interrupted:
            raise
        except Exception as e:                                     # noqa: BLE001
            import traceback
            key = f"{adapter.name}@{adapter.run_tag}"
            ctx.errors.append((key, f"{task_name}/{protocol}", repr(e), traceback.format_exc()))
            log.error(f"[{key}] {task_name} ({protocol}) failed: {e!r} - other protocols still run")
            progress.get().error(f"{key} {task_name}/{protocol}: {e!r}", model=adapter.name, task=task_name)
            return None


# =========================================================================== #
class RecoveryTask(Task):
    name, capability = "recovery", RECOVERY

    def run(self, adapter, ctx):
        batch, cfg, recs = ctx.windows["test"], ctx.cfg, []
        runs = [(k, r, f"recovery/{k}@{r:g}") for k in cfg.recovery_kinds for r in cfg.recovery_ratios]
        runs += [(k, r, f"recovery/{k}:{r:g}") for k, r in (cfg.recovery_schemes or ())]
        for kind, ratio, task in runs:
            # A seed only changes random/block masks; the fixed schemes are scored once.
            seeds = cfg.eval_seeds if kind in ("random", "block") else cfg.eval_seeds[:1]
            for seed in seeds:
                mask = make_mask(len(batch), batch.length, ratio, kind, seed, cfg.recovery_keep_endpoints)
                hidden = TargetGuard.hide_masked(batch, mask)
                with ctx.timed(_key(adapter), self.capability, len(batch)):
                    plat, plon = adapter.reconstruct(hidden, mask)
                score = lambda la, lo, m=mask: (recovery_metrics(la, lo, batch.lat, batch.lon, m, ctx.grid,
                                                                 cfg.recovery_dtw), {})
                ck = ("recovery_baselines", kind, ratio, seed)
                if ck not in ctx.cache:
                    base = {"linear_interp": score(*B.linear_interpolation(hidden, mask)),
                            "last_observed": score(*B.last_observed(hidden, mask))}
                    if kind == "last":        # interpolation cannot extrapolate: add dead reckoning
                        base["constant_velocity"] = score(*B.constant_velocity(hidden, mask))
                    ctx.cache[ck] = base
                skill = ["constant_velocity"] if kind == "last" else ["linear_interp"]
                recs += self.emit(ctx, adapter, task, len(batch), score(plat, plon), ctx.cache[ck], skill, seed)
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
        rm = ranking_metrics(p, v.tgt_cell, ks=(1, 5, 10, 20))     # 10 and 20: TrajGPT's Acc@k
        if not ranked:                                  # point predictor: top-k / NLL undefined
            rm = {"acc@1": rm["acc@1"]}
        d = haversine_m(top1_latlon[:, 0], top1_latlon[:, 1], v.tgt_lat, v.tgt_lon)
        rm.update({"dist_err_m": d, "median_dist_err_m": d, "acc_1km": (d <= 1000).astype(float)})
        return rm, {}

    # A dense (n, n_cells) score matrix is 18 GB for 5,000 samples on a 500 m grid covering a
    # region - more than a whole job's memory allowance, and the pipeline builds one per model
    # and one per baseline. Every location metric is per-sample, so scoring in chunks is exact:
    # the peak drops by the chunking factor and the numbers do not move at all.
    SCORE_CHUNK = 512

    @classmethod
    def _score_chunked(cls, prob_fn, v, grid, ranked: bool, top1=None) -> Scores:
        """Score in chunks. `prob_fn(sub_batch, idx) -> (len(idx), n_cells)`.

        `top1` is the per-sample predicted (lat, lon) when the predictor has its own notion of
        a best location (a token centroid); otherwise the argmax cell centroid is used.
        """
        parts: Dict[str, List[np.ndarray]] = {}
        for s in range(0, len(v), cls.SCORE_CHUNK):
            idx = np.arange(s, min(s + cls.SCORE_CHUNK, len(v)))
            sub = v.take(idx)
            p = prob_fn(sub, idx)
            best = top1[idx] if top1 is not None else np.column_stack(grid.centroid(p.argmax(1)))
            for k, arr in cls._score(p, best, sub, ranked)[0].items():
                parts.setdefault(k, []).append(arr)
            del p
        return {k: np.concatenate(vs) for k, vs in parts.items()}, {}

    @classmethod
    def _model_scores(cls, pred, v, grid) -> Scores:
        """Score a LocationPrediction without ever holding a full-grid matrix for every sample.

        `token_scores` and `latlon` are compact per-sample forms that only become huge once
        mapped onto the shared grid, so that mapping is done a chunk at a time. `grid_scores` is
        already dense when the adapter hands it over, so there is nothing left to save there.
        """
        if pred.grid_scores is not None:
            p = normalise_probs(pred.grid_scores)
            return cls._score(p, np.column_stack(grid.centroid(p.argmax(1))), v, True)
        if pred.token_scores is not None:
            pt = normalise_probs(pred.token_scores)
            tl = np.asarray(pred.token_latlon, float)
            cells = grid.cell_of(tl[:, 0], tl[:, 1])
            M = sparse.csr_matrix((np.ones(len(cells)), (np.arange(len(cells)), cells)),
                                  shape=(len(cells), grid.n_cells))
            # top-1 comes from the model's own vocabulary, which is finer than the shared grid
            return cls._score_chunked(lambda sub, idx: np.asarray((M.T @ pt[idx].T).T), v, grid,
                                      True, top1=tl[pt.argmax(1)])
        ll = np.asarray(pred.latlon, float)

        def point_probs(sub, idx):
            p = np.zeros((len(idx), grid.n_cells))
            p[np.arange(len(idx)), grid.cell_of(ll[idx, 0], ll[idx, 1])] = 1.0
            return p

        return cls._score_chunked(point_probs, v, grid, False, top1=ll)

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
        rm = candidate_ranking_metrics(probs, cand, v.tgt_cell, counts, ks=(1, 5, 10, 20))
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
        if _no_visits(ctx, adapter, self.name):
            return []
        v, grid, recs = ctx.visits["test"], ctx.grid, []
        seed = ctx.cfg.eval_seeds[0]                   # deterministic task: one pass
        if "loc_baselines" not in ctx.cache:
            lb = B.LocationBaselines(ctx.visits["train"], grid.n_cells)
            ctx.cache["loc_baselines"] = {}
            for bn, fn in [("markov1", lb.markov1), ("user_frequent", lb.user_frequent),
                           ("global_popular", lb.global_popular)]:
                # chunked: each of these would otherwise build its own full-grid score matrix
                ctx.cache["loc_baselines"][bn] = self._score_chunked(
                    lambda sub, idx, _f=fn: _f(sub), v, grid, True)
        for protocol in ctx.cfg.location_protocols:
            if protocol == "native" and NEXT_LOCATION in adapter.capabilities:
                def _native():
                    adapter.prepare(self.name, ctx.visits.get("train"), ctx.visits.get("val"))
                    with ctx.timed(_key(adapter), self.capability, len(v)):
                        pred = adapter.predict_location(TargetGuard.hide_visits(v), grid)
                    pred.check()
                    return self._model_scores(pred, v, grid)
                scores = self.try_protocol(ctx, adapter, self.name, protocol, _native)
            elif protocol == "linear_probe" and EMBEDDING in adapter.capabilities:
                scores = self.try_protocol(ctx, adapter, self.name, protocol,
                                           lambda: self._probe_score(adapter, ctx, v, grid))
            else:
                continue
            if scores is None:                      # this protocol failed; the others still run
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

    # TrajGPT's factorisation p(region) p(travel | region) p(duration | region, travel): travel time
    # given the next visit's location, duration given its location and arrival. (The released code's
    # heads instead read the target's own arrival/departure - a leak an honest evaluation cannot
    # reproduce; see MODELS.md and paper_metrics.py.)
    PAPER_REVEAL = {"travel_time": ("location",), "duration": ("location", "arrival")}

    def run(self, adapter, ctx):
        if _no_visits(ctx, adapter, self.name):
            return []
        cfg, recs = ctx.cfg, []
        reveals = [tuple(cfg.continuous_reveal.get(self.target, ()))]
        if cfg.continuous_paper_conditioning and self.PAPER_REVEAL[self.target] not in reveals:
            reveals.append(self.PAPER_REVEAL[self.target])
        for reveal in reveals:
            recs += self._run(adapter, ctx, reveal)
        return recs

    def _run(self, adapter, ctx, reveal):
        cfg, v, recs = ctx.cfg, ctx.visits["test"], []
        keep = self._valid(self._y(v), cfg)
        seed, y = cfg.eval_seeds[0], self._y(v)
        ytr_all = self._y(ctx.visits["train"])
        # TrajGPT's P(+-t) truncates every forecast to [0, max]; the original takes max as the 99th
        # percentile of ALL splits, which reads the test data. The TRAIN percentile is used here.
        upper = float(np.nanpercentile(ytr_all[self._valid(ytr_all, cfg)], 99))
        ck = ("cont_baselines", self.target, reveal)
        if ck not in ctx.cache:
            ytr = ytr_all
            cb = B.ContinuousBaselines(ytr[self._valid(ytr, cfg)], seed=seed)
            n = int(keep.sum())
            prob = continuous_metrics(y[keep], None, cb.mixture(n), support_max=upper)
            pt = continuous_metrics(y[keep], cb.point(n))
            prob["mae_min"], prob["rmse_min"], prob["mape"] = pt["mae_min"], pt["rmse_min"], pt["mape"]
            ctx.cache[ck] = {"train_marginal": self._with_pit(prob)}
        for protocol in cfg.continuous_protocols:
            def _native():
                """-> (point, mixture, samples, sigma) from the model's own head."""
                adapter.prepare(self.name, ctx.visits.get("train"), ctx.visits.get("val"))
                with ctx.timed(_key(adapter), f"{self.capability}/{self.target}", len(v)):
                    pred = adapter.predict_continuous(TargetGuard.hide_visits(v, reveal), self.target)
                pt, mx, sm = self._to_minutes(pred)
                sg, vv = None, ctx.visits.get("val")
                if mx is None and sm is None and vv is not None and len(vv):
                    # point model: sigma from VALIDATION residuals. `.get` because a split can
                    # legitimately yield no visit sequences, and a KeyError here would take down
                    # the whole continuous task rather than just the predictive spread.
                    vp = adapter.predict_continuous(TargetGuard.hide_visits(vv, reveal), self.target).point
                    okv = self._valid(self._y(vv), cfg)
                    resid = (self._y(vv) - np.asarray(vp) / 60.0)[okv]
                    s = float(np.std(resid)) if resid.size > 1 else 0.0
                    sg = s if np.isfinite(s) and s > 0 else 1.0
                return pt, mx, sm, sg

            if protocol == "native" and CONTINUOUS in adapter.capabilities:
                got = self.try_protocol(ctx, adapter, self.name, protocol, _native)
            elif protocol == "linear_probe" and EMBEDDING in adapter.capabilities:
                got = self.try_protocol(ctx, adapter, self.name, protocol,
                                        lambda: (*self._probe(adapter, ctx, reveal), None, None))
            else:
                continue
            if got is None:                         # this protocol failed; the others still run
                continue
            point, mix, samples, sigma = got
            sel = lambda a: None if a is None else a[keep]
            m = mix
            if m is not None:
                from .metrics.probabilistic import Mixture
                m = Mixture(m.weights[keep], m.means[keep], m.stds[keep], m.space)
            model = self._with_pit(continuous_metrics(y[keep], sel(point), m, sel(samples), sigma, upper))
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

    def applicable(self, adapter):
        # A reconstruction model can be rolled out into a generator (see mobeval.rollout), so
        # it belongs here too - under its own protocol, never mixed with a native generator.
        return bool({GENERATION, RECOVERY} & adapter.capabilities)

    def _generate(self, adapter, ctx, protocol, n, seed):
        """-> (generated dataset, extra baselines for this protocol)."""
        cfg = ctx.cfg
        if protocol == "native":
            adapter.reference_staypoints = ctx.staypoints["train"]
            with ctx.timed(_key(adapter), self.capability, n):
                return adapter.generate(ctx.splits["train"], n, seed), {}
        from .rollout import masked_rollout, seed_only
        kw = dict(seed_points=cfg.rollout_seed_points, block=cfg.rollout_block,
                  window=cfg.window_length, noise_m=cfg.rollout_noise_m, max_len=cfg.rollout_max_len)
        with ctx.timed(_key(adapter), RECOVERY, n):
            gen = masked_rollout(adapter, ctx.splits["train"], n, seed, **kw)
        # The seed prefix is real data and already fixes much of a trajectory's statistics.
        # This control keeps the same prefix and generates nothing after it, so a model that
        # does not beat it has added nothing of its own.
        ck = ("gen_seed_only", seed)
        if ck not in ctx.cache:
            ctx.cache[ck] = self._stats(ctx, seed_only(ctx.splits["train"], n, seed,
                                                       seed_points=cfg.rollout_seed_points,
                                                       max_len=cfg.rollout_max_len), generated=True)
        return gen, {"seed_only": ctx.cache[ck]}

    def _stats(self, ctx, ds, generated: bool = False, split: Optional[str] = None):
        # generators emit dwell points (e.g. arrival + departure per visit), so generated data always uses
        # point-based detection per trajectory; real data uses the configured method
        if generated:
            sp = detect_staypoints(ds, ctx.cfg.staypoint_dist_m, ctx.cfg.staypoint_time_s, by_trajectory=True)
        elif split is not None and split in ctx.staypoints:
            sp = ctx.staypoints[split]          # already computed for this split; do not redo it
        else:
            sp = ctx.detect_staypoints(ds)
        return G.trajectory_stats(ds.points, sp, ctx.grid)

    def run(self, adapter, ctx):
        cfg, recs = ctx.cfg, []
        test = ctx.splits["test"]
        n = min(test.points.traj_id.nunique(), cfg.generation_max_trajectories)
        if "gen_real" not in ctx.cache:
            # The real reference distribution. Capped: these are per-trajectory statistics whose
            # distribution is estimated fine from tens of thousands of trajectories, and the
            # uncapped groupby over a multi-million-trajectory panel is what the job dies on.
            real_pts = G.sample_trajectories(test.points, cfg.generation_max_real_trajectories, cfg.split_seed)
            capped = len(real_pts) < len(test.points)
            if capped:
                log.info(f"generation: using {real_pts.traj_id.nunique():,} of "
                         f"{test.points.traj_id.nunique():,} test trajectories for the reference "
                         f"statistics (eval.generation_max_real_trajectories)")
            ctx.cache["gen_real"] = self._stats(ctx, MobilityDataset(real_pts, "real"),
                                                split=None if capped else "test")
            # real-vs-real noise floor: n REAL train trajectories, same sample size as the generator's output
            tr = ctx.splits["train"].points
            ids = np.random.default_rng(cfg.split_seed).choice(tr.traj_id.unique(), n, replace=False)
            ctx.cache["gen_floor"] = self._stats(ctx, MobilityDataset(tr[tr.traj_id.isin(ids)], "floor"))
        real, floor = ctx.cache["gen_real"], ctx.cache["gen_floor"]
        protocols = [p for p in cfg.generation_protocols
                     if (p == "native" and GENERATION in adapter.capabilities)
                     or (p == "rollout" and RECOVERY in adapter.capabilities
                         and GENERATION not in adapter.capabilities)]
        for protocol, seed in [(p, s) for p in protocols for s in cfg.eval_seeds]:
            got = self.try_protocol(ctx, adapter, self.name, protocol,
                                    lambda p=protocol, s=seed: self._generate(adapter, ctx, p, n, s))
            if got is None:
                continue
            gen_ds, extra = got
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
                        baseline="uniform_bbox", baseline_value=lv.get(metric), protocol=protocol,
                        skill=skill_score(metric, mv[metric], lv.get(metric), fv.get(metric)),
                        eval_seed=seed, dataset=ctx.dataset_name))
                    # keyed by PROTOCOL too: the rollout rows need their own baseline rows, and
                    # the seed-only control only exists for that protocol
                    fk = ("gen_ref_emitted", protocol, stat, metric)
                    if fk not in ctx.emitted_baselines and seed == cfg.eval_seeds[0]:
                        ctx.emitted_baselines.add(fk)
                        refs = [("real_noise_floor", fv), ("uniform_bbox", lv)]
                        for nm, st in extra.items():
                            if stat in st:
                                refs.append((nm, G.compare_stat(r, st[stat].to_numpy(float), bins)))
                        for nm, val in refs:
                            if metric in val:
                                recs.append(ResultRecord(model=f"baseline:{nm}", run_tag="-",
                                                         task=f"generation/{stat}", metric=f"{metric}:{stat}",
                                                         value=val[metric], eval_seed=seed, protocol=protocol,
                                                         dataset=ctx.dataset_name))
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
                                         metric="jsd:visited_cells", value=v, protocol=protocol,
                                         baseline="uniform_bbox" if b is not None else None,
                                         baseline_value=b,
                                         skill=None if b is None else skill_score("jsd", v, b, f),
                                         eval_seed=seed, dataset=ctx.dataset_name))
            # memorisation check: distributional metrics cannot tell a copier from a good model
            trp = ctx.splits["train"].points
            nn_kw = {"max_train": cfg.generation_nn_max_train, "max_query": cfg.generation_nn_max_query,
                     "seed": cfg.split_seed}
            if "gen_nn_ref" not in ctx.cache:
                ctx.cache["gen_nn_ref"] = np.percentile(G.nearest_train_distance(test.points, trp, **nn_kw), 5)
            # the generated set is already small, so it is compared in full
            nn = G.nearest_train_distance(gen_ds.points, trp, **{**nn_kw, "max_query": None})
            thr = ctx.cache["gen_nn_ref"]
            for metric, vals in (("copy_rate", (nn < thr).astype(float)), ("nn_train_dist_m", nn)):
                res = evaluate_with_ci(metric, vals, n_boot=min(cfg.n_boot, 200), seed=seed)
                recs.append(ResultRecord(model=adapter.name, run_tag=adapter.run_tag, task="generation/memorisation",
                                         metric=metric, n=len(nn), eval_seed=seed, protocol=protocol,
                                         dataset=ctx.dataset_name, **res))
            rho = G.paired_spearman(real["radius_of_gyration"], gen["radius_of_gyration"])
            if rho is not None:
                recs.append(ResultRecord(model=adapter.name, run_tag=adapter.run_tag, task="generation/radius_of_gyration",
                                         metric="spearman_paired:radius_of_gyration", value=rho, eval_seed=seed,
                                         protocol=protocol, dataset=ctx.dataset_name))
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
class RetrievalTask(Task):
    """Trajectory retrieval from embeddings: MR, MRR, HR@k (similarity) and CR@k (conditions).

    Database: `retrieval_db_size` test windows; queries: the first `retrieval_queries` of them. Every
    query's answer is its own window, so a rank is well defined and there is exactly one hit.

      odd_even      every model with embeddings. Query = the window's odd-indexed points, database =
                    the even-indexed points of every window (the t2vec / TrajCL protocol): the two
                    halves describe the same trip with no point in common.
      cross_modal   models that embed other representations (OmniTraj): query = the window's topology,
                    road segments, regions or a fusion of them; database = the GPS embeddings of the
                    full windows. The paper's Table 2.
      condition     OmniTraj's condition-based retrieval (Table 3): query = the SET of regions (road
                    segments) of a window; CR@k = share of those elements present in the union of the
                    top-k retrieved windows' elements.

    Baselines, scored on the same queries and database: the Hausdorff distance between point sets
    (the strongest heuristic in the OmniTraj paper; for cross_modal only against topology queries,
    which are points) and random ranking.
    """
    name, capability = "retrieval", EMBEDDING

    def applicable(self, adapter):
        return EMBEDDING in adapter.capabilities

    @staticmethod
    def _sample(ctx):
        key = ("retrieval_sample",)
        if key not in ctx.cache:
            w = ctx.windows.get("test")
            if w is None or len(w) < 2:
                ctx.cache[key] = None
            else:
                cfg = ctx.cfg
                rng = np.random.default_rng(cfg.eval_seeds[0])
                n = min(len(w), cfg.retrieval_db_size)
                db = w.take(np.sort(rng.choice(len(w), n, replace=False)))
                # queries drawn at random from the database: its first rows would be a few users only
                q = np.sort(rng.choice(n, min(n, cfg.retrieval_queries), replace=False))
                ctx.cache[key] = (db, q)
        return ctx.cache[key]

    @staticmethod
    def _retag(b: TrajectoryBatch, tag: str) -> TrajectoryBatch:
        return TrajectoryBatch(b.lat, b.lon, b.t, b.user_id, np.array([f"{t}#{tag}" for t in b.traj_id], dtype=object), b.mode)

    @staticmethod
    def _ranks(Q: np.ndarray, D: np.ndarray, truth: np.ndarray, higher_is_closer: bool = True) -> np.ndarray:
        """Rank of the true item (1 = first); ties count against the query."""
        ranks = np.empty(len(Q))
        for s in range(0, len(Q), 256):
            S = Q[s:s + 256] @ D.T if higher_is_closer else -Q[s:s + 256]
            tv = S[np.arange(len(S)), truth[s:s + 256]]
            ranks[s:s + 256] = 1 + (S > tv[:, None]).sum(1) + (S == tv[:, None]).sum(1) - 1
        return ranks

    @staticmethod
    def _unit(E):
        E = np.asarray(E, float)
        return E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)

    def _scores(self, ranks) -> Scores:
        out = {"mean_rank": ranks.astype(float), "mrr": 1.0 / ranks}
        for k in self.cfg_ks:
            out[f"hr@{k}"] = (ranks <= k).astype(float)
        return out, {}

    @staticmethod
    def _xy(ctx, lat, lon):
        from .geo import LocalProjection
        key = ("retrieval_proj",)
        if key not in ctx.cache:
            p = ctx.splits["train"].points
            ctx.cache[key] = LocalProjection(float(p.lat.mean()), float(p.lon.mean()))
        x, y = ctx.cache[key].to_xy(np.asarray(lat, float), np.asarray(lon, float))
        return np.stack([x, y], -1).astype(np.float32)

    @staticmethod
    def _hausdorff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Symmetric Hausdorff between point set a (n, 2) and each window of b (N, L, 2)."""
        d = np.sqrt(((a[None, :, None, :] - b[:, None, :, :]) ** 2).sum(-1))            # (N, n, L)
        return np.maximum(d.min(2).max(1), d.min(1).max(1))

    @staticmethod
    def _hausdorff_ranks(qpts: List[np.ndarray], dpts: np.ndarray, truth: np.ndarray) -> np.ndarray:
        """Rank of the true window by Hausdorff distance, exactly, without the full Q x N x n x L
        tensor: H(q, j) >= max over q's points of the distance to j's bounding box, so only windows
        whose bound is below H(q, true) need the exact distance."""
        lo, hi = dpts.min(1), dpts.max(1)                                                 # (N, 2)
        ranks = np.empty(len(qpts))
        for i, a in enumerate(qpts):
            h_true = RetrievalTask._hausdorff(a, dpts[truth[i]:truth[i] + 1])[0]
            gap = np.maximum(np.maximum(lo[None] - a[:, None], a[:, None] - hi[None]), 0)  # (n, N, 2)
            lb = np.sqrt((gap ** 2).sum(-1)).max(0)                                        # (N,)
            cand = np.where(lb <= h_true)[0]
            cand = cand[cand != truth[i]]
            h = np.concatenate([RetrievalTask._hausdorff(a, dpts[cand[s:s + 512]]) for s in range(0, len(cand), 512)]) \
                if len(cand) else np.array([])
            ranks[i] = 1 + int((h <= h_true).sum())                                        # ties against the query
        return ranks

    def run(self, adapter, ctx):
        got = self._sample(ctx)
        if got is None:
            ctx.skipped.append((_key(adapter), self.name, "fewer than 2 test windows"))
            return []
        db, q = got
        cfg, recs, seed = ctx.cfg, [], cfg_seed(ctx)
        self.cfg_ks = tuple(cfg.retrieval_ks)
        truth = q.copy()
        # every random baseline has its own fixed stream, so it is the same whichever model runs first
        random_ranks = lambda n, k: np.random.default_rng([seed, k]).integers(1, len(db) + 1, size=n).astype(float)
        protos = set(cfg.retrieval_protocols)
        # ---------------------------------------------------------------- odd / even
        if "odd_even" in protos and db.length >= 4:
            ck = ("retrieval_base", "odd_even")
            qb = self._retag(TrajectoryBatch(db.lat[q][:, 1::2], db.lon[q][:, 1::2], db.t[q][:, 1::2],
                                             db.user_id[q], db.traj_id[q]), "odd")
            dbb = self._retag(TrajectoryBatch(db.lat[:, ::2], db.lon[:, ::2], db.t[:, ::2], db.user_id, db.traj_id), "even")
            if ck not in ctx.cache:
                qp = [self._xy(ctx, qb.lat[i], qb.lon[i]) for i in range(len(qb))]
                ctx.cache[ck] = {"hausdorff": self._scores(self._hausdorff_ranks(qp, self._xy(ctx, dbb.lat, dbb.lon), truth)),
                                 "random": self._scores(random_ranks(len(q), 0))}

            def _oe():
                with ctx.timed(_key(adapter), EMBEDDING, len(qb) + len(dbb)):
                    return self._ranks(self._unit(adapter.embed(qb)), self._unit(adapter.embed(dbb)), truth)
            ranks = self.try_protocol(ctx, adapter, "retrieval/odd_even", "embedding", _oe)
            if ranks is not None:
                recs += self.emit(ctx, adapter, "retrieval/odd_even", len(q), self._scores(ranks), ctx.cache[ck],
                                  ["hausdorff", "random"], seed, "embedding")
        if CROSS_MODAL not in adapter.capabilities:
            return recs
        # ---------------------------------------------------------------- cross-modal (OmniTraj)
        qb = db.take(q)
        D = None
        mods = [m for m in cfg.retrieval_modalities if m in adapter.query_modalities()]
        if "cross_modal" in protos and mods:
            D = self._unit(adapter.embed_database(db))
            for m in mods:
                ck = ("retrieval_base", "cross_modal", m)
                if ck not in ctx.cache:
                    base = {"random": self._scores(random_ranks(len(q), 1))}
                    if m == "topology":
                        from .nn.omnitraj_prep import resample, topology
                        qp = []
                        for i in range(len(qb)):
                            tp = topology(resample(qb.lat[i], qb.lon[i]))
                            qp.append(self._xy(ctx, tp[:, 1], tp[:, 0]))
                        base["hausdorff"] = self._scores(self._hausdorff_ranks(qp, self._xy(ctx, db.lat, db.lon), truth))
                    ctx.cache[ck] = base
                ranks = self.try_protocol(ctx, adapter, f"retrieval/cross_modal:{m}", "native",
                                          lambda m=m: self._ranks(self._unit(adapter.embed_query(qb, m)), D, truth))
                if ranks is not None:
                    recs += self.emit(ctx, adapter, f"retrieval/cross_modal:{m}", len(q), self._scores(ranks),
                                      ctx.cache[ck], ["hausdorff", "random"], seed)
        # ---------------------------------------------------------------- condition-based (OmniTraj)
        if "condition" in protos:
            D = self._unit(adapter.embed_database(db)) if D is None else D
            for m in ("region", "road"):
                if m not in adapter.query_modalities():
                    continue

                def _cr(m=m):
                    elems = adapter.elements(db, m)
                    # Eq. 14 divides by the query's size: a window with no matched segment is no query
                    qi = np.array([i for i in q if elems[i]], int)
                    if len(qi) < len(q):
                        ctx.skipped.append((_key(adapter), f"retrieval/condition:{m}",
                                            f"{len(q) - len(qi)} of {len(q)} queries have no {m} and were left out"))
                    if not len(qi):
                        return None
                    Q = self._unit(adapter.embed_query(db.take(qi), m))
                    top = np.argsort(-(Q @ D.T), axis=1, kind="stable")[:, :5]
                    r = np.random.default_rng([seed, 2 if m == "region" else 3])
                    rtop = np.stack([r.permutation(len(db))[:5] for _ in range(len(qi))])
                    out = {}
                    for name, T in (("model", top), ("random", rtop)):
                        cr = {1: [], 5: []}
                        for row, i in enumerate(qi):
                            want = elems[i]
                            for k in (1, 5):
                                got = set().union(*[elems[j] for j in T[row, :k]])
                                cr[k].append(len(want & got) / len(want))
                        out[name] = ({"cr@1": np.asarray(cr[1]), "cr@5": np.asarray(cr[5])}, {})
                    return out, len(qi)
                res = self.try_protocol(ctx, adapter, f"retrieval/condition:{m}", "native", _cr)
                if res is not None:
                    res, nq = res
                    recs += self.emit(ctx, adapter, f"retrieval/condition:{m}", nq, res["model"],
                                      {"random": res["random"]}, ["random"], seed)
        return recs


def cfg_seed(ctx) -> int:
    return int(ctx.cfg.eval_seeds[0])


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


TASK_NAMES = ("recovery", "next_location", "continuous", "mode_classification", "generation",
              "user_identification", "anomaly_detection", "retrieval", "efficiency")
VISIT_TASKS = {"next_location", "continuous", "generation"}      # need staypoints / visit sequences


def default_tasks(cfg) -> List[Task]:
    tasks = ([RecoveryTask(), NextLocationTask()] + [ContinuousValueTask(t) for t in cfg.continuous_targets]
             + [ModeClassificationTask(), GenerationTask(), UserIdentificationTask(),
                AnomalyDetectionTask(), RetrievalTask(), EfficiencyTask()])
    if cfg.tasks is None:
        return tasks
    unknown = set(cfg.tasks) - set(TASK_NAMES)
    if unknown:
        raise ValueError(f"unknown tasks {sorted(unknown)}; known: {', '.join(TASK_NAMES)}")
    return [t for t in tasks if t.name.split("/")[0] in set(cfg.tasks)]
