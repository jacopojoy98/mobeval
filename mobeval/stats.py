"""Uncertainty: bootstrap CIs over evaluation samples, paired with baselines."""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np

from .metrics.registry import get_spec


def aggregate(values: np.ndarray, how: str) -> float:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan
    if how == "mean":
        return float(v.mean())
    if how == "rmse":
        return float(np.sqrt(v.mean()))
    if how == "median":
        return float(np.median(v))
    if how == "p90":
        return float(np.percentile(v, 90))
    raise ValueError(how)


def skill_score(metric: str, value: float, baseline: float, floor: Optional[float] = None) -> Optional[float]:
    """Unit-free comparison against a baseline evaluated on the SAME samples.

    ratio      1 - v/b             errors >= 0 (m, min, CRPS): fraction of baseline error removed
    bounded    (v-b)/(1-b)         scores in [0,1]: fraction of the gap to perfect closed
    difference b - v               log scores (nats/sample): already a log-likelihood ratio
    floor      (b-v)/(b-f)         divergences: 1 = as close as real-vs-real noise floor
    """
    spec = get_spec(metric)
    if baseline is None or not np.isfinite(baseline) or not np.isfinite(value):
        return None
    if spec.skill == "ratio":
        return None if baseline <= 0 else 1 - value / baseline
    if spec.skill == "bounded":
        return None if baseline >= 1 else (value - baseline) / (1 - baseline)
    if spec.skill == "difference":
        return baseline - value
    if spec.skill == "floor":
        if floor is None or baseline - floor <= 1e-12:
            return None
        return (baseline - value) / (baseline - floor)
    return None


def bootstrap(statistic: Callable[[np.ndarray], float], n: int, n_boot: int = 1000,
              alpha: float = 0.05, seed: int = 0) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = np.array([statistic(rng.integers(0, n, n)) for _ in range(n_boot)], float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (np.nan, np.nan)
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))


def evaluate_with_ci(metric: str, model_vals, baseline_vals=None, set_fn=None, baseline_set_fn=None,
                     n_boot: int = 1000, seed: int = 0) -> Dict[str, Optional[float]]:
    """Point estimate + CI for the metric, and (paired) CI for its skill score."""
    spec = get_spec(metric)
    if set_fn is None:
        mv = np.asarray(model_vals, float)
        n = len(mv)
        stat = lambda i: aggregate(mv[i], spec.aggregate)
    else:
        n = int(model_vals)          # when set_fn is given, model_vals is the sample count
        stat = set_fn
    idx_all = np.arange(n)
    res = {"value": stat(idx_all)}
    res["ci_low"], res["ci_high"] = bootstrap(stat, n, n_boot, seed=seed)
    res["baseline_value"] = res["skill"] = res["skill_ci_low"] = res["skill_ci_high"] = None
    if baseline_vals is not None or baseline_set_fn is not None:
        if baseline_set_fn is None:
            bv = np.asarray(baseline_vals, float)
            bstat = lambda i: aggregate(bv[i], spec.aggregate)
        else:
            bstat = baseline_set_fn
        res["baseline_value"] = bstat(idx_all)
        res["skill"] = skill_score(metric, res["value"], res["baseline_value"])
        if res["skill"] is not None:
            def sk(i):
                s = skill_score(metric, stat(i), bstat(i))
                return np.nan if s is None else s
            res["skill_ci_low"], res["skill_ci_high"] = bootstrap(sk, n, n_boot, seed=seed + 1)
    return res
