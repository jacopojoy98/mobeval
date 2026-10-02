"""Generative-quality metrics on mobility statistics (rg, jump length, stay
duration, daily locations, spatial visit distribution). Bins are fixed from
REAL data so every model is histogrammed identically."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wasserstein_distance

from ..geo import haversine_m, radius_of_gyration_m

LOG_STATS = {"radius_of_gyration", "jump_length", "stay_duration"}


def trajectory_stats(points: pd.DataFrame, staypoints: pd.DataFrame, grid) -> Dict[str, pd.Series]:
    """Per-unit statistics. Returns Series indexed by user (for pairing) where sensible."""
    rg = points.groupby(["user_id", "traj_id"]).apply(
        lambda g: radius_of_gyration_m(g.lat.to_numpy(), g.lon.to_numpy()), include_groups=False)
    stats = {"radius_of_gyration": rg.droplevel(1)}
    if len(staypoints):
        sp = staypoints.sort_values(["user_id", "t_arrive"])
        same = sp.user_id.to_numpy()[1:] == sp.user_id.to_numpy()[:-1]
        jl = haversine_m(sp.lat.to_numpy()[:-1], sp.lon.to_numpy()[:-1], sp.lat.to_numpy()[1:], sp.lon.to_numpy()[1:])
        stats["jump_length"] = pd.Series(jl[same], index=sp.user_id.to_numpy()[1:][same])
        stats["stay_duration"] = pd.Series((sp.t_leave - sp.t_arrive).to_numpy() / 60.0, index=sp.user_id.to_numpy())
        day = (sp.t_arrive // 86400).astype(int)
        cells = grid.cell_of(sp.lat.to_numpy(), sp.lon.to_numpy())
        dl = pd.DataFrame({"u": sp.user_id.to_numpy(), "d": day.to_numpy(), "c": cells}).groupby(["u", "d"]).c.nunique()
        stats["daily_locations"] = pd.Series(dl.to_numpy(), index=dl.index.get_level_values(0))
        stats["_cells"] = pd.Series(cells)
    return stats


def make_bins(real: np.ndarray, name: str, n_bins: int = 30) -> np.ndarray:
    real = real[np.isfinite(real)]
    if name == "daily_locations":
        return np.arange(0.5, max(real.max(), 1) + 1.5 + 5)            # integer bins
    if name in LOG_STATS:
        lo = max(np.percentile(real, 0.5), 1e-3)
        hi = np.percentile(real, 99.5)
        edges = np.logspace(np.log10(lo), np.log10(max(hi, lo * 10)), n_bins + 1)
    else:
        edges = np.linspace(np.percentile(real, 0.5), np.percentile(real, 99.5), n_bins + 1)
    return np.concatenate([[-np.inf], edges, [np.inf]])                  # overflow bins keep mass


def jsd_bits(p_counts: np.ndarray, q_counts: np.ndarray, eps: float = 1e-12) -> float:
    p = p_counts / max(p_counts.sum(), eps); q = q_counts / max(q_counts.sum(), eps)
    m = 0.5 * (p + q)
    kl = lambda a, b: np.sum(np.where(a > 0, a * np.log2((a + eps) / (b + eps)), 0.0))
    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def compare_stat(real: np.ndarray, gen: np.ndarray, bins: np.ndarray) -> Dict[str, float]:
    real, gen = real[np.isfinite(real)], gen[np.isfinite(gen)]
    if len(gen) == 0:
        return {"jsd": 1.0, "w1": np.nan}
    return {"jsd": jsd_bits(np.histogram(real, bins)[0], np.histogram(gen, bins)[0]),
            "w1": float(wasserstein_distance(real, gen))}


def cell_jsd(real_cells: np.ndarray, gen_cells: np.ndarray, n_cells: int) -> float:
    return jsd_bits(np.bincount(real_cells, minlength=n_cells).astype(float),
                    np.bincount(gen_cells, minlength=n_cells).astype(float))


def paired_spearman(real: pd.Series, gen: pd.Series) -> Optional[float]:
    """Rank correlation of per-user MEDIAN statistic, users present in both."""
    r, g = real.groupby(level=0).median(), gen.groupby(level=0).median()
    common = r.index.intersection(g.index)
    if len(common) < 5:
        return None
    return float(spearmanr(r[common], g[common]).statistic)


def sample_trajectories(points: pd.DataFrame, max_n: Optional[int], seed: int = 0) -> pd.DataFrame:
    """Keep at most `max_n` whole trajectories, chosen with a fixed seed.

    Subsampling has to happen BEFORE any per-trajectory work. Resampling every trajectory in a
    40-million-point panel and then keeping 3,000 of them costs gigabytes and is what made the
    generation task get OOM-killed on a real dataset.
    """
    if max_n is None:
        return points
    ids = points.traj_id.unique()
    if len(ids) <= max_n:
        return points
    keep = np.random.default_rng(seed).choice(ids, max_n, replace=False)
    return points[points.traj_id.isin(keep)]


def _resample(points: pd.DataFrame, k: int = 16) -> np.ndarray:
    """(T, k, 2) lat/lon of every trajectory resampled to k points by arc index."""
    out = []
    for _, g in points.groupby("traj_id", sort=False):
        la, lo = g.lat.to_numpy(), g.lon.to_numpy()
        x = np.linspace(0, len(la) - 1, k)
        out.append(np.column_stack([np.interp(x, np.arange(len(la)), la), np.interp(x, np.arange(len(lo)), lo)]))
    return np.stack(out) if out else np.zeros((0, k, 2))


# The pairwise comparison below materialises (n_query, block, k) doubles. Bounding the element
# count rather than fixing `block` keeps that array the same size whatever the query count is -
# at a fixed block of 256 it was 9.8 GB for 300k query trajectories.
_MAX_PAIRWISE_ELEMENTS = 8_000_000


def nearest_train_distance(query: pd.DataFrame, train: pd.DataFrame, k: int = 16, max_train: int = 3000,
                           max_query: Optional[int] = 2000, seed: int = 0) -> np.ndarray:
    """Per query trajectory: mean point-wise haversine distance (m) to its closest
    training trajectory. Low values for GENERATED data indicate copying.

    Both sides are subsampled to whole trajectories first. The result is a distance
    distribution, and the statistics taken from it (a 5th percentile, a copy rate) are
    estimated just as well from a few thousand trajectories as from millions.
    """
    Q = _resample(sample_trajectories(query, max_query, seed), k)
    T = _resample(sample_trajectories(train, max_train, seed), k)
    best = np.full(len(Q), np.inf)
    if not len(Q) or not len(T):
        return best
    block = max(1, min(len(T), _MAX_PAIRWISE_ELEMENTS // max(len(Q) * k, 1)))
    for i in range(0, len(T), block):
        t = T[i:i + block]
        d = haversine_m(Q[:, None, :, 0], Q[:, None, :, 1], t[None, :, :, 0], t[None, :, :, 1]).mean(-1)
        best = np.minimum(best, d.min(1))
    return best
