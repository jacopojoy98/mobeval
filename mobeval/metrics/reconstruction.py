"""Recovery metrics. All return PER-TRAJECTORY arrays so the pipeline can
bootstrap over trajectories and pair them with baselines on identical samples."""
from __future__ import annotations

from typing import Dict

import numpy as np

from ..geo import haversine_m


def _masked_rows(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean of `values` over masked positions per row."""
    cnt = mask.sum(1)
    return np.where(cnt > 0, (values * mask).sum(1) / np.maximum(cnt, 1), np.nan)


def block_ends(mask: np.ndarray) -> np.ndarray:
    """True at the last position of every contiguous masked block."""
    nxt = np.zeros_like(mask)
    nxt[:, :-1] = mask[:, 1:]
    return mask & ~nxt


def dtw_masked(pred_lat, pred_lon, true_lat, true_lon, mask) -> np.ndarray:
    """DTW between predicted and true masked sub-sequences (haversine cost),
    normalised by the number of aligned pairs on the optimal path.
    Rows must have the same number of masked points (pipeline masks do)."""
    k = mask.sum(1)
    if not np.all(k == k[0]):
        raise ValueError("dtw_masked expects an equal number of masked points per row")
    k = int(k[0])
    n = mask.shape[0]
    sel = lambda a: a[mask].reshape(n, k)
    pl, po, tl, to = sel(pred_lat), sel(pred_lon), sel(true_lat), sel(true_lon)
    cost = haversine_m(pl[:, :, None], po[:, :, None], tl[:, None, :], to[:, None, :])  # (n,k,k)
    D = np.full((n, k + 1, k + 1), np.inf)
    L = np.zeros((n, k + 1, k + 1))
    D[:, 0, 0] = 0.0
    for i in range(1, k + 1):
        for j in range(1, k + 1):
            cand = np.stack([D[:, i - 1, j - 1], D[:, i - 1, j], D[:, i, j - 1]], 1)
            lens = np.stack([L[:, i - 1, j - 1], L[:, i - 1, j], L[:, i, j - 1]], 1)
            arg = cand.argmin(1)
            D[:, i, j] = cost[:, i - 1, j - 1] + cand[np.arange(n), arg]
            L[:, i, j] = lens[np.arange(n), arg] + 1
    return D[:, k, k] / L[:, k, k]


def recovery_metrics(pred_lat, pred_lon, true_lat, true_lon, mask, grid=None,
                     with_dtw: bool = True) -> Dict[str, np.ndarray]:
    pred_lat, pred_lon = np.asarray(pred_lat, float), np.asarray(pred_lon, float)
    if pred_lat.shape != true_lat.shape:
        raise ValueError(f"prediction shape {pred_lat.shape} != target shape {true_lat.shape}")
    if not np.isfinite(pred_lat[mask]).all() or not np.isfinite(pred_lon[mask]).all():
        raise ValueError("non-finite coordinates at masked positions")
    d = haversine_m(pred_lat, pred_lon, true_lat, true_lon)
    ade = _masked_rows(d, mask)
    out = {
        "ade_m": ade,
        "median_ade_m": ade,
        "p90_ade_m": ade,
        "rmse_m": _masked_rows(d ** 2, mask),          # per-row MSE; aggregate='rmse' takes sqrt(mean)
        "fde_m": _masked_rows(d, block_ends(mask)),
        "acc_100m": _masked_rows((d <= 100).astype(float), mask),
        "acc_500m": _masked_rows((d <= 500).astype(float), mask),
    }
    if grid is not None:
        hit = grid.cell_of(pred_lat, pred_lon) == grid.cell_of(true_lat, true_lon)
        out["grid_acc"] = _masked_rows(hit.astype(float), mask)
    if with_dtw:
        out["dtw_m"] = dtw_masked(pred_lat, pred_lon, true_lat, true_lon, mask)
    return out
