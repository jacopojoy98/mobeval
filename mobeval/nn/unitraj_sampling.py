"""UniTraj's own training-sample construction (utils/dataset.py of github.com/Yasoz/UniTraj),
reproduced so the model can be retrained with the original methodology.

Per trajectory:
  1. ATR resampling. With probability 0.3, and only for trajectories of >= 360 points,
     interval-consistent resampling: points are averaged in bins of k seconds, with k drawn
     from 8-15 (L > 540), 6-10 (360 < L <= 540) or 3-6 (L = 360). Otherwise a random subset of
     int(L * r) points is kept, with r falling logarithmically from 1 (L <= 36) to 0.35 (L >= 600).
  2. One masking strategy (ratio 0.5 of min(L, 200) points): random 70%, RDP key points 15%,
     a 5-14-point block plus random fill 5%, the last 3-7 points plus random fill 10%.
  3. Offsets from the FIRST point (masked or not), z-normalised, padded/truncated to 200.
  4. The mask is topped up at random, over real AND padding positions, to exactly
     int(200 * ratio) hidden tokens, so every sample shows the encoder 100 tokens - some of
     them padding zeros when the trajectory is shorter than 200.
  5. Loss: mean over (batch, 2, 200) of the squared error at hidden real points, divided by 0.5.

Two faithful-but-necessary changes, both documented in MODELS.md:
  * pandas' `resample(...).mean()` emits empty bins as NaN rows (harmless on WorldTrace's 1 s data,
    fatal on sparser data); empty bins are dropped.
  * numpy/random global state is replaced by an explicit generator, for reproducibility.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

MIN_POINTS, MAX_POINTS, MIN_SAMPLING_RATIO = 36, 600, 0.35
ORIGINAL_MIX = {"random": 0.7, "rdp": 0.15, "block": 0.05, "last_n": 0.10}


def log_sampling_ratio(length: int) -> float:
    if length <= MIN_POINTS:
        return 1.0
    if length >= MAX_POINTS:
        return MIN_SAMPLING_RATIO
    r = 1.0 - math.log(length - MIN_POINTS + 1) / math.log(MAX_POINTS - MIN_POINTS + 1) * (1.0 - MIN_SAMPLING_RATIO)
    return max(r, MIN_SAMPLING_RATIO)


def atr_resample(lat, lon, t, rng: np.random.Generator, p_interval: float = 0.3):
    """-> (lat, lon, t, intervals) after UniTraj's adaptive trajectory resampling."""
    lat, lon, t = np.asarray(lat, float), np.asarray(lon, float), np.asarray(t, float)
    L = len(t)
    if rng.random() < p_interval and L >= 360:
        # python's random.randint is inclusive at both ends
        k = int(rng.integers(8, 16)) if L > 540 else int(rng.integers(6, 11)) if L > 360 else int(rng.integers(3, 7))
        day0 = math.floor(t[0] / 86400.0) * 86400.0            # pandas' default origin: start of the day
        b = np.floor((t - day0) / k).astype(np.int64)
        u, inv = np.unique(b, return_inverse=True)             # non-empty bins only (see module doc)
        cnt = np.bincount(inv)
        lat = np.bincount(inv, lat) / cnt
        lon = np.bincount(inv, lon) / cnt
        t = day0 + u * float(k)
    else:
        n = int(L * log_sampling_ratio(L))
        keep = np.sort(rng.choice(L, size=n, replace=False))
        lat, lon, t = lat[keep], lon[keep], t[keep]
    iv = np.zeros(len(t))
    iv[1:] = np.diff(t)
    return lat, lon, t, iv


def rdp_keypoints(xy: np.ndarray, epsilon: float) -> np.ndarray:
    """Ramer-Douglas-Peucker as in the `rdp` package (return_mask=True): True = kept point.
    Distance is to the infinite line through the segment ends (to the start point if they coincide)."""
    n = len(xy)
    keep = np.zeros(n, bool)
    if n == 0:
        return keep
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        s, e = stack.pop()
        if e <= s + 1:
            continue
        a, b = xy[s], xy[e]
        pts = xy[s + 1:e]
        d = b - a
        norm = math.hypot(d[0], d[1])
        if norm == 0:
            dist = np.hypot(pts[:, 0] - a[0], pts[:, 1] - a[1])
        else:
            dist = np.abs(d[0] * (a[1] - pts[:, 1]) - d[1] * (a[0] - pts[:, 0])) / norm
        i = int(np.argmax(dist))
        if dist[i] > epsilon:
            m = s + 1 + i
            keep[m] = True
            stack += [(s, m), (m, e)]
    return keep


def _fill(mask: np.ndarray, target: int, rng, allowed: Optional[np.ndarray] = None):
    free = np.where(~mask & (True if allowed is None else allowed))[0]
    extra = max(0, min(target - int(mask.sum()), len(free)))
    if extra:
        mask[rng.choice(free, size=extra, replace=False)] = True
    return mask


def strategy_mask(lon, lat, strategy: str, ratio: float, rng: np.random.Generator,
                  mask_endpoints: bool = True, rdp_epsilon: float = 1e-4) -> np.ndarray:
    """One of UniTraj's four masks over len(lon) points (already truncated to max_len).
    mask_endpoints=False keeps the first and last point visible (mobeval's evaluation convention)."""
    L = len(lon)
    num = int(L * ratio)
    allowed = np.ones(L, bool)
    if not mask_endpoints:
        allowed[[0, -1]] = False
        num = min(num, int(allowed.sum()))
    mask = np.zeros(L, bool)
    if strategy == "random":
        mask[rng.choice(np.where(allowed)[0], size=num, replace=False)] = True
    elif strategy == "rdp":
        # float32, as the original runs rdp on the float32 trajectory tensor: in float64 the key-point
        # sets differ for most WorldTrace trajectories (distances near epsilon = 1e-4 degrees)
        key = rdp_keypoints(np.column_stack([lon, lat]).astype(np.float32), rdp_epsilon)
        key[0] = key[-1] = False
        idx = np.where(key & allowed)[0]
        if len(idx) > num:
            idx = rng.choice(idx, size=num, replace=False)
        mask[idx] = True
        _fill(mask, num, rng, allowed)
    elif strategy == "block":
        bs = min(int(rng.integers(5, 15)), max(1, int(allowed.sum())))
        lo, hi = (0, L) if mask_endpoints else (1, L - 1)
        s = int(rng.integers(lo, max(lo, hi - bs) + 1))
        mask[s:s + bs] = True
        mask &= allowed
        _fill(mask, num, rng, allowed)
    elif strategy == "last_n":
        n = int(rng.integers(3, 8))
        end = L if mask_endpoints else L - 1
        mask[max(0, end - n):end] = True
        _fill(mask, num, rng, allowed)
    else:
        raise ValueError(f"unknown UniTraj mask strategy '{strategy}'")
    return mask


def build_batch(samples: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]], rng: np.random.Generator,
                max_len: int, norm: Dict, ratio: float = 0.5, mix: Optional[Dict[str, float]] = None,
                resample: str = "atr", mask_endpoints: bool = True, fixed_hidden_count: bool = True,
                offset_from: str = "first", rdp_epsilon: float = 1e-4):
    """-> x (B,2,max_len) float32 normalised offsets (lon, lat) with hidden positions zeroed,
          x_true (B,2,max_len), intervals (B,max_len), hidden (B,max_len) bool, target (B,max_len) bool.

    fixed_hidden_count=True reproduces the original top-up to exactly int(max_len*ratio) hidden tokens over
    real and padding positions (padding not drawn stays VISIBLE); False hides every padding position
    and the masked points only, which gives rows unequal visible counts, so the rows' counts are then
    equalised by hiding extra random real points (the encoder needs one count per batch)."""
    mix = mix or ORIGINAL_MIX
    names, probs = list(mix), np.asarray(list(mix.values()), float)
    probs = probs / probs.sum()
    B = len(samples)
    mean, std = np.asarray(norm["mean"], float), np.asarray(norm["std"], float)
    x_true = np.zeros((B, 2, max_len), np.float32)
    iv = np.zeros((B, max_len), np.float32)
    hid = np.zeros((B, max_len), bool)
    real = np.zeros((B, max_len), bool)
    for i, (lat, lon, t) in enumerate(samples):
        if resample == "atr":
            lat, lon, t, ivs = atr_resample(lat, lon, t, rng)
        else:
            ivs = np.zeros(len(t))
            ivs[1:] = np.diff(np.asarray(t, float))
        L = min(len(lat), max_len)
        lat, lon, ivs = np.asarray(lat)[:L], np.asarray(lon)[:L], ivs[:L]
        strategy = names[int(rng.choice(len(names), p=probs))]
        m = strategy_mask(lon, lat, strategy, ratio, rng, mask_endpoints, rdp_epsilon)
        if offset_from == "first":
            j = 0
        else:                                   # first visible point
            vis = np.where(~m)[0]
            j = int(vis[0]) if len(vis) else 0
        x_true[i, 0, :L] = ((lon - lon[j]) - mean[0]) / std[0]
        x_true[i, 1, :L] = ((lat - lat[j]) - mean[1]) / std[1]
        iv[i, :L] = ivs
        real[i, :L] = True
        hid[i, :L] = m
        if fixed_hidden_count:
            # over real and padding positions; a first-visible anchor must stay visible, or the offsets
            # would be taken from a hidden point
            allowed = np.ones(max_len, bool)
            if offset_from != "first":
                allowed[j] = False
            _fill(hid[i], int(max_len * ratio), rng, allowed)
        else:
            hid[i, L:] = True
    if not fixed_hidden_count:
        n_vis = (~hid).sum(1)
        target_vis = int(n_vis.min())
        for i in np.where(n_vis > target_vis)[0]:
            vis = np.where(~hid[i])[0]
            if offset_from != "first":
                vis = vis[1:]                   # never hide the anchor point
            hid[i, rng.choice(vis, size=int(n_vis[i] - target_vis), replace=False)] = True
    target = hid & real
    x = np.where(hid[:, None, :], 0.0, x_true).astype(np.float32)
    return x, x_true, iv, hid, target


def trajectories(ds, min_points: int = MIN_POINTS, max_points: Optional[int] = None,
                 max_gap_s: Optional[float] = None) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Whole trajectories (one per traj_id, split at gaps > max_gap_s) as (lat, lon, t) arrays."""
    out = []
    for _, g in ds.points.sort_values(["traj_id", "t"], kind="stable").groupby("traj_id", sort=False):
        la, lo, tt = g.lat.to_numpy(), g.lon.to_numpy(), g.t.to_numpy().astype(float)
        cuts = np.where(np.diff(tt) > max_gap_s)[0] + 1 if max_gap_s else []
        for seg in np.split(np.arange(len(tt)), cuts):
            if len(seg) >= min_points and (max_points is None or len(seg) <= max_points):
                out.append((la[seg], lo[seg], tt[seg]))
    return out
