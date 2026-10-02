"""Generation by masked rollout: turning a reconstruction model into a generator.

UniTraj, TransferTraj and CLIP-Mobility have no `generate` method, but they can all fill in
hidden positions. Hiding *the future* rather than a random subset turns that into generation:
seed the model with the first few real points of a trajectory, ask it to fill the rest, feed
its own output back, and continue. The same procedure runs for every model, so what is being
compared is the models and not three separately hand-written decoders.

Two things have to be right or the resulting numbers are worse than useless.

**Comparability.** The generation metrics compare distributions of per-trajectory statistics -
radius of gyration, jump length - against real trajectories. Generating fixed-length windows
would compare 64-point fragments with whole trips, and every model would look wrong for a
reason that has nothing to do with the model. The rollout therefore reproduces each seed
trajectory's own length and its own timestamps, sliding the model's fixed window forward as
needed, so a generated trajectory is the same kind of object as a real one.

**Attribution.** The seed prefix is real data, and a prefix alone already fixes much of a
trajectory's statistics. `seed_only` freezes after the seed and generates nothing; it is
reported as a baseline for exactly this protocol, and a model that does not beat it has added
nothing to the seed it was given.

A caveat that belongs in any write-up: a masked reconstruction model trained with a squared
error predicts a conditional MEAN, not a sample. Its rollouts are therefore smoother and
shorter than real trajectories, and the distributional metrics will show that as a penalty.
That is a property of decoding a regression model, not evidence about the representation.
`noise_m` injects calibrated noise if you want to correct the marginal spread; it is off by
default because the right scale is a modelling choice, not something the pipeline should pick.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .data import MobilityDataset, TrajectoryBatch

log = logging.getLogger("mobeval.rollout")

DEFAULT_SEED_POINTS = 4
DEFAULT_BLOCK = 8


def _pick(reference: MobilityDataset, n: int, seed: int, min_points: int):
    """n real trajectories, long enough to seed and then say something."""
    p = reference.points
    sizes = p.groupby("traj_id").size()
    ok = sizes[sizes >= min_points].index.to_numpy()
    if not len(ok):
        raise ValueError(f"no trajectory in the reference split has {min_points} points; "
                         f"lower rollout_seed_points or the block size")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(ok, size=n, replace=len(ok) < n)
    sub = p[p.traj_id.isin(set(chosen.tolist()))]
    return [g for _, g in sub.groupby("traj_id", sort=False)], chosen


def _stack(groups, max_len: Optional[int]):
    """(n, L) lat/lon/t with the real time axis, plus each trajectory's true length.

    Shorter trajectories are padded at the end; the padding is filled by the model like any
    other hidden position and then discarded, so it never reaches the metrics.
    """
    lens = [min(len(g), max_len or len(g)) for g in groups]
    L = max(lens)
    lat = np.zeros((len(groups), L)); lon = np.zeros_like(lat); t = np.zeros_like(lat)
    for i, (g, k) in enumerate(zip(groups, lens)):
        lat[i, :k] = g.lat.to_numpy()[:k]
        lon[i, :k] = g.lon.to_numpy()[:k]
        t[i, :k] = g.t.to_numpy()[:k]
        if k < L:                                   # keep time strictly increasing in the padding
            step = np.diff(t[i, :k]).mean() if k > 1 else 60.0
            t[i, k:] = t[i, k - 1] + step * np.arange(1, L - k + 1)
            lat[i, k:], lon[i, k:] = lat[i, k - 1], lon[i, k - 1]
    return lat, lon, t, np.asarray(lens)


def _as_dataset(lat, lon, t, lens, users, name: str) -> MobilityDataset:
    rows = []
    for i, k in enumerate(lens):
        rows.append(pd.DataFrame({"user_id": users[i], "traj_id": f"{name}{i}",
                                  "t": t[i, :k], "lat": lat[i, :k], "lon": lon[i, :k]}))
    return MobilityDataset(pd.concat(rows, ignore_index=True), name)


def masked_rollout(adapter, reference: MobilityDataset, n_trajectories: int, seed: int = 0,
                   seed_points: int = DEFAULT_SEED_POINTS, block: int = DEFAULT_BLOCK,
                   window: Optional[int] = None, noise_m: Optional[float] = None,
                   max_len: Optional[int] = 512) -> MobilityDataset:
    """Generate by repeatedly asking a reconstruction model to fill in the future.

    `window` bounds what the model sees at once (UniTraj refuses anything beyond its fixed
    trajectory length); the window slides forward so trajectories longer than it can still be
    produced in full.
    """
    groups, _ = _pick(reference, n_trajectories, seed, min_points=seed_points + 1)
    lat, lon, t, lens = _stack(groups, max_len)
    users = [g.user_id.iloc[0] for g in groups]
    n, L = lat.shape
    W = min(window or L, L)
    if seed_points >= W:
        raise ValueError(f"rollout_seed_points={seed_points} leaves no room in a window of {W}")
    rng = np.random.default_rng(seed)

    known = seed_points                             # positions [0, known) are committed
    traj_id = np.array([f"roll{i}" for i in range(n)], dtype=object)
    while known < L:
        stop = min(known + block, L)
        start = max(0, stop - W)                    # slide so the window ends at `stop`
        sl = slice(start, stop)
        sub = TrajectoryBatch(lat[:, sl].copy(), lon[:, sl].copy(), t[:, sl],
                              np.asarray(users), traj_id, None)
        mask = np.zeros(sub.lat.shape, bool)
        mask[:, known - start:] = True              # everything not yet committed is hidden
        sub.lat[mask] = np.nan
        sub.lon[mask] = np.nan
        plat, plon = adapter.reconstruct(sub, mask)
        new = slice(known - start, stop - start)
        lat[:, known:stop] = np.asarray(plat)[:, new]
        lon[:, known:stop] = np.asarray(plon)[:, new]
        if noise_m:                                 # optional: restore marginal spread
            lat[:, known:stop] += rng.normal(0, noise_m / 111_195.0, (n, stop - known))
            lon[:, known:stop] += rng.normal(0, noise_m / 70_000.0, (n, stop - known))
        known = stop
    lat, lon = _bound(lat, lon, reference, seed_points)
    return _as_dataset(lat, lon, t, lens, users, "roll")


def _bound(lat, lon, reference: MobilityDataset, seed_points: int):
    """Keep generated coordinates on the planet, and say when they had to be pulled back.

    Rolling forward asks a model to EXTRAPOLATE - every hidden position is after the last
    visible one - and some reconstruction models diverge when used that way; a spline fitted to
    a short prefix will happily run off the map. Non-finite values are replaced and wild ones
    are clipped to a box twice the size of the reference data's own extent, which is far outside
    anything real and so still shows up in the metrics as a bad trajectory. The count is logged,
    because "the decoder diverged" and "the model generates poorly" are different findings and
    a reader needs to know which one a number reflects.
    """
    p = reference.points
    lo_lat, hi_lat = float(p.lat.min()), float(p.lat.max())
    lo_lon, hi_lon = float(p.lon.min()), float(p.lon.max())
    pad_lat, pad_lon = max(hi_lat - lo_lat, 1e-3), max(hi_lon - lo_lon, 1e-3)
    box = (max(lo_lat - pad_lat, -89.9), min(hi_lat + pad_lat, 89.9),
           max(lo_lon - pad_lon, -179.9), min(hi_lon + pad_lon, 179.9))

    bad = ~np.isfinite(lat) | ~np.isfinite(lon)
    if bad.any():
        lat, lon = lat.copy(), lon.copy()
        for i, j in zip(*np.nonzero(bad)):           # hold the last good position
            k = max(j - 1, 0)
            lat[i, j], lon[i, j] = lat[i, k], lon[i, k]
    out = (lat < box[0]) | (lat > box[1]) | (lon < box[2]) | (lon > box[3])
    if out.any():
        rows = int(out.any(1).sum())
        log.warning(f"rollout: {out.sum():,} generated positions in {rows} of {len(lat)} trajectories fell "
                    f"outside twice the reference extent and were clipped. Rolling forward is pure "
                    f"extrapolation, and a model that diverges under it will score badly for that reason "
                    f"rather than for the quality of what it generates.")
        lat = np.clip(lat, box[0], box[1])
        lon = np.clip(lon, box[2], box[3])
    if bad.any():
        log.warning(f"rollout: {int(bad.sum()):,} non-finite positions were replaced by the previous one")
    return lat, lon


def seed_only(reference: MobilityDataset, n_trajectories: int, seed: int = 0,
              seed_points: int = DEFAULT_SEED_POINTS, max_len: Optional[int] = 512) -> MobilityDataset:
    """The control for the rollout protocol: keep the real seed prefix, then stand still.

    Every rollout is handed the same real prefix, and a prefix already determines a good deal
    of a trajectory's statistics. This generates nothing at all beyond it, so a model whose
    scores match this one has contributed nothing of its own.
    """
    groups, _ = _pick(reference, n_trajectories, seed, min_points=seed_points + 1)
    lat, lon, t, lens = _stack(groups, max_len)
    users = [g.user_id.iloc[0] for g in groups]
    lat[:, seed_points:] = lat[:, seed_points - 1][:, None]
    lon[:, seed_points:] = lon[:, seed_points - 1][:, None]
    return _as_dataset(lat, lon, t, lens, users, "seed")
