"""OmniTraj's input construction, reconstructed from the paper and its sample data.

The authors' preprocessing code is not public. The repository ships only the network, the training
script and 1,000 already-preprocessed Chengdu trips (data/trajectory.pkl). Each step below is
either stated in the paper or recovered from that sample, and says which and how well:

  resample     Paper Sec. 3.2.1: "up/down-sample to a fixed length L using cubic spline
               interpolation", L = 200 (Table 4). A quarter of the sample's steps are EXACTLY zero
               (the vehicle standing still), which rules out arc-length parametrisation and also rules
               out scipy's CubicSpline, whose overshoot leaves tiny non-zero steps around every stop.
               A piecewise-cubic Hermite interpolant (PCHIP), which keeps flat stretches flat, fits
               both the paper's wording and the zeros; it is the default, over the point index.
               Linear interpolation over the index is ruled out by the sample. What exactly the authors
               ran remains an inference (`interpolation` selects "pchip", "cubic" or "linear").
  topology     Paper Def. 2: "critical points". RDP with epsilon 1e-4 degrees, run on the 200 resampled
               points, reproduces the sample's topology exactly for 1000 of 1000 trips (endpoints always
               kept, 13.3 points on average as in the paper's Table 5).
  regions      Paper B.1: the city "divided into 16 x 16 = 256 grids". Fitting the sample's cell ids
               gives square cells (0.006 deg lon x 0.005137 deg lat, ~571 m in Chengdu) numbered
               1 + lon_index * 16 + lat_index over the resampled points; with the fitted origin this
               reproduces 99.9% of the sample's 200,000 cell ids. 0 is padding.
  roads        Paper Def. 3: map-matched segments "between two intersections"; one segment id per
               ORIGINAL point in the sample (not per resampled point). Deduplicated in order of first
               occurrence (dict.fromkeys) by the original dataset class.
  normalise    utils/dataset.py: z-normalisation of (lon, lat) with fixed per-city mean and std.

The dataset class itself (padding, truncation, BOS/EOS, augmentations) is in the repository and is
reproduced line by line in `build_sample`, with Python's global `random` replaced by a generator.

For training, `prepare_units` computes the deterministic part of every sample once (resampling,
topology, region and road ids, in parallel over the job's CPUs) and `PreparedUnits.batch` adds only
what changes from step to step (augmentations, padding). That is how the original is organised -
preprocessed trips on disk, augmented by the dataset class - and it gives the same batches as
`build_sample` for the same random generator.
"""
from __future__ import annotations

import contextlib
import importlib.machinery
import logging
import math
import multiprocessing as mp
import os
import sys
import types
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .unitraj_sampling import rdp_keypoints

log = logging.getLogger("mobeval.nn.omnitraj_prep")

TRAJ_LEN, TOPOL_LEN, MAX_ROAD_LEN, MAX_REGION_LEN = 200, 128, 128, 64
TOPOLOGY_EPS = 1e-4
PAPER_GRID_N = 16
CHENGDU_CELL_M = 571.0            # the sample's cell size; used when a cell size rather than a count is wanted


def resample(lat, lon, n: int = TRAJ_LEN, interpolation: str = "pchip") -> np.ndarray:
    """(L,) lat/lon -> (n, 2) array of (lon, lat), interpolated over the point index."""
    from scipy.interpolate import CubicSpline, PchipInterpolator
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    L = len(lat)
    if L == 1:
        return np.repeat(np.array([[lon[0], lat[0]]]), n, 0)
    x = np.arange(L, dtype=float)
    q = np.linspace(0.0, L - 1.0, n)
    if L == 2:                                    # a cubic spline needs 3 points; 2 is a line
        return np.column_stack([np.interp(q, x, lon), np.interp(q, x, lat)])
    if interpolation == "linear":
        return np.column_stack([np.interp(q, x, lon), np.interp(q, x, lat)])
    f = {"pchip": PchipInterpolator, "cubic": CubicSpline}.get(interpolation)
    if f is None:
        raise ValueError("interpolation must be 'pchip', 'cubic' or 'linear'")
    return np.column_stack([f(x, lon)(q), f(x, lat)(q)])


def topology(traj: np.ndarray, eps: float = TOPOLOGY_EPS) -> np.ndarray:
    """Critical points of a resampled (n, 2) (lon, lat) trajectory: RDP in float64 degrees."""
    return traj[rdp_keypoints(np.asarray(traj, float), eps)]


@dataclass
class RegionGrid:
    """Square grid over the study area. `n` cells per side (the paper's 16) or `cell_m` metres.
    Ids 1 + ix * ny + iy as in the sample; with more cells than `max_cells`, only cells seen in
    training get an id (1..V, by frequency) and every other cell shares id V + 1."""
    lat0: float
    lon0: float
    dlat: float
    dlon: float
    nx: int
    ny: int
    vocab: Optional[Dict[int, int]] = None        # raw id -> compact id, when compacted

    @classmethod
    def fit(cls, lat, lon, n: Optional[int] = PAPER_GRID_N, cell_m: Optional[float] = None,
            max_cells: int = 20_000) -> "RegionGrid":
        lat, lon = np.asarray(lat, float), np.asarray(lon, float)
        la0, la1, lo0, lo1 = lat.min(), lat.max(), lon.min(), lon.max()
        k = math.cos(math.radians((la0 + la1) / 2))
        h_m, w_m = (la1 - la0) * 111_195.0, (lo1 - lo0) * 111_195.0 * k
        if cell_m is None:                         # n x n square cells over the square hull of the area
            cell_m = max(h_m, w_m) / n * (1 + 1e-9)
            nx = ny = int(n)
        else:
            nx, ny = int(w_m // cell_m) + 1, int(h_m // cell_m) + 1
        g = cls(la0, lo0, cell_m / 111_195.0, cell_m / (111_195.0 * k), nx, ny)
        if nx * ny > max_cells:
            raw = g.raw_ids(lat, lon)
            ids, cnt = np.unique(raw, return_counts=True)
            order = ids[np.argsort(-cnt, kind="stable")][:max_cells]
            g.vocab = {int(r): i + 1 for i, r in enumerate(order)}
        return g

    @property
    def num_grids(self) -> int:
        return len(self.vocab) + 1 if self.vocab is not None else self.nx * self.ny

    def raw_ids(self, lat, lon) -> np.ndarray:
        ix = np.clip(np.floor((np.asarray(lon, float) - self.lon0) / self.dlon), 0, self.nx - 1).astype(int)
        iy = np.clip(np.floor((np.asarray(lat, float) - self.lat0) / self.dlat), 0, self.ny - 1).astype(int)
        return 1 + ix * self.ny + iy

    def ids(self, lat, lon) -> np.ndarray:
        raw = self.raw_ids(lat, lon)
        if self.vocab is None:
            return raw
        unk = len(self.vocab) + 1
        return np.array([self.vocab.get(int(r), unk) for r in np.ravel(raw)]).reshape(np.shape(raw))

    def state(self) -> dict:
        return {"lat0": self.lat0, "lon0": self.lon0, "dlat": self.dlat, "dlon": self.dlon, "nx": self.nx,
                "ny": self.ny, "vocab": None if self.vocab is None else [[k, v] for k, v in self.vocab.items()]}

    @classmethod
    def from_state(cls, s: dict) -> "RegionGrid":
        v = s.get("vocab")
        return cls(s["lat0"], s["lon0"], s["dlat"], s["dlon"], s["nx"], s["ny"],
                   None if v is None else {int(a): int(b) for a, b in v})


@dataclass
class RoadVocab:
    """Map-matched segment ids -> compact ids 1..R (train segments, by frequency); unseen -> R + 1.
    Special tokens as in the original dataset: MASK = num_roads - 3, EOS - 2, BOS - 1, PAD 0.
    (The original feeds raw segment ids and notes that id 0 collides with padding; compact ids avoid
    that and keep the embedding table at the size of the network actually used.)"""
    index: Dict[int, int] = field(default_factory=dict)

    @classmethod
    def fit(cls, segment_ids: Sequence[np.ndarray], max_roads: int = 200_000) -> "RoadVocab":
        allid = np.concatenate([np.asarray(s, np.int64) for s in segment_ids if len(s)]) if segment_ids else np.array([], np.int64)
        allid = allid[allid >= 0]
        ids, cnt = np.unique(allid, return_counts=True)
        order = ids[np.argsort(-cnt, kind="stable")][:max_roads]
        return cls({int(r): i + 1 for i, r in enumerate(order)})

    @property
    def num_roads(self) -> int:
        return len(self.index) + 5                 # PAD, 1..R, UNK, MASK, EOS, BOS

    @property
    def unk(self) -> int:
        return len(self.index) + 1

    def encode(self, seg: np.ndarray) -> np.ndarray:
        seg = np.asarray(seg, np.int64)
        seg = seg[seg >= 0]                        # -1 = unmatched point
        return np.array([self.index.get(int(s), self.unk) for s in seg], np.int64)

    def state(self) -> list:
        return [[k, v] for k, v in self.index.items()]

    @classmethod
    def from_state(cls, s) -> "RoadVocab":
        return cls({int(a): int(b) for a, b in s})


def _dedup(seq) -> np.ndarray:
    return np.asarray(list(dict.fromkeys(np.asarray(seq).tolist())), np.int64)


def _pad_or_truncate(x: np.ndarray, target: int, pad=0):
    """utils/dataset.py pad_or_truncate: 2-D sequences keep their first `target` rows; 1-D ones keep
    the first element, elements 1..target-2 and the LAST element (so a trailing EOS survives)."""
    n = len(x)
    if n > target:
        out = x[:target] if x.ndim == 2 else np.concatenate([x[:1], x[1:target - 1], x[-1:]])
        return out, np.ones(target, np.float32)
    shape = (target,) + x.shape[1:]
    out = np.full(shape, pad, dtype=x.dtype)
    out[:n] = x
    m = np.zeros(target, np.float32)
    m[:n] = 1
    return out, m


def augment_road(seq: np.ndarray, rng: np.random.Generator, mask_idx: int, num_masks: int = 5,
                 window: int = 3) -> np.ndarray:
    """TrajectoryDataset.augment_road_sequence (augment_prob 0.7, five types, Python randint inclusive)."""
    if rng.random() >= 0.7:
        return seq
    kind = ("subsequence", "reverse", "random_mask", "random_drop", "random_shuffle_local")[int(rng.integers(5))]
    n = len(seq)
    if kind == "subsequence":
        if n <= 3:
            return seq
        start = int(rng.integers(0, n - 2 + 1))
        length = int(rng.integers(2, n - start + 1))
        return seq[start:start + length]
    if kind == "reverse":
        return seq[::-1].copy()
    if kind == "random_mask":
        if n <= num_masks * 3:
            return seq
        out = seq.copy()
        out[rng.permutation(n)[:num_masks]] = mask_idx
        return out
    if kind == "random_drop":
        if n <= 3:
            return seq
        return seq[rng.random(n) > 0.1]
    if n <= window:                                # random_shuffle_local
        return seq
    out = seq.copy()
    for i in range(0, n - window + 1, window):
        out[i:i + window] = out[i:i + window][rng.permutation(window)]
    return out


def augment_region(seq: np.ndarray, rng: np.random.Generator, p_shuffle: float = 0.5,
                   p_remove: float = 0.7) -> np.ndarray:
    """TrajectoryDataset.augment_region_sequence. One draw decides both steps, as in the original:
    a shuffled sequence is therefore always also thinned."""
    prob = rng.random()
    seq = seq.copy()
    if prob < p_shuffle:
        w = min(5, len(seq) - 1)
        start = int(rng.integers(0, len(seq) - w + 1))
        seq[start:start + w] = seq[start:start + w][rng.permutation(w)]
    if prob < p_remove:
        keep = rng.random(len(seq)) > 0.2
        if keep.sum() > 0:
            seq = seq[keep]
    return seq


def base_sample(lat, lon, grid: RegionGrid, norm: Dict[str, Sequence[float]], topology_eps: float = TOPOLOGY_EPS,
                interpolation: str = "pchip"):
    """The part of a sample that does not depend on the random generator: the normalised 200-point
    trajectory, the normalised topology (not padded) and the deduplicated region ids."""
    traj = resample(lat, lon, TRAJ_LEN, interpolation)
    mean, std = np.asarray(norm["mean"], float), np.asarray(norm["std"], float)
    topo = ((topology(traj, topology_eps) - mean) / std).astype(np.float32)
    reg = _dedup(grid.ids(traj[:, 1], traj[:, 0]))
    return ((traj - mean) / std).astype(np.float32), topo, reg


def road_tokens(roads: Optional[np.ndarray], vocab: RoadVocab) -> np.ndarray:
    """Deduplicated road tokens of one trip, from one segment id per original point (-1 = unmatched)."""
    return _dedup(vocab.encode(roads if roads is not None else np.array([], np.int64)))


def finish_sample(topo: np.ndarray, reg: np.ndarray, road: Optional[np.ndarray], num_roads: Optional[int],
                  rng: Optional[np.random.Generator], augment: bool, trajectory: Optional[np.ndarray] = None) -> dict:
    """Augmentations (region first, then road, as the original draws them) and padding. The inputs
    are not modified, so they can be cached and reused every epoch."""
    out = {} if trajectory is None else {"trajectory": trajectory}
    out["topology"], out["topology_attention_mask"] = _pad_or_truncate(topo, TOPOL_LEN)
    if augment:
        reg = augment_region(reg, rng)
    out["region"], out["region_attention_mask"] = _pad_or_truncate(reg, MAX_REGION_LEN, 0)
    if road is not None:
        r = road
        if augment:
            r = augment_road(r, rng, num_roads - 3)
        r = np.concatenate([[num_roads - 1], r, [num_roads - 2]]).astype(np.int64)
        out["road"], out["road_attention_mask"] = _pad_or_truncate(r, MAX_ROAD_LEN, 0)
    return out


def build_sample(lat, lon, roads: Optional[np.ndarray], grid: RegionGrid, vocab: Optional[RoadVocab],
                 norm: Dict[str, Sequence[float]], rng: Optional[np.random.Generator], augment: bool,
                 topology_eps: float = TOPOLOGY_EPS, interpolation: str = "pchip") -> dict:
    """One training/evaluation example, as TrajectoryDataset.__getitem__ returns it, from raw points.
    `roads` holds one map-matched segment id per ORIGINAL point (-1 = unmatched), or None."""
    traj, topo, reg = base_sample(lat, lon, grid, norm, topology_eps, interpolation)
    road = road_tokens(roads, vocab) if vocab is not None else None
    return finish_sample(topo, reg, road, vocab.num_roads if vocab is not None else None, rng, augment, traj)


class PreparedUnits:
    """`base_sample` (and the road tokens) of every training unit, computed once.

    `batch(idx, rng, augment)` returns exactly `collate([build_sample(...) for i in idx])` for the
    same generator state, but only augments and pads: the resampling, the topology and the id
    lookups - nearly all of the CPU time of a step - are not redone every epoch."""

    def __init__(self, trajectory: np.ndarray, topology: List[np.ndarray], region: List[np.ndarray],
                 road: Optional[List[np.ndarray]], num_roads: Optional[int]):
        self.trajectory = trajectory          # (N, 200, 2) float32, normalised
        self.topology = topology              # N x (k, 2) float32, normalised, not padded
        self.region = region                  # N x deduplicated cell ids
        self.road = road                      # N x deduplicated road tokens, or None (no road encoder)
        self.num_roads = num_roads

    def __len__(self) -> int:
        return len(self.topology)

    @property
    def nbytes(self) -> int:
        parts = [self.topology, self.region] + ([self.road] if self.road is not None else [])
        return int(self.trajectory.nbytes + sum(a.nbytes for p in parts for a in p))

    def batch(self, idx, rng: Optional[np.random.Generator], augment: bool) -> Dict[str, np.ndarray]:
        idx = np.asarray(idx)
        out = collate([finish_sample(self.topology[i], self.region[i], None if self.road is None else self.road[i],
                                     self.num_roads, rng, augment) for i in idx])
        out["trajectory"] = self.trajectory[idx]
        return out


def available_cpus() -> int:
    """CPUs this job may use: PBS's NCPUS (or Slurm's), else the process's CPU affinity."""
    for var in ("NCPUS", "SLURM_CPUS_PER_TASK"):
        v = os.environ.get(var, "")
        if v.isdigit() and int(v) > 0:
            return int(v)
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:                                  # not on Linux
        return os.cpu_count() or 1


_WORK = None                                                # (grid, norm, eps, interpolation) in each worker


def _init_worker(grid, norm, eps, interp):
    global _WORK
    _WORK = (grid, norm, eps, interp)


def _base_list(rows):
    grid, norm, eps, interp = _WORK
    return [base_sample(lat, lon, grid, norm, eps, interp) for lat, lon in rows]


@contextlib.contextmanager
def _caller_script_hidden():
    """multiprocessing imports the caller's __main__ in every spawned or forkserver child; a script
    without an `if __name__ == "__main__":` guard would then run again in each worker. The workers
    need only this module, so while they start, __main__ is a stand-in they skip."""
    main = sys.modules.get("__main__")
    stand_in = types.ModuleType("__main__")
    stand_in.__spec__ = importlib.machinery.ModuleSpec("__main__", None)
    sys.modules["__main__"] = stand_in
    try:
        yield
    finally:
        sys.modules["__main__"] = main


def prepare_units(units: Sequence[tuple], roads: Optional[Sequence[np.ndarray]], grid: RegionGrid,
                  vocab: Optional[RoadVocab], norm: Dict[str, Sequence[float]], topology_eps: float = TOPOLOGY_EPS,
                  interpolation: str = "pchip", workers: Optional[int] = None, chunk: int = 1000,
                  chunk_timeout_s: float = 600.0) -> PreparedUnits:
    """PreparedUnits for `units` = [(traj_id, lat, lon, t), ...]; `roads` = one segment-id array per
    unit (or None without roads). The resampling and topology run in `workers` processes (default:
    the job's CPUs), forked from a fresh single-threaded server ("forkserver") rather than from this
    process: the training process has CUDA, torch and progress threads, and a child forked from it can
    inherit one of their locks held and wait on it forever. The workers load only this module, never
    the calling script. A chunk takes seconds; if one fails or takes longer than `chunk_timeout_s`, the
    workers are stopped and the work is done in this process instead."""
    n = len(units)
    workers = max(1, int(workers or available_cpus()))
    bases = None
    if workers > 1 and n >= 2 * chunk:
        pool = None
        try:
            ctx = mp.get_context("forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn")
            if ctx.get_start_method() == "forkserver":
                ctx.set_forkserver_preload([__name__])
            with _caller_script_hidden():
                pool = ctx.Pool(min(workers, (n + chunk - 1) // chunk), initializer=_init_worker,
                                initargs=(grid, norm, topology_eps, interpolation))
            parts = pool.imap(_base_list, ([(u[1], u[2]) for u in units[s:s + chunk]] for s in range(0, n, chunk)))
            bases = []
            for _ in range(0, n, chunk):
                bases += parts.next(timeout=chunk_timeout_s)
            pool.close()
            pool.join()
        except Exception as e:                              # noqa: BLE001 - fall back, never fail training
            log.warning(f"parallel preprocessing failed ({e!r}); doing it in one process")
            if pool is not None:
                pool.terminate()
            bases = None
    if bases is None:
        bases = [base_sample(u[1], u[2], grid, norm, topology_eps, interpolation) for u in units]
    traj = np.stack([b[0] for b in bases]) if bases else np.zeros((0, TRAJ_LEN, 2), np.float32)
    road = None
    if vocab is not None:
        road = [road_tokens(r, vocab) for r in (roads if roads is not None else [None] * n)]
    return PreparedUnits(traj, [b[1] for b in bases], [b[2] for b in bases], road,
                         vocab.num_roads if vocab is not None else None)


def collate(samples: List[dict]) -> Dict[str, np.ndarray]:
    return {k: np.stack([s[k] for s in samples]) for k in samples[0]}
