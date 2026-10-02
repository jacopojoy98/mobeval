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
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .unitraj_sampling import rdp_keypoints

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


def build_sample(lat, lon, roads: Optional[np.ndarray], grid: RegionGrid, vocab: Optional[RoadVocab],
                 norm: Dict[str, Sequence[float]], rng: Optional[np.random.Generator], augment: bool,
                 topology_eps: float = TOPOLOGY_EPS, interpolation: str = "pchip") -> dict:
    """One training/evaluation example, as TrajectoryDataset.__getitem__ returns it, from raw points.
    `roads` holds one map-matched segment id per ORIGINAL point (-1 = unmatched), or None."""
    traj = resample(lat, lon, TRAJ_LEN, interpolation)
    mean, std = np.asarray(norm["mean"], float), np.asarray(norm["std"], float)
    topo = topology(traj, topology_eps)
    topo_n, topo_m = _pad_or_truncate(((topo - mean) / std).astype(np.float32), TOPOL_LEN)
    out = {"trajectory": ((traj - mean) / std).astype(np.float32), "topology": topo_n,
           "topology_attention_mask": topo_m}
    reg = _dedup(grid.ids(traj[:, 1], traj[:, 0]))
    if augment:
        reg = augment_region(reg, rng)
    out["region"], out["region_attention_mask"] = _pad_or_truncate(reg, MAX_REGION_LEN, 0)
    if vocab is not None:
        r = _dedup(vocab.encode(roads if roads is not None else np.array([], np.int64)))
        if augment:
            r = augment_road(r, rng, vocab.num_roads - 3)
        r = np.concatenate([[vocab.num_roads - 1], r, [vocab.num_roads - 2]]).astype(np.int64)
        out["road"], out["road_attention_mask"] = _pad_or_truncate(r, MAX_ROAD_LEN, 0)
    return out


def collate(samples: List[dict]) -> Dict[str, np.ndarray]:
    return {k: np.stack([s[k] for s in samples]) for k in samples[0]}
