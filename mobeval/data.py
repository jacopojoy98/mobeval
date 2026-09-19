"""Canonical data layer.

A single raw GPS dataset is the source of truth. Every model-specific view
(fixed-length GPS windows for UniTraj / your model, staypoint visit sequences
for TrajGPT, grid tokens for location prediction) is DERIVED from it with the
same split, so all models are scored on the same underlying people and days.

Canonical point table columns:
    user_id, traj_id, t (unix seconds, float), lat, lon, [mode]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .geo import LocalProjection, haversine_m

REQUIRED_COLUMNS = ("user_id", "traj_id", "t", "lat", "lon")


# --------------------------------------------------------------------------- #
# Dataset + splits
# --------------------------------------------------------------------------- #
class MobilityDataset:
    def __init__(self, points: pd.DataFrame, name: str = "dataset"):
        missing = [c for c in REQUIRED_COLUMNS if c not in points.columns]
        if missing:
            raise ValueError(f"points table missing columns: {missing}")
        df = points.copy()
        df["t"] = df["t"].astype(float)
        if not (df["lat"].between(-90, 90).all() and df["lon"].between(-180, 180).all()):
            raise ValueError("lat/lon out of range - are the coordinates normalised or swapped?")
        self.points = df.sort_values(["user_id", "traj_id", "t"]).reset_index(drop=True)
        self.name = name

    def __len__(self):
        return len(self.points)

    @property
    def has_modes(self) -> bool:
        return "mode" in self.points.columns

    def subset_users(self, users) -> "MobilityDataset":
        return MobilityDataset(self.points[self.points.user_id.isin(set(users))], self.name)

    def split(self, by: str = "time", ratios=(0.8, 0.1, 0.1), seed: int = 0, val_by: str = "time",
              val_fraction: float = 0.1) -> Dict[str, "MobilityDataset"]:
        """by='time': chronological split of trajectories (UniTE protocol, 8:1:1).
        by='user':  disjoint users (tests generalisation to unseen people).
        by='predefined': use the `split` column (train/test, optionally val). Without val rows, a
        validation set is carved from TRAIN only (`val_by` = 'time': latest trajectories, or 'user')."""
        names = ("train", "val", "test")
        if by == "predefined":
            return self._predefined_split(val_by, val_fraction, seed)
        assert abs(sum(ratios) - 1) < 1e-9
        if by == "user":
            users = np.array(sorted(self.points.user_id.unique()))
            np.random.default_rng(seed).shuffle(users)
            cuts = np.cumsum([int(round(r * len(users))) for r in ratios[:-1]])
            groups = np.split(users, cuts)
            return {n: self.subset_users(g) for n, g in zip(names, groups)}
        if by == "time":
            starts = self.points.groupby("traj_id")["t"].min().sort_values()
            ids = starts.index.to_numpy()
            cuts = np.cumsum([int(round(r * len(ids))) for r in ratios[:-1]])
            groups = np.split(ids, cuts)
            return {n: MobilityDataset(self.points[self.points.traj_id.isin(set(g))], self.name)
                    for n, g in zip(names, groups)}
        raise ValueError("by must be 'time', 'user' or 'predefined'")

    def _predefined_split(self, val_by: str, val_fraction: float, seed: int):
        if "split" not in self.points.columns:
            raise ValueError("split_by='predefined' needs a 'split' column (train/test[/val]); "
                             "from_csv(train_path=..., test_path=...) adds it")
        p = self.points
        labels = set(p["split"].unique())
        if not labels <= {"train", "val", "test"} or not {"train", "test"} <= labels:
            raise ValueError(f"'split' column must contain train and test (and optionally val), found {sorted(labels)}")
        both = set(p.loc[p["split"] == "train", "traj_id"]) & set(p.loc[p["split"] == "test", "traj_id"])
        if both:
            raise ValueError(f"{len(both)} trajectories appear in both train and test, e.g. {sorted(both)[:3]}")
        sub = lambda mask: MobilityDataset(p[mask], self.name)
        out = {"test": sub(p["split"] == "test")}
        if "val" in labels:
            out["train"], out["val"] = sub(p["split"] == "train"), sub(p["split"] == "val")
            return out
        tr = p[p["split"] == "train"]
        if val_by == "user":
            users = np.array(sorted(tr.user_id.unique()))
            np.random.default_rng(seed).shuffle(users)
            val_ids = set(tr.loc[tr.user_id.isin(users[:max(1, int(round(val_fraction * len(users))))]), "traj_id"])
        elif val_by == "time":
            starts = tr.groupby("traj_id")["t"].min().sort_values()
            val_ids = set(starts.index[len(starts) - max(1, int(round(val_fraction * len(starts)))):])
        else:
            raise ValueError("val_by must be 'time' or 'user'")
        is_val = tr.traj_id.isin(val_ids)
        out["train"], out["val"] = MobilityDataset(tr[~is_val], self.name), MobilityDataset(tr[is_val], self.name)
        return out


# --------------------------------------------------------------------------- #
# Shared spatial discretisation
# --------------------------------------------------------------------------- #
class SpatialGrid:
    """Square metric grid shared by ALL models. Models with their own tokenizer
    (e.g. TrajGPT's H3 cells) are mapped onto it via token centroids, so
    location accuracy is always measured on the same label space."""

    def __init__(self, lat_min, lat_max, lon_min, lon_max, cell_size_m: float = 500.0):
        self.cell_size_m = float(cell_size_m)
        self.proj = LocalProjection((lat_min + lat_max) / 2, (lon_min + lon_max) / 2)
        x0, y0 = self.proj.to_xy(lat_min, lon_min)
        x1, y1 = self.proj.to_xy(lat_max, lon_max)
        self.x0, self.y0 = float(x0), float(y0)
        self.nx = int(np.floor((x1 - x0) / self.cell_size_m)) + 1
        self.ny = int(np.floor((y1 - y0) / self.cell_size_m)) + 1
        self.bounds = (lat_min, lat_max, lon_min, lon_max)

    @classmethod
    def from_dataset(cls, ds: MobilityDataset, cell_size_m=500.0, pad_m=1000.0) -> "SpatialGrid":
        p = ds.points
        pad_lat = pad_m / 111_195.0
        pad_lon = pad_m / (111_195.0 * np.cos(np.radians(p.lat.mean())))
        return cls(p.lat.min() - pad_lat, p.lat.max() + pad_lat,
                   p.lon.min() - pad_lon, p.lon.max() + pad_lon, cell_size_m)

    @property
    def n_cells(self) -> int:
        return self.nx * self.ny

    def cell_of(self, lat, lon) -> np.ndarray:
        x, y = self.proj.to_xy(lat, lon)
        ix = np.clip(np.floor((np.asarray(x) - self.x0) / self.cell_size_m), 0, self.nx - 1).astype(int)
        iy = np.clip(np.floor((np.asarray(y) - self.y0) / self.cell_size_m), 0, self.ny - 1).astype(int)
        return iy * self.nx + ix

    def centroid(self, cell):
        cell = np.asarray(cell)
        ix, iy = cell % self.nx, cell // self.nx
        return self.proj.to_latlon(self.x0 + (ix + 0.5) * self.cell_size_m,
                                   self.y0 + (iy + 0.5) * self.cell_size_m)


# --------------------------------------------------------------------------- #
# View 1: fixed-length GPS windows (recovery, mode classification, embeddings)
# --------------------------------------------------------------------------- #
@dataclass
class TrajectoryBatch:
    lat: np.ndarray            # (N, L) degrees
    lon: np.ndarray            # (N, L) degrees
    t: np.ndarray              # (N, L) unix seconds
    user_id: np.ndarray        # (N,)
    traj_id: np.ndarray        # (N,)
    mode: Optional[np.ndarray] = None   # (N,) window label or None

    def __len__(self):
        return self.lat.shape[0]

    @property
    def length(self):
        return self.lat.shape[1]

    def take(self, idx) -> "TrajectoryBatch":
        return TrajectoryBatch(self.lat[idx], self.lon[idx], self.t[idx], self.user_id[idx],
                               self.traj_id[idx], None if self.mode is None else self.mode[idx])


def make_windows(ds: MobilityDataset, length: int = 64, stride: Optional[int] = None,
                 max_gap_s: float = 300.0, min_mode_purity: float = 0.8,
                 moving_only_for_modes: bool = True) -> TrajectoryBatch:
    """Cut every trajectory into fixed-length windows without large sampling gaps."""
    stride = stride or length
    lats, lons, ts, users, trajs, modes = [], [], [], [], [], []
    has_mode = ds.has_modes
    for (uid, tid), g in ds.points.groupby(["user_id", "traj_id"], sort=False):
        la, lo, tt = g.lat.to_numpy(), g.lon.to_numpy(), g.t.to_numpy()
        md = g["mode"].to_numpy() if has_mode else None
        breaks = np.where(np.diff(tt) > max_gap_s)[0] + 1
        for seg in np.split(np.arange(len(tt)), breaks):
            for s in range(0, len(seg) - length + 1, stride):
                idx = seg[s:s + length]
                label = None
                if has_mode:
                    m = pd.Series(md[idx]).dropna()
                    if len(m) and m.value_counts(normalize=True).iloc[0] >= min_mode_purity \
                            and (not moving_only_for_modes or len(m) >= min_mode_purity * length):
                        label = m.value_counts().index[0]
                lats.append(la[idx]); lons.append(lo[idx]); ts.append(tt[idx])
                users.append(uid); trajs.append(tid); modes.append(label)
    if not lats:
        raise ValueError("no windows produced - reduce `length` or increase `max_gap_s`")
    return TrajectoryBatch(np.stack(lats), np.stack(lons), np.stack(ts), np.array(users),
                           np.array(trajs), np.array(modes, dtype=object) if has_mode else None)


def make_mask(n: int, length: int, ratio: float, kind: str = "random", seed: int = 0,
              keep_endpoints: bool = True) -> np.ndarray:
    """Boolean mask, True = hidden from the model. Generated ONCE by the pipeline
    (seeded) and passed to every model, so all models reconstruct the same points.
    Endpoints are kept observed so interpolation baselines are well defined."""
    rng = np.random.default_rng(seed)
    lo, hi = (1, length - 1) if keep_endpoints else (0, length)
    avail = hi - lo
    k = int(np.clip(round(ratio * length), 1, avail))
    mask = np.zeros((n, length), dtype=bool)
    for i in range(n):
        if kind == "random":
            mask[i, lo + rng.choice(avail, size=k, replace=False)] = True
        elif kind == "block":
            s = rng.integers(lo, hi - k + 1)
            mask[i, s:s + k] = True
        else:
            raise ValueError("kind must be 'random' or 'block'")
    return mask


# --------------------------------------------------------------------------- #
# View 2: staypoints / visit sequences (TrajGPT-style tasks)
# --------------------------------------------------------------------------- #
def detect_staypoints(ds: MobilityDataset, dist_thresh_m: float = 200.0,
                      time_thresh_s: float = 20 * 60, by_trajectory: bool = False) -> pd.DataFrame:
    """Classic staypoint detection (Li et al., 2008). Returns one row per visit.
    by_trajectory=True never merges points of different trajectories (use for
    generated data, where several trajectories of one user may overlap in time)."""
    rows = []
    keys = ["user_id", "traj_id"] if by_trajectory else "user_id"
    for key, g in ds.points.groupby(keys, sort=False):
        uid = key[0] if by_trajectory else key
        g = g.sort_values("t", kind="stable")             # chronological, whatever the trajectory ids are
        la, lo, tt = g.lat.to_numpy(), g.lon.to_numpy(), g.t.to_numpy()
        tids = g.traj_id.to_numpy()
        n, i = len(tt), 0
        while i < n:
            j = i + 1
            while j < n and haversine_m(la[i], lo[i], la[j], lo[j]) <= dist_thresh_m:
                j += 1
            if tt[j - 1] - tt[i] >= time_thresh_s:
                rows.append((uid, tids[i], float(la[i:j].mean()), float(lo[i:j].mean()), tt[i], tt[j - 1]))
                i = j
            else:
                i += 1
    sp = pd.DataFrame(rows, columns=["user_id", "traj_id", "lat", "lon", "t_arrive", "t_leave"])
    return sp.sort_values(["user_id", "t_arrive"]).reset_index(drop=True)


def staypoints_from_trips(ds: MobilityDataset, min_stay_s: float = 20 * 60, max_link_dist_m: float = 500.0,
                          max_stay_s: float = 3 * 86400) -> pd.DataFrame:
    """Staypoints for trip-segmented data (e.g. vehicle black boxes that record only while moving).
    A stay is the gap between the end of one trip and the start of the user's next trip, if it lasts
    at least `min_stay_s`. Location: midpoint of trip end and next trip start when they are within
    `max_link_dist_m`, otherwise the trip end. Gaps above `max_stay_s` are treated as missing data."""
    p = ds.points.sort_values(["user_id", "t"], kind="stable")
    trips = p.groupby(["user_id", "traj_id"], sort=False).agg(
        t_start=("t", "first"), t_end=("t", "last"), lat_s=("lat", "first"), lon_s=("lon", "first"),
        lat_e=("lat", "last"), lon_e=("lon", "last")).reset_index().sort_values(["user_id", "t_start"])
    rows = []
    for uid, g in trips.groupby("user_id", sort=False):
        g = g.reset_index(drop=True)
        for i in range(len(g) - 1):
            a, b = g.iloc[i], g.iloc[i + 1]
            gap = b.t_start - a.t_end
            if gap < min_stay_s or gap > max_stay_s:
                continue
            d = haversine_m(a.lat_e, a.lon_e, b.lat_s, b.lon_s)
            lat, lon = ((a.lat_e + b.lat_s) / 2, (a.lon_e + b.lon_s) / 2) if d <= max_link_dist_m else (a.lat_e, a.lon_e)
            rows.append((uid, a.traj_id, float(lat), float(lon), float(a.t_end), float(b.t_start)))
    sp = pd.DataFrame(rows, columns=["user_id", "traj_id", "lat", "lon", "t_arrive", "t_leave"])
    return sp.sort_values(["user_id", "t_arrive"]).reset_index(drop=True)


def clean_points(df: pd.DataFrame, max_speed_mps: float = 70.0, min_points: int = 2,
                 drop_duplicate_times: bool = True) -> pd.DataFrame:
    """Remove NaN/invalid coordinates, duplicate timestamps within a trajectory, points implying an
    impossible speed from the previous kept point, and trajectories with fewer than `min_points`."""
    n0 = len(df)
    df = df.dropna(subset=["user_id", "traj_id", "t", "lat", "lon"])
    df = df[df.lat.between(-90, 90) & df.lon.between(-180, 180) & ~((df.lat == 0) & (df.lon == 0))]
    df = df.sort_values(["traj_id", "t"], kind="stable")
    if drop_duplicate_times:
        df = df.drop_duplicates(["traj_id", "t"])
    keep = np.ones(len(df), bool)
    lat, lon, t, tid = df.lat.to_numpy(), df.lon.to_numpy(), df.t.to_numpy(), df.traj_id.to_numpy()
    last = 0
    for i in range(1, len(df)):
        if tid[i] != tid[last] or not keep[last]:
            last = i
            continue
        v = haversine_m(lat[last], lon[last], lat[i], lon[i]) / max(t[i] - t[last], 1.0)
        if v > max_speed_mps:
            keep[i] = False
        else:
            last = i
    df = df[keep]
    df = df[df.groupby("traj_id").t.transform("size") >= min_points]
    import logging
    logging.getLogger("mobeval.data").info(f"clean_points: kept {len(df)}/{n0} points "
                                           f"({df.traj_id.nunique()} trajectories)")
    return df.reset_index(drop=True)


@dataclass
class VisitBatch:
    """Context of C visits -> one target visit. Times in seconds."""
    ctx_lat: np.ndarray        # (N, C)
    ctx_lon: np.ndarray
    ctx_cell: np.ndarray       # (N, C) shared-grid cell ids
    ctx_t_arrive: np.ndarray
    ctx_t_leave: np.ndarray
    user_id: np.ndarray        # (N,)
    tgt_lat: np.ndarray        # (N,)
    tgt_lon: np.ndarray
    tgt_cell: np.ndarray
    tgt_travel_time_s: np.ndarray   # arrival(target) - departure(last context visit)
    tgt_duration_s: np.ndarray      # leave(target) - arrive(target)
    tgt_t_arrive: np.ndarray

    def __len__(self):
        return len(self.tgt_cell)


def make_visit_sequences(staypoints: pd.DataFrame, grid: SpatialGrid, context: int = 8,
                         stride: int = 1, target_traj_ids=None) -> VisitBatch:
    """Staypoints should come from the FULL history of each user. With
    `target_traj_ids`, only targets belonging to those trajectories (i.e. to one
    split) are kept, while the context may use the user's earlier visits - this
    is what a deployed next-location model sees, and it avoids an empty test set
    under chronological splits. Targets never leak into another split's targets."""
    keep_targets = None if target_traj_ids is None else set(target_traj_ids)
    buf = {k: [] for k in VisitBatch.__dataclass_fields__}
    for uid, g in staypoints.groupby("user_id", sort=False):
        la, lo = g.lat.to_numpy(), g.lon.to_numpy()
        ta, tl = g.t_arrive.to_numpy(), g.t_leave.to_numpy()
        cells = grid.cell_of(la, lo)
        tr = g.traj_id.to_numpy() if "traj_id" in g else None
        for s in range(0, len(g) - context, stride):
            c, t = slice(s, s + context), s + context
            if keep_targets is not None and tr[t] not in keep_targets:
                continue
            buf["ctx_lat"].append(la[c]); buf["ctx_lon"].append(lo[c]); buf["ctx_cell"].append(cells[c])
            buf["ctx_t_arrive"].append(ta[c]); buf["ctx_t_leave"].append(tl[c])
            buf["user_id"].append(uid)
            buf["tgt_lat"].append(la[t]); buf["tgt_lon"].append(lo[t]); buf["tgt_cell"].append(cells[t])
            buf["tgt_travel_time_s"].append(ta[t] - tl[t - 1]); buf["tgt_duration_s"].append(tl[t] - ta[t])
            buf["tgt_t_arrive"].append(ta[t])
    if not buf["tgt_cell"]:
        raise ValueError("no visit sequences produced - reduce `context`")
    return VisitBatch(**{k: np.asarray(v) for k, v in buf.items()})


# --------------------------------------------------------------------------- #
# Synthetic data (for tests / smoke runs only)
# --------------------------------------------------------------------------- #
MODE_SPEEDS = {"walk": 1.4, "bike": 4.5, "bus": 7.0, "car": 11.0, "train": 20.0}


def synthetic_dataset(n_users: int = 20, n_days: int = 6, seed: int = 0,
                      lat0: float = 55.676, lon0: float = 12.568) -> MobilityDataset:
    """Anchor-based synthetic mobility (home/work/others), GPS noise, 5 modes."""
    rng = np.random.default_rng(seed)
    proj = LocalProjection(lat0, lon0)
    rows, day0 = [], 1_700_000_000.0
    for u in range(n_users):
        n_anchor = rng.integers(3, 6)
        anchors = rng.normal(0, 4000 + 2000 * rng.random(), size=(n_anchor, 2))
        for d in range(n_days):
            tid = f"u{u}_d{d}"
            t = day0 + d * 86400 + 7 * 3600 + rng.normal(0, 1800)
            seq = [0] + list(rng.choice(np.arange(1, n_anchor), size=rng.integers(1, 4))) + [0]
            for a, b in zip(seq[:-1], seq[1:]):
                stay = rng.uniform(25, 240) * 60
                for k in range(int(stay // 300)):          # sparse sampling while staying
                    x, y = anchors[a] + rng.normal(0, 15, 2)
                    rows.append((f"u{u}", tid, t + k * 300, *proj.to_latlon(x, y), None))
                t += stay
                if a == b:
                    continue
                dist = np.linalg.norm(anchors[b] - anchors[a])
                mode = ("walk" if dist < 1200 else "bike" if dist < 3000 else
                        rng.choice(["bus", "car", "train"], p=[0.4, 0.4, 0.2]))
                v = MODE_SPEEDS[mode] * rng.uniform(0.8, 1.2)
                n_pts = max(int(dist / v / 15), 2)                 # 15 s sampling while moving
                bend = rng.normal(0, 0.15 * dist, 2)
                for k in range(n_pts):
                    s = k / (n_pts - 1)
                    xy = (1 - s) * anchors[a] + s * anchors[b] + 4 * s * (1 - s) * bend + rng.normal(0, 8, 2)
                    rows.append((f"u{u}", tid, t + k * 15, *proj.to_latlon(*xy), mode))
                t += n_pts * 15
    df = pd.DataFrame(rows, columns=["user_id", "traj_id", "t", "lat", "lon", "mode"])
    return MobilityDataset(df, name="synthetic")
