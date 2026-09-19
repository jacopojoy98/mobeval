"""Dataset loaders -> canonical MobilityDataset."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .data import MobilityDataset

def to_unix_seconds(x) -> pd.Series:
    """Resolution-independent datetime -> unix seconds (pandas>=2 may store s/ms/us/ns)."""
    return (pd.to_datetime(x) - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)


GEOLIFE_MODE_MAP = {"walk": "walk", "bike": "bike", "bus": "bus", "car": "car", "taxi": "car",
                    "train": "train", "subway": "train", "railway": "train"}


def _read_table(path):
    p = Path(path)
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)


def from_csv(path: Optional[str] = None, name: Optional[str] = None, time_col: str = "t",
             train_path: Optional[str] = None, test_path: Optional[str] = None, val_path: Optional[str] = None,
             query: Optional[str] = None, clean: Optional[dict] = None, keep_columns=(), **rename) -> MobilityDataset:
    """CSV/Parquet GPS table(s) -> MobilityDataset.

    path                     one file (mobeval splits it), OR
    train_path / test_path   pre-split files (+ optional val_path); adds a `split` column for
                             EvalConfig(split_by="predefined")
    time_col                 unix seconds or a datetime column. Naive datetimes are kept as local
                             clock time, so time-of-day features refer to local time
    rename                   canonical=original, e.g. user_id="uid", traj_id="trip_id", lon="lng"
    query                    optional pandas query applied before renaming, e.g. "QUALITY >= 2"
    clean                    optional clean_points() arguments, e.g. {max_speed_mps: 70}; {} = defaults
    keep_columns             extra original columns to keep (after renaming)
    """
    from .data import clean_points
    files = {"all": path} if path else {k: v for k, v in (("train", train_path), ("val", val_path), ("test", test_path)) if v}
    if not files or (path and (train_path or test_path)):
        raise ValueError("give either `path` or `train_path` + `test_path`")
    if not path and not (train_path and test_path):
        raise ValueError("pre-split data needs both train_path and test_path")
    frames = []
    for split, f in files.items():
        df = _read_table(f)
        if query:
            df = df.query(query)
        if split != "all":
            df = df.assign(split=split)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    mapping = {orig: canon for canon, orig in rename.items()}
    missing = [c for c in list(mapping) + [time_col] if c not in df.columns]
    if missing:
        raise ValueError(f"columns not found: {missing}; available: {list(df.columns)}")
    df = df.rename(columns={**mapping, time_col: "t"})
    if not pd.api.types.is_numeric_dtype(df["t"]):
        df["t"] = to_unix_seconds(df["t"])
    wanted = ["user_id", "traj_id", "t", "lat", "lon"] + [c for c in ("mode", "split") if c in df.columns] + list(keep_columns)
    lacking = [c for c in ("user_id", "traj_id", "lat", "lon") if c not in df.columns]
    if lacking:
        raise ValueError(f"no column mapped to {lacking}; pass e.g. user_id='uid', traj_id='trip_id', lon='lng'")
    df = df[wanted].copy()
    # trajectory ids only need to be unique per user in many datasets: make them globally unique
    df["user_id"] = df["user_id"].astype(str)
    df["traj_id"] = df["user_id"] + ":" + df["traj_id"].astype(str)
    if "split" in df.columns:
        per_traj = df.groupby("traj_id")["split"].nunique()
        if (per_traj > 1).any():
            raise ValueError(f"{int((per_traj > 1).sum())} trajectories occur in more than one split file")
    if clean is not None:
        df = clean_points(df, **clean)
    return MobilityDataset(df, name or Path(path or train_path).stem)


def load_geolife(root: str, users=None, labelled_only: bool = False, max_speed_mps: float = 60.0) -> MobilityDataset:
    """GeoLife 1.3 (root = folder containing Data/). Mode labels from labels.txt are mapped
    to 5 classes (taxi->car, subway/railway->train); unlabelled points get mode=None.
    Implausible jumps (> max_speed_mps) are removed, following UniTraj-style filtering."""
    root = Path(root) / "Data" if (Path(root) / "Data").exists() else Path(root)
    frames = []
    for udir in sorted(p for p in root.iterdir() if p.is_dir()):
        if users is not None and udir.name not in set(users):
            continue
        labels = None
        lf = udir / "labels.txt"
        if lf.exists():
            labels = pd.read_csv(lf, sep="\t")
            labels.columns = ["start", "end", "mode"]
            labels["start"] = to_unix_seconds(labels.start)
            labels["end"] = to_unix_seconds(labels.end)
            labels["mode"] = labels["mode"].str.lower().map(GEOLIFE_MODE_MAP)
            labels = labels.dropna().sort_values("start")
        if labelled_only and labels is None:
            continue
        for plt in sorted((udir / "Trajectory").glob("*.plt")):
            d = pd.read_csv(plt, skiprows=6, header=None, usecols=[0, 1, 5, 6], names=["lat", "lon", "date", "time"])
            if d.empty:
                continue
            d["t"] = to_unix_seconds(d.date + " " + d.time)
            d = d.drop(columns=["date", "time"]).sort_values("t")
            d["user_id"], d["traj_id"] = udir.name, f"{udir.name}_{plt.stem}"
            d["mode"] = None
            if labels is not None and len(labels):
                i = np.searchsorted(labels.start.to_numpy(), d.t.to_numpy(), side="right") - 1
                ok = (i >= 0) & (d.t.to_numpy() <= labels.end.to_numpy()[np.clip(i, 0, None)])
                d.loc[ok, "mode"] = labels["mode"].to_numpy()[i[ok]]
            frames.append(d)
    if not frames:
        raise FileNotFoundError(f"no GeoLife trajectories under {root}")
    df = pd.concat(frames, ignore_index=True)
    df = df[df.lat.between(-90, 90) & df.lon.between(-180, 180)]
    # drop points implying impossible speed w.r.t. the previous point of the same trajectory
    from .geo import haversine_m
    same = df.traj_id.eq(df.traj_id.shift())
    dist = haversine_m(df.lat.shift(), df.lon.shift(), df.lat, df.lon)
    dt = (df.t - df.t.shift()).clip(lower=1)
    df = df[~(same & (dist / dt > max_speed_mps))]
    if labelled_only:
        df = df[df["mode"].notna() | df.traj_id.isin(df.loc[df["mode"].notna(), "traj_id"].unique())]
    return MobilityDataset(df.reset_index(drop=True), "geolife")
