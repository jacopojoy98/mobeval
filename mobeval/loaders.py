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


def from_csv(path: str, name: Optional[str] = None, time_col: str = "t", **rename) -> MobilityDataset:
    """Generic CSV/Parquet with columns user_id, traj_id, t, lat, lon[, mode].
    `t` may be unix seconds or a datetime string. Pass rename=dict(old=new) as kwargs."""
    p = Path(path)
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    df = df.rename(columns={v: k for k, v in rename.items()}) if rename else df
    if not np.issubdtype(df[time_col].dtype, np.number):
        df[time_col] = to_unix_seconds(df[time_col])
    return MobilityDataset(df.rename(columns={time_col: "t"}), name or p.stem)


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
