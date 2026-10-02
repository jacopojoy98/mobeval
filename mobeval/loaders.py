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


def load_geolife(root: str, users=None, labelled_only: bool = False,
                 max_speed_mps: Optional[float] = 60.0) -> MobilityDataset:
    """GeoLife 1.3 (root = folder containing Data/). Mode labels from labels.txt are mapped
    to 5 classes (taxi->car, subway/railway->train); unlabelled points get mode=None.
    Implausible jumps (> max_speed_mps) are removed, following UniTraj-style filtering;
    max_speed_mps: null keeps every point, as TrajGPT's trackintel reader does."""
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
    if max_speed_mps is not None:
        from .geo import haversine_m
        same = df.traj_id.eq(df.traj_id.shift())
        dist = haversine_m(df.lat.shift(), df.lon.shift(), df.lat, df.lon)
        dt = (df.t - df.t.shift()).clip(lower=1)
        df = df[~(same & (dist / dt > max_speed_mps))]
    if labelled_only:
        df = df[df["mode"].notna() | df.traj_id.isin(df.loc[df["mode"].notna(), "traj_id"].unique())]
    return MobilityDataset(df.reset_index(drop=True), "geolife")


# --------------------------------------------------------------------------- #
# Datasets of the original papers (download instructions: DATASETS.md)
# --------------------------------------------------------------------------- #
def load_worldtrace(root: str, max_trajectories: Optional[int] = None, seed: int = 0, pattern: str = "**/*.csv",
                    use_matched: bool = False) -> MobilityDataset:
    """WorldTrace (huggingface.co/datasets/OpenTrace/WorldTrace, ODbL), UniTraj's pre-training data.

    `root` is either the downloaded folder of per-trajectory CSV files (columns `time`, `latitude`,
    `longitude`, ...; 1 s sampling) or a pickle in UniTraj's own format (a DataFrame with a `time`
    series and a `trajectory` array of (lat, lon) per row, like data/worldtrace_sample.pkl).
    There are no users: each trajectory is its own user. The full release is 2.45M trajectories
    (~35 GB); `max_trajectories` draws a random subset (the paper's curated 1.1M subset is not public)."""
    p = Path(root)
    rng = np.random.default_rng(seed)
    frames = []
    if p.is_file() and p.suffix in (".pkl", ".pickle"):
        df = pd.read_pickle(p)
        idx = np.arange(len(df))
        if max_trajectories and len(idx) > max_trajectories:
            idx = np.sort(rng.choice(idx, max_trajectories, replace=False))
        for i in idx:
            r = df.iloc[i]
            xy = np.asarray(r["trajectory"], float)
            frames.append(pd.DataFrame({"t": to_unix_seconds(pd.Series(r["time"]).to_numpy()),
                                        "lat": xy[:, 0], "lon": xy[:, 1], "traj_id": f"wt{i}"}))
    else:
        files = sorted(p.glob(pattern))
        if not files:
            raise FileNotFoundError(f"no WorldTrace CSV files matching {pattern} under {root}")
        if max_trajectories and len(files) > max_trajectories:
            files = [files[i] for i in np.sort(rng.choice(len(files), max_trajectories, replace=False))]
        lat_c, lon_c = ("matched_latitude", "matched_longitude") if use_matched else ("latitude", "longitude")
        for f in files:
            d = pd.read_csv(f, usecols=["time", lat_c, lon_c])
            frames.append(pd.DataFrame({"t": to_unix_seconds(d["time"]), "lat": d[lat_c], "lon": d[lon_c],
                                        "traj_id": f.stem}))
    df = pd.concat(frames, ignore_index=True).dropna(subset=["lat", "lon", "t"])
    df["user_id"] = df["traj_id"]
    return MobilityDataset(df, "worldtrace")


def load_porto(path: str, max_trajectories: Optional[int] = None, seed: int = 0, step_s: float = 15.0,
               drop_missing: bool = True) -> MobilityDataset:
    """Porto taxi trips (Kaggle 'pkdd-15-predict-taxi-service-trajectory-i', train.csv), one of
    TransferTraj's datasets. POLYLINE is a JSON list of [lon, lat] every 15 s from TIMESTAMP;
    trips flagged MISSING_DATA are dropped. user_id = TAXI_ID, traj_id = TRIP_ID."""
    import json
    df = pd.read_csv(path, usecols=["TRIP_ID", "TAXI_ID", "TIMESTAMP", "MISSING_DATA", "POLYLINE"])
    if drop_missing:
        df = df[~df.MISSING_DATA.astype(str).str.lower().eq("true")]
    if max_trajectories and len(df) > max_trajectories:
        df = df.sample(max_trajectories, random_state=seed)
    rows = []
    for trip, taxi, t0, poly in df[["TRIP_ID", "TAXI_ID", "TIMESTAMP", "POLYLINE"]].itertuples(index=False):
        pts = json.loads(poly)
        if not pts:
            continue
        a = np.asarray(pts, float)
        rows.append(pd.DataFrame({"user_id": str(taxi), "traj_id": str(trip), "t": float(t0) + step_s * np.arange(len(a)),
                                  "lon": a[:, 0], "lat": a[:, 1]}))
    return MobilityDataset(pd.concat(rows, ignore_index=True), "porto")


def load_transfertraj_h5(path: str, context_out: Optional[str] = None) -> MobilityDataset:
    """TransferTraj's processed DiDi files (samples/small_chengdu.h5 and the full Chengdu / Xi'an
    releases in the same layout): HDF5 with /trips (trip, seq_i, time, lng, lat, ...), /trip_info
    (trip, driver, ...), /pois (lng, lat, ...) and /road_info (road_lng, road_lat, ...).
    With `context_out`, also writes poi_latlon.npy and road_latlon.npy there, row-aligned with the
    repository's *_poi_embed.npy / *_road_embed.npy, ready for the adapter's `context:` option."""
    with pd.HDFStore(path, "r") as st:
        trips = st["trips"]
        info = st["trip_info"] if "/trip_info" in st.keys() else None
        if context_out:
            out = Path(context_out)
            out.mkdir(parents=True, exist_ok=True)
            if "/pois" in st.keys():
                np.save(out / "poi_latlon.npy", st["pois"][["lat", "lng"]].to_numpy(float))
            if "/road_info" in st.keys():
                np.save(out / "road_latlon.npy", st["road_info"][["road_lat", "road_lng"]].to_numpy(float))
    df = trips.sort_values(["trip", "seq_i"]).rename(columns={"lng": "lon"})
    df["t"] = to_unix_seconds(df["time"])
    driver = info.set_index("trip")["driver"] if info is not None and "driver" in info else None
    df["user_id"] = (df["trip"].map(driver) if driver is not None else df["trip"]).astype(str)
    df["traj_id"] = df["trip"].astype(str)
    return MobilityDataset(df[["user_id", "traj_id", "t", "lat", "lon"]], Path(path).stem)


def prepare(ds: MobilityDataset, min_interval_s: Optional[float] = None, every_nth: Optional[int] = None,
            min_traj_points: Optional[int] = None, max_traj_points: Optional[int] = None,
            time_from: Optional[str] = None, time_to: Optional[str] = None,
            min_user_trajectories: Optional[int] = None) -> MobilityDataset:
    """Per-dataset preprocessing steps the papers describe, applied after loading (order as listed):

    time_from / time_to    keep points in [time_from, time_to] (e.g. TrajGPT: GeoLife 2007-2008)
    every_nth              keep every n-th point of each trajectory (TransferTraj's "three-hop
                           resampling" of the DiDi data: every_nth: 3)
    min_interval_s         keep a point only if at least this long after the last kept one
                           (UniTraj's evaluation data at 3 s: min_interval_s: 3)
    min/max_traj_points    drop trajectories outside this length (TransferTraj: 5..120)
    min_user_trajectories  drop users with fewer trajectories"""
    p = ds.points
    if time_from is not None:
        p = p[p.t >= to_unix_seconds(pd.Series([time_from])).iloc[0]]
    if time_to is not None:
        p = p[p.t <= to_unix_seconds(pd.Series([time_to])).iloc[0]]
    p = p.sort_values(["traj_id", "t"], kind="stable")
    if every_nth and every_nth > 1:
        p = p[p.groupby("traj_id").cumcount() % int(every_nth) == 0]
    if min_interval_s:
        keep = np.zeros(len(p), bool)
        tt, tid = p.t.to_numpy(), p.traj_id.to_numpy()
        last_t, last_id = -np.inf, None
        for i in range(len(p)):
            if tid[i] != last_id or tt[i] - last_t >= min_interval_s:
                keep[i], last_t, last_id = True, tt[i], tid[i]
        p = p[keep]
    if min_traj_points or max_traj_points:
        n = p.groupby("traj_id").t.transform("size")
        p = p[(n >= (min_traj_points or 0)) & (n <= (max_traj_points or np.inf))]
    if min_user_trajectories:
        k = p.groupby("user_id").traj_id.transform("nunique")
        p = p[k >= min_user_trajectories]
    if p.empty:
        raise ValueError("preprocessing removed every point; check the prepare: settings")
    return MobilityDataset(p.reset_index(drop=True), ds.name)
