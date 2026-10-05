#!/usr/bin/env python
"""How was UniTraj's WorldTrace data derived from the public WorldTrace files?

Two things, in one pass over the raw per-trajectory CSV files:

  1. a one-line profile of every raw file (length, sampling, speed, map-matching quality), so the
     raw distribution can be compared with the UniTraj repository sample (same columns, written for
     the sample too);
  2. an exact search for the repository sample's trajectories inside the raw files: a sample
     trajectory is found when its first point (timestamp, latitude, longitude at 6 decimals)
     occurs in a raw file, in its matched or in its raw coordinates. The offset says whether the
     sample trajectory is a whole file or a slice of one.

    python tools/worldtrace_profile.py --raw /scratch/$USER/data/WorldTrace \
        --sample /path/to/UniTraj/data/worldtrace_sample.pkl --out /scratch/$USER/wt_profile --workers 32

Writes raw_profile.csv, sample_profile.csv and matches.csv into --out. Run it as a PBS job: the
full dataset is 2.45M files. `--max-files N` profiles a random subset (fine for the distributions;
the exact search needs every file, since the sample is 2,000 trajectories out of millions).
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

R = 6_371_008.8
SAMPLE_KEYS: dict = {}          # (timestamp, lat6, lon6) of each sample trajectory's first point -> row


def _path_m(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    h = np.sin(np.diff(la) / 2) ** 2 + np.cos(la[:-1]) * np.cos(la[1:]) * np.sin(np.diff(lo) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(h))


def _decimals(v) -> int:
    s = repr(float(v))
    return len(s.split(".")[1]) if "." in s else 0


def profile(t, lat, lon) -> dict:
    """Statistics of one trajectory: t in seconds, coordinates in degrees."""
    step = _path_m(lat, lon) if len(lat) > 1 else np.array([0.0])
    dt = np.diff(t) if len(t) > 1 else np.array([0.0])
    dur = float(t[-1] - t[0])
    return {"n_points": len(t), "duration_s": dur, "dt_median_s": float(np.median(dt)), "dt_max_s": float(dt.max()),
            "path_m": float(step.sum()), "mean_speed_kmh": float(step.sum() / dur * 3.6) if dur > 0 else np.nan,
            "max_step_mps": float((step / np.maximum(dt, 1e-9)).max()), "share_still": float((step < 0.5).mean()),
            "share_over_6_decimals": float(np.mean([max(_decimals(a), _decimals(b)) > 6 for a, b in zip(lat, lon)])),
            "lat0": float(lat[0]), "lon0": float(lon[0])}


def _init(keys):
    global SAMPLE_KEYS
    SAMPLE_KEYS = keys


def one_file(path: str):
    try:
        df = pd.read_csv(path)
        ts = pd.to_datetime(df["time"], format="ISO8601")
        t = ts.to_numpy().astype("datetime64[s]").astype(np.int64).astype(float)
        has_m = "matched_latitude" in df
        lat = df["matched_latitude" if has_m else "latitude"].to_numpy(float)
        lon = df["matched_longitude" if has_m else "longitude"].to_numpy(float)
        row = {"file": path, "start": str(ts.iloc[0]), **profile(t, lat, lon)}
        if has_m:
            row["raw_vs_matched_m_mean"] = float(df["matched_distance"].mean()) if "matched_distance" in df else np.nan
            row["matched_type_true"] = float(df["matched_type"].astype(str).str.lower().eq("true").mean()) \
                if "matched_type" in df else np.nan
            row["matched_missing"] = float(df["matched_latitude"].isna().mean())
        hits = []
        if SAMPLE_KEYS:
            tt = ts.dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy()
            for which, la, lo in (("matched", "matched_latitude", "matched_longitude"), ("raw", "latitude", "longitude")):
                if la not in df:
                    continue
                a, b = np.round(df[la].to_numpy(float), 6), np.round(df[lo].to_numpy(float), 6)
                for i in range(len(df)):
                    k = SAMPLE_KEYS.get((tt[i], a[i], b[i]))
                    if k is not None:
                        hits.append({"sample_row": k, "file": path, "coords": which, "offset_in_file": i,
                                     "file_points": len(df)})
        return row, hits
    except Exception as e:                                           # noqa: BLE001 - one bad file must not stop the scan
        return {"file": path, "error": repr(e)}, []


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raw", help="folder with the WorldTrace per-trajectory CSV files (searched recursively)")
    ap.add_argument("--sample", help="UniTraj's data/worldtrace_sample.pkl (or a larger file in the same format)")
    ap.add_argument("--out", default="wt_profile")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--max-files", type=int, help="profile a random subset of the raw files")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    keys, sample_len = {}, {}
    if a.sample:
        s = pd.read_pickle(a.sample)
        rows = []
        for idx, (tm, tr) in zip(s.index, zip(s["time"], s["trajectory"])):
            tr = np.asarray(tr, float)
            t = pd.to_datetime(pd.Series(tm).reset_index(drop=True))
            rows.append({"sample_row": idx, "start": str(t.iloc[0]),
                         **profile(t.to_numpy().astype("datetime64[s]").astype(np.int64).astype(float), tr[:, 0], tr[:, 1])})
            keys[(t.iloc[0].strftime("%Y-%m-%d %H:%M:%S"), round(tr[0, 0], 6), round(tr[0, 1], 6))] = idx
            sample_len[idx] = len(tr)
        pd.DataFrame(rows).to_csv(out / "sample_profile.csv", index=False)
        print(f"sample: {len(rows):,} trajectories -> {out / 'sample_profile.csv'}")

    if a.raw:
        files = sorted(str(p) for p in Path(a.raw).rglob("*.csv"))
        if a.max_files and len(files) > a.max_files:
            files = list(np.random.default_rng(a.seed).choice(files, a.max_files, replace=False))
        print(f"raw: {len(files):,} files, {a.workers} workers")
        prof, hits = [], []
        with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(keys,)) as ex:
            for n, (row, h) in enumerate(ex.map(one_file, files, chunksize=256), 1):
                prof.append(row); hits += h
                if n % 100_000 == 0:
                    print(f"  {n:,} files, {len(hits)} sample trajectories found", flush=True)
        pd.DataFrame(prof).to_csv(out / "raw_profile.csv", index=False)
        m = pd.DataFrame(hits)
        if len(m):
            m["sample_points"] = m.sample_row.map(sample_len)
            m["whole_file"] = (m.offset_in_file == 0) & (m.sample_points == m.file_points)
        m.to_csv(out / "matches.csv", index=False)
        print(f"-> {out / 'raw_profile.csv'}, {out / 'matches.csv'}")
        if keys:
            found = m.sample_row.nunique() if len(m) else 0
            print(f"sample trajectories found in the raw files: {found} of {len(keys)}")
            if len(m):
                print(m.groupby("coords").agg(n=("sample_row", "nunique"), whole_file=("whole_file", "mean"),
                                              median_offset=("offset_in_file", "median")).to_string())


if __name__ == "__main__":
    main()
