#!/usr/bin/env python
"""Sample trajectories straight out of WorldTrace's Trajectory.zip into ONE file, without unzipping.

The download is a single 27 GB archive of 2.45M small CSV files; extracting it takes hours and leaves
millions of files behind. This reads only the sampled members from inside the archive and writes a
single .npz that `loader: worldtrace` opens directly.

    python tools/worldtrace_subset.py --zip /home/$USER/data/WorldTrace/Trajectory.zip \
        --n 300000 --out /scratch/$USER/worldtrace_300k.npz --workers 16

The file holds, concatenated over trajectories: t (unix seconds), latitude / longitude (raw GPS),
matched_latitude / matched_longitude (map-matched), plus `length` and `names` per trajectory.
That is 40 bytes per point: about 4.5 GB for 300,000 trajectories. Run it as a job (jobs/worldtrace.pbs).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mobeval.loaders import WT_COLUMNS, read_worldtrace_csv, zip_index, zip_read  # noqa: E402

KEYS = ("t",) + WT_COLUMNS


def read_chunk(args):
    path, members, min_points = args
    out = {k: [] for k in KEYS}
    names, lengths, bad = [], [], 0
    with open(path, "rb") as fh:
        for name, off, size, method in members:
            try:
                d = read_worldtrace_csv(zip_read(fh, off, size, method))
            except Exception:                                       # noqa: BLE001 - skip a broken member
                bad += 1
                continue
            if len(d["t"]) < min_points:
                continue
            for k in KEYS:
                out[k].append(d[k])
            names.append(name)
            lengths.append(len(d["t"]))
    cat = {k: (np.concatenate(v) if v else np.empty(0)) for k, v in out.items()}
    return cat, names, lengths, bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--zip", required=True, help="WorldTrace's Trajectory.zip")
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--n", type=int, default=300_000, help="trajectories to sample (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-points", type=int, default=0, help="skip shorter trajectories")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    a = ap.parse_args()

    t0 = time.time()
    members = zip_index(a.zip)
    print(f"{len(members):,} trajectory files in {a.zip} (index read in {time.time() - t0:.0f}s)", flush=True)
    if a.n and len(members) > a.n:
        pick = np.sort(np.random.default_rng(a.seed).choice(len(members), a.n, replace=False))
        members = [members[i] for i in pick]
    chunks = [(a.zip, members[i:i + 2000], a.min_points) for i in range(0, len(members), 2000)]
    parts, names, lengths, bad = {k: [] for k in KEYS}, [], [], 0
    with ProcessPoolExecutor(a.workers) as ex:
        for n, (cat, nm, ln, b) in enumerate(ex.map(read_chunk, chunks), 1):
            for k in KEYS:
                parts[k].append(cat[k])
            names += nm; lengths += ln; bad += b
            if n % 10 == 0 or n == len(chunks):
                print(f"  {min(n * 2000, len(members)):,} / {len(members):,} files, {sum(lengths):,} points "
                      f"({time.time() - t0:.0f}s)", flush=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, names=np.array(names), length=np.array(lengths, np.int32),
             **{k: np.concatenate(parts[k]) for k in KEYS})
    print(f"wrote {out}: {len(names):,} trajectories, {sum(lengths):,} points, "
          f"{out.stat().st_size / 1e9:.2f} GB" + (f"; {bad} unreadable files skipped" if bad else ""))


if __name__ == "__main__":
    main()
