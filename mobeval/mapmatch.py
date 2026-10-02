"""Map matching: one road-segment id per GPS point, for OmniTraj's road modality.

The OmniTraj paper does not say which matcher produced its road sequences. Two routes are
supported, and both write the same table (traj_id, t, seg; seg = -1 where unmatched):

* Built-in HMM matcher (Newson & Krumm, 2009), no extra install:
      mobeval mapmatch --config my.yaml --roads roads/rome.npz --out roads/rome_matched.parquet --workers 16
  Emission: Gaussian in the point-to-road distance (sigma_m). Transition: exponential in the
  difference between the network route length and the great-circle distance of consecutive points
  (beta_m). Candidates within radius_m, at most max_candidates segments per point. When no route
  connects two consecutive points (a gap, a tunnel, a missing road), the chain restarts there
  instead of failing the whole trajectory.
* FMM (github.com/cyang-kth/fmm), much faster at country scale:
      mobeval roads ... --fmm roads/rome_fmm.gpkg        # network in FMM's format
      mobeval mapmatch --config my.yaml --fmm-export gps.csv
      fmm ... --network roads/rome_fmm.gpkg --gps gps.csv --output fmm_out.csv --output_fields opath
      mobeval mapmatch --config my.yaml --fmm-import fmm_out.csv --fmm-network roads/rome_fmm.gpkg --out matched.parquet
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .geo import LocalProjection
from .roads import RoadNetwork

log = logging.getLogger("mobeval.mapmatch")


class HMMMatcher:
    def __init__(self, net: RoadNetwork, sigma_m: float = 20.0, beta_m: float = 50.0, radius_m: float = 60.0,
                 max_candidates: int = 8, margin_m: float = 1500.0):
        from scipy import sparse
        from shapely import STRtree, linestrings
        self.net, self.sigma, self.beta, self.radius, self.K, self.margin = net, sigma_m, beta_m, radius_m, max_candidates, margin_m
        self.proj = LocalProjection(float(np.mean(net.node_lat)), float(np.mean(net.node_lon)))
        gx, gy = self.proj.to_xy(net.geom_lat, net.geom_lon)
        ptr = net.geom_ptr
        # pieces = consecutive vertex pairs of every segment
        a = np.concatenate([np.arange(ptr[i], ptr[i + 1] - 1) for i in range(len(net))]) if len(net) else np.array([], int)
        self.piece_seg = np.repeat(np.arange(len(net)), np.diff(ptr) - 1)
        self.p0 = np.column_stack([gx[a], gy[a]])
        self.p1 = np.column_stack([gx[a + 1], gy[a + 1]])
        plen = np.hypot(*(self.p1 - self.p0).T)
        # offset (metres along the segment, in geometry order) at the start of each piece
        first = np.r_[True, self.piece_seg[1:] != self.piece_seg[:-1]]
        cum = np.cumsum(plen) - plen
        start = np.maximum.accumulate(np.where(first, cum, 0))
        self.piece_off = cum - start
        self.piece_len = plen
        self.seg_len = np.bincount(self.piece_seg, plen, minlength=len(net))
        self.tree = STRtree(linestrings(np.stack([self.p0, self.p1], 1)))
        # directed travel edges (u -> v, and v -> u unless one-way); weights are projected lengths, so they
        # are consistent with the offsets along segments
        two = ~net.oneway
        self.e_seg = np.r_[np.arange(len(net)), np.where(two)[0]]
        self.e_from = np.r_[net.u, net.v[two]]
        self.e_to = np.r_[net.v, net.u[two]]
        self._sparse = sparse

    # ------------------------------------------------------------------ candidates
    def _candidates(self, xy):
        from shapely import points
        pi, ci = self.tree.query(points(xy), predicate="dwithin", distance=self.radius)
        out = [[] for _ in range(len(xy))]
        if len(pi) == 0:
            return out
        p0, d = self.p0[ci], self.p1[ci] - self.p0[ci]
        L2 = np.maximum((d ** 2).sum(1), 1e-12)
        t = np.clip(((xy[pi] - p0) * d).sum(1) / L2, 0, 1)
        proj = p0 + t[:, None] * d
        dist = np.hypot(*(xy[pi] - proj).T)
        seg = self.piece_seg[ci]
        off = self.piece_off[ci] + t * self.piece_len[ci]
        df = pd.DataFrame({"p": pi, "seg": seg, "dist": dist, "off": off}).sort_values(["p", "dist"])
        df = df.drop_duplicates(["p", "seg"])
        for p, g in df.groupby("p", sort=False):
            out[p] = list(g.head(self.K)[["seg", "dist", "off"]].itertuples(index=False, name=None))
        return out

    def _states(self, cands):
        """(seg, forward, position along travel direction, distance) per point."""
        st = []
        for c in cands:
            s = []
            for seg, dist, off in c:
                s.append((seg, True, off, dist))
                if not self.net.oneway[seg]:
                    s.append((seg, False, self.seg_len[seg] - off, dist))
            st.append(s)
        return st

    def _subgraph(self, xy, margin):
        """Directed graph over the segments near the trajectory's bounding box (+ margin). Parallel
        segments between the same two nodes keep the SHORTER length: a sparse matrix would add them."""
        from shapely import box
        lo, hi = xy.min(0) - margin, xy.max(0) + margin
        segs = np.unique(self.piece_seg[self.tree.query(box(lo[0], lo[1], hi[0], hi[1]))])
        keep = np.isin(self.e_seg, segs)
        f, t, w = self.e_from[keep], self.e_to[keep], self.seg_len[self.e_seg[keep]] + 1e-6
        nodes = np.unique(np.r_[f, t, self.net.u[segs], self.net.v[segs]])
        local = np.full(len(self.net.node_lat), -1, np.int64)
        local[nodes] = np.arange(len(nodes))
        df = pd.DataFrame({"f": local[f], "t": local[t], "w": w}).groupby(["f", "t"], as_index=False).w.min()
        G = self._sparse.csr_matrix((df.w.to_numpy(), (df.f.to_numpy(), df.t.to_numpy())), shape=(len(nodes), len(nodes)))
        return G, local

    # ------------------------------------------------------------------ matching
    def match(self, lat, lon) -> np.ndarray:
        """Segment id per point (-1 where no road is within radius_m)."""
        from scipy.sparse.csgraph import dijkstra
        lat, lon = np.asarray(lat, float), np.asarray(lon, float)
        n = len(lat)
        out = np.full(n, -1, np.int64)
        if n == 0 or len(self.net) == 0:
            return out
        xy = np.column_stack(self.proj.to_xy(lat, lon))
        states = self._states(self._candidates(xy))
        obs = [i for i in range(n) if states[i]]
        if not obs:
            return out
        exit_node = lambda s: self.net.v[s[0]] if s[1] else self.net.u[s[0]]
        entry_node = lambda s: self.net.u[s[0]] if s[1] else self.net.v[s[0]]
        gcs = [np.hypot(*(xy[b] - xy[a])) for a, b in zip(obs[:-1], obs[1:])]
        limit = 3.0 * (max(gcs) if gcs else 0.0) + 2 * self.radius + self.margin
        # Route search on the part of the network around this trajectory only: a dense (sources x all
        # nodes) distance matrix over a whole city would cost gigabytes per trajectory.
        G, local = self._subgraph(xy, self.margin)
        sources = np.unique([exit_node(s) for i in obs for s in states[i]])
        D = dijkstra(G, directed=True, indices=local[sources], limit=limit)
        row = {int(s): k for k, s in enumerate(sources)}
        entry_col = lambda s: local[int(entry_node(s))]
        emis = lambda s: -0.5 * (s[3] / self.sigma) ** 2
        # Viterbi; a step no route reaches starts a new chain (scores[k] kept for backtracking)
        scores = [np.array([emis(s) for s in states[obs[0]]])]
        back = [None]
        for k in range(1, len(obs)):
            prev, cur, gc = states[obs[k - 1]], states[obs[k]], gcs[k - 1]
            T = np.full((len(prev), len(cur)), -np.inf)
            for i, a in enumerate(prev):
                for j, b in enumerate(cur):
                    if a[0] == b[0] and a[1] == b[1] and b[2] >= a[2] - 1.0:
                        route = b[2] - a[2]
                    else:
                        d = D[row[int(exit_node(a))], entry_col(b)]
                        if not np.isfinite(d):
                            continue
                        route = (self.seg_len[a[0]] - a[2]) + d + b[2]
                    T[i, j] = -abs(route - gc) / self.beta
            tot = scores[-1][:, None] + T
            bi = tot.argmax(0)
            best = tot[bi, np.arange(len(cur))]
            e = np.array([emis(s) for s in cur])
            if np.isfinite(best).any():
                back.append(bi)
                scores.append(best + e)
            else:                                              # unreachable: restart the chain
                back.append(None)
                scores.append(e)
        j = int(np.argmax(scores[-1]))
        for k in range(len(obs) - 1, -1, -1):
            out[obs[k]] = states[obs[k]][j][0]
            if k == 0:
                break
            j = int(back[k][j]) if back[k] is not None else int(np.argmax(scores[k - 1]))
        return out


# ---------------------------------------------------------------------- tables
def write_table(df: pd.DataFrame, path: str):
    """.parquet (needs pyarrow) or .csv / .csv.gz."""
    if str(path).endswith(".parquet"):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def read_table(path: str) -> pd.DataFrame:
    if str(path).endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype={"traj_id": str})


# ---------------------------------------------------------------------- whole datasets
_MATCHER = None


def _match_one(args):
    tid, lat, lon, t = args
    seg = _MATCHER.match(lat, lon)
    return pd.DataFrame({"traj_id": tid, "t": t, "seg": seg})


def match_dataset(ds, net: RoadNetwork, workers: int = 1, max_trajectories: Optional[int] = None, seed: int = 0,
                  **matcher_kw) -> pd.DataFrame:
    """Match every trajectory of a MobilityDataset. Returns (traj_id, t, seg)."""
    global _MATCHER
    _MATCHER = HMMMatcher(net, **matcher_kw)
    p = ds.points.sort_values(["traj_id", "t"], kind="stable")
    groups = [(tid, g.lat.to_numpy(), g.lon.to_numpy(), g.t.to_numpy()) for tid, g in p.groupby("traj_id", sort=False)]
    if max_trajectories and len(groups) > max_trajectories:
        keep = np.random.default_rng(seed).choice(len(groups), max_trajectories, replace=False)
        groups = [groups[i] for i in np.sort(keep)]
    log.info(f"map matching {len(groups):,} trajectories on {len(net):,} segments with {workers} worker(s)")
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers) as pool:
            parts = pool.map(_match_one, groups, chunksize=16)
    else:
        parts = [_match_one(g) for g in groups]
    out = pd.concat(parts, ignore_index=True)
    log.info(f"matched {float((out.seg >= 0).mean()):.1%} of {len(out):,} points")
    return out


# ---------------------------------------------------------------------- FMM interoperability
def fmm_export(ds, path: str) -> pd.DataFrame:
    """GPS points in FMM's point-CSV format (id;x;y;timestamp). Returns the id -> traj_id mapping,
    also written next to `path` as <path>.ids.csv."""
    p = ds.points.sort_values(["traj_id", "t"], kind="stable")
    ids = pd.DataFrame({"traj_id": p.traj_id.unique()})
    ids["id"] = np.arange(len(ids))
    q = p.merge(ids, on="traj_id")
    q[["id", "lon", "lat", "t"]].rename(columns={"lon": "x", "lat": "y", "t": "timestamp"}).to_csv(path, sep=";", index=False)
    ids.to_csv(str(path) + ".ids.csv", index=False)
    return ids


def fmm_import(ds, result_csv: str, network_gpkg: str) -> pd.DataFrame:
    """FMM output (with the `opath` field: one edge per point) -> (traj_id, t, seg). FMM ids are the
    trajectory numbers fmm_export assigned, which depend only on the dataset, so the same config must
    be used for export and import."""
    import geopandas as gpd
    edges = gpd.read_file(network_gpkg, ignore_geometry=True).set_index("id")["segment"]
    p = ds.points.sort_values(["traj_id", "t"], kind="stable")
    tids = p.traj_id.unique()
    groups = {k: g.t.to_numpy() for k, g in p.groupby("traj_id", sort=False)}
    res = pd.read_csv(result_csv, sep=";")
    rows, bad = [], 0
    for r in res.itertuples():
        tid = tids[int(r.id)]
        t = groups[tid]
        op = [int(x) for x in str(r.opath).split(",") if x not in ("", "nan")]
        seg = np.full(len(t), -1, np.int64)
        if len(op) == len(t):
            seg = edges.reindex(op).fillna(-1).astype(np.int64).to_numpy()
        else:
            bad += 1
        rows.append(pd.DataFrame({"traj_id": tid, "t": t, "seg": seg}))
    if bad:
        log.warning(f"{bad} FMM results had an opath of the wrong length and were left unmatched")
    return pd.concat(rows, ignore_index=True)
