"""Road network for map matching (OmniTraj's road modality), built from an OpenStreetMap extract.

    mobeval roads --pbf italy-latest.osm.pbf --bbox 41.75 42.05 12.30 12.70 --out roads/rome.npz \
                  [--fmm roads/rome_fmm.gpkg]

Segments follow OmniTraj's Definition 3: "a continuous stretch of road between two intersections".
Drivable OSM ways are cut at every node shared by two or more ways (and at way ends), and each piece
gets a segment id. One-way streets keep their direction; everything else can be travelled both ways.

Run it as a PBS job for a whole country: it reads the .pbf once, keeping node locations in memory
(pyosmium's flex_mem index). `--bbox` keeps only ways with at least one node inside.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

log = logging.getLogger("mobeval.roads")

DRIVABLE = {"motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
            "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
            "living_street", "road"}


@dataclass
class RoadNetwork:
    """Segments between intersections. Node ids are compact (0..N-1)."""
    seg_id: np.ndarray        # (S,) int64
    u: np.ndarray             # (S,) start node
    v: np.ndarray             # (S,) end node
    oneway: np.ndarray        # (S,) bool: only u -> v
    length_m: np.ndarray      # (S,)
    geom_ptr: np.ndarray      # (S+1,) offsets into geom_lat / geom_lon
    geom_lat: np.ndarray
    geom_lon: np.ndarray
    node_lat: np.ndarray      # (N,)
    node_lon: np.ndarray

    def __len__(self):
        return len(self.seg_id)

    def geometry(self, i: int):
        s, e = self.geom_ptr[i], self.geom_ptr[i + 1]
        return self.geom_lat[s:e], self.geom_lon[s:e]

    def save(self, path: str):
        np.savez_compressed(path, **{k: getattr(self, k) for k in self.__dataclass_fields__})

    @classmethod
    def load(cls, path: str) -> "RoadNetwork":
        z = np.load(path)
        return cls(**{k: z[k] for k in cls.__dataclass_fields__})

    @classmethod
    def from_polylines(cls, lines: Sequence[np.ndarray], oneway: Optional[Sequence[bool]] = None) -> "RoadNetwork":
        """Build from polylines of (lat, lon) rows, cutting at shared vertices (exact coordinate
        equality). Used by tests and for networks from other sources."""
        return _segment([np.asarray(l, float) for l in lines], None,
                        list(oneway) if oneway is not None else [False] * len(lines))

    def to_fmm(self, path: str):
        """GeoPackage for FMM (github.com/cyang-kth/fmm): one directed edge per travel direction, with
        FMM's `id`, `source`, `target` fields and `segment`, the id FMM results are mapped back to."""
        import geopandas as gpd
        from shapely.geometry import LineString
        rows = []
        eid = 0
        for i in range(len(self)):
            la, lo = self.geometry(i)
            pts = list(zip(lo, la))
            rows.append((eid, int(self.u[i]), int(self.v[i]), int(self.seg_id[i]), LineString(pts)))
            eid += 1
            if not self.oneway[i]:
                rows.append((eid, int(self.v[i]), int(self.u[i]), int(self.seg_id[i]), LineString(pts[::-1])))
                eid += 1
        g = gpd.GeoDataFrame(rows, columns=["id", "source", "target", "segment", "geometry"], crs="EPSG:4326")
        g.to_file(path, driver="GPKG")
        log.info(f"wrote {len(g):,} directed edges for FMM to {path}")


def _length_m(lat, lon) -> float:
    from .geo import haversine_m
    return float(np.sum(haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:]))) if len(lat) > 1 else 0.0


def _segment(ways: Sequence[np.ndarray], refs: Optional[Sequence[np.ndarray]], oneway: Sequence[bool]) -> RoadNetwork:
    """Cut ways at intersections. `refs` are OSM node ids (None: vertices identified by coordinates)."""
    keys = refs if refs is not None else [[(round(a, 9), round(b, 9)) for a, b in w] for w in ways]
    count = {}
    for k in keys:
        for j, n in enumerate(k):
            count[n] = count.get(n, 0) + (2 if j in (0, len(k) - 1) else 1)
    node_index, node_ll = {}, []

    def nid(key, ll):
        if key not in node_index:
            node_index[key] = len(node_ll)
            node_ll.append(ll)
        return node_index[key]

    U, V, OW, L, ptr, glat, glon = [], [], [], [], [0], [], []
    for w, k, ow in zip(ways, keys, oneway):
        if len(w) < 2:
            continue
        cut = [0] + [j for j in range(1, len(k) - 1) if count[k[j]] >= 2] + [len(k) - 1]
        for a, b in zip(cut[:-1], cut[1:]):
            piece = w[a:b + 1]
            U.append(nid(k[a], w[a])); V.append(nid(k[b], w[b])); OW.append(bool(ow))
            L.append(_length_m(piece[:, 0], piece[:, 1]))
            glat.extend(piece[:, 0]); glon.extend(piece[:, 1]); ptr.append(len(glat))
    nl = np.asarray(node_ll, float).reshape(-1, 2)
    S = len(U)
    return RoadNetwork(np.arange(S, dtype=np.int64), np.asarray(U, np.int64), np.asarray(V, np.int64),
                       np.asarray(OW, bool), np.asarray(L, float), np.asarray(ptr, np.int64),
                       np.asarray(glat, float), np.asarray(glon, float), nl[:, 0], nl[:, 1])


def from_osm(pbf: str, bbox: Optional[Sequence[float]] = None, highways=DRIVABLE) -> RoadNetwork:
    """Drivable network from an .osm.pbf (or .osm) file. bbox = (lat_min, lat_max, lon_min, lon_max)."""
    import osmium
    ways, refs, oneway = [], [], []
    la0, la1, lo0, lo1 = bbox if bbox is not None else (-90, 90, -180, 180)

    class H(osmium.SimpleHandler):
        def way(self, w):
            hw = w.tags.get("highway")
            if hw not in highways:
                return
            try:
                ll = np.array([(n.location.lat, n.location.lon) for n in w.nodes], float)
            except osmium.InvalidLocationError:
                return
            if len(ll) < 2:
                return
            inside = (ll[:, 0] >= la0) & (ll[:, 0] <= la1) & (ll[:, 1] >= lo0) & (ll[:, 1] <= lo1)
            if not inside.any():
                return
            r = np.array([n.ref for n in w.nodes], np.int64)
            ow = w.tags.get("oneway", "")
            forward = (ow in ("yes", "true", "1") or hw in ("motorway", "motorway_link")
                       or w.tags.get("junction") == "roundabout")
            if ow == "-1":
                ll, r, forward = ll[::-1], r[::-1], True
            ways.append(ll); refs.append(r.tolist()); oneway.append(forward and ow != "no")

    H().apply_file(pbf, locations=True, idx="flex_mem")
    if not ways:
        raise ValueError(f"no drivable ways found in {pbf} within {bbox}")
    net = _segment(ways, refs, oneway)
    log.info(f"{len(ways):,} ways -> {len(net):,} segments between intersections, {len(net.node_lat):,} nodes")
    return net
