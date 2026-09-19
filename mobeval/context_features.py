"""Build the POI and road-network features that TransferTraj can use, from OpenStreetMap.

Produces four arrays, which is the format every mobeval adapter expects for context features:

    poi_embed.npy   (N_poi, d)    one embedding row per POI
    poi_latlon.npy  (N_poi, 2)    (lat, lon) of each POI, WGS84 degrees
    road_embed.npy  (N_road, d)   one embedding row per road sample point
    road_latlon.npy (N_road, 2)   (lat, lon) of each road sample point

Two ways in, because compute nodes usually have no internet:

  * live        `from_osm(...)` downloads with osmnx (Overpass). Run it on a login node or laptop.
  * from files  `pois_from_file(...)` / `roads_from_file(...)` read a GeoJSON or CSV you exported
                beforehand (Overpass Turbo, QGIS, a Geofabrik extract processed with osmium/pyrosm).

Embeddings are one-hot over the most frequent OSM categories, so nothing has to be downloaded or
trained. The model puts a LayerNorm + Linear in front of them, so any real-valued matrix works:
supply your own (text embeddings of POI names, for instance) by writing the .npy files yourself.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .geo import LocalProjection, haversine_m

log = logging.getLogger("mobeval.context")

# OSM tag keys searched for a category, in order of preference
POI_TAG_KEYS = ("amenity", "shop", "tourism", "leisure", "office", "healthcare", "public_transport",
                "railway", "aeroway", "craft", "historic", "emergency")
DEFAULT_POI_TAGS = {k: True for k in ("amenity", "shop", "tourism", "leisure", "office", "public_transport")}
ROAD_TAG_KEYS = ("highway",)
COLUMNS = ["lat", "lon", "category"]


# --------------------------------------------------------------------------- bounding box
def dataset_bbox(ds, pad_km: float = 2.0) -> Tuple[float, float, float, float]:
    """(lat_min, lat_max, lon_min, lon_max) covering a MobilityDataset, padded by `pad_km`."""
    p = ds.points
    dlat = pad_km / 111.195
    dlon = pad_km / (111.195 * max(np.cos(np.radians(float(p.lat.mean()))), 0.01))
    return (float(p.lat.min()) - dlat, float(p.lat.max()) + dlat,
            float(p.lon.min()) - dlon, float(p.lon.max()) + dlon)


def bbox_area_km2(bbox) -> float:
    lat_min, lat_max, lon_min, lon_max = bbox
    return (haversine_m(lat_min, lon_min, lat_max, lon_min) / 1000) * (
        haversine_m(lat_min, lon_min, lat_min, lon_max) / 1000)


# --------------------------------------------------------------------------- OSM (live)
def pois_from_osm(bbox, tags: Optional[dict] = None, timeout: int = 300) -> pd.DataFrame:
    """Download POIs with osmnx. `bbox` is (lat_min, lat_max, lon_min, lon_max)."""
    ox = _osmnx(timeout)
    lat_min, lat_max, lon_min, lon_max = bbox
    gdf = ox.features_from_bbox(bbox=(lon_min, lat_min, lon_max, lat_max), tags=tags or DEFAULT_POI_TAGS)
    return _gdf_to_points(gdf, POI_TAG_KEYS)


def roads_from_osm(bbox, network_type: str = "drive", max_spacing_m: float = 100.0,
                   timeout: int = 300) -> pd.DataFrame:
    """Download the road network with osmnx and turn each edge into one or more sample points."""
    ox = _osmnx(timeout)
    lat_min, lat_max, lon_min, lon_max = bbox
    g = ox.graph_from_bbox(bbox=(lon_min, lat_min, lon_max, lat_max), network_type=network_type, simplify=True)
    edges = ox.graph_to_gdfs(g, nodes=False)
    rows = []
    for geom, hw in zip(edges.geometry, edges.get("highway", ["road"] * len(edges))):
        cat = f"highway={_first(hw)}"
        for lon, lat in _sample_line(list(geom.coords), max_spacing_m):
            rows.append((lat, lon, cat))
    return pd.DataFrame(rows, columns=COLUMNS)


def _osmnx(timeout):
    try:
        import osmnx as ox
    except ImportError as e:                                    # pragma: no cover - depends on env
        raise ImportError("live OSM download needs osmnx (`pip install osmnx`). Without it, export the "
                          "data once (Overpass Turbo / QGIS / a Geofabrik extract) and use "
                          "pois_from_file / roads_from_file.") from e
    ox.settings.requests_timeout = timeout
    return ox


def _gdf_to_points(gdf, tag_keys) -> pd.DataFrame:
    """Geometry -> representative point (centroid for ways/areas), tags -> 'key=value' category."""
    rows = []
    for geom, props in zip(gdf.geometry, gdf.to_dict("records")):
        if geom is None or geom.is_empty:
            continue
        pt = geom if geom.geom_type == "Point" else geom.centroid
        rows.append((float(pt.y), float(pt.x), _category(props, tag_keys)))
    return pd.DataFrame(rows, columns=COLUMNS)


# --------------------------------------------------------------------------- files (offline)
def pois_from_file(path, category_keys: Sequence[str] = POI_TAG_KEYS, **kw) -> pd.DataFrame:
    return _from_file(path, category_keys, **kw)


def roads_from_file(path, category_keys: Sequence[str] = ROAD_TAG_KEYS, max_spacing_m: float = 100.0,
                    **kw) -> pd.DataFrame:
    return _from_file(path, category_keys, max_spacing_m=max_spacing_m, **kw)


def _from_file(path, category_keys, lat_col: str = "lat", lon_col: str = "lon",
               category_col: Optional[str] = None, max_spacing_m: Optional[float] = None) -> pd.DataFrame:
    """Read POIs / roads from GeoJSON (no geopandas needed) or CSV/Parquet."""
    path = Path(path)
    if path.suffix.lower() in (".geojson", ".json"):
        rows = []
        for feat in json.loads(path.read_text()).get("features", []):
            geom, props = feat.get("geometry") or {}, feat.get("properties") or {}
            coords = _flatten_coords(geom)
            if not coords:
                continue
            cat = _category(props, category_keys)
            if max_spacing_m and len(coords) > 1:
                rows += [(lat, lon, cat) for lon, lat in _sample_line(coords, max_spacing_m)]
            else:
                arr = np.asarray(coords, float)
                rows.append((float(arr[:, 1].mean()), float(arr[:, 0].mean()), cat))
        return pd.DataFrame(rows, columns=COLUMNS)
    df = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    missing = [c for c in (lat_col, lon_col) if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: columns {missing} not found; available: {list(df.columns)}")
    cat = (df[category_col].astype(str) if category_col and category_col in df.columns
           else df.apply(lambda r: _category(r.to_dict(), category_keys), axis=1))
    return pd.DataFrame({"lat": df[lat_col].astype(float), "lon": df[lon_col].astype(float),
                         "category": cat}).reset_index(drop=True)


def _flatten_coords(geom) -> List[Tuple[float, float]]:
    """All (lon, lat) pairs of any GeoJSON geometry."""
    if not geom:
        return []
    if geom.get("type") == "GeometryCollection":
        return [c for g in geom.get("geometries", []) for c in _flatten_coords(g)]
    out, stack = [], [geom.get("coordinates")]
    while stack:
        item = stack.pop()
        if isinstance(item, (list, tuple)) and item and isinstance(item[0], (int, float)):
            out.append((float(item[0]), float(item[1])))
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
    return out


def _sample_line(coords: Sequence[Tuple[float, float]], max_spacing_m: float) -> List[Tuple[float, float]]:
    """Points along a (lon, lat) line, at most `max_spacing_m` apart, so long edges are not
    represented by a single midpoint."""
    c = np.asarray(coords, float)
    if len(c) < 2:
        return [tuple(c[0])] if len(c) else []
    seg = haversine_m(c[:-1, 1], c[:-1, 0], c[1:, 1], c[1:, 0])
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 0:
        return [tuple(c[0])]
    n = max(int(np.ceil(total / max_spacing_m)), 1)
    targets = (np.arange(n) + 0.5) * total / n
    return [(float(np.interp(t, cum, c[:, 0])), float(np.interp(t, cum, c[:, 1]))) for t in targets]


def _first(v):
    if isinstance(v, (list, tuple)) and len(v):
        return str(v[0])
    return str(v)


def _category(props: dict, keys: Iterable[str]) -> str:
    for k in keys:
        v = props.get(k)
        if v is not None and not (isinstance(v, float) and np.isnan(v)) and str(v).lower() not in ("nan", "none", ""):
            return f"{k}={_first(v)}"
    return "other"


# --------------------------------------------------------------------------- embeddings
def category_embeddings(categories: Sequence[str], dim: int = 64) -> Tuple[np.ndarray, List[str]]:
    """One-hot over the (dim - 1) most frequent categories, with everything else in a shared bucket."""
    counts = pd.Series(list(categories)).value_counts()
    vocab = list(counts.index[:max(dim - 1, 1)])
    index = {c: i for i, c in enumerate(vocab)}
    emb = np.zeros((len(categories), dim), np.float32)
    emb[np.arange(len(categories)), [index.get(c, dim - 1) for c in categories]] = 1.0
    return emb, vocab + ["<other>"]


# --------------------------------------------------------------------------- assembly
def clip_to_bbox(df: pd.DataFrame, bbox) -> pd.DataFrame:
    lat_min, lat_max, lon_min, lon_max = bbox
    return df[df.lat.between(lat_min, lat_max) & df.lon.between(lon_min, lon_max)].reset_index(drop=True)


def thin(df: pd.DataFrame, max_features: Optional[int], seed: int = 0) -> pd.DataFrame:
    """Random subsample. The model compares every trajectory point with every context entry, so the
    cost is linear in the number of entries; keeping hundreds of thousands is rarely worthwhile."""
    if max_features is None or len(df) <= max_features:
        return df
    idx = np.sort(np.random.default_rng(seed).choice(len(df), max_features, replace=False))
    log.warning(f"thinning {len(df)} -> {max_features} entries (cost is linear in this number)")
    return df.iloc[idx].reset_index(drop=True)


def write_context(out_dir, poi_df: Optional[pd.DataFrame] = None, road_df: Optional[pd.DataFrame] = None,
                  dim: int = 64) -> Dict[str, str]:
    """Write the .npy files and return the `context` mapping to paste into a config."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ctx = {}
    for kind, df in (("poi", poi_df), ("road", road_df)):
        if df is None or not len(df):
            continue
        emb, vocab = category_embeddings(df.category.tolist(), dim)
        np.save(out / f"{kind}_embed.npy", emb)
        np.save(out / f"{kind}_latlon.npy", df[["lat", "lon"]].to_numpy(np.float64))
        (out / f"{kind}_categories.json").write_text(json.dumps(vocab, indent=1))
        ctx[f"{kind}_embed"] = str(out / f"{kind}_embed.npy")
        ctx[f"{kind}_latlon"] = str(out / f"{kind}_latlon.npy")
        log.info(f"{kind}: {len(df)} entries, {emb.shape[1]}-d embeddings, {len(vocab)} categories "
                 f"(top: {', '.join(vocab[:5])})")
    return ctx


def suggest_radius_m(poi_df: Optional[pd.DataFrame], bbox, target_neighbours: float = 8.0) -> float:
    """Radius whose disc holds ~`target_neighbours` POIs at the average density of the area.
    TransferTraj compares SQUARED distances, so `poi_dist` in a config is this value squared."""
    if poi_df is None or not len(poi_df):
        return 500.0
    density = len(poi_df) / max(bbox_area_km2(bbox), 1e-6)          # per km²
    return float(np.clip(np.sqrt(target_neighbours / (np.pi * density)) * 1000, 50.0, 2000.0))


def build_from_osm(bbox, out_dir, dim: int = 64, poi_tags: Optional[dict] = None,
                   network_type: str = "drive", max_spacing_m: float = 100.0,
                   max_features: Optional[int] = 50_000, timeout: int = 300) -> Dict[str, str]:
    """Download POIs and roads for `bbox` and write the four .npy files. Needs internet + osmnx."""
    area = bbox_area_km2(bbox)
    log.info(f"downloading OSM features for {area:,.0f} km² ...")
    if area > 5000:
        log.warning(f"{area:,.0f} km² is a large area for a single Overpass query; if it times out, "
                    f"use a Geofabrik extract and the *_from_file readers instead")
    poi = thin(clip_to_bbox(pois_from_osm(bbox, poi_tags, timeout), bbox), max_features)
    road = thin(clip_to_bbox(roads_from_osm(bbox, network_type, max_spacing_m, timeout), bbox), max_features)
    ctx = write_context(out_dir, poi, road, dim)
    ctx["_suggested_poi_dist"] = round(suggest_radius_m(poi, bbox) ** 2)
    return ctx
