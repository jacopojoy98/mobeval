"""Tests for POI / road context features and their use by TransferTraj."""
import json

import numpy as np
import pandas as pd
import pytest

from mobeval import EvalConfig, synthetic_dataset
from mobeval.context_features import (bbox_area_km2, category_embeddings, clip_to_bbox, dataset_bbox,
                                      pois_from_file, roads_from_file, suggest_radius_m, thin, write_context)
from mobeval.geo import haversine_m


def _geojson(path, kind="poi", n=50, seed=0):
    rng = np.random.default_rng(seed)
    feats = []
    for _ in range(n):
        lon, lat = float(rng.uniform(12.5, 12.6)), float(rng.uniform(55.6, 55.7))
        if kind == "poi":
            feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]},
                          "properties": {"amenity": str(rng.choice(["cafe", "bank", "school"]))}})
        else:
            feats.append({"type": "Feature",
                          "geometry": {"type": "LineString", "coordinates": [[lon, lat], [lon + 0.01, lat + 0.005]]},
                          "properties": {"highway": str(rng.choice(["primary", "residential"]))}})
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    return path


def test_dataset_bbox_covers_all_points_with_padding():
    ds = synthetic_dataset(n_users=6, n_days=3, seed=0)
    lat_min, lat_max, lon_min, lon_max = dataset_bbox(ds, pad_km=2.0)
    p = ds.points
    assert lat_min < p.lat.min() and lat_max > p.lat.max() and lon_min < p.lon.min() and lon_max > p.lon.max()
    assert haversine_m(p.lat.min(), p.lon.min(), lat_min, p.lon.min()) == pytest.approx(2000, rel=0.02)
    assert bbox_area_km2((lat_min, lat_max, lon_min, lon_max)) > 0


def test_geojson_points_and_linestring_sampling(tmp_path):
    poi = pois_from_file(_geojson(tmp_path / "p.geojson", "poi", 30))
    assert len(poi) == 30 and set(poi.columns) == {"lat", "lon", "category"}
    assert poi.category.str.startswith("amenity=").all()
    # each ~1 km edge becomes several sample points, never fewer than one
    road = roads_from_file(_geojson(tmp_path / "r.geojson", "road", 10), max_spacing_m=200.0)
    assert len(road) > 10 and road.category.str.startswith("highway=").all()
    coarse = roads_from_file(tmp_path / "r.geojson", max_spacing_m=5000.0)
    assert len(coarse) == 10                                    # one point per short edge


def test_csv_source_and_explicit_category_column(tmp_path):
    pd.DataFrame({"lat": [55.6, 55.61], "lon": [12.5, 12.51], "kind": ["shop", "cafe"]}).to_csv(
        tmp_path / "p.csv", index=False)
    df = pois_from_file(tmp_path / "p.csv", category_col="kind")
    assert list(df.category) == ["shop", "cafe"]


def test_category_embeddings_are_onehot_with_shared_other_bucket():
    cats = ["a"] * 10 + ["b"] * 5 + ["c"] * 2 + ["d"]
    emb, vocab = category_embeddings(cats, dim=3)
    assert emb.shape == (18, 3) and (emb.sum(1) == 1).all()
    assert vocab == ["a", "b", "<other>"]
    assert np.array_equal(emb[-1], emb[-2])                     # c and d share the "other" column


def test_clip_and_thin(tmp_path):
    df = pd.DataFrame({"lat": [55.0, 55.65, 56.0], "lon": [12.55, 12.55, 12.55], "category": ["x"] * 3})
    assert len(clip_to_bbox(df, (55.6, 55.7, 12.5, 12.6))) == 1
    big = pd.DataFrame({"lat": np.zeros(100), "lon": np.zeros(100), "category": ["x"] * 100})
    assert len(thin(big, 10)) == 10 and len(thin(big, None)) == 100


def test_suggested_radius_scales_with_density():
    bbox = (55.6, 55.7, 12.5, 12.6)
    dense = pd.DataFrame({"lat": np.zeros(5000), "lon": np.zeros(5000), "category": ["x"] * 5000})
    sparse = pd.DataFrame({"lat": np.zeros(20), "lon": np.zeros(20), "category": ["x"] * 20})
    assert suggest_radius_m(dense, bbox) < suggest_radius_m(sparse, bbox)


def test_write_context_produces_the_arrays_adapters_expect(tmp_path):
    poi = pois_from_file(_geojson(tmp_path / "p.geojson", "poi", 40))
    road = roads_from_file(_geojson(tmp_path / "r.geojson", "road", 12), max_spacing_m=300.0)
    ctx = write_context(tmp_path / "out", poi, road, dim=16)
    assert set(ctx) == {"poi_embed", "poi_latlon", "road_embed", "road_latlon"}
    e, c = np.load(ctx["poi_embed"]), np.load(ctx["poi_latlon"])
    assert e.shape == (40, 16) and c.shape == (40, 2)
    assert c[:, 0].min() > 55 and c[:, 1].max() < 13          # (lat, lon) order, not swapped
    assert json.loads((tmp_path / "out" / "poi_categories.json").read_text())[-1] == "<other>"
