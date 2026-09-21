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


# --------------------------------------------------------------------- GeoPackage (.gpkg)
def _gpkg(path, rows, table="features", srs=4326, envelope=False):
    """A minimal but valid GeoPackage, written with sqlite3 only (no GDAL needed to test)."""
    import sqlite3
    import struct
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE gpkg_spatial_ref_sys (srs_name TEXT, srs_id INTEGER PRIMARY KEY,
            organization TEXT, organization_coordsys_id INTEGER, definition TEXT, description TEXT);
        CREATE TABLE gpkg_contents (table_name TEXT PRIMARY KEY, data_type TEXT, identifier TEXT,
            description TEXT, last_change TEXT, min_x REAL, min_y REAL, max_x REAL, max_y REAL, srs_id INTEGER);
        CREATE TABLE gpkg_geometry_columns (table_name TEXT, column_name TEXT, geometry_type_name TEXT,
            srs_id INTEGER, z TINYINT, m TINYINT);
    """)
    con.execute(f'CREATE TABLE "{table}" (fid INTEGER PRIMARY KEY, geom BLOB, amenity TEXT, highway TEXT)')
    con.execute("INSERT INTO gpkg_contents VALUES (?,?,?,?,?,?,?,?,?,?)",
                (table, "features", table, "", "2026-01-01T00:00:00Z", 0, 0, 1, 1, srs))
    con.execute("INSERT INTO gpkg_geometry_columns VALUES (?,?,?,?,?,?)", (table, "geom", "GEOMETRY", srs, 0, 0))
    for wkb, amenity, highway in rows:
        flags, env = (0x03, struct.pack("<dddd", 0, 1, 0, 1)) if envelope else (0x01, b"")
        blob = b"GP" + bytes([0, flags]) + struct.pack("<i", srs) + env + wkb
        con.execute(f'INSERT INTO "{table}" (geom, amenity, highway) VALUES (?,?,?)', (blob, amenity, highway))
    con.commit()
    con.close()
    return path


def _pt(x, y, z=None):
    import struct
    return struct.pack("<BIdd", 1, 1, x, y) if z is None else struct.pack("<BIddd", 1, 1001, x, y, z)


def _line(pts):
    import struct
    return struct.pack("<BII", 1, 2, len(pts)) + b"".join(struct.pack("<dd", x, y) for x, y in pts)


@pytest.mark.parametrize("envelope", [False, True])       # real writers (QGIS, ogr2ogr) add an envelope
def test_gpkg_points_lines_and_3d(tmp_path, envelope):
    from mobeval.context_features import gpkg_layers
    path = _gpkg(str(tmp_path / f"p{int(envelope)}.gpkg"),
                 [(_pt(12.50, 55.60), "cafe", None),
                  (_pt(12.51, 55.61, 30.0), "bank", None)],        # PointZ: extra dimension skipped
                 envelope=envelope)
    poi = pois_from_file(path)
    assert list(poi.category) == ["amenity=cafe", "amenity=bank"]
    assert poi.lat.tolist() == [55.60, 55.61] and poi.lon.tolist() == [12.50, 12.51]
    assert gpkg_layers(path) == ["features"]

    road = roads_from_file(_gpkg(str(tmp_path / f"r{int(envelope)}.gpkg"),
                                 [(_line([(12.50, 55.60), (12.53, 55.60)]), None, "primary")],
                                 envelope=envelope), max_spacing_m=400)
    assert len(road) > 1 and set(road.category) == {"highway=primary"}
    assert road.lon.between(12.50, 12.53).all()


def test_gpkg_layer_selection(tmp_path):
    path = str(tmp_path / "multi.gpkg")
    _gpkg(path, [(_pt(12.5, 55.6), "cafe", None)], table="pois")
    import sqlite3
    con = sqlite3.connect(path)
    con.execute('CREATE TABLE "roads" (fid INTEGER PRIMARY KEY, geom BLOB, amenity TEXT, highway TEXT)')
    con.execute("INSERT INTO gpkg_contents VALUES ('roads','features','roads','','2026-01-01T00:00:00Z',0,0,1,1,4326)")
    con.execute("INSERT INTO gpkg_geometry_columns VALUES ('roads','geom','GEOMETRY',4326,0,0)")
    con.commit()
    con.close()
    from mobeval.context_features import gpkg_layers
    assert gpkg_layers(path) == ["pois", "roads"]
    assert len(pois_from_file(path, layer="pois")) == 1          # explicit layer
    assert len(pois_from_file(path)) == 1                        # first layer, with a warning
    with pytest.raises(ValueError, match="not found"):
        pois_from_file(path, layer="nope")


def test_gpkg_geometry_header_variants():
    import struct
    from mobeval.context_features import _gpkg_geom_coords
    big = b"GP" + bytes([0, 0x00]) + struct.pack(">i", 4326) + struct.pack(">BIdd", 0, 1, 12.5, 55.6)
    assert _gpkg_geom_coords(big) == [(12.5, 55.6)]              # big-endian header and WKB
    assert _gpkg_geom_coords(b"GP" + bytes([0, 0x11]) + struct.pack("<i", 4326)) == []   # empty flag
    assert _gpkg_geom_coords(b"not a gpkg blob") == []


def test_geopackage_is_read_and_reprojected_to_wgs84(tmp_path):
    """QGIS exports often use a projected CRS; mobeval works in WGS84 degrees throughout."""
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import LineString, Point
    gpkg = tmp_path / "osm.gpkg"
    gpd.GeoDataFrame({"amenity": ["cafe", "bank"]},
                     geometry=[Point(10.40, 43.71), Point(10.41, 43.72)], crs=4326
                     ).to_crs(32632).to_file(gpkg, layer="pois", driver="GPKG")
    gpd.GeoDataFrame({"highway": ["primary"]},
                     geometry=[LineString([(10.40, 43.71), (10.41, 43.715)])], crs=4326
                     ).to_crs(32632).to_file(gpkg, layer="roads", driver="GPKG")

    poi = pois_from_file(gpkg, layer="pois")
    assert list(poi.category) == ["amenity=cafe", "amenity=bank"]
    assert poi.lat.round(3).tolist() == [43.71, 43.72] and poi.lon.round(3).tolist() == [10.40, 10.41]
    road = roads_from_file(gpkg, layer="roads", max_spacing_m=200.0)
    assert len(road) > 1 and (road.category == "highway=primary").all()
    assert road.lat.between(43.70, 43.72).all() and road.lon.between(10.39, 10.42).all()
