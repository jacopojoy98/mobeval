"""Model-facing encodings of the canonical mobeval data (numpy in, numpy out).

* RegionTokenizer - discrete location vocabulary (metric grid by default, H3 if installed),
  fitted on TRAIN visits only; unseen cells map to the nearest known region.
* point_tokens    - per-GPS-point vectors for the CLIP trajectory view.
* visit_tokens    - per-visit vectors for the CLIP visit view.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from ..data import SpatialGrid
from ..geo import LocalProjection, haversine_m

POINT_TOKEN_DIM = 9
MAX_OFFSET_KM = 200.0      # clip for local offsets within a window
MAX_SPEED_MPS = 100.0      # clip for point speeds (360 km/h)
VISIT_TOKEN_DIM = 9
COORD_CHANNELS = (0, 1)
KINEMATIC_CHANNELS = (3, 4, 5)


class RegionTokenizer:
    def __init__(self, backend: str = "grid", cell_m: float = 1000.0, h3_resolution: int = 7, offset: int = 4):
        self.backend, self.cell_m, self.res, self.offset = backend, float(cell_m), int(h3_resolution), int(offset)
        self.keys = None
        self._grid: Optional[SpatialGrid] = None

    # -- raw cell keys --------------------------------------------------------------
    def _raw(self, lat, lon):
        if self.backend == "h3":
            import h3
            f = getattr(h3, "latlng_to_cell", None) or h3.geo_to_h3
            return np.array([f(a, b, self.res) for a, b in zip(np.ravel(lat), np.ravel(lon))]).reshape(np.shape(lat))
        return self._grid.cell_of(lat, lon)

    def _centroid(self, keys):
        if self.backend == "h3":
            import h3
            f = getattr(h3, "cell_to_latlng", None) or h3.h3_to_geo
            return np.array([f(k) for k in keys], float)
        return np.column_stack(self._grid.centroid(np.asarray(keys)))

    def fit(self, lat, lon, bounds=None) -> "RegionTokenizer":
        lat, lon = np.ravel(lat), np.ravel(lon)
        if self.backend == "grid":
            b = bounds or (lat.min() - 0.05, lat.max() + 0.05, lon.min() - 0.05, lon.max() + 0.05)
            self._grid = SpatialGrid(*b, cell_size_m=self.cell_m)
        self.keys = np.unique(self._raw(lat, lon))
        self._build()
        return self

    def _build(self):
        self._index = {k: i for i, k in enumerate(self.keys.tolist())}
        self.latlon = self._centroid(self.keys)
        self._proj = LocalProjection.from_points(self.latlon[:, 0], self.latlon[:, 1])
        self._tree = cKDTree(np.column_stack(self._proj.to_xy(self.latlon[:, 0], self.latlon[:, 1])))

    @property
    def n_regions(self) -> int:
        return len(self.keys)

    def tokens(self, lat, lon) -> np.ndarray:
        """Token ids (>= offset). Unseen cells -> nearest known region centroid."""
        lat, lon = np.asarray(lat, float), np.asarray(lon, float)
        raw = self._raw(lat, lon).ravel()
        out = np.array([self._index.get(k, -1) for k in raw.tolist()])
        miss = out < 0
        if miss.any():
            xy = np.column_stack(self._proj.to_xy(lat.ravel()[miss], lon.ravel()[miss]))
            out[miss] = self._tree.query(xy)[1]
        return (out + self.offset).reshape(lat.shape)

    def token_latlon(self) -> np.ndarray:
        """(n_regions, 2) centroids in token order (without special-token offset)."""
        return self.latlon

    def state(self) -> dict:
        return {"backend": self.backend, "cell_m": self.cell_m, "h3_resolution": self.res, "offset": self.offset,
                "keys": self.keys.tolist(), "grid_bounds": None if self._grid is None else list(self._grid.bounds)}

    @classmethod
    def from_state(cls, s: dict) -> "RegionTokenizer":
        tok = cls(s["backend"], s["cell_m"], s["h3_resolution"], s["offset"])
        if s["backend"] == "grid":
            tok._grid = SpatialGrid(*s["grid_bounds"], cell_size_m=s["cell_m"])
        tok.keys = np.array(s["keys"])
        tok._build()
        return tok


def _tod_week(t):
    t = np.asarray(t, float)
    tod = (t % 86400) / 86400 * 2 * np.pi
    weekday = ((t // 86400 + 3) % 7) / 6.0                     # 1970-01-01 was a Thursday
    return np.sin(tod), np.cos(tod), weekday


def point_tokens(lat, lon, t, hidden: Optional[np.ndarray] = None) -> np.ndarray:
    """(N, L, 9): dx_km, dy_km (from first visible point), log1p(dt)/5, speed/10, sin/cos heading,
    sin/cos time-of-day, weekday. Hidden points have unknown spatial/kinematic channels (0)."""
    lat, lon, t = np.asarray(lat, float), np.asarray(lon, float), np.asarray(t, float)
    N, L = lat.shape
    hidden = np.zeros((N, L), bool) if hidden is None else hidden
    first = np.argmax(~hidden, 1)
    lat0, lon0 = lat[np.arange(N), first], lon[np.arange(N), first]
    kx = np.radians(1.0) * 6_371_008.8 * np.cos(np.radians(lat0))[:, None]
    ky = np.radians(1.0) * 6_371_008.8
    x = np.where(hidden, 0.0, (lon - lon0[:, None]) * kx) / 1000.0
    y = np.where(hidden, 0.0, (lat - lat0[:, None]) * ky) / 1000.0
    tok = np.zeros((N, L, POINT_TOKEN_DIM), np.float32)
    tok[..., 0], tok[..., 1] = x, y
    dt = np.diff(t, axis=1, prepend=t[:, :1])
    tok[..., 2] = np.log1p(np.maximum(dt, 0)) / 5.0
    dx, dy = np.diff(x, axis=1, prepend=x[:, :1]) * 1000, np.diff(y, axis=1, prepend=y[:, :1]) * 1000
    known = ~hidden & ~np.roll(hidden, 1, axis=1)
    known[:, 0] = False
    tok[..., 3] = np.where(known, np.hypot(dx, dy) / np.maximum(dt, 1.0) / 10.0, 0.0)
    head = np.arctan2(dy, dx)
    tok[..., 4], tok[..., 5] = np.where(known, np.sin(head), 0.0), np.where(known, np.cos(head), 0.0)
    s, c, w = _tod_week(t)
    tok[..., 6], tok[..., 7], tok[..., 8] = s, c, w
    # physical bounds: a single GPS glitch (e.g. a jump of 1000 km) must not produce huge inputs
    np.clip(tok[..., :2], -MAX_OFFSET_KM, MAX_OFFSET_KM, out=tok[..., :2])
    np.clip(tok[..., 3], 0.0, MAX_SPEED_MPS / 10.0, out=tok[..., 3])
    if not np.isfinite(tok).all():
        raise ValueError("non-finite point tokens: check the input for NaN/inf coordinates or timestamps")
    return tok


def tokens_to_latlon(tok_xy_km: np.ndarray, lat0: np.ndarray, lon0: np.ndarray):
    kx = np.radians(1.0) * 6_371_008.8 * np.cos(np.radians(lat0))[:, None]
    ky = np.radians(1.0) * 6_371_008.8
    return lat0[:, None] + tok_xy_km[..., 1] * 1000 / ky, lon0[:, None] + tok_xy_km[..., 0] * 1000 / kx


def visit_tokens(lat, lon, t_arrive, t_leave, proj: LocalProjection) -> np.ndarray:
    """(N, C, 9): x/10km, y/10km (dataset projection), sin/cos arrival tod, sin/cos departure tod,
    log1p(duration h), log1p(travel h from previous visit), weekday."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    ta, tl = np.asarray(t_arrive, float), np.asarray(t_leave, float)
    x, y = proj.to_xy(lat, lon)
    tok = np.zeros(lat.shape + (VISIT_TOKEN_DIM,), np.float32)
    tok[..., 0], tok[..., 1] = x / 10_000.0, y / 10_000.0
    s, c, w = _tod_week(ta)
    tok[..., 2], tok[..., 3], tok[..., 8] = s, c, w
    s2, c2, _ = _tod_week(tl)
    tok[..., 4], tok[..., 5] = s2, c2
    tok[..., 6] = np.log1p(np.maximum(tl - ta, 0) / 3600.0)
    travel = ta - np.concatenate([tl[..., :1], tl[..., :-1]], axis=-1)
    tok[..., 7] = np.log1p(np.clip(travel, 0, None) / 3600.0)
    np.clip(tok[..., :2], -50.0, 50.0, out=tok[..., :2])        # +-500 km around the data centre
    return tok
