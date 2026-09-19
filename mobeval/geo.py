"""Geodesic helpers. Every spatial error in the pipeline is computed in METRES on
de-normalised (lat, lon) coordinates, so models that train on normalised or
projected coordinates can never leak their internal units into the results."""
from __future__ import annotations

import numpy as np

EARTH_RADIUS_M = 6_371_008.8


def haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in metres (vectorised, broadcasting)."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(a, dtype=float)) for a in (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


class LocalProjection:
    """Equirectangular projection around a reference point -> local x/y metres.
    Accurate to well under 1% within a city-scale region (< ~100 km)."""

    def __init__(self, lat0: float, lon0: float):
        self.lat0, self.lon0 = float(lat0), float(lon0)
        self._kx = np.radians(1.0) * EARTH_RADIUS_M * np.cos(np.radians(self.lat0))
        self._ky = np.radians(1.0) * EARTH_RADIUS_M

    @classmethod
    def from_points(cls, lat, lon) -> "LocalProjection":
        return cls(float(np.nanmean(lat)), float(np.nanmean(lon)))

    def to_xy(self, lat, lon):
        return (np.asarray(lon) - self.lon0) * self._kx, (np.asarray(lat) - self.lat0) * self._ky

    def to_latlon(self, x, y):
        return np.asarray(y) / self._ky + self.lat0, np.asarray(x) / self._kx + self.lon0


def radius_of_gyration_m(lat, lon, weights=None) -> float:
    """Radius of gyration (Gonzalez et al., 2008) in metres."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    if lat.size == 0:
        return np.nan
    w = np.ones_like(lat) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    clat, clon = np.sum(w * lat), np.sum(w * lon)
    d = haversine_m(lat, lon, clat, clon)
    return float(np.sqrt(np.sum(w * d ** 2)))
