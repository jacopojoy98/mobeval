"""Naive-but-honest baselines, evaluated on EXACTLY the same samples as the
models. They replace the placeholder 0.05 and make skill scores meaningful."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import MobilityDataset, TrajectoryBatch, VisitBatch
from .geo import LocalProjection, haversine_m
from .metrics.probabilistic import Mixture


# ---------------------------------------------------------------- recovery ---
def linear_interpolation(batch: TrajectoryBatch, mask: np.ndarray):
    """Time-aware linear interpolation between observed neighbours."""
    lat, lon = batch.lat.copy(), batch.lon.copy()
    for i in range(len(batch)):
        obs = ~mask[i]
        if obs.sum() < 2:
            continue
        t = batch.t[i]
        lat[i, mask[i]] = np.interp(t[mask[i]], t[obs], batch.lat[i, obs])
        lon[i, mask[i]] = np.interp(t[mask[i]], t[obs], batch.lon[i, obs])
    return lat, lon


def last_observed(batch: TrajectoryBatch, mask: np.ndarray):
    lat, lon = batch.lat.copy(), batch.lon.copy()
    for i in range(len(batch)):
        for j in range(1, batch.length):
            if mask[i, j]:
                lat[i, j], lon[i, j] = lat[i, j - 1], lon[i, j - 1]
    return lat, lon


# ---------------------------------------------------------------- location ---
class LocationBaselines:
    """Fitted on TRAIN visit sequences only."""

    def __init__(self, train_visits: VisitBatch, n_cells: int, alpha: float = 0.1):
        self.n = n_cells
        cells = np.concatenate([train_visits.ctx_cell[:, -1], train_visits.tgt_cell])
        self.popularity = np.bincount(cells, minlength=n_cells).astype(float) + alpha
        self.popularity /= self.popularity.sum()
        self.trans: Dict[int, np.ndarray] = {}
        for a, b in zip(train_visits.ctx_cell[:, -1], train_visits.tgt_cell):
            self.trans.setdefault(int(a), np.zeros(n_cells))[b] += 1

    def global_popular(self, v: VisitBatch) -> np.ndarray:
        return np.tile(self.popularity, (len(v), 1))

    def user_frequent(self, v: VisitBatch) -> np.ndarray:
        """Frequency of cells in the visible context, backed off to popularity."""
        out = np.tile(self.popularity * 1e-3, (len(v), 1))
        for i, row in enumerate(v.ctx_cell):
            np.add.at(out[i], row, 1.0)
        return out / out.sum(1, keepdims=True)

    def markov1(self, v: VisitBatch) -> np.ndarray:
        out = np.empty((len(v), self.n))
        for i, last in enumerate(v.ctx_cell[:, -1]):
            counts = self.trans.get(int(last))
            row = self.popularity * 1e-2 if counts is None else counts + self.popularity * 1e-2
            out[i] = row / row.sum()
        return out


# -------------------------------------------------------------- continuous ---
class ContinuousBaselines:
    """Train-set marginal distribution of the target, in minutes."""

    def __init__(self, train_values_min: np.ndarray, n_components: int = 3, seed: int = 0):
        v = np.asarray(train_values_min, float)
        v = v[np.isfinite(v) & (v > 0)]
        self.median = float(np.median(v))
        gm = GaussianMixture(n_components, random_state=seed).fit(np.log(v)[:, None])
        self._w, self._m = gm.weights_, gm.means_.ravel()
        self._s = np.sqrt(gm.covariances_.ravel())

    def point(self, n: int) -> np.ndarray:
        return np.full(n, self.median)

    def mixture(self, n: int) -> Mixture:
        tile = lambda a: np.tile(a, (n, 1))
        return Mixture(tile(self._w), tile(self._m), tile(self._s), space="log")


# ---------------------------------------------------------- classification ---
def handcrafted_features(batch: TrajectoryBatch) -> np.ndarray:
    """Speed/acceleration/heading statistics - a strong classical mode baseline."""
    d = haversine_m(batch.lat[:, :-1], batch.lon[:, :-1], batch.lat[:, 1:], batch.lon[:, 1:])
    dt = np.maximum(np.diff(batch.t, axis=1), 1.0)
    v = d / dt
    a = np.diff(v, axis=1) / dt[:, 1:]
    proj = LocalProjection.from_points(batch.lat, batch.lon)
    x, y = proj.to_xy(batch.lat, batch.lon)
    head = np.arctan2(np.diff(y, axis=1), np.diff(x, axis=1))
    turn = np.abs(np.angle(np.exp(1j * np.diff(head, axis=1))))
    q = lambda arr, p: np.percentile(arr, p, axis=1)
    feats = [v.mean(1), q(v, 50), q(v, 85), q(v, 95), v.std(1), np.abs(a).mean(1), q(np.abs(a), 90),
             turn.mean(1), (v < 0.5).mean(1), d.sum(1), (batch.t[:, -1] - batch.t[:, 0])]
    return np.stack(feats, 1)


def majority_class_probs(train_y: np.ndarray, n: int, n_classes: int) -> np.ndarray:
    """Class-prior predictor: argmax = majority class, probabilities = train priors."""
    p = np.bincount(train_y, minlength=n_classes).astype(float) + 1e-3
    return np.tile(p / p.sum(), (n, 1))


def handcrafted_classifier(train: TrajectoryBatch, train_y, test: TrajectoryBatch, seed: int = 0) -> np.ndarray:
    clf = HistGradientBoostingClassifier(random_state=seed, max_iter=200)
    clf.fit(handcrafted_features(train), train_y)
    proba = clf.predict_proba(handcrafted_features(test))
    full = np.full((len(test), int(max(train_y.max(), proba.shape[1] - 1)) + 1), 1e-9)
    full[:, clf.classes_] = proba
    return full


def linear_probe(train_emb, train_y, test_emb, n_classes: int, seed: int = 0) -> np.ndarray:
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, random_state=seed))
    clf.fit(train_emb, train_y)
    proba = clf.predict_proba(test_emb)
    full = np.full((len(test_emb), n_classes), 1e-9)
    full[:, clf.classes_] = proba
    return full


# -------------------------------------------------------------- generation ---
def uniform_bbox_generator(reference: MobilityDataset, n_trajectories: int, seed: int = 0) -> MobilityDataset:
    """Lower reference: trips with realistic timing but uniformly random places."""
    rng = np.random.default_rng(seed)
    p = reference.points
    tids = rng.choice(p.traj_id.unique(), size=n_trajectories, replace=True)
    rows = []
    for k, tid in enumerate(tids):
        g = p[p.traj_id == tid]
        n_stops = max(2, int(rng.integers(2, 5)))
        stops = np.column_stack([rng.uniform(p.lat.min(), p.lat.max(), n_stops),
                                 rng.uniform(p.lon.min(), p.lon.max(), n_stops)])
        seg = np.array_split(np.arange(len(g)), n_stops)
        for s, idx in enumerate(seg):
            rows.append(pd.DataFrame({"user_id": g.user_id.iloc[0], "traj_id": f"gen{k}",
                                      "t": g.t.to_numpy()[idx],
                                      "lat": stops[s, 0] + rng.normal(0, 1e-4, len(idx)),
                                      "lon": stops[s, 1] + rng.normal(0, 1e-4, len(idx))}))
    return MobilityDataset(pd.concat(rows, ignore_index=True), "uniform_bbox")
