"""Small non-neural reference models. They exist to (a) smoke-test the pipeline
end to end, (b) show every output form an adapter can return. Not baselines."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from ..baselines import handcrafted_features
from ..data import MobilityDataset, SpatialGrid
from ..metrics.probabilistic import Mixture
from .base import (CONTINUOUS, EMBEDDING, GENERATION, MODE_CLASSIFICATION, NEXT_LOCATION, RECOVERY,
                   ContinuousPrediction, LocationPrediction, MobilityModelAdapter)


class KinematicReference(MobilityModelAdapter):
    """PCHIP recovery · own-tokenizer location scores · log-normal duration mixture ·
    feature embeddings (probe) · jitter-copy generator (a deliberate 'memoriser')."""
    name = "KinematicRef"
    capabilities = {RECOVERY, NEXT_LOCATION, CONTINUOUS, EMBEDDING, GENERATION}

    def __init__(self, token_cell_m: float = 300.0, seed: int = 0):
        self.token_cell_m, self.seed = token_cell_m, seed
        self._proj = np.random.default_rng(seed).normal(size=(11, 32))
        self._norm = None                  # fixed on first embed() so the mapping is row-wise

    def num_parameters(self):
        return 11 * 32

    def reconstruct(self, batch, mask):
        lat, lon = batch.lat.copy(), batch.lon.copy()
        for i in range(len(batch)):
            o = ~mask[i]
            t = batch.t[i]
            lat[i, mask[i]] = PchipInterpolator(t[o], batch.lat[i, o])(t[mask[i]])
            lon[i, mask[i]] = PchipInterpolator(t[o], batch.lon[i, o])(t[mask[i]])
        return lat, lon

    def predict_location(self, visits, grid):
        b = grid.bounds
        own = SpatialGrid(b[0], b[1], b[2], b[3], self.token_cell_m)          # model-specific vocabulary
        tok = own.cell_of(visits.ctx_lat, visits.ctx_lon)
        scores = np.full((len(visits), own.n_cells), 1e-4)
        C = tok.shape[1]
        w = 0.6 ** np.arange(C)[::-1]                                           # recency-weighted frequency
        for i in range(len(visits)):
            np.add.at(scores[i], tok[i], w)
        # "return home" prior: the first context visit gets a boost
        scores[np.arange(len(visits)), tok[:, 0]] += 0.5
        scores /= scores.sum(1, keepdims=True)
        return LocationPrediction(token_scores=scores, token_latlon=np.column_stack(own.centroid(np.arange(own.n_cells))))

    def predict_continuous(self, visits, target):
        if target == "duration":
            d = np.maximum(visits.ctx_t_leave - visits.ctx_t_arrive, 60.0)
        else:
            d = np.maximum(visits.ctx_t_arrive[:, 1:] - visits.ctx_t_leave[:, :-1], 30.0)
        ld = np.log(d)
        mu, sd = ld.mean(1), ld.std(1) + 0.3
        return ContinuousPrediction(mixture=Mixture(np.ones((len(d), 1)), mu[:, None], sd[:, None], space="log"))

    def embed(self, batch):
        """`embed` must be a pure function of each row (see MobilityModelAdapter.embed).

        This used to standardise with statistics of the batch it was handed, which made a
        window's embedding depend on what it was embedded alongside. Train and test are
        embedded in separate calls, so they landed in different spaces - and in the anomaly
        task the injected anomalies themselves set the test-batch scale, squashing every test
        embedding toward zero and hiding the very thing being detected. The normalisation is
        now fixed on first use and reused for every later batch.
        """
        f = handcrafted_features(batch)
        if self._norm is None:
            self._norm = (f.mean(0), f.std(0) + 1e-9)
        mu, sd = self._norm
        return np.tanh(((f - mu) / sd) @ self._proj)

    def generate(self, reference: MobilityDataset, n_trajectories: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        p = reference.points
        tids = rng.choice(p.traj_id.unique(), n_trajectories, replace=True)
        parts = []
        for k, tid in enumerate(tids):
            g = p[p.traj_id == tid].copy()
            g["lat"] += rng.normal(0, 2e-4, len(g)); g["lon"] += rng.normal(0, 3e-4, len(g))
            g["traj_id"] = f"gen{k}"
            parts.append(g[["user_id", "traj_id", "t", "lat", "lon"]])
        return MobilityDataset(pd.concat(parts, ignore_index=True), "generated")


class WeakReference(MobilityModelAdapter):
    """Deliberately weak: noisy recovery, point location, constant point durations,
    rule-based native mode classifier. Exercises the point-prediction code paths."""
    name = "WeakRef"
    capabilities = {RECOVERY, NEXT_LOCATION, CONTINUOUS, MODE_CLASSIFICATION}

    def __init__(self, noise_m: float = 150.0, seed: int = 0):
        self.noise_m, self.seed = noise_m, seed

    def num_parameters(self):
        return 5

    def reconstruct(self, batch, mask):
        from ..baselines import linear_interpolation
        lat, lon = linear_interpolation(batch, mask)
        rng = np.random.default_rng(self.seed)
        lat = lat + mask * rng.normal(0, self.noise_m / 111_195, lat.shape)
        lon = lon + mask * rng.normal(0, self.noise_m / 70_000, lon.shape)
        return lat, lon

    def predict_location(self, visits, grid):
        return LocationPrediction(latlon=np.column_stack([visits.ctx_lat[:, -2], visits.ctx_lon[:, -2]]))

    def predict_continuous(self, visits, target):
        return ContinuousPrediction(point=np.full(len(visits), 30 * 60.0))

    def classify_mode(self, batch, classes):
        v = handcrafted_features(batch)[:, 1]                                   # median speed
        centres = {"walk": 1.4, "bike": 4.5, "bus": 7.0, "car": 11.0, "train": 20.0}
        c = np.array([centres.get(k, 5.0) for k in classes])
        return -np.abs(np.log(v[:, None] + 0.1) - np.log(c[None, :])) * 3     # logits
