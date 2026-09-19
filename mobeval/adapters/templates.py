"""Adapter templates for UniTraj, TrajGPT and your model.

The model code was not available when this pipeline was written, so the
model-specific calls are marked `TODO(model)`. Everything else - unit
conversion, de-normalisation, vocabulary mapping, log-space mixtures - is
already implemented, because that is where the logged inconsistencies came from
(ADE ~0.5 vs ~950, MAE of 138 "hours", NLL of -5.6).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..metrics.probabilistic import Mixture
from .base import (CONTINUOUS, EMBEDDING, GENERATION, MODE_CLASSIFICATION, NEXT_LOCATION, RECOVERY,
                   ContinuousPrediction, LocationPrediction, MobilityModelAdapter)


def _torch():
    import torch  # lazy: the pipeline itself does not require torch
    return torch


def count_parameters(module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


# --------------------------------------------------------------------------- #
class GPSMaskedModelAdapter(MobilityModelAdapter):
    """Shared logic for continuous-GPS masked models (UniTraj, your model).

    Most such models train on NORMALISED coordinates (z-score or min-max per
    dataset). The adapter must normalise inputs with the TRAINING statistics and
    de-normalise outputs back to degrees - metrics are then computed in metres
    by the pipeline. Never report errors in normalised space.
    """
    capabilities = {RECOVERY, MODE_CLASSIFICATION, EMBEDDING}

    def __init__(self, checkpoint: str, norm_stats: dict, device: str = "cpu", batch_size: int = 256,
                 run_tag: str = "default"):
        # norm_stats: {"lat_mean":..,"lat_std":..,"lon_mean":..,"lon_std":..} or min/max - FROM TRAINING DATA
        self.checkpoint, self.norm, self.device, self.bs, self.run_tag = checkpoint, norm_stats, device, batch_size, run_tag
        self.model = self._load(checkpoint)
        self.mode_head = None

    # ---- TODO(model): the only model-specific pieces ------------------------
    def _load(self, checkpoint):
        raise NotImplementedError("TODO(model): build the network and load the state dict")

    def _forward_reconstruct(self, x_norm: np.ndarray, t: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """x_norm (N,L,2) with masked positions zeroed/mask-token'd -> (N,L,2) normalised predictions."""
        raise NotImplementedError("TODO(model)")

    def _forward_embed(self, x_norm: np.ndarray, t: np.ndarray) -> np.ndarray:
        """-> (N,d) pooled encoder output (mean over tokens or CLS)."""
        raise NotImplementedError("TODO(model)")

    # ---- canonical conversions ------------------------------------------------
    def normalise(self, lat, lon):
        n = self.norm
        if "lat_std" in n:
            return np.stack([(lat - n["lat_mean"]) / n["lat_std"], (lon - n["lon_mean"]) / n["lon_std"]], -1)
        return np.stack([(lat - n["lat_min"]) / (n["lat_max"] - n["lat_min"]),
                         (lon - n["lon_min"]) / (n["lon_max"] - n["lon_min"])], -1)

    def denormalise(self, xy):
        n = self.norm
        if "lat_std" in n:
            return xy[..., 0] * n["lat_std"] + n["lat_mean"], xy[..., 1] * n["lon_std"] + n["lon_mean"]
        return (xy[..., 0] * (n["lat_max"] - n["lat_min"]) + n["lat_min"],
                xy[..., 1] * (n["lon_max"] - n["lon_min"]) + n["lon_min"])

    def _batched(self, fn, *arrays):
        outs = [fn(*(a[i:i + self.bs] for a in arrays)) for i in range(0, len(arrays[0]), self.bs)]
        return np.concatenate(outs, 0)

    def reconstruct(self, batch, mask):
        x = np.nan_to_num(self.normalise(batch.lat, batch.lon), nan=0.0)   # masked positions arrive as NaN
        t = batch.t - batch.t[:, :1]                                          # relative seconds
        pred = self._batched(self._forward_reconstruct, x, t, mask)
        lat, lon = self.denormalise(pred)
        return lat, lon

    def embed(self, batch):
        x = self.normalise(batch.lat, batch.lon)
        return self._batched(self._forward_embed, x, batch.t - batch.t[:, :1])

    def prepare(self, task, train, val, protocol="native", label_fraction=1.0, labels=None, classes=None,
                seed=0, **kw):
        """Native mode classification = fine-tuned head on the SAME label subset
        the pipeline gives to the baselines (so few-shot comparisons are fair)."""
        if task == "mode_classification" and protocol == "native":
            # TODO(model): fine-tune (or load) a classification head on `train` / `labels`,
            # early-stop on `val`. Must be re-run for each (label_fraction, seed).
            self.mode_head = None

    def classify_mode(self, batch, classes: Sequence[str]):
        raise NotImplementedError("TODO(model): logits (N, K) from self.mode_head, columns ordered as `classes`")

    def num_parameters(self):
        try:
            return count_parameters(self.model)
        except Exception:
            return None


class UniTrajAdapter(GPSMaskedModelAdapter):
    name = "UniTraj"

    def _load(self, checkpoint):
        import torch
        from  .. import UniTraj
        model = UniTraj.load_from_checkpoint(checkpoint, map_location=self.device)
        model.eval()
        return model
    # UniTraj can also generate; add GENERATION once `generate` is implemented (return a MobilityDataset).


class MyModelAdapter(GPSMaskedModelAdapter):
    name = "MyModel"


# --------------------------------------------------------------------------- #
class TrajGPTAdapter(MobilityModelAdapter):
    """Visit-based model: region token + arrival/departure via Gaussian-mixture heads.

    Conversions handled here:
      * region tokens (H3) -> centroids, so the pipeline maps them onto the shared grid
        and also computes distance errors (no need for the shared grid to be H3);
      * GMM heads -> `Mixture` in SECONDS. If the head models log-time (a negative NLL
        such as -5.6 strongly suggests a scaled or log target), use space="log" and make
        sure the scale matches: mixture over log(seconds). If it models time in hours,
        multiply means/stds by 3600 (linear) or add log(3600) to means (log).
      * infilling with a separator token reorders the sequence; the adapter must put
        predictions back at the ORIGINAL positions before returning.
    """
    name = "TrajGPT"
    capabilities = {NEXT_LOCATION, CONTINUOUS, GENERATION}

    def __init__(self, checkpoint: str, h3_resolution: int, time_unit_seconds: float = 3600.0,
                 time_space: str = "log", device: str = "cpu", run_tag: str = "default"):
        self.checkpoint, self.res, self.device, self.run_tag = checkpoint, h3_resolution, device, run_tag
        self.time_unit_s, self.time_space = time_unit_seconds, time_space
        self.model, self.vocab = self._load(checkpoint)          # vocab: list of H3 cell ids, index = token id

    def _load(self, checkpoint):
        raise NotImplementedError("TODO(model): load TrajGPT and its region vocabulary")

    def _encode_context(self, visits):
        """TODO(model): ctx_lat/lon -> H3 tokens (h3.latlng_to_cell(lat, lon, self.res) -> vocab index),
        arrival/departure times in the model's time unit, day-of-week/time-of-day features."""
        raise NotImplementedError

    def _forward_next(self, enc):
        """TODO(model): -> region logits (N,V), travel-time GMM (w, mu, sigma) (N,M) x3, duration GMM."""
        raise NotImplementedError

    def token_latlon(self) -> np.ndarray:
        import h3
        return np.array([h3.cell_to_latlng(c) for c in self.vocab])   # (V, 2) lat, lon

    def predict_location(self, visits, grid):
        logits, _, _ = self._forward_next(self._encode_context(visits))
        return LocationPrediction(token_scores=logits, token_latlon=self.token_latlon())

    def predict_continuous(self, visits, target):
        _, tt, du = self._forward_next(self._encode_context(visits))
        w, mu, sd = tt if target == "travel_time" else du
        mix = Mixture(w, mu, sd, space=self.time_space).rescale(self.time_unit_s)   # -> seconds
        return ContinuousPrediction(mixture=mix)

    def generate(self, reference, n_trajectories, seed=0):
        """TODO(model): sample visit sequences, convert tokens to centroids and emit a
        point table with two points per visit (arrival, departure) so staypoint detection
        recovers the visits: rows (user_id, traj_id, t, lat, lon)."""
        raise NotImplementedError
