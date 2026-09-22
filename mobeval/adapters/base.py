"""The adapter contract. Wrapping a model = implementing the subset of methods
it natively supports and declaring them in `capabilities`. The pipeline owns
data, masks, units, metrics and baselines; the adapter only converts between
the canonical formats below and the model's own I/O.

Canonical formats (never deviate - convert inside the adapter):
  * coordinates: WGS84 degrees (lat, lon), de-normalised
  * times:       unix seconds;  continuous targets returned in SECONDS
  * locations:   shared-grid scores, OR own-token scores + token centroids, OR coordinates
"""
from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Optional, Sequence, Set

import numpy as np

from ..data import MobilityDataset, SpatialGrid, TrajectoryBatch, VisitBatch
from ..metrics.probabilistic import Mixture

RECOVERY = "recovery"
NEXT_LOCATION = "next_location"
CONTINUOUS = "continuous"            # travel time / stay duration
MODE_CLASSIFICATION = "mode_classification"
EMBEDDING = "embedding"
GENERATION = "generation"
ALL_CAPABILITIES = {RECOVERY, NEXT_LOCATION, CONTINUOUS, MODE_CLASSIFICATION, EMBEDDING, GENERATION}


@dataclass
class LocationPrediction:
    """Provide exactly ONE of the three forms."""
    grid_scores: Optional[np.ndarray] = None      # (N, grid.n_cells) probabilities or logits
    token_scores: Optional[np.ndarray] = None     # (N, V) over the model's own vocabulary (e.g. H3)
    token_latlon: Optional[np.ndarray] = None     # (V, 2) centroid (lat, lon) of each token
    latlon: Optional[np.ndarray] = None           # (N, 2) point prediction

    def check(self):
        forms = [self.grid_scores is not None, self.token_scores is not None, self.latlon is not None]
        if sum(forms) != 1:
            raise ValueError("LocationPrediction: provide exactly one of grid_scores / token_scores / latlon")
        if self.token_scores is not None and self.token_latlon is None:
            raise ValueError("token_scores requires token_latlon (centroids) to map onto the shared grid")


@dataclass
class ContinuousPrediction:
    """Provide a point estimate, a Mixture, or samples - all in SECONDS
    (for a log-space mixture: over log(seconds))."""
    point: Optional[np.ndarray] = None            # (N,)
    mixture: Optional[Mixture] = None
    samples: Optional[np.ndarray] = None          # (N, S)


class MobilityModelAdapter(ABC):
    name: str = "unnamed"
    run_tag: str = "default"
    capabilities: Set[str] = set()

    # ---- bookkeeping ---------------------------------------------------------
    def num_parameters(self) -> Optional[int]:
        return None

    def prepare(self, task: str, train, val, protocol: str = "native", label_fraction: float = 1.0,
                **kwargs) -> None:
        """Optional hook: fine-tune a head / build vocab / load a task checkpoint
        on TRAIN (and select on VAL) before evaluation. Default: no-op."""

    # ---- capabilities (override the ones you support) -------------------------
    def reconstruct(self, batch: TrajectoryBatch, mask: np.ndarray):
        """Return (lat, lon) arrays of shape (N, L). Masked positions are hidden:
        the adapter MUST NOT read batch.lat[mask] / batch.lon[mask]."""
        raise NotImplementedError

    def predict_location(self, visits: VisitBatch, grid: SpatialGrid) -> LocationPrediction:
        """Predict the location of the target visit from ctx_* fields only."""
        raise NotImplementedError

    def predict_continuous(self, visits: VisitBatch, target: str) -> ContinuousPrediction:
        """target in {'travel_time', 'duration'}; must not read tgt_* fields."""
        raise NotImplementedError

    def classify_mode(self, batch: TrajectoryBatch, classes: Sequence[str]) -> np.ndarray:
        """(N, K) probabilities or logits, columns ordered as `classes`."""
        raise NotImplementedError

    def embed(self, batch: TrajectoryBatch) -> np.ndarray:
        """(N, d) frozen trajectory embeddings (used for linear probing).

        MUST be a pure function of each row: `embed(b)[i]` may depend on row `i` and on the
        model's weights, never on the other rows in the batch. The pipeline embeds train, val
        and test in separate calls and compares the results directly (linear probes, kNN
        anomaly scores), so any per-batch statistic - standardising by the batch mean, fitting
        a projection to the batch extent - puts those sets in different spaces and the
        comparison silently measures the difference between batches. It bites hardest in
        anomaly detection, where the injected anomalies would set the test batch's scale and
        hide themselves.
        """
        raise NotImplementedError

    def generate(self, reference: MobilityDataset, n_trajectories: int, seed: int = 0) -> MobilityDataset:
        """Generate trajectories (canonical point table). `reference` is TRAIN data
        (for conditioning, e.g. user ids / start times); never test data."""
        raise NotImplementedError


class TargetGuard:
    """Wraps a batch so that reading hidden fields raises - catches leakage bugs."""

    @staticmethod
    def hide_visits(v: VisitBatch, reveal=()) -> VisitBatch:
        """Blank target fields. `reveal` may contain 'location' and/or 'arrival' for
        teacher-forced conditional tasks (e.g. TrajGPT predicts duration GIVEN region)."""
        nan = lambda a: np.full(a.shape, np.nan)
        loc = "location" in reveal
        return VisitBatch(v.ctx_lat, v.ctx_lon, v.ctx_cell, v.ctx_t_arrive, v.ctx_t_leave, v.user_id,
                          v.tgt_lat if loc else nan(v.tgt_lat), v.tgt_lon if loc else nan(v.tgt_lon),
                          v.tgt_cell if loc else np.full(v.tgt_cell.shape, -1),
                          nan(v.tgt_travel_time_s), nan(v.tgt_duration_s),
                          v.tgt_t_arrive if "arrival" in reveal else nan(v.tgt_t_arrive))

    @staticmethod
    def hide_masked(b: TrajectoryBatch, mask: np.ndarray) -> TrajectoryBatch:
        lat, lon = b.lat.copy(), b.lon.copy()
        lat[mask] = np.nan; lon[mask] = np.nan
        return TrajectoryBatch(lat, lon, b.t, b.user_id, b.traj_id, None)
