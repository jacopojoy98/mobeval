"""Configuration and the shared evaluation context (built once, reused by every model)."""
from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .data import (MobilityDataset, SpatialGrid, TrajectoryBatch, VisitBatch, detect_staypoints,
                   make_visit_sequences, make_windows)


@dataclass
class EvalConfig:
    # data & splits
    split_by: str = "time"                      # 'time' (UniTE 8:1:1), 'user', or 'predefined' (dataset `split` column)
    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    split_seed: int = 0
    val_by: str = "time"                        # predefined split without val: carve val from train by time/user
    val_fraction: float = 0.1
    window_length: int = 64
    window_stride: Optional[int] = None
    max_gap_s: float = 300.0
    grid_cell_m: float = 500.0
    staypoint_dist_m: float = 200.0
    staypoint_time_s: float = 20 * 60
    staypoint_method: str = "points"            # 'points' (dwell points recorded) or 'trips' (gaps between trips)
    trip_link_dist_m: float = 500.0             # 'trips': max distance trip end -> next start to average them
    trip_max_stay_s: float = 3 * 86400          # 'trips': longer gaps are missing data, not stays
    visit_context: int = 8
    max_eval_samples: Optional[int] = 2000      # seeded subsample of test views (same for all models)
    # statistics
    eval_seeds: Sequence[int] = (0, 1, 2)       # repeated masks / label subsets
    n_boot: int = 500
    # tasks
    recovery_ratios: Sequence[float] = (0.25, 0.5, 0.75)
    recovery_kinds: Sequence[str] = ("random", "block")
    recovery_dtw: bool = True
    continuous_targets: Sequence[str] = ("travel_time", "duration")
    continuous_reveal: Dict[str, Tuple[str, ...]] = field(default_factory=dict)  # e.g. {"duration": ("location",)}
    # gaps between consecutive staypoints longer than this are missing data (phone off, overnight), not travel;
    # they are excluded from the travel-time task for every model and baseline (TrajGPT uses the same 4 h rule)
    travel_time_max_h: Optional[float] = 4.0
    mode_protocols: Sequence[str] = ("native", "linear_probe")
    label_fractions: Sequence[float] = (1.0, 0.1)
    generation_max_trajectories: int = 500


class EvalContext:
    def __init__(self, dataset: MobilityDataset, cfg: EvalConfig):
        self.cfg = cfg
        self.dataset_name = dataset.name
        self.splits = dataset.split(cfg.split_by, cfg.split_ratios, cfg.split_seed, cfg.val_by, cfg.val_fraction)
        self.grid = SpatialGrid.from_dataset(dataset, cfg.grid_cell_m)
        self.rng = np.random.default_rng(cfg.split_seed)
        self.windows: Dict[str, TrajectoryBatch] = {}
        self.staypoints: Dict[str, pd.DataFrame] = {}
        self.visits: Dict[str, VisitBatch] = {}
        all_sp = self.detect_staypoints(dataset)
        for name, ds in self.splits.items():
            try:
                w = make_windows(ds, cfg.window_length, cfg.window_stride, cfg.max_gap_s)
                self.windows[name] = self._cap(w) if name != "train" else w
            except ValueError:
                pass
            tids = set(ds.points.traj_id.unique())
            self.staypoints[name] = all_sp[all_sp.traj_id.isin(tids)].reset_index(drop=True)
            try:
                v = make_visit_sequences(all_sp, self.grid, cfg.visit_context, target_traj_ids=tids)
                self.visits[name] = self._cap_visits(v) if name != "train" else v
            except ValueError:
                pass
        self.cache: Dict = {}
        self.emitted_baselines = set()
        self.latency = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))   # model -> capability -> [sec, n]
        self.skipped, self.errors = [], []

    def detect_staypoints(self, ds: MobilityDataset) -> pd.DataFrame:
        """Staypoints of REAL data with the configured method."""
        from .data import staypoints_from_trips
        c = self.cfg
        if c.staypoint_method == "trips":
            return staypoints_from_trips(ds, c.staypoint_time_s, c.trip_link_dist_m, c.trip_max_stay_s)
        if c.staypoint_method == "points":
            return detect_staypoints(ds, c.staypoint_dist_m, c.staypoint_time_s)
        raise ValueError("staypoint_method must be 'points' or 'trips'")

    @property
    def fingerprint(self) -> str:
        """Hash of the split assignment (which trajectories are train/val/test) + view parameters."""
        import hashlib
        h = hashlib.sha1()
        for name in ("train", "val", "test"):
            h.update(name.encode())
            h.update("|".join(map(str, sorted(self.splits[name].points.traj_id.unique()))).encode())
        c = self.cfg
        h.update(repr((c.split_by, tuple(c.split_ratios), c.split_seed, c.window_length, c.max_gap_s,
                       c.staypoint_dist_m, c.staypoint_time_s, c.visit_context, c.staypoint_method)).encode())
        return h.hexdigest()[:16]

    def provenance(self, **extra) -> dict:
        import datetime
        return {"train_fingerprint": self.fingerprint, "dataset": self.dataset_name,
                "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
                **{k: v for k, v in extra.items() if v is not None}}

    def _cap(self, b: TrajectoryBatch) -> TrajectoryBatch:
        m = self.cfg.max_eval_samples
        if m is None or len(b) <= m:
            return b
        return b.take(np.sort(np.random.default_rng(self.cfg.split_seed).choice(len(b), m, replace=False)))

    def _cap_visits(self, v: VisitBatch) -> VisitBatch:
        m = self.cfg.max_eval_samples
        if m is None or len(v) <= m:
            return v
        idx = np.sort(np.random.default_rng(self.cfg.split_seed).choice(len(v), m, replace=False))
        return VisitBatch(**{k: getattr(v, k)[idx] for k in VisitBatch.__dataclass_fields__})

    @contextmanager
    def timed(self, model_key: str, capability: str, n: int):
        t0 = time.perf_counter()
        yield
        rec = self.latency[model_key][capability]
        rec[0] += time.perf_counter() - t0
        rec[1] += n

    def summary(self) -> str:
        lines = [f"dataset={self.dataset_name} split_by={self.cfg.split_by} grid={self.grid.nx}x{self.grid.ny} "
                 f"cells @ {self.cfg.grid_cell_m:.0f} m"]
        for s in ("train", "val", "test"):
            lines.append(f"  {s:5s}: users={self.splits[s].points.user_id.nunique():5d} points={len(self.splits[s].points):7d} "
                         f"windows={len(self.windows[s]) if s in self.windows else 0:6d} "
                         f"staypoints={len(self.staypoints[s]):6d} "
                         f"visit_seqs={len(self.visits[s]) if s in self.visits else 0:6d}")
        return "\n".join(lines)
