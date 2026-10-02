"""Configuration and the shared evaluation context (built once, reused by every model)."""
from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import logging

import numpy as np
import pandas as pd

from .data import (MobilityDataset, SpatialGrid, TrajectoryBatch, VisitBatch, detect_staypoints,
                   make_visit_sequences, make_windows)

log = logging.getLogger("mobeval.context")


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
    # "h3" makes the shared location label space H3 cells (TrajGPT's evaluation uses resolution 7)
    grid_backend: str = "square"
    grid_h3_resolution: int = 7
    staypoint_dist_m: float = 200.0
    staypoint_time_s: float = 20 * 60
    staypoint_method: str = "points"            # 'points' (dwell points recorded) or 'trips' (gaps between trips)
    trip_link_dist_m: float = 500.0             # 'trips': max distance trip end -> next start to average them
    trip_max_stay_s: float = 3 * 86400          # 'trips': longer gaps are missing data, not stays
    visit_context: int = 8
    max_eval_samples: Optional[int] = 2000      # seeded subsample of val/test views (same for all models)
    # Training-set size. Visit sequences are built with a sliding window, so with stride 1 consecutive
    # samples share visit_context-1 of their visits: a panel dataset easily yields millions of nearly
    # identical sequences. `visit_stride` thins them at the source (train split only, so evaluation
    # semantics are untouched) and `max_train_samples` caps the training views outright.
    visit_stride: int = 1
    max_train_samples: Optional[int] = None
    # statistics
    eval_seeds: Sequence[int] = (0, 1, 2)       # repeated masks / label subsets
    n_boot: int = 500
    # tasks. None = all; otherwise a subset of tasks.TASK_NAMES. Without next_location, continuous
    # and generation no staypoints are detected, which saves hours on large GPS collections.
    tasks: Optional[Sequence[str]] = None
    recovery_ratios: Sequence[float] = (0.25, 0.5, 0.75)
    recovery_kinds: Sequence[str] = ("random", "block")
    # Keep the first and last point of every window observed under random/block masking, so the
    # interpolation baselines are defined. UniTraj's own evaluation may mask them: set false for it.
    recovery_keep_endpoints: bool = True
    # Fixed-shape schemes taken from the papers, run once each besides kinds x ratios:
    # ("last", 5) = predict the final 5 points (TransferTraj, UniTraj trajectory prediction);
    # ("keep_every", 8) = keep every 8th point and the last, recover the rest (TransferTraj TRec).
    recovery_schemes: Sequence[Tuple[str, float]] = (("last", 5), ("keep_every", 8))
    recovery_dtw: bool = True
    continuous_targets: Sequence[str] = ("travel_time", "duration")
    continuous_reveal: Dict[str, Tuple[str, ...]] = field(default_factory=dict)  # e.g. {"duration": ("location",)}
    # Also score travel time given the target location, and duration given location + arrival:
    # the teacher-forced conditions of TrajGPT's own P(+-t) evaluation (see PAPER_REVEAL).
    continuous_paper_conditioning: bool = True
    # gaps between consecutive staypoints longer than this are missing data (phone off, overnight), not travel;
    # they are excluded from the travel-time task for every model and baseline (TrajGPT uses the same 4 h rule)
    travel_time_max_h: Optional[float] = 4.0
    mode_protocols: Sequence[str] = ("native", "linear_probe")
    label_fractions: Sequence[float] = (1.0, 0.1)
    generation_max_trajectories: int = 500
    # A reconstruction model has no `generate`, but hiding the FUTURE instead of a random subset
    # turns filling-in into generation. The same rollout runs for every such model, so what is
    # compared is the models and not three hand-written decoders (see mobeval.rollout).
    generation_protocols: Sequence[str] = ("native", "rollout")
    rollout_seed_points: int = 4                # real points the model is seeded with
    rollout_block: int = 8                      # points committed per step before re-feeding
    rollout_noise_m: Optional[float] = None     # optional: restore marginal spread (see rollout.py)
    rollout_max_len: Optional[int] = 512        # cap on generated trajectory length
    # Caps for the generation task's reference computations. These bound per-trajectory work on
    # the FULL splits, which is what makes the task run out of memory on a large panel; they do
    # not change what the generator is asked to produce.
    generation_max_real_trajectories: Optional[int] = 20_000   # test trajectories behind the reference stats
    generation_nn_max_train: int = 3_000       # train trajectories in the memorisation comparison
    generation_nn_max_query: int = 2_000       # test trajectories compared against them
    # Protocols for the prediction tasks. 'native' uses the model's own head; 'linear_probe'
    # fits a linear head on frozen embeddings, which is the only way to put an encoder with no
    # such head (UniTraj, TransferTraj) on the same axis as a generative model.
    location_protocols: Sequence[str] = ("native", "linear_probe")
    continuous_protocols: Sequence[str] = ("native", "linear_probe")
    probe_top_k: int = 1000                     # candidate cells the location probe may predict
    probe_max_train: Optional[int] = 100_000    # cap on probe training samples (speed, not semantics)
    # Representation-level tasks. Both read the frozen embedding and never fine-tune it.
    user_id_max_users: int = 100                # re-identification is over this many users
    user_id_min_windows: int = 8                # ... each needing this many windows per split
    anomaly_kinds: Sequence[str] = ("teleport", "detour", "loop")
    anomaly_rate: float = 0.1                   # share of test windows corrupted
    anomaly_knn_k: int = 10
    anomaly_protocols: Sequence[str] = ("embedding_knn", "reconstruction")
    anomaly_mask_ratio: float = 0.3             # masking used by the 'reconstruction' protocol


class EvalContext:
    def __init__(self, dataset: MobilityDataset, cfg: EvalConfig):
        self.cfg = cfg
        self.dataset_name = dataset.name
        self.splits = dataset.split(cfg.split_by, cfg.split_ratios, cfg.split_seed, cfg.val_by, cfg.val_fraction)
        self.rng = np.random.default_rng(cfg.split_seed)
        self.windows: Dict[str, TrajectoryBatch] = {}
        self.staypoints: Dict[str, pd.DataFrame] = {}
        self.visits: Dict[str, VisitBatch] = {}
        from .tasks import VISIT_TASKS
        needs_visits = cfg.tasks is None or bool(VISIT_TASKS & set(cfg.tasks))
        all_sp = (self.detect_staypoints(dataset) if needs_visits else
                  pd.DataFrame(columns=["user_id", "traj_id", "lat", "lon", "t_arrive", "t_leave"]))
        if cfg.grid_backend == "square":
            self.grid = SpatialGrid.from_dataset(dataset, cfg.grid_cell_m)
        elif cfg.grid_backend == "h3":
            from .data import H3Grid
            self.grid = H3Grid.from_dataset(dataset, cfg.grid_h3_resolution, all_sp)
        else:
            raise ValueError("grid_backend must be 'square' or 'h3'")
        for name, ds in self.splits.items():
            try:
                w = make_windows(ds, cfg.window_length, cfg.window_stride, cfg.max_gap_s)
                self.windows[name] = self._cap(w) if name != "train" else self._cap_train(w, "windows")
            except ValueError:
                pass
            tids = set(ds.points.traj_id.unique())
            self.staypoints[name] = all_sp[all_sp.traj_id.isin(tids)].reset_index(drop=True)
            try:
                v = make_visit_sequences(all_sp, self.grid, cfg.visit_context,
                                         stride=cfg.visit_stride if name == "train" else 1,
                                         target_traj_ids=tids)
                self.visits[name] = (self._cap_visits(v) if name != "train"
                                     else self._cap_train(v, "visit sequences", visits=True))
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
        rec = getattr(self, "active_recipe", None)
        return {"train_fingerprint": self.fingerprint, "dataset": self.dataset_name,
                "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
                **({"recipe": rec} if rec else {}),
                **{k: v for k, v in extra.items() if v is not None}}

    def _subsample(self, n: int, m: Optional[int]) -> Optional[np.ndarray]:
        if m is None or n <= m:
            return None
        return np.sort(np.random.default_rng(self.cfg.split_seed).choice(n, m, replace=False))

    LARGE_TRAIN = 500_000

    def _cap_train(self, batch, what: str, visits: bool = False):
        idx = self._subsample(len(batch), self.cfg.max_train_samples)
        if idx is None:
            if len(batch) > self.LARGE_TRAIN:
                log.warning(
                    f"train {what}: {len(batch):,} samples - one epoch will be {len(batch) // 64:,} steps at "
                    f"batch size 64. Thin them with `visit_stride` (overlapping sequences) and/or "
                    f"`max_train_samples`, or cap work per epoch with train.max_steps_per_epoch.")
            return batch
        log.info(f"train {what}: {len(batch)} -> {len(idx)} (max_train_samples)")
        return (VisitBatch(**{k: getattr(batch, k)[idx] for k in VisitBatch.__dataclass_fields__})
                if visits else batch.take(idx))

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

    def split_overlap(self) -> Optional[dict]:
        """How much of the test set is in places and among people the training set knows.

        A location model cannot predict a cell it has never seen, and a tokenizer-based model
        will silently snap such a target onto its nearest known region - which may be a hundred
        kilometres away. That failure looks exactly like "the model is bad" in every metric, so
        the overlap is measured up front rather than inferred afterwards from strange numbers.
        """
        tr, te = self.staypoints.get("train"), self.staypoints.get("test")
        if tr is None or te is None or not len(tr) or not len(te):
            return None
        tr_cells = self.grid.cell_of(tr.lat.to_numpy(), tr.lon.to_numpy())
        te_cells = self.grid.cell_of(te.lat.to_numpy(), te.lon.to_numpy())
        seen = np.unique(tr_cells)
        inside = np.isin(te_cells, seen)
        out = {"test_cells_seen_in_train": float(inside.mean()),
               "train_cells": int(len(seen)),
               "test_cells": int(len(np.unique(te_cells)))}
        if not inside.all():                       # how far an unseen target would be snapped
            from scipy.spatial import cKDTree
            from .geo import LocalProjection
            slat, slon = self.grid.centroid(seen)
            proj = LocalProjection.from_points(slat, slon)
            tree = cKDTree(np.column_stack(proj.to_xy(slat, slon)))
            miss = ~inside
            d = tree.query(np.column_stack(proj.to_xy(te.lat.to_numpy()[miss], te.lon.to_numpy()[miss])))[0]
            out["median_snap_m"] = float(np.median(d))
            out["p95_snap_m"] = float(np.percentile(d, 95))
        users = {"train": set(tr.user_id.unique()), "test": set(te.user_id.unique())}
        out["test_users_seen_in_train"] = (len(users["test"] & users["train"]) / len(users["test"])
                                           if users["test"] else 0.0)
        return out

    def summary(self) -> str:
        lines = [f"dataset={self.dataset_name} split_by={self.cfg.split_by} grid={self.grid.describe()}"]
        for s in ("train", "val", "test"):
            lines.append(f"  {s:5s}: users={self.splits[s].points.user_id.nunique():5d} points={len(self.splits[s].points):7d} "
                         f"windows={len(self.windows[s]) if s in self.windows else 0:6d} "
                         f"staypoints={len(self.staypoints[s]):6d} "
                         f"visit_seqs={len(self.visits[s]) if s in self.visits else 0:6d}")
        ov = self.split_overlap()
        if ov:
            lines.append(f"  overlap: {ov['test_cells_seen_in_train']:.1%} of test staypoints are in cells seen in "
                         f"train ({ov['train_cells']:,} train cells, {ov['test_cells']:,} test cells); "
                         f"{ov['test_users_seen_in_train']:.1%} of test users appear in train")
            if "median_snap_m" in ov:
                lines.append(f"           unseen test staypoints are {ov['median_snap_m'] / 1000:.1f} km "
                             f"(p95 {ov['p95_snap_m'] / 1000:.1f} km) from the nearest cell train has seen")
        return "\n".join(lines)
