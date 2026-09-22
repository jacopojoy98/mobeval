"""Synthetic anomalies injected into evaluation windows.

There is no ground-truth anomaly label in a raw GPS panel, so detection has to be measured
against corruptions we introduce ourselves. The kinds below are deliberately graded by how
much of the signal is *kinematic*, because a detector that only notices impossible speeds is
not doing representation learning - it is doing a speed check, and a handcrafted baseline
does that better and for free:

  teleport  a chunk is displaced far away          -> a huge step; trivially kinematic
  speed     displacements are scaled up            -> impossible speeds; trivially kinematic
  noise     heavy GPS jitter added                 -> step-length distribution changes
  detour    the middle heading is rotated          -> STEP LENGTHS ARE UNCHANGED
  loop      the middle is traversed and retraced   -> STEP LENGTHS ARE UNCHANGED

`detour` and `loop` are the interesting ones. They resample nothing: every consecutive
displacement magnitude is preserved exactly, so every speed, acceleration and distance
statistic is unchanged and a speed check is at chance on them by construction (measured
ROC-AUC 0.54).

They are NOT invisible to all kinematics, and the distinction matters when reading results:
splicing a rotated or reversed span inserts a sharp turn, so turning-angle features do carry
signal - `turn.mean` alone reaches ROC-AUC 0.66 on detour and 0.70 on loop, and the full
`kinematic_knn` baseline reaches 0.60-0.64. THAT, not 0.5, is the bar a model has to clear
before its skill can be attributed to having learned plausible routes.

Every kind keeps the timestamps untouched, so a detector cannot cheat off the time axis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

from .data import TrajectoryBatch
from .geo import LocalProjection

KINDS = ("teleport", "speed", "noise", "detour", "loop")
PRESERVE_STEP_LENGTHS = ("detour", "loop")      # kinematic features are unchanged by these


@dataclass
class AnomalySet:
    """A test batch with some windows corrupted, and the labels saying which."""
    batch: TrajectoryBatch
    is_anomalous: np.ndarray        # (N,) bool
    kind: str

    def __len__(self):
        return len(self.is_anomalous)

    @property
    def rate(self) -> float:
        return float(self.is_anomalous.mean())


def _middle_span(L: int, rng: np.random.Generator, frac=(0.3, 0.6)) -> Tuple[int, int]:
    """A contiguous span inside the window, never touching the first or last point."""
    n = max(2, int(round(L * rng.uniform(*frac))))
    n = min(n, L - 2)
    start = int(rng.integers(1, max(2, L - n)))
    return start, start + n


def inject(batch: TrajectoryBatch, kind: str, rate: float = 0.1, seed: int = 0,
           teleport_km: Tuple[float, float] = (5.0, 50.0), speed_factor: float = 6.0,
           noise_m: float = 300.0) -> AnomalySet:
    """Corrupt `rate` of the windows with one kind of anomaly. Timestamps are never touched."""
    if kind not in KINDS:
        raise ValueError(f"unknown anomaly kind '{kind}'; known: {KINDS}")
    rng = np.random.default_rng(seed)
    N, L = batch.lat.shape
    if L < 6:
        raise ValueError(f"windows of length {L} are too short to corrupt meaningfully (need >= 6)")
    n_bad = max(1, int(round(rate * N)))
    bad = rng.choice(N, size=min(n_bad, N - 1), replace=False)     # always leave some normal windows
    is_bad = np.zeros(N, bool)
    is_bad[bad] = True

    proj = LocalProjection.from_points(batch.lat, batch.lon)
    x, y = proj.to_xy(batch.lat, batch.lon)                        # metres, shape (N, L)
    x, y = x.copy(), y.copy()

    for i in bad:
        s, e = _middle_span(L, rng)
        if kind == "teleport":
            d = rng.uniform(*teleport_km) * 1000.0
            a = rng.uniform(0, 2 * np.pi)
            x[i, s:e] += d * np.cos(a)
            y[i, s:e] += d * np.sin(a)
        elif kind == "speed":
            dx, dy = np.diff(x[i]), np.diff(y[i])
            dx[s:e] *= speed_factor
            dy[s:e] *= speed_factor
            x[i, 1:], y[i, 1:] = x[i, 0] + np.cumsum(dx), y[i, 0] + np.cumsum(dy)
        elif kind == "noise":
            x[i] += rng.normal(0, noise_m, L)
            y[i] += rng.normal(0, noise_m, L)
        elif kind == "detour":
            # Rotate the displacements of the middle span, then carry the offset through the
            # rest so the path stays continuous. Step LENGTHS are identical to the original.
            dx, dy = np.diff(x[i]), np.diff(y[i])
            a = rng.uniform(np.pi / 3, 5 * np.pi / 3)              # never ~0: that is a no-op
            ca, sa = np.cos(a), np.sin(a)
            rx = ca * dx[s:e] - sa * dy[s:e]
            ry = sa * dx[s:e] + ca * dy[s:e]
            dx[s:e], dy[s:e] = rx, ry
            x[i, 1:], y[i, 1:] = x[i, 0] + np.cumsum(dx), y[i, 0] + np.cumsum(dy)
        elif kind == "loop":
            # The vehicle suddenly retraces its route in reverse. Reversing the order of the
            # span's displacements AND negating them is a permutation with sign flips, so every
            # step length is preserved exactly - only the direction of travel is wrong.
            dx, dy = np.diff(x[i]), np.diff(y[i])
            dx[s:e], dy[s:e] = -dx[s:e][::-1], -dy[s:e][::-1]
            x[i, 1:], y[i, 1:] = x[i, 0] + np.cumsum(dx), y[i, 0] + np.cumsum(dy)

    lat, lon = proj.to_latlon(x, y)
    # Fresh trajectory ids. Adapters cache embeddings on (traj_id, first time, last time), and
    # these windows keep their original timestamps by design - reusing the ids would hand back
    # the CLEAN window's embedding and make every detector look like chance for no visible
    # reason. Tagging by kind and seed also keeps different corruptions apart in the cache.
    tag = f"#anom:{kind}:{seed}"
    ids = np.array([f"{t}{tag}" for t in np.asarray(batch.traj_id).tolist()], dtype=object)
    out = TrajectoryBatch(lat, lon, batch.t.copy(), np.asarray(batch.user_id).copy(), ids,
                          None if batch.mode is None else np.asarray(batch.mode))
    return AnomalySet(out, is_bad, kind)


def step_lengths(batch: TrajectoryBatch) -> np.ndarray:
    from .geo import haversine_m
    return haversine_m(batch.lat[:, :-1], batch.lon[:, :-1], batch.lat[:, 1:], batch.lon[:, 1:])


def describe(original: TrajectoryBatch, anomalies: AnomalySet) -> Dict[str, float]:
    """How much the corruption shows up in plain kinematics, used to verify that the
    step-length-preserving kinds really are invisible to a speed detector.

    `step_length_deviation` is the largest relative change in the sorted multiset of step
    lengths. For `detour` and `loop` it is not exactly zero because displacements are rotated
    in a local planar frame and then mapped back to lat/lon, and that projection is not
    perfectly conformal - the residual is around 0.1%, far below anything a detector can use.
    """
    a = step_lengths(original)[anomalies.is_anomalous]
    b = step_lengths(anomalies.batch)[anomalies.is_anomalous]
    sa, sb = np.sort(a, axis=1), np.sort(b, axis=1)
    dev = np.abs(sb - sa) / np.maximum(sa, 1.0)
    return {"max_step_ratio": float(np.median(b.max(1) / np.maximum(a.max(1), 1e-9))),
            "total_distance_ratio": float(np.median(b.sum(1) / np.maximum(a.sum(1), 1e-9))),
            "step_length_deviation": float(np.max(dev))}


def all_kinds(batch: TrajectoryBatch, kinds: Sequence[str], rate: float, seed: int) -> Dict[str, AnomalySet]:
    return {k: inject(batch, k, rate, seed) for k in kinds}
