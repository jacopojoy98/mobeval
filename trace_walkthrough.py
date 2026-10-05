"""Follow one number through the pipeline by hand (see CODE_GUIDE.md, section 6).

    python examples/trace_walkthrough.py        # from the repository root; synthetic data, CPU, ~1 min

The model has random weights: the point is to see every intermediate array, not the score.
"""
import numpy as np
from mobeval.data import synthetic_dataset, make_mask
from mobeval.context import EvalConfig, EvalContext
from mobeval.adapters.base import TargetGuard
from mobeval.adapters.unitraj import UniTrajAdapter
from mobeval.metrics.reconstruction import recovery_metrics
from mobeval.stats import evaluate_with_ci
from mobeval import baselines as B

# 1. data -> context (splits, windows, staypoints, visit sequences, shared grid)
ctx = EvalContext(synthetic_dataset(n_users=20, n_days=6), EvalConfig(window_length=32, visit_context=4))
print({k: len(v) for k, v in ctx.windows.items()}, {k: len(v) for k, v in ctx.visits.items()})

# 2. the model (random weights here: enough to follow the data, not to get a good number)
model = UniTrajAdapter(device="cpu")

# 3. what RecoveryTask.run does for one (kind, ratio, seed)
batch = ctx.windows["test"]
mask = make_mask(len(batch), batch.length, 0.5, "block", seed=0)      # True = hidden
hidden = TargetGuard.hide_masked(batch, mask)                          # hidden points become NaN
plat, plon = model.reconstruct(hidden, mask)                           # -> UniTrajAdapter._encode / _decode
per_window = recovery_metrics(plat, plon, batch.lat, batch.lon, mask, ctx.grid)
base = recovery_metrics(*B.linear_interpolation(hidden, mask), batch.lat, batch.lon, mask, ctx.grid)

# 4. what Task.emit does for one metric
print(evaluate_with_ci("ade_m", per_window["ade_m"], baseline_vals=base["ade_m"], n_boot=200))

# 5. next location through a linear probe on the frozen embedding (UniTraj has no location head)
from mobeval.tasks import NextLocationTask
scores, _ = NextLocationTask()._probe_score(model, ctx, ctx.visits["test"], ctx.grid)
print({k: round(float(np.mean(v)), 3) for k, v in scores.items()})
