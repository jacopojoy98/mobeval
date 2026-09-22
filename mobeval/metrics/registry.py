"""Central metric registry: the ONLY place where direction, unit, valid range and
skill-score formula are defined. Per-model scripts never declare these."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

INF = float("inf")


@dataclass(frozen=True)
class MetricSpec:
    name: str
    family: str                 # recovery | location | continuous | classification | generation | efficiency
    higher_is_better: bool
    unit: str
    skill: str                  # 'ratio' (1-v/b) | 'bounded' ((v-b)/(1-b)) | 'difference' (b-v) | 'floor' | 'none'
    valid_range: Tuple[float, float] = (-INF, INF)
    plausible_max: Optional[float] = None   # soft sanity bound -> flagged, never silently dropped
    aggregate: str = "mean"     # mean | rmse | median | p90
    description: str = ""


_S = MetricSpec
_SPECS = [
    # --- recovery (masked points, metres, de-normalised haversine) -----------
    _S("ade_m", "recovery", False, "m", "ratio", (0, INF), 20_000, "mean", "Mean haversine error on masked points"),
    _S("median_ade_m", "recovery", False, "m", "ratio", (0, INF), 20_000, "median", "Median per-trajectory ADE"),
    _S("p90_ade_m", "recovery", False, "m", "ratio", (0, INF), 50_000, "p90", "90th pct per-trajectory ADE"),
    _S("rmse_m", "recovery", False, "m", "ratio", (0, INF), 50_000, "rmse", "sqrt(mean squared haversine error)"),
    _S("fde_m", "recovery", False, "m", "ratio", (0, INF), 20_000, "mean", "Error at last point of each masked block"),
    _S("dtw_m", "recovery", False, "m", "ratio", (0, INF), 20_000, "mean", "Path-normalised DTW over masked points"),
    _S("acc_100m", "recovery", True, "fraction", "bounded", (0, 1), None, "mean", "Masked points within 100 m"),
    _S("acc_500m", "recovery", True, "fraction", "bounded", (0, 1), None, "mean", "Masked points within 500 m"),
    _S("grid_acc", "recovery", True, "fraction", "bounded", (0, 1), None, "mean", "Masked point in true shared-grid cell"),
    # --- location prediction on the SHARED grid ------------------------------
    _S("acc@1", "location", True, "fraction", "bounded", (0, 1)),
    _S("acc@5", "location", True, "fraction", "bounded", (0, 1)),
    _S("mrr@20", "location", True, "fraction", "bounded", (0, 1)),
    _S("loc_nll", "location", False, "nats", "difference", (0, INF), None, "mean", "NLL of true cell (eps-smoothed)"),
    _S("dist_err_m", "location", False, "m", "ratio", (0, INF), 50_000, "mean", "Top-1 location -> true visit"),
    _S("median_dist_err_m", "location", False, "m", "ratio", (0, INF), 50_000, "median"),
    _S("acc_1km", "location", True, "fraction", "bounded", (0, 1), None, "mean", "Top-1 within 1 km"),
    _S("probe_coverage", "location", True, "fraction", "none", (0, 1), None, "mean",
       "Share of test targets the location probe's candidate cells can reach at all - this is "
       "its accuracy ceiling, so read acc@1 against it and not against 1.0"),
    # --- continuous values (canonical unit: minutes) -------------------------
    _S("mae_min", "continuous", False, "min", "ratio", (0, INF), 24 * 60),
    _S("rmse_min", "continuous", False, "min", "ratio", (0, INF), 24 * 60, "rmse"),
    _S("crps_min", "continuous", False, "min", "ratio", (0, INF), 24 * 60, "mean", "CRPS; equals MAE for point forecasts"),
    _S("nll", "continuous", False, "nats", "difference", (-INF, INF), None, "mean", "NLL of value measured in minutes"),
    _S("pseudo_nll", "continuous", False, "nats", "difference", (-INF, INF), None, "mean",
       "Gaussian NLL, sigma fitted on VALIDATION residuals (point models only)"),
    _S("coverage80", "continuous", True, "fraction", "none", (0, 1), None, "mean", "Ideal value 0.80"),
    _S("pit_ks", "continuous", False, "stat", "none", (0, 1), None, "mean", "KS distance of PIT from U(0,1)"),
    # --- classification -------------------------------------------------------
    _S("accuracy", "classification", True, "fraction", "bounded", (0, 1)),
    _S("balanced_accuracy", "classification", True, "fraction", "bounded", (0, 1)),
    _S("macro_f1", "classification", True, "fraction", "bounded", (0, 1)),
    _S("cls_nll", "classification", False, "nats", "difference", (0, INF)),
    _S("ece", "classification", False, "fraction", "ratio", (0, 1), None, "mean", "Expected calibration error"),
    # --- generation (distributional) -----------------------------------------
    _S("jsd", "generation", False, "bits", "floor", (0, 1), None, "mean", "JSD (base 2) on bins fixed from real data"),
    _S("w1", "generation", False, "native", "floor", (0, INF), None, "mean", "1-Wasserstein on raw values"),
    _S("copy_rate", "generation", False, "fraction", "none", (0, 1), None, "mean",
       "Share of generated trajectories closer to a train trajectory than 95% of REAL test ones are "
       "(ideal ~0.05; much higher = memorisation)"),
    _S("nn_train_dist_m", "generation", True, "m", "none", (0, INF), None, "median",
       "Median distance from generated trajectory to nearest train trajectory"),
    _S("spearman_paired", "generation", True, "rho", "none", (-1, 1), None, "mean", "Per-user stat, real vs generated"),
    # --- user identification from the latent space ------------------------------
    # Separate names from the location family: these are a property of the representation,
    # not of a prediction task, and must not be averaged into the location summary.
    _S("user_acc@1", "identity", True, "fraction", "bounded", (0, 1), None, "mean",
       "Correct user identified from one window's embedding"),
    _S("user_acc@5", "identity", True, "fraction", "bounded", (0, 1)),
    _S("user_mrr@20", "identity", True, "fraction", "bounded", (0, 1)),
    _S("user_macro_f1", "identity", True, "fraction", "bounded", (0, 1)),
    _S("user_nll", "identity", False, "nats", "difference", (0, INF), None, "mean", "NLL of the true user"),
    # --- anomaly detection --------------------------------------------------------
    _S("roc_auc", "anomaly", True, "auc", "bounded", (0, 1), None, "mean",
       "Ranking quality; 0.5 = chance, threshold-free"),
    _S("pr_auc", "anomaly", True, "auc", "bounded", (0, 1), None, "mean",
       "Average precision; compare against the anomaly rate, not against 0.5"),
    _S("precision@k", "anomaly", True, "fraction", "bounded", (0, 1), None, "mean",
       "Precision among the k highest-scored windows, k = number of injected anomalies"),
    # --- efficiency -------------------------------------------------------------
    _S("n_parameters", "efficiency", False, "count", "none", (0, INF)),
    _S("latency_ms_per_sample", "efficiency", False, "ms", "none", (0, INF)),
]
REGISTRY = {s.name: s for s in _SPECS}

# Non-redundant metrics used in cross-family summaries (ADE/median/p90/RMSE are highly
# correlated; averaging all of them would silently up-weight recovery error).
HEADLINE = {"ade_m", "fde_m", "dtw_m", "acc_100m", "acc@1", "acc@5", "dist_err_m", "crps_min", "mae_min",
            "macro_f1", "balanced_accuracy", "jsd", "user_acc@1", "user_macro_f1", "roc_auc", "pr_auc"}


def get_spec(name: str) -> MetricSpec:
    """Metric names may carry a qualifier after ':' (e.g. 'jsd:radius_of_gyration')."""
    base = name.split(":")[0]
    if base in REGISTRY:
        return REGISTRY[base]
    raise KeyError(f"metric '{name}' is not registered - add it to metrics/registry.py")
