"""Continuous-value metrics that work for BOTH probabilistic and point models.

CRPS is the bridge: for a point forecast CRPS == absolute error, for a
Gaussian mixture it has a closed form. So TrajGPT's GMM head and a point
predictor land on the same scale (minutes) with a proper scoring rule."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

LOG_2PI = np.log(2 * np.pi)


@dataclass
class Mixture:
    """Gaussian mixture per sample. space='linear': over the value itself;
    space='log': over log(value) (i.e. a log-normal mixture)."""
    weights: np.ndarray   # (N, M), rows sum to 1
    means: np.ndarray     # (N, M)
    stds: np.ndarray      # (N, M)
    space: str = "linear"

    def __post_init__(self):
        self.weights = np.asarray(self.weights, float)
        self.weights = self.weights / self.weights.sum(1, keepdims=True)
        self.means, self.stds = np.asarray(self.means, float), np.maximum(np.asarray(self.stds, float), 1e-9)
        if self.space not in ("linear", "log"):
            raise ValueError("space must be 'linear' or 'log'")

    def rescale(self, factor: float) -> "Mixture":
        """Change units of the VALUE (e.g. seconds -> minutes: factor=1/60)."""
        if self.space == "linear":
            return Mixture(self.weights, self.means * factor, self.stds * factor, "linear")
        return Mixture(self.weights, self.means + np.log(factor), self.stds, "log")

    def mean(self) -> np.ndarray:
        if self.space == "linear":
            return (self.weights * self.means).sum(1)
        return (self.weights * np.exp(self.means + 0.5 * self.stds ** 2)).sum(1)

    def median(self, iters: int = 60) -> np.ndarray:
        """Mixture median by vectorised bisection (the MAE-optimal point forecast;
        the mean of a skewed log-normal mixture badly inflates MAE)."""
        lo = (self.means - 8 * self.stds).min(1)
        hi = (self.means + 8 * self.stds).max(1)
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            z = mid[:, None]
            below = (self.weights * norm.cdf((z - self.means) / self.stds)).sum(1) < 0.5
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        m = 0.5 * (lo + hi)
        return np.exp(m) if self.space == "log" else m

    def cdf(self, y) -> np.ndarray:
        y = np.asarray(y, float)[:, None]
        if self.space == "log":
            y = np.log(np.maximum(y, 1e-12))
        return (self.weights * norm.cdf((y - self.means) / self.stds)).sum(1)

    def nll(self, y) -> np.ndarray:
        """NLL of y in the VALUE's units (log-space mixtures get the Jacobian term)."""
        y = np.asarray(y, float)
        z = np.log(np.maximum(y, 1e-12)) if self.space == "log" else y
        comp = np.log(self.weights) - 0.5 * LOG_2PI - np.log(self.stds) - 0.5 * ((z[:, None] - self.means) / self.stds) ** 2
        nll = -logsumexp(comp, axis=1)
        if self.space == "log":
            nll = nll + np.log(np.maximum(y, 1e-12))
        return nll

    def sample(self, n: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        N, M = self.weights.shape
        cum = np.cumsum(self.weights, 1)
        comp = (rng.random((N, n, 1)) > cum[:, None, :]).sum(-1).clip(0, M - 1)
        mu = np.take_along_axis(self.means, comp, 1); sd = np.take_along_axis(self.stds, comp, 1)
        x = mu + sd * rng.standard_normal((N, n))
        return np.exp(x) if self.space == "log" else x


def _A(mu, var):
    s = np.sqrt(var)
    return 2 * s * norm.pdf(mu / s) + mu * (2 * norm.cdf(mu / s) - 1)


def crps_mixture(mix: Mixture, y, n_samples: int = 500, seed: int = 0) -> np.ndarray:
    y = np.asarray(y, float)
    if mix.space == "linear":   # closed form (Grimit et al., 2006)
        w, m, v = mix.weights, mix.means, mix.stds ** 2
        t1 = (w * _A(y[:, None] - m, v)).sum(1)
        t2 = (w[:, :, None] * w[:, None, :] * _A(m[:, :, None] - m[:, None, :], v[:, :, None] + v[:, None, :])).sum((1, 2))
        return t1 - 0.5 * t2
    return crps_samples(mix.sample(n_samples, seed), y)


def crps_samples(samples: np.ndarray, y) -> np.ndarray:
    """Unbiased-ish ensemble CRPS: E|X-y| - 0.5 E|X-X'| via sorted samples."""
    x = np.sort(np.asarray(samples, float), 1)
    n = x.shape[1]
    t1 = np.abs(x - np.asarray(y, float)[:, None]).mean(1)
    coef = (2 * np.arange(1, n + 1) - n - 1)
    t2 = (x * coef).sum(1) / (n * n)
    return t1 - t2


def gaussian_pseudo_nll(pred, y, sigma: float) -> np.ndarray:
    r = np.asarray(y, float) - np.asarray(pred, float)
    return 0.5 * LOG_2PI + np.log(sigma) + 0.5 * (r / sigma) ** 2


def continuous_metrics(y, point: Optional[np.ndarray] = None, mixture: Optional[Mixture] = None,
                       samples: Optional[np.ndarray] = None, pseudo_sigma: Optional[float] = None
                       ) -> Dict[str, np.ndarray]:
    """y and every prediction must already be in canonical units (minutes)."""
    y = np.asarray(y, float)
    out: Dict[str, np.ndarray] = {}
    if mixture is not None:
        pt = mixture.median() if point is None else point
        out["crps_min"] = crps_mixture(mixture, y)
        out["nll"] = mixture.nll(y)
        pit = mixture.cdf(y)
    elif samples is not None:
        pt = np.median(samples, 1) if point is None else point
        out["crps_min"] = crps_samples(samples, y)
        pit = (samples <= y[:, None]).mean(1)
    elif point is not None:
        pt = np.asarray(point, float)
        out["crps_min"] = np.abs(pt - y)
        pit = None
        if pseudo_sigma is not None:
            out["pseudo_nll"] = gaussian_pseudo_nll(pt, y, pseudo_sigma)
    else:
        raise ValueError("need point, mixture or samples")
    out["mae_min"] = np.abs(pt - y)
    out["rmse_min"] = (pt - y) ** 2
    if pit is not None:
        out["coverage80"] = ((pit >= 0.1) & (pit <= 0.9)).astype(float)
        out["_pit"] = pit
    return out


def pit_ks(pit: np.ndarray) -> float:
    p = np.sort(pit)
    n = len(p)
    grid = np.arange(1, n + 1) / n
    return float(max(np.max(grid - p), np.max(p - (grid - 1 / n))))
