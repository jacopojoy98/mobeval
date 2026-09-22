"""Classification & location-ranking metrics.

Per-sample metrics are returned as arrays; set-level metrics (macro-F1,
balanced accuracy, ECE) are returned as callables idx -> float so the
bootstrap can recompute them on resamples."""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np

EPS = 1e-9


def normalise_probs(scores: np.ndarray, assume: str = "auto") -> np.ndarray:
    """Turn scores into probabilities. 'auto': softmax unless rows already sum to 1."""
    s = np.asarray(scores, dtype=np.float64)
    if assume == "probs" or (assume == "auto" and np.all(s >= 0) and np.allclose(s.sum(1), 1, atol=1e-3)):
        return s / s.sum(1, keepdims=True)
    s = s - s.max(1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(1, keepdims=True)


def rank_of_true(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """1-based rank of the true label (ties broken pessimistically)."""
    p_true = probs[np.arange(len(y)), y]
    return (probs > p_true[:, None]).sum(1) + (probs == p_true[:, None]).sum(1)


def ranking_metrics(probs: np.ndarray, y: np.ndarray, ks=(1, 5), mrr_cutoff: int = 20,
                    smooth: float = 1e-6) -> Dict[str, np.ndarray]:
    r = rank_of_true(probs, y)
    out = {f"acc@{k}": (r <= k).astype(float) for k in ks}
    out[f"mrr@{mrr_cutoff}"] = np.where(r <= mrr_cutoff, 1.0 / r, 0.0)
    K = probs.shape[1]
    p = (1 - smooth) * probs[np.arange(len(y)), y] + smooth / K
    out["loc_nll"] = -np.log(p)
    return out


def macro_f1(y, pred, n_classes) -> float:
    f1s = []
    for c in range(n_classes):
        tp = np.sum((pred == c) & (y == c)); fp = np.sum((pred == c) & (y != c)); fn = np.sum((pred != c) & (y == c))
        if tp + fn == 0:
            continue                                   # class absent from this (re)sample
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else np.nan


def balanced_accuracy(y, pred, n_classes) -> float:
    rec = [np.mean(pred[y == c] == c) for c in range(n_classes) if np.any(y == c)]
    return float(np.mean(rec))


def ece(probs, y, n_bins: int = 10) -> float:
    conf, pred = probs.max(1), probs.argmax(1)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.mean() * abs(np.mean(pred[m] == y[m]) - conf[m].mean())
    return float(total)


def classification_metrics(probs: np.ndarray, y: np.ndarray
                           ) -> Tuple[Dict[str, np.ndarray], Dict[str, Callable[[np.ndarray], float]]]:
    probs = normalise_probs(probs)
    K = probs.shape[1]
    pred = probs.argmax(1)
    per_sample = {
        "accuracy": (pred == y).astype(float),
        "cls_nll": -np.log(np.clip(probs[np.arange(len(y)), y], EPS, 1)),
    }
    set_level = {
        "balanced_accuracy": lambda i: balanced_accuracy(y[i], pred[i], K),
        "macro_f1": lambda i: macro_f1(y[i], pred[i], K),
        "ece": lambda i: ece(probs[i], y[i]),
    }
    return per_sample, set_level


def candidate_ranking_metrics(cand_probs: np.ndarray, cand_cells: np.ndarray, y: np.ndarray,
                              popularity: np.ndarray, ks=(1, 5), mrr_cutoff: int = 20,
                              backoff: float = 1e-3) -> Dict[str, np.ndarray]:
    """Ranking metrics for a predictor that only scores a CANDIDATE SET of cells.

    A linear probe cannot have one output per cell when the shared grid has hundreds of
    thousands of them, so it scores the `top_k` most visited training cells and everything
    else falls back to the training popularity. Densifying that to (N, n_cells) would cost
    tens of gigabytes at realistic grid sizes, so the metrics are computed directly from the
    candidate scores:

      * a target inside the candidate set is ranked among the candidates as usual;
      * a target outside it is unreachable - it counts as a miss for acc@k and MRR, which is
        the honest accounting, since the probe genuinely cannot predict that cell;
      * its likelihood is the popularity backoff, so `loc_nll` stays finite and comparable.

    The share of targets that are reachable at all is reported separately as the probe's
    ceiling (`_coverage`), because acc@1 should be read against it and not against 1.0.
    """
    cand_probs = np.asarray(cand_probs, float)
    n, K = cand_probs.shape
    pos = np.full(int(np.max(cand_cells)) + 1, -1, int)     # cell id -> column, -1 if not a candidate
    pos[cand_cells] = np.arange(K)
    y = np.asarray(y, int)
    col = np.where(y < len(pos), pos[np.minimum(y, len(pos) - 1)], -1)
    inside = col >= 0

    # Outside the candidate set the target is unreachable: it sits behind every candidate AND
    # behind the popularity-ordered remainder of the grid. Using K+1 would let it earn MRR
    # credit whenever the candidate set happens to be smaller than the cutoff, so the rank is
    # infinite rather than merely large - acc@k and MRR are then 0 for any K.
    rank = np.full(n, np.inf, float)
    if inside.any():
        p_true = cand_probs[np.arange(n)[inside], col[inside]]
        sub = cand_probs[inside]
        rank[inside] = (sub > p_true[:, None]).sum(1) + (sub == p_true[:, None]).sum(1)

    out = {f"acc@{k}": (rank <= k).astype(float) for k in ks}
    out[f"mrr@{mrr_cutoff}"] = np.where(rank <= mrr_cutoff, 1.0 / rank, 0.0)
    pop = np.asarray(popularity, float)
    pop = pop / max(pop.sum(), EPS)
    p_out = backoff * np.where(y < len(pop), pop[np.minimum(y, len(pop) - 1)], 0.0)
    p = np.where(inside, (1 - backoff) * cand_probs[np.arange(n), np.maximum(col, 0)], p_out)
    out["loc_nll"] = -np.log(np.clip(p, EPS, 1.0))
    out["_coverage"] = inside.astype(float)
    return out
