"""Metrics for scoring tasks where the output is one number per sample and the label is
binary (anomaly detection). Threshold-free, because a threshold is a deployment choice and
comparing models at one arbitrary cut-off says more about the cut-off than about the models.

All three metrics are set-level: they cannot be computed per sample, so the pipeline's
bootstrap resamples the whole set through the `set_fn` mechanism.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, Tuple

import numpy as np

log = logging.getLogger("mobeval.detection")


def _rank_average(x: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged, which is what makes AUC correct when scores are tied."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), float)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    """P(score of a random positive > score of a random negative), ties counted as half.

    Computed from rank sums (the Mann-Whitney U identity) rather than by sweeping
    thresholds, so it is exact with ties and O(n log n).
    """
    y = np.asarray(y).astype(bool)
    score = np.asarray(score, float)
    ok = np.isfinite(score)
    y, score = y[ok], score[ok]
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = _rank_average(score)
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _tie_groups(y: np.ndarray, score: np.ndarray):
    """Sort by descending score and collapse equal scores. Returns the cumulative count and
    cumulative positives at the END of each tie group.

    Ties have to be handled explicitly or a detector that outputs a constant scores far above
    chance purely through the tie-break order of the sort.
    """
    order = np.argsort(-score, kind="mergesort")
    y, score = y[order], score[order]
    last = np.r_[np.flatnonzero(np.diff(score)), len(score) - 1]     # index of each group's last element
    return last + 1, np.cumsum(y)[last]


def pr_auc(y: np.ndarray, score: np.ndarray) -> float:
    """Average precision: the step-wise area under precision-recall, evaluated at each
    distinct score threshold so that tied scores are not silently broken in the model's favour.

    Read it against the anomaly rate (what a random scorer gets), not against 0.5.
    """
    y = np.asarray(y).astype(bool)
    score = np.asarray(score, float)
    ok = np.isfinite(score)
    y, score = y[ok], score[ok]
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    n, tp = _tie_groups(y, score)
    precision = tp / n
    recall = tp / n_pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def precision_at_k(y: np.ndarray, score: np.ndarray, k: int = None) -> float:
    """Precision among the k highest-scored samples. k defaults to the number of positives,
    so the ceiling is 1.0 and the chance level is the anomaly rate.

    When the k-th place falls inside a group of tied scores, the tied group contributes its
    own positive rate rather than whichever members the sort happened to put first.
    """
    y = np.asarray(y).astype(bool)
    score = np.asarray(score, float)
    ok = np.isfinite(score)
    y, score = y[ok], score[ok]
    k = int(y.sum()) if k is None else int(k)
    if k <= 0 or k > len(y):
        return float("nan")
    n, tp = _tie_groups(y, score)
    full = np.searchsorted(n, k, side="left")                 # first group that reaches k
    before_n = n[full - 1] if full > 0 else 0
    before_tp = tp[full - 1] if full > 0 else 0
    group_n = n[full] - before_n
    group_tp = tp[full] - before_tp
    taken = k - before_n                                      # how many of the tied group fit in k
    return float((before_tp + group_tp * taken / group_n) / k)


def detection_metrics(y: np.ndarray, score: np.ndarray
                      ) -> Tuple[Dict[str, np.ndarray], Dict[str, Callable[[np.ndarray], float]]]:
    """(per-sample, set-level) in the shape the pipeline's `emit` expects.

    Non-finite scores are ranked last (most normal) rather than dropped. Dropping them would
    evaluate the model on fewer samples than its baseline while the record still claimed the
    full `n`, and would shrink `k` in precision@k to the anomalies that happened to survive -
    so a model that failed to score half the set could be reported as beating one that scored
    all of it. Ranking them last instead keeps every comparison on identical samples and lets
    a model that cannot produce a score simply lose the credit for those rows.
    """
    y = np.asarray(y).astype(bool)
    score = np.asarray(score, float)
    bad = ~np.isfinite(score)
    if bad.any():
        log.warning(f"anomaly scoring: {bad.sum()}/{len(score)} scores are not finite; they are ranked "
                    f"as the most normal samples so every model is scored on the same set")
        finite = score[~bad]
        floor = (finite.min() - 1.0) if finite.size else 0.0
        score = np.where(bad, floor, score)
    return {}, {"roc_auc": lambda i: roc_auc(y[i], score[i]),
                "pr_auc": lambda i: pr_auc(y[i], score[i]),
                "precision@k": lambda i: precision_at_k(y[i], score[i])}


# ------------------------------------------------------------------ scoring rules
def knn_distance(train_emb: np.ndarray, test_emb: np.ndarray, k: int = 10,
                 standardise: bool = True, block: int = 2048) -> np.ndarray:
    """Mean distance to the k nearest TRAIN embeddings: high = unlike anything seen in training.

    Standardising first stops one high-variance embedding dimension from dominating the
    distance, which would make the score a proxy for that single coordinate. Computed in
    blocks so a large test set does not need an (n_test x n_train) matrix in memory.
    """
    train_emb = np.asarray(train_emb, float)
    test_emb = np.asarray(test_emb, float)
    if standardise:
        mu = train_emb.mean(0)
        sd = train_emb.std(0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        train_emb, test_emb = (train_emb - mu) / sd, (test_emb - mu) / sd
    k = max(1, min(int(k), len(train_emb)))
    tr_sq = (train_emb ** 2).sum(1)
    out = np.empty(len(test_emb), float)
    for s in range(0, len(test_emb), block):
        te = test_emb[s:s + block]
        d2 = (te ** 2).sum(1)[:, None] + tr_sq[None, :] - 2.0 * te @ train_emb.T
        np.maximum(d2, 0.0, out=d2)
        part = np.partition(d2, k - 1, axis=1)[:, :k]
        out[s:s + block] = np.sqrt(part).mean(1)
    return out


def mahalanobis_distance(train_emb: np.ndarray, test_emb: np.ndarray, shrinkage: float = 0.1) -> np.ndarray:
    """Distance to the training embedding distribution under a shrunk Gaussian.

    Cheaper than kNN and smoother, but it assumes one elliptical blob; kNN is the better
    default when the training embeddings are multi-modal, which mobility embeddings usually are.
    """
    train_emb = np.asarray(train_emb, float)
    test_emb = np.asarray(test_emb, float)
    mu = train_emb.mean(0)
    X = train_emb - mu
    cov = X.T @ X / max(len(X) - 1, 1)
    cov = (1 - shrinkage) * cov + shrinkage * np.trace(cov) / len(cov) * np.eye(len(cov))
    prec = np.linalg.pinv(cov)
    d = test_emb - mu
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, prec, d), 0.0))
