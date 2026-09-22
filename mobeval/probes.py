"""Linear probes on frozen embeddings.

A probe answers a different question from a native head: not "can this model do the task"
but "is the information needed for the task linearly decodable from its representation".
That is what makes UniTraj and TransferTraj comparable with TrajGPT on next location and
travel time at all - they have no such heads, and inventing one would measure the head.

Three rules make the numbers honest and keep them comparable across models:

  * the probe is LINEAR. No hidden layer, so its score is a property of the representation
    rather than of the capacity bolted on top of it.
  * the encoder is FROZEN. Embeddings are computed once and never receive gradients.
  * every model gets the same probe, the same training data and the same hyper-parameters.

Results are always reported with `protocol='linear_probe'`, never mixed with native ones.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

log = logging.getLogger("mobeval.probes")


# ------------------------------------------------------------------ location
def candidate_cells(train_cell: np.ndarray, n_cells: int, top_k: int) -> np.ndarray:
    """The most-visited training cells, which is what the probe is allowed to predict.

    One output per grid cell is impossible at realistic grid sizes (a 500 m grid over a
    region is easily 500k cells), so the probe predicts the `top_k` most visited ones.
    Everything else falls back to the training popularity - see
    `metrics.classification.candidate_ranking_metrics`.
    """
    counts = np.bincount(np.asarray(train_cell, int), minlength=n_cells).astype(float)
    cand = np.argsort(-counts, kind="mergesort")[:top_k]
    return cand[counts[cand] > 0]


def location_probe(train_emb: np.ndarray, train_cell: np.ndarray, test_emb: np.ndarray,
                   n_cells: int, top_k: int = 1000, seed: int = 0, device: str = "auto",
                   max_train: Optional[int] = 100_000, epochs: int = 40
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Multinomial linear probe over the candidate cells.

    Returns (probabilities over candidates, candidate cell ids, training cell counts).
    The probability matrix is (n_test, len(candidates)), never (n_test, n_cells): the dense
    form would be tens of gigabytes on a realistic grid.
    """
    train_cell = np.asarray(train_cell, int)
    cand = candidate_cells(train_cell, n_cells, top_k)
    counts = np.bincount(train_cell, minlength=n_cells).astype(float)
    if len(cand) < 2:
        raise ValueError("fewer than two distinct training cells - nothing to probe")

    remap = np.full(n_cells, -1, int)
    remap[cand] = np.arange(len(cand))
    keep = np.flatnonzero(remap[train_cell] >= 0)
    if max_train is not None and len(keep) > max_train:
        keep = np.random.default_rng(seed).choice(keep, max_train, replace=False)
    y = remap[train_cell[keep]]
    X = np.asarray(train_emb, float)[keep]

    probs = _linear_softmax(X, y, np.asarray(test_emb, float), len(cand), seed, device, epochs)
    return probs, cand, counts


def _linear_softmax(X, y, Xte, n_classes, seed, device, epochs) -> np.ndarray:
    """A linear softmax classifier. Uses the project's own torch trainer when torch is
    available (needed for thousands of classes, and it can use the GPU), and falls back to
    scikit-learn for the small-class case so the core stays usable without torch."""
    try:
        from .nn.common import TrainConfig, fit_classifier_head, predict_head
    except ImportError:                                           # no torch in this environment
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
        clf.fit(X, y)
        out = np.full((len(Xte), n_classes), 1e-12)
        out[:, clf.classes_] = clf.predict_proba(Xte)
        return out / out.sum(1, keepdims=True)
    cfg = TrainConfig(epochs=epochs, batch_size=512, lr=1e-2, weight_decay=1e-4, patience=6,
                      device=device, seed=seed, log_every=0)
    # hidden=None -> a single linear layer, which is what makes this a probe and not a model.
    # class_weighted=False -> the probe should inherit the task's natural class imbalance,
    # exactly like the baselines it is compared against.
    head = fit_classifier_head(X, y, n_classes, cfg, hidden=None, class_weighted=False)
    return predict_head(head, Xte)


# ------------------------------------------------------------------ continuous
def continuous_probe(train_emb: np.ndarray, train_y_min: np.ndarray, test_emb: np.ndarray,
                     val_emb: Optional[np.ndarray] = None, val_y_min: Optional[np.ndarray] = None,
                     alpha: float = 1.0):
    """Ridge regression on log(minutes), returning a log-normal predictive distribution.

    The pipeline scores continuous targets with CRPS, NLL and PIT, so a bare point estimate
    is not enough: the probe needs a distribution. Ridge gives the mean of log(value) and the
    spread comes from the residuals on VALIDATION data - never on the data the metrics are
    computed on, and never on the ridge's own training residuals, which are optimistic.

    Returns (point estimate in minutes, Mixture over log(minutes)).
    """
    from .metrics.probabilistic import Mixture
    X, Xte = np.asarray(train_emb, float), np.asarray(test_emb, float)
    y = np.log(np.maximum(np.asarray(train_y_min, float), 1e-3))
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    X, y = X[ok], y[ok]
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Z = np.column_stack([(X - mu) / sd, np.ones(len(X))])
    A = Z.T @ Z + alpha * np.eye(Z.shape[1])
    A[-1, -1] -= alpha                                  # never regularise the intercept
    w = np.linalg.solve(A, Z.T @ y)

    predict = lambda E: np.column_stack([(np.asarray(E, float) - mu) / sd, np.ones(len(E))]) @ w
    # Calibrate on validation: both the OFFSET and the spread. Taking only a centred standard
    # deviation would hide a systematic train->val shift - which is exactly what a chronological
    # split produces - inside a small sigma, and the metrics would then report the probe as
    # over-confident when the real defect is that it is biased.
    bias = 0.0
    if val_emb is not None and val_y_min is not None and len(val_emb):
        vy = np.log(np.maximum(np.asarray(val_y_min, float), 1e-3))
        resid = vy - predict(val_emb)
        resid = resid[np.isfinite(resid)]
        if len(resid) > 1:
            bias, sigma = float(resid.mean()), float(resid.std())
        else:
            sigma = 1.0
    else:                                               # no validation split: fall back, and say so
        sigma = float(np.std(y - predict(X)))
        log.warning("continuous probe: no validation data, so the predictive spread comes from "
                    "training residuals and will be over-confident")
    if not np.isfinite(sigma) or sigma <= 0:
        # `float(np.std(...)) or 1.0` would let NaN through (NaN is truthy), and a NaN sigma
        # turns every downstream nll/crps into NaN with nothing in the logs to explain it.
        log.warning("continuous probe: residual spread is not finite; falling back to sigma=1")
        sigma = 1.0
    sigma = max(sigma, 1e-3)
    m = predict(Xte) + bias
    n = len(m)
    mixture = Mixture(np.ones((n, 1)), m.reshape(n, 1), np.full((n, 1), sigma), space="log")
    # The MEDIAN, exp(m), not the mean exp(m + sigma^2/2). Every native adapter returns a
    # mixture with no point estimate, and `continuous_metrics` then scores its median; the
    # train_marginal baseline uses the median too. Handing over the mean instead would inflate
    # this one row's MAE by exp(sigma^2/2) - 38% at sigma=0.8 - and make the probe look worse
    # than its own predictive distribution warrants, for reasons of bookkeeping alone.
    return np.exp(m), mixture
