"""Shared machinery for PyTorch-backed adapters."""
from __future__ import annotations

import logging
from typing import Dict, Optional, Sequence

import numpy as np

from ..data import TrajectoryBatch
from .base import MobilityModelAdapter

log = logging.getLogger("mobeval.adapters")


class TorchAdapter(MobilityModelAdapter):
    """Adds: device handling, parameter counting, cached embeddings, and a native
    mode-classification head trained on frozen embeddings (re-fitted for every label
    subset the pipeline passes, so few-shot comparisons stay fair)."""

    model_type = "torch"

    def __init__(self, device: str = "auto", batch_size: int = 128, run_tag: str = "default",
                 name: Optional[str] = None, head_train: Optional[dict] = None, head_hidden: Optional[int] = None,
                 head_class_weighted: bool = False):
        from ..nn.common import TrainConfig, resolve_device
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self.run_tag = run_tag
        if name:
            self.name = name
        self.head_cfg = TrainConfig.from_dict({"epochs": 200, "batch_size": 256, "lr": 1e-3, "weight_decay": 1e-4,
                                               "patience": 15, "device": str(self.device), "log_every": 0,
                                               **(head_train or {})})
        # Native classification head. Defaults mirror the pipeline's linear probe (linear, unweighted), so
        # "native" vs "linear_probe" differ only in the optimiser, not in capacity or class balancing.
        self.head_hidden, self.head_class_weighted = head_hidden, head_class_weighted
        self.provenance: dict = {}       # e.g. train split fingerprint, stored in checkpoints
        self.net = None
        self._emb_cache: Dict[tuple, np.ndarray] = {}
        self._mode_head = None

    # ----------------------------------------------------------------- bookkeeping
    def num_parameters(self):
        from ..nn.common import count_parameters
        return None if self.net is None else count_parameters(self.net)

    def _require_net(self):
        if self.net is None:
            raise RuntimeError(f"{self.name}: no network loaded - use from_checkpoint(...) or pretrain(...) first")

    def epoch_checkpointer(self, out: Optional[str]):
        """Callback for `nn.common.fit`: keep `out` at the best-so-far weights.

        Without this a checkpoint only appeared once training had finished, so a crash at
        epoch 40 of 100 - or a PBS walltime kill - threw away every epoch of work. Saving on
        improvement means the file on disk is always the best model seen so far, and it is
        mirrored to durable storage by `save_checkpoint` as it is written.
        """
        if not out:
            return None

        def save_best(history):
            # complete=False: these are the best weights so far, not the end of training. If the
            # job dies here, a resumed run continues from them instead of treating them as final.
            self.save(out, history, quiet=True, complete=False)

        return save_best

    # ----------------------------------------------------------------- embeddings
    def _embed_batch(self, batch: TrajectoryBatch) -> np.ndarray:
        raise NotImplementedError

    def embed(self, batch: TrajectoryBatch) -> np.ndarray:
        """Embeddings, cached on (traj_id, first time, last time).

        That key assumes a trajectory id identifies its coordinates. Anything that CHANGES the
        coordinates while keeping the id - injected anomalies, most obviously - must give the
        batch fresh ids, or it silently receives the original window's embedding back. See
        `anomalies.inject`, which re-tags every window for exactly this reason.
        """
        self._require_net()
        keys = list(zip(batch.traj_id.tolist(), batch.t[:, 0].tolist(), batch.t[:, -1].tolist()))
        todo = [i for i, k in enumerate(keys) if k not in self._emb_cache]
        for s in range(0, len(todo), self.batch_size):
            idx = np.asarray(todo[s:s + self.batch_size], dtype=int)      # dtype matters when empty
            for i, z in zip(idx, self._embed_batch(batch.take(idx))):
                self._emb_cache[keys[i]] = z
        return np.stack([self._emb_cache[k] for k in keys])

    def invalidate_cache(self):
        self._emb_cache.clear()
        self._mode_head = None

    # --------------------------------------------------------- mode classification
    def prepare(self, task, train, val, protocol="native", label_fraction=1.0, labels=None, classes=None,
                seed=0, **kw):
        if task == "mode_classification" and protocol == "native":
            from ..nn.common import fit_classifier_head
            cfg = type(self.head_cfg)(**{**self.head_cfg.__dict__, "seed": seed})
            self._mode_head = fit_classifier_head(self.embed(train), np.asarray(labels), len(classes), cfg,
                                                  hidden=self.head_hidden, class_weighted=self.head_class_weighted)
            self._mode_classes = list(classes)

    def classify_mode(self, batch, classes: Sequence[str]):
        from ..nn.common import predict_head
        if self._mode_head is None or list(classes) != self._mode_classes:
            raise RuntimeError("mode head not fitted - prepare('mode_classification', ...) must run first")
        return predict_head(self._mode_head, self.embed(batch))
