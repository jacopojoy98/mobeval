"""Adapter for the CLIP-style dual-view mobility model (trajectory transformer + visit transformer).

    adapter = CLIPMobilityAdapter.pretrain(ctx, out="ckpt/clip.pt")
    adapter = CLIPMobilityAdapter.from_checkpoint("ckpt/clip.pt")

Views built from canonical data (see nn/features.py):
  * trajectory view - one 9-d token per GPS point of a window (local offsets in km, time step,
    speed, heading, time of day, weekday);
  * visit view      - one 9-d token per staypoint (position, arrival/departure time of day,
    duration, travel time, weekday). A window is paired with the same user's last
    `visit_context` visits that ENDED BEFORE the window starts, so no future information.

Pre-training objective: next-token regression on the trajectory view + symmetric InfoNCE
between the two views (in-batch negatives, bounded temperature).

Capabilities:
  * recovery        - native: autoregressive infilling with the next-token head (the causal
                      model only uses points BEFORE a gap - a structural handicap versus
                      bidirectional models that should be kept in mind when reading results);
  * embedding       - L2-normalised trajectory embedding (the CLIP space);
  * mode classif.   - head on frozen embeddings (re-fitted per label subset);
  * next location / travel time / duration - heads on the frozen visit encoder's last-token
                      state, trained on the pipeline's TRAIN visit sequences on first use.

Custom features (e.g. OSM road type, POI density as in the original tokenizer) can be added by
passing `extra_point_features=callable(TrajectoryBatch) -> (N, L, E)`; they must be computable
for masked points without their coordinates, or be zeroed there.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np

from ..data import TrajectoryBatch, VisitBatch
from ..geo import LocalProjection
from ..metrics.probabilistic import Mixture
from .base import (CONTINUOUS, EMBEDDING, MODE_CLASSIFICATION, NEXT_LOCATION, RECOVERY, ContinuousPrediction,
                   LocationPrediction)
from .torch_base import TorchAdapter

log = logging.getLogger("mobeval.adapters.clip")
DEFAULT_ARCH = dict(d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.1, learned_pos_encoding=False,
                    embedding_dim=128, temperature=0.07, trajectory_max_seq_length=512, visit_max_seq_length=128)


class CLIPMobilityAdapter(TorchAdapter):
    name = "CLIPMobility"
    model_type = "clip_mobility"
    capabilities = {RECOVERY, EMBEDDING, MODE_CLASSIFICATION, NEXT_LOCATION, CONTINUOUS}

    def __init__(self, arch: Optional[dict] = None, visit_center=(0.0, 0.0), visit_context: int = 8,
                 extra_point_features: Optional[Callable] = None, extra_dim: int = 0, **kw):
        super().__init__(**kw)
        from ..nn.clip_net import CLIPMobilityModel
        from ..nn.features import POINT_TOKEN_DIM, VISIT_TOKEN_DIM
        self.arch = {**DEFAULT_ARCH, **(arch or {})}
        self.extra, self.extra_dim = extra_point_features, int(extra_dim)
        self.visit_proj, self.visit_context = LocalProjection(*visit_center), int(visit_context)
        self.net = CLIPMobilityModel(trajectory_token_dim=POINT_TOKEN_DIM + self.extra_dim, visit_token_dim=VISIT_TOKEN_DIM,
                                     **self.arch).to(self.device).eval()
        self._train_visits = None
        self._loc_head, self._time_heads = None, {}

    # ------------------------------------------------------------------ persistence
    def save(self, path, history=None):
        from ..nn.common import save_checkpoint
        save_checkpoint(path, self.net, self.model_type, {"arch": self.arch, "extra_dim": self.extra_dim},
                        {"visit_center": [self.visit_proj.lat0, self.visit_proj.lon0], "visit_context": self.visit_context,
                         "provenance": self.provenance},
                        history)

    @classmethod
    def from_checkpoint(cls, path, extra_point_features=None, **kw) -> "CLIPMobilityAdapter":
        from ..nn.common import load_checkpoint
        ck = load_checkpoint(path)
        if ck["format"] == "raw":
            raise ValueError("raw CLIP state dicts do not record token definitions; token layouts from the original "
                             "tokenizer differ from mobeval's, so retrain with CLIPMobilityAdapter.pretrain(...)")
        ad = cls(arch=ck["config"]["arch"], visit_center=ck["meta"]["visit_center"],
                 visit_context=ck["meta"]["visit_context"], extra_point_features=extra_point_features,
                 extra_dim=ck["config"].get("extra_dim", 0), **kw)
        ad.net.load_state_dict(ck["state_dict"], strict=True)
        ad.net.eval()
        ad.provenance = ck["meta"].get("provenance", {})
        return ad

    # ------------------------------------------------------------------ token builders
    def _point_tokens(self, batch: TrajectoryBatch, hidden=None):
        from ..nn.features import point_tokens
        tok = point_tokens(np.nan_to_num(batch.lat), np.nan_to_num(batch.lon), batch.t, hidden)
        if self.extra_dim:
            ex = np.asarray(self.extra(batch), np.float32)
            if hidden is not None:
                ex = np.where(hidden[..., None], 0.0, ex)
            tok = np.concatenate([tok, ex], -1)
        return tok

    def _visit_tokens(self, lat, lon, ta, tl):
        from ..nn.features import visit_tokens
        return visit_tokens(lat, lon, ta, tl, self.visit_proj)

    def _T(self, a, dtype=None):
        import torch
        return torch.as_tensor(a, dtype=dtype or torch.float32, device=self.device)

    # ------------------------------------------------------------------ recovery (autoregressive infilling)
    def reconstruct(self, batch: TrajectoryBatch, mask: np.ndarray):
        import torch
        from ..nn.features import KINEMATIC_CHANNELS, tokens_to_latlon
        self._require_net()
        lat_out, lon_out = np.empty_like(batch.lat), np.empty_like(batch.lon)
        fill = list((0, 1) + KINEMATIC_CHANNELS)
        with torch.no_grad():
            for s in range(0, len(batch), self.batch_size):
                sl = slice(s, s + self.batch_size)
                b, m = batch.take(np.arange(len(batch))[sl]), mask[sl]
                tok = self._T(self._point_tokens(b, m))
                for j in np.where(m.any(0))[0]:
                    if j == 0:
                        continue
                    rows = torch.as_tensor(m[:, j], device=self.device)
                    nxt = self.net.trajectory_transformer(tok[:, :j])[:, -1]
                    for c in fill:
                        tok[rows, j, c] = nxt[rows, c]
                first = np.argmax(~m, 1)
                idx = np.arange(len(b))
                la, lo = tokens_to_latlon(tok[..., :2].cpu().numpy(), b.lat[idx, first], b.lon[idx, first])
                lat_out[sl], lon_out[sl] = np.where(m, la, b.lat), np.where(m, lo, b.lon)
        return lat_out, lon_out

    # ------------------------------------------------------------------ embeddings
    def _embed_batch(self, batch):
        import torch
        with torch.no_grad():
            return self.net.encode_trajectory(self._T(self._point_tokens(batch))).cpu().numpy()

    # ------------------------------------------------------------------ visit-encoder heads
    def prepare(self, task, train, val, **kw):
        if task in ("next_location", "continuous/travel_time", "continuous/duration") and isinstance(train, VisitBatch):
            if self._train_visits is None or len(self._train_visits) != len(train):
                self._train_visits, self._loc_head, self._time_heads = train, None, {}
            return
        super().prepare(task, train, val, **kw)

    def _visit_state(self, v: VisitBatch) -> np.ndarray:
        """Frozen visit-encoder state at the last context visit (causal => summarises the context)."""
        import torch
        out = []
        with torch.no_grad():
            for s in range(0, len(v), self.batch_size):
                sl = slice(s, s + self.batch_size)
                tok = self._T(self._visit_tokens(v.ctx_lat[sl], v.ctx_lon[sl], v.ctx_t_arrive[sl], v.ctx_t_leave[sl]))
                out.append(self.net.visit_transformer.hidden_states(tok)[:, -1].cpu().numpy())
        return np.concatenate(out)

    def _need_train_visits(self):
        if self._train_visits is None:
            raise RuntimeError("visit heads need training visits: the pipeline calls prepare(...) first; "
                               "when calling directly, use adapter.prepare('next_location', ctx.visits['train'], None)")

    def predict_location(self, visits: VisitBatch, grid) -> LocationPrediction:
        from ..nn.common import fit_classifier_head, predict_head
        self._need_train_visits()
        if self._loc_head is None or self._loc_head_cells != grid.n_cells:
            tv = self._train_visits
            self._loc_head = fit_classifier_head(self._visit_state(tv), tv.tgt_cell, grid.n_cells, self.head_cfg,
                                                 hidden=256, class_weighted=False)
            self._loc_head_cells = grid.n_cells
        return LocationPrediction(grid_scores=predict_head(self._loc_head, self._visit_state(visits)))

    def predict_continuous(self, visits: VisitBatch, target: str) -> ContinuousPrediction:
        import torch
        from ..nn.common import GMMHead, fit, resolve_device, set_seed
        self._need_train_visits()
        if target not in self._time_heads:
            tv = self._train_visits
            y_s = tv.tgt_travel_time_s if target == "travel_time" else tv.tgt_duration_s
            ok = np.isfinite(y_s) & (y_s > 0)
            Z = self._T(self._visit_state(tv)[ok])
            mu, sd = Z.mean(0, keepdim=True), Z.std(0, keepdim=True) + 1e-6
            y = self._T(np.log(y_s[ok] / 3600.0))
            set_seed(self.head_cfg.seed)
            head = GMMHead(Z.shape[1]).to(self.device)
            rng = np.random.default_rng(self.head_cfg.seed)
            perm = rng.permutation(len(y)); n_val = len(y) // 7
            va, tr = perm[:n_val], perm[n_val:]
            loss = lambda i, training: GMMHead.nll(head(((Z - mu) / sd)[(tr if training else va)[i]]), y[(tr if training else va)[i]])
            fit(head, len(tr), len(va), loss, self.head_cfg, drop_last=False)
            self._time_heads[target] = (head, mu, sd)
        head, mu, sd = self._time_heads[target]
        with torch.no_grad():
            w, m, s = (a.cpu().numpy() for a in head((self._T(self._visit_state(visits)) - mu) / sd))
        return ContinuousPrediction(mixture=Mixture(w, m, s, space="log").rescale(3600.0))

    # ------------------------------------------------------------------ pre-training
    @staticmethod
    def pair_windows_with_visits(windows: TrajectoryBatch, staypoints, context: int, proj: LocalProjection):
        """For each window: the user's last `context` staypoints that ended before the window started."""
        from ..nn.features import VISIT_TOKEN_DIM, visit_tokens
        N = len(windows)
        tok = np.zeros((N, context, VISIT_TOKEN_DIM), np.float32)
        pad = np.ones((N, context), bool)
        groups = {u: g.sort_values("t_leave") for u, g in staypoints.groupby("user_id")}
        for i in range(N):
            g = groups.get(windows.user_id[i])
            if g is None:
                continue
            k = np.searchsorted(g.t_leave.to_numpy(), windows.t[i, 0], side="right")
            h = g.iloc[max(0, k - context):k]
            if len(h) == 0:
                continue
            n = len(h)
            tok[i, context - n:] = visit_tokens(h.lat.to_numpy()[None], h.lon.to_numpy()[None], h.t_arrive.to_numpy()[None],
                                                h.t_leave.to_numpy()[None], proj)[0]
            pad[i, context - n:] = False
        return tok, pad

    @classmethod
    def pretrain(cls, ctx, train: Optional[dict] = None, out: Optional[str] = None, arch: Optional[dict] = None,
                 init_from: Optional[str] = None, clip_weight: float = 1.0, prediction_weight: float = 1.0,
                 min_visits: int = 2, extra_point_features=None, extra_dim: int = 0, **kw) -> "CLIPMobilityAdapter":
        import pandas as pd
        import torch
        import torch.nn.functional as F
        from ..nn.common import TrainConfig
        from ..nn.common import fit as fit_loop
        cfg = TrainConfig.from_dict({"lr": 1e-4, "batch_size": 64, **(train or {})})
        kw.pop("device", None)                     # the training device (train.device) is used for the adapter too
        all_sp = pd.concat([ctx.staypoints[s] for s in ("train", "val", "test")], ignore_index=True)
        if init_from:
            ad = cls.from_checkpoint(init_from, extra_point_features=extra_point_features, device=cfg.device, **kw)
        else:
            sp = ctx.staypoints["train"]
            ad = cls(arch=arch, visit_center=(float(sp.lat.mean()), float(sp.lon.mean())),
                     visit_context=ctx.cfg.visit_context, extra_point_features=extra_point_features, extra_dim=extra_dim,
                     device=cfg.device, **kw)
        if ctx.windows["train"].length > ad.arch["trajectory_max_seq_length"]:
            raise ValueError("window_length exceeds trajectory_max_seq_length")

        def build(split):
            w = ctx.windows[split]
            vt, vp = cls.pair_windows_with_visits(w, all_sp, ad.visit_context, ad.visit_proj)
            keep = (~vp).sum(1) >= min_visits
            log.info(f"{split}: {keep.sum()}/{len(w)} windows paired with >= {min_visits} previous visits")
            w = w.take(np.where(keep)[0])
            return (torch.as_tensor(ad._point_tokens(w), device=ad.device), torch.as_tensor(vt[keep], device=ad.device),
                    torch.as_tensor(vp[keep], device=ad.device))

        data = {True: build("train"), False: build("val")}

        def loss_fn(idx, training):
            pt, vt, vp = (a[idx] for a in data[training])
            outp = ad.net(pt[:, :-1], vt, None, vp, compute_alignment=True)
            pred = F.mse_loss(outp["trajectory_predictions"], pt[:, 1:])
            return prediction_weight * pred + clip_weight * outp["clip_loss"]

        ad.provenance = ctx.provenance(init_from=init_from)
        ad.net.train()
        history = fit_loop(ad.net, len(data[True][0]), len(data[False][0]), loss_fn, cfg)
        ad.invalidate_cache()
        if out:
            ad.save(out, history)
        return ad
