"""TransferTraj adapter: masked trajectory recovery, embeddings, mode classification, pre-training.

    adapter = TransferTrajAdapter.pretrain(ctx, out="ckpt/transfertraj.pt")
    adapter = TransferTrajAdapter.from_checkpoint("ckpt/transfertraj.pt")

Conventions reproduced from the original code: metre coordinates relative to the first point of
each trajectory (the absolute first point is passed separately for the POI/road lookup), features
[x, y, timestamp, delta_t] each paired with a token channel, causal encoder with coordinate-driven
rotary attention, and the pre-training objective of span masking + per-point feature masking.

Deviations, all optional and recorded in the checkpoint:
  * projection - the original converts to UTM with pyproj; mobeval uses its own local metric
    projection (equirectangular around the data centre), which avoids the dependency, keeps metres,
    and stays within a fraction of a percent at city scale;
  * coord_scale - the original feeds raw metres, so the spatial MSE term starts around 1e5 and
    dominates the gradient. Dividing coordinates by `coord_scale` (and multiplying predictions back)
    changes nothing about the architecture; use 1.0 to reproduce the original exactly;
  * POI / road embeddings are optional. Without them the model keeps the same shape but the two
    context pathways contribute only their token embedding (see nn/transfertraj_net.py).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from ..data import TrajectoryBatch
from ..geo import LocalProjection
from .base import EMBEDDING, MODE_CLASSIFICATION, RECOVERY
from .torch_base import TorchAdapter

log = logging.getLogger("mobeval.adapters.transfertraj")
DEFAULT_ARCH = dict(embed_size=64, d_model=128, rafee_layer=2, poi_dist=100, rn_dist=100)
KNOWN_TOKEN, MASK_TOKEN, UNKNOWN_TOKEN, PAD_TOKEN = 0, 1, 4, 5


class TransferTrajAdapter(TorchAdapter):
    name = "TransferTraj"
    model_type = "transfertraj"
    capabilities = {RECOVERY, EMBEDDING, MODE_CLASSIFICATION}

    def __init__(self, arch: Optional[dict] = None, center=(0.0, 0.0), coord_scale: float = 1000.0,
                 context: Optional[dict] = None, pooling: str = "mean", **kw):
        super().__init__(**kw)
        import torch
        from ..nn.transfertraj_net import TransferTraj
        self.arch = {**DEFAULT_ARCH, **(arch or {})}
        self.proj = LocalProjection(*center)
        self.coord_scale = float(coord_scale)
        self.pooling = pooling
        self.context = dict(context or {})
        ctx = self._load_context()
        self.net = TransferTraj(**self.arch, **ctx).to(self.device).eval()
        self._torch = torch

    def _load_context(self) -> dict:
        """Optional POI / road-network features: .npy embeddings plus .npy (lat, lon) coordinates."""
        out = {}
        for kind in ("poi", "road"):
            emb, coo = self.context.get(f"{kind}_embed"), self.context.get(f"{kind}_latlon")
            if not (emb and coo):
                continue
            for f in (emb, coo):
                if not Path(f).exists():
                    raise FileNotFoundError(
                        f"{kind} context file {f} not found. Paths recorded at training time are absolute; "
                        f"override them with `adapter: {{context: {{...}}}}` in the config, or rebuild the "
                        f"features with `mobeval context`.")
            if True:
                e, c = np.load(emb), np.load(coo)
                if c.ndim != 2 or c.shape[1] != 2 or len(c) != len(e):
                    raise ValueError(f"{kind}: embeddings {e.shape} and coordinates {c.shape} must have the "
                                     f"same length, with coordinates shaped (N, 2) as (lat, lon)")
                x, y = self.proj.to_xy(c[:, 0], c[:, 1])
                out[f"{kind}_embed"] = np.asarray(e, np.float32)
                out[f"{kind}_coors"] = np.stack([x, y], 1).astype(np.float32) / self.coord_scale
                log.info(f"{kind}: {len(e)} entries with {e.shape[1]}-d embeddings")
        return out

    # ------------------------------------------------------------------ persistence
    def save(self, path, history=None, quiet: bool = False, complete: bool = True):
        from ..nn.common import save_checkpoint
        save_checkpoint(path, self.net, self.model_type, {"arch": self.arch},
                        {"center": [self.proj.lat0, self.proj.lon0], "coord_scale": self.coord_scale,
                         "context": self.context, "pooling": self.pooling, "provenance": self.provenance}, history,
                        quiet=quiet, complete=complete)

    @classmethod
    def from_checkpoint(cls, path, **kw) -> "TransferTrajAdapter":
        from ..nn.common import load_checkpoint
        ck = load_checkpoint(path)
        if ck["format"] == "raw":
            raise ValueError("raw TransferTraj state dicts do not record the projection or coordinate scale; "
                             "use from_original_state_dict(...)")
        m = ck["meta"]
        # kw wins over the stored values: context file paths in particular are absolute and move
        # between machines and jobs (staging to /scratch, for instance), so a config may override them.
        ad = cls(**{"arch": ck["config"]["arch"], "center": m["center"], "coord_scale": m["coord_scale"],
                    "context": m.get("context"), "pooling": m.get("pooling", "mean"), **kw})
        ad.net.load_state_dict(ck["state_dict"], strict=True)
        ad.net.eval()
        ad.provenance = m.get("provenance", {})
        return ad

    @classmethod
    def from_original_state_dict(cls, path, center, arch=None, context=None, **kw) -> "TransferTrajAdapter":
        """Load weights trained with the original repository. `center` is the (lat, lon) the metre
        coordinates are measured around; keep coord_scale=1.0, since the original feeds raw metres."""
        import torch
        ad = cls(arch=arch, center=center, coord_scale=1.0, context=context, **kw)
        ad.net.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
        ad.net.eval()
        return ad

    # ------------------------------------------------------------------ encoding
    def _encode(self, lat, lon, t, hidden):
        """-> input_seq (B, L, 4, 2), positions (B, L), first_point (B, 2) in scaled metres."""
        T = self._torch
        B, L = lat.shape
        x, y = self.proj.to_xy(np.nan_to_num(lat), np.nan_to_num(lon))
        x, y = x / self.coord_scale, y / self.coord_scale
        first = np.argmax(~hidden, 1)
        rows = np.arange(B)
        x0, y0, t0 = x[rows, first], y[rows, first], t[rows, first]
        feats = np.stack([np.where(hidden, 0.0, x - x0[:, None]), np.where(hidden, 0.0, y - y0[:, None]),
                          t, t - t0[:, None]], -1).astype(np.float32)
        tok = np.zeros((B, L, 4), np.float32)
        tok[..., 0] = tok[..., 1] = np.where(hidden, MASK_TOKEN, KNOWN_TOKEN)      # spatial masked
        seq = np.stack([feats, tok], -1)
        return (T.as_tensor(seq, device=self.device),
                T.as_tensor(np.tile(np.arange(L), (B, 1)), device=self.device).long(),
                T.as_tensor(np.stack([x0, y0], 1).astype(np.float32), device=self.device))

    # ------------------------------------------------------------------ capabilities
    def reconstruct(self, batch: TrajectoryBatch, mask: np.ndarray):
        self._require_net()
        lat_out, lon_out = batch.lat.copy(), batch.lon.copy()
        with self._torch.no_grad():
            for s in range(0, len(batch), self.batch_size):
                sl = slice(s, s + self.batch_size)
                seq, pos, fp = self._encode(batch.lat[sl], batch.lon[sl], batch.t[sl], mask[sl])
                _, mem = self.net(seq, pos, fp)
                pred = (self.net.pred(mem)[0] + fp.unsqueeze(1)).cpu().numpy() * self.coord_scale
                la, lo = self.proj.to_latlon(pred[..., 0], pred[..., 1])
                m = mask[sl]
                lat_out[sl], lon_out[sl] = np.where(m, la, batch.lat[sl]), np.where(m, lo, batch.lon[sl])
        return lat_out, lon_out

    def _embed_batch(self, batch):
        hidden = np.zeros(batch.lat.shape, bool)
        with self._torch.no_grad():
            seq, pos, fp = self._encode(batch.lat, batch.lon, batch.t, hidden)
            _, mem = self.net(seq, pos, fp)
            return (mem[:, -1] if self.pooling == "last" else mem.mean(1)).cpu().numpy()

    # ------------------------------------------------------------------ pre-training
    @staticmethod
    def _pretrain_masks(n, L, rng, span_div_ratio, span_mask_ratio, feature_mask_prob):
        """Span masking (both modalities) plus per-point single-modality masking, as in PretrainPadder."""
        span_hidden = np.zeros((n, L), bool)
        feat_hidden = np.zeros((n, L, 2), bool)                       # [spatial, temporal]
        for i in range(n):
            cuts = sorted({0, L} | set(rng.choice(L, int(np.ceil(L * span_div_ratio)), replace=False).tolist()))
            spans = list(zip(cuts[:-1], cuts[1:]))
            for j in rng.choice(len(spans), int(np.ceil(len(spans) * span_mask_ratio)), replace=False):
                lo, hi = spans[j]
                span_hidden[i, lo:hi] = True
            pick = rng.random(L) < feature_mask_prob
            spatial = rng.random(L) < 0.5
            feat_hidden[i, :, 0] = pick & spatial
            feat_hidden[i, :, 1] = pick & ~spatial
        feat_hidden[..., 0] |= span_hidden
        feat_hidden[..., 1] |= span_hidden
        return feat_hidden

    def _pretrain_batch(self, b: TrajectoryBatch, hidden2: np.ndarray):
        """Build (input_seq, target_seq) with independent spatial/temporal masking."""
        T = self._torch
        seq, pos, fp = self._encode(b.lat, b.lon, b.t, np.zeros(b.lat.shape, bool))
        target = seq.clone()
        inp = seq.clone()
        h = T.as_tensor(hidden2, device=self.device)
        sp, tp = h[..., 0], h[..., 1]
        for cols, m in ((slice(0, 2), sp), (slice(2, 4), tp)):
            inp[:, :, cols, 0] = T.where(m.unsqueeze(-1), T.zeros_like(inp[:, :, cols, 0]), inp[:, :, cols, 0])
            inp[:, :, cols, 1] = T.where(m.unsqueeze(-1), T.full_like(inp[:, :, cols, 1], MASK_TOKEN),
                                         inp[:, :, cols, 1])
            target[:, :, cols, 1] = T.where(m.unsqueeze(-1), T.full_like(target[:, :, cols, 1], UNKNOWN_TOKEN),
                                            target[:, :, cols, 1])
        return inp, target, pos, fp

    @classmethod
    def pretrain(cls, ctx, train: Optional[dict] = None, out: Optional[str] = None, arch: Optional[dict] = None,
                 init_from: Optional[str] = None, coord_scale: float = 1000.0, context: Optional[dict] = None,
                 span_div_ratio: float = 0.2, span_mask_ratio: float = 0.4, feature_mask_prob: float = 0.2,
                 **kw) -> "TransferTrajAdapter":
        """Span-masked pre-training (the original objective) on ctx.windows['train'], early stopping on 'val'."""
        from ..nn.common import TrainConfig
        from ..nn.common import fit as fit_loop
        cfg = TrainConfig.from_dict({"lr": 1e-3, "batch_size": 32, **(train or {})})
        kw.pop("device", None)
        if init_from:
            ad = cls.from_checkpoint(init_from, device=cfg.device, **kw)
        else:
            p = ctx.splits["train"].points
            ad = cls(arch=arch, center=(float(p.lat.mean()), float(p.lon.mean())), coord_scale=coord_scale,
                     context=context, device=cfg.device, **kw)
        tr, va = ctx.windows["train"], ctx.windows["val"]
        rng = np.random.default_rng(cfg.seed)

        def loss_fn(idx, training):
            b = (tr if training else va).take(idx)
            seed = rng if training else np.random.default_rng(int(idx[0]))
            hidden2 = ad._pretrain_masks(len(idx), b.length, seed, span_div_ratio, span_mask_ratio,
                                         feature_mask_prob if training else 0.0)
            return ad.net.loss(*ad._pretrain_batch(b, hidden2))

        ad.provenance = ctx.provenance(init_from=init_from)
        ad.net.train()
        history = fit_loop(ad.net, len(tr), len(va), loss_fn, cfg, on_best=ad.epoch_checkpointer(out))
        ad.invalidate_cache()
        if out:
            ad.save(out, history)
        return ad
