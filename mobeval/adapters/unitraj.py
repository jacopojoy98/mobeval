"""UniTraj adapter: masked-trajectory recovery, embeddings, native mode head, and (re)training.

    adapter = UniTrajAdapter.from_checkpoint("model.pt")            # public checkpoint
    adapter = UniTrajAdapter.pretrain(ctx, out="ckpt/unitraj.pt")  # train from scratch on ctx train split
    adapter = UniTrajAdapter.pretrain(ctx, init_from="model.pt")   # continue from the public weights

Input conventions reproduced from the original code: (longitude, latitude) order, offsets
from the first point in degrees, z-normalised with fixed statistics, time intervals in
seconds, fixed length 200. Windows shorter than 200 are right-padded; padding positions are
always hidden from the encoder (never visible tokens), so they cannot distort predictions.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..data import TrajectoryBatch, make_mask
from .base import EMBEDDING, MODE_CLASSIFICATION, RECOVERY
from .torch_base import TorchAdapter

log = logging.getLogger("mobeval.adapters.unitraj")

DEFAULT_ARCH = dict(trajectory_length=200, patch_size=1, embedding_dim=128, encoder_layers=8, encoder_heads=4,
                    decoder_layers=4, decoder_heads=4)
PUBLIC_NORM = {"mean": [5.3311563533497974e-05, -7.49477039789781e-05],       # (lon, lat) offset, degrees
               "std": [0.049923088401556015, 0.040688566863536835]}


class UniTrajAdapter(TorchAdapter):
    name = "UniTraj"
    model_type = "unitraj"
    capabilities = {RECOVERY, EMBEDDING, MODE_CLASSIFICATION}

    def __init__(self, arch: Optional[dict] = None, norm: Optional[dict] = None, pooling: str = "cls", **kw):
        super().__init__(**kw)
        import torch
        from ..nn.unitraj_net import UniTraj
        self.arch = {**DEFAULT_ARCH, **(arch or {})}
        if self.arch["patch_size"] != 1:
            raise ValueError("only patch_size=1 is supported (padding would otherwise mix into patches)")
        self.norm = norm or PUBLIC_NORM
        self.pooling = pooling
        self.net = UniTraj(**self.arch).to(self.device).eval()
        self._torch = torch

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_checkpoint(cls, path: str, **kw) -> "UniTrajAdapter":
        from ..nn.common import load_checkpoint
        ck = load_checkpoint(path)
        ad = cls(arch=ck["config"].get("arch"), norm=ck["meta"].get("norm"), **kw)
        ad.net.load_state_dict(ck["state_dict"], strict=True)
        ad.net.eval()
        ad.provenance = ck["meta"].get("provenance", {})
        log.info(f"loaded UniTraj ({'mobeval' if ck['format'] != 'raw' else 'original'} checkpoint) from {path}")
        return ad

    def save(self, path: str, history=None):
        from ..nn.common import save_checkpoint
        save_checkpoint(path, self.net, self.model_type, {"arch": self.arch},
                        {"norm": self.norm, "provenance": self.provenance}, history)

    # ------------------------------------------------------------------ encoding
    def _encode(self, lat, lon, t, hidden):
        """-> traj [B,2,Lm], intervals [B,Lm], hidden_full [B,Lm], origin (lon0, lat0), window length."""
        B, L = lat.shape
        Lm = self.arch["trajectory_length"]
        if L > Lm:
            raise ValueError(f"window length {L} exceeds UniTraj's fixed length {Lm}; set EvalConfig.window_length <= {Lm}")
        first = np.argmax(~hidden, 1)
        lon0, lat0 = lon[np.arange(B), first], lat[np.arange(B), first]
        mean, std = np.asarray(self.norm["mean"]), np.asarray(self.norm["std"])
        x = np.zeros((B, 2, Lm), np.float32)
        x[:, 0, :L] = np.where(hidden, 0.0, ((np.nan_to_num(lon) - lon0[:, None]) - mean[0]) / std[0])
        x[:, 1, :L] = np.where(hidden, 0.0, ((np.nan_to_num(lat) - lat0[:, None]) - mean[1]) / std[1])
        iv = np.zeros((B, Lm), np.float32)
        iv[:, 1:L] = np.diff(t, axis=1)
        hid = np.ones((B, Lm), bool)
        hid[:, :L] = hidden
        T = self._torch
        return (T.as_tensor(x, device=self.device), T.as_tensor(iv, device=self.device), hid, lon0, lat0, L)

    def _decode(self, pred, lon0, lat0, L):
        mean, std = np.asarray(self.norm["mean"]), np.asarray(self.norm["std"])
        p = pred.detach().cpu().numpy()[:, :, :L]
        return p[:, 1] * std[1] + mean[1] + lat0[:, None], p[:, 0] * std[0] + mean[0] + lon0[:, None]

    # ------------------------------------------------------------------ capabilities
    def reconstruct(self, batch: TrajectoryBatch, mask: np.ndarray):
        self._require_net()
        rng = np.random.default_rng(0)
        lat_out, lon_out = np.empty_like(batch.lat), np.empty_like(batch.lon)
        with self._torch.no_grad():
            for s in range(0, len(batch), self.batch_size):
                sl = slice(s, s + self.batch_size)
                x, iv, hid, lon0, lat0, L = self._encode(batch.lat[sl], batch.lon[sl], batch.t[sl], mask[sl])
                pred, _ = self.net(x, iv, hid, rng)
                lat_out[sl], lon_out[sl] = self._decode(pred, lon0, lat0, L)
        return lat_out, lon_out

    def _embed_batch(self, batch):
        hidden = np.zeros(batch.lat.shape, bool)
        with self._torch.no_grad():
            x, iv, hid, *_ = self._encode(batch.lat, batch.lon, batch.t, hidden)
            return self.net.embed(x, iv, hid, self.pooling).cpu().numpy()

    # ------------------------------------------------------------------ training
    @classmethod
    def pretrain(cls, ctx, train: Optional[dict] = None, out: Optional[str] = None, init_from: Optional[str] = None,
                 arch: Optional[dict] = None, mask_ratio: float = 0.5, block_prob: float = 0.3,
                 fit_norm: Optional[bool] = None, **kw) -> "UniTrajAdapter":
        """Masked-reconstruction pre-training on ctx.windows['train'], early stopping on 'val'.
        With `init_from`, continues from existing weights and keeps their normalisation."""
        import torch
        from ..nn.common import TrainConfig
        from ..nn.common import fit as fit_loop
        cfg = TrainConfig.from_dict({"lr": 1e-3, "batch_size": 128, **(train or {})})
        kw.pop("device", None)                     # the training device (train.device) is used for the adapter too
        if init_from:
            ad = cls.from_checkpoint(init_from, device=cfg.device, **kw)
        else:
            ad = cls(arch=arch, device=cfg.device, **kw)
        tr, va = ctx.windows["train"], ctx.windows["val"]
        if fit_norm if fit_norm is not None else not init_from:
            off = np.stack([(tr.lon - tr.lon[:, :1]).ravel(), (tr.lat - tr.lat[:, :1]).ravel()], 1)
            ad.norm = {"mean": off.mean(0).tolist(), "std": (off.std(0) + 1e-9).tolist()}
            log.info(f"fitted normalisation on train windows: {ad.norm}")
        rng = np.random.default_rng(cfg.seed)
        L = tr.length

        def masks(n):
            m = make_mask(n, L, mask_ratio, "random", int(rng.integers(1 << 31)))
            is_block = rng.random(n) < block_prob
            if is_block.any():
                m[is_block] = make_mask(int(is_block.sum()), L, mask_ratio, "block", int(rng.integers(1 << 31)))
            return m

        def loss_fn(idx, training):
            b = (tr if training else va).take(idx)
            m = masks(len(idx)) if training else make_mask(len(idx), L, mask_ratio, "random", int(idx[0]))
            x_true, iv, _, _, _, _ = ad._encode(b.lat, b.lon, b.t, np.zeros_like(m))
            x_in, _, hid, _, _, _ = ad._encode(b.lat, b.lon, b.t, m)
            pred, _ = ad.net(x_in, iv, hid, rng)
            target_mask = torch.zeros(hid.shape, dtype=torch.bool, device=ad.device)
            target_mask[:, :L] = torch.as_tensor(m, device=ad.device)
            err = ((pred - x_true) ** 2).sum(1)                   # [B, Lm], normalised units
            return err[target_mask].mean()

        ad.provenance = ctx.provenance(init_from=init_from)
        ad.net.train()
        history = fit_loop(ad.net, len(tr), len(va), loss_fn, cfg)
        ad.invalidate_cache()
        if out:
            ad.save(out, history)
        return ad
