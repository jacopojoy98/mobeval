"""TrajGPT adapter: next-visit location, travel time and duration (Gaussian mixtures),
autoregressive visit generation, and (re)training on the pipeline's visit sequences.

    adapter = TrajGPTAdapter.train(ctx, out="ckpt/trajgpt.pt")
    adapter = TrajGPTAdapter.from_checkpoint("ckpt/trajgpt.pt")

Conventions reproduced from the original code: region vocabulary with 4 special tokens,
x/y in metres around the data centroid, arrival/departure in days since a reference time,
travel time and duration in HOURS, clipped at the train 99th percentile, 3-component GMMs.

Unconditional time predictions. The travel/duration heads are conditioned on the target
visit (TrajGPT's factorisation). When the pipeline hides the target location, the adapter
marginalises over the top-k predicted regions: the returned distribution is the mixture
sum_k p(region_k) * p(time | region_k), itself a Gaussian mixture. For duration, the
unknown arrival time is set to the last departure plus the median predicted travel time.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from ..data import MobilityDataset, VisitBatch
from ..geo import LocalProjection
from ..metrics.probabilistic import Mixture
from .base import CONTINUOUS, GENERATION, NEXT_LOCATION, ContinuousPrediction, LocationPrediction
from .torch_base import TorchAdapter

log = logging.getLogger("mobeval.adapters.trajgpt")
DEFAULT_ARCH = dict(num_heads=2, num_layers=4, num_gaussians=3, d_feedforward=32, d_embed=32, lambda_min=1.0,
                    input_order="fixed")


class TrajGPTAdapter(TorchAdapter):
    name = "TrajGPT"
    model_type = "trajgpt"
    capabilities = {NEXT_LOCATION, CONTINUOUS, GENERATION}

    def __init__(self, tokenizer, proj: LocalProjection, t0: float, max_travel_h: float, max_duration_h: float,
                 lambda_max: float, sequence_len: int, arch: Optional[dict] = None, top_k_regions: int = 5,
                 min_scale_h: float = 1.0 / 60, time_reference: str = "week", max_valid_travel_h: float = 4.0, **kw):
        super().__init__(**kw)
        from ..nn.trajgpt_net import TrajGPT
        self.arch = {**DEFAULT_ARCH, **(arch or {})}
        self.tok, self.proj, self.t0 = tokenizer, proj, float(t0)
        self.max_travel_h, self.max_duration_h = float(max_travel_h), float(max_duration_h)
        self.lambda_max, self.sequence_len, self.top_k = float(lambda_max), int(sequence_len), int(top_k_regions)
        self.min_scale_h = float(min_scale_h)
        if time_reference not in ("week", "global"):
            raise ValueError("time_reference must be 'week' or 'global'")
        self.time_reference = time_reference
        self.max_valid_travel_h = float(max_valid_travel_h)
        self.net = TrajGPT(self.tok.n_regions, self.sequence_len, self.lambda_max, **self.arch).to(self.device).eval()
        if self.arch["input_order"] == "legacy":
            log.warning("TrajGPT legacy input order: the original time heads read the target's own arrival/"
                        "departure. mobeval never provides them, so predictions are honest but the model was "
                        "trained with that leak; retrain with input_order='fixed' for meaningful time metrics.")

    # ------------------------------------------------------------------ persistence
    def _meta(self):
        return {"tokenizer": self.tok.state(), "proj": [self.proj.lat0, self.proj.lon0], "t0": self.t0,
                "max_travel_h": self.max_travel_h, "max_duration_h": self.max_duration_h,
                "lambda_max": self.lambda_max, "sequence_len": self.sequence_len, "min_scale_h": self.min_scale_h,
                "time_reference": self.time_reference, "max_valid_travel_h": self.max_valid_travel_h,
                "provenance": self.provenance}

    def save(self, path, history=None):
        from ..nn.common import save_checkpoint
        save_checkpoint(path, self.net, self.model_type, {"arch": self.arch}, self._meta(), history)

    @classmethod
    def from_checkpoint(cls, path, **kw) -> "TrajGPTAdapter":
        from ..nn.common import load_checkpoint
        from ..nn.features import RegionTokenizer
        ck = load_checkpoint(path)
        if ck["format"] == "raw":
            raise ValueError("raw TrajGPT state dicts carry no vocabulary/scales; use from_original_state_dict(...)")
        m = ck["meta"]
        ad = cls(RegionTokenizer.from_state(m["tokenizer"]), LocalProjection(*m["proj"]), m["t0"], m["max_travel_h"],
                 m["max_duration_h"], m["lambda_max"], m["sequence_len"], arch=ck["config"]["arch"],
                 min_scale_h=m.get("min_scale_h", 1.0 / 60), time_reference=m.get("time_reference", "global"),
                 max_valid_travel_h=m.get("max_valid_travel_h", 4.0), **kw)
        ad.net.load_state_dict(ck["state_dict"], strict=True)
        ad.net.eval()
        ad.provenance = m.get("provenance", {})
        return ad

    @classmethod
    def from_original_state_dict(cls, path, region_h3_cells, lambda_max, max_travel_h, max_duration_h, t0, center_latlon,
                                 h3_resolution=7, sequence_len=128, arch=None, **kw) -> "TrajGPTAdapter":
        """Load a state dict saved from the ORIGINAL repository. `region_h3_cells` must list the H3 cells in
        region-id order (region_id - 4), as produced by its preprocessing (category codes of `h3_index`)."""
        import torch
        from ..nn.features import RegionTokenizer
        tok = RegionTokenizer("h3", h3_resolution=h3_resolution)
        tok.keys = np.asarray(region_h3_cells)
        tok._build()
        ad = cls(tok, LocalProjection(*center_latlon), t0, max_travel_h, max_duration_h, lambda_max, sequence_len,
                 arch={**(arch or {}), "input_order": "legacy"}, min_scale_h=1e-6, time_reference="global", **kw)
        ad.net.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
        return ad

    # ------------------------------------------------------------------ encoding
    def _sequence(self, lat, lon, ta, tl):
        """Arrays (N, S) -> model input dict of tensors."""
        import torch
        x, y = self.proj.to_xy(lat, lon)
        ta, tl = np.asarray(ta, float), np.asarray(tl, float)
        if self.time_reference == "global":        # original: days since the dataset's first arrival
            ref = self.t0
        else:                                      # Monday 00:00 UTC before each sequence's first visit
            day = np.floor(ta[:, :1] / 86400.0)
            ref = (day - (day + 3) % 7) * 86400.0  # 1970-01-01 was a Thursday
        T = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=self.device)
        return {"region_id": T(self.tok.tokens(lat, lon), torch.long), "x": T(x), "y": T(y),
                "arrival_time": T((ta - ref) / 86400.0), "departure_time": T((tl - ref) / 86400.0)}

    def _with_target(self, v: VisitBatch, lat, lon, ta, tl):
        cat = lambda c, t: np.concatenate([c, np.asarray(t, float)[:, None]], 1)
        return (cat(v.ctx_lat, lat), cat(v.ctx_lon, lon), cat(v.ctx_t_arrive, ta), cat(v.ctx_t_leave, tl))

    def _check_len(self, C):
        if C + 1 > self.sequence_len:
            raise ValueError(f"visit context {C} + 1 exceeds the model sequence length {self.sequence_len}")

    # ------------------------------------------------------------------ capabilities
    def predict_location(self, visits: VisitBatch, grid) -> LocationPrediction:
        import torch
        self._check_len(visits.ctx_lat.shape[1])
        out = []
        with torch.no_grad():
            for s in range(0, len(visits), self.batch_size):
                v = _slice(visits, slice(s, s + self.batch_size))
                last = lambda a: a[:, -1]
                inp = self._sequence(*self._with_target(v, last(v.ctx_lat), last(v.ctx_lon), last(v.ctx_t_leave),
                                                        last(v.ctx_t_leave)))
                out.append(self.net(inp)["region_id"][:, -1, 4:].float().cpu().numpy())
        return LocationPrediction(token_scores=np.concatenate(out), token_latlon=self.tok.token_latlon())

    def predict_continuous(self, visits: VisitBatch, target: str) -> ContinuousPrediction:
        import torch
        self._check_len(visits.ctx_lat.shape[1])
        W, MU, SD = [], [], []
        with torch.no_grad():
            for s in range(0, len(visits), self.batch_size):
                w, mu, sd = self._continuous_batch(_slice(visits, slice(s, s + self.batch_size)), target, torch)
                W.append(w); MU.append(mu); SD.append(sd)
        mix_hours = Mixture(np.concatenate(W), np.concatenate(MU), np.concatenate(SD), space="linear")
        return ContinuousPrediction(mixture=mix_hours.rescale(3600.0))

    def _continuous_batch(self, v, target, torch):
        N = len(v)
        last_dep = v.ctx_t_leave[:, -1]
        known_loc = np.isfinite(v.tgt_lat).all()
        if known_loc:
            cand_lat, cand_lon, p = v.tgt_lat[:, None], v.tgt_lon[:, None], np.ones((N, 1))
        else:
            logits = self.predict_location(v, None).token_scores
            k = min(self.top_k, logits.shape[1])
            top = np.argsort(-logits, 1)[:, :k]
            pr = np.exp(logits - logits.max(1, keepdims=True))
            p = np.take_along_axis(pr, top, 1)
            p = p / p.sum(1, keepdims=True)
            cand = self.tok.token_latlon()[top]                    # (N, k, 2)
            cand_lat, cand_lon = cand[..., 0], cand[..., 1]
        K = cand_lat.shape[1]
        rep = lambda a: np.repeat(a, K, axis=0)
        vk = VisitBatch(**{f: rep(getattr(v, f)) for f in VisitBatch.__dataclass_fields__})
        arrival = rep(v.tgt_t_arrive) if np.isfinite(v.tgt_t_arrive).all() else rep(last_dep)
        inp = self._sequence(*self._with_target(vk, cand_lat.ravel(), cand_lon.ravel(), arrival, arrival))
        out = self.net(inp)
        head = out["travel_time"]
        if target == "duration":
            if not np.isfinite(v.tgt_t_arrive).all():       # plug in arrival = last departure + median travel
                tw, tmu, tsd = (head[k][:, -1].float().cpu().numpy() for k in ("weight", "loc", "scale"))
                tm = Mixture(tw, tmu, np.maximum(tsd, self.min_scale_h))
                arrival = rep(last_dep) + np.clip(tm.median(), 0, self.max_travel_h) * 3600.0
                inp = self._sequence(*self._with_target(vk, cand_lat.ravel(), cand_lon.ravel(), arrival, arrival))
                out = self.net(inp)
            head = out["duration"]
        w, mu, sd = (head[k][:, -1].float().cpu().numpy() for k in ("weight", "loc", "scale"))
        sd = np.maximum(sd, self.min_scale_h)
        M = w.shape[1]
        w = (w / w.sum(1, keepdims=True)).reshape(N, K, M) * p[:, :, None]
        return w.reshape(N, K * M), mu.reshape(N, K * M), sd.reshape(N, K * M)

    def generate(self, reference: MobilityDataset, n_trajectories: int, seed: int = 0, n_visits: int = 12,
                 jitter_m: float = 30.0) -> MobilityDataset:
        """Seed with real visit sequences from TRAIN, then sample n_visits new visits per sequence.
        Emits two points per generated visit (arrival, departure) so staypoint detection recovers them."""
        import torch
        from ..data import make_visit_sequences
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        C = self.sequence_len - 1
        sp = self._reference_staypoints(reference)
        seeds = make_visit_sequences(sp, _GridShim(), min(C, self._seed_context), stride=1)
        pick = rng.choice(len(seeds), n_trajectories, replace=len(seeds) < n_trajectories)
        v = _slice(seeds, pick)
        lat, lon, ta, tl = v.ctx_lat.copy(), v.ctx_lon.copy(), v.ctx_t_arrive.copy(), v.ctx_t_leave.copy()
        latlon = self.tok.token_latlon()
        rows = []
        with torch.no_grad():
            for step in range(n_visits):
                inp = self._sequence(*(np.concatenate([a, a[:, -1:]], 1) for a in (lat, lon, ta, tl)))
                logits = self.net(inp)["region_id"][:, -1, 4:].float()
                reg = torch.multinomial(torch.softmax(logits, -1), 1).squeeze(1).cpu().numpy()
                nlat, nlon = latlon[reg, 0], latlon[reg, 1]
                inp = self._sequence(*(np.concatenate([a, b[:, None]], 1)
                                       for a, b in ((lat, nlat), (lon, nlon), (ta, tl[:, -1]), (tl, tl[:, -1]))))
                travel = self._sample(self.net(inp)["travel_time"], rng, self.max_travel_h)
                arr = tl[:, -1] + travel * 3600.0
                inp = self._sequence(*(np.concatenate([a, b[:, None]], 1)
                                       for a, b in ((lat, nlat), (lon, nlon), (ta, arr), (tl, arr))))
                dur = self._sample(self.net(inp)["duration"], rng, self.max_duration_h)
                dep = arr + dur * 3600.0
                lat, lon = np.concatenate([lat[:, 1:], nlat[:, None]], 1), np.concatenate([lon[:, 1:], nlon[:, None]], 1)
                ta, tl = np.concatenate([ta[:, 1:], arr[:, None]], 1), np.concatenate([tl[:, 1:], dep[:, None]], 1)
                j = rng.normal(0, jitter_m / 111_195.0, (len(reg), 2))
                for i in range(len(reg)):
                    for tt in (arr[i], dep[i]):
                        rows.append((v.user_id[i], f"trajgpt_gen{i}", tt, nlat[i] + j[i, 0], nlon[i] + j[i, 1]))
        return MobilityDataset(pd.DataFrame(rows, columns=["user_id", "traj_id", "t", "lat", "lon"]), "trajgpt_generated")

    _seed_context = 8

    def _reference_staypoints(self, reference):
        if getattr(self, "reference_staypoints", None) is not None:       # set by the pipeline (configured method)
            return self.reference_staypoints
        from ..data import detect_staypoints
        key = id(reference)
        if getattr(self, "_ref_key", None) != key:
            self._ref_sp, self._ref_key = detect_staypoints(reference), key
        return self._ref_sp

    def _sample(self, head, rng, max_h):
        w, mu, sd = (head[k][:, -1].float().cpu().numpy() for k in ("weight", "loc", "scale"))
        sd = np.maximum(sd, self.min_scale_h)
        return np.clip(Mixture(w, mu, sd).sample(1, int(rng.integers(1 << 31)))[:, 0], 0.0, max_h)

    # ------------------------------------------------------------------ training
    @classmethod
    def train(cls, ctx, train: Optional[dict] = None, out: Optional[str] = None, arch: Optional[dict] = None,
              tokenizer: Optional[dict] = None, init_from: Optional[str] = None, **kw) -> "TrajGPTAdapter":
        """Teacher-forced training on ctx.visits['train'] (context + target = one sequence), early
        stopping on ctx.visits['val']. Loss = region CE + travel NLL + duration NLL (paper), with:
        travel gaps above max_valid_travel_h excluded (TrajGPT's own "missing span" rule, applied to
        the loss too), a minimum GMM scale, and data-initialised GMM heads."""
        import torch
        import torch.nn.functional as F
        from ..nn.common import TrainConfig
        from ..nn.common import fit as fit_loop
        from ..nn.features import RegionTokenizer
        from ..nn.trajgpt_net import gmm_nll, init_gmm_head
        cfg = TrainConfig.from_dict({"lr": 1e-3, "batch_size": 64, "patience": 10, **(train or {})})
        kw.pop("device", None)                     # the training device (train.device) is used for the adapter too
        tr, va = ctx.visits["train"], ctx.visits["val"]
        C = tr.ctx_lat.shape[1]
        if init_from:
            ad = cls.from_checkpoint(init_from, device=cfg.device, **kw)
            ad._check_len(C)
        else:
            sp_train = ctx.staypoints["train"]
            tok = RegionTokenizer(**{"backend": "grid", "cell_m": 1000.0, **(tokenizer or {})}).fit(sp_train.lat, sp_train.lon)
            proj = LocalProjection.from_points(sp_train.lat, sp_train.lon)
            dur_h = (sp_train.t_leave - sp_train.t_arrive).to_numpy() / 3600.0
            travel_h = np.concatenate([tr.tgt_travel_time_s, (tr.ctx_t_arrive[:, 1:] - tr.ctx_t_leave[:, :-1]).ravel()]) / 3600.0
            x, y = proj.to_xy(sp_train.lat.to_numpy(), sp_train.lon.to_numpy())
            lam = float(np.hypot(np.ptp(x), np.ptp(y))) or 1000.0
            ad = cls(tok, proj, float(sp_train.t_arrive.min()), np.nanpercentile(travel_h, 99),
                     np.nanpercentile(dur_h, 99), lam, C + 1, arch=arch, device=cfg.device, **kw)
        log.info(f"TrajGPT: {ad.tok.n_regions} regions, seq_len {ad.sequence_len}, input_order {ad.arch['input_order']}, "
                 f"time_reference {ad.time_reference}, max duration {ad.max_duration_h:.1f} h")

        def tensors(v):
            lat, lon, ta, tl = ad._with_target(v, v.tgt_lat, v.tgt_lon, v.tgt_t_arrive, v.tgt_t_arrive + v.tgt_duration_s)
            travel = (ta[:, 1:] - tl[:, :-1]) / 3600.0
            T = lambda a: torch.as_tensor(a, dtype=torch.float32, device=ad.device)
            return ((lat, lon, ta, tl), T(np.clip(travel, 0, None)), T(travel <= ad.max_valid_travel_h),
                    T(np.clip((tl[:, 1:] - ta[:, 1:]) / 3600.0, 0, ad.max_duration_h)))

        data = {True: tensors(tr), False: tensors(va)}
        if not init_from:
            _, trav, ok, dur = data[True]
            init_gmm_head(ad.net.travel_head, trav[ok.bool()])
            init_gmm_head(ad.net.duration_head, dur.ravel())

        def loss_fn(idx, training):
            (lat, lon, ta, tl), travel, ok, dur = data[training]
            inp = ad._sequence(lat[idx], lon[idx], ta[idx], tl[idx])
            out = ad.net(inp)
            reg = out["region_id"]
            ce = F.cross_entropy(reg.reshape(-1, reg.shape[-1]), inp["region_id"][:, 1:].reshape(-1))
            ms = ad.min_scale_h
            t_nll = gmm_nll(out["travel_time"], travel[idx], ok[idx].bool(), ms)
            d_nll = gmm_nll(out["duration"], dur[idx], torch.ones_like(ok[idx], dtype=torch.bool), ms)
            return ce + (t_nll.mean() if t_nll.numel() else 0.0) + d_nll.mean()

        ad.provenance = ctx.provenance(init_from=init_from)
        ad.net.train()
        history = fit_loop(ad.net, len(tr), len(va), loss_fn, cfg)
        if out:
            ad.save(out, history)
        return ad


def _slice(v: VisitBatch, idx) -> VisitBatch:
    return VisitBatch(**{f: getattr(v, f)[idx] for f in VisitBatch.__dataclass_fields__})


class _GridShim:
    """make_visit_sequences needs a grid for ctx_cell; generation only uses coordinates."""
    n_cells = 1

    @staticmethod
    def cell_of(lat, lon):
        return np.zeros(np.shape(lat), int)
