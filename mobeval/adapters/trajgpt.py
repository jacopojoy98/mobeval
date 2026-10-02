"""TrajGPT adapter: next-visit location, travel time and duration (Gaussian mixtures),
autoregressive visit generation, and (re)training on the pipeline's visit sequences.

    adapter = TrajGPTAdapter.train(ctx, out="ckpt/trajgpt.pt")
    adapter = TrajGPTAdapter.from_checkpoint("ckpt/trajgpt.pt")

Conventions reproduced from the original code: region vocabulary with 4 special tokens,
x/y in metres around the data centroid, arrival/departure in days since a reference time,
travel time and duration in HOURS, 3-component GMMs. Targets: durations are clipped at the TRAIN
99th percentile (the original uses the 99th percentile of all splits, i.e. it reads test data);
travel times are not clipped but gaps above `max_valid_travel_h` are masked out of the loss (the
original clips them at the p99 and excludes gaps over MAX_VALID_TRAVEL_TIME = 4 h from its metrics
only); the train-p99 travel time bounds sampling.

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
# `sequence_len` only sizes the pre-computed causal masks (no parameters depend on it, and the masks are
# non-persistent, so checkpoints are portable across values). It is kept at least this long - the
# original repository's RAW_SEQ_LEN - so a checkpoint is not locked to the context it was trained with.
MIN_SEQUENCE_LEN = 128


class TrajGPTAdapter(TorchAdapter):
    name = "TrajGPT"
    model_type = "trajgpt"
    capabilities = {NEXT_LOCATION, CONTINUOUS, GENERATION}

    def __init__(self, tokenizer, proj: LocalProjection, t0: float, max_travel_h: float, max_duration_h: float,
                 lambda_max: float, sequence_len: int, arch: Optional[dict] = None, top_k_regions: int = 5,
                 min_scale_h: float = 1.0 / 60, time_reference: str = "week", max_valid_travel_h: float = 4.0,
                 time_input_unit_s: float = 86400.0, travel_unit_h: float = 1.0, **kw):
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
        # Unit of the arrival/departure times fed to Time2Vec: days in the original since commit
        # 2d47f78, hours in the paper-era code (49aad40). Targets are in hours either way.
        self.time_input_unit_s = float(time_input_unit_s)
        self.travel_unit_h = float(travel_unit_h)
        self._warned_context = False
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
                "time_input_unit_s": self.time_input_unit_s, "travel_unit_h": self.travel_unit_h,
                "provenance": self.provenance}

    def save(self, path, history=None, quiet: bool = False, complete: bool = True):
        from ..nn.common import save_checkpoint
        save_checkpoint(path, self.net, self.model_type, {"arch": self.arch}, self._meta(), history,
                        quiet=quiet, complete=complete)

    @classmethod
    def from_checkpoint(cls, path, **kw) -> "TrajGPTAdapter":
        from ..nn.common import load_checkpoint
        from ..nn.features import RegionTokenizer
        ck = load_checkpoint(path)
        if ck["format"] == "raw":
            raise ValueError("raw TrajGPT state dicts carry no vocabulary/scales; use from_original_state_dict(...)")
        m = ck["meta"]
        ad = cls(RegionTokenizer.from_state(m["tokenizer"]), LocalProjection(*m["proj"]), m["t0"], m["max_travel_h"],
                 m["max_duration_h"], m["lambda_max"], m["sequence_len"],
                 **{"arch": ck["config"]["arch"], "min_scale_h": m.get("min_scale_h", 1.0 / 60),
                    "time_reference": m.get("time_reference", "global"),
                    "max_valid_travel_h": m.get("max_valid_travel_h", 4.0),
                    "time_input_unit_s": m.get("time_input_unit_s", 86400.0),
                    "travel_unit_h": m.get("travel_unit_h", 1.0), **kw})
        ad.net.load_state_dict(ck["state_dict"], strict=True)
        ad.net.eval()
        ad.provenance = m.get("provenance", {})
        return ad

    @classmethod
    def from_original_state_dict(cls, path, region_h3_cells, lambda_max, max_travel_h, max_duration_h, t0, center_latlon,
                                 h3_resolution=7, sequence_len=128, arch=None, revision="cf959ca",
                                 **kw) -> "TrajGPTAdapter":
        """Load a state dict saved from the ORIGINAL repository. `region_h3_cells` must list the H3 cells in
        region-id order (region_id - 4), as produced by its preprocessing (category codes of `h3_index`).

        The vocabulary sizes are read from the state dict, not derived from `region_h3_cells`: the original
        code adds the 4 special tokens twice (main.py and the modules), so the embedding has nr+8 rows and
        the head nr+8 outputs at HEAD, nr+4 at the paper-era commit 49aad40. Either loads.

        `revision` names the code that trained the checkpoint. The versions differ in ways the weights
        cannot reveal, so it must be stated - a wrong value loads without error and silently changes the
        predictions:
          "49aad40"  paper era (and earlier): Space2Vec scales g^(s/S - 1) (an operator-precedence slip),
                     Time2Vec inputs in hours, travel time and duration in hours, head nr+4
          "2d47f78"  2d47f78 and b9f1ae2: Space2Vec g^(s/(S-1)), inputs in days, duration in hours but
                     travel time in DAYS, head nr+4
          "cf959ca"  cf959ca up to HEAD (276fdee): inputs in days, both targets in hours, head nr+8
                     ("HEAD" is accepted as an alias)"""
        import torch
        from ..nn.features import RegionTokenizer
        from ..nn.trajgpt_net import N_SPECIAL_TOKENS
        sd = torch.load(path, map_location="cpu")
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) and "state_dict" in sd else sd
        emb_rows = int(sd["input.region_embedding.weight"].shape[0])
        head_rows = int(sd["region_id_head.weight"].shape[0])
        usable = min(emb_rows, head_rows) - N_SPECIAL_TOKENS
        if len(region_h3_cells) > usable:
            raise ValueError(f"{len(region_h3_cells)} H3 cells given, but the checkpoint has room for {usable} "
                             f"regions (embedding {emb_rows} rows, head {head_rows} outputs): wrong vocabulary?")
        if len(region_h3_cells) < usable - N_SPECIAL_TOKENS:
            log.warning(f"checkpoint has room for {usable} regions but only {len(region_h3_cells)} H3 cells were "
                        f"given; is this the vocabulary it was trained with?")
        tok = RegionTokenizer("h3", h3_resolution=h3_resolution)
        tok.keys = np.asarray(region_h3_cells)
        tok._build()
        revision = {"HEAD": "cf959ca"}.get(revision, revision)
        if revision not in ("49aad40", "2d47f78", "cf959ca"):
            raise ValueError("revision must be '49aad40', '2d47f78' or 'cf959ca' (alias 'HEAD')")
        arch = {**(arch or {}), "input_order": "legacy", "embedding_rows": emb_rows, "head_rows": head_rows,
                "space2vec_exponent": "s/S-1" if revision == "49aad40" else "s/(S-1)"}
        ad = cls(tok, LocalProjection(*center_latlon), t0, max_travel_h, max_duration_h, lambda_max, sequence_len,
                 arch=arch, min_scale_h=1e-6, time_reference="global",
                 time_input_unit_s=3600.0 if revision == "49aad40" else 86400.0,
                 travel_unit_h=24.0 if revision == "2d47f78" else 1.0, **kw)
        ad.net.load_state_dict(sd, strict=True)
        ad.net.eval()
        ad.provenance = {"source": "original_repository", "revision": revision, "path": str(path)}
        return ad

    def _region_logits(self, out):
        """Scores over the tokenizer's regions only: drops the 4 special tokens in front and any unused
        rows a checkpoint from the original repository carries behind (see from_original_state_dict)."""
        return out["region_id"][:, -1, 4:4 + self.tok.n_regions]

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
                "arrival_time": T((ta - ref) / self.time_input_unit_s),
                "departure_time": T((tl - ref) / self.time_input_unit_s)}

    def _with_target(self, v: VisitBatch, lat, lon, ta, tl):
        cat = lambda c, t: np.concatenate([c, np.asarray(t, float)[:, None]], 1)
        return (cat(v.ctx_lat, lat), cat(v.ctx_lon, lon), cat(v.ctx_t_arrive, ta), cat(v.ctx_t_leave, tl))

    def _check_len(self, C):
        if C + 1 > self.sequence_len:
            raise ValueError(f"visit context {C} + 1 exceeds the model sequence length {self.sequence_len}")
        trained = self.provenance.get("train_context")
        if trained is not None and trained != C and not self._warned_context:
            self._warned_context = True
            log.warning(f"evaluating with a context of {C} visits, but this checkpoint was trained with "
                        f"{trained}. The model sees sequence positions it never saw in training; retrain "
                        f"with the same `visit_context` for a like-for-like comparison.")

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
                out.append(self._region_logits(self.net(inp)).float().cpu().numpy())
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
                tw, tmu, tsd = self._gmm(head, "travel_time")
                tm = Mixture(tw, tmu, np.maximum(tsd, self.min_scale_h))
                arrival = rep(last_dep) + np.clip(tm.median(), 0, self.max_travel_h) * 3600.0
                inp = self._sequence(*self._with_target(vk, cand_lat.ravel(), cand_lon.ravel(), arrival, arrival))
                out = self.net(inp)
            head = out["duration"]
        w, mu, sd = self._gmm(head, target)
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
                logits = self._region_logits(self.net(inp)).float()
                reg = torch.multinomial(torch.softmax(logits, -1), 1).squeeze(1).cpu().numpy()
                nlat, nlon = latlon[reg, 0], latlon[reg, 1]
                inp = self._sequence(*(np.concatenate([a, b[:, None]], 1)
                                       for a, b in ((lat, nlat), (lon, nlon), (ta, tl[:, -1]), (tl, tl[:, -1]))))
                travel = self._sample(self.net(inp)["travel_time"], rng, self.max_travel_h, "travel_time")
                arr = tl[:, -1] + travel * 3600.0
                inp = self._sequence(*(np.concatenate([a, b[:, None]], 1)
                                       for a, b in ((lat, nlat), (lon, nlon), (ta, arr), (tl, arr))))
                dur = self._sample(self.net(inp)["duration"], rng, self.max_duration_h, "duration")
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

    def _gmm(self, head, kind):
        """Last-position GMM parameters in HOURS. Checkpoints of the original's commits 2d47f78..b9f1ae2
        predict travel time in days (`travel_unit_h` = 24)."""
        w, mu, sd = (head[k][:, -1].float().cpu().numpy() for k in ("weight", "loc", "scale"))
        u = self.travel_unit_h if kind == "travel_time" else 1.0
        return w, mu * u, sd * u

    def _sample(self, head, rng, max_h, kind):
        w, mu, sd = self._gmm(head, kind)
        sd = np.maximum(sd, self.min_scale_h)
        return np.clip(Mixture(w, mu, sd).sample(1, int(rng.integers(1 << 31)))[:, 0], 0.0, max_h)

    # ------------------------------------------------------------------ training
    @classmethod
    def train(cls, ctx, train: Optional[dict] = None, out: Optional[str] = None, arch: Optional[dict] = None,
              tokenizer: Optional[dict] = None, init_from: Optional[str] = None, sequences: str = "context",
              seq_len: int = 128, instance_stride: int = 1, travel_loss_mask: bool = True,
              init_gmm: bool = True, clip_travel: bool = False, **kw) -> "TrajGPTAdapter":
        """Teacher-forced training, early stopping on the validation split. Loss = region CE + travel
        NLL + duration NLL (paper).

        sequences="context" (default): one sample per ctx.visits entry (visit_context visits + target).
        sequences="original": the original repository's instances - per user, a window of `seq_len`
        visits starting at every `instance_stride`-th position that still has seq_len visits after it,
        plus each user's first window, shorter ones left-padded (utils/preprocess.py,
        split_indices_for_next_prediction). Built lazily, so memory does not grow with seq_len.

        travel_loss_mask: exclude travel gaps above max_valid_travel_h from the loss (the original
        excludes them from its metrics only). init_gmm: start the GMM heads at the data (the original
        does not). clip_travel: clip travel-time targets at the train 99th percentile, as the original
        does (durations are always clipped). In every mode, loss positions whose visit belongs to
        another split are excluded."""
        import torch
        import torch.nn.functional as F
        from ..nn.common import TrainConfig
        from ..nn.common import fit as fit_loop
        from ..nn.features import RegionTokenizer
        from ..nn.trajgpt_net import gmm_nll, init_gmm_head
        cfg = TrainConfig.from_dict({"lr": 1e-3, "batch_size": 64, "patience": 10, **(train or {})})
        kw.pop("device", None)                     # the training device (train.device) is used for the adapter too
        if sequences not in ("context", "original"):
            raise ValueError("sequences must be 'context' or 'original'")
        tr, va = ctx.visits.get("train"), ctx.visits.get("val")
        if sequences == "context" and (tr is None or va is None):
            raise ValueError("no train/val visit sequences: lower eval.visit_context or check the staypoints")
        C = tr.ctx_lat.shape[1] if sequences == "context" else seq_len - 1
        if init_from:
            ad = cls.from_checkpoint(init_from, device=cfg.device, **kw)
            ad._check_len(C)
        else:
            sp_train = ctx.staypoints["train"]
            if len(sp_train) < 2:
                raise ValueError("TrajGPT needs staypoints in the train split and found none: check "
                                 "eval.staypoint_method / staypoint_dist_m / staypoint_time_s (and eval.tasks, "
                                 "which skips staypoint detection when no visit task is selected)")
            tok = RegionTokenizer(**{"backend": "grid", "cell_m": 1000.0, **(tokenizer or {})}).fit(sp_train.lat, sp_train.lon)
            proj = LocalProjection.from_points(sp_train.lat, sp_train.lon)
            dur_h = (sp_train.t_leave - sp_train.t_arrive).to_numpy() / 3600.0
            if tr is not None:
                travel_h = np.concatenate([tr.tgt_travel_time_s,
                                           (tr.ctx_t_arrive[:, 1:] - tr.ctx_t_leave[:, :-1]).ravel()]) / 3600.0
            else:                                      # consecutive train staypoints of each user
                g = sp_train.sort_values(["user_id", "t_arrive"])
                same = g.user_id.eq(g.user_id.shift()).to_numpy()
                travel_h = ((g.t_arrive - g.t_leave.shift()).to_numpy() / 3600.0)[same]
            x, y = proj.to_xy(sp_train.lat.to_numpy(), sp_train.lon.to_numpy())
            lam = float(np.hypot(np.ptp(x), np.ptp(y))) or 1000.0
            ad = cls(tok, proj, float(sp_train.t_arrive.min()), np.nanpercentile(travel_h, 99),
                     np.nanpercentile(dur_h, 99), lam, max(C + 1, MIN_SEQUENCE_LEN),
                     arch=arch, device=cfg.device, **kw)
        if sequences == "original":
            return cls._train_original(ad, ctx, cfg, out, init_from, seq_len, instance_stride, travel_loss_mask,
                                       init_gmm, clip_travel)
        log.info(f"TrajGPT: {ad.tok.n_regions} regions, visit_context {C} ({C} targets per sample), "
                 f"mask capacity {ad.sequence_len}, input_order {ad.arch['input_order']}, "
                 f"time_reference {ad.time_reference}, max duration {ad.max_duration_h:.1f} h")

        def tensors(v):
            lat, lon, ta, tl = ad._with_target(v, v.tgt_lat, v.tgt_lon, v.tgt_t_arrive, v.tgt_t_arrive + v.tgt_duration_s)
            travel = (ta[:, 1:] - tl[:, :-1]) / 3600.0
            T = lambda a: torch.as_tensor(a, dtype=torch.float32, device=ad.device)
            # Output position k predicts visit k+1 of [ctx_0 .. ctx_{C-1}, target]. The target is in
            # this split by construction; context visits may come from another split, and a loss
            # taken on them would train on (or early-stop on) visits that are not this split's.
            own = np.concatenate([v.ctx_in_split[:, 1:], np.ones((len(v), 1), bool)], 1)
            return ((lat, lon, ta, tl), T(np.clip(travel, 0, None)), T((travel <= ad.max_valid_travel_h) & own),
                    T(np.clip((tl[:, 1:] - ta[:, 1:]) / 3600.0, 0, ad.max_duration_h)), T(own))

        data = {True: tensors(tr), False: tensors(va)}
        if clip_travel:
            for k in data:
                d = list(data[k])
                d[1] = d[1].clamp(max=ad.max_travel_h)
                data[k] = tuple(d)
        if not travel_loss_mask:
            for k, v in ((True, tr), (False, va)):
                d = list(data[k])
                d[2] = d[4].clone()                 # every own-split position, however long the gap
                data[k] = tuple(d)
        n_out = int((~tr.ctx_in_split[:, 1:]).sum())
        if n_out:
            log.info(f"TrajGPT: {n_out:,} context positions belong to other splits and are excluded "
                     f"from the training loss (they remain visible as history)")
        if not init_from and init_gmm:
            _, trav, ok, dur, own = data[True]
            init_gmm_head(ad.net.travel_head, trav[ok.bool()])
            init_gmm_head(ad.net.duration_head, dur[own.bool()])

        def loss_fn(idx, training):
            (lat, lon, ta, tl), travel, ok, dur, own = data[training]
            inp = ad._sequence(lat[idx], lon[idx], ta[idx], tl[idx])
            out = ad.net(inp)
            reg = out["region_id"]
            w = own[idx]
            ce = F.cross_entropy(reg.reshape(-1, reg.shape[-1]), inp["region_id"][:, 1:].reshape(-1),
                                 reduction="none")
            ce = (ce * w.reshape(-1)).sum() / w.sum().clamp(min=1)
            ms = ad.min_scale_h
            t_nll = gmm_nll(out["travel_time"], travel[idx], ok[idx].bool(), ms)
            d_nll = gmm_nll(out["duration"], dur[idx], w.bool(), ms)
            return ce + (t_nll.mean() if t_nll.numel() else 0.0) + d_nll.mean()

        ad.provenance = ctx.provenance(init_from=init_from, train_context=C)
        ad.net.train()
        history = fit_loop(ad.net, len(tr), len(va), loss_fn, cfg, on_best=ad.epoch_checkpointer(out))
        if out:
            ad.save(out, history)
        return ad

    @classmethod
    def _train_original(cls, ad, ctx, cfg, out, init_from, seq_len, stride, travel_loss_mask, init_gmm, clip_travel):
        import torch
        import torch.nn.functional as F
        from ..nn.common import fit as fit_loop
        from ..nn.trajgpt_net import PAD, gmm_nll, init_gmm_head
        S = int(seq_len)
        if ad.sequence_len < S:
            raise ValueError(f"seq_len {S} exceeds the model's mask capacity {ad.sequence_len}")
        data = {k: _original_instances(ctx, k, S, int(stride)) for k in ("train", "val")}
        log.info(f"TrajGPT original instances: {len(data['train'][1]):,} train / {len(data['val'][1]):,} val "
                 f"windows of up to {S} visits (stride {stride}), left-padded")
        T = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=ad.device)

        def batch(split, idx):
            users, inst = data[split]
            B = len(idx)
            lat, lon = np.full((B, S), np.nan), np.full((B, S), np.nan)
            ta, tl = np.zeros((B, S)), np.zeros((B, S))
            real, own, first = np.zeros((B, S), bool), np.zeros((B, S), bool), np.zeros((B, S), bool)
            for r, i in enumerate(idx):
                u, s0, e = inst[i]
                la, lo, a, l, o = users[u]
                n = e - s0 + 1
                sl = slice(S - n, S)
                lat[r, sl], lon[r, sl], ta[r, sl], tl[r, sl] = la[s0:e + 1], lo[s0:e + 1], a[s0:e + 1], l[s0:e + 1]
                real[r, sl], own[r, sl] = True, o[s0:e + 1]
                first[r, S - n] = s0 == 0                   # the user's first visit: travel time 0
            fill = lambda a: np.where(real, a, np.nanmean(np.where(real, a, np.nan), 1, keepdims=True))
            j0 = (~real).sum(1)                              # first real position of each row
            first_t = lambda a: np.where(real, a, a[np.arange(B), j0][:, None])   # keeps the week reference right
            inp = ad._sequence(fill(lat), fill(lon), first_t(ta), first_t(tl))
            pad = T(~real, torch.bool)
            inp["region_id"] = inp["region_id"].masked_fill(pad, PAD)
            for k in ("x", "y", "arrival_time", "departure_time"):
                inp[k] = inp[k].masked_fill(pad, 0.0)
            travel = np.zeros((B, S - 1))
            prev_ok = real[:, :-1] & real[:, 1:]
            travel[prev_ok] = ((ta[:, 1:] - tl[:, :-1]) / 3600.0)[prev_ok]
            travel = np.clip(travel, 0, ad.max_travel_h if clip_travel else None)
            dur = np.clip((tl[:, 1:] - ta[:, 1:]) / 3600.0, 0, ad.max_duration_h)
            tgt = real[:, 1:] & own[:, 1:]
            ok = tgt & (prev_ok | first[:, 1:])
            if travel_loss_mask:
                ok &= travel <= ad.max_valid_travel_h
            return inp, T(travel), T(ok, torch.bool), T(dur), T(tgt, torch.bool)

        if not init_from and init_gmm:
            n = min(len(data["train"][1]), 20000)
            _, trav, ok, dur, tgt = batch("train", np.arange(n))
            init_gmm_head(ad.net.travel_head, trav[ok])
            init_gmm_head(ad.net.duration_head, dur[tgt])

        def loss_fn(idx, training):
            inp, travel, ok, dur, tgt = batch("train" if training else "val", idx)
            o = ad.net(inp)
            reg = o["region_id"]
            ce = F.cross_entropy(reg.reshape(-1, reg.shape[-1]), inp["region_id"][:, 1:].reshape(-1), reduction="none")
            w = tgt.float().reshape(-1)
            ce = (ce * w).sum() / w.sum().clamp(min=1)
            ms = ad.min_scale_h
            t_nll = gmm_nll(o["travel_time"], travel, ok, ms)
            d_nll = gmm_nll(o["duration"], dur, tgt, ms)
            return ce + (t_nll.mean() if t_nll.numel() else 0.0) + (d_nll.mean() if d_nll.numel() else 0.0)

        ad.provenance = ctx.provenance(init_from=init_from, train_context=S - 1)
        ad.net.train()
        history = fit_loop(ad.net, len(data["train"][1]), len(data["val"][1]), loss_fn, cfg,
                           on_best=ad.epoch_checkpointer(out))
        if out:
            ad.save(out, history)
        return ad


def _original_instances(ctx, split: str, seq_len: int, stride: int):
    """Per-user staypoint arrays + (user, start, end) instances with >= 1 target in `split`."""
    import pandas as pd
    sp = pd.concat([ctx.staypoints[s] for s in ctx.staypoints], ignore_index=True)
    sp = sp.drop_duplicates(["user_id", "t_arrive", "lat", "lon"]).sort_values(["user_id", "t_arrive"], kind="stable")
    own_ids = set(ctx.splits[split].points.traj_id.unique())
    users, inst = [], []
    for uid, g in sp.groupby("user_id", sort=False):
        own = g.traj_id.isin(own_ids).to_numpy()
        if not own.any() or len(g) < 2:          # the original keeps users with >= 2 visits
            continue
        u = len(users)
        users.append((g.lat.to_numpy(), g.lon.to_numpy(), g.t_arrive.to_numpy().astype(float),
                      g.t_leave.to_numpy().astype(float), own))
        n = len(g)
        starts = sorted({0} | set(range(0, max(0, n - seq_len), stride)))
        for s0 in starts:
            e = min(s0 + seq_len - 1, n - 1)
            pad = seq_len - (e - s0 + 1)
            first_target = s0 if pad else s0 + 1        # left padding makes the first real visit a target
            if own[first_target:e + 1].any():
                inst.append((u, s0, e))
    return users, inst


def _slice(v: VisitBatch, idx) -> VisitBatch:
    return VisitBatch(**{f: getattr(v, f)[idx] for f in VisitBatch.__dataclass_fields__})


class _GridShim:
    """make_visit_sequences needs a grid for ctx_cell; generation only uses coordinates."""
    n_cells = 1

    @staticmethod
    def cell_of(lat, lon):
        return np.zeros(np.shape(lat), int)
