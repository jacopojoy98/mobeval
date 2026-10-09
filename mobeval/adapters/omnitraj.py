"""OmniTraj adapter (Zhu et al., KDD 2025; github.com/Yasoz/OmniTraj).

OmniTraj embeds a trajectory from four representations - the GPS points, its "topology" (critical
points), its road segments and the grid regions it crosses - with one encoder each, aligned by
contrastive learning. It has no decoder, so in mobeval it provides:

  embed()          the GPS trajectory's embedding (every linear-probe task, user identification,
                   anomaly detection, mode classification, similarity retrieval);
  embed_query()    the embedding of a topology / road / region query, or of a fusion of them, in the
                   same space - the paper's cross-modal and condition-based retrieval.

The network is the original code, vendored unmodified (mobeval/nn/omnitraj). The input
construction is reconstructed from the paper and the repository's sample data, because the
authors' preprocessing is not public (see nn/omnitraj_prep.py for what was verified and how).

Road segments need map matching (mobeval roads / mobeval mapmatch). Without a `roads_file`, the road
encoder is left out: the model trains and evaluates on trajectory, topology and regions, and this is
recorded in the checkpoint.

    adapter = OmniTrajAdapter.pretrain(ctx, out="ckpt/omnitraj.pt", roads_file="roads/matched.parquet")
"""
from __future__ import annotations

import copy
import logging
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..data import TrajectoryBatch
from ..nn import omnitraj_prep as prep
from .base import CROSS_MODAL, EMBEDDING, MODE_CLASSIFICATION
from .torch_base import TorchAdapter

log = logging.getLogger("mobeval.adapters.omnitraj")

# utils/config.py of the original repository (paths removed; num_roads / num_grids set from the data)
ENCODER_CONFIGS = {
    "trajectory": {"input_dim": 2, "seq_length": 200, "patch_size": 5, "embed_dim": 256, "depth": 6, "num_heads": 8,
                   "drop_rate": 0.1, "attn_drop_rate": 0.1, "drop_path_rate": 0.1, "pooling_strategy": "cls",
                   "patch_embed_type": "linear"},
    "topology": {"input_dim": 2, "embed_dim": 256, "num_heads": 8, "num_layers": 6, "dropout": 0.1,
                 "max_position_embeddings": 256, "pooling_strategy": "cls"},
    "road": {"num_roads": 8000, "output_dim": 256, "embed_dim": 256, "num_heads": 8, "num_layers": 6, "dropout": 0.1,
             "pooling_strategy": "cls"},
    "region": {"num_grids": 256, "output_dim": 256, "embedding_dim": 256, "num_heads": 8, "num_layers": 6,
               "dropout": 0.1, "pooling_strategy": "cls"},
}
# main.py: the contrast pairs and the fusions contrasted with the trajectory
CONTRAST_PAIRS = [("trajectory", "topology"), ("topology", "road"), ("topology", "region")]
# the paper (Sec. 3.3, Eq. 10): the trajectory against each modality
PAPER_PAIRS = [("trajectory", "topology"), ("trajectory", "road"), ("trajectory", "region")]
FUSIONS = ["topology_road", "topology_region", "road_region", "topology_road_region"]
QUERY_MODALITIES = ["topology", "road", "region", "region+topology", "road+topology", "region+road+topology"]
_FUSION_OF = {frozenset(["topology", "road"]): "topology_road", frozenset(["topology", "region"]): "topology_region",
              frozenset(["road", "region"]): "road_region",
              frozenset(["topology", "road", "region"]): "topology_road_region"}


def _ns(d):
    from types import SimpleNamespace
    return SimpleNamespace(**{k: _ns(v) if isinstance(v, dict) else v for k, v in d.items()})


_NET_CLASSES = None


def _net_classes():
    """The original OmniModel with three additions that leave what it computes unchanged:

      - the contrastive loss always runs in float32, so mixed precision (train.amp) only lowers
        the precision of the encoders - the similarity logits stay exact;
      - `parallelize(device_ids)`: the four encoders run data-parallel over several GPUs, and their
        outputs are gathered before the loss, which therefore still contrasts every trip with the
        whole batch (plain DataParallel/DDP would compute it per GPU, on half the negatives);
      - the paper's loss as a subclass (PaperLossOmniModel)."""
    global _NET_CLASSES
    if _NET_CLASSES is not None:
        return _NET_CLASSES
    import torch
    import torch.nn.functional as F
    from torch import nn
    from ..nn.omnitraj.omni_semantic import OmniModel

    class _EncodeAll(nn.Module):
        """What DataParallel replicates: every modality's projection for a slice of the batch."""

        def __init__(self, net, modalities):
            super().__init__()
            self.net, self.modalities = net, list(modalities)

        def forward(self, batch):
            return {m: self.net.encode_modality(m, batch[m], batch.get(f"{m}_attention_mask"), normalize=False)
                    for m in self.modalities}

    class MobevalOmniModel(OmniModel):
        _dp = None                     # DataParallel over _EncodeAll; kept out of the module tree
        _pre = None                    # projections computed by it for the batch being processed

        def parallelize(self, device_ids):
            dp = None
            if device_ids and len(device_ids) > 1:
                mods = sorted(set(sum(self.contrast_pairs, ())))
                dp = nn.DataParallel(_EncodeAll(self, mods), device_ids=list(device_ids))
            object.__setattr__(self, "_dp", dp)          # not registered: state_dict and .to() unchanged

        def forward(self, batch, fusion_modality=None):
            if self._dp is None:
                return super().forward(batch, fusion_modality)
            object.__setattr__(self, "_pre", self._dp(batch))
            try:
                return super().forward(batch, fusion_modality)       # the original loss, on the whole batch
            finally:
                object.__setattr__(self, "_pre", None)

        def encode_modality(self, modality, x, attention_mask=None, normalize=False):
            pre = self._pre
            if pre is not None and not normalize and modality in pre:
                return pre[modality]
            return super().encode_modality(modality, x, attention_mask, normalize)

        def compute_contrastive_loss(self, z1, z2):
            if z1.device.type in ("cuda", "cpu"):
                with torch.autocast(z1.device.type, enabled=False):
                    return self._contrastive(z1.float(), z2.float())
            return self._contrastive(z1, z2)

        def _contrastive(self, z1, z2):
            return OmniModel.compute_contrastive_loss(self, z1, z2)

    class PaperLossOmniModel(MobevalOmniModel):
        """The loss as the paper states it (Eqs. 9-10): InfoNCE on COSINE similarity with temperature tau,
        both directions summed. The released code instead uses soft targets from within-modality
        similarities, on unnormalised projections (compute_contrastive_loss in omni_semantic.py)."""

        def _contrastive(self, z1, z2):
            z1, z2 = F.normalize(z1, dim=-1), F.normalize(z2, dim=-1)
            logits = z1 @ z2.T / self.temperature.clamp(min=1e-2)
            y = torch.arange(len(z1), device=z1.device)
            return F.cross_entropy(logits, y, reduction="none") + F.cross_entropy(logits.T, y, reduction="none")

    _NET_CLASSES = {"code": MobevalOmniModel, "paper": PaperLossOmniModel}
    return _NET_CLASSES


def _build_net(arch: dict, use_road: bool, projection_dim: int, loss: str, pairs: str = "code"):
    cfg = copy.deepcopy(arch)
    cfg["enabled_encoders"] = ["trajectory", "topology", "road", "region"] if use_road else ["trajectory", "topology", "region"]
    cfg["freeze_encoders"] = []
    if pairs not in ("code", "paper"):
        raise ValueError("pairs must be 'code' or 'paper'")
    pairs = [p for p in (CONTRAST_PAIRS if pairs == "code" else PAPER_PAIRS) if use_road or "road" not in p]
    if loss not in ("code", "paper"):
        raise ValueError("loss must be 'code' or 'paper'")
    return _net_classes()[loss](_ns(cfg), pairs, projection_dim=projection_dim)


# the transformer layers of each vendored encoder: the unit gradient checkpointing recomputes
_ENCODER_LAYERS = {"trajectory": lambda e: e.blocks, "topology": lambda e: e.roformer.encoder.layer,
                   "road": lambda e: e.roformer.encoder.layer, "region": lambda e: e.transformer.layers}


_CHECKPOINTED = {}


def _checkpointed_class(cls):
    """`cls` whose forward recomputes itself in the backward pass. It is a CLASS change, not a
    wrapper stored on the instance: DataParallel's replicas copy an instance's attributes, so a
    stored wrapper would make every replica run the ORIGINAL layer (on the first GPU's weights)."""
    if cls not in _CHECKPOINTED:
        import torch
        from torch.utils.checkpoint import checkpoint

        class Checkpointed(cls):
            def forward(self, *args, **kwargs):
                if self.training and torch.is_grad_enabled():
                    return checkpoint(super().forward, *args, use_reentrant=False, **kwargs)
                return super().forward(*args, **kwargs)

            def __delattr__(self, name):
                if name == "forward":                   # `del layer.forward` undoes the checkpointing
                    object.__setattr__(self, "__class__", cls)
                else:
                    super().__delattr__(name)

        Checkpointed.__name__ = Checkpointed.__qualname__ = cls.__name__
        _CHECKPOINTED[cls] = Checkpointed
    return _CHECKPOINTED[cls]


def _checkpoint_layers(net) -> list:
    """Gradient checkpointing on every encoder layer, applied from outside the vendored code.

    Only each layer's input is kept for the backward pass, and the layer is recomputed there with
    the same dropout draws (checkpoint restores the RNG state), so the gradients are unchanged. The
    recipes' batch of 1536 needs about 73 GB of activations without it and about 10 GB with it
    (measured: 49 vs 6.7 MB per trajectory), for roughly 35% longer steps. Inactive outside training
    (no grad, or eval mode). Returns the wrapped layers; `del layer.forward` restores each one.
    Parameters, state_dict and checkpoints are those of the original layers."""
    layers = [layer for name, enc in net.encoders.items() for layer in _ENCODER_LAYERS[name](enc)]
    for layer in layers:
        layer.__class__ = _checkpointed_class(type(layer))
    return layers


class OmniTrajAdapter(TorchAdapter):
    name = "OmniTraj"
    model_type = "omnitraj"
    capabilities = {EMBEDDING, MODE_CLASSIFICATION, CROSS_MODAL}

    def __init__(self, arch: Optional[dict] = None, norm: Optional[dict] = None, grid: Optional[dict] = None,
                 road_vocab: Optional[list] = None, projection_dim: int = 512, loss: str = "code",
                 embedding: str = "projected", roads_file: Optional[str] = None,
                 topology_eps: float = prep.TOPOLOGY_EPS, interpolation: str = "pchip", pairs: str = "code", **kw):
        super().__init__(**kw)
        import torch
        self._torch = torch
        self.arch = copy.deepcopy(ENCODER_CONFIGS)          # partial overrides: {"trajectory": {"depth": 2}}
        for k, v in (arch or {}).items():
            self.arch[k] = {**self.arch.get(k, {}), **v} if isinstance(v, dict) else v
        self.norm = norm or {"mean": [0.0, 0.0], "std": [1.0, 1.0]}
        self.grid = prep.RegionGrid.from_state(grid) if isinstance(grid, dict) else grid
        self.vocab = prep.RoadVocab.from_state(road_vocab) if road_vocab is not None else None
        self.projection_dim, self.loss, self.embedding = int(projection_dim), loss, embedding
        self.roads_file, self.topology_eps = roads_file, float(topology_eps)
        self.interpolation, self.pairs = interpolation, pairs
        if embedding not in ("projected", "encoder"):
            raise ValueError("embedding must be 'projected' (the retrieval space) or 'encoder'")
        if self.grid is not None:
            self.arch["region"]["num_grids"] = self.grid.num_grids
        if self.vocab is not None:
            self.arch["road"]["num_roads"] = self.vocab.num_roads
        self.net = _build_net(self.arch, self.vocab is not None, self.projection_dim, loss, pairs).to(self.device).eval()
        self._roads = None

    @property
    def use_road(self) -> bool:
        return self.vocab is not None

    # ------------------------------------------------------------------ persistence
    def _meta(self):
        return {"norm": self.norm, "grid": None if self.grid is None else self.grid.state(),
                "road_vocab": None if self.vocab is None else self.vocab.state(), "projection_dim": self.projection_dim,
                "loss": self.loss, "embedding": self.embedding, "roads_file": self.roads_file,
                "topology_eps": self.topology_eps, "interpolation": self.interpolation, "pairs": self.pairs,
                "provenance": self.provenance}

    def save(self, path, history=None, quiet: bool = False, complete: bool = True):
        from ..nn.common import save_checkpoint
        save_checkpoint(path, self.net, self.model_type, {"arch": self.arch}, self._meta(), history,
                        quiet=quiet, complete=complete)

    @classmethod
    def from_checkpoint(cls, path, **kw) -> "OmniTrajAdapter":
        from ..nn.common import load_checkpoint
        ck = load_checkpoint(path)
        if ck["format"] == "raw":
            raise ValueError("raw OmniTraj state dicts carry no grid, road vocabulary or normalisation; "
                             "no pretrained weights were released, so train with `pretrain`")
        m = ck["meta"]
        fixed = {"arch": ck["config"]["arch"], "norm": m["norm"], "grid": m["grid"], "road_vocab": m["road_vocab"],
                 "projection_dim": m["projection_dim"], "loss": m.get("loss", "code"), "pairs": m.get("pairs", "code")}
        clash = {k for k in fixed if k in kw and kw[k] != fixed[k]}
        if clash:
            log.warning(f"{path}: {sorted(clash)} are fixed by the checkpoint's weights; the given values are ignored")
        kw = {k: v for k, v in kw.items() if k not in fixed}
        ad = cls(**fixed, **{"embedding": m.get("embedding", "projected"), "roads_file": m.get("roads_file"),
                             "topology_eps": m.get("topology_eps", prep.TOPOLOGY_EPS),
                             "interpolation": m.get("interpolation", "pchip"), **kw})
        ad.net.load_state_dict(ck["state_dict"], strict=True)
        ad.net.eval()
        ad.provenance = m.get("provenance", {})
        return ad

    # ------------------------------------------------------------------ inputs
    def _road_lookup(self):
        """traj_id -> (sorted t, segment id per point) from the map-matching table."""
        if self._roads is None:
            from ..mapmatch import read_table
            r = read_table(self.roads_file) if self.roads_file else None
            self._roads = {} if r is None else {k: (g.t.to_numpy(float), g.seg.to_numpy(np.int64))
                                                for k, g in r.sort_values(["traj_id", "t"]).groupby("traj_id")}
        return self._roads

    def _segments(self, traj_id, t) -> np.ndarray:
        look = self._road_lookup().get(traj_id)
        if look is None:
            return np.full(len(t), -1, np.int64)
        tt, seg = look
        i = np.clip(np.searchsorted(tt, t), 0, len(tt) - 1)
        return np.where(np.abs(tt[i] - t) < 1e-6, seg[i], -1)

    def _samples(self, rows: List[tuple], rng=None, augment: bool = False) -> Dict[str, np.ndarray]:
        """rows: (traj_id, lat, lon, t) -> collated original-format batch."""
        out = []
        for tid, lat, lon, t in rows:
            roads = self._segments(tid, np.asarray(t, float)) if self.use_road else None
            out.append(prep.build_sample(lat, lon, roads, self.grid, self.vocab, self.norm, rng, augment,
                                         self.topology_eps, self.interpolation))
        return prep.collate(out)

    def _tensors(self, b: Dict[str, np.ndarray]):
        T = self._torch
        return {k: T.as_tensor(v, device=self.device) for k, v in b.items()}

    @staticmethod
    def _rows(batch: TrajectoryBatch) -> List[tuple]:
        return [(batch.traj_id[i], batch.lat[i], batch.lon[i], batch.t[i]) for i in range(len(batch))]

    # ------------------------------------------------------------------ capabilities
    def _embed_batch(self, batch: TrajectoryBatch, space: Optional[str] = None) -> np.ndarray:
        T = self._torch
        mean, std = np.asarray(self.norm["mean"]), np.asarray(self.norm["std"])
        x = np.stack([(prep.resample(batch.lat[i], batch.lon[i], interpolation=self.interpolation) - mean) / std
                      for i in range(len(batch))])
        x = T.as_tensor(x.astype(np.float32), device=self.device)
        with T.no_grad():
            if (space or self.embedding) == "encoder":
                z = self.net.encoders["trajectory"](x, None)
            else:
                z = self.net.encode_modality("trajectory", x, None, normalize=True)
        return z.float().cpu().numpy()

    def embed_database(self, batch: TrajectoryBatch) -> np.ndarray:
        """GPS embeddings in the space the queries live in (the normalised projection), whatever
        `embedding` is set to for the probing tasks."""
        if self.embedding == "projected":
            return self.embed(batch)
        return np.concatenate([self._embed_batch(batch.take(np.arange(s, min(s + self.batch_size, len(batch)))),
                                                 "projected") for s in range(0, len(batch), self.batch_size)])

    def query_modalities(self) -> List[str]:
        return [m for m in QUERY_MODALITIES if self.use_road or "road" not in m]

    def embed_query(self, batch: TrajectoryBatch, modalities: str) -> np.ndarray:
        """Embedding of the batch's topology / road / region representation ("region+topology", ...),
        computed as the original get_embeddings: each modality projected and L2-normalised, fused by
        the fusion layer when there are several, and normalised again."""
        mods = modalities.split("+")
        if "road" in mods and not self.use_road:
            raise ValueError("road queries need a model trained with map-matched roads (roads_file)")
        T = self._torch
        outs = []
        for s in range(0, len(batch), self.batch_size):
            b = self._tensors(self._samples(self._rows(batch.take(np.arange(s, min(s + self.batch_size, len(batch)))))))
            with T.no_grad():
                e = {m: self.net.encode_modality(m, b[m], b.get(f"{m}_attention_mask"), normalize=True) for m in mods}
                if len(mods) > 1:
                    z = self.net.encode_fusion(_FUSION_OF[frozenset(mods)], e, normalize=True)
                else:
                    z = e[mods[0]]
            outs.append(z.float().cpu().numpy())
        return np.concatenate(outs)

    def elements(self, batch: TrajectoryBatch, modality: str) -> List[set]:
        """The set of regions / road segments of each trajectory, as the model sees them (for CR@k)."""
        sets = []
        for tid, lat, lon, t in self._rows(batch):
            if modality == "region":
                tr = prep.resample(lat, lon, interpolation=self.interpolation)
                ids = set(self.grid.ids(tr[:, 1], tr[:, 0]).tolist())
                if self.grid.vocab is not None:                    # the shared "other cell" id is no region
                    ids.discard(len(self.grid.vocab) + 1)
                sets.append(ids)
            elif modality == "road":
                seg = self._segments(tid, np.asarray(t, float))
                sets.append(set(self.vocab.encode(seg).tolist()) - {self.vocab.unk})
            else:
                raise ValueError(modality)
        return sets

    # ------------------------------------------------------------------ training
    @classmethod
    def pretrain(cls, ctx, train: Optional[dict] = None, out: Optional[str] = None, init_from: Optional[str] = None,
                 roads_file: Optional[str] = None, sample_unit: str = "trajectory", min_points: int = 20,
                 grid_n: Optional[int] = prep.PAPER_GRID_N, grid_cell_m: Optional[float] = None,
                 max_regions: int = 20_000, max_roads: int = 200_000, augment_val: bool = True,
                 max_train_units: Optional[int] = None, loss: str = "code", pairs: str = "code",
                 gradient_checkpointing: bool = True, gpus: int = 1, prep_workers: Optional[int] = None,
                 **kw) -> "OmniTrajAdapter":
        """Contrastive training as main.py: trajectory<->topology, topology<->road, topology<->region
        (pairs="paper": the trajectory against each modality, Eq. 10), plus trajectory<->each fusion, best
        model on the validation loss. `recipe: code | paper` sets main.py's / the paper's optimisation;
        without a recipe the defaults are mobeval's (shorter) ones.

        sample_unit: "trajectory" (whole trips of >= min_points points, as the paper) or "window".
        grid_n / grid_cell_m: the region grid, n x n over the train area (paper: 16) or fixed-size cells.
        augment_val: the original also augments the validation set (its dataset settings are shared).
        gradient_checkpointing: recompute encoder layers in the backward pass (see _checkpoint_layers);
        same gradients, a fraction of the memory, slower steps. Off only pays for small batches.

        Speed (none of these changes what is computed): the resampling, topology and id lookups of
        every unit are done once, over `prep_workers` processes (default: the job's CPUs), and each
        step only augments and pads; `gpus: 2` splits the encoders over two GPUs with the loss still
        on the whole batch; `train: {amp: true}` runs the encoders in mixed precision."""
        from ..nn.common import TrainConfig
        from ..nn.common import fit as fit_loop
        cfg = TrainConfig.from_dict({"lr": 2e-4, "batch_size": 256, "optimizer": "adamw", "weight_decay": 1e-4,
                                     "scheduler": "cosine", "cosine_eta_min": 1e-5, "grad_clip": 1.0,
                                     **(train or {})})
        kw.pop("device", None)
        rng = np.random.default_rng(cfg.seed)
        # ---- units
        if sample_unit == "trajectory":
            # whole trips, split where the recording pauses longer than max_gap_s (as the evaluation
            # windows are): a spline across a two-hour gap would invent a straight-line trip
            gap = getattr(ctx.cfg, "max_gap_s", None)
            units = {}
            for s in ("train", "val"):
                u = []
                p = ctx.splits[s].points.sort_values(["traj_id", "t"], kind="stable")
                for tid, g in p.groupby("traj_id", sort=False):
                    la, lo, tt = g.lat.to_numpy(), g.lon.to_numpy(), g.t.to_numpy(float)
                    cuts = np.where(np.diff(tt) > gap)[0] + 1 if gap else []
                    for seg in np.split(np.arange(len(tt)), cuts):
                        if len(seg) >= min_points:
                            u.append((tid, la[seg], lo[seg], tt[seg]))
                units[s] = u
        elif sample_unit == "window":
            units = {s: cls._rows(ctx.windows[s]) for s in ("train", "val")}
        else:
            raise ValueError("sample_unit must be 'trajectory' or 'window'")
        if max_train_units and len(units["train"]) > max_train_units:
            keep = np.sort(rng.choice(len(units["train"]), max_train_units, replace=False))
            units["train"] = [units["train"][i] for i in keep]
        if not units["train"] or not units["val"]:
            raise ValueError(f"no {sample_unit}s with >= {min_points} points in train/val")
        if init_from:
            ad = cls.from_checkpoint(init_from, device=cfg.device, loss=loss, pairs=pairs,
                                     **({"roads_file": roads_file} if roads_file else {}), **kw)
        else:
            # ---- normalisation, grid and road vocabulary, all from TRAIN
            interp = kw.get("interpolation", "pchip")
            res = [prep.resample(la, lo, interpolation=interp) for _, la, lo, _ in units["train"][:20000]]
            allxy = np.concatenate(res)
            norm = {"mean": allxy.mean(0).tolist(), "std": (allxy.std(0) + 1e-12).tolist()}
            lat = np.concatenate([la for _, la, _, _ in units["train"]])
            lon = np.concatenate([lo for _, _, lo, _ in units["train"]])
            grid = prep.RegionGrid.fit(lat, lon, n=None if grid_cell_m else grid_n, cell_m=grid_cell_m,
                                       max_cells=max_regions)
            vocab = None
            if roads_file:
                tmp = cls.__new__(cls)
                tmp.roads_file, tmp._roads = roads_file, None
                segs = [tmp._segments(tid, t) for tid, _, _, t in units["train"]]
                matched = float(np.mean(np.concatenate(segs) >= 0)) if segs else 0.0
                vocab = prep.RoadVocab.fit(segs, max_roads)
                log.info(f"OmniTraj roads: {len(vocab.index):,} segments in train, {matched:.1%} of train points matched")
                if matched < 0.5:
                    log.warning("fewer than half of the train points have a road segment: check that roads_file "
                                "was produced from this dataset (same traj_id and timestamps)")
            else:
                log.warning("OmniTraj without roads_file: the road encoder is left out (trajectory, topology and "
                            "region only). Run `mobeval roads` and `mobeval mapmatch` for the full model.")
            ad = cls(norm=norm, grid=grid.state(), road_vocab=None if vocab is None else vocab.state(),
                     loss=loss, pairs=pairs, roads_file=roads_file, device=cfg.device, **kw)
            if roads_file:
                ad._roads = tmp._roads                          # the map-matching table, already read
        log.info(f"OmniTraj: {len(units['train']):,} train / {len(units['val']):,} val {'trajectories' if sample_unit == 'trajectory' else 'windows'}, "
                 f"{ad.grid.num_grids} regions ({ad.grid.nx}x{ad.grid.ny}"
                 f"{', compacted' if ad.grid.vocab is not None else ''}), "
                 f"{'no roads' if not ad.use_road else f'{ad.vocab.num_roads} road tokens'}, loss {ad.loss}")
        fusions = [f for f in FUSIONS if ad.use_road or "road" not in f]

        # ---- the deterministic part of every sample, once (the original preprocesses offline too)
        t_prep = time.time()
        cache = {}
        for s in ("train", "val"):
            segs = [ad._segments(tid, np.asarray(t, float)) for tid, _, _, t in units[s]] if ad.use_road else None
            cache[s] = prep.prepare_units(units[s], segs, ad.grid, ad.vocab, ad.norm, ad.topology_eps,
                                          ad.interpolation, workers=prep_workers)
        log.info(f"OmniTraj: inputs of {len(cache['train']) + len(cache['val']):,} units prepared in "
                 f"{time.time() - t_prep:.0f}s ({prep.available_cpus() if not prep_workers else prep_workers} "
                 f"processes, {(cache['train'].nbytes + cache['val'].nbytes) / 1e9:.2f} GB); each step now only "
                 "augments and pads")
        clock = {"batch": 0.0, "epochs": 0}

        def loss_fn(idx, training):
            t = time.perf_counter()
            g = rng if training else np.random.default_rng(int(idx[0]))
            b = ad._tensors(cache["train" if training else "val"].batch(idx, g, training or augment_val))
            b["label"] = ad._torch.as_tensor(idx, device=ad.device)
            clock["batch"] += time.perf_counter() - t
            loss = ad.net(b, fusions)
            return loss.mean() if loss.dim() > 0 else loss

        def on_epoch(history):
            h = history[-1]
            h["batch_seconds"] = round(clock["batch"], 1)
            clock["batch"] = 0.0
            clock["epochs"] += 1
            if clock["epochs"] == 1 or h["epoch"] % 25 == 0:          # first epoch of this job, then every 25th
                log.info(f"OmniTraj epoch {h['epoch']}: {h['seconds']:.0f}s, of which {h['batch_seconds']:.0f}s "
                         f"building batches on the CPU; the rest is the network on {ad.device}")

        ad.provenance = ctx.provenance(init_from=init_from, sample_unit=sample_unit)
        ad.net.train()
        wrapped = _checkpoint_layers(ad.net) if gradient_checkpointing else []
        if wrapped:
            log.info(f"OmniTraj: gradient checkpointing on {len(wrapped)} encoder layers")

        # ---- several GPUs: the encoders split each batch, the loss sees all of it
        gpus = int(gpus or 1)
        if gpus > 1:
            T = ad._torch
            n_dev = T.cuda.device_count() if T.cuda.is_available() else 0
            if ad.device.type != "cuda" or n_dev < 2:
                log.warning(f"OmniTraj: gpus={gpus} asked, but {n_dev} CUDA device(s) visible to this job "
                            f"(device {ad.device}): training on one. On PBS, ask for them: ngpus={gpus}.")
            else:
                first = ad.device.index if ad.device.index is not None else T.cuda.current_device()
                ids = [first] + [i for i in range(n_dev) if i != first][:gpus - 1]
                ad.net.parallelize(ids)
                log.info(f"OmniTraj: encoders data-parallel over GPUs {ids}; the contrastive loss is computed on "
                         f"the whole batch of {cfg.batch_size} on GPU {ids[0]}")
        ad.provenance["compute"] = {"amp": bool(cfg.amp) and ad.device.type == "cuda",
                                    "gpus": len(ad.net._dp.device_ids) if ad.net._dp is not None else 1,
                                    "gradient_checkpointing": bool(wrapped)}
        try:
            history = fit_loop(ad.net, len(units["train"]), len(units["val"]), loss_fn, cfg,
                               on_best=ad.epoch_checkpointer(out), on_epoch=on_epoch)
        finally:
            ad.net.parallelize(None)
            for layer in wrapped:
                del layer.forward
        ad.net.eval()
        ad.invalidate_cache()
        if out:
            ad.save(out, history)
        return ad
