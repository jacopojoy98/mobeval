"""Model registry: build, train and load adapters from plain dictionaries (config files)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

log = logging.getLogger("mobeval.registry")


def _unitraj():
    from .adapters.unitraj import UniTrajAdapter
    return UniTrajAdapter


def _trajgpt():
    from .adapters.trajgpt import TrajGPTAdapter
    return TrajGPTAdapter


def _clip():
    from .adapters.clip_mobility import CLIPMobilityAdapter
    return CLIPMobilityAdapter


def _transfertraj():
    from .adapters.transfertraj import TransferTrajAdapter
    return TransferTrajAdapter


def _kinematic():
    from .adapters.reference import KinematicReference
    return KinematicReference


def _weak():
    from .adapters.reference import WeakReference
    return WeakReference


MODEL_TYPES = {"unitraj": _unitraj, "trajgpt": _trajgpt, "clip_mobility": _clip, "transfertraj": _transfertraj,
               "kinematic_ref": _kinematic, "weak_ref": _weak}
TRAIN_METHOD = {"unitraj": "pretrain", "trajgpt": "train", "clip_mobility": "pretrain", "transfertraj": "pretrain"}


class ProvenanceError(RuntimeError):
    pass


def checkpoint_path(spec: dict, output_dir) -> Path:
    return Path(spec.get("checkpoint") or Path(output_dir) / "checkpoints" / f"{spec['name']}.pt")


def train_model(spec: dict, ctx, output_dir) -> Path:
    """Train (or fine-tune, with train.init_from) the model described by `spec` on ctx's train split."""
    mtype = spec["type"]
    if mtype not in TRAIN_METHOD:
        raise ValueError(f"model type '{mtype}' has no training routine")
    cls = MODEL_TYPES[mtype]()
    tr = dict(spec.get("train", {}))
    out = Path(tr.pop("out", None) or checkpoint_path(spec, output_dir))
    kwargs = dict(tr.pop("options", {}))
    if "arch" in spec:
        kwargs["arch"] = spec["arch"]
    init_from = tr.pop("init_from", None)
    if init_from:
        kwargs["init_from"] = init_from
    log.info(f"training {spec['name']} ({mtype}) -> {out}")
    getattr(cls, TRAIN_METHOD[mtype])(ctx, train=tr, out=str(out), **kwargs, **spec.get("adapter", {}))
    return out


def load_model(spec: dict, ctx=None, output_dir=".", check_provenance: str = "error"):
    """Instantiate an adapter. Torch models load `checkpoint` (default <output>/checkpoints/<name>.pt).
    check_provenance: 'error' | 'warn' | 'off' - compare the checkpoint's train-split fingerprint with ctx."""
    mtype = spec["type"]
    cls = MODEL_TYPES[mtype]()
    common = {k: spec[k] for k in ("run_tag",) if k in spec}
    if mtype not in TRAIN_METHOD:                                   # reference models: no weights
        ad = cls(**{k: v for k, v in spec.get("adapter", {}).items() if k != "device"})
    else:
        path = checkpoint_path(spec, output_dir)
        if not path.exists():
            raise FileNotFoundError(f"{spec['name']}: checkpoint {path} not found - run `mobeval train` first")
        ad = cls.from_checkpoint(str(path), **spec.get("adapter", {}), **common)
        fp = getattr(ad, "provenance", {}).get("train_fingerprint")
        if ctx is not None and check_provenance != "off" and not spec.get("external_pretraining", False):
            if fp is None:
                msg = (f"{spec['name']}: checkpoint has no split fingerprint (trained outside mobeval?). Make sure its "
                       f"training data does not overlap the test split, then set external_pretraining: true.")
            elif fp != ctx.fingerprint:
                msg = (f"{spec['name']}: checkpoint was trained on a different split ({fp} != {ctx.fingerprint}); "
                       f"its training data may overlap this test set.")
            else:
                msg = None
            if msg:
                if check_provenance == "error":
                    raise ProvenanceError(msg)
                log.warning(msg)
    ad.name = spec.get("name", ad.name)
    if "run_tag" in spec:
        ad.run_tag = spec["run_tag"]
    return ad
