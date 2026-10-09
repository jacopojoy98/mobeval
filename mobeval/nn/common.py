"""Training utilities shared by all model adapters: device/seed handling, a generic
early-stopping fit loop, checkpoint I/O and small task heads for frozen encoders."""
from __future__ import annotations

import contextlib
import copy
import json
import logging
import math
import os
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional

from .. import progress

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger("mobeval.train")
CHECKPOINT_FORMAT = "mobeval-checkpoint-v1"


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 0.01
    patience: int = 8                      # early stopping on validation loss (epochs)
    grad_clip: float = 1.0                 # 0 / None: no clipping
    optimizer: str = "adamw"               # adamw | adam (L2-coupled weight decay, as torch.optim.Adam) | adafactor
    scheduler: str = "plateau"             # plateau | step | cosine | none
    cosine_eta_min: float = 0.0            # CosineAnnealingLR over `epochs` (scheduler: cosine)
    step_size: int = 5                     # StepLR (scheduler: step)
    step_gamma: float = 0.5
    restore_best: bool = True              # False: keep the last epoch's weights (original TransferTraj)
    drop_last: bool = True                 # drop the last incomplete training batch (the originals keep it)
    plateau_factor: float = 0.5
    plateau_patience: Optional[int] = None # default: max(1, patience // 3)
    max_steps_per_epoch: Optional[int] = None
    device: str = "auto"
    seed: int = 0
    log_every: int = 50
    amp: bool = False                      # mixed precision on CUDA GPUs: float16 with loss scaling before
                                           # Ampere (V100), bfloat16 from Ampere on; weights stay float32
    save_train_state: bool = True          # interim checkpoints also hold optimizer, schedule and data
                                           # order, so a resumed job continues the run exactly

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "TrainConfig":
        d = dict(d or {})
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown TrainConfig keys: {sorted(unknown)}")
        return cls(**d)


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def minibatches(n: int, batch_size: int, shuffle: bool, rng: Optional[np.random.Generator] = None,
                drop_last: bool = False) -> Iterator[np.ndarray]:
    idx = (rng or np.random.default_rng()).permutation(n) if shuffle else np.arange(n)
    stop = n - (n % batch_size) if drop_last and n >= batch_size else n
    for s in range(0, stop, batch_size):
        yield idx[s:s + batch_size]


def make_optimizer(params, cfg: TrainConfig):
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adafactor":
        # torch's defaults (lr 1e-2, relative-step decay), as TrajGPT's HEAD calls Adafactor(params)
        # without arguments; `lr` is ignored so the default recipe cannot silently override it.
        return torch.optim.Adafactor(params)
    raise ValueError("optimizer must be 'adamw', 'adam' or 'adafactor'")


# Set while `fit` runs: the optimiser/schedule state written into interim checkpoints.
_LIVE_STATE: Optional[Callable[[], dict]] = None
# Set by `resuming(path)`: the next `fit` continues the run stored in that checkpoint.
_RESUME_FROM: Optional[str] = None


@contextlib.contextmanager
def resuming(path):
    """Within this block, the (first) `fit` continues the interrupted run whose interim checkpoint
    is `path`: epoch count, learning-rate schedule, early-stopping state and history carry on,
    and the optimizer state and data order too when the checkpoint holds them. The weights
    themselves are loaded by the adapter (init_from), as before."""
    global _RESUME_FROM
    _RESUME_FROM = str(path) if path else None
    try:
        yield
    finally:
        _RESUME_FROM = None


def _amp_context(cfg: TrainConfig, model: nn.Module):
    """(autocast context factory, GradScaler or None) for `cfg.amp`, or (None, None)."""
    if not cfg.amp:
        return None, None
    try:
        dev = next(model.parameters()).device
    except StopIteration:
        return None, None
    if dev.type != "cuda":
        log.info(f"amp: mixed precision is for GPUs; training on {dev.type} in float32")
        return None, None
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(dev)[0] >= 8 else torch.float16
    scaler = None
    if dtype == torch.float16:
        try:
            scaler = torch.amp.GradScaler("cuda")
        except (AttributeError, TypeError):                         # torch < 2.3
            scaler = torch.cuda.amp.GradScaler()
    log.info(f"mixed precision: {str(dtype).replace('torch.', '')} on {dev}"
             + (" with loss scaling" if scaler is not None else ""))
    return (lambda: torch.autocast(dev.type, dtype=dtype)), scaler


def _replay(history: List[dict], sched) -> tuple:
    """Best validation loss and epochs-without-improvement after `history`, stepping `sched` through
    it exactly as the original run did (so the learning rate is where that run left it)."""
    best, bad = math.inf, 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)                # "scheduler.step() before optimizer.step()"
        for h in history:
            vl = float(h["val_loss"])
            if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                sched.step(vl)
            elif sched is not None:
                sched.step()
            if vl < best - 1e-6:
                best, bad = vl, 0
            else:
                bad += 1
    return best, bad


def fit(model: nn.Module, n_train: int, n_val: int, loss_fn: Callable[[np.ndarray, bool], torch.Tensor],
        cfg: TrainConfig, params: Optional[Iterable] = None, drop_last: bool = True,
        on_best: Optional[Callable[[List[dict]], None]] = None,
        on_epoch: Optional[Callable[[List[dict]], None]] = None) -> List[dict]:
    """Generic loop. `loss_fn(indices, train)` builds the batch for those sample indices
    and returns a scalar loss. Restores the best validation state at the end.

    `on_best(history)` is called every time validation improves, with the model's weights
    already at that best state. Adapters pass a checkpoint saver, so a run that is killed
    at epoch 40 of 100 still leaves the best-so-far model on disk instead of nothing.
    `on_epoch(history)` is called after every epoch (before `on_best`); it may add fields to
    history[-1], which are then saved with the checkpoint.

    Inside `resuming(path)` the loop continues the run stored in that checkpoint instead of
    starting at epoch 1 (see `resuming`).
    """
    global _LIVE_STATE, _RESUME_FROM
    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    opt = make_optimizer(params if params is not None else model.parameters(), cfg)
    if cfg.scheduler == "plateau":
        pp = cfg.plateau_patience if cfg.plateau_patience is not None else max(1, cfg.patience // 3)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=cfg.plateau_factor, patience=pp)
    elif cfg.scheduler == "step":
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=cfg.step_size, gamma=cfg.step_gamma)
    elif cfg.scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.cosine_eta_min)
    elif cfg.scheduler == "none":
        sched = None
    else:
        raise ValueError("scheduler must be 'plateau', 'step', 'cosine' or 'none'")
    autocast, scaler = _amp_context(cfg, model)
    amp = autocast if autocast is not None else contextlib.nullcontext
    best, best_state, bad, history, start = math.inf, None, 0, [], 1
    resume_from, _RESUME_FROM = _RESUME_FROM, None                 # only the first fit of a resumed training
    if resume_from:
        ck = torch.load(resume_from, map_location="cpu", weights_only=False)
        history = [dict(h) for h in (ck.get("history") or [])]
        state = ck.get("train_state") or {}
        if history:
            start = int(history[-1]["epoch"]) + 1
            best, bad = _replay(history, sched)
            how = "learning-rate schedule replayed"
            if state.get("epoch") == history[-1]["epoch"]:
                if state.get("optimizer") is not None:
                    lrs = [g["lr"] for g in opt.param_groups]
                    try:
                        opt.load_state_dict(state["optimizer"])
                        how += ", optimizer state restored"
                    except (ValueError, KeyError, RuntimeError) as e:   # e.g. another optimizer in the config
                        how += f", optimizer starts afresh ({e})"
                    for g, lr in zip(opt.param_groups, lrs):        # the schedule of THIS config (epochs may
                        g["lr"] = lr                                 # have changed) decides the learning rate
                if scaler is not None and state.get("scaler"):
                    scaler.load_state_dict(state["scaler"])
                if state.get("rng") is not None:
                    rng.bit_generator.state = state["rng"]
                    how += ", data order continued"
                if state.get("torch_rng") is not None:                  # dropout continues its stream too
                    torch.set_rng_state(state["torch_rng"])
                if state.get("cuda_rng") is not None and torch.cuda.is_available():
                    try:
                        torch.cuda.set_rng_state_all(state["cuda_rng"])
                    except (RuntimeError, IndexError):                  # another number of GPUs
                        pass
            else:
                rng = np.random.default_rng([cfg.seed, start])
                how += ("; the optimizer starts afresh (its moments were not saved in this checkpoint, "
                        "written by an older mobeval) - they re-adapt within a few hundred steps")
            best_state = copy.deepcopy(model.state_dict())          # the weights loaded ARE the best so far
            log.info(f"continuing the interrupted training at epoch {start} of {cfg.epochs} "
                     f"(best val {best:.4f}; {how})")
    rep = progress.get()
    who = getattr(rep, "scope", None) or "model"
    rep.model(who, state="training", detail=f"epoch {start - 1}/{cfg.epochs}")
    epoch = start - 1

    def train_state():
        return {"epoch": epoch, "epochs": cfg.epochs, "best": best, "bad": bad, "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None, "rng": rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

    _LIVE_STATE = train_state if cfg.save_train_state else None
    try:
        for epoch in range(start, cfg.epochs + 1):
            model.train()
            t0, tr_losses = time.time(), []
            for step, idx in enumerate(minibatches(n_train, cfg.batch_size, True, rng,
                                                   drop_last and cfg.drop_last and n_train > cfg.batch_size)):
                if cfg.max_steps_per_epoch and step >= cfg.max_steps_per_epoch:
                    break
                with amp():
                    loss = loss_fn(idx, True)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite training loss at epoch {epoch}, step {step}. Common causes: extreme or invalid input "
                        f"values (clean the GPS data, see mobeval.data.clean_points), or a learning rate that is too high "
                        f"(current {opt.param_groups[0]['lr']:.2e})"
                        + (". Mixed precision is on (amp): try again without it." if autocast is not None else "."))
                opt.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    if cfg.grad_clip:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    if cfg.grad_clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    opt.step()
                tr_losses.append(loss.item())
                if cfg.log_every and step % cfg.log_every == 0:
                    log.debug(f"epoch {epoch} step {step} loss {loss.item():.4f}")
            model.eval()
            with torch.no_grad(), amp():
                va = [loss_fn(idx, False).item() for idx in minibatches(n_val, cfg.batch_size, False)] if n_val else []
            tr, vl = float(np.mean(tr_losses)), (float(np.mean(va)) if va else float(np.mean(tr_losses)))
            if not np.isfinite(vl):
                bad_batches = int(np.sum(~np.isfinite(va))) if va else 0
                raise FloatingPointError(
                    f"non-finite validation loss at epoch {epoch} ({bad_batches}/{len(va)} batches). Early stopping "
                    "cannot work with NaN; check the validation inputs for NaN/inf or extreme values.")
            history.append({"epoch": epoch, "train_loss": tr, "val_loss": vl, "seconds": time.time() - t0})
            if on_epoch is not None:
                on_epoch(history)
            log.info(f"epoch {epoch:3d}  train {tr:.4f}  val {vl:.4f}  ({time.time() - t0:.0f}s)")
            improved = vl < best - 1e-6
            rep.model(who, state="training", epoch=epoch, epochs=cfg.epochs, train_loss=tr, val_loss=vl,
                      best_val=min(best, vl), detail=f"epoch {epoch}/{cfg.epochs}  val {vl:.4g}"
                                                     f"  best {min(best, vl):.4g}")
            rep.event("epoch", model=who, epoch=epoch, epochs=cfg.epochs, train_loss=float(f"{tr:.6g}"),
                      val_loss=float(f"{vl:.6g}"), improved=improved, seconds=round(time.time() - t0, 1))
            if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                sched.step(vl)
            elif sched is not None:
                sched.step()
            if vl < best - 1e-6:
                best, bad, best_state = vl, 0, copy.deepcopy(model.state_dict())
                if on_best is not None:
                    # The live weights ARE the best weights at this instant, so the saver can just
                    # write model.state_dict(). Never let a failed save abort a good training run.
                    try:
                        on_best(history)
                    except Exception as e:                                  # noqa: BLE001
                        log.warning(f"could not save the epoch-{epoch} checkpoint: {e!r}")
                        rep.event("checkpoint_failed", model=who, epoch=epoch, error=repr(e))
                    else:
                        rep.event("checkpoint", model=who, epoch=epoch, val_loss=float(f"{vl:.6g}"),
                                  message=f"{who}: saved checkpoint at epoch {epoch} (val {vl:.4g})")
            else:
                bad += 1
                if bad >= cfg.patience:
                    log.info(f"early stopping at epoch {epoch} (best val {best:.4f})")
                    rep.event("early_stop", model=who, epoch=epoch, best_val=best,
                              message=f"{who}: early stop at epoch {epoch} (best val {best:.4g})")
                    break
    finally:
        _LIVE_STATE = None
    if best_state is not None and cfg.restore_best:
        model.load_state_dict(best_state)
    model.eval()
    return history


# ------------------------------------------------------------------ checkpoints
def save_checkpoint(path, model: nn.Module, model_type: str, config: dict, meta: Optional[dict] = None,
                    history: Optional[list] = None, quiet: bool = False, complete: bool = True):
    """Write a checkpoint atomically, then mirror it to durable storage.

    Atomicity matters because this is now called after every improving epoch: a job killed
    part-way through `torch.save` must not be able to destroy the previous good checkpoint.
    We write to a temporary file in the same directory and rename, which is atomic on POSIX.
    """
    from .. import layout
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # `complete` marks the difference between "this is the best epoch so far" and "training
    # finished". A training job that is killed leaves complete=False, which tells a resumed run
    # to continue from these weights instead of accepting a half-trained model as final.
    blob = {"format": CHECKPOINT_FORMAT, "model_type": model_type, "config": config, "complete": complete,
            "meta": meta or {}, "history": history or [], "state_dict": model.state_dict()}
    if not complete and _LIVE_STATE is not None:
        # An interim checkpoint written from inside `fit`: keep what a resumed job needs to carry
        # on exactly (optimizer moments, data order). Final checkpoints do not carry it.
        try:
            blob["train_state"] = _LIVE_STATE()
        except Exception as e:                                      # noqa: BLE001
            log.debug(f"train state not saved: {e!r}")
    sidecar = {"model_type": model_type, "config": config, "complete": complete,
               "meta": meta or {}, "history": history or []}
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    torch.save(blob, tmp)
    os.replace(tmp, path)
    json_path = path.with_suffix(".json")
    tmp_json = json_path.with_suffix(f".json.tmp{os.getpid()}")
    tmp_json.write_text(json.dumps(sidecar, indent=2,
                                   default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    os.replace(tmp_json, json_path)
    kept = layout.mirror(path)
    layout.mirror(json_path)
    if not quiet:
        log.info(f"saved checkpoint -> {path}" + (f"  (kept in {kept.parent})" if kept else ""))


def checkpoint_status(path) -> str:
    """'missing' | 'partial' | 'complete', read from the sidecar so nothing has to be unpickled.

    'partial' means the last training of this model was interrupted: the weights are the best
    epoch it reached, and training should continue from them rather than be skipped.
    """
    path = Path(path)
    if not path.exists():
        return "missing"
    side = path.with_suffix(".json")
    if side.exists():
        try:
            return "complete" if json.loads(side.read_text()).get("complete", True) else "partial"
        except (OSError, json.JSONDecodeError):
            pass
    return "complete"                       # checkpoints from before this field existed


def last_epoch(path) -> Optional[int]:
    side = Path(path).with_suffix(".json")
    try:
        hist = json.loads(side.read_text()).get("history") or []
    except (OSError, json.JSONDecodeError):
        return None
    return hist[-1].get("epoch") if hist else None


def load_checkpoint(path, map_location="cpu") -> dict:
    ck = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(ck, dict) and ck.get("format") == CHECKPOINT_FORMAT:
        return ck
    # raw state dict (e.g. the public UniTraj model.pt, or a TrajGPT best_state_dict)
    if isinstance(ck, dict) and "model_state_dict" in ck:
        ck = ck["model_state_dict"]
    return {"format": "raw", "model_type": None, "config": {}, "meta": {}, "history": [], "state_dict": ck}


# ------------------------------------------------------------- frozen-encoder heads
class MLPHead(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        self.net = (nn.Linear(d_in, d_out) if not hidden else
                    nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, d_out)))

    def forward(self, x):
        return self.net(x)


class GMMHead(nn.Module):
    """Mixture of Gaussians over log(value), predicted from a representation."""

    def __init__(self, d_in: int, n_components: int = 3, hidden: int = 128):
        super().__init__()
        self.k = n_components
        self.body = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU())
        self.out = nn.Linear(hidden, 3 * n_components)

    def forward(self, x):
        w, mu, s = self.out(self.body(x)).chunk(3, dim=-1)
        return F.softmax(w, -1), mu, F.softplus(s) + 1e-3

    @staticmethod
    def nll(params, y_log):
        w, mu, s = params
        comp = torch.log(w + 1e-12) - torch.log(s) - 0.5 * math.log(2 * math.pi) - 0.5 * ((y_log[:, None] - mu) / s) ** 2
        return -torch.logsumexp(comp, -1).mean()


def _split_train_val(n: int, frac: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = int(round(frac * n)) if n >= 20 else 0
    return idx[n_val:], idx[:n_val]


def fit_classifier_head(Z: np.ndarray, y: np.ndarray, n_classes: int, cfg: TrainConfig,
                        hidden: Optional[int] = 256, class_weighted: bool = True) -> MLPHead:
    """Head on frozen embeddings; 15% of the (few-shot) train labels are held out for early stopping."""
    dev = resolve_device(cfg.device)
    set_seed(cfg.seed)
    Zt = torch.as_tensor(Z, dtype=torch.float32, device=dev)
    mu, sd = Zt.mean(0, keepdim=True), Zt.std(0, keepdim=True) + 1e-6
    Zt = (Zt - mu) / sd
    yt = torch.as_tensor(y, dtype=torch.long, device=dev)
    head = MLPHead(Z.shape[1], n_classes, hidden).to(dev)
    head.register_buffer("mu", mu); head.register_buffer("sd", sd)
    counts = torch.bincount(yt, minlength=n_classes).float()
    weight = (counts.sum() / (n_classes * counts.clamp(min=1))) if class_weighted else None
    tr, va = _split_train_val(len(y), 0.15, cfg.seed)
    loss = lambda i, train: F.cross_entropy(head((Zt[(tr if train else va)[i]])), yt[(tr if train else va)[i]], weight=weight)
    fit(head, len(tr), len(va), loss, cfg, drop_last=False)
    return head


def predict_head(head: MLPHead, Z: np.ndarray, softmax: bool = True) -> np.ndarray:
    dev = next(head.parameters()).device
    with torch.no_grad():
        z = (torch.as_tensor(Z, dtype=torch.float32, device=dev) - head.mu) / head.sd
        out = head(z)
        return (F.softmax(out, -1) if softmax else out).cpu().numpy()
