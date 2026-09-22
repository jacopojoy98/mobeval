"""Experiment configuration files (YAML or JSON).

    output_dir: results/geolife
    persist_dir: /home/me/results       # optional: keep checkpoints/results here as they are produced,
                                        # so a job killed in scratch does not lose finished work
    run_dirs: true                      # each run writes output_dir/runs/<run_id>/ (default)
    checkpoint_dir: null                # defaults to output_dir/checkpoints, shared by every run
    dataset:   {loader: geolife, path: /data/geolife}          # or csv / synthetic
    eval:      {window_length: 64, eval_seeds: [0, 1, 2], ...} # any EvalConfig field
    check_provenance: error                                    # error | warn | off
    models:
      - {name: UniTraj-ft, type: unitraj, train: {init_from: model.pt, epochs: 20, device: cuda}}
      - {name: TrajGPT, type: trajgpt, train: {epochs: 100}, arch: {input_order: fixed}}
      - {name: CLIP, type: clip_mobility, train: {epochs: 50}}
      - {name: UniTraj-public, type: unitraj, checkpoint: model.pt, external_pretraining: true}
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from .context import EvalConfig


def load_config(path) -> dict:
    text = Path(path).read_text()
    if str(path).endswith((".yaml", ".yml")):
        import yaml
        cfg = yaml.safe_load(text)
    else:
        cfg = json.loads(text)
    for key in ("dataset", "models"):
        if key not in cfg:
            raise ValueError(f"config is missing '{key}'")
    return apply_defaults(cfg)


# Every key a model entry may have. A misspelled key used to be silently ignored, which is the
# worst possible outcome: `chechpoint:` simply fell back to the default checkpoint path and the
# run looked fine while evaluating a different file than the one named in the config.
MODEL_KEYS = {"name", "type", "checkpoint", "train", "arch", "adapter", "run_tag", "external_pretraining"}
TOP_LEVEL_KEYS = {"title", "output_dir", "persist_dir", "progress_dir", "checkpoint_dir", "run_dirs",
                  "dataset", "eval", "models", "check_provenance"}


def _did_you_mean(key: str, known) -> str:
    import difflib
    close = difflib.get_close_matches(key, sorted(known), n=1, cutoff=0.6)
    return f" - did you mean '{close[0]}'?" if close else ""


def _check_keys(d: dict, known, where: str):
    unknown = sorted(set(d) - known)
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {unknown}"
                         + "".join(_did_you_mean(k, known) for k in unknown)
                         + f". Valid keys: {sorted(known)}")


def apply_defaults(cfg: dict) -> dict:
    names = [m["name"] for m in cfg["models"]]
    if len(set(names)) != len(names):
        raise ValueError(f"model names must be unique: {names}")
    _check_keys(cfg, TOP_LEVEL_KEYS, "config")
    for m in cfg["models"]:
        for required in ("name", "type"):
            if required not in m:
                raise ValueError(f"model entry {m} is missing '{required}'")
        _check_keys(m, MODEL_KEYS, f"model '{m['name']}'")
    cfg.setdefault("output_dir", "results")
    cfg.setdefault("eval", {})
    cfg.setdefault("check_provenance", "error")
    # Each invocation writes its report into output_dir/runs/<run_id>/ so concurrent or repeated
    # jobs never overwrite each other. Checkpoints stay in the shared output_dir/checkpoints/.
    cfg.setdefault("run_dirs", True)
    cfg.setdefault("persist_dir", None)        # durable directory; results are copied there as produced
    cfg.setdefault("checkpoint_dir", None)     # defaults to output_dir/checkpoints
    return cfg


def eval_config(d: dict) -> EvalConfig:
    fields = {f.name for f in dataclasses.fields(EvalConfig)}
    unknown = set(d) - fields
    if unknown:
        raise ValueError(f"unknown eval keys: {sorted(unknown)}")
    d = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()}
    if "continuous_reveal" in d:
        d["continuous_reveal"] = {k: tuple(v) for k, v in d["continuous_reveal"].items()}
    return EvalConfig(**d)


def load_dataset(d: dict):
    from .data import synthetic_dataset
    from .loaders import from_csv, load_geolife
    kind = d.get("loader", "csv")
    kw = {k: v for k, v in d.items() if k not in ("loader", "path")}
    if kind == "geolife":
        return load_geolife(d["path"], **kw)
    if kind == "csv":
        return from_csv(d.get("path"), **kw)
    if kind == "synthetic":
        return synthetic_dataset(**kw)
    raise ValueError(f"unknown dataset loader '{kind}'")
