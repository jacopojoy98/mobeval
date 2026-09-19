"""Experiment configuration files (YAML or JSON).

    output_dir: results/geolife
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


def apply_defaults(cfg: dict) -> dict:
    names = [m["name"] for m in cfg["models"]]
    if len(set(names)) != len(names):
        raise ValueError(f"model names must be unique: {names}")
    cfg.setdefault("output_dir", "results")
    cfg.setdefault("eval", {})
    cfg.setdefault("check_provenance", "error")
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
