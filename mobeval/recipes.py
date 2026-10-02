"""Training recipes that reproduce each model's original methodology.

    models:
      - {name: UniTraj-paper, type: unitraj, recipe: paper}
      - {name: TrajGPT-code,  type: trajgpt, recipe: code}

`paper` follows the publication. Where the paper is silent it follows the released code, and it
does NOT reproduce defects of the released code that would corrupt an evaluation (target leaks,
test data read during preprocessing, a training loop that stops after one batch). Every such
choice is listed under `not_reproduced` and printed when the recipe is used.

`code` follows the released code, including its differences from the paper, except where the
code cannot be what produced the paper's results (listed under `not_reproduced` too).

Anything set explicitly in the config overrides the recipe, and every override is logged and stored
in the checkpoint's provenance, so a "paper" checkpoint with changes never passes for the original.

A recipe covers the MODEL's training. The paper's DATA preparation and evaluation settings are in
`eval` (checked against the run's `eval:` section, with a warning for every difference) and in the
ready-made configs examples/configs/paper_<model>.yaml, which also point at the paper's dataset.

CLIP-Mobility has no publication, so it has no recipe.
"""
from __future__ import annotations

import copy
import logging
from typing import Dict, List, Tuple

log = logging.getLogger("mobeval.recipes")

_ADAM = {"optimizer": "adam", "weight_decay": 0.0, "grad_clip": 0.0, "drop_last": False}
# Options that point at data files rather than change the method (e.g. TransferTraj's POI/road
# embeddings): setting them is not an override of the recipe.
DATA_OPTIONS = {"context", "roads_file"}

RECIPES: Dict[str, Dict[str, dict]] = {
    # ============================================================== UniTraj
    "unitraj": {
        "paper": {
            "sources": ["UniTraj paper (Zhu et al.), Sec. on ATR / masking, Table 5 (mask mix)",
                        "github.com/Yasoz/UniTraj main.py, utils/dataset.py, utils/config.py"],
            "arch": {"trajectory_length": 200, "patch_size": 1, "embedding_dim": 128, "encoder_layers": 8,
                     "encoder_heads": 4, "decoder_layers": 4, "decoder_heads": 4},
            "train": {**_ADAM, "lr": 1e-3, "batch_size": 1024, "epochs": 200, "patience": 20,
                      "scheduler": "plateau", "plateau_factor": 0.5, "plateau_patience": 2},
            "options": {"sampling": "original", "sample_unit": "trajectory", "resample": "atr",
                        "mask_mix": {"random": 0.7, "rdp": 0.15, "block": 0.05, "last_n": 0.10},
                        "mask_ratio": 0.5, "mask_endpoints": True, "fixed_hidden_count": True,
                        "offset_from": "first_visible", "loss_norm": "original", "rdp_epsilon": 1e-4,
                        "fit_norm": False, "min_points": 28},
            "eval": {"window_length": 200, "recovery_keep_endpoints": False},
            "dataset": "worldtrace",
            "not_reproduced": [
                "anchoring offsets at trajectory[0] even when that point is masked (at evaluation this "
                "hands the model a hidden coordinate); the first visible point is used",
                "the repository's training loop `break`s after one batch per epoch; the paper's 200 "
                "epochs over the data are used instead",
                "empty resampling bins are dropped (pandas would emit NaN rows; see nn/unitraj_sampling.py)",
                "trajectories under 28 points are skipped: the original has no filter, but its block mask "
                "(5-14 points plus a fill to half the length) fails on them",
                "the curated 1.1M-trajectory WorldTrace subset is not released; the public WorldTrace is used",
            ],
        },
        "code": {
            "sources": ["github.com/Yasoz/UniTraj main.py, utils/dataset.py, utils/config.py"],
            "arch": {"trajectory_length": 200, "patch_size": 1, "embedding_dim": 128, "encoder_layers": 8,
                     "encoder_heads": 4, "decoder_layers": 4, "decoder_heads": 4},
            "train": {**_ADAM, "lr": 1e-3, "batch_size": 1024, "epochs": 1000, "patience": 20,
                      "scheduler": "plateau", "plateau_factor": 0.5, "plateau_patience": 2},
            "options": {"sampling": "original", "sample_unit": "trajectory", "resample": "atr",
                        "mask_mix": {"random": 0.7, "rdp": 0.15, "block": 0.05, "last_n": 0.10},
                        "mask_ratio": 0.5, "mask_endpoints": True, "fixed_hidden_count": True,
                        "offset_from": "first", "loss_norm": "original", "rdp_epsilon": 1e-4,
                        "fit_norm": False, "min_points": 28},
            "eval": {"window_length": 200, "recovery_keep_endpoints": False},
            "dataset": "worldtrace",
            "not_reproduced": [
                "the one-batch-per-epoch `break` (1000 'epochs' would be 1000 batches in total, which "
                "cannot have produced the published model); full epochs are used",
                "empty resampling bins are dropped (pandas would emit NaN rows)",
                "evaluation always anchors offsets at the first visible point (training uses trajectory[0])",
            ],
        },
    },
    # ============================================================== TrajGPT
    "trajgpt": {
        "paper": {
            "sources": ["TrajGPT paper (Hsu et al.), experimental settings for GeoLife",
                        "github.com/ktxlh/TrajGPT at the paper-era commit 49aad40"],
            "arch": {"num_layers": 2, "num_heads": 8, "d_feedforward": 32, "d_embed": 32, "num_gaussians": 3,
                     "input_order": "fixed"},
            "train": {**_ADAM, "lr": 1e-4, "batch_size": 64, "epochs": 2000, "patience": 10, "scheduler": "none",
                      "seed": 0},
            "options": {"sequences": "original", "seq_len": 128, "instance_stride": 1, "travel_loss_mask": False,
                        "init_gmm": False, "clip_travel": True, "min_scale_h": 1e-6, "time_reference": "global",
                        "time_input_unit_s": 3600.0, "tokenizer": {"backend": "h3", "h3_resolution": 7}},
            "eval": {"staypoint_method": "points", "staypoint_dist_m": 200.0, "staypoint_time_s": 600.0,
                     "visit_context": 127, "grid_backend": "h3", "grid_h3_resolution": 7,
                     "travel_time_max_h": 4.0, "continuous_paper_conditioning": True},
            "dataset": "geolife",
            "not_reproduced": [
                "the code's input order [location, arrival, departure, region], which lets the time heads "
                "read the target's own arrival/departure; the paper's factorisation order "
                "[region, location, arrival, departure] is used",
                "the region vocabulary, time origin and 99th-percentile clipping come from the TRAIN split; "
                "the original computes them over all splits, i.e. reads the test data",
                "instances are assigned to splits by their target visits, with other splits' visits kept as "
                "history but excluded from the loss; the original splits overlapping instances by start "
                "time, so test visits also appear in training instances",
                "visit infilling (the paper's first task) is not implemented; next-visit prediction is",
                "local metric projection instead of UTM (sub-percent difference at city scale)",
            ],
        },
        "code": {
            "sources": ["github.com/ktxlh/TrajGPT at the paper-era commit 49aad40 (main.py, utils/*)"],
            "arch": {"num_layers": 4, "num_heads": 2, "d_feedforward": 32, "d_embed": 32, "num_gaussians": 3,
                     "input_order": "legacy"},
            "train": {**_ADAM, "lr": 1e-4, "batch_size": 64, "epochs": 2000, "patience": 50, "scheduler": "none"},
            "options": {"sequences": "original", "seq_len": 128, "instance_stride": 1, "travel_loss_mask": False,
                        "init_gmm": False, "clip_travel": True, "min_scale_h": 1e-6, "time_reference": "global",
                        "time_input_unit_s": 3600.0, "tokenizer": {"backend": "h3", "h3_resolution": 7}},
            "eval": {"staypoint_method": "points", "staypoint_dist_m": 100.0, "staypoint_time_s": 300.0,
                     "visit_context": 127, "grid_backend": "h3", "grid_h3_resolution": 7,
                     "travel_time_max_h": 4.0, "continuous_paper_conditioning": True},
            "dataset": "geolife",
            "not_reproduced": [
                "vocabulary, time origin and percentile clipping from the TRAIN split, not all splits",
                "split assignment by target visit (see the paper recipe)",
                "trackintel's 15-minute gap threshold in staypoint detection",
                "the later HEAD code (Adafactor, times in days, users with >= 20 visits) - this is 49aad40",
            ],
            "warning": "input_order 'legacy' reproduces the original's target leak in the travel-time and "
                       "duration heads; their training loss is not a valid measure of forecasting skill "
                       "(mobeval never feeds the target's times at evaluation, so its scores stay honest).",
        },
    },
    # ============================================================== TransferTraj
    "transfertraj": {
        "paper": {
            "sources": ["TransferTraj paper (arXiv 2505.12672), Sec. 4.1-4.2 and implementation details",
                        "github.com/wtl52656/TransferTraj models/, data.py, pipeline.py"],
            "arch": {"embed_size": 64, "d_model": 128, "rafee_layer": 2, "poi_dist": 10_000, "rn_dist": 10_000},
            "train": {**_ADAM, "lr": 1e-3, "batch_size": 64, "epochs": 30, "patience": 10**6, "scheduler": "none",
                      "restore_best": False},
            "options": {"coord_scale": 1.0, "masking": "paper", "feature_mask_prob": 0.2, "objective": "pretrain"},
            "eval": {"recovery_schemes": [["last", 5], ["keep_every", 8]]},
            "dataset": "didi_h5",
            "finetune": {"train": {**_ADAM, "lr": 1e-3, "batch_size": 64, "epochs": 30, "patience": 10**6,
                                   "scheduler": "step", "step_size": 5, "step_gamma": 0.5, "restore_best": False},
                         "note": "the paper fine-tunes per task (objective: tp or trec, with init_from the "
                                 "pre-trained checkpoint); it does not state the fine-tuning epochs, 30 are used"},
            "not_reproduced": [
                "fixed-length windows instead of whole trips of 5-120 points",
                "local metric projection instead of UTM",
                "routing noise only in training mode (the original also adds it at evaluation)",
                "cross-city zero-/few-shot transfer is a data choice: train on one city's config, evaluate "
                "on another's",
            ],
        },
        "code": {
            "sources": ["github.com/wtl52656/TransferTraj data.py (PretrainPadder), settings/local_test.json"],
            "arch": {"embed_size": 64, "d_model": 128, "rafee_layer": 2, "poi_dist": 100, "rn_dist": 100},
            "train": {**_ADAM, "lr": 1e-3, "batch_size": 64, "epochs": 30, "patience": 10**6, "scheduler": "none",
                      "restore_best": False},
            "options": {"coord_scale": 1.0, "masking": "code", "span_div_ratio": 0.2, "span_mask_ratio": 0.4,
                        "feature_mask_prob": 0.2, "objective": "pretrain"},
            "eval": {"recovery_schemes": [["last", 5], ["keep_every", 8]]},
            "dataset": "didi_h5",
            "not_reproduced": [
                "batch size and epochs: the repository only ships a smoke-test settings file (batch 16, "
                "2 epochs); the paper's 64 and 30 are used",
                "fixed-length windows instead of whole trips; local projection instead of UTM",
            ],
        },
    },
    # ============================================================== OmniTraj
    "omnitraj": {
        "code": {
            "sources": ["github.com/Yasoz/OmniTraj main.py, utils/config.py, utils/dataset.py (commit 3ce11d6)"],
            "arch": {},
            "train": {"optimizer": "adamw", "lr": 2e-4, "weight_decay": 1e-4, "scheduler": "cosine",
                      "cosine_eta_min": 1e-5, "epochs": 500, "batch_size": 1536, "grad_clip": 1.0,
                      "patience": 10**6, "restore_best": True, "drop_last": False},
            "options": {"loss": "code", "pairs": "code", "sample_unit": "trajectory", "min_points": 20, "grid_n": 16,
                        "augment_val": True, "projection_dim": 512, "interpolation": "pchip"},
            "eval": {"retrieval_db_size": 20000},
            "dataset": "omnitraj_city",
            "not_reproduced": [
                "the authors' preprocessing is not public; it is reconstructed (nn/omnitraj_prep.py): RDP "
                "topology reproduces the sample exactly, the 16x16 grid 99.9% of its cell ids; the "
                "interpolant (PCHIP over the point index) and the map matcher are inferred, not confirmed",
                "the Chengdu / Xi'an data (1.2M trips each) are no longer available; any city subset of other "
                "data is a transfer of the method, not a reproduction of the numbers",
                "road-segment and region ids are compacted to the training vocabulary (the original feeds raw "
                "ids, where segment 0 collides with padding)",
                "evaluation retrieves among test WINDOWS resampled to 200 points, not whole trips",
            ],
        },
        "paper": {
            "sources": ["OmniTraj paper (KDD 2025), Sec. 3.3 (Eqs. 9-10), Appendix A and B.1",
                        "github.com/Yasoz/OmniTraj where the paper is silent"],
            "arch": {},
            "train": {"optimizer": "adam", "lr": 2e-4, "weight_decay": 0.0, "scheduler": "cosine",
                      "cosine_eta_min": 1e-5, "epochs": 500, "batch_size": 1536, "grad_clip": 1.0,
                      "patience": 10**6, "restore_best": True, "drop_last": False},
            "options": {"loss": "paper", "pairs": "paper", "sample_unit": "trajectory", "min_points": 20,
                        "grid_n": 16, "augment_val": True, "projection_dim": 512, "interpolation": "pchip"},
            "eval": {"retrieval_db_size": 20000},
            "dataset": "omnitraj_city",
            "not_reproduced": [
                "the authors' preprocessing is not public; it is reconstructed (nn/omnitraj_prep.py): RDP "
                "topology reproduces the sample exactly, the 16x16 grid 99.9% of its cell ids; the "
                "interpolant (PCHIP over the point index) and the map matcher are inferred, not confirmed",
                "the Chengdu / Xi'an data (1.2M trips each) are no longer available; any city subset of other "
                "data is a transfer of the method, not a reproduction of the numbers",
                "road-segment and region ids are compacted to the training vocabulary (the original feeds raw "
                "ids, where segment 0 collides with padding)",
                "evaluation retrieves among test WINDOWS resampled to 200 points, not whole trips",
                "the paper's loss is vanilla InfoNCE on cosine similarity with the trajectory contrasted "
                "against each modality (Eq. 10), both used here; its temperature is not stated, so the "
                "code's learnable temperature (initial value 1.0, soft for cosine logits) is kept",
                "epochs, batch size, schedule and clipping are not in the paper; the code's values are used",
            ],
        },
    },
}


def available(model_type: str) -> List[str]:
    return sorted(RECIPES.get(model_type, {}))


def get(model_type: str, name: str) -> dict:
    if model_type not in RECIPES:
        why = ("CLIP-Mobility has no publication to reproduce" if model_type == "clip_mobility"
               else f"no recipes for model type '{model_type}'")
        raise ValueError(f"recipe '{name}': {why}")
    if name not in RECIPES[model_type]:
        raise ValueError(f"unknown recipe '{name}' for {model_type}; available: {available(model_type)}")
    return copy.deepcopy(RECIPES[model_type][name])


def apply(spec: dict) -> Tuple[dict, dict]:
    """Merge a model spec over its recipe. Returns (resolved spec, recipe record for provenance).
    Explicit values in the spec win; each one that differs from the recipe is reported."""
    name = spec.get("recipe")
    if not name:
        return spec, {}
    r = get(spec["type"], name)
    out = copy.deepcopy(spec)
    user_train = dict(spec.get("train", {}))
    user_opts = dict(user_train.pop("options", {}))
    if user_opts.get("objective", "pretrain") != "pretrain" and "finetune" in r:
        # task fine-tuning (e.g. TransferTraj tp / trec) has its own optimisation settings
        r["train"] = r["finetune"]["train"]
        r["options"] = {**r["options"], "objective": user_opts["objective"]}
        r.setdefault("not_reproduced", []).append(r["finetune"]["note"])
    overrides = []
    for key, preset, user in (("arch", r.get("arch", {}), spec.get("arch", {}) or {}),
                              ("train", r.get("train", {}), {k: v for k, v in user_train.items()
                                                             if k not in ("out", "init_from", "device")}),
                              ("options", r.get("options", {}), user_opts)):
        for k, v in user.items():
            if k in preset and preset[k] != v:
                overrides.append(f"{key}.{k}: {preset[k]!r} -> {v!r}")
            elif k not in preset and key != "train" and k not in DATA_OPTIONS:
                overrides.append(f"{key}.{k}: (not in recipe) {v!r}")
    arch = {**r.get("arch", {}), **(spec.get("arch") or {})}
    opts = {**r.get("options", {}), **user_opts}
    for k, v in (spec.get("adapter") or {}).items():       # adapter: {...} is passed to training too
        if k in opts:
            if opts[k] != v:
                overrides.append(f"adapter.{k}: {opts[k]!r} -> {v!r}")
            del opts[k]
    train = {**r.get("train", {}), **user_train, "options": opts}
    if arch:
        out["arch"] = arch
    out["train"] = train
    record = {"recipe": name, "overrides": overrides, "not_reproduced": r.get("not_reproduced", []),
              "sources": r.get("sources", [])}
    log.info(f"{spec['name']}: {spec['type']} recipe '{name}' ({'; '.join(r.get('sources', []))})")
    for item in r.get("not_reproduced", []):
        log.info(f"{spec['name']}: recipe '{name}' deliberately does not reproduce: {item}")
    if r.get("warning"):
        log.warning(f"{spec['name']}: {r['warning']}")
    for o in overrides:
        log.warning(f"{spec['name']}: overrides the '{name}' recipe - {o}. Results are no longer the "
                    f"original methodology.")
    return out, record


def check_eval(spec_name: str, model_type: str, recipe: str, cfg) -> List[str]:
    """Differences between the run's evaluation/data settings and the recipe's. Warned, not enforced:
    a benchmark on other data legitimately differs, but it must not be mistaken for a reproduction."""
    diffs = []
    for k, v in get(model_type, recipe).get("eval", {}).items():
        have = getattr(cfg, k, None)
        norm = lambda x: [list(i) if isinstance(i, (list, tuple)) else i for i in x] if isinstance(x, (list, tuple)) else x
        if k == "recovery_schemes":
            missing = [s for s in norm(v) if s not in norm(have or [])]
            if missing:
                diffs.append(f"eval.recovery_schemes lacks {missing}")
        elif norm(have) != norm(v):
            diffs.append(f"eval.{k} = {have!r}, the paper uses {v!r}")
    for d in diffs:
        log.warning(f"{spec_name}: {d} - the model is trained with the '{recipe}' recipe, but this run's "
                    f"data/evaluation settings differ from the paper's")
    return diffs
