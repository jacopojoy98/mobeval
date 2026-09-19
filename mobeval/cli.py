"""Command line interface.

    python -m mobeval train    --config exp.yaml [--models A B]   # train models that have a `train` section
    python -m mobeval evaluate --config exp.yaml [--models A B]   # evaluate all models, write report
    python -m mobeval run      --config exp.yaml                  # train (missing checkpoints) + evaluate
    python -m mobeval smoke    [--out DIR]                        # tiny synthetic end-to-end check
    python -m mobeval info     --config exp.yaml                  # dataset/split summary + fingerprint
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

log = logging.getLogger("mobeval")


def _select(cfg, names):
    models = cfg["models"]
    if names:
        missing = set(names) - {m["name"] for m in models}
        if missing:
            raise SystemExit(f"unknown model(s): {sorted(missing)}")
        models = [m for m in models if m["name"] in names]
    return models


def _context(cfg):
    from .config import eval_config, load_dataset
    from .runner import EvaluationPipeline
    pipe = EvaluationPipeline(eval_config(cfg["eval"]))
    ctx = pipe.prepare(load_dataset(cfg["dataset"]))
    return pipe, ctx


def cmd_train(cfg, names, ctx=None, only_missing=False):
    from .registry import TRAIN_METHOD, checkpoint_path, train_model
    ctx = ctx or _context(cfg)[1]
    for spec in _select(cfg, names):
        if "train" not in spec or spec["type"] not in TRAIN_METHOD:
            continue
        if only_missing and checkpoint_path(spec, cfg["output_dir"]).exists():
            log.info(f"{spec['name']}: checkpoint exists, skipping training")
            continue
        t0 = time.time()
        train_model(spec, ctx, cfg["output_dir"])
        log.info(f"{spec['name']}: trained in {time.time() - t0:.0f}s")
    return ctx


def cmd_evaluate(cfg, names, pipe=None, ctx=None):
    from .registry import load_model
    from .report import family_summary, leaderboard, markdown_report
    if ctx is None:
        pipe, ctx = _context(cfg)
    out = Path(cfg["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    adapters = [load_model(s, ctx, cfg["output_dir"], cfg["check_provenance"]) for s in _select(cfg, names)]
    store = pipe.run(adapters, ctx)
    store.save(out / "results.jsonl")
    leaderboard(store).to_csv(out / "leaderboard.csv", index=False)
    family_summary(store.to_frame()).to_csv(out / "family_summary.csv")
    (out / "report.md").write_text(markdown_report(store, ctx, cfg.get("title", f"Evaluation on {ctx.dataset_name}")))
    (out / "run_info.json").write_text(json.dumps({"fingerprint": ctx.fingerprint, "config": cfg,
                                                   "errors": [e[:3] for e in ctx.errors], "skipped": ctx.skipped},
                                                  indent=2, default=str))
    for m, t, e, _ in ctx.errors:
        log.error(f"{m} · {t}: {e}")
    log.info(f"wrote {out}/report.md ({len(store.records)} records)")
    return store, ctx


SMOKE = {
    "output_dir": "mobeval_smoke", "title": "mobeval smoke test",
    "dataset": {"loader": "synthetic", "n_users": 20, "n_days": 8, "seed": 0},
    "eval": {"window_length": 32, "recovery_ratios": [0.5], "recovery_kinds": ["block"], "eval_seeds": [0],
             "n_boot": 50, "max_eval_samples": 150, "visit_context": 4, "label_fractions": [1.0],
             "generation_max_trajectories": 20},
    "models": [
        {"name": "UniTraj-tiny", "type": "unitraj", "arch": {"trajectory_length": 32, "embedding_dim": 32,
                                                              "encoder_layers": 1, "decoder_layers": 1},
         "train": {"epochs": 2, "batch_size": 32, "lr": 1e-3}},
        {"name": "TrajGPT-tiny", "type": "trajgpt", "arch": {"num_layers": 1},
         "train": {"epochs": 2, "batch_size": 32, "options": {"tokenizer": {"cell_m": 1000}}}},
        {"name": "CLIP-tiny", "type": "clip_mobility", "arch": {"d_model": 32, "nhead": 2, "num_layers": 1,
                                                                 "dim_feedforward": 64, "embedding_dim": 16},
         "train": {"epochs": 2, "batch_size": 32, "lr": 1e-3}, "adapter": {"head_train": {"epochs": 5}}},
        {"name": "KinematicRef", "type": "kinematic_ref"},
    ],
}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mobeval", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["train", "evaluate", "run", "smoke", "info"])
    ap.add_argument("--config")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--out", help="override output_dir")
    ap.add_argument("--device", help="override train.device for all models (e.g. cuda, cpu)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    if args.command == "smoke":
        from .config import apply_defaults
        cfg = apply_defaults(json.loads(json.dumps(SMOKE)))
    else:
        if not args.config:
            ap.error("--config is required")
        from .config import load_config
        cfg = load_config(args.config)
    if args.out:
        cfg["output_dir"] = args.out
    if args.device:
        for m in cfg["models"]:
            if "train" in m:
                m["train"]["device"] = args.device
            m.setdefault("adapter", {})["device"] = args.device

    if args.command == "info":
        _, ctx = _context(cfg)
        print(ctx.summary()); print("fingerprint:", ctx.fingerprint)
    elif args.command == "train":
        cmd_train(cfg, args.models)
    elif args.command == "evaluate":
        cmd_evaluate(cfg, args.models)
    else:                                   # run / smoke
        pipe, ctx = _context(cfg)
        cmd_train(cfg, args.models, ctx, only_missing=args.command == "run")
        store, ctx = cmd_evaluate(cfg, args.models, pipe, ctx)
        if args.command == "smoke":
            ok = not ctx.errors
            print(("SMOKE TEST PASSED" if ok else "SMOKE TEST FAILED") + f" - {len(store.records)} records in {cfg['output_dir']}/")
            return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
