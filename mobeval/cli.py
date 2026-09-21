"""Command line interface.

    python -m mobeval train    --config exp.yaml [--models A B]   # train models that have a `train` section
    python -m mobeval evaluate --config exp.yaml [--models A B]   # evaluate all models, write report
    python -m mobeval run      --config exp.yaml                  # train (missing checkpoints) + evaluate
    python -m mobeval run      --config exp.yaml --resume         # continue the newest run after a crash
    python -m mobeval context  --config exp.yaml --out data/context   # POI/road features from OSM
    python -m mobeval status   [--config exp.yaml | --progress-dir DIR] [--watch 10]  # how is it going?
    python -m mobeval smoke    [--out DIR]                        # tiny synthetic end-to-end check
    python -m mobeval info     --config exp.yaml                  # dataset/split summary + fingerprint

Nothing finished ever waits for the end of the job: a checkpoint is written (and copied to
--persist-dir) after every epoch that improves, and results are written after every task. Each
invocation gets its own <output_dir>/runs/<run_id>/ directory, with <output_dir>/latest pointing
at the newest, so repeated or concurrent jobs never overwrite each other.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

from . import progress
from .progress import Interrupted

log = logging.getLogger("mobeval")


RUN_COMMANDS = ("train", "evaluate", "run", "smoke")


def ensure_progress(directory, command, meta=None):
    """Start progress reporting, or update the already-running reporter."""
    rep = progress.get()
    if rep.active:
        if meta:
            rep.state.update(meta)
            rep._flush()
        return rep
    return progress.start(directory, command, meta=meta)


def progress_dir(cfg, args=None):
    """Where live progress is written. On a cluster this must be visible from the submitting
    machine, so it deliberately does NOT follow output_dir when that points at node-local scratch."""
    explicit = getattr(args, "progress_dir", None) or os.environ.get("MOBEVAL_PROGRESS_DIR")
    if explicit:
        return Path(explicit)
    if cfg and cfg.get("progress_dir"):
        return Path(cfg["progress_dir"])
    return Path(cfg["output_dir"]) / "progress" if cfg else None


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


def _resume_training(spec, cfg, only_missing):
    """Should this model be trained, and should it continue from an existing checkpoint?

    A checkpoint written by an interrupted job holds the best epoch it reached, marked
    incomplete. Skipping it would quietly evaluate a half-trained model, so it is continued
    from instead - which is also what makes a re-submitted PBS job pick up where it left off.
    """
    from .nn.common import checkpoint_status, last_epoch
    from .registry import checkpoint_path
    path = checkpoint_path(spec, cfg["output_dir"])
    status = checkpoint_status(path)
    if not only_missing or status == "missing":
        return True, None
    if status == "complete":
        log.info(f"{spec['name']}: checkpoint exists, skipping training")
        return False, None
    epoch = last_epoch(path)
    log.warning(f"{spec['name']}: {path.name} is from an interrupted run"
                + (f" (best epoch {epoch})" if epoch else "")
                + " - continuing training from those weights")
    return True, str(path)


def cmd_train(cfg, names, ctx=None, only_missing=False):
    from .registry import TRAIN_METHOD, train_model
    ctx = ctx or _context(cfg)[1]
    rep = progress.get()
    todo = [m for m in _select(cfg, names)
            if "train" in m and m["type"] in TRAIN_METHOD and _resume_training(m, cfg, only_missing)[0]]
    rep.set_plan(rep.state.get("steps_total", 0) + len(todo) if rep.active else len(todo))
    for m in _select(cfg, names):
        rep.model(m["name"], state="pending" if m in todo else "ready")
    trained_any = False
    for spec in _select(cfg, names):
        if "train" not in spec or spec["type"] not in TRAIN_METHOD:
            why = (f"its type '{spec['type']}' has no training routine" if spec["type"] not in TRAIN_METHOD
                   else "it has no `train:` section in the config (add one, or it is meant to be "
                        "loaded from an existing `checkpoint:`)")
            log.warning(f"NOT training {spec['name']}: {why}")
            progress.get().model(spec["name"], state="not trained", detail=why)
            continue
        train_it, continue_from = _resume_training(spec, cfg, only_missing)
        if not train_it:
            continue
        if continue_from:
            spec = {**spec, "train": {**spec["train"], "init_from": continue_from}}
        t0 = time.time()
        trained_any = True
        train_model(spec, ctx, cfg["output_dir"])
        log.info(f"{spec['name']}: trained in {time.time() - t0:.0f}s")
    if names and not trained_any:
        log.warning(f"nothing was trained for {names}: no checkpoint will be written")
    return ctx


def cmd_evaluate(cfg, names, pipe=None, ctx=None, resume=False):
    from . import layout
    from .registry import load_model
    from .report import family_summary, leaderboard, markdown_report
    from .results import ResultStore
    if ctx is None:
        pipe, ctx = _context(cfg)
    lay = layout.get()
    out = lay.run_dir if lay else Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    adapters = [load_model(s, ctx, cfg["output_dir"], cfg["check_provenance"]) for s in _select(cfg, names)]

    # Bound to its file from the start: every finished task is written out immediately, so a
    # crash (or a walltime kill) half-way through leaves the completed tasks on disk.
    store = ResultStore(out / "results.jsonl", resume=resume)
    if lay is not None:
        store.on_persist = lay.mirror
    if store.resumed:
        log.info(f"resuming run {out.name}: {store.resumed} records kept from the previous attempt")
    store = pipe.run(adapters, ctx, store)

    leaderboard(store).to_csv(out / "leaderboard.csv", index=False)
    family_summary(store.to_frame()).to_csv(out / "family_summary.csv")
    progress.get().set_phase("writing report")
    (out / "report.md").write_text(markdown_report(store, ctx, cfg.get("title", f"Evaluation on {ctx.dataset_name}")))
    (out / "run_info.json").write_text(json.dumps({"fingerprint": ctx.fingerprint, "config": cfg,
                                                   "run_id": lay.run_id if lay else None,
                                                   "errors": [e[:3] for e in ctx.errors], "skipped": ctx.skipped},
                                                  indent=2, default=str))
    for name in ("leaderboard.csv", "family_summary.csv", "report.md", "run_info.json"):
        layout.mirror(out / name)
    for m, t, e, _ in ctx.errors:
        log.error(f"{m} · {t}: {e}")
    log.info(f"wrote {out}/report.md ({len(store.records)} records)")
    return store, ctx


def cmd_context(cfg, args):
    """Build POI / road-network context features for the dataset's area (TransferTraj uses them)."""
    from .config import load_dataset
    from .context_features import (bbox_area_km2, build_from_osm, clip_to_bbox, dataset_bbox, pois_from_file,
                                   roads_from_file, suggest_radius_m, thin, write_context)
    ds = load_dataset(cfg["dataset"])
    bbox = dataset_bbox(ds, args.pad_km)
    out = Path(args.out or Path(cfg["output_dir"]) / "context")
    log.info(f"dataset area: {bbox_area_km2(bbox):,.0f} km²  bbox lat {bbox[0]:.4f}..{bbox[1]:.4f} "
             f"lon {bbox[2]:.4f}..{bbox[3]:.4f}")
    if args.poi_file or args.road_file:
        poi = (thin(clip_to_bbox(pois_from_file(args.poi_file, layer=args.poi_layer), bbox), args.max_features)
               if args.poi_file else None)
        road = (thin(clip_to_bbox(roads_from_file(args.road_file, layer=args.road_layer,
                                                  max_spacing_m=args.road_spacing), bbox),
                     args.max_features) if args.road_file else None)
        ctx = write_context(out, poi, road, args.dim)
        ctx["_suggested_poi_dist"] = round(suggest_radius_m(poi, bbox) ** 2)
    else:
        ctx = build_from_osm(bbox, out, dim=args.dim, max_spacing_m=args.road_spacing,
                             max_features=args.max_features)
    radius = ctx.pop("_suggested_poi_dist", 250000)
    print("\nAdd this to the TransferTraj model in your config (paths are read at run time):\n")
    print("    adapter:")
    print("      context:")
    for k, v in ctx.items():
        print(f"        {k}: {v}")
    print(f"\n    arch: {{poi_dist: {radius}, rn_dist: {radius}}}   "
          f"# SQUARED metres, i.e. a {radius ** 0.5:.0f} m radius")
    return ctx


def cmd_status(args, cfg=None):
    """Show how running (and recent) jobs are doing, from their progress files."""
    d = progress_dir(cfg, args)
    if d is None:
        raise SystemExit("give --progress-dir, --config, or set MOBEVAL_PROGRESS_DIR")

    def render():
        runs = progress.load_runs(d, limit=args.limit)
        text = progress.format_runs(runs, verbose=args.verbose, ascii_only=args.ascii)
        if args.events and runs:
            text += f"\n\nlast {args.events} events of {runs[0]['run_id']}:\n"
            for e in progress.tail_events(runs[0]["_events"], args.events):
                text += "  " + progress.event_line(e) + "\n"
        return f"progress: {d}\n\n{text}"

    if not args.watch:
        print(render())
        return
    try:
        while True:
            print("\033[2J\033[H" + render(), flush=True)
            time.sleep(args.watch)
    except KeyboardInterrupt:
        pass


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
        {"name": "TransferTraj-tiny", "type": "transfertraj", "arch": {"embed_size": 16, "d_model": 32, "rafee_layer": 1},
         "train": {"epochs": 2, "batch_size": 16, "lr": 1e-3}},
        {"name": "KinematicRef", "type": "kinematic_ref"},
    ],
}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mobeval", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["train", "evaluate", "run", "smoke", "info", "context", "status"])
    ap.add_argument("--config")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--out", help="override output_dir")
    ap.add_argument("--device", help="override train.device for all models (e.g. cuda, cpu)")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--progress-dir", help="where live progress is written/read (must be visible from "
                                           "the submitting machine, not node-local scratch)")
    ap.add_argument("--persist-dir", help="durable directory (home/project, NOT scratch). Checkpoints are "
                                          "copied there as each model finishes training and results after "
                                          "each task, so a crash or walltime kill loses nothing finished.")
    ap.add_argument("--resume", nargs="?", const=True, metavar="RUN_ID",
                    help="continue the newest run (or the named one): keep its finished tasks and "
                         "only compute what is missing")
    s = ap.add_argument_group("status")
    s.add_argument("--watch", type=float, nargs="?", const=10.0, help="refresh every N seconds (default 10)")
    s.add_argument("--events", type=int, default=0, help="also show the last N events of the newest run")
    s.add_argument("--limit", type=int, default=5, help="how many runs to show (default 5)")
    s.add_argument("--ascii", action="store_true", help="plain ASCII progress bars")
    g = ap.add_argument_group("context")
    g.add_argument("--poi-file", help="GeoJSON/GeoPackage/shapefile/CSV of POIs (offline instead of OSM)")
    g.add_argument("--road-file", help="GeoJSON/GeoPackage/shapefile/CSV of road geometries (offline)")
    g.add_argument("--poi-layer", help="layer name, for a multi-layer GeoPackage")
    g.add_argument("--road-layer", help="layer name, for a multi-layer GeoPackage")
    g.add_argument("--dim", type=int, default=64, help="embedding dimension (default 64)")
    g.add_argument("--pad-km", type=float, default=2.0, help="padding around the data's bounding box")
    g.add_argument("--road-spacing", type=float, default=100.0, help="metres between road sample points")
    g.add_argument("--max-features", type=int, default=50000, help="cap on POIs / road points")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    if args.command == "status" and not args.config:
        return cmd_status(args) or 0                # inspecting only needs the progress directory

    # Start reporting BEFORE the config is read, whenever the directory is known without it, so that
    # a failure while loading the config still leaves a visible "failed" run rather than nothing.
    early = getattr(args, "progress_dir", None) or os.environ.get("MOBEVAL_PROGRESS_DIR")
    if early and args.command in RUN_COMMANDS:
        ensure_progress(early, args.command, {"config_path": args.config})
        progress.get().set_phase("loading config")
    try:
        cfg = _load_cfg(args, ap)
    except BaseException as e:                                          # noqa: BLE001
        progress.stop("failed", failure=repr(e))
        raise
    if args.command == "status":
        return cmd_status(args, cfg) or 0
    if args.command == "context":
        return cmd_context(cfg, args) and 0
    if args.command == "info":
        from . import layout
        _, ctx = _context(cfg)
        print(ctx.summary())
        print("fingerprint:", ctx.fingerprint)
        print("progress dir:", progress_dir(cfg, args))
        print(layout.from_config(cfg, "info", persist_dir=args.persist_dir).describe())
        return 0

    rep = ensure_progress(progress_dir(cfg, args), args.command,
                          {"config_path": args.config, "output_dir": cfg["output_dir"]})
    lay = _setup_layout(cfg, args, rep)
    log.info(f"progress: {rep.status_path if rep.active else 'disabled'}")
    log.info("\n" + lay.describe())
    if rep.active:
        rep.state.update({"run_dir": str(lay.run_dir), "persist_dir": str(lay.persist_dir or "")})
    _install_signal_handlers()
    try:
        if args.command == "train":
            # Plain `train` retrains. With --resume it skips models that finished and continues
            # one whose checkpoint is marked incomplete, which is what a re-submitted job wants.
            cmd_train(cfg, args.models, only_missing=bool(args.resume))
        elif args.command == "evaluate":
            cmd_evaluate(cfg, args.models, resume=bool(args.resume))
        else:                                   # run / smoke
            pipe, ctx = _context(cfg)
            cmd_train(cfg, args.models, ctx, only_missing=args.command == "run")
            store, ctx = cmd_evaluate(cfg, args.models, pipe, ctx, resume=bool(args.resume))
            if args.command == "smoke":
                ok = not ctx.errors
                progress.stop("done" if ok else "failed")
                print(("SMOKE TEST PASSED" if ok else "SMOKE TEST FAILED")
                      + f" - {len(store.records)} records in {lay.run_dir}/")
                return 0 if ok else 1
    except Interrupted as e:
        # SIGTERM, which on PBS means the walltime ran out. Everything finished so far is
        # already on disk; say where, and exit non-zero so `depend=afterok` does not continue.
        log.warning(f"{e}. Finished work is in {lay.persist_dir or lay.run_dir}; "
                    f"re-submit with --resume to carry on from here.")
        progress.stop("interrupted", failure=str(e))
        return 143
    except BaseException as e:                                          # noqa: BLE001
        progress.stop("failed", failure=repr(e))
        log.error(f"run failed; finished work is in {lay.persist_dir or lay.run_dir} "
                  f"(re-submit with --resume to keep it)")
        raise
    progress.stop("done")
    log.info(f"run {lay.run_id} complete -> {lay.persist_dir or lay.output_dir}")
    return 0


def _install_signal_handlers():
    """Turn SIGTERM/SIGINT into an ordinary exception, so the `except` clauses above run.

    PBS sends SIGTERM before SIGKILL when the walltime expires. Without this the process dies
    where it stands, the progress file is left saying "running" for ever, and no summary is
    written - even though the per-task results and per-epoch checkpoints are safely on disk.
    """
    def handler(signum, _frame):
        raise Interrupted(f"received {signal.Signals(signum).name}")

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):           # not on the main thread; nothing we can do
            pass


def _setup_layout(cfg, args, rep):
    """Decide where this run writes, and make a resumed run reuse the earlier directory."""
    from . import layout
    resume_dir = None
    if args.resume:
        probe = layout.from_config(cfg, args.command, run_id=rep.run_id if rep.active else None,
                                   persist_dir=args.persist_dir)
        if args.resume is not True and str(args.resume) != "latest":
            resume_dir = probe.output_dir / layout.RUNS / str(args.resume)
            if not resume_dir.is_dir():
                raise SystemExit(f"--resume {args.resume}: no such run directory {resume_dir}")
        else:
            previous = probe.previous_runs()
            resume_dir = previous[-1] if previous else None
            if resume_dir is None:
                log.info("--resume: no previous run found, starting a fresh one")
    lay = layout.from_config(cfg, args.command, run_id=rep.run_id if rep.active else None,
                             persist_dir=args.persist_dir, resume_dir=resume_dir)
    layout.activate(lay).create()
    lay.restore_checkpoints()                  # reuse weights an earlier job already mirrored home
    if resume_dir is not None:
        lay.restore_run()                      # the earlier attempt may only survive in persist_dir
    return lay


def _load_cfg(args, ap):
    """Build the configuration dictionary, applying the command-line overrides."""
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
    return cfg


if __name__ == "__main__":
    sys.exit(main())
