"""Template: evaluate UniTraj, TrajGPT and your model on one dataset with one protocol.

    python examples/run_real.py --geolife /data/geolife --out results_geolife

Fill in the TODO(model) parts of mobeval/adapters/templates.py first. Every model
receives the SAME splits, masks, visit sequences, label subsets and baselines.
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mobeval import EvalConfig, EvaluationPipeline, markdown_report
from mobeval.loaders import load_geolife
from mobeval.report import family_summary, leaderboard

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def build_adapters(args, train_points):
    from mobeval.adapters.templates import MyModelAdapter, TrajGPTAdapter, UniTrajAdapter
    # Normalisation statistics MUST come from the training split (or from each model's own
    # pre-training stats if it was pre-trained elsewhere - then use exactly those).
    norm = {"lat_mean": train_points.lat.mean(), "lat_std": train_points.lat.std(),
            "lon_mean": train_points.lon.mean(), "lon_std": train_points.lon.std()}
    adapters = []
    for tag, ckpt in enumerate(args.mymodel or []):          # several checkpoints = several training seeds
        adapters.append(MyModelAdapter(ckpt, norm, args.device, run_tag=f"seed{tag}"))
    if args.unitraj:
        adapters.append(UniTrajAdapter(args.unitraj, norm, args.device))
    if args.trajgpt:
        adapters.append(TrajGPTAdapter(args.trajgpt, h3_resolution=args.h3_res, device=args.device))
    return adapters


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--geolife", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--mymodel", nargs="*")
    ap.add_argument("--unitraj")
    ap.add_argument("--trajgpt")
    ap.add_argument("--h3-res", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--split-by", default="time", choices=["time", "user"])
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ds = load_geolife(args.geolife)
    cfg = EvalConfig(
        split_by=args.split_by,
        window_length=64,            # match the models' context length; resample upstream if they differ
        grid_cell_m=500,             # ~ H3 res 8 edge (~460 m); keep token size and grid comparable
        recovery_ratios=(0.25, 0.5, 0.75), recovery_kinds=("random", "block"),
        eval_seeds=(0, 1, 2), n_boot=1000, max_eval_samples=5000,
        label_fractions=(1.0, 0.1, 0.01),
        # TrajGPT predicts duration after the region: evaluate that conditional variant explicitly
        continuous_reveal={"duration": ("location",)},
    )
    pipe = EvaluationPipeline(cfg)
    ctx = pipe.prepare(ds)
    store = pipe.run(build_adapters(args, ctx.splits["train"].points), ctx)
    store.save(out / "results.jsonl")
    leaderboard(store).to_csv(out / "leaderboard.csv", index=False)
    family_summary(store.to_frame()).to_csv(out / "family_summary.csv")
    (out / "report.md").write_text(markdown_report(store, ctx, f"Evaluation on {ds.name}"))
