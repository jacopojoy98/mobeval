"""End-to-end smoke run on synthetic data with the two reference adapters."""
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mobeval import EvalConfig, EvaluationPipeline, markdown_report, synthetic_dataset
from mobeval.adapters.reference import KinematicReference, WeakReference

logging.basicConfig(level=logging.INFO, format="%(message)s")

if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "results_synthetic")
    out.mkdir(exist_ok=True)
    t0 = time.time()
    ds = synthetic_dataset(n_users=30, n_days=10, seed=0)
    cfg = EvalConfig(window_length=32, recovery_ratios=(0.25, 0.5), eval_seeds=(0, 1), n_boot=200,
                     max_eval_samples=600, visit_context=6, label_fractions=(1.0, 0.2))
    pipe = EvaluationPipeline(cfg)
    ctx = pipe.prepare(ds)
    store = pipe.run([KinematicReference(), WeakReference()], ctx)
    store.save(out / "results.jsonl")
    (out / "report.md").write_text(markdown_report(store, ctx, "Synthetic smoke run"))
    from mobeval.report import leaderboard
    leaderboard(store).to_csv(out / "leaderboard.csv", index=False)
    print(f"\n{len(store.records)} records in {time.time() - t0:.1f}s -> {out}/")
    for m, t, e, tb in ctx.errors:
        print("ERROR", m, t, e, tb, sep="\n")
