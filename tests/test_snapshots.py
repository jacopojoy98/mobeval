"""`mobeval snapshots`: which epoch of a training does best (snapshots from train.save_every)."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mobeval import snapshots as S


# ------------------------------------------------------------------ choosing the files and epochs
def test_choose_every_and_max():
    eps = list(range(10, 210, 10))                      # 10 .. 200
    assert S.choose(eps) == eps
    assert S.choose(eps, every=50) == [50, 100, 150, 200]
    assert S.choose(eps, max_n=3) == [10, 110, 200]    # evenly spread, first and last kept
    assert S.choose(eps, every=20, max_n=4) == [20, 80, 140, 200]
    assert S.choose(eps, max_n=0) == eps and S.choose(eps, max_n=1) == [200]
    assert S.choose([], every=5) == []


def _snap(folder: Path, model: str, epoch: int, val_loss: float, mtime=None) -> Path:
    d = folder / model
    d.mkdir(parents=True, exist_ok=True)
    pt = d / f"epoch_{epoch:04d}.pt"
    pt.write_bytes(b"weights")
    hist = [{"epoch": e, "train_loss": 2.0 / e, "val_loss": val_loss if e == epoch else 9.0} for e in range(1, epoch + 1)]
    pt.with_suffix(".json").write_text(json.dumps({"history": hist}))
    if mtime is not None:
        os.utime(pt, (mtime, mtime))
    return pt


def test_find_snapshots_takes_the_newer_copy_of_an_epoch(tmp_path):
    a, b = tmp_path / "staged_out", tmp_path / "mirrored"
    old = _snap(a, "M", 10, 1.0, mtime=1_000_000)
    new = _snap(b, "M", 10, 1.0, mtime=2_000_000)
    _snap(a, "M", 20, 1.0)
    (a / "M" / "epoch_0030.json").write_text("{}")      # a sidecar without weights is not a snapshot
    (a / "M" / "notes.txt").write_text("x")
    found = S.find_snapshots("M", [a, b, tmp_path / "missing"])
    assert list(found) == [10, 20] and found[10] == new and found[10] != old


def _cfg(tmp_path, models):
    return {"output_dir": str(tmp_path / "out"), "models": models}


def _checkpoint(path: Path, history):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"weights")
    path.with_suffix(".json").write_text(json.dumps({"history": history, "complete": True}))
    return path


def test_plan_scores_the_kept_weights_once(tmp_path):
    snaps = tmp_path / "snapshots"
    for e in (2, 4, 6, 8):
        _snap(snaps, "A", e, val_loss=1.0 / e)
        _snap(snaps, "B", e, val_loss=1.0 / e)
    hist = [{"epoch": e, "train_loss": 1.0, "val_loss": v} for e, v in [(1, 5), (2, 4), (3, 3.5), (4, 2), (5, 3), (6, 3)]]
    ck = {"A": _checkpoint(tmp_path / "ck" / "A.pt", hist),                 # kept epoch 4: one of the snapshots
          "B": _checkpoint(tmp_path / "ck" / "B.pt", hist[:3])}             # kept epoch 3: not a snapshot
    models = [{"name": "A", "type": "trajgpt", "train": {"epochs": 8}, "recipe": "paper",
               "adapter": {"device": "cuda", "batch_size": 64}},
              {"name": "B", "type": "trajgpt", "train": {"epochs": 8}},
              {"name": "KinematicRef", "type": "kinematic_ref"}]
    specs, rows, hists = S.plan(_cfg(tmp_path, models), None, [snaps], trainable={"trajgpt"},
                                checkpoint_for=lambda s: ck[s["name"]])
    names = [s["name"] for s in specs]
    assert names == ["A-ep0002", "A-ep0004", "A-ep0006", "A-ep0008", "B", "B-ep0002", "B-ep0004", "B-ep0006",
                     "B-ep0008"]
    a4 = next(s for s in specs if s["name"] == "A-ep0004")
    assert a4 == {"name": "A-ep0004", "type": "trajgpt", "checkpoint": str(snaps / "A" / "epoch_0004.pt"),
                  "adapter": {"device": "cuda", "batch_size": 64}}          # no train, no recipe: load only
    r = rows.set_index("model")
    assert r.loc["A-ep0004", "kept"] and not r.loc["A-ep0006", "kept"]
    assert r.loc["B", "kind"] == S.CHECKPOINT and r.loc["B", "kept"] and r.loc["B", "epoch"] == 3
    assert r.loc["A-ep0006", "val_loss"] == pytest.approx(1 / 6)           # the loss at that epoch
    assert len(hists["A"]) == 8                                            # newest snapshot's history: longer


def test_plan_explains_missing_snapshots_and_name_clashes(tmp_path):
    snaps = tmp_path / "snapshots"
    models = [{"name": "A", "type": "trajgpt", "train": {}}, {"name": "B", "type": "trajgpt", "train": {}}]
    cfg = _cfg(tmp_path, models)
    with pytest.raises(SystemExit, match="no snapshots for \\['A'\\].*save_every"):
        S.plan(cfg, ["A"], [snaps], trainable={"trajgpt"}, checkpoint_for=lambda s: tmp_path / "none.pt")
    with pytest.raises(SystemExit, match="unknown model"):
        S.plan(cfg, ["Z"], [snaps], trainable={"trajgpt"})
    _snap(snaps, "A", 5, 1.0)
    with pytest.raises(SystemExit, match="every 10"):                      # filtered away by --every
        S.plan(cfg, ["A"], [snaps], every=10, trainable={"trajgpt"}, checkpoint_for=lambda s: tmp_path / "none.pt")
    clash = _cfg(tmp_path, models + [{"name": "A-ep0005", "type": "trajgpt"}])
    with pytest.raises(SystemExit, match="already in the config"):
        S.plan(clash, ["A"], [snaps], trainable={"trajgpt"}, checkpoint_for=lambda s: tmp_path / "none.pt")


# ------------------------------------------------------------------ which epoch is best
def _lb(rows):
    cols = ["family", "task", "metric", "protocol", "model", "value", "ci_low", "ci_high", "higher_is_better", "unit"]
    return pd.DataFrame(rows, columns=cols)


def test_summary_best_epoch_ties_kept_and_family_balanced_overall():
    snaps = pd.DataFrame({"model": ["M-ep0010", "M-ep0020", "M-ep0030"], "base": "M", "kind": S.SNAPSHOT,
                          "kept": [False, True, False], "epoch": [10, 20, 30], "val_loss": [3.0, 1.0, 2.0],
                          "checkpoint": [f"/r/snapshots/M/epoch_{e:04d}.pt" for e in (10, 20, 30)]})
    rows = []
    # recovery (error, lower is better): epoch 30 best, 20 inside its interval, 10 outside
    for m, v in (("M-ep0010", 900.0), ("M-ep0020", 520.0), ("M-ep0030", 500.0)):
        rows.append(("recovery", "recovery/random@0.5", "ade_m", "native", m, v, v - 30, v + 30, False, "m"))
    rows.append(("recovery", "recovery/random@0.5", "ade_m", "native", "baseline:linear_interp", 200.0, 190, 210, False, "m"))
    # anomaly, three rows: epoch 10 best on all of them
    for kind in ("detour", "loop", "teleport"):
        for m, v in (("M-ep0010", 0.9), ("M-ep0020", 0.7), ("M-ep0030", 0.6)):
            rows.append(("anomaly", f"anomaly/{kind}", "roc_auc", "embedding_knn", m, v, v - 0.01, v + 0.01, True, "auc"))
    # a non-primary metric is ignored
    rows.append(("recovery", "recovery/random@0.5", "fde_m", "native", "M-ep0010", 1.0, 0, 2, False, "m"))
    summary, overall = S.summarize(_lb(rows), snaps)
    rec = summary[summary.family.eq("recovery")].iloc[0]
    assert (rec.best_epoch, rec.best_value, rec.tied_epochs) == (30, 500.0, "20")
    assert rec.kept_epoch == 20 and rec.kept_value == 520.0 and rec.last_epoch == 30
    assert rec.baseline == "linear_interp" and rec.gain_vs_kept == pytest.approx(20 / 520)
    assert len(summary) == 4
    o = overall.set_index("model")
    # per family: recovery positions 1, .5, 0 and anomaly 0, .5, 1 -> each family counts once
    assert o.loc["M-ep0010", "position"] == pytest.approx(0.5)
    assert o.loc["M-ep0030", "position"] == pytest.approx(0.5)
    assert o.loc["M-ep0010", "best_on"] == 3 and o.loc["M-ep0010", "tasks"] == 4
    assert o.loc["M-ep0020", "kept"] and o.loc["M-ep0020", "val_loss_position"] == 0.0
    best = S.best_overall(overall, "M")
    assert best.epoch == 10                                    # tie on position: the earlier epoch
    md = S.markdown(summary, overall, snaps, "val", {"M": {"type": "trajgpt", "adapter": {"device": "cuda"}}})
    assert "Best overall: **epoch 10**" in md and "validation split" in md
    assert "checkpoint: /r/snapshots/M/epoch_0010.pt" in md and "type: trajgpt" in md
    assert "optimistic" not in md
    assert "optimistic" in S.markdown(summary, overall, snaps, "test")


def test_resume_must_keep_the_split(tmp_path):
    S.check_resume(tmp_path, {"split": "val", "tasks": []}, resuming=False)
    S.check_resume(tmp_path, {"split": "val", "tasks": []}, resuming=True)
    with pytest.raises(SystemExit, match="validation|val split"):
        S.check_resume(tmp_path, {"split": "test", "tasks": []}, resuming=True)
    other = tmp_path / "evaluate_run"
    other.mkdir()
    (other / "results.jsonl").write_text("")
    with pytest.raises(SystemExit, match="not a snapshot run"):
        S.check_resume(other, {"split": "val", "tasks": []}, resuming=True)


# ------------------------------------------------------------------ the generation task follows the scored split
def test_generation_reference_comes_from_the_scored_split(monkeypatch):
    pytest.importorskip("torch")
    from mobeval import EvalConfig, synthetic_dataset
    from mobeval import tasks as T
    from mobeval.context import EvalContext
    from mobeval.registry import load_model
    ctx = EvalContext(synthetic_dataset(n_users=16, n_days=5, seed=2),
                      EvalConfig(window_length=24, visit_context=4, max_eval_samples=60, eval_seeds=(0,),
                                 generation_max_trajectories=10))
    seen = []
    orig = T.G.sample_trajectories

    def spy(points, *a, **kw):
        seen.append(set(points.traj_id.unique()))
        return orig(points, *a, **kw)

    monkeypatch.setattr(T.G, "sample_trajectories", spy)
    view = ctx.split_view("val")
    ad = load_model({"name": "KinematicRef", "type": "kinematic_ref"}, view)
    T.GenerationTask().run(ad, view)
    assert seen and seen[0] <= set(ctx.splits["val"].points.traj_id.unique())
    assert not seen[0] & set(ctx.splits["test"].points.traj_id.unique())


# ------------------------------------------------------------------ end to end: train, sweep, resume, figures
@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    pytest.importorskip("torch")
    from mobeval.cli import main
    root = tmp_path_factory.mktemp("sweep")
    cfg = {"output_dir": str(root / "out"), "title": "snapshot sweep test",
           "dataset": {"loader": "synthetic", "n_users": 20, "n_days": 8, "seed": 0},
           "eval": {"window_length": 32, "eval_seeds": [0], "n_boot": 50, "max_eval_samples": 150,
                    "visit_context": 4, "recovery_ratios": [0.5], "recovery_kinds": ["block"]},
           "models": [{"name": "TrajGPT", "type": "trajgpt", "arch": {"num_layers": 1},
                       "train": {"epochs": 4, "batch_size": 32, "save_every": 1, "device": "cpu",
                                 "options": {"tokenizer": {"cell_m": 1000}}}},
                      {"name": "KinematicRef", "type": "kinematic_ref"}]}
    path = root / "cfg.json"
    path.write_text(json.dumps(cfg))
    assert main(["train", "--config", str(path)]) in (0, None)
    return root, path


def _runs(root):
    return sorted((root / "out" / "runs").glob("*-snapshots"))


def test_sweep_end_to_end(trained):
    from mobeval.cli import main
    root, path = trained
    assert len(list((root / "out" / "snapshots" / "TrajGPT").glob("epoch_*.pt"))) == 4
    main(["snapshots", "--config", str(path), "--models", "TrajGPT", "--max-snapshots", "3",
          "--tasks", "next_location", "continuous"])
    run = _runs(root)[-1]
    snaps = pd.read_csv(run / "snapshots.csv")
    assert snaps[snaps.kind.eq("snapshot")].model.tolist() == ["TrajGPT-ep0001", "TrajGPT-ep0003", "TrajGPT-ep0004"]
    assert snaps.kept.sum() == 1                                  # the kept weights, scored once
    lb = pd.read_csv(run / "leaderboard.csv")
    assert set(snaps.model) <= set(lb.model)
    assert not lb.family.eq("efficiency").any()                  # the same for every snapshot: left out
    for f in ("best_snapshots.csv", "snapshot_ranking.csv", "training_history.csv", "report.md"):
        assert (run / f).exists(), f
    assert json.loads((run / "run_info.json").read_text())["eval_split"] == "val"
    assert (run / "report.md").read_text().startswith("## Which epoch is best")
    hist = pd.read_csv(run / "training_history.csv")
    assert hist.epoch.tolist() == [1, 2, 3, 4]

    # resuming keeps the split, and refuses to switch it
    with pytest.raises(SystemExit, match="split"):
        main(["snapshots", "--config", str(path), "--models", "TrajGPT", "--max-snapshots", "3",
              "--tasks", "next_location", "continuous", "--split", "test", "--resume", "latest"])
    n = len((run / "results.jsonl").read_text().splitlines())
    main(["snapshots", "--config", str(path), "--models", "TrajGPT", "--max-snapshots", "3",
          "--tasks", "next_location", "continuous", "--resume", "latest"])
    assert _runs(root)[-1] == run and len((run / "results.jsonl").read_text().splitlines()) == n

    # on the test split the same weights give other numbers
    main(["snapshots", "--config", str(path), "--models", "TrajGPT", "--max-snapshots", "3",
          "--tasks", "next_location", "continuous", "--split", "test"])
    test_run = _runs(root)[-1]
    assert test_run != run
    assert json.loads((test_run / "run_info.json").read_text())["eval_split"] == "test"
    a = lb.set_index(["task", "metric", "protocol", "model"]).value
    b = pd.read_csv(test_run / "leaderboard.csv").set_index(["task", "metric", "protocol", "model"]).value
    common = a.index.intersection(b.index)
    assert len(common) and not np.allclose(a.loc[common], b.loc[common])
    assert "optimistic" in (test_run / "report.md").read_text()

    # the figures
    pytest.importorskip("matplotlib")
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import plot_snapshots
    out = root / "figs"
    assert plot_snapshots.main([str(run), "--out", str(out)]) == 0
    assert {p.name for p in out.iterdir()} >= {"40_snapshots_TrajGPT.png", "41_snapshot_heatmap.png",
                                               "42_snapshot_ranking.png", "best_snapshots.csv"}


def test_a_model_without_snapshots_is_reported_before_loading_data(trained, tmp_path):
    from mobeval.cli import main
    root, path = trained
    cfg = json.loads(path.read_text())
    cfg["models"].append({"name": "CLIP", "type": "clip_mobility", "train": {"epochs": 1}})
    p2 = tmp_path / "cfg.json"
    p2.write_text(json.dumps(cfg))
    with pytest.raises(SystemExit, match="no snapshots for \\['CLIP'\\]"):
        main(["snapshots", "--config", str(p2), "--models", "CLIP"])
