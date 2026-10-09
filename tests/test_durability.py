"""What must survive a crash, a walltime kill, or two jobs sharing an output directory.

The failure these guard against is not a wrong number, it is losing hours of finished work
because the process died before the single write at the end of the job.
"""
import json

import numpy as np
import pytest
import torch
import torch.nn as nn

from mobeval import EvalConfig, EvaluationPipeline, synthetic_dataset
from mobeval import layout as layout_mod
from mobeval.adapters.reference import KinematicReference, WeakReference
from mobeval.nn.common import TrainConfig, fit, load_checkpoint, save_checkpoint
from mobeval.results import ResultRecord, ResultStore


@pytest.fixture(autouse=True)
def _no_layout():
    """Tests must not inherit a layout from each other; mirroring is process-wide."""
    yield
    layout_mod.activate(None)


# --------------------------------------------------------------------- checkpoints
def _tiny_problem(seed=0):
    """A model whose validation loss improves for a few epochs, then a loss that explodes."""
    torch.manual_seed(seed)
    model = nn.Linear(4, 1)
    x = torch.randn(64, 4)
    y = x @ torch.tensor([1.0, -2.0, 0.5, 0.0]) + 0.1 * torch.randn(64)

    def loss_fn(idx, training):
        i = torch.as_tensor(np.asarray(idx) % 64)
        return ((model(x[i]).squeeze(-1) - y[i]) ** 2).mean()

    return model, loss_fn


def test_checkpoint_is_written_on_every_improving_epoch(tmp_path):
    model, loss_fn = _tiny_problem()
    out = tmp_path / "ck" / "m.pt"
    seen = []

    def on_best(history):
        seen.append(history[-1]["epoch"])
        save_checkpoint(out, model, "test", {}, history=history, quiet=True)
        # the file must be complete and loadable the instant it is written, mid-training
        assert load_checkpoint(out)["history"][-1]["epoch"] == history[-1]["epoch"]

    fit(model, 64, 32, loss_fn, TrainConfig(epochs=6, batch_size=16, lr=0.05, device="cpu"),
        drop_last=False, on_best=on_best)
    assert len(seen) >= 2, "expected several improving epochs to have been saved"
    assert out.exists()


def test_a_crash_mid_training_leaves_the_best_epoch_so_far_on_disk(tmp_path):
    """The whole point: dying at epoch 4 of 50 must not throw away epochs 1-3."""
    model, loss_fn = _tiny_problem()
    out = tmp_path / "ck" / "m.pt"
    calls = {"n": 0}

    def exploding(idx, training):
        calls["n"] += 1
        if calls["n"] > 12:
            raise RuntimeError("simulated crash (OOM, preempted node, ...)")
        return loss_fn(idx, training)

    def on_best(history):
        save_checkpoint(out, model, "test", {}, history=history, quiet=True)

    with pytest.raises(RuntimeError, match="simulated crash"):
        fit(model, 64, 32, exploding, TrainConfig(epochs=50, batch_size=16, lr=0.05, device="cpu"),
            drop_last=False, on_best=on_best)
    ck = load_checkpoint(out)
    assert ck["history"], "a checkpoint from before the crash must survive"
    assert set(ck["state_dict"]) == set(model.state_dict())


def test_checkpoint_write_is_atomic(tmp_path):
    """A half-written save must not be able to destroy the previous good checkpoint."""
    model, _ = _tiny_problem()
    out = tmp_path / "m.pt"
    save_checkpoint(out, model, "test", {"v": 1}, history=[{"epoch": 1}], quiet=True)
    good = out.read_bytes()

    real_save = torch.save

    def failing_save(obj, path, *a, **k):
        real_save(obj, path, *a, **k)          # writes the temporary file...
        raise OSError("disk full")             # ...then dies before the rename

    torch.save = failing_save
    try:
        with pytest.raises(OSError):
            save_checkpoint(out, model, "test", {"v": 2}, history=[{"epoch": 2}], quiet=True)
    finally:
        torch.save = real_save
    assert out.read_bytes() == good, "the previous checkpoint must be untouched"
    assert load_checkpoint(out)["config"] == {"v": 1}


# --------------------------------------------------------------------- run layout
def test_runs_get_separate_directories_and_share_checkpoints(tmp_path):
    a = layout_mod.Layout(tmp_path / "out", run_id="run-a").create()
    b = layout_mod.Layout(tmp_path / "out", run_id="run-b").create()
    assert a.run_dir != b.run_dir and a.run_dir.is_dir() and b.run_dir.is_dir()
    assert a.checkpoint_dir == b.checkpoint_dir, "training must be reusable across runs"
    latest = tmp_path / "out" / "latest"
    assert latest.is_dir() and latest.resolve() == b.run_dir.resolve()


def test_persist_dir_keeps_work_and_restores_checkpoints(tmp_path):
    scratch, home = tmp_path / "scratch", tmp_path / "home"
    lay = layout_mod.Layout(scratch, run_id="run-1", persist_dir=home).create()
    ck = lay.checkpoint("UniTraj")
    ck.write_text("weights")
    assert lay.mirror(ck) == home / "checkpoints" / "UniTraj.pt"
    res = lay.run_dir / "results.jsonl"
    res.write_text('{"a": 1}\n')
    assert lay.mirror(res).read_text() == '{"a": 1}\n'

    # scratch is wiped; a later job on a different node restores the weights from home
    import shutil
    shutil.rmtree(scratch)
    later = layout_mod.Layout(scratch, run_id="run-2", persist_dir=home).create()
    assert later.restore_checkpoints() == ["UniTraj"]
    assert later.checkpoint("UniTraj").read_text() == "weights"


def test_mirroring_never_raises_when_the_destination_is_unusable(tmp_path):
    """A full or misconfigured persist directory must not take down a training run."""
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    (blocked / "runs").write_text("not a directory")      # mkdir below will fail
    lay = layout_mod.Layout(tmp_path / "out", run_id="r", persist_dir=blocked).create()
    f = lay.run_dir / "results.jsonl"
    f.write_text("x")
    assert lay.mirror(f) is None                          # warns, does not raise


# --------------------------------------------------------------------- results
def _rec(model, task_unit, task, value=1.0):
    return ResultRecord(model=model, run_tag="default", task=task, metric="ade_m", value=value,
                        n=10, task_unit=task_unit)


def test_results_are_on_disk_before_the_run_finishes(tmp_path):
    store = ResultStore(tmp_path / "results.jsonl")
    store.extend([_rec("A", "recovery", "recovery/block@0.5")])
    store.persist()
    assert len(ResultStore.load(tmp_path / "results.jsonl").records) == 1
    store.extend([_rec("A", "generation", "generation/jump_length")])
    store.persist()
    assert len(ResultStore.load(tmp_path / "results.jsonl").records) == 2


def test_a_truncated_line_costs_one_record_not_the_file(tmp_path):
    p = tmp_path / "results.jsonl"
    store = ResultStore(p)
    store.extend([_rec("A", "recovery", "recovery/block@0.5"), _rec("A", "generation", "generation/jump")])
    store.persist()
    with open(p, "a") as f:                      # the job was killed part-way through a line
        f.write('{"model": "A", "run_tag": "def')
    assert len(ResultStore.load(p).records) == 2


def test_resume_keeps_finished_tasks_and_recomputes_nothing(tmp_path):
    ds = synthetic_dataset(n_users=10, n_days=5, seed=3)
    cfg = EvalConfig(window_length=24, recovery_ratios=(0.5,), recovery_kinds=("block",), eval_seeds=(0,),
                     n_boot=20, max_eval_samples=80, visit_context=4, label_fractions=(1.0,),
                     generation_max_trajectories=20)
    pipe = EvaluationPipeline(cfg)
    ctx = pipe.prepare(ds)
    path = tmp_path / "results.jsonl"

    first = pipe.run([KinematicReference(), WeakReference()], ctx, ResultStore(path))
    assert first.records and path.exists()
    units = first.done_tasks()
    assert units, "records must carry the task unit a resumed run checks off"

    # a second attempt resuming from that file must reuse everything and recompute nothing
    ran = []
    for t in pipe.tasks:
        original = t.run
        t.run = (lambda *a, _o=original, _n=t.name, **k: (ran.append(_n), _o(*a, **k))[1])
    resumed = pipe.run([KinematicReference(), WeakReference()], ctx, ResultStore(path, resume=True))
    assert not ran, f"resumed run recomputed {ran}"
    assert len(resumed.records) == len(first.records)


def test_an_interrupted_run_resumed_gives_exactly_the_same_records_as_one_clean_run(tmp_path):
    """The point of resuming: identical output, not just "most of it".

    Baseline rows are emitted once per task by whichever adapter reaches them first, tracked in
    memory. A resumed process starts with that tracking empty and must not write a second copy -
    including the generation task's baselines, which use a different key shape.
    """
    def fresh_ctx():
        ds = synthetic_dataset(n_users=10, n_days=5, seed=7)
        cfg = EvalConfig(window_length=24, recovery_ratios=(0.5,), recovery_kinds=("block",),
                         eval_seeds=(0,), n_boot=20, max_eval_samples=80, visit_context=4,
                         label_fractions=(1.0,), generation_max_trajectories=20)
        pipe = EvaluationPipeline(cfg)
        return pipe, pipe.prepare(ds)

    def key(r):
        return (r.model, r.run_tag, r.task, r.metric, r.eval_seed, r.protocol)

    pipe, ctx = fresh_ctx()
    clean = pipe.run([KinematicReference(), WeakReference()], ctx, ResultStore(tmp_path / "clean.jsonl"))

    # a run that stops after the first model, then a fresh process resuming from its file
    partial_path = tmp_path / "partial.jsonl"
    pipe, ctx = fresh_ctx()
    pipe.run([KinematicReference()], ctx, ResultStore(partial_path))
    pipe, ctx = fresh_ctx()                      # fresh context: emitted_baselines starts empty
    resumed = pipe.run([KinematicReference(), WeakReference()], ctx,
                       ResultStore(partial_path, resume=True))

    assert sorted(map(key, resumed.records)) == sorted(map(key, clean.records))
    assert len(resumed.records) == len(clean.records), "resuming duplicated records"
    # the generation baselines are the ones with the odd key shape - check them by name
    gen = [r for r in resumed.records if r.model.startswith("baseline:") and r.task.startswith("generation/")]
    assert gen and len({key(r) for r in gen}) == len(gen), "generation baselines were duplicated"


def test_binding_without_resume_keeps_the_earlier_file_as_previous(tmp_path):
    p = tmp_path / "results.jsonl"
    ResultStore(p).extend([_rec("A", "recovery", "recovery/block@0.5")])
    ResultStore(p).persist()
    s = ResultStore(p)
    s.extend([_rec("B", "recovery", "recovery/block@0.5")])
    s.persist()
    assert (tmp_path / "results.jsonl.previous").exists()
    assert [r.model for r in ResultStore.load(p).records] == ["B"]


# --------------------------------------------------------------------- interrupted training
def test_an_interrupted_model_is_continued_not_accepted_as_final(tmp_path):
    """A checkpoint from a killed job is the best epoch so far, not a trained model."""
    from mobeval.cli import _resume_training
    from mobeval.nn.common import checkpoint_status

    model, loss_fn = _tiny_problem()
    cfg = {"output_dir": str(tmp_path)}
    spec = {"name": "M", "type": "unitraj", "train": {"epochs": 10}}
    ck = tmp_path / "checkpoints" / "M.pt"

    assert _resume_training(spec, cfg, only_missing=True) == (True, None)      # nothing there yet

    save_checkpoint(ck, model, "unitraj", {}, history=[{"epoch": 3}], quiet=True, complete=False)
    assert checkpoint_status(ck) == "partial"
    train_it, continue_from = _resume_training(spec, cfg, only_missing=True)
    assert train_it and continue_from == str(ck), "a half-trained model must be continued"

    save_checkpoint(ck, model, "unitraj", {}, history=[{"epoch": 10}], quiet=True, complete=True)
    assert checkpoint_status(ck) == "complete"
    assert _resume_training(spec, cfg, only_missing=True) == (False, None)     # finished: skip it

    assert _resume_training(spec, cfg, only_missing=False)[0] is True          # plain `train` retrains


def test_checkpoints_from_before_this_field_existed_still_count_as_complete(tmp_path):
    from mobeval.nn.common import checkpoint_status
    model, _ = _tiny_problem()
    ck = tmp_path / "old.pt"
    save_checkpoint(ck, model, "unitraj", {}, quiet=True)
    side = json.loads(ck.with_suffix(".json").read_text())
    del side["complete"]
    ck.with_suffix(".json").write_text(json.dumps(side))
    assert checkpoint_status(ck) == "complete"


def test_run_info_and_report_land_in_the_run_directory(tmp_path):
    from mobeval.cli import main
    out = tmp_path / "out"
    assert main(["smoke", "--out", str(out), "--device", "cpu",
                 "--persist-dir", str(tmp_path / "home")]) == 0
    run = json.loads((out / "latest" / "run_info.json").read_text())
    assert run["run_id"] and (out / "runs" / run["run_id"] / "report.md").exists()
    # everything durable also reached the persist directory, without waiting for a stage-out step
    home = tmp_path / "home"
    assert (home / "checkpoints" / "TrajGPT-tiny.pt").exists()
    assert (home / "runs" / run["run_id"] / "results.jsonl").exists()
    assert (home / "runs" / run["run_id"] / "report.md").exists()


# --------------------------------------------------------------------- config typos
def test_a_misspelled_model_key_is_refused_not_ignored(tmp_path):
    """`chechpoint:` silently fell back to the default path, so a run looked fine while
    evaluating a different file than the config named."""
    from mobeval.config import apply_defaults
    cfg = {"dataset": {}, "models": [{"name": "TrajGPT", "type": "trajgpt",
                                      "chechpoint": "/home/me/TrajGPT.pt"}]}
    with pytest.raises(ValueError, match="did you mean 'checkpoint'"):
        apply_defaults(cfg)
    with pytest.raises(ValueError, match="did you mean 'models'"):
        apply_defaults({"dataset": {}, "modles": [], "models": [{"name": "A", "type": "unitraj"}]})
    # the spelling that was meant still works
    ok = apply_defaults({"dataset": {}, "models": [{"name": "TrajGPT", "type": "trajgpt",
                                                    "checkpoint": "/home/me/TrajGPT.pt"}]})
    assert ok["models"][0]["checkpoint"].endswith("TrajGPT.pt")


def test_generation_survives_a_baseline_with_no_staypoints(tmp_path):
    """Regression for KeyError('_cells'): the uniform-bbox baseline can produce no stays at all
    on data where stays are inferred from trip gaps, which killed the whole generation task."""
    import pandas as pd
    from mobeval.metrics import generative as G
    from mobeval.data import SpatialGrid
    pts = pd.DataFrame({"user_id": [1, 1, 1], "traj_id": [1, 1, 1],
                        "lat": [45.0, 45.01, 45.02], "lon": [9.0, 9.01, 9.02], "t": [0.0, 60.0, 120.0]})
    grid = SpatialGrid(44.9, 45.1, 8.9, 9.1, 500.0)
    empty = G.trajectory_stats(pts, pd.DataFrame(columns=["user_id", "lat", "lon", "t_arrive", "t_leave"]), grid)
    assert "_cells" not in empty, "no staypoints must mean no _cells - the task has to cope with that"


# --------------------------------------------------------------------- resuming the same run
def _dropout_problem(seed=0):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Dropout(0.2), nn.Linear(16, 1))
    g = torch.Generator().manual_seed(1)
    x = torch.randn(64, 4, generator=g)
    y = x @ torch.tensor([1.0, -2.0, 0.5, 0.0]) + 0.1 * torch.randn(64, generator=g)

    def loss_fn(idx, training):
        i = torch.as_tensor(np.asarray(idx) % 64)
        return ((model(x[i]).squeeze(-1) - y[i]) ** 2).mean()

    return model, loss_fn


def _losses(history):
    return [(h["epoch"], h["train_loss"], h["val_loss"]) for h in history]


def test_a_resumed_training_continues_the_run_exactly(tmp_path):
    """Killed after epoch 3 of 6 and resumed: epoch count, cosine schedule, optimizer moments,
    data order and dropout carry on, so the result is the uninterrupted run's, to the bit."""
    from mobeval.nn.common import resuming
    cfg = TrainConfig(epochs=6, batch_size=16, lr=0.02, device="cpu", scheduler="cosine", patience=100)
    model, loss_fn = _dropout_problem()
    clean = fit(model, 64, 32, loss_fn, cfg, drop_last=False)
    clean_w = {k: v.clone() for k, v in model.state_dict().items()}
    assert [h["val_loss"] for h in clean[:3]] == sorted((h["val_loss"] for h in clean[:3]), reverse=True)  # 1-3 improve

    model, loss_fn = _dropout_problem()
    out, calls = tmp_path / "m.pt", {"n": 0}

    def killed_in_epoch_4(idx, training):
        calls["n"] += training
        if calls["n"] > 12:                                     # 4 steps per epoch
            raise RuntimeError("walltime")
        return loss_fn(idx, training)

    with pytest.raises(RuntimeError, match="walltime"):
        fit(model, 64, 32, killed_in_epoch_4, cfg, drop_last=False,
            on_best=lambda h: save_checkpoint(out, model, "test", {}, history=h, quiet=True, complete=False))
    ck = load_checkpoint(out)
    assert ck["history"][-1]["epoch"] == 3 and ck["train_state"]["epoch"] == 3

    model, loss_fn = _dropout_problem(seed=99)                  # fresh process: other initial weights...
    model.load_state_dict(ck["state_dict"])                     # ...replaced by the checkpoint's (init_from)
    with resuming(out):
        resumed = fit(model, 64, 32, loss_fn, cfg, drop_last=False)
    assert _losses(resumed) == _losses(clean)
    assert all(torch.equal(clean_w[k], v) for k, v in model.state_dict().items())


def test_a_checkpoint_without_train_state_still_continues_epochs_and_schedule(tmp_path, monkeypatch):
    """Checkpoints written before the train state was saved (or with save_train_state: false):
    the optimizer starts afresh, but the epoch count and the learning-rate schedule carry on."""
    import mobeval.nn.common as C
    opts, lrs = [], []
    real = C.make_optimizer
    monkeypatch.setattr(C, "make_optimizer", lambda p, cfg: opts.append(real(p, cfg)) or opts[-1])
    cfg = TrainConfig(epochs=6, batch_size=16, lr=0.02, device="cpu", scheduler="cosine", patience=100,
                      save_train_state=False)
    model, loss_fn = _dropout_problem()
    fit(model, 64, 32, loss_fn, cfg, drop_last=False, on_epoch=lambda h: lrs.append(opts[-1].param_groups[0]["lr"]))
    clean_lrs, lrs[:] = list(lrs), []

    model, loss_fn = _dropout_problem()
    out = tmp_path / "m.pt"

    def keep_epoch_3(h):                                        # as if the job had been killed in epoch 4
        if len(h) == 3:
            save_checkpoint(out, model, "test", {}, history=h, quiet=True, complete=False)

    fit(model, 64, 32, loss_fn, cfg, drop_last=False, on_best=keep_epoch_3)
    assert "train_state" not in load_checkpoint(out)
    model.load_state_dict(load_checkpoint(out)["state_dict"])
    lrs[:] = []
    with C.resuming(out):
        resumed = fit(model, 64, 32, loss_fn, cfg, drop_last=False,
                      on_epoch=lambda h: lrs.append(opts[-1].param_groups[0]["lr"]))
    assert [h["epoch"] for h in resumed] == [1, 2, 3, 4, 5, 6]
    assert lrs == clean_lrs[3:]                                 # epochs 4-6 at the schedule's learning rates


def test_train_model_resumes_only_when_continuing_its_own_checkpoint(tmp_path, monkeypatch):
    """`--resume` continues the run; `init_from` alone (fine-tuning) starts a new one."""
    from types import SimpleNamespace
    import mobeval.nn.common as C
    from mobeval import registry
    seen = []

    class Fake:
        @classmethod
        def pretrain(cls, ctx, train=None, out=None, init_from=None, **kw):
            seen.append((init_from, C._RESUME_FROM, "resume" in (train or {})))

    monkeypatch.setitem(registry.MODEL_TYPES, "omnitraj", lambda: Fake)
    ctx = SimpleNamespace(cfg=None)
    ck = str(tmp_path / "checkpoints" / "O.pt")
    registry.train_model({"name": "O", "type": "omnitraj", "train": {"init_from": ck, "resume": True}}, ctx, tmp_path)
    registry.train_model({"name": "O", "type": "omnitraj", "train": {"init_from": ck}}, ctx, tmp_path)
    assert seen == [(ck, ck, False), (ck, None, False)] and C._RESUME_FROM is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="mixed precision is for CUDA GPUs")
def test_mixed_precision_training_on_gpu_learns_like_float32():
    out = {}
    for amp in (False, True):
        model, loss_fn = _dropout_problem()
        model.cuda()
        x, y = torch.randn(64, 4, device="cuda"), torch.randn(64, device="cuda")

        def loss_fn(idx, training):
            i = torch.as_tensor(np.asarray(idx) % 64, device="cuda")
            return ((model(x[i]).squeeze(-1) - y[i]) ** 2).mean()

        torch.manual_seed(0)
        x.copy_(torch.randn(64, 4)); y.copy_(x[:, 0] - 2 * x[:, 1])
        fit(model, 64, 32, loss_fn, TrainConfig(epochs=4, batch_size=16, lr=0.02, device="cuda", amp=amp),
            drop_last=False)
        out[amp] = {k: v.cpu() for k, v in model.state_dict().items()}
    assert all(v.dtype == torch.float32 for v in out[True].values())          # weights stay float32
    assert all(torch.allclose(out[True][k], out[False][k], atol=0.05) for k in out[False])
