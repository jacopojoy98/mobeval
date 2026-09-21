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
