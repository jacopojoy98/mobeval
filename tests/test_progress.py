"""Progress reporting: what a job writes, and what `mobeval status` shows."""
import json
import time
from pathlib import Path

import pytest

from mobeval import progress


def test_reporter_writes_snapshot_and_event_log(tmp_path):
    r = progress.Reporter(tmp_path, "run", heartbeat_s=0)
    r.set_plan(2, "starting")
    with r.stage("prepare", "preparing data", step=True):
        pass
    r.model("UniTraj", state="training", detail="epoch 1/10")
    r.count(records=5)
    r.close("done")
    status = json.loads(Path(r.status_path).read_text())
    assert status["state"] == "done" and status["steps_done"] == 1 and status["steps_total"] == 2
    assert status["models"]["UniTraj"]["detail"] == "epoch 1/10" and status["counters"]["records"] == 5
    events = [json.loads(x) for x in Path(r.events_path).read_text().splitlines()]
    assert [e["event"] for e in events][:2] == ["run_start", "stage_start"]
    assert events[-1]["event"] == "run_end"


def test_snapshot_is_never_partially_written(tmp_path):
    """Readers poll while the job writes; a torn file would crash `mobeval status`."""
    r = progress.Reporter(tmp_path, "run", heartbeat_s=0)
    for i in range(30):
        r.model(f"m{i}", state="training", detail="x" * 500)
        assert json.loads(Path(r.status_path).read_text())["run_id"] == r.run_id
    assert not list(Path(tmp_path).glob("*.tmp*"))
    r.close()


def test_failure_is_recorded_and_stage_reraises(tmp_path):
    r = progress.Reporter(tmp_path, "run", heartbeat_s=0)
    with pytest.raises(ValueError):
        with r.stage("train", "training X"):
            raise ValueError("boom")
    r.error("training X failed", model="X")
    r.close("failed", failure="ValueError('boom')")
    runs = progress.load_runs(tmp_path)
    assert runs[0]["state"] == "failed" and runs[0]["counters"]["errors"] == 1
    text = progress.format_runs(runs)
    assert "FAILED" in text and "training X failed" in text


def test_stale_run_is_flagged(tmp_path):
    r = progress.Reporter(tmp_path, "run", heartbeat_s=0)
    r.set_phase("training")
    state = json.loads(Path(r.status_path).read_text())
    state["updated"] = time.time() - (progress.STALE_AFTER_S + 60)
    Path(r.status_path).write_text(json.dumps(state))
    run = progress.load_runs(tmp_path)[0]
    assert run.get("stale") and run["state"] == "running"
    assert "may have been killed" in progress.format_runs([run])


def test_several_runs_share_a_directory_newest_first(tmp_path):
    ids = []
    for i in range(3):
        r = progress.Reporter(tmp_path, "train", run_id=f"run{i}", heartbeat_s=0)
        r.close()
        ids.append(r.run_id)
        time.sleep(0.02)
    runs = progress.load_runs(tmp_path)
    assert [r["run_id"] for r in runs] == ids[::-1]
    assert len(progress.load_runs(tmp_path, limit=2)) == 2


def test_module_level_reporter_is_a_noop_without_a_directory():
    progress.start(None, "run")
    rep = progress.get()
    assert not rep.active
    rep.set_plan(3); rep.model("x", state="y"); rep.error("z")      # must not raise
    with rep.stage("s"), rep.scoped("m"):
        pass
    progress.stop()


def test_format_helpers():
    assert progress.human_time(45) == "45s" and progress.human_time(3725).startswith("1h")
    assert progress.bar(1, 4, width=4, ascii_only=True) == "[#...] 1/4"
    assert progress.bar(0, 0) == ""
    assert "no runs found" in progress.format_runs([]).lower()
    line = progress.event_line({"ts": time.time(), "event": "epoch", "model": "M", "epoch": 3})
    assert "epoch" in line and "model=M" in line


def test_pipeline_reports_stages_tasks_and_counts(tmp_path):
    """An actual evaluation must fill in phases, per-model state and the step counter."""
    from mobeval import EvalConfig, EvaluationPipeline, synthetic_dataset
    from mobeval.adapters.reference import KinematicReference
    progress.start(tmp_path, "run")
    pipe = EvaluationPipeline(EvalConfig(window_length=24, eval_seeds=(0,), n_boot=20, max_eval_samples=40,
                                         recovery_ratios=(0.5,), recovery_kinds=("block",), visit_context=4,
                                         generation_max_trajectories=10))
    ctx = pipe.prepare(synthetic_dataset(n_users=8, n_days=5, seed=1))
    store = pipe.run([KinematicReference()], ctx)
    progress.stop("done")

    run = progress.load_runs(tmp_path)[0]
    assert run["state"] == "done"
    assert run["steps_total"] > 0 and run["steps_done"] == run["steps_total"]
    assert run["counters"]["records"] == len(store.records)
    assert run["models"]["KinematicRef"]["state"] == "evaluated"
    events = [json.loads(x) for x in Path(run["_events"]).read_text().splitlines()]
    kinds = {e["event"] for e in events}
    assert {"run_start", "stage_start", "prepared", "task_start", "task_end", "run_end"} <= kinds
    assert any(e["event"] == "task_end" and e["task"] == "recovery" for e in events)


def test_config_failure_is_recorded_when_the_directory_is_known_early(tmp_path, monkeypatch):
    """A broken config must leave a visible failed run, not an empty progress directory."""
    from mobeval.cli import main
    d = tmp_path / "progress"
    monkeypatch.setenv("MOBEVAL_PROGRESS_DIR", str(d))
    bad = tmp_path / "bad.yaml"
    bad.write_text("output_dir: /tmp/x\ndataset: {loader: csv, path: /nope.csv}\n"
                   "models: [{name: A, type: unitraj}, {name: A, type: unitraj}]\n")
    with pytest.raises(ValueError, match="unique"):
        main(["train", "--config", str(bad)])
    run = progress.load_runs(d)[0]
    assert run["state"] == "failed" and run["phase"] == "loading config" and "unique" in run["failure"]
