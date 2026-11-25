"""The training log: per-epoch losses (total and per term), the training loss measured like the
validation loss, learning rate and gradient norm, written to <checkpoint>.losses.csv every epoch."""
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from torch import nn

from mobeval import EvalConfig, synthetic_dataset
from mobeval.context import EvalContext
from mobeval.nn.common import (TrainConfig, evaluating, fit, load_checkpoint, loss_log_path, report_loss_parts,
                               resuming, save_checkpoint)

FAST = {"epochs": 2, "batch_size": 32, "device": "cpu", "lr": 1e-3}


def _problem(seed=0, noisy_eval=False, jitter=True):
    """A small regression with dropout, a private training stream (as the adapters' mask
    generators) and, optionally, global random draws even in eval mode."""
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Dropout(0.2), nn.Linear(16, 1))
    g = torch.Generator().manual_seed(1)
    x = torch.randn(96, 4, generator=g)
    y = x @ torch.tensor([1.0, -2.0, 0.5, 0.0]) + 0.1 * torch.randn(96, generator=g)
    private = np.random.default_rng(7)

    def loss_fn(idx, training):
        i = torch.as_tensor(np.asarray(idx) + (0 if training else 64))
        stochastic = training and not evaluating() and jitter
        noise = private.normal(0, 0.05, size=(len(i), 4)) if stochastic else np.zeros((len(i), 4))
        if noisy_eval:                                   # third-party code drawing from the global streams
            torch.rand(1), np.random.rand()
        err = model(x[i] + torch.as_tensor(noise, dtype=torch.float32)).squeeze(-1) - y[i]
        mse, mae = (err ** 2).mean(), err.abs().mean()
        report_loss_parts(mse=mse, mae=mae, empty=torch.zeros(0))
        return mse

    return model, loss_fn


def _cfg(**kw):
    return TrainConfig(**{"epochs": 5, "batch_size": 16, "lr": 0.02, "device": "cpu", "scheduler": "cosine",
                          "patience": 100, **kw})


def test_every_epoch_records_losses_terms_lr_and_gradient_norm(tmp_path):
    model, loss_fn = _problem()
    out = tmp_path / "m.pt"
    h = fit(model, 64, 32, loss_fn, _cfg(train_eval_samples=32), drop_last=False, log_to=out)
    assert len(h) == 5
    for e in h:
        for k in ("train_loss", "train_eval_loss", "val_loss", "best_val_loss", "improved", "lr", "grad_norm",
                  "grad_norm_max", "train_loss:mse", "train_loss:mae", "train_eval_loss:mse", "val_loss:mse"):
            assert k in e and np.isfinite(float(e[k])), k
        assert not any("empty" in k for k in e)                     # empty terms are skipped
        assert e["val_loss:mse"] == pytest.approx(e["val_loss"], rel=1e-6)   # the loss IS the mse term
        assert e["train_eval_loss:mse"] == pytest.approx(e["train_eval_loss"], rel=1e-6)
        assert e["grad_norm_max"] >= e["grad_norm"] > 0
    assert h[0]["lr"] == pytest.approx(0.02) and h[-1]["lr"] < h[0]["lr"]     # this epoch's rate, cosine
    assert [e["improved"] for e in h] == [e["val_loss"] <= min(x["val_loss"] for x in h[:i + 1])
                                          for i, e in enumerate(h)]

    csv = loss_log_path(out)
    assert csv.name == "m.losses.csv"
    t = pd.read_csv(csv)
    assert list(t.columns[:10]) == ["epoch", "train_loss", "train_eval_loss", "val_loss", "best_val_loss",
                                    "improved", "lr", "grad_norm", "grad_norm_max", "seconds"]
    assert list(t.columns[10:13]) == ["train_loss:mse", "train_eval_loss:mse", "val_loss:mse"]
    assert t.epoch.tolist() == [1, 2, 3, 4, 5]
    assert np.allclose(t.val_loss, [e["val_loss"] for e in h], rtol=1e-5)


def test_the_train_eval_pass_leaves_the_training_unchanged(tmp_path):
    """Same weights and losses with and without it: it draws nothing from the data order, the
    loss function's own training stream (evaluating()), or the global random streams."""
    runs = []
    for n in (0, 48):
        model, loss_fn = _problem(noisy_eval=True)
        h = fit(model, 64, 32, loss_fn, _cfg(train_eval_samples=n), drop_last=False, log_to=tmp_path / f"m{n}.pt")
        runs.append((h, {k: v.clone() for k, v in model.state_dict().items()}))
    (h0, w0), (h1, w1) = runs
    assert all(torch.equal(w0[k], w1[k]) for k in w0)
    assert [(e["train_loss"], e["val_loss"]) for e in h0] == [(e["train_loss"], e["val_loss"]) for e in h1]
    assert "train_eval_loss" not in h0[0] and "train_eval_loss" in h1[0]


def test_train_eval_batches_are_fixed_whole_and_built_like_validation_batches():
    from mobeval.nn.common import _train_eval_batches
    b = _train_eval_batches(1000, 64, 150, seed=0)
    assert len(b) == 3 and all(len(x) == 64 and np.all(np.diff(x) == 1) for x in b)   # consecutive, as val
    assert [x.tolist() for x in b] == [x.tolist() for x in _train_eval_batches(1000, 64, 150, seed=0)]
    assert len({int(x[0]) // 64 for x in b}) == 3
    assert [len(x) for x in _train_eval_batches(10, 64, 150, seed=0)] == [10]
    assert _train_eval_batches(1000, 64, 0, seed=0) == []


def test_no_log_and_no_extra_pass_without_log_to(tmp_path):
    model, loss_fn = _problem()
    calls = {"n": 0}

    def counted(idx, training):
        calls["n"] += 1
        return loss_fn(idx, training)

    h = fit(model, 64, 32, counted, _cfg(epochs=2, train_eval_samples=64), drop_last=False)
    assert calls["n"] == 2 * (4 + 2)                         # 4 training + 2 validation batches per epoch
    assert "train_eval_loss" not in h[0] and "val_loss:mse" in h[0]
    assert not list(tmp_path.iterdir())


def test_the_log_keeps_every_epoch_even_when_the_checkpoint_does_not(tmp_path):
    """The checkpoint's .json is rewritten only when validation improves; the CSV every epoch, so
    a job killed after a run of bad epochs still shows them."""
    model, loss_fn = _problem()
    out, epoch = tmp_path / "m.pt", {"n": 0}

    def worse_after_two(idx, training):                      # validation improves twice, then never
        loss = loss_fn(idx, training)
        return loss if training or epoch["n"] < 2 else loss.detach() * 0 + 10.0 + epoch["n"]

    def bump(h):
        epoch["n"] += 1

    def killed_in_epoch_5(h):
        if len(h) == 4:
            raise KeyboardInterrupt

    def on_epoch(h):
        bump(h)
        killed_in_epoch_5(h)

    with pytest.raises(KeyboardInterrupt):
        fit(model, 64, 32, worse_after_two, _cfg(epochs=8), drop_last=False, log_to=out, on_epoch=on_epoch,
            on_best=lambda h: save_checkpoint(out, model, "test", {}, history=h, quiet=True, complete=False))
    assert json.loads(out.with_suffix(".json").read_text())["history"][-1]["epoch"] == 2
    t = pd.read_csv(loss_log_path(out))
    assert t.epoch.tolist() == [1, 2, 3]                     # epoch 4 died in on_epoch, before its row
    assert t.improved.tolist() == [1, 1, 0]


def test_a_resumed_recorded_training_is_still_exact(tmp_path):
    """With the log and the train-eval pass on, a run killed in epoch 4 and resumed ends with the
    uninterrupted run's weights and losses, and the log holds all six epochs. (Without a private
    training stream: an adapter's own generator restarts with the job, as it always has.)"""
    cfg = _cfg(epochs=6, train_eval_samples=32)
    model, loss_fn = _problem(jitter=False)
    clean = fit(model, 64, 32, loss_fn, cfg, drop_last=False, log_to=tmp_path / "clean.pt")
    clean_w = {k: v.clone() for k, v in model.state_dict().items()}

    model, loss_fn = _problem(jitter=False)
    out, calls = tmp_path / "m.pt", {"n": 0}

    def killed_in_epoch_4(idx, training):
        calls["n"] += training and not evaluating()
        if calls["n"] > 12:                                  # 4 training steps per epoch
            raise RuntimeError("walltime")
        return loss_fn(idx, training)

    with pytest.raises(RuntimeError, match="walltime"):
        fit(model, 64, 32, killed_in_epoch_4, cfg, drop_last=False, log_to=out,
            on_best=lambda h: save_checkpoint(out, model, "test", {}, history=h, quiet=True, complete=False))
    model, loss_fn = _problem(seed=99, jitter=False)
    model.load_state_dict(load_checkpoint(out)["state_dict"])
    with resuming(out):
        resumed = fit(model, 64, 32, loss_fn, cfg, drop_last=False, log_to=out)
    key = lambda h: [(e["epoch"], e["train_loss"], e["val_loss"], e["train_eval_loss"]) for e in h]
    assert key(resumed) == key(clean)
    assert all(torch.equal(clean_w[k], v) for k, v in model.state_dict().items())
    assert pd.read_csv(loss_log_path(out)).epoch.tolist() == [1, 2, 3, 4, 5, 6]


def test_a_failing_log_never_stops_the_training(tmp_path, monkeypatch, caplog):
    import mobeval.nn.common as C
    monkeypatch.setattr(C, "write_loss_log", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    model, loss_fn = _problem()
    h = fit(model, 64, 32, loss_fn, _cfg(epochs=3), drop_last=False, log_to=tmp_path / "m.pt")
    assert len(h) == 3
    assert sum("could not write the loss log" in r.message for r in caplog.records) == 1


# ------------------------------------------------------------------ every model's training writes the log
@pytest.fixture(scope="module")
def ctx():
    ds = synthetic_dataset(n_users=14, n_days=6, seed=3)
    return EvalContext(ds, EvalConfig(window_length=24, visit_context=4, max_eval_samples=60))


def _train(kind, ctx, out, **train):
    from mobeval.nn.common import set_seed
    set_seed(0)                                              # the networks' initial weights
    tr = {**FAST, **train}
    if kind == "unitraj":
        from mobeval.adapters.unitraj import UniTrajAdapter
        return UniTrajAdapter.pretrain(ctx, train=tr, out=str(out), arch=dict(trajectory_length=32, embedding_dim=32,
                                                                                encoder_layers=1, decoder_layers=1))
    if kind == "unitraj-original":
        from mobeval.adapters.unitraj import UniTrajAdapter
        return UniTrajAdapter.pretrain(ctx, train=tr, out=str(out), sampling="original", min_points=8,
                                       arch=dict(trajectory_length=32, embedding_dim=32, encoder_layers=1,
                                                 decoder_layers=1))
    if kind == "trajgpt":
        from mobeval.adapters.trajgpt import TrajGPTAdapter
        return TrajGPTAdapter.train(ctx, train=tr, out=str(out), arch={"num_layers": 1})
    if kind == "trajgpt-original":
        from mobeval.adapters.trajgpt import TrajGPTAdapter
        return TrajGPTAdapter.train(ctx, train=tr, out=str(out), arch={"num_layers": 1}, sequences="original",
                                    seq_len=6, instance_stride=3)
    if kind == "clip":
        from mobeval.adapters.clip_mobility import CLIPMobilityAdapter
        return CLIPMobilityAdapter.pretrain(ctx, train=tr, out=str(out), arch=dict(d_model=32, nhead=2, num_layers=1,
                                                                                   dim_feedforward=64, embedding_dim=16))
    if kind == "transfertraj":
        from mobeval.adapters.transfertraj import TransferTrajAdapter
        return TransferTrajAdapter.pretrain(ctx, train=tr, out=str(out), batch_size=32,
                                            arch=dict(embed_size=16, d_model=32, rafee_layer=1))
    raise ValueError(kind)


TERMS = {"unitraj": [], "unitraj-original": [], "trajgpt": ["location", "travel_time", "duration"],
         "trajgpt-original": ["location", "travel_time", "duration"], "clip": ["prediction", "alignment"],
         "transfertraj": ["spatial", "temporal", "token"]}


@pytest.mark.parametrize("kind", list(TERMS))
def test_each_model_logs_its_terms_and_trains_the_same_with_or_without_the_extra_pass(kind, ctx, tmp_path):
    a = _train(kind, ctx, tmp_path / "a.pt", train_eval_samples=64)
    t = pd.read_csv(tmp_path / "a.losses.csv")
    assert t.epoch.tolist() == [1, 2]
    assert t[["train_loss", "train_eval_loss", "val_loss", "lr", "grad_norm"]].notna().all().all()
    for term in TERMS[kind]:
        assert {f"train_loss:{term}", f"train_eval_loss:{term}", f"val_loss:{term}"} <= set(t.columns), term
    if TERMS[kind]:                                          # the terms (weights 1 here) add up to the loss
        parts = t[[f"val_loss:{p}" for p in TERMS[kind]]].sum(axis=1)
        assert np.allclose(parts, t.val_loss, rtol=1e-4)
    side = json.loads((tmp_path / "a.json").read_text())
    assert "train_eval_loss" in side["history"][-1]          # the checkpoint's history carries it too

    b = _train(kind, ctx, tmp_path / "b.pt", train_eval_samples=0)
    wa, wb = a.net.state_dict(), b.net.state_dict()
    assert all(torch.equal(wa[k], wb[k]) for k in wa), "the extra pass changed the training"
    assert "train_eval_loss" not in pd.read_csv(tmp_path / "b.losses.csv").columns


def test_omnitraj_logs_each_contrasted_pair(tmp_path):
    pytest.importorskip("timm")
    pytest.importorskip("transformers")
    from mobeval.adapters.omnitraj import OmniTrajAdapter
    ds = synthetic_dataset(n_users=8, n_days=5, seed=1)
    p = ds.points
    roads = pd.DataFrame({"traj_id": p.traj_id, "t": p.t,
                          "seg": (np.floor(p.lat * 300) * 1000 + np.floor(p.lon * 300)).astype(np.int64) % 100000})
    roads.to_csv(tmp_path / "roads.csv", index=False)
    ctx = EvalContext(ds, EvalConfig(window_length=24, visit_context=4, max_eval_samples=40))
    tiny = {"trajectory": {"embed_dim": 32, "num_heads": 2, "depth": 1, "output_dim": 32},
            "topology": {"embed_dim": 32, "output_dim": 32, "num_layers": 1, "num_heads": 2},
            "road": {"embed_dim": 32, "output_dim": 32, "num_layers": 1, "num_heads": 2},
            "region": {"output_dim": 32, "embed_dim": 32, "num_layers": 1, "num_heads": 2}}
    from mobeval.nn.common import set_seed
    nets = []
    for n in (32, 0):
        set_seed(0)                                          # the networks' initial weights
        ad = OmniTrajAdapter.pretrain(ctx, train={"epochs": 2, "batch_size": 16, "device": "cpu",
                                                  "train_eval_samples": n},
                                      out=str(tmp_path / f"o{n}.pt"), arch=tiny, projection_dim=16, min_points=10,
                                      roads_file=str(tmp_path / "roads.csv"), prep_workers=1)
        nets.append(ad.net.state_dict())
    t = pd.read_csv(tmp_path / "o32.losses.csv")
    pairs = [c.split(":", 1)[1] for c in t.columns if c.startswith("val_loss:")]
    assert "trajectory-topology" in pairs and any(p.startswith("trajectory-topology_") for p in pairs)
    assert np.allclose(t[[f"val_loss:{p}" for p in pairs]].mean(axis=1), t.val_loss, rtol=1e-4)   # loss = mean of pairs
    assert t.train_eval_loss.notna().all() and "batch_seconds" in t.columns
    assert all(torch.equal(nets[0][k], nets[1][k]) for k in nets[0]), "the extra pass changed the training"


# ------------------------------------------------------------------ report and figures
def test_report_shows_the_gap_at_the_best_epoch_and_reads_old_histories():
    from mobeval.report import fit_section, training_curves
    new = [{"epoch": 1, "train_loss": 3.0, "train_eval_loss": 2.0, "val_loss": 2.5},
           {"epoch": 2, "train_loss": 2.0, "train_eval_loss": 1.0, "val_loss": 1.5},
           {"epoch": 3, "train_loss": 1.0, "train_eval_loss": 0.5, "val_loss": 1.8}]
    old = [{k: e[k] for k in ("epoch", "train_loss", "val_loss")} for e in new]
    c = training_curves({"New": new, "Old": old}).set_index("model")
    assert c.loc["New", "best_epoch"] == 2 and c.loc["New", "gap_at_best"] == pytest.approx(0.5)
    assert np.isnan(c.loc["Old", "gap_at_best"])
    md = fit_section(None, c.reset_index())
    assert "gap_at_best" in md and "+50%" in md


def test_plot_fit_prefers_the_log_with_more_epochs_and_draws_the_figures(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    import plot_fit
    model, loss_fn = _problem()
    ck = tmp_path / "checkpoints"
    out = ck / "TrajGPT.pt"
    fit(model, 64, 32, loss_fn, _cfg(epochs=4, train_eval_samples=32), drop_last=False, log_to=out,
        on_best=lambda h: len(h) <= 2 and save_checkpoint(out, model, "test", {}, history=h, quiet=True,
                                                          complete=False))
    hist = plot_fit.load_histories([ck])
    assert len(hist) == 1 and hist[0][0] == "TrajGPT" and len(hist[0][1]) == 4 and hist[0][2] is False
    figs = tmp_path / "figs"
    figs.mkdir()
    plot_fit.style()
    plot_fit.fig_curves(hist, figs, ["png"])
    assert plot_fit.fig_terms(hist, figs, ["png"]) and plot_fit.fig_optimisation(hist, figs, ["png"])
    assert {p.name for p in figs.iterdir()} == {"22_training_curves.png", "23_loss_terms.png",
                                                "24_learning_rate_and_gradients.png"}


# ------------------------------------------------------------------ periodic snapshots (train.save_every)
@pytest.mark.parametrize("kind", ["trajgpt", "clip"])
def test_save_every_keeps_each_nth_epoch_beside_the_checkpoints(kind, ctx, tmp_path):
    from mobeval.registry import load_model
    ck = tmp_path / "checkpoints"
    ad = _train(kind, ctx, ck / "M.pt", epochs=5, save_every=2)
    snaps = tmp_path / "snapshots" / "M"
    assert sorted(p.name for p in snaps.glob("*.pt")) == ["epoch_0002.pt", "epoch_0004.pt"]
    assert not (ck / "snapshots").exists()                  # jobs never copy them back to scratch
    side = json.loads((snaps / "epoch_0004.json").read_text())
    assert side["history"][-1]["epoch"] == 4                 # the weights of epoch 4, whatever was best
    spec = {"name": "M-epoch4", "type": "trajgpt" if kind == "trajgpt" else "clip_mobility",
            "checkpoint": str(snaps / "epoch_0004.pt"), "adapter": {"device": "cpu"}}
    snap = load_model(spec, ctx, str(tmp_path), check_provenance="error")
    assert snap.name == "M-epoch4"                          # loads, and passes the split check


def test_no_snapshots_by_default(ctx, tmp_path):
    _train("trajgpt", ctx, tmp_path / "checkpoints" / "M.pt", epochs=3)
    assert not (tmp_path / "snapshots").exists()


def test_a_failing_snapshot_never_stops_the_training(tmp_path, caplog):
    model, loss_fn = _problem()

    def saver(h):
        pass

    def broken(h, epoch):
        raise OSError("disk full")

    saver.snapshot = broken
    h = fit(model, 64, 32, loss_fn, _cfg(epochs=4, save_every=2), drop_last=False, on_best=saver)
    assert len(h) == 4
    assert sum("could not save the epoch" in r.message and "snapshot" in r.message for r in caplog.records) == 2
