"""Tests for the PyTorch model adapters, training, checkpoints, provenance and CLI."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mobeval import EvalConfig, synthetic_dataset
from mobeval.adapters.base import TargetGuard
from mobeval.context import EvalContext
from mobeval.data import make_mask

FAST = {"epochs": 1, "batch_size": 32, "device": "cpu", "lr": 1e-3}


@pytest.fixture(scope="module")
def ctx():
    ds = synthetic_dataset(n_users=14, n_days=6, seed=3)
    return EvalContext(ds, EvalConfig(window_length=24, visit_context=4, max_eval_samples=60))


# ------------------------------------------------------------------ UniTraj
@pytest.fixture(scope="module")
def unitraj(ctx, tmp_path_factory):
    from mobeval.adapters.unitraj import UniTrajAdapter
    out = tmp_path_factory.mktemp("ck") / "unitraj.pt"
    arch = dict(trajectory_length=32, embedding_dim=32, encoder_layers=1, decoder_layers=1)
    ad = UniTrajAdapter.pretrain(ctx, arch=arch, train=FAST, out=str(out))
    return ad, out


def test_unitraj_recovery_shapes_roundtrip_and_provenance(ctx, unitraj):
    from mobeval.adapters.unitraj import UniTrajAdapter
    ad, path = unitraj
    b = ctx.windows["test"]
    m = make_mask(len(b), b.length, 0.5, "block", 0)
    la, lo = ad.reconstruct(TargetGuard.hide_masked(b, m), m)
    assert la.shape == b.lat.shape and np.isfinite(la[m]).all() and np.isfinite(lo[m]).all()
    ad2 = UniTrajAdapter.from_checkpoint(str(path), device="cpu")
    la2, _ = ad2.reconstruct(TargetGuard.hide_masked(b, m), m)
    assert np.allclose(la, la2)
    assert ad2.provenance["train_fingerprint"] == ctx.fingerprint
    assert ad2.embed(b).shape == (len(b), 32)


def test_unitraj_rejects_windows_longer_than_model(ctx, unitraj):
    from mobeval.data import make_windows
    ad, _ = unitraj
    long = make_windows(ctx.splits["train"], 40, max_gap_s=400)
    with pytest.raises(ValueError, match="exceeds"):
        ad.reconstruct(long, make_mask(len(long), 40, 0.5))


def test_native_head_defaults_match_linear_probe_setup(unitraj):
    ad, _ = unitraj
    assert ad.head_hidden is None and ad.head_class_weighted is False


# ------------------------------------------------------------------ TrajGPT
def test_trajgpt_legacy_order_leaks_target_departure_fixed_does_not():
    from mobeval.nn.trajgpt_net import TrajGPT
    torch.manual_seed(0)
    B, L = 3, 8
    inp = dict(region_id=torch.randint(4, 20, (B, L)), x=torch.randn(B, L) * 500, y=torch.randn(B, L) * 500,
               arrival_time=torch.rand(B, L) * 5, departure_time=torch.rand(B, L) * 5)
    moved = {k: v.clone() for k, v in inp.items()}
    moved["departure_time"][:, -1] += 0.5
    for order, leaks in (("legacy", True), ("fixed", False)):
        net = TrajGPT(16, L, 5000.0, input_order=order).eval()
        with torch.no_grad():
            d = (net(inp)["duration"]["loc"][:, -1] - net(moved)["duration"]["loc"][:, -1]).abs().max().item()
        assert (d > 1e-6) == leaks


def test_trajgpt_train_predict_generate(ctx, tmp_path):
    from mobeval.adapters.trajgpt import TrajGPTAdapter
    path = tmp_path / "trajgpt.pt"
    TrajGPTAdapter.train(ctx, train=FAST, out=str(path), arch={"num_layers": 1})
    ad = TrajGPTAdapter.from_checkpoint(str(path), device="cpu")
    v = TargetGuard.hide_visits(ctx.visits["test"])
    loc = ad.predict_location(v, ctx.grid)
    assert loc.token_scores.shape == (len(v), ad.tok.n_regions)
    for target in ("travel_time", "duration"):
        mix = ad.predict_continuous(v, target).mixture
        assert np.allclose(mix.weights.sum(1), 1) and np.isfinite(mix.means).all() and (mix.stds > 0).all()
    gen = ad.generate(ctx.splits["train"], 5, seed=0, n_visits=3)
    assert {"user_id", "traj_id", "t", "lat", "lon"} <= set(gen.points.columns) and len(gen.points) == 5 * 3 * 2


# ------------------------------------------------------------------ CLIP
def test_clip_pairs_only_past_visits(ctx):
    import pandas as pd
    from mobeval.adapters.clip_mobility import CLIPMobilityAdapter
    from mobeval.geo import LocalProjection
    sp = pd.concat(ctx.staypoints.values(), ignore_index=True)
    w = ctx.windows["train"]
    tok, pad = CLIPMobilityAdapter.pair_windows_with_visits(w, sp, 4, LocalProjection(55.676, 12.568))
    assert tok.shape == (len(w), 4, 9)
    for i in range(0, len(w), max(1, len(w) // 20)):
        g = sp[sp.user_id == w.user_id[i]]
        assert (~pad[i]).sum() == min(4, int((g.t_leave <= w.t[i, 0]).sum()))


def test_clip_pretrain_recovery_embeddings(ctx, tmp_path):
    from mobeval.adapters.clip_mobility import CLIPMobilityAdapter
    path = tmp_path / "clip.pt"
    arch = dict(d_model=32, nhead=2, num_layers=1, dim_feedforward=64, embedding_dim=16)
    CLIPMobilityAdapter.pretrain(ctx, arch=arch, train=FAST, out=str(path))
    ad = CLIPMobilityAdapter.from_checkpoint(str(path), device="cpu", head_train={"epochs": 2})
    b = ctx.windows["test"]
    m = make_mask(len(b), b.length, 0.3, "random", 1)
    la, lo = ad.reconstruct(TargetGuard.hide_masked(b, m), m)
    assert np.isfinite(la).all() and np.array_equal(la[~m], b.lat[~m])
    z = ad.embed(b)
    assert np.allclose(np.linalg.norm(z, axis=1), 1, atol=1e-5)
    ad.prepare("next_location", ctx.visits["train"], None)
    assert ad.predict_location(TargetGuard.hide_visits(ctx.visits["test"]), ctx.grid).grid_scores.shape[1] == ctx.grid.n_cells


# ------------------------------------------------------------------ registry / CLI
def test_provenance_blocks_checkpoint_from_other_split(ctx, unitraj):
    from mobeval.registry import ProvenanceError, load_model
    _, path = unitraj
    other = EvalContext(synthetic_dataset(n_users=14, n_days=6, seed=3),
                        EvalConfig(window_length=24, visit_context=4, split_by="user"))
    spec = {"name": "U", "type": "unitraj", "checkpoint": str(path), "adapter": {"device": "cpu"}}
    assert load_model(spec, ctx).name == "U"
    with pytest.raises(ProvenanceError):
        load_model(spec, other)
    load_model(spec, other, check_provenance="warn")


def test_cli_smoke(tmp_path):
    from mobeval.cli import main
    assert main(["smoke", "--out", str(tmp_path / "smoke"), "--device", "cpu"]) == 0
    assert (tmp_path / "smoke" / "report.md").exists() and (tmp_path / "smoke" / "checkpoints" / "TrajGPT-tiny.pt").exists()
