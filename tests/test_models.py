"""Tests for the PyTorch model adapters, training, checkpoints, provenance and CLI."""
from pathlib import Path

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
    out = tmp_path / "smoke"
    assert main(["smoke", "--out", str(out), "--device", "cpu"]) == 0
    # Checkpoints are shared across runs; the report belongs to this run, reachable via `latest`.
    assert (out / "checkpoints" / "TrajGPT-tiny.pt").exists()
    assert (out / "latest" / "report.md").exists()
    runs = sorted((out / "runs").iterdir())
    assert len(runs) == 1 and (runs[0] / "results.jsonl").exists()


def test_clip_validation_is_finite_with_short_visit_histories():
    """Regression: left-padded visits + causal mask gave NaN in PyTorch's fast inference path."""
    import pandas as pd
    from mobeval.adapters.clip_mobility import CLIPMobilityAdapter
    from mobeval.data import TrajectoryBatch
    from mobeval.geo import LocalProjection
    from mobeval.nn.clip_net import CLIPMobilityModel
    sp = pd.DataFrame({"user_id": ["a", "a", "b"], "lat": [55.6, 55.61, 55.7], "lon": [12.5, 12.51, 12.6],
                       "t_arrive": [0., 5000, 0], "t_leave": [3000., 9000, 4000]})
    w = TrajectoryBatch(np.full((2, 20), 55.6), np.full((2, 20), 12.5), np.tile(np.arange(20) * 15. + 10000, (2, 1)),
                        np.array(["a", "b"]), np.arange(2))
    tok, pad = CLIPMobilityAdapter.pair_windows_with_visits(w, sp, 8, LocalProjection(55.6, 12.5))
    assert not pad[:, 0].any() and (~pad).sum(1).tolist() == [2, 1]
    m = CLIPMobilityModel(9, 64, 9, 8, d_model=32, nhead=2, num_layers=2, dim_feedforward=64, embedding_dim=16).eval()
    with torch.no_grad():
        assert torch.isfinite(m(torch.randn(2, 20, 9), torch.as_tensor(tok), None, torch.as_tensor(pad))["clip_loss"])
    with pytest.raises(ValueError, match="RIGHT"):
        m.visit_transformer.hidden_states(torch.as_tensor(tok), torch.as_tensor(pad[:, ::-1].copy()))


def test_point_tokens_are_bounded_under_gps_glitches():
    from mobeval.nn.features import MAX_OFFSET_KM, point_tokens
    lat = np.full((1, 10), 45.0); lon = np.full((1, 10), 9.0); lat[0, 5] = 60.0      # 1,600 km jump
    tok = point_tokens(lat, lon, np.arange(10)[None] * 1.0)
    assert np.abs(tok[..., :2]).max() <= MAX_OFFSET_KM and np.isfinite(tok).all()


# ------------------------------------------------------------------ TransferTraj
@pytest.fixture(scope="module")
def transfertraj(ctx, tmp_path_factory):
    from mobeval.adapters.transfertraj import TransferTrajAdapter
    out = tmp_path_factory.mktemp("ck") / "transfertraj.pt"
    ad = TransferTrajAdapter.pretrain(ctx, arch=dict(embed_size=16, d_model=32, rafee_layer=1), train=FAST,
                                      out=str(out), batch_size=32)
    return ad, out


def test_transfertraj_recovery_roundtrip_and_embeddings(ctx, transfertraj):
    from mobeval.adapters.transfertraj import TransferTrajAdapter
    ad, path = transfertraj
    b = ctx.windows["test"]
    m = make_mask(len(b), b.length, 0.5, "random", 0)
    la, lo = ad.reconstruct(TargetGuard.hide_masked(b, m), m)
    assert np.isfinite(la[m]).all() and np.array_equal(la[~m], b.lat[~m])          # observed points untouched
    ad2 = TransferTrajAdapter.from_checkpoint(str(path), device="cpu")
    la2, _ = ad2.reconstruct(TargetGuard.hide_masked(b, m), m)
    assert np.allclose(la, la2) and ad2.provenance["train_fingerprint"] == ctx.fingerprint
    assert ad2.embed(b).shape == (len(b), 32)


def test_transfertraj_encoding_is_invertible(ctx, transfertraj):
    """Relative scaled metres -> degrees must round-trip, whatever coord_scale is."""
    ad, _ = transfertraj
    b = ctx.windows["test"].take(np.arange(4))
    seq, _, fp = ad._encode(b.lat, b.lon, b.t, np.zeros(b.lat.shape, bool))
    xy = (seq[..., :2, 0] + fp.unsqueeze(1)).numpy() * ad.coord_scale
    la, lo = ad.proj.to_latlon(xy[..., 0], xy[..., 1])
    assert np.allclose(la, b.lat, atol=1e-6) and np.allclose(lo, b.lon, atol=1e-6)


def test_transfertraj_pretrain_masks_cover_both_modalities():
    from mobeval.adapters.transfertraj import TransferTrajAdapter
    h = TransferTrajAdapter._pretrain_masks(8, 32, np.random.default_rng(0), 0.2, 0.4, 0.2)
    assert h.shape == (8, 32, 2) and h[..., 0].any() and h[..., 1].any() and not h.all()


def test_transfertraj_works_without_poi_or_road_data(ctx, transfertraj):
    """The optional context pathways must keep the architecture intact when no data is given."""
    ad, _ = transfertraj
    assert ad.net.poi_embed_mat.shape == (1, 1) and float(ad.net.poi_coors.min()) > 1e10
    assert "poi_embed_mat" not in ad.net.state_dict()          # non-persistent: not part of checkpoints


def test_transfertraj_inference_is_deterministic(ctx, transfertraj):
    """The MoE router's noise must not leak into evaluation (the original randomises every forward)."""
    ad, _ = transfertraj
    b = ctx.windows["test"].take(np.arange(8))
    m = make_mask(len(b), b.length, 0.5, "random", 2)
    hidden = TargetGuard.hide_masked(b, m)
    assert np.array_equal(ad.reconstruct(hidden, m)[0], ad.reconstruct(hidden, m)[0])
    assert np.array_equal(ad.embed(b), ad._embed_batch(b))


# ------------------------------------------------- TransferTraj context features (POI / road)
def _context_files(tmp_path, ctx, n_poi=40, n_road=25, dim=8, seed=0):
    """POI / road arrays scattered over the dataset's own area."""
    from mobeval.context_features import dataset_bbox
    rng = np.random.default_rng(seed)
    lat_min, lat_max, lon_min, lon_max = dataset_bbox(ctx.splits["train"], pad_km=0.0)
    out = {}
    for kind, n in (("poi", n_poi), ("road", n_road)):
        np.save(tmp_path / f"{kind}_embed.npy", rng.normal(size=(n, dim)).astype(np.float32))
        np.save(tmp_path / f"{kind}_latlon.npy",
                np.column_stack([rng.uniform(lat_min, lat_max, n), rng.uniform(lon_min, lon_max, n)]))
        out[f"{kind}_embed"] = str(tmp_path / f"{kind}_embed.npy")
        out[f"{kind}_latlon"] = str(tmp_path / f"{kind}_latlon.npy")
    return out


def test_context_lookup_matches_the_naive_masked_mean():
    """The chunked matmul must equal materialising (B, L, N, D) and averaging, at any chunk size."""
    from mobeval.nn.transfertraj_net import TransferTraj
    torch.manual_seed(0)
    net = TransferTraj(embed_size=8, d_model=16, poi_embed=torch.randn(37, 5),
                       poi_coors=torch.randn(37, 2) * 400, poi_dist=250_000.0).eval()
    B, L = 2, 6
    spatial, first = torch.randn(B, L, 2) * 300, torch.randn(B, 2) * 100
    fmask, token_e = torch.zeros(B, L, dtype=torch.bool), torch.zeros(B, L, 16)
    with torch.no_grad():
        emb = net.poi_embed_layer(net.poi_embed_mat)
        d = ((net.poi_coors[None, None] - (spatial + first.unsqueeze(1)).unsqueeze(2)) ** 2).sum(-1)
        m = (d < net.poi_dist).unsqueeze(-1)
        naive = (emb[None, None] * m).sum(2) / m.sum(2).clamp(min=1)
        for chunk in (1024, 8, 3):
            got = net._context_embed(net.poi_embed_layer, net.poi_embed_mat, net.poi_coors, spatial, first,
                                     net.poi_dist, fmask, token_e, chunk=chunk)
            assert torch.allclose(got, naive, atol=1e-5), chunk


def test_transfertraj_uses_context_and_survives_a_checkpoint_roundtrip(ctx, tmp_path):
    from mobeval.adapters.transfertraj import TransferTrajAdapter
    context = _context_files(tmp_path, ctx)
    arch = dict(embed_size=16, d_model=32, rafee_layer=1, poi_dist=250_000.0, rn_dist=250_000.0)
    path = tmp_path / "tt_ctx.pt"
    ad = TransferTrajAdapter.pretrain(ctx, arch=arch, train=FAST, out=str(path), context=context, batch_size=32)
    assert ad.net.poi_embed_mat.shape == (40, 8) and ad.net.road_embed_mat.shape == (25, 8)
    b = ctx.windows["test"].take(np.arange(6))
    m = make_mask(len(b), b.length, 0.5, "block", 0)
    hidden = TargetGuard.hide_masked(b, m)
    ad2 = TransferTrajAdapter.from_checkpoint(str(path), device="cpu")
    assert np.allclose(ad.reconstruct(hidden, m)[0], ad2.reconstruct(hidden, m)[0])
    # POI/road matrices are inputs, not weights: they must stay out of the state dict
    assert not [k for k in ad.net.state_dict() if k.endswith(("poi_embed_mat", "road_coors"))]


def test_context_paths_can_be_overridden_and_missing_files_are_reported(ctx, tmp_path):
    from mobeval.adapters.transfertraj import TransferTrajAdapter
    context = _context_files(tmp_path, ctx)
    path = tmp_path / "tt_ov.pt"
    TransferTrajAdapter.pretrain(ctx, arch=dict(embed_size=16, d_model=32, rafee_layer=1), train=FAST,
                                 out=str(path), context=context, batch_size=32)
    moved = tmp_path / "moved"
    moved.mkdir()
    for k, v in context.items():                       # simulate staging to a different directory
        (moved / Path(v).name).write_bytes(Path(v).read_bytes())
        Path(v).unlink()
    with pytest.raises(FileNotFoundError, match="override them"):
        TransferTrajAdapter.from_checkpoint(str(path), device="cpu")
    ad = TransferTrajAdapter.from_checkpoint(
        str(path), device="cpu", context={k: str(moved / Path(v).name) for k, v in context.items()})
    assert ad.net.poi_embed_mat.shape == (40, 8)


def test_mismatched_context_arrays_are_rejected(ctx, tmp_path):
    from mobeval.adapters.transfertraj import TransferTrajAdapter
    np.save(tmp_path / "e.npy", np.zeros((10, 4), np.float32))
    np.save(tmp_path / "c.npy", np.zeros((7, 2)))
    with pytest.raises(ValueError, match="same length"):
        TransferTrajAdapter(center=(55.6, 12.5), device="cpu",
                            context={"poi_embed": str(tmp_path / "e.npy"), "poi_latlon": str(tmp_path / "c.npy")})


def test_trajgpt_mask_capacity_is_generous_and_context_mismatch_warns(ctx, tmp_path, caplog):
    """sequence_len only sizes the causal masks: no parameter depends on it and it is not persisted,
    so a checkpoint must not be locked to the context it was trained with - but using a different
    one must say so."""
    from mobeval.adapters.trajgpt import MIN_SEQUENCE_LEN, TrajGPTAdapter
    path = tmp_path / "tg_ctx.pt"
    ad = TrajGPTAdapter.train(ctx, train=FAST, out=str(path), arch={"num_layers": 1})
    trained_context = ctx.visits["train"].ctx_lat.shape[1]
    assert ad.sequence_len >= MIN_SEQUENCE_LEN
    assert ad.provenance["train_context"] == trained_context

    reloaded = TrajGPTAdapter.from_checkpoint(str(path), device="cpu")
    with caplog.at_level("WARNING"):
        reloaded._check_len(trained_context)                      # same context: silent
        assert not [r for r in caplog.records if "trained with" in r.message]
        reloaded._check_len(trained_context + 4)                  # different: warns once
        assert [r for r in caplog.records if "trained with" in r.message]


def test_transfertraj_context_radius_is_in_metres_whatever_the_coord_scale(tmp_path):
    """poi_dist / rn_dist are squared METRES, as in the original. The network sees coordinates
    divided by coord_scale, so passing the threshold through unchanged turned the original 10 m
    radius into 10 km at coord_scale=1000: every point averaged nearly every POI in the city."""
    import numpy as np
    from mobeval.adapters.transfertraj import TransferTrajAdapter
    rng = np.random.default_rng(0)
    lat0, lon0 = 45.0, 9.0
    poi = np.column_stack([lat0 + rng.normal(0, 0.02, 3000), lon0 + rng.normal(0, 0.02, 3000)])
    np.save(tmp_path / "pe.npy", rng.normal(size=(3000, 8)).astype(np.float32))
    np.save(tmp_path / "pl.npy", poi)
    ctx = {"poi_embed": str(tmp_path / "pe.npy"), "poi_latlon": str(tmp_path / "pl.npy")}
    pts_lat, pts_lon = lat0 + rng.normal(0, 0.02, 500), lon0 + rng.normal(0, 0.02, 500)
    counts = {}
    for scale in (1.0, 1000.0):
        ad = TransferTrajAdapter(arch={"embed_size": 8, "d_model": 16, "rafee_layer": 1,
                                       "poi_dist": 100.0 ** 2, "rn_dist": 100.0 ** 2},
                                 center=(lat0, lon0), coord_scale=scale, context=ctx, device="cpu")
        x, y = ad.proj.to_xy(pts_lat, pts_lon)
        pts = np.column_stack([x, y]) / scale
        coors = ad.net.poi_coors.cpu().numpy()
        d2 = ((coors[None, :, :] - pts[:, None, :]) ** 2).sum(-1)
        counts[scale] = (d2 < ad.net.poi_dist).sum(1)
        assert ad.arch["poi_dist"] == 100.0 ** 2, "the checkpoint must keep the user-facing squared metres"
    assert np.array_equal(counts[1.0], counts[1000.0]), "matched POIs changed with coord_scale"
    assert 0 < counts[1.0].mean() < 100, f"a 100 m radius matched {counts[1.0].mean():.0f} POIs per point"


def test_visit_sequences_flag_context_visits_from_other_splits():
    """Contexts may reach into a user's visits from another split (that is history a deployed
    model has), but anything trained on every context position must know which ones they are."""
    import numpy as np, pandas as pd
    from mobeval.data import SpatialGrid, make_visit_sequences
    n = 12
    sp = pd.DataFrame({"user_id": "u", "traj_id": [f"{'tr' if i % 2 == 0 else 'te'}{i}" for i in range(n)],
                       "lat": 45 + np.arange(n) * 1e-3, "lon": 9.0,
                       "t_arrive": np.arange(n) * 7200.0, "t_leave": np.arange(n) * 7200.0 + 3600})
    grid = SpatialGrid(44.9, 45.1, 8.9, 9.1, 500.0)
    train_ids = {t for t in sp.traj_id if t.startswith("tr")}
    v = make_visit_sequences(sp, grid, context=4, target_traj_ids=train_ids)
    assert len(v) > 0
    for i in range(len(v)):
        start = int(round((v.tgt_t_arrive[i] / 7200.0))) - 4
        expected = [sp.traj_id[start + j] in train_ids for j in range(4)]
        assert list(v.ctx_in_split[i]) == expected
    # without a split restriction everything counts as the sample's own split
    assert make_visit_sequences(sp, grid, context=4).ctx_in_split.all()


def test_trajgpt_does_not_train_on_other_splits_visits(caplog):
    """Regression: TrajGPT's loss is taken at every position, and under interleaved predefined
    splits those positions include test visits - the model was training on the test set."""
    import logging
    import numpy as np, pandas as pd
    from mobeval import EvalConfig, EvaluationPipeline, synthetic_dataset
    from mobeval.adapters.trajgpt import TrajGPTAdapter
    from mobeval.data import MobilityDataset
    ds = synthetic_dataset(n_users=12, n_days=10, seed=0)
    pts = ds.points.copy()
    order = {t: i for i, t in enumerate(pts.drop_duplicates("traj_id").sort_values("t").traj_id)}
    pts["split"] = ["test" if order[t] % 3 == 2 else "train" for t in pts.traj_id]   # interleaved
    cfg = EvalConfig(split_by="predefined", window_length=16, visit_context=4, max_eval_samples=100)
    ctx = EvaluationPipeline(cfg).prepare(MobilityDataset(pts, "interleaved"))
    assert (~ctx.visits["train"].ctx_in_split).any(), "the fixture must actually interleave splits"
    with caplog.at_level(logging.INFO, logger="mobeval"):
        TrajGPTAdapter.train(ctx, train={"epochs": 1, "batch_size": 32, "device": "cpu"},
                             arch={"num_layers": 1}, tokenizer={"cell_m": 1000})
    assert any("excluded from the training loss" in r.message for r in caplog.records)


@pytest.mark.parametrize("head_extra", [4, 0])     # HEAD commit: nr+8 outputs; paper-era 49aad40: nr+4
def test_trajgpt_loads_original_repository_state_dicts(tmp_path, head_extra):
    """The original main.py passes num_regions + 4 and the modules add 4 again: the embedding has
    nr+8 rows, the head nr+8 (HEAD) or nr+4 (49aad40). Both must load, and only real regions scored."""
    torch = pytest.importorskip("torch")
    h3 = pytest.importorskip("h3")
    from mobeval.adapters.trajgpt import TrajGPTAdapter
    from mobeval.nn.trajgpt_net import TrajGPT
    nr = 12
    cells = sorted({h3.latlng_to_cell(43.7 + 0.08 * i, 10.4 + 0.08 * (i % 3), 7) for i in range(40)})[:nr]
    orig = TrajGPT(nr + 4, 128, 20000.0, embedding_rows=nr + 8, head_rows=nr + 4 + head_extra)
    p = tmp_path / "orig.pt"
    torch.save(orig.state_dict(), p)
    ad = TrajGPTAdapter.from_original_state_dict(p, cells, 20000.0, 4.0, 24.0, 0.0, (43.9, 10.5), device="cpu")
    assert ad.net.region_id_head.out_features == nr + 4 + head_extra
    B, S = 2, 6
    inp = dict(region_id=torch.randint(4, 4 + nr, (B, S)), x=torch.randn(B, S), y=torch.randn(B, S),
               arrival_time=torch.linspace(0, 1, S).repeat(B, 1), departure_time=torch.linspace(0, 1, S).repeat(B, 1))
    with torch.no_grad():
        assert ad._region_logits(ad.net(inp)).shape == (B, nr)
    with pytest.raises(ValueError, match="room for"):
        TrajGPTAdapter.from_original_state_dict(p, cells * 3, 20000.0, 4.0, 24.0, 0.0, (43.9, 10.5), device="cpu")


def test_unitraj_rejects_unequal_visible_counts():
    from mobeval.nn.unitraj_net import UniTraj
    hidden = np.zeros((2, 10), bool)
    hidden[0, :5] = True
    hidden[1, :4] = True
    with pytest.raises(ValueError, match="same number of visible"):
        UniTraj.permutations(hidden, np.random.default_rng(0))
