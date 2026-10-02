"""Original-methodology support: paper metrics, recipes, original sampling, paper datasets."""
import numpy as np
import pandas as pd
import pytest

from mobeval import EvalConfig, synthetic_dataset
from mobeval.data import make_mask


# ------------------------------------------------------------------ masks from the papers
def test_last_and_keep_every_masks_match_the_original_padders():
    m = make_mask(3, 20, 5, "last")
    assert m.shape == (3, 20) and (m.sum(1) == 5).all() and m[:, -5:].all()
    k = make_mask(2, 20, 8, "keep_every")
    kept = np.where(~k[0])[0].tolist()
    assert kept == [0, 8, 16, 19]                      # TRecPadder: every 8th + the last point
    k2 = make_mask(1, 17, 8, "keep_every")             # (L-1) % 8 == 0: the last is already kept
    assert np.where(~k2[0])[0].tolist() == [0, 8, 16]


def test_constant_velocity_extrapolates_a_straight_line_exactly():
    from mobeval import baselines as B
    from mobeval.data import TrajectoryBatch
    t = np.arange(10, dtype=float)[None] * 10
    b = TrajectoryBatch(45 + 1e-4 * t, 9 + 2e-4 * t, t, np.array(["u"]), np.array(["x"]))
    m = make_mask(1, 10, 3, "last")
    la, lo = B.constant_velocity(b, m)
    assert np.allclose(la, b.lat) and np.allclose(lo, b.lon)


# ------------------------------------------------------------------ TrajGPT's P(+-t)
def test_p_within_matches_trajgpt_definition():
    from mobeval.metrics.probabilistic import Mixture, p_within
    y = np.array([30.0, 60.0])
    sharp = Mixture(np.ones((2, 1)), y[:, None], np.full((2, 1), 0.1))
    out = p_within(y, 600.0, mixture=sharp)
    assert np.allclose(out["p_within_5min"], 1.0)
    wide = Mixture(np.ones((2, 1)), y[:, None], np.full((2, 1), 1e4))
    # nearly flat on [0, 600]: mass within +-10 is ~20/600 once truncated and renormalised
    assert np.allclose(p_within(y, 600.0, mixture=wide)["p_within_10min"], 20 / 600, atol=2e-3)
    pt = p_within(y, 600.0, point=np.array([34.0, 80.0]))
    assert pt["p_within_5min"].tolist() == [1.0, 0.0] and pt["p_within_20min"].tolist() == [1.0, 1.0]


# ------------------------------------------------------------------ paper metric registry
def test_paper_metric_levels_depend_on_the_protocol():
    from mobeval.paper_metrics import lookup
    cfg = EvalConfig()
    from types import SimpleNamespace
    from mobeval.data import TrajectoryBatch
    t3 = np.arange(200, dtype=float)[None].repeat(4, 0) * 3.0
    w3 = TrajectoryBatch(np.zeros((4, 200)), np.zeros((4, 200)), t3, np.zeros(4), np.zeros(4))
    ctx = lambda c, t=t3: SimpleNamespace(cfg=c, windows={"test": TrajectoryBatch(w3.lat, w3.lon, t, w3.user_id, w3.traj_id)})
    paper_uni = EvalConfig(window_length=200, recovery_keep_endpoints=False)
    assert lookup("trajgpt", "next_location", "acc@10", "native", ctx(cfg))[0] == "near"
    assert lookup("trajgpt", "next_location", "median_dist_err_m", "native", ctx(cfg)) is None
    assert lookup("trajgpt", "continuous/duration|given:location+arrival", "p_within_5min", "native",
                  ctx(cfg))[0] == "near"            # the original's heads read the target's own times
    assert lookup("unitraj", "recovery/random@0.5", "ade_m", "native", ctx(cfg))[0] == "near"
    assert lookup("unitraj", "recovery/random@0.5", "ade_m", "native", ctx(paper_uni))[0] == "exact"
    assert lookup("unitraj", "recovery/random@0.5", "ade_m", "native", ctx(paper_uni, t3 / 3))[0] == "near"  # 1 s
    assert lookup("unitraj", "recovery/random@0.5", "ade_m", "native", paper_uni)[0] == "near"       # unverifiable
    assert lookup("transfertraj", "continuous/travel_time/linear_probe|given:location|<=4h", "mape",
                  "linear_probe", cfg)[0] == "near"
    assert lookup("clip_mobility", "recovery/random@0.5", "ade_m", "native", cfg) is None
    # a metric is only marked for the model whose paper reported it
    assert lookup("unitraj", "next_location", "acc@1", "linear_probe", cfg) is None


# ------------------------------------------------------------------ recipes
def test_recipe_merge_records_every_override():
    from mobeval import recipes
    spec = {"name": "T", "type": "trajgpt", "recipe": "paper",
            "train": {"epochs": 3, "device": "cpu", "options": {"seq_len": 16}}, "arch": {"num_layers": 2}}
    out, rec = recipes.apply(spec)
    assert out["train"]["optimizer"] == "adam" and out["train"]["lr"] == 1e-4 and out["train"]["epochs"] == 3
    assert out["train"]["options"]["sequences"] == "original" and out["arch"]["num_heads"] == 8
    assert any("train.epochs" in o for o in rec["overrides"]) and any("seq_len" in o for o in rec["overrides"])
    assert not any("num_layers" in o for o in rec["overrides"])          # same value as the recipe
    assert not any("device" in o for o in rec["overrides"])


def test_recipe_data_options_and_finetuning():
    from mobeval import recipes
    spec = {"name": "T", "type": "transfertraj", "recipe": "paper",
            "train": {"options": {"context": {"poi_embed": "x.npy"}, "objective": "tp"}}}
    out, rec = recipes.apply(spec)
    assert rec["overrides"] == []                                         # context is data, not method
    assert out["train"]["scheduler"] == "step" and out["train"]["options"]["objective"] == "tp"
    with pytest.raises(ValueError, match="no publication"):
        recipes.apply({"name": "C", "type": "clip_mobility", "recipe": "paper"})
    with pytest.raises(ValueError, match="unknown recipe"):
        recipes.apply({"name": "U", "type": "unitraj", "recipe": "papr"})


def test_recipe_adapter_keys_are_not_passed_twice():
    from mobeval import recipes
    out, rec = recipes.apply({"name": "T", "type": "transfertraj", "recipe": "paper",
                              "adapter": {"coord_scale": 1000.0}})
    assert "coord_scale" not in out["train"]["options"]
    assert any("adapter.coord_scale" in o for o in rec["overrides"])


def test_every_recipe_option_is_accepted_by_its_training_method():
    import inspect
    from mobeval import recipes
    from mobeval.nn.common import TrainConfig
    from mobeval.registry import MODEL_TYPES, TRAIN_METHOD
    for mtype, rs in recipes.RECIPES.items():
        cls = MODEL_TYPES[mtype]()
        sig = inspect.signature(getattr(cls, TRAIN_METHOD[mtype]))
        ctor = inspect.signature(cls.__init__)
        for name, r in rs.items():
            TrainConfig.from_dict({k: v for k, v in r["train"].items()})
            for k in r["options"]:
                assert k in sig.parameters or k in ctor.parameters, (mtype, name, k)
            for k in r.get("eval", {}):
                assert k in EvalConfig.__dataclass_fields__, (mtype, name, k)


# ------------------------------------------------------------------ UniTraj's own sampling
def test_unitraj_original_masks_and_fixed_hidden_count():
    from mobeval.nn import unitraj_sampling as us
    rng = np.random.default_rng(0)
    L = 150
    lat, lon = 45 + np.cumsum(rng.normal(0, 1e-4, L)), 9 + np.cumsum(rng.normal(0, 1e-4, L))
    for s in ("random", "rdp", "block", "last_n"):
        m = us.strategy_mask(lon, lat, s, 0.5, rng)
        assert m.sum() == 75, s
        k = us.strategy_mask(lon, lat, s, 0.5, rng, mask_endpoints=False)
        assert not k[0] and not k[-1], s
    t = np.arange(L, dtype=float) + 1.7e9
    norm = {"mean": [0, 0], "std": [0.05, 0.04]}
    x, xt, iv, hid, tgt = us.build_batch([(lat, lon, t)] * 4, rng, 200, norm, resample="none")
    assert (hid.sum(1) == 100).all()                      # exactly max_len * ratio hidden, as the original
    assert not tgt[:, L:].any() and (tgt.sum(1) >= 75).all()
    assert np.all(x[hid[:, None, :].repeat(2, 1)] == 0)
    x, xt, iv, hid, tgt = us.build_batch([(lat, lon, t), (lat[:90], lon[:90], t[:90])], rng, 200, norm,
                                         resample="none", fixed_hidden_count=False, offset_from="first_visible")
    assert len(set((~hid).sum(1))) == 1                   # one visible count per batch


def test_unitraj_atr_resampling():
    from mobeval.nn.unitraj_sampling import atr_resample, log_sampling_ratio
    assert log_sampling_ratio(30) == 1.0 and log_sampling_ratio(700) == 0.35
    rng = np.random.default_rng(1)
    t = 1.7e9 + np.arange(600, dtype=float)
    lat, lon = np.linspace(45, 45.01, 600), np.linspace(9, 9.01, 600)
    seen = set()
    for _ in range(40):
        la, lo, tt, iv = atr_resample(lat, lon, t, rng)
        assert np.all(np.diff(tt) > 0) and iv[0] == 0 and np.allclose(iv[1:], np.diff(tt))
        gaps = np.unique(np.diff(tt))
        seen.add("interval" if len(gaps) == 1 and gaps[0] >= 8 else "ratio")
    assert seen == {"interval", "ratio"}


# ------------------------------------------------------------------ H3 label space
def test_h3_grid_maps_points_to_their_cells():
    pytest.importorskip("h3")
    from mobeval.data import H3Grid
    ds = synthetic_dataset(n_users=4, n_days=2, seed=0)
    g = H3Grid.from_dataset(ds, 8)
    p = ds.points
    c = g.cell_of(p.lat.to_numpy()[:50], p.lon.to_numpy()[:50])
    la, lo = g.centroid(c)
    from mobeval.geo import haversine_m
    assert haversine_m(la, lo, p.lat.to_numpy()[:50], p.lon.to_numpy()[:50]).max() < 600   # res-8 cells
    assert g.cell_of(np.array([0.0]), np.array([0.0])).shape == (1,)                       # outside -> nearest


# ------------------------------------------------------------------ optimisers of the originals
@pytest.mark.parametrize("opt,sched", [("adam", "none"), ("adafactor", "none"), ("adam", "step")])
def test_train_config_optimisers(opt, sched):
    import torch
    from mobeval.nn.common import TrainConfig, fit
    net = torch.nn.Linear(3, 1)
    X, y = torch.randn(64, 3), torch.randn(64, 1)
    cfg = TrainConfig(epochs=3, batch_size=16, optimizer=opt, scheduler=sched, weight_decay=0.0, grad_clip=0,
                      device="cpu", restore_best=False, log_every=0)
    h = fit(net, 64, 16, lambda idx, tr: ((net(X[idx]) - y[idx]) ** 2).mean(), cfg)
    assert len(h) == 3


# ------------------------------------------------------------------ datasets of the papers
def test_transfertraj_h5_loader_and_prepare(tmp_path):
    pytest.importorskip("tables")
    from mobeval.loaders import load_transfertraj_h5, prepare
    n = 30
    trips = pd.DataFrame({"trip": np.repeat([0, 1, 2], n), "seq_i": np.tile(np.arange(n), 3),
                          "time": pd.to_datetime(1.5e9 + np.tile(np.arange(n) * 6.0, 3) + np.repeat([0, 5e3, 9e3], n), unit="s"),
                          "lng": 104 + np.random.default_rng(0).random(3 * n) * 1e-3,
                          "lat": 30.7 + np.random.default_rng(1).random(3 * n) * 1e-3})
    info = pd.DataFrame({"trip": [0, 1, 2], "driver": [7, 7, 8]})
    pois = pd.DataFrame({"lng": [104.0, 104.1], "lat": [30.7, 30.8]})
    f = tmp_path / "c.h5"
    with pd.HDFStore(f, "w") as st:
        st["trips"], st["trip_info"], st["pois"] = trips, info, pois
    ds = load_transfertraj_h5(str(f), context_out=str(tmp_path / "ctx"))
    assert set(ds.points.user_id) == {"7", "8"} and ds.points.traj_id.nunique() == 3
    assert np.load(tmp_path / "ctx" / "poi_latlon.npy").tolist() == [[30.7, 104.0], [30.8, 104.1]]
    d3 = prepare(ds, every_nth=3)
    assert len(d3) == 3 * 10
    d12 = prepare(ds, min_interval_s=12, max_traj_points=15)
    assert d12.points.groupby("traj_id").t.diff().dropna().min() >= 12


def test_porto_loader(tmp_path):
    from mobeval.loaders import load_porto
    f = tmp_path / "train.csv"
    f.write_text('TRIP_ID,CALL_TYPE,ORIGIN_CALL,ORIGIN_STAND,TAXI_ID,TIMESTAMP,DAY_TYPE,MISSING_DATA,POLYLINE\n'
                 '"1","C","","","20000589","1372636858","A","False","[[-8.61,41.14],[-8.62,41.15]]"\n'
                 '"2","C","","","20000590","1372636858","A","True","[[-8.61,41.14]]"\n')
    p = load_porto(str(f)).points
    assert len(p) == 2 and p.t.diff().iloc[1] == 15.0 and p.lon.iloc[0] == -8.61


# ------------------------------------------------------------------ TrajGPT's original instances
def test_trajgpt_original_instances_follow_the_repository():
    from mobeval.adapters.trajgpt import _original_instances
    from mobeval.context import EvalContext
    ctx = EvalContext(synthetic_dataset(n_users=6, n_days=6, seed=4), EvalConfig(window_length=24, visit_context=4))
    users, inst = _original_instances(ctx, "train", 8, 1)
    assert inst and all(e - s0 + 1 <= 8 for _, s0, e in inst)
    for u, (la, *_ , own) in enumerate(users):
        n = len(la)
        starts = sorted(s0 for uu, s0, _ in inst if uu == u)
        assert 0 in starts or not own[1:].any()
        assert all(s0 <= max(0, n - 8) for s0 in starts)          # never a window past the last full one


def test_unitraj_first_visible_anchor_is_never_hidden_by_the_top_up():
    from mobeval.nn import unitraj_sampling as us
    rng = np.random.default_rng(3)
    norm = {"mean": [0, 0], "std": [0.05, 0.04]}
    samples = []
    for L in (40, 60, 100, 150):
        t = np.arange(L, dtype=float) + 1.7e9
        samples.append((45 + np.cumsum(rng.normal(0, 1e-4, L)), 9 + np.cumsum(rng.normal(0, 1e-4, L)), t))
    for _ in range(30):
        x, xt, iv, hid, tgt = us.build_batch(samples, rng, 200, norm, resample="none", offset_from="first_visible")
        for i in range(len(samples)):
            j = int(np.argmax(~hid[i]))                           # first visible position
            assert np.allclose(xt[i, :, j], -np.asarray(norm["mean"]) / np.asarray(norm["std"]))  # offset 0 there


def test_drop_last_is_configurable():
    import torch
    from mobeval.nn.common import TrainConfig, fit
    seen = []
    net = torch.nn.Linear(1, 1)
    def loss(idx, tr):
        if tr:
            seen.append(len(idx))
        return net(torch.ones(len(idx), 1)).mean()
    fit(net, 10, 2, loss, TrainConfig(epochs=1, batch_size=4, drop_last=False, device="cpu", log_every=0))
    assert sorted(seen) == [2, 4, 4]
