import numpy as np
import pandas as pd
import pytest

from mobeval import EvalConfig, EvaluationPipeline, synthetic_dataset
from mobeval.adapters.base import NEXT_LOCATION, RECOVERY, LocationPrediction, MobilityModelAdapter
from mobeval.adapters.reference import KinematicReference, WeakReference
from mobeval.baselines import linear_interpolation
from mobeval.data import SpatialGrid, TrajectoryBatch, make_mask
from mobeval.geo import haversine_m
from mobeval.metrics.classification import ranking_metrics
from mobeval.metrics.probabilistic import Mixture, continuous_metrics, crps_mixture, crps_samples
from mobeval.metrics.reconstruction import block_ends, recovery_metrics
from mobeval.results import ResultRecord
from mobeval.stats import skill_score
from mobeval.tasks import NextLocationTask


def test_haversine_one_degree_at_equator():
    assert haversine_m(0, 0, 0, 1) == pytest.approx(111_195, rel=1e-3)


def test_masks():
    m = make_mask(50, 20, 0.5, "random", seed=1)
    assert (m.sum(1) == 10).all() and not m[:, 0].any() and not m[:, -1].any()
    b = make_mask(50, 20, 0.3, "block", seed=1)
    assert (block_ends(b).sum(1) == 1).all()                     # exactly one contiguous block
    assert np.array_equal(make_mask(5, 20, .5, seed=3), make_mask(5, 20, .5, seed=3))


def _line_batch(n=4, L=20):
    t = np.tile(np.arange(L) * 15.0, (n, 1))
    lat = 55.0 + t * 1e-5
    lon = 12.0 + t * 2e-5
    return TrajectoryBatch(lat, lon, t, np.arange(n), np.arange(n))


def test_interpolation_exact_on_straight_line_and_zero_errors():
    b = _line_batch()
    mask = make_mask(len(b), b.length, 0.5, "block", 0)
    lat, lon = linear_interpolation(b, mask)
    m = recovery_metrics(lat, lon, b.lat, b.lon, mask)
    assert m["ade_m"].max() < 1e-3 and m["dtw_m"].max() < 1e-3 and (m["acc_100m"] == 1).all()


def test_nan_leak_is_caught():
    b = _line_batch()
    mask = make_mask(len(b), b.length, 0.5, "random", 0)
    lat = b.lat.copy(); lat[mask] = np.nan
    with pytest.raises(ValueError):
        recovery_metrics(lat, b.lon, b.lat, b.lon, mask)


def test_crps_point_equals_mae_and_mixture_closed_form_matches_samples():
    rng = np.random.default_rng(0)
    y = rng.normal(10, 3, 200)
    pt = y + rng.normal(0, 1, 200)
    out = continuous_metrics(y, point=pt)
    assert np.allclose(out["crps_min"], np.abs(pt - y))
    mix = Mixture(rng.random((200, 3)), rng.normal(10, 2, (200, 3)), rng.uniform(.5, 2, (200, 3)))
    cf = crps_mixture(mix, y)
    mc = crps_samples(mix.sample(4000, seed=1), y)
    assert np.mean(cf) == pytest.approx(np.mean(mc), rel=0.02)


@pytest.mark.parametrize("space", ["linear", "log"])
def test_unit_change_shifts_nll_by_log_jacobian(space):
    rng = np.random.default_rng(1)
    y_s = rng.uniform(300, 5000, 50)
    means = np.log(rng.uniform(300, 5000, (50, 2))) if space == "log" else rng.uniform(300, 5000, (50, 2))
    stds = rng.uniform(.3, 1, (50, 2)) if space == "log" else rng.uniform(200, 900, (50, 2))
    mix_s = Mixture(np.ones((50, 2)), means, stds, space)
    mix_min = mix_s.rescale(1 / 60)
    assert np.allclose(mix_min.nll(y_s / 60), mix_s.nll(y_s) - np.log(60))
    assert np.allclose(mix_min.median(), mix_s.median() / 60, rtol=1e-4)


def test_mixture_median_matches_samples():
    mix = Mixture(np.array([[.7, .3]]), np.array([[np.log(600), np.log(6000)]]), np.array([[.4, .6]]), "log")
    assert mix.median()[0] == pytest.approx(np.median(mix.sample(200_000)[0]), rel=0.02)


def test_skill_scores():
    assert skill_score("ade_m", 50, 100) == pytest.approx(0.5)
    assert skill_score("acc@1", 0.6, 0.2) == pytest.approx(0.5)
    assert skill_score("nll", 2.0, 2.5) == pytest.approx(0.5)
    assert skill_score("jsd", 0.2, 0.5, floor=0.1) == pytest.approx(0.75)


def test_record_flags_percent_scale_and_implausible_units():
    assert ResultRecord("m", "r", "t", "acc@1", 66.0).flags          # 66 instead of 0.66
    assert ResultRecord("m", "r", "t", "mae_min", 138.8 * 60).flags  # 138.8 hours in minutes
    assert not ResultRecord("m", "r", "t", "ade_m", 30.0).flags
    assert ResultRecord("m", "r", "t", "ade_m", 30.0).higher_is_better is False   # direction from registry


def test_token_scores_equal_grid_scores_when_vocab_is_the_grid():
    grid = SpatialGrid(55.6, 55.7, 12.5, 12.6, 1000)
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(grid.n_cells), 30)
    cent = np.column_stack(grid.centroid(np.arange(grid.n_cells)))
    p1, _, _ = NextLocationTask._grid_probs(LocationPrediction(grid_scores=probs), grid, 30)
    p2, _, _ = NextLocationTask._grid_probs(LocationPrediction(token_scores=probs, token_latlon=cent), grid, 30)
    y = rng.integers(0, grid.n_cells, 30)
    assert np.allclose(ranking_metrics(p1, y)["acc@5"], ranking_metrics(p2, y)["acc@5"])


class _Cheater(MobilityModelAdapter):
    name, capabilities = "Cheater", {RECOVERY}

    def reconstruct(self, batch, mask):
        return batch.lat, batch.lon          # tries to return the hidden truth -> NaN -> error


def test_end_to_end_and_leakage_guard():
    ds = synthetic_dataset(n_users=12, n_days=6, seed=2)
    cfg = EvalConfig(window_length=24, recovery_ratios=(0.5,), recovery_kinds=("block",), eval_seeds=(0,),
                     n_boot=30, max_eval_samples=150, visit_context=4, label_fractions=(1.0,),
                     generation_max_trajectories=40)
    pipe = EvaluationPipeline(cfg)
    ctx = pipe.prepare(ds)
    store = pipe.run([KinematicReference(), WeakReference(), _Cheater()], ctx)
    df = store.to_frame()
    assert [e for e in ctx.errors if e[0].startswith("Cheater")], "leakage guard should reject the cheater"
    assert not [e for e in ctx.errors if not e[0].startswith("Cheater")], ctx.errors
    fams = set(df[df.model == "KinematicRef"].family)
    assert {"recovery", "location", "continuous", "classification", "generation", "efficiency"} <= fams
    # baselines scored on identical samples: skill must be reproducible from logged values
    r = df[(df.model == "KinematicRef") & (df.metric == "ade_m")].iloc[0]
    assert r.skill == pytest.approx(1 - r.value / r.baseline_value)


def test_geolife_loader_times_and_labels(tmp_path):
    from mobeval.loaders import load_geolife
    d = tmp_path / "Data" / "007" / "Trajectory"
    d.mkdir(parents=True)
    hdr = "Geolife trajectory\nWGS 84\nAltitude is in Feet\nReserved 3\n0,2,255,My Track,0,0,2,8421376\n0\n"
    rows = [f"39.98{k:02d},116.30{k:02d},0,100,39000.1,2008-10-23,10:{k // 12:02d}:{(k * 5) % 60:02d}" for k in range(60)]
    (d / "20081023100000.plt").write_text(hdr + "\n".join(rows) + "\n")
    (tmp_path / "Data" / "007" / "labels.txt").write_text(
        "Start Time\tEnd Time\tTransportation Mode\n2008/10/23 10:00:00\t2008/10/23 10:02:00\tsubway\n")
    p = load_geolife(str(tmp_path)).points
    assert p.t.iloc[0] == pd.Timestamp("2008-10-23 10:00:00").timestamp()
    assert p.t.diff().iloc[1] == pytest.approx(5.0)
    assert set(p["mode"].dropna()) == {"train"} and p["mode"].notna().sum() == 25


def _vehicle_csvs(tmp_path):
    """Trip-only recordings with the column layout of a vehicle GPS panel, pre-split in two files."""
    p = synthetic_dataset(n_users=10, n_days=8, seed=5).points
    p = p[p["mode"].notna()].copy()
    p["trip"] = (p.groupby("user_id").t.diff().fillna(1e9) > 120).groupby(p.user_id).cumsum()
    df = pd.DataFrame({"uid": p.user_id, "lat": p.lat, "lng": p.lon, "QUALITY": 3,
                       "datetime": pd.to_datetime(p.t, unit="s").dt.strftime("%Y-%m-%d %H:%M:%S"), "trip_id": p.trip})
    cut = df.datetime.sort_values().iloc[int(0.8 * len(df))]
    last = df.groupby(["uid", "trip_id"]).datetime.transform("min") >= cut
    df[~last].to_csv(tmp_path / "train.csv", index=False)
    df[last].to_csv(tmp_path / "test.csv", index=False)
    return tmp_path / "train.csv", tmp_path / "test.csv"


def test_predefined_split_csv_and_trip_staypoints(tmp_path):
    from mobeval.context import EvalContext
    from mobeval.data import staypoints_from_trips
    from mobeval.loaders import from_csv
    tr, te = _vehicle_csvs(tmp_path)
    ds = from_csv(train_path=str(tr), test_path=str(te), time_col="datetime", user_id="uid", traj_id="trip_id",
                  lon="lng", query="QUALITY >= 2", clean={})
    splits = ds.split("predefined", val_by="time", val_fraction=0.1)
    ids = {k: set(v.points.traj_id) for k, v in splits.items()}
    assert not (ids["train"] & ids["val"]) and not (ids["train"] & ids["test"]) and ids["val"] and ids["test"]
    # validation is carved from the train file only, from its latest trajectories
    test_file_ids = set(ds.points.loc[ds.points["split"] == "test", "traj_id"])
    assert ids["test"] == test_file_ids and not (ids["val"] & test_file_ids)
    assert splits["val"].points.groupby("traj_id").t.min().min() >= splits["train"].points.groupby("traj_id").t.min().max() - 1
    sp = staypoints_from_trips(splits["train"])
    assert len(sp) > 0 and ((sp.t_leave - sp.t_arrive) >= 20 * 60).all()
    ctx = EvalContext(ds, EvalConfig(split_by="predefined", staypoint_method="trips", window_length=16, visit_context=3))
    assert len(ctx.visits["test"]) > 0 and len(ctx.windows["test"]) > 0


def test_staypoints_follow_time_not_trajectory_id_order():
    from mobeval.data import MobilityDataset, detect_staypoints
    rows = [("u", "b_first", 0 + 60 * k, 45.0, 9.0) for k in range(30)] + \
           [("u", "a_second", 10_000 + 60 * k, 45.1, 9.1) for k in range(30)]
    sp = detect_staypoints(MobilityDataset(pd.DataFrame(rows, columns=["user_id", "traj_id", "t", "lat", "lon"])))
    assert list(sp.traj_id) == ["b_first", "a_second"] and (sp.t_leave > sp.t_arrive).all()


def test_training_views_can_be_thinned_without_touching_evaluation():
    """Panel data yields millions of overlapping visit sequences; train must be limitable."""
    from mobeval.context import EvalContext
    ds = synthetic_dataset(n_users=20, n_days=12, seed=7)
    base = EvalConfig(window_length=32, visit_context=6, max_eval_samples=50)
    plain = EvalContext(ds, base)
    thin = EvalContext(ds, EvalConfig(window_length=32, visit_context=6, max_eval_samples=50,
                                      visit_stride=4, max_train_samples=100))
    assert len(thin.windows["train"]) == 100 < len(plain.windows["train"])
    assert 0 < len(thin.visits["train"]) < len(plain.visits["train"]) / 2
    for split in ("val", "test"):                       # evaluation views must be identical
        assert len(thin.visits[split]) == len(plain.visits[split])
        assert len(thin.windows[split]) == len(plain.windows[split])
        assert np.array_equal(thin.visits[split].tgt_cell, plain.visits[split].tgt_cell)
    assert thin.fingerprint == plain.fingerprint        # same split: checkpoints stay compatible
