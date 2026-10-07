"""OmniTraj: reconstructed preprocessing, map matching, adapter, retrieval task."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mobeval import EvalConfig, synthetic_dataset
from mobeval.nn import omnitraj_prep as prep

SAMPLE = Path(__file__).parent / "data" / "omnitraj_sample.pkl"      # 40 trips of the repository's Chengdu sample


# ------------------------------------------------------------------ preprocessing vs the authors' sample
def test_topology_is_rdp_1e4_on_the_resampled_trajectory():
    d = pd.read_pickle(SAMPLE)
    for r in d.itertuples():
        tr = np.asarray(r.trajectory, float)
        assert np.allclose(prep.topology(tr), np.asarray(r.topology))


def test_region_ids_follow_the_reconstructed_grid():
    d = pd.read_pickle(SAMPLE)
    g = prep.RegionGrid(30.65172, 104.03534, 0.005137, 0.006, 16, 16)
    xy = np.concatenate([np.asarray(t) for t in d.trajectory])
    c = np.concatenate([np.asarray(x) for x in d.cell_sequence])
    assert (g.ids(xy[:, 1], xy[:, 0]) == c).mean() > 0.99


def test_grid_fit_gives_square_cells_and_compacts_large_areas():
    rng = np.random.default_rng(0)
    lat, lon = 41.9 + rng.random(500) * 0.1, 12.5 + rng.random(500) * 0.1
    g = prep.RegionGrid.fit(lat, lon, n=16)
    assert g.nx == g.ny == 16 and g.num_grids == 256 and g.vocab is None
    mid = (lat.min() + lat.max()) / 2
    assert abs(g.dlat * 111195 - g.dlon * 111195 * np.cos(np.radians(mid))) < 1e-6      # square cells
    ids = g.ids(lat, lon)
    assert ids.min() >= 1 and ids.max() <= 256
    big = prep.RegionGrid.fit(lat, lon, n=None, cell_m=50, max_cells=100)
    assert big.num_grids == 101 and big.ids(np.array([0.0]), np.array([0.0]))[0] == 101   # unseen -> shared id


def test_resample_hits_the_endpoints_and_is_identity_at_200_points():
    lat, lon = np.linspace(41.9, 41.95, 37), np.linspace(12.5, 12.52, 37) ** 1.01
    r = prep.resample(lat, lon)
    assert r.shape == (200, 2) and np.allclose(r[0], [lon[0], lat[0]]) and np.allclose(r[-1], [lon[-1], lat[-1]])
    tr = np.asarray(pd.read_pickle(SAMPLE).trajectory.iloc[0], float)
    assert np.allclose(prep.resample(tr[:, 1], tr[:, 0]), tr)


def test_sample_layout_matches_the_original_dataset_class():
    d = pd.read_pickle(SAMPLE)
    r = d.iloc[0]
    tr = np.asarray(r.trajectory)
    g = prep.RegionGrid(30.65172, 104.03534, 0.005137, 0.006, 16, 16)
    v = prep.RoadVocab.fit([np.asarray(x) for x in d.roads])
    s = prep.build_sample(tr[:, 1], tr[:, 0], np.asarray(r.roads), g, v,
                          {"mean": [104.07596303, 30.68085491], "std": [2.15106194e-02, 1.89193207e-02]}, None, False)
    road = s["road"][s["road_attention_mask"] > 0]
    assert road[0] == v.num_roads - 1 and road[-1] == v.num_roads - 2          # BOS ... EOS
    assert len(road) == len(dict.fromkeys(np.asarray(r.roads).tolist())) + 2   # deduplicated, in order
    assert s["topology_attention_mask"].sum() == len(r.topology)
    # regions: the resampled points' cells, deduplicated in order (the grid itself is checked above)
    assert list(s["region"][s["region_attention_mask"] > 0]) == list(dict.fromkeys(g.ids(tr[:, 1], tr[:, 0]).tolist()))
    long = np.arange(1, 300)
    out, m = prep._pad_or_truncate(long, 128)
    assert len(out) == 128 and out[0] == 1 and out[-1] == 299 and m.all()       # keeps first and LAST


def test_augmentations_only_reorder_or_thin():
    rng = np.random.default_rng(0)
    seq = np.arange(1, 40)
    for _ in range(200):
        a = prep.augment_road(seq, rng, mask_idx=999)
        assert set(a.tolist()) <= set(seq.tolist()) | {999}
        b = prep.augment_region(seq, rng)
        assert set(b.tolist()) <= set(seq.tolist()) and len(b) >= 1


# ------------------------------------------------------------------ roads and map matching
def _grid_network(lat0=41.9, lon0=12.5, n=5, step=200.0):
    from mobeval.roads import RoadNetwork
    dl, dn = step / 111195, step / (111195 * np.cos(np.radians(lat0)))
    lines = []
    for i in range(n):
        lines.append(np.array([[lat0 + i * dl, lon0 + j * dn] for j in range(n)]))
        lines.append(np.array([[lat0 + j * dl, lon0 + i * dn] for j in range(n)]))
    return RoadNetwork.from_polylines(lines), dl, dn


def test_hmm_matcher_follows_the_driven_streets():
    from mobeval.mapmatch import HMMMatcher
    net, dl, dn = _grid_network()
    assert len(net) == 40                                   # 5 x 5 street grid cut at every crossing
    path = [(1, x) for x in np.linspace(0.05, 2.95, 30)] + [(y, 3) for y in np.linspace(1.05, 2.95, 20)]
    rng = np.random.default_rng(0)
    lat = np.array([41.9 + y * dl for y, _ in path]) + rng.normal(0, 4 / 111195, len(path))
    lon = np.array([12.5 + x * dn for _, x in path]) + rng.normal(0, 4 / 111195, len(path))
    seg = HMMMatcher(net, sigma_m=10).match(lat, lon)
    la_mid = [net.geometry(s)[0].mean() for s in seg]
    lo_mid = [net.geometry(s)[1].mean() for s in seg]
    on_row = [abs((la - 41.9) / dl - 1) < 1e-6 for la in la_mid[:30]]
    on_col = [abs((lo - 12.5) / dn - 3) < 1e-6 for lo in lo_mid[30:]]
    assert np.mean(on_row) == 1.0 and np.mean(on_col) == 1.0


def test_hmm_matcher_restarts_across_disconnected_roads():
    from mobeval.mapmatch import HMMMatcher
    from mobeval.roads import RoadNetwork
    a = np.array([[41.9, 12.50], [41.9, 12.51]])
    b = np.array([[41.95, 12.50], [41.95, 12.51]])          # 5.5 km north, not connected
    net = RoadNetwork.from_polylines([a, b])
    lat = np.r_[np.full(5, 41.9), np.full(5, 41.95)]
    lon = np.r_[np.linspace(12.501, 12.509, 5), np.linspace(12.501, 12.509, 5)]
    seg = HMMMatcher(net).match(lat, lon)
    assert list(seg[:5]) == [0] * 5 and list(seg[5:]) == [1] * 5
    assert (HMMMatcher(net).match(np.array([0.0]), np.array([0.0])) == -1).all()


def test_osm_extraction_cuts_at_intersections(tmp_path):
    pytest.importorskip("osmium")
    from mobeval.roads import from_osm
    f = tmp_path / "t.osm"
    f.write_text("""<?xml version="1.0" encoding="UTF-8"?><osm version="0.6">
 <node id="1" lat="41.900" lon="12.500"/><node id="2" lat="41.900" lon="12.502"/><node id="3" lat="41.900" lon="12.504"/>
 <node id="4" lat="41.902" lon="12.502"/><node id="5" lat="41.898" lon="12.502"/><node id="6" lat="41.901" lon="12.505"/>
 <way id="10"><nd ref="1"/><nd ref="2"/><nd ref="3"/><tag k="highway" v="primary"/></way>
 <way id="11"><nd ref="4"/><nd ref="2"/><nd ref="5"/><tag k="highway" v="residential"/><tag k="oneway" v="yes"/></way>
 <way id="12"><nd ref="3"/><nd ref="6"/><tag k="highway" v="footway"/></way></osm>""")
    net = from_osm(str(f))
    assert len(net) == 4 and net.oneway.tolist() == [False, False, True, True]     # footway dropped
    net.save(str(tmp_path / "n.npz"))
    from mobeval.roads import RoadNetwork
    assert len(RoadNetwork.load(str(tmp_path / "n.npz"))) == 4


def test_fmm_roundtrip(tmp_path):
    pytest.importorskip("geopandas")
    from mobeval.mapmatch import fmm_export, fmm_import
    net, _, _ = _grid_network(n=3)
    net.to_fmm(str(tmp_path / "net.gpkg"))
    ds = synthetic_dataset(n_users=2, n_days=1, seed=0)
    fmm_export(ds, str(tmp_path / "gps.csv"))
    gps = pd.read_csv(tmp_path / "gps.csv", sep=";")
    # a fake FMM result: every point on edge 2 (segment 1)
    res = gps.groupby("id").size().reset_index(name="n")
    res["opath"] = res.n.map(lambda n: ",".join(["2"] * n))
    res[["id", "opath"]].to_csv(tmp_path / "out.csv", sep=";", index=False)
    m = fmm_import(ds, str(tmp_path / "out.csv"), str(tmp_path / "net.gpkg"))
    assert len(m) == len(ds.points) and set(m.seg) == {1}


# ------------------------------------------------------------------ adapter
TINY = {"trajectory": {"embed_dim": 32, "depth": 1, "num_heads": 2},
        "topology": {"embed_dim": 32, "num_layers": 1, "num_heads": 2},
        "road": {"embed_dim": 32, "output_dim": 32, "num_layers": 1, "num_heads": 2},
        "region": {"output_dim": 32, "embed_dim": 32, "num_layers": 1, "num_heads": 2}}


@pytest.fixture(scope="module")
def omni(tmp_path_factory):
    pytest.importorskip("timm")
    pytest.importorskip("transformers")
    from mobeval.adapters.omnitraj import OmniTrajAdapter
    from mobeval.context import EvalContext
    d = tmp_path_factory.mktemp("omni")
    ds = synthetic_dataset(n_users=8, n_days=5, seed=1)
    p = ds.points
    roads = pd.DataFrame({"traj_id": p.traj_id, "t": p.t,
                          "seg": (np.floor(p.lat * 300) * 1000 + np.floor(p.lon * 300)).astype(np.int64) % 100000})
    roads.to_csv(d / "roads.csv", index=False)
    ctx = EvalContext(ds, EvalConfig(window_length=24, visit_context=4, max_eval_samples=40))
    ad = OmniTrajAdapter.pretrain(ctx, train={"epochs": 1, "batch_size": 16, "device": "cpu"}, out=str(d / "o.pt"),
                                  arch=TINY, projection_dim=16, roads_file=str(d / "roads.csv"), min_points=10)
    return ad, ctx, d


def test_omnitraj_embeddings_are_row_pure_normalised_and_reload(omni):
    from mobeval.adapters.omnitraj import OmniTrajAdapter
    ad, ctx, d = omni
    w = ctx.windows["test"]
    e = ad.embed(w)
    assert e.shape == (len(w), 16) and np.allclose(np.linalg.norm(e, axis=1), 1, atol=1e-4)
    ad.invalidate_cache()
    part = ad.embed(w.take(np.arange(3)))
    assert np.allclose(part, e[:3], atol=1e-5)                   # a row does not depend on its batch
    ad2 = OmniTrajAdapter.from_checkpoint(str(d / "o.pt"), device="cpu")
    assert np.allclose(ad2.embed(w), e, atol=1e-5)
    assert ad2.use_road and "road" in ad2.query_modalities()
    for m in ad2.query_modalities():
        assert ad2.embed_query(w.take(np.arange(4)), m).shape == (4, 16)


def test_omnitraj_without_roads_leaves_the_road_encoder_out(omni):
    from mobeval.adapters.omnitraj import OmniTrajAdapter
    _, ctx, _ = omni
    ad = OmniTrajAdapter.pretrain(ctx, train={"epochs": 1, "batch_size": 16, "device": "cpu"}, arch=TINY,
                                  projection_dim=16, min_points=10, loss="paper")
    assert "road" not in ad.net.encoders and "road" not in ",".join(ad.query_modalities())
    with pytest.raises(ValueError, match="road"):
        ad.embed_query(ctx.windows["test"].take(np.arange(2)), "road")


def test_omnitraj_gradient_checkpointing_keeps_the_gradients_and_is_removed_after_training(omni):
    import copy
    import torch
    from mobeval.adapters.omnitraj import FUSIONS, _checkpoint_layers
    ad, ctx, _ = omni
    assert not [m for m in ad.net.modules() if "forward" in m.__dict__]   # pretrain restored every layer
    net = copy.deepcopy(ad.net).train()
    b = ad._tensors(ad._samples(ad._rows(ctx.windows["train"].take(np.arange(8)))))
    b["label"] = torch.arange(8, device=ad.device)

    def grads():
        torch.manual_seed(0)                                   # dropout is on: the same draws in both runs
        net.zero_grad(set_to_none=True)
        net(b, FUSIONS).backward()
        return {n: p.grad.clone() for n, p in net.named_parameters() if p.grad is not None}

    plain = grads()
    assert len(_checkpoint_layers(net)) == 4                   # TINY: one layer per encoder
    ckpt = grads()
    assert plain.keys() == ckpt.keys() and all(torch.allclose(plain[n], ckpt[n], atol=1e-6) for n in plain)


# ------------------------------------------------------------------ retrieval
def test_hausdorff_ranks_match_brute_force():
    from mobeval.tasks import RetrievalTask as R
    rng = np.random.default_rng(0)
    db = (np.cumsum(rng.normal(0, 40, (400, 16, 2)), 1) + rng.uniform(0, 3000, (400, 1, 2))).astype(np.float32)
    q = [db[i] + rng.normal(0, 60, (16, 2)).astype(np.float32) for i in range(40)]
    fast = R._hausdorff_ranks(q, db, np.arange(40))
    brute = [1 + int((R._hausdorff(a, db) <= R._hausdorff(a, db[i:i + 1])[0]).sum()) - 1 for i, a in enumerate(q)]
    assert np.array_equal(fast, np.asarray(brute, float)) and fast.max() > 1


def test_retrieval_ranks_count_ties_against_the_query():
    from mobeval.tasks import RetrievalTask as R
    Q = np.eye(3)
    D = np.array([[1.0, 0, 0], [1.0, 0, 0], [0, 0, 1.0]])
    assert R._ranks(Q, D, np.array([0, 1, 2])).tolist() == [2, 3, 1]


class _Perfect:
    """Embeds a window by its own coordinates: odd and even halves of one window nearly coincide."""
    name, run_tag, capabilities = "Perfect", "default", {"embedding"}

    def embed(self, b):
        return np.column_stack([b.lat.mean(1), b.lon.mean(1), b.lat[:, -1] - b.lat[:, 0]]) * 1e3


def test_retrieval_task_scores_a_near_perfect_embedding():
    from mobeval.context import EvalContext
    from mobeval.tasks import RetrievalTask
    ctx = EvalContext(synthetic_dataset(n_users=6, n_days=4, seed=2),
                      EvalConfig(window_length=16, retrieval_db_size=60, retrieval_queries=20, n_boot=20, eval_seeds=(0,)))
    recs = RetrievalTask().run(_Perfect(), ctx)
    got = {(r.model, r.metric): r.value for r in recs if r.task == "retrieval/odd_even"}
    assert got[("Perfect", "hr@10")] > 0.8 and got[("baseline:random", "mean_rank")] > 5
    assert {"mean_rank", "mrr", "hr@1", "hr@5", "hr@10"} <= {m for _, m in got}


def test_prepare_bbox_keeps_whole_trajectories_inside():
    from mobeval.loaders import prepare
    ds = synthetic_dataset(n_users=6, n_days=3, seed=0)
    p = ds.points
    box = [p.lat.quantile(0.2), p.lat.quantile(0.8), p.lon.quantile(0.2), p.lon.quantile(0.8)]
    out = prepare(ds, bbox=box).points
    assert out.lat.between(box[0], box[1]).all() and out.lon.between(box[2], box[3]).all()
    kept = set(out.traj_id)
    assert all((p[p.traj_id == t].lat.between(box[0], box[1]).all()) for t in kept)


def test_omnitraj_fine_tunes_from_its_checkpoint_and_retrieves_with_encoder_embeddings(omni):
    from mobeval.adapters.omnitraj import OmniTrajAdapter
    from mobeval.tasks import RetrievalTask
    _, ctx, d = omni
    ad = OmniTrajAdapter.pretrain(ctx, train={"epochs": 1, "batch_size": 16, "device": "cpu"}, init_from=str(d / "o.pt"),
                                  arch=TINY, projection_dim=16, min_points=10, embedding="encoder")
    assert ad.embed(ctx.windows["test"].take(np.arange(2))).shape[1] == 32      # encoder space for probes
    ctx.cfg.retrieval_db_size, ctx.cfg.retrieval_queries = 30, 10
    recs = RetrievalTask().run(ad, ctx)                                         # database in the query space
    tasks = {r.task for r in recs}
    assert "retrieval/cross_modal:topology" in tasks and "retrieval/condition:region" in tasks


def test_parallel_segments_keep_the_shorter_route():
    from mobeval.mapmatch import HMMMatcher
    from mobeval.roads import RoadNetwork
    k = 111195 * np.cos(np.radians(41.9))
    a = np.array([[41.9, 12.5], [41.9, 12.5 + 100 / k]])
    detour = np.array([a[0], [41.9 + 100 / 111195, a[0][1]], [41.9 + 100 / 111195, a[1][1]], a[1]])
    G, _ = HMMMatcher(RoadNetwork.from_polylines([a, detour]))._subgraph(np.zeros((2, 2)), 5000)
    assert np.allclose(sorted(G.data), [100, 100], atol=0.5)
