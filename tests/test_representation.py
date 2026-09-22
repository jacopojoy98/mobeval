"""Linear probes, user identification and anomaly detection.

These tasks are easy to get subtly wrong in ways that produce plausible-looking numbers: a
probe that cannot decode anything looks the same as a representation with no information, and
a detector scoring cached embeddings of the CLEAN windows looks the same as a model with no
skill. Each test below pins down one of those confusions.
"""
import numpy as np
import pytest

from mobeval import EvalConfig, EvaluationPipeline, synthetic_dataset
from mobeval import anomalies as A
from mobeval.adapters.base import EMBEDDING, TargetGuard
from mobeval.adapters.reference import KinematicReference
from mobeval.baselines import kinematic_knn_score, max_step_score
from mobeval.metrics.classification import candidate_ranking_metrics
from mobeval.metrics.detection import knn_distance, precision_at_k, pr_auc, roc_auc
from mobeval.probes import continuous_probe, location_probe


def _ctx(**kw):
    cfg = EvalConfig(window_length=32, max_eval_samples=300, visit_context=5, eval_seeds=(0,),
                     n_boot=30, recovery_ratios=(0.5,), recovery_kinds=("block",),
                     label_fractions=(1.0,), generation_max_trajectories=20, **kw)
    pipe = EvaluationPipeline(cfg)
    return pipe, pipe.prepare(synthetic_dataset(n_users=40, n_days=12, seed=0))


# --------------------------------------------------------------------- detection metrics
def test_detection_metrics_match_the_reference_implementation():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    for shape in ("continuous", "tied", "constant"):
        y = rng.random(300) < 0.2
        s = rng.normal(y * 1.0, 1.0)
        if shape == "tied":
            s = np.round(s)
        if shape == "constant":
            s = np.zeros(300)
        assert roc_auc(y, s) == pytest.approx(sk.roc_auc_score(y, s))
        assert pr_auc(y, s) == pytest.approx(sk.average_precision_score(y, s))


def test_a_constant_scorer_gets_exactly_the_anomaly_rate():
    """Ties must not be broken in the detector's favour, or a model that outputs one number
    for everything scores far above chance purely through the sort order."""
    y = np.r_[np.ones(20, bool), np.zeros(80, bool)]
    s = np.zeros(100)
    assert roc_auc(y, s) == pytest.approx(0.5)
    assert pr_auc(y, s) == pytest.approx(0.2)
    assert precision_at_k(y, s) == pytest.approx(0.2)


# --------------------------------------------------------------------- location probe
def _codebook(rng, n_codes=40, d=24):
    return rng.normal(size=(n_codes, d))


def test_location_probe_recovers_a_target_the_embedding_encodes():
    """If the answer is linearly present, the probe must find it - otherwise a low score on a
    real model cannot be attributed to the representation."""
    rng = np.random.default_rng(0)
    codes = _codebook(rng)
    tr_cell, te_cell = rng.choice(40, 2000), rng.choice(40, 400)
    emb = lambda c, s: codes[c] + rng.normal(0, s, (len(c), codes.shape[1]))
    p, cand, counts = location_probe(emb(tr_cell, 0.3), tr_cell, emb(te_cell, 0.3), 5000,
                                     top_k=100, seed=0, device="cpu", epochs=50)
    m = candidate_ranking_metrics(p, cand, te_cell, counts)
    assert m["acc@1"].mean() > 0.9, m["acc@1"].mean()


def test_location_probe_reports_chance_when_the_embedding_is_noise():
    """The other half of the previous test: no information must yield no skill, not a
    flattering number."""
    rng = np.random.default_rng(1)
    tr_cell, te_cell = rng.choice(40, 2000), rng.choice(40, 400)
    noise = lambda n: rng.normal(size=(n, 24))
    p, cand, counts = location_probe(noise(2000), tr_cell, noise(400), 5000,
                                     top_k=100, seed=0, device="cpu", epochs=30)
    m = candidate_ranking_metrics(p, cand, te_cell, counts)
    assert m["acc@1"].mean() < 0.12, m["acc@1"].mean()      # 40 equiprobable cells -> 0.025


def test_targets_outside_the_candidate_set_count_as_misses_with_finite_nll():
    rng = np.random.default_rng(2)
    cand_probs = rng.dirichlet(np.ones(10), 200)
    cand = np.arange(10)
    y = np.full(200, 500)                                   # never a candidate
    pop = np.ones(5000) / 5000
    m = candidate_ranking_metrics(cand_probs, cand, y, pop)
    assert m["_coverage"].mean() == 0.0
    assert m["acc@1"].sum() == 0 and m["mrr@20"].sum() == 0
    assert np.isfinite(m["loc_nll"]).all(), "unreachable targets must still get a finite likelihood"


def test_continuous_probe_takes_its_spread_from_validation_not_from_the_test_set():
    rng = np.random.default_rng(3)
    d, w = 12, rng.normal(size=12)
    E = lambda n: rng.normal(size=(n, d))
    f = lambda X, s: np.exp(1.5 + 0.5 * (X @ w) / np.sqrt(d) + rng.normal(0, s, len(X)))
    Etr, Ev, Ete = E(2000), E(500), E(500)
    point, mix = continuous_probe(Etr, f(Etr, 0.4), Ete, Ev, f(Ev, 0.4))
    assert float(mix.stds[0, 0]) == pytest.approx(0.4, abs=0.08)
    assert mix.space == "log" and len(point) == len(Ete)


def test_the_probe_cannot_see_the_target():
    """The probe embeds the visit CONTEXT. Two batches differing only in their target must
    produce byte-identical inputs to the encoder, or the probe is scoring leakage."""
    from mobeval.tasks import visits_as_windows
    _, ctx = _ctx()
    v = ctx.visits["test"]
    hidden = TargetGuard.hide_visits(v)
    w1 = visits_as_windows(hidden)
    shuffled = TargetGuard.hide_visits(v)
    shuffled.tgt_cell = shuffled.tgt_cell[::-1].copy()      # scramble whatever survived hiding
    shuffled.tgt_lat, shuffled.tgt_lon = shuffled.tgt_lat[::-1].copy(), shuffled.tgt_lon[::-1].copy()
    w2 = visits_as_windows(shuffled)
    assert np.array_equal(w1.lat, w2.lat) and np.array_equal(w1.lon, w2.lon)
    assert np.array_equal(w1.t, w2.t)


# --------------------------------------------------------------------- anomalies
@pytest.mark.parametrize("kind", ["detour", "loop"])
def test_step_preserving_anomalies_really_preserve_step_lengths(kind):
    """`detour` and `loop` are the only anomalies that test route understanding rather than a
    speed check. That claim is only true if every displacement magnitude survives."""
    _, ctx = _ctx()
    w = ctx.windows["test"]
    s = A.inject(w, kind, rate=0.3, seed=0)
    d = A.describe(w, s)
    assert d["step_length_deviation"] < 0.02, d          # projection round-trip only
    assert d["max_step_ratio"] == pytest.approx(1.0, abs=0.02)
    assert d["total_distance_ratio"] == pytest.approx(1.0, abs=0.02)
    # The precise claim: speed/distance statistics carry no signal...
    assert roc_auc(s.is_anomalous, max_step_score(s.batch)) < 0.62, "a speed check must be near chance"
    # ...but turning angles do, because the splice inserts a sharp turn. The bar a model has to
    # clear is therefore kinematic_knn, not 0.5, and the docs say so. Pin both ends of that: the
    # baseline must be meaningfully above chance yet far below what it gets on the easy kinds.
    kin = roc_auc(s.is_anomalous, kinematic_knn_score(ctx.windows["train"], s.batch))
    assert 0.55 < kin < 0.85, f"kinematic_knn on {kind} = {kin:.3f}; the documented bar is ~0.60-0.64"


@pytest.mark.parametrize("kind", ["teleport", "speed", "noise"])
def test_kinematic_anomalies_are_caught_by_the_kinematic_baseline(kind):
    """The graded difficulty has to hold in both directions: these must be easy, so that a
    model beating the baseline on them cannot be reported as understanding anything."""
    _, ctx = _ctx()
    s = A.inject(ctx.windows["test"], kind, rate=0.2, seed=0)
    assert roc_auc(s.is_anomalous, max_step_score(s.batch)) > 0.9


def test_injection_does_not_touch_timestamps_or_labels():
    _, ctx = _ctx()
    w = ctx.windows["test"]
    s = A.inject(w, "teleport", rate=0.2, seed=0)
    assert np.array_equal(w.t, s.batch.t), "a detector must not be able to cheat off the time axis"
    assert np.array_equal(w.user_id, s.batch.user_id)
    assert 0 < s.is_anomalous.sum() < len(s), "both classes must be present"


def test_anomalous_windows_get_fresh_embeddings_not_the_clean_cached_ones():
    """Regression. Injection keeps the original timestamps, and adapters cache embeddings on
    (traj_id, t_first, t_last). Reusing the ids handed back the CLEAN embedding, which made
    every detector score at chance with nothing in the logs to show why."""
    torch = pytest.importorskip("torch")
    from mobeval.adapters.unitraj import UniTrajAdapter
    _, ctx = _ctx()
    ad = UniTrajAdapter(arch={"trajectory_length": 32, "embedding_dim": 32, "encoder_layers": 1,
                              "decoder_layers": 1}, device="cpu")
    clean = ctx.windows["test"]
    s = A.inject(clean, "teleport", rate=0.3, seed=0)
    e_clean = ad.embed(clean)                       # populate the cache first, as the pipeline does
    e_anom = ad.embed(s.batch)
    unchanged = np.isclose(e_clean, e_anom).all(1)
    assert not unchanged[s.is_anomalous].any(), "corrupted windows returned the clean embedding"
    assert unchanged[~s.is_anomalous].all(), "untouched windows should still hit the cache"


def test_embedding_knn_detects_teleports_on_a_real_encoder():
    torch = pytest.importorskip("torch")
    from mobeval.adapters.unitraj import UniTrajAdapter
    _, ctx = _ctx()
    ad = UniTrajAdapter.pretrain(ctx, train={"epochs": 4, "batch_size": 64, "device": "cpu"},
                                 arch={"trajectory_length": 32, "embedding_dim": 32,
                                       "encoder_layers": 1, "decoder_layers": 1})
    s = A.inject(ctx.windows["test"], "teleport", rate=0.2, seed=0)
    score = knn_distance(ad.embed(ctx.windows["train"]), ad.embed(s.batch), k=10)
    assert roc_auc(s.is_anomalous, score) > 0.8


# --------------------------------------------------------------------- user identification
def test_user_cohort_is_the_same_for_every_model():
    """Selecting users per model would make the score depend on evaluation order."""
    from mobeval.tasks import UserIdentificationTask
    _, ctx = _ctx(user_id_min_windows=4, user_id_max_users=15)
    a = UserIdentificationTask._cohort(ctx)
    b = UserIdentificationTask._cohort(ctx)
    assert a is not None and all(np.array_equal(x, y) for x, y in zip(a[:4], b[:4]))
    itr, ite, ytr, yte, K = a
    assert K >= 2 and len(itr) == len(ytr) and len(ite) == len(yte)
    assert set(np.unique(yte)) <= set(np.unique(ytr)), "closed-set: test users must be known"


def test_user_identification_always_reports_the_location_control():
    """The mean-location baseline is what makes the number interpretable - mobility identity is
    mostly home location, so it must never be silently absent."""
    from mobeval.tasks import UserIdentificationTask
    _, ctx = _ctx(user_id_min_windows=4, user_id_max_users=12)
    recs = UserIdentificationTask().run(KinematicReference(), ctx)
    assert recs, "task produced nothing"
    names = {r.model for r in recs}
    assert "baseline:mean_location" in names and "baseline:majority" in names
    assert all(r.protocol == "linear_probe" for r in recs)
    assert {r.metric for r in recs} >= {"user_acc@1", "user_macro_f1"}


def test_representation_metrics_stay_out_of_the_prediction_families():
    from mobeval.metrics.registry import get_spec
    assert get_spec("user_acc@1").family == "identity"
    assert get_spec("roc_auc").family == "anomaly"
    assert get_spec("acc@1").family == "location"       # unchanged


def test_probe_results_are_labelled_as_probes():
    """A probe number must never be mistaken for a native capability in a report."""
    pipe, ctx = _ctx()
    store = pipe.run([KinematicReference()], ctx)
    probes = [r for r in store.records if r.protocol == "linear_probe"]
    assert probes, "no probe results were produced"
    assert all("/linear_probe" in r.task or r.task.startswith("user_identification")
               for r in probes if not r.model.startswith("baseline:"))


# --------------------------------------------------------------------- review regressions
def test_embed_must_be_a_pure_function_of_each_row():
    """Per-batch standardisation put train and test in different spaces. In the anomaly task
    the injected anomalies then set the test batch's scale and hid themselves: KinematicRef
    scored 0.73 on teleports it actually separates perfectly."""
    from mobeval.baselines import handcrafted_features
    _, ctx = _ctx()
    w = ctx.windows["test"]
    assert np.allclose(handcrafted_features(w)[:10], handcrafted_features(w.take(np.arange(10))))
    ad = KinematicReference()
    full = ad.embed(w)
    assert np.allclose(full[:10], ad.embed(w.take(np.arange(10))))
    # the end-to-end consequence
    s = A.inject(w, "teleport", rate=0.2, seed=0)
    ad2 = KinematicReference()
    score = knn_distance(ad2.embed(ctx.windows["train"]), ad2.embed(s.batch), k=10)
    assert roc_auc(s.is_anomalous, score) > 0.95


def test_non_finite_scores_do_not_shrink_the_evaluation_set():
    """Dropping NaN scores evaluated the model on fewer samples than its baseline while the
    record still claimed the full n, and shrank k in precision@k to the surviving anomalies."""
    from mobeval.metrics.detection import detection_metrics
    rng = np.random.default_rng(0)
    y = rng.random(300) < 0.1
    s = rng.normal(y * 2.0, 1.0)
    s[rng.choice(300, 40, replace=False)] = np.nan
    _, sets = detection_metrics(y, s)
    idx = np.arange(300)
    # k is the full anomaly count, and the NaN rows are ranked as the most normal
    assert sets["precision@k"](idx) == pytest.approx(precision_at_k(y, np.nan_to_num(s, nan=s[np.isfinite(s)].min() - 1)))
    assert np.isfinite(sets["roc_auc"](idx)) and np.isfinite(sets["pr_auc"](idx))


def test_the_probe_point_estimate_is_the_median_like_every_other_predictor():
    """Native adapters and the train_marginal baseline are scored on the mixture MEDIAN. Handing
    the probe's row the log-normal MEAN instead inflated its MAE by exp(sigma^2/2) - 38% at
    sigma=0.8 - purely as a bookkeeping artefact."""
    rng = np.random.default_rng(0)
    d, w = 8, rng.normal(size=8)
    E = lambda n: rng.normal(size=(n, d))
    f = lambda X: np.exp(1.0 + 0.3 * (X @ w) / np.sqrt(d) + rng.normal(0, 0.8, len(X)))
    Etr, Ev, Ete = E(3000), E(600), E(600)
    point, mix = continuous_probe(Etr, f(Etr), Ete, Ev, f(Ev))
    assert np.allclose(point, np.exp(mix.means[:, 0])), "point estimate must be the median, exp(m)"
    assert np.allclose(point, mix.median(), rtol=1e-6)


def test_the_continuous_probe_absorbs_a_train_to_validation_shift():
    """A centred std hides a systematic offset inside a small sigma, and the metrics then report
    over-confidence when the real defect is bias - exactly what a chronological split produces."""
    rng = np.random.default_rng(1)
    d = 6
    E = lambda n: rng.normal(size=(n, d))
    Etr, Ev, Ete = E(2000), E(500), E(500)
    ytr = np.exp(1.0 + rng.normal(0, 0.2, 2000))
    yv = np.exp(1.0 + 0.5 + rng.normal(0, 0.2, 500))      # validation runs 0.5 higher in log space
    point, mix = continuous_probe(Etr, ytr, Ete, Ev, yv)
    assert float(np.mean(mix.means)) == pytest.approx(1.5, abs=0.15), "the offset must be corrected"
    assert float(mix.stds[0, 0]) == pytest.approx(0.2, abs=0.06), "the spread must stay honest"


def test_a_non_finite_sigma_never_reaches_the_mixture():
    rng = np.random.default_rng(2)
    E = lambda n: rng.normal(size=(n, 5))
    Etr, Ev, Ete = E(200), E(50), E(50)
    yv = np.full(50, np.nan)
    point, mix = continuous_probe(Etr, np.exp(rng.normal(0, 1, 200)), Ete, Ev, yv)
    assert np.isfinite(mix.stds).all() and np.isfinite(point).all()


def test_visit_context_ids_are_unique_across_splits():
    """Ids restarting at 0 in every split can collide on coarsely quantised data, and a test
    context would then be handed a training sample's cached embedding."""
    from mobeval.tasks import visits_as_windows
    _, ctx = _ctx()
    ids = set()
    for name in ("train", "val", "test"):
        v = ctx.visits.get(name)
        if v is None or not len(v):
            continue
        new = set(visits_as_windows(v, tag=name).traj_id.tolist())
        assert not (ids & new), f"{name} reuses ids from an earlier split"
        ids |= new


def test_user_identification_reports_both_position_controls():
    from mobeval.tasks import UserIdentificationTask
    _, ctx = _ctx(user_id_min_windows=4, user_id_max_users=12)
    recs = UserIdentificationTask().run(KinematicReference(), ctx)
    names = {r.model for r in recs}
    assert {"baseline:mean_location", "baseline:mean_location_gbdt"} <= names


def test_the_location_probe_reports_its_own_ceiling():
    """acc@1 cannot exceed the share of targets the candidate set can reach, so that share has
    to appear next to it rather than only in a job log."""
    pipe, ctx = _ctx(probe_top_k=5)
    store = pipe.run([KinematicReference()], ctx)
    cov = [r for r in store.records if r.metric == "probe_coverage"]
    assert cov, "probe_coverage was not emitted"
    acc = [r for r in store.records if r.metric == "acc@1" and r.protocol == "linear_probe"
           and not r.model.startswith("baseline:")]
    assert acc and acc[0].value <= cov[0].value + 1e-9, "accuracy exceeded the reachable share"
