"""TransferTraj: the context coverage check logged before training."""
import logging

import numpy as np
import pytest

from mobeval import EvalConfig, synthetic_dataset


@pytest.fixture(scope="module")
def ctx():
    from mobeval.context import EvalContext
    return EvalContext(synthetic_dataset(n_users=4, n_days=3, seed=0), EvalConfig(window_length=16))


def _context(tmp_path, latlon, name):
    np.save(tmp_path / f"{name}_latlon.npy", latlon)
    np.save(tmp_path / f"{name}_embed.npy", np.random.default_rng(0).normal(size=(len(latlon), 8)).astype(np.float32))
    return {"poi_embed": str(tmp_path / f"{name}_embed.npy"), "poi_latlon": str(tmp_path / f"{name}_latlon.npy")}


def test_context_coverage_counts_points_with_context_within_the_radius(ctx, tmp_path, caplog):
    from mobeval.adapters.transfertraj import TransferTrajAdapter
    w = ctx.windows["train"]
    pts = np.column_stack([w.lat.ravel(), w.lon.ravel()])
    pts = pts[np.isfinite(pts).all(1)]
    centre = (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
    on_route = _context(tmp_path, pts, "near")                            # POIs on the GPS points themselves
    far_away = _context(tmp_path, pts + [1.0, 1.0], "far")                # the same, ~140 km off
    near = TransferTrajAdapter(arch={"poi_dist": 10_000}, center=centre, context=on_route, device="cpu")
    cov = near.context_coverage(w)["poi"]
    assert cov["radius_m"] == pytest.approx(100.0) and cov["share_with_any"] == 1.0
    with caplog.at_level(logging.WARNING, logger="mobeval.adapters.transfertraj"):
        far = TransferTrajAdapter(arch={"poi_dist": 10_000}, center=centre, context=far_away, device="cpu")
        assert far.context_coverage(w)["poi"]["share_with_any"] == 0.0
    assert "nearly empty" in caplog.text
    assert TransferTrajAdapter(center=centre, device="cpu").context_coverage(w) == {}        # no context files
