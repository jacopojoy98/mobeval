"""Pictures of what the models actually produce, on a map.

Numbers say how far off a reconstruction is; a picture says in what way - a straight line through a
bend, a jump across town, a trajectory that never leaves its starting point. Two kinds of figure:

  recovery     one figure per sampled test window and masking scheme. One panel per model (and one
               for linear interpolation): the real trajectory, the points the model was shown, the
               points hidden from it, and what it put in their place.
  generation   one figure per seed trajectory. The real trip next to what each model generates
               from the same first points (masked rollout), and, for models that generate visit
               sequences themselves (TrajGPT), a figure of generated sequences next to real ones.

Everything is drawn over a road background, from whichever of these is available:
  * a road network written by `mobeval roads` (network.npz): the road geometry as lines;
  * road sample points (the road_latlon.npy that `mobeval context` writes): dots along the roads;
  * neither: the training GPS points themselves, faintly - vehicle traces draw the roads on their own.

    eval: {visualize_samples: 6, visualize_roads: /home/me/data/roads/milan/network.npz}   # with `evaluate`
    mobeval visualize --config c.yaml [--models A B] [--n 6] [--roads network.npz]    # -> <output_dir>/samples

The arrays behind every figure are saved next to it (samples.npz), so a figure can be redrawn in
another style without the models.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .adapters.base import GENERATION, RECOVERY, TargetGuard
from .data import MobilityDataset, TrajectoryBatch, make_mask
from .geo import haversine_m

log = logging.getLogger("mobeval.visualize")

# One colour per model, the same in every figure (and the same as plot_results.py).
MODEL_ORDER = ["UniTraj-zeroshot", "UniTraj-finetuned", "TransferTraj", "TrajGPT", "CLIPMobility", "OmniTraj"]
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#898781", "#ffffff"
ROAD, ROAD_DOT, BASELINE = "#d3d1c7", "#c9c7bd", "#6f6e69"
MAX_ROAD_SEGMENTS, MAX_BG_POINTS = 60_000, 15_000


def model_color(name: str, names: Sequence[str]) -> str:
    known = MODEL_ORDER + [n for n in names if n not in MODEL_ORDER]
    return PALETTE[min(known.index(name), len(PALETTE) - 1)] if name in known else BASELINE


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "bold",
                         "axes.titlelocation": "left", "text.color": INK, "legend.frameon": False})
    return plt


# --------------------------------------------------------------------------- background
class Background:
    """The road layer under every panel: network lines, road sample points, or GPS traces."""

    def __init__(self, network=None, points: Optional[np.ndarray] = None, kind: str = "none"):
        self.network, self.points, self.kind = network, points, kind
        if network is not None:
            ptr = np.asarray(network.geom_ptr[:-1], np.int64)
            self._box = (np.minimum.reduceat(network.geom_lat, ptr), np.maximum.reduceat(network.geom_lat, ptr),
                         np.minimum.reduceat(network.geom_lon, ptr), np.maximum.reduceat(network.geom_lon, ptr))

    @classmethod
    def load(cls, path: Optional[str], ctx=None) -> "Background":
        """`path`: network.npz from `mobeval roads`, or an (N, 2) .npy of (lat, lon) road points.
        Without a usable path, the train split's GPS points are used."""
        if path:
            p = Path(path)
            try:
                if p.suffix == ".npz":
                    from .roads import RoadNetwork
                    net = RoadNetwork.load(str(p))
                    log.info(f"road background: {len(net):,} segments from {p.name}")
                    return cls(network=net, kind="road network")
                pts = np.load(p)
                if pts.ndim == 2 and pts.shape[1] == 2:
                    log.info(f"road background: {len(pts):,} road points from {p.name}")
                    return cls(points=np.asarray(pts, float), kind="road points")
                log.warning(f"{p}: expected an (N, 2) array of (lat, lon); using GPS traces instead")
            except Exception as e:                                   # noqa: BLE001 - a background must never stop a run
                log.warning(f"could not read the road background {p}: {e!r}; using GPS traces instead")
        if ctx is not None and "train" in ctx.splits:
            pts = ctx.splits["train"].points[["lat", "lon"]].to_numpy(float)
            if len(pts) > 3_000_000:
                pts = pts[np.random.default_rng(0).choice(len(pts), 3_000_000, replace=False)]
            return cls(points=pts, kind="training GPS points")
        return cls()

    def draw(self, ax, box: Tuple[float, float, float, float]):
        la0, la1, lo0, lo1 = box
        if self.network is not None:
            from matplotlib.collections import LineCollection
            b = self._box
            idx = np.flatnonzero((b[1] >= la0) & (b[0] <= la1) & (b[3] >= lo0) & (b[2] <= lo1))
            if len(idx) > MAX_ROAD_SEGMENTS:                         # a whole region in view: thin it
                idx = np.sort(np.random.default_rng(0).choice(idx, MAX_ROAD_SEGMENTS, replace=False))
            net = self.network
            lines = [np.column_stack([net.geom_lon[net.geom_ptr[i]:net.geom_ptr[i + 1]],
                                      net.geom_lat[net.geom_ptr[i]:net.geom_ptr[i + 1]]]) for i in idx]
            ax.add_collection(LineCollection(lines, colors=ROAD, linewidths=0.8, zorder=1))
        elif self.points is not None:
            p = self.points
            m = (p[:, 0] >= la0) & (p[:, 0] <= la1) & (p[:, 1] >= lo0) & (p[:, 1] <= lo1)
            q = p[m]
            if len(q) > MAX_BG_POINTS:
                q = q[np.random.default_rng(0).choice(len(q), MAX_BG_POINTS, replace=False)]
            # fewer, fainter dots when a wide area is in view, so the traces stay a background
            ax.scatter(q[:, 1], q[:, 0], s=2.5 if len(q) < 5000 else 1.5, color=ROAD_DOT,
                       alpha=1.0 if len(q) < 5000 else 0.55, linewidths=0, zorder=1, rasterized=True)


# --------------------------------------------------------------------------- map helpers
def _view(lat, lon, pad: float = 0.25, min_m: float = 400.0) -> Tuple[float, float, float, float]:
    """A box around the given points, padded, at least `min_m` on a side and roughly square."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    ok = np.isfinite(lat) & np.isfinite(lon)
    lat, lon = lat[ok], lon[ok]
    c_lat, c_lon = (lat.min() + lat.max()) / 2, (lon.min() + lon.max()) / 2
    k = max(np.cos(np.radians(c_lat)), 0.05)
    h = max((lat.max() - lat.min()) * 111_195.0, (lon.max() - lon.min()) * 111_195.0 * k, min_m) * (1 + 2 * pad) / 2
    return (c_lat - h / 111_195.0, c_lat + h / 111_195.0, c_lon - h / (111_195.0 * k), c_lon + h / (111_195.0 * k))


def _setup_map(ax, box, bg: Background):
    la0, la1, lo0, lo1 = box
    bg.draw(ax, box)
    ax.set_xlim(lo0, lo1); ax.set_ylim(la0, la1)
    ax.set_aspect(1.0 / max(np.cos(np.radians((la0 + la1) / 2)), 0.05))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#c3c2b7"); s.set_linewidth(0.8)
    # scale bar: a round length close to a quarter of the width
    width_m = haversine_m(la0, lo0, la0, lo1)
    nice = np.array([50, 100, 200, 500, 1000, 2000, 5000, 10_000, 20_000, 50_000, 100_000, 200_000])
    L = float(nice[np.argmin(np.abs(nice - width_m / 4))])
    x0 = lo0 + 0.05 * (lo1 - lo0)
    x1 = x0 + (lo1 - lo0) * L / width_m
    y = la0 + 0.05 * (la1 - la0)
    ax.plot([x0, x1], [y, y], color=INK2, linewidth=2, solid_capstyle="butt", zorder=9)
    ax.text((x0 + x1) / 2, y + 0.015 * (la1 - la0), f"{L / 1000:g} km" if L >= 1000 else f"{L:g} m",
            ha="center", va="bottom", fontsize=7.5, color=INK2, zorder=9,
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1, alpha=0.8))


def _off_map(lat, lon, box) -> float:
    la0, la1, lo0, lo1 = box
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    return float(np.mean((lat < la0) | (lat > la1) | (lon < lo0) | (lon > lo1))) if len(lat) else 0.0


def _dist(v: float) -> str:
    return f"{v / 1000:.1f} km" if v >= 1000 else f"{v:.0f} m"


def _grid(n_panels: int, max_cols: int = 4):
    cols = min(n_panels, max_cols)
    return int(np.ceil(n_panels / cols)), cols


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight", pad_inches=0.15)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- recovery
def pick_windows(batch: TrajectoryBatch, n: int, seed: int = 0) -> np.ndarray:
    """`n` test windows worth looking at: drawn at random (seeded) from the half with the larger
    spatial extent, one per user where possible. A window in which the vehicle barely moves shows
    nothing about a reconstruction."""
    ext = np.hypot((batch.lat.max(1) - batch.lat.min(1)) * 111_195.0,
                   (batch.lon.max(1) - batch.lon.min(1)) * 111_195.0 * np.cos(np.radians(batch.lat.mean(1))))
    cand = np.flatnonzero(ext >= np.median(ext))
    rng = np.random.default_rng(seed)
    order = rng.permutation(cand)
    seen, out = set(), []
    for i in order:                                                 # first pass: distinct users
        if batch.user_id[i] not in seen:
            seen.add(batch.user_id[i]); out.append(int(i))
        if len(out) == n:
            break
    out += [int(i) for i in order if int(i) not in out][:max(0, n - len(out))]
    return np.asarray(out[:n], int)


def collect_recovery(ctx, adapters, n: int = 6, schemes=(("block", 0.5), ("random", 0.5)), seed: int = 0) -> dict:
    """Reconstructions of the same `n` test windows by every model with a recovery capability, and
    by linear interpolation, under each masking scheme."""
    from . import baselines as B
    batch = ctx.windows["test"]
    idx = pick_windows(batch, min(n, len(batch)), seed)
    sub = batch.take(idx)
    out = {"lat": sub.lat, "lon": sub.lon, "t": sub.t, "traj_id": sub.traj_id.astype(str), "schemes": {}}
    for kind, ratio in schemes:
        mask = make_mask(len(sub), sub.length, ratio, kind, seed, ctx.cfg.recovery_keep_endpoints)
        hidden = TargetGuard.hide_masked(sub, mask)
        preds = {}
        if kind != "last":
            preds["linear interpolation"] = tuple(np.asarray(a, float) for a in B.linear_interpolation(hidden, mask))
        else:
            preds["constant velocity"] = tuple(np.asarray(a, float) for a in B.constant_velocity(hidden, mask))
        for ad in adapters:
            if RECOVERY not in ad.capabilities:
                continue
            try:
                la, lo = ad.reconstruct(hidden, mask)
                preds[ad.name] = (np.asarray(la, float), np.asarray(lo, float))
            except Exception as e:                                   # noqa: BLE001 - one model must not lose the others
                log.warning(f"visualize: {ad.name} could not reconstruct ({kind}@{ratio:g}): {e!r}")
        out["schemes"][f"{kind}@{ratio:g}"] = {"mask": mask, "preds": preds}
    return out


def plot_recovery(data: dict, bg: Background, out_dir, model_names: Sequence[str]) -> List[Path]:
    plt = _plt()
    from matplotlib.lines import Line2D
    files = []
    for scheme, s in data["schemes"].items():
        mask, preds = s["mask"], s["preds"]
        kind, ratio = scheme.split("@")
        what = {"block": "one contiguous gap", "random": "scattered points", "last": "the final points",
                "keep_every": "all but every n-th point"}.get(kind, kind)
        for i in range(len(data["lat"])):
            tla, tlo, m = data["lat"][i], data["lon"][i], mask[i]
            box = _view(tla, tlo)
            rows, cols = _grid(len(preds))
            fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.5 * rows + 1.1), squeeze=False)
            fig.subplots_adjust(top=1 - 1.05 / (3.5 * rows + 1.1), bottom=0.02, left=0.01, right=0.99,
                                wspace=0.06, hspace=0.16)
            for k, ax in enumerate(axes.ravel()):
                if k >= len(preds):
                    ax.set_visible(False)
                    continue
                name = list(preds)[k]
                pla, plo = preds[name][0][i], preds[name][1][i]
                col = BASELINE if name not in model_names else model_color(name, model_names)
                _setup_map(ax, box, bg)
                ax.plot(tlo, tla, color=INK, linewidth=1.6, zorder=3)                       # the real trip
                ax.plot(tlo[m], tla[m], linestyle="none", marker="o", markersize=4.5, markerfacecolor=SURFACE,
                        markeredgecolor=INK, markeredgewidth=1.0, zorder=4)                  # hidden, real
                ax.plot(tlo[~m], tla[~m], linestyle="none", marker="o", markersize=3.2, color=INK, zorder=4)
                path_la, path_lo = np.where(m, pla, tla), np.where(m, plo, tlo)              # what the model says
                ax.plot(path_lo, path_la, color=col, linewidth=1.8, zorder=5, alpha=0.95)
                ax.plot(plo[m], pla[m], linestyle="none", marker="o", markersize=4.5, color=col,
                        markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=6)
                err = float(np.mean(haversine_m(pla[m], plo[m], tla[m], tlo[m])))
                ax.set_title(f"{name}  ·  {_dist(err)} mean error", color=INK)
                off = _off_map(pla[m], plo[m], box)
                if off > 0:
                    ax.text(0.97, 0.96, f"{off:.0%} of its points are off this map", transform=ax.transAxes,
                            ha="right", va="top", fontsize=7.5, color=INK2,
                            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=2, alpha=0.85), zorder=10)
            span = haversine_m(tla[0], tlo[0], tla[-1], tlo[-1])
            dur = float(data["t"][i][-1] - data["t"][i][0])
            fig.text(0.01, 0.995, f"Recovery, {what} hidden ({float(ratio):.0%})" if float(ratio) < 1
                     else f"Recovery, {what} hidden", ha="left", va="top", fontsize=13, fontweight="bold")
            fig.text(0.01, 0.995 - 0.30 / fig.get_figheight(),
                     f"Test window {i + 1}: {len(tla)} points over {dur / 60:.0f} min, {_dist(span)} start to end. "
                     f"Background: {bg.kind}.", ha="left", va="top", fontsize=9, color=INK2)
            handles = [Line2D([], [], color=INK, linewidth=1.6, marker="o", markersize=3.2, label="real trajectory, points shown to the model"),
                       Line2D([], [], linestyle="none", marker="o", markersize=4.5, markerfacecolor=SURFACE,
                              markeredgecolor=INK, label="real points hidden from the model"),
                       Line2D([], [], color=MUTED, linewidth=1.8, marker="o", markersize=4.5,
                              markeredgecolor=SURFACE, label="the model's reconstruction (in its colour)")]
            fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005, 0.995 - 0.52 / fig.get_figheight()),
                       ncol=3, fontsize=8, handlelength=2.2, columnspacing=1.6)
            files.append(_save(fig, Path(out_dir) / f"recovery_{kind}{float(ratio) * 100 if float(ratio) < 1 else float(ratio):g}_window{i + 1}.png"))
    return files


# --------------------------------------------------------------------------- generation
def collect_generation(ctx, adapters, n: int = 6, seed: int = 0) -> dict:
    """Rollouts of every reconstruction model from the same `n` real seed trajectories (drawn from
    the train split, as in the generation task), and visit sequences from native generators."""
    from .rollout import _stack, masked_rollout
    cfg = ctx.cfg
    out = {"rollout": None, "native": {}}
    train = ctx.splits["train"]
    roll_models = [a for a in adapters if RECOVERY in a.capabilities and GENERATION not in a.capabilities]
    if roll_models:
        p = train.points
        sizes = p.groupby("traj_id").size()
        need = max(cfg.rollout_seed_points + cfg.rollout_block, 20)
        ok = sizes[sizes >= need].index.to_numpy()
        if len(ok):
            chosen = np.random.default_rng(seed).choice(ok, size=min(n, len(ok)), replace=False)
            sub = MobilityDataset(p[p.traj_id.isin(set(chosen.tolist()))].copy(), "seeds")
            groups = [g for _, g in sub.points.groupby("traj_id", sort=False)]
            la, lo, tt, lens = _stack(groups, cfg.rollout_max_len)
            real = [(la[i, :k], lo[i, :k]) for i, k in enumerate(lens)]
            gen = {}
            for ad in roll_models:
                try:
                    g = masked_rollout(ad, sub, len(groups), seed, seed_points=cfg.rollout_seed_points,
                                       block=cfg.rollout_block, window=cfg.window_length,
                                       noise_m=cfg.rollout_noise_m, max_len=cfg.rollout_max_len)
                    by = {k: v for k, v in g.points.groupby("traj_id", sort=False)}
                    gen[ad.name] = [(by[f"roll{i}"].lat.to_numpy(), by[f"roll{i}"].lon.to_numpy())
                                    for i in range(len(groups))]
                except Exception as e:                               # noqa: BLE001
                    log.warning(f"visualize: rollout of {ad.name} failed: {e!r}")
            out["rollout"] = {"real": real, "gen": gen, "seed_points": cfg.rollout_seed_points}
    for ad in adapters:
        if GENERATION not in ad.capabilities:
            continue
        try:
            ad.reference_staypoints = ctx.staypoints["train"]
            g = ad.generate(train, n, seed)
            seqs = [(v.lat.to_numpy(), v.lon.to_numpy()) for _, v in g.points.groupby("traj_id", sort=False)][:n]
            real = []
            k = max(int(np.median([len(s[0]) for s in seqs]) // 2), 3)          # generators emit 2 points per visit
            for split in ("test", "train"):                                      # real sequences of the same length
                sp = ctx.staypoints.get(split)
                if sp is None or not len(sp):
                    continue
                users = sp.user_id.value_counts()
                users = users[users >= k].index.to_numpy()
                for u in np.random.default_rng(seed).permutation(users)[:n]:
                    v = sp[sp.user_id == u].sort_values("t_arrive").head(k)
                    real.append((v.lat.to_numpy(), v.lon.to_numpy()))
                if real:
                    break
            out["native"][ad.name] = {"gen": seqs, "real": real}
        except Exception as e:                                       # noqa: BLE001
            log.warning(f"visualize: generation by {ad.name} failed: {e!r}")
    return out


def plot_generation(data: dict, bg: Background, out_dir, model_names: Sequence[str]) -> List[Path]:
    plt = _plt()
    files = []
    r = data.get("rollout")
    if r and r["gen"]:
        k = r["seed_points"]
        names = list(r["gen"])
        for i, (rla, rlo) in enumerate(r["real"]):
            box = _view(rla, rlo)
            rows, cols = _grid(1 + len(names))
            fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.5 * rows + 0.95), squeeze=False)
            fig.subplots_adjust(top=1 - 0.9 / (3.5 * rows + 0.95), bottom=0.02, left=0.01, right=0.99,
                                wspace=0.06, hspace=0.16)
            panels = [("real trip", (rla, rlo), INK)] + [(nm, r["gen"][nm][i], model_color(nm, model_names)) for nm in names]
            for j, ax in enumerate(axes.ravel()):
                if j >= len(panels):
                    ax.set_visible(False)
                    continue
                name, (la, lo), col = panels[j]
                _setup_map(ax, box, bg)
                if j:                                                # the real trip, faint, under every model
                    ax.plot(rlo, rla, color="#9a9890", linewidth=1.0, zorder=2)
                ax.plot(lo[k - 1:], la[k - 1:], color=col, linewidth=1.8, zorder=4)
                ax.plot(lo[:k], la[:k], color=INK, linewidth=1.6, marker="o", markersize=3.5, zorder=5)
                ax.plot(lo[-1], la[-1], marker="s", markersize=5, color=col, markeredgecolor=SURFACE, zorder=6)
                length = float(haversine_m(la[:-1], lo[:-1], la[1:], lo[1:]).sum())
                ax.set_title(f"{name}  ·  {_dist(length)} long", color=INK)
                off = _off_map(la, lo, box)
                if off > 0:
                    ax.text(0.97, 0.96, f"{off:.0%} of its points are off this map", transform=ax.transAxes,
                            ha="right", va="top", fontsize=7.5, color=INK2,
                            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=2, alpha=0.85), zorder=10)
            fig.text(0.01, 0.995, "Generation by rollout: the real trip and each model's continuation",
                     ha="left", va="top", fontsize=13, fontweight="bold")
            fig.text(0.01, 0.995 - 0.30 / fig.get_figheight(),
                     f"Seed trajectory {i + 1} ({len(rla)} points). Black dots: the {k} real points every model "
                     f"starts from. Coloured line: what it generates after them (square = its end). Grey: the "
                     f"real trip. Background: {bg.kind}.", ha="left", va="top", fontsize=9, color=INK2)
            files.append(_save(fig, Path(out_dir) / f"generation_rollout_seed{i + 1}.png"))
    for name, d in (data.get("native") or {}).items():
        seqs = [("generated", s) for s in d["gen"]] + [("real", s) for s in d["real"]]
        if not seqs:
            continue
        n_gen = len(d["gen"])
        cols = max(n_gen, len(d["real"]), 1)
        rows = 2 if d["real"] else 1
        fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows + 0.95), squeeze=False)
        fig.subplots_adjust(top=1 - 0.9 / (3.2 * rows + 0.95), bottom=0.02, left=0.01, right=0.99, wspace=0.06, hspace=0.16)
        col = model_color(name, model_names)
        for rr, group in enumerate((d["gen"], d["real"])[:rows]):
            for c in range(cols):
                ax = axes[rr][c]
                if c >= len(group):
                    ax.set_visible(False)
                    continue
                la, lo = group[c]
                box = _view(la, lo, min_m=2000.0)
                _setup_map(ax, box, bg)
                cc = col if rr == 0 else INK
                ax.plot(lo, la, color=cc, linewidth=1.0, alpha=0.7, zorder=3)
                ax.plot(lo, la, linestyle="none", marker="o", markersize=5, color=cc, markeredgecolor=SURFACE,
                        markeredgewidth=0.8, zorder=4)
                ax.plot(lo[0], la[0], marker="o", markersize=8, markerfacecolor="none", markeredgecolor=cc,
                        markeredgewidth=1.4, zorder=5)
                places = len({(round(a, 3), round(b, 3)) for a, b in zip(la, lo)})
                ax.set_title(f"{'generated' if rr == 0 else 'real'} {c + 1}  ·  {places} places", color=INK)
        fig.text(0.01, 0.995, f"Generation by {name}: generated visit sequences (top) and real ones (bottom)"
                 if rows == 2 else f"Generation by {name}: generated visit sequences",
                 ha="left", va="top", fontsize=13, fontweight="bold")
        fig.text(0.01, 0.995 - 0.30 / fig.get_figheight(),
                 f"Each dot is a visit, lines join consecutive visits, the ring marks the first. Every panel has its "
                 f"own scale. Background: {bg.kind}.", ha="left", va="top", fontsize=9, color=INK2)
        files.append(_save(fig, Path(out_dir) / f"generation_native_{name}.png"))
    return files


# --------------------------------------------------------------------------- entry point
def make_samples(ctx, adapters, out_dir, n: int = 6, roads: Optional[str] = None, seed: Optional[int] = None,
                 tasks: Sequence[str] = ("recovery", "generation")) -> List[Path]:
    """Collect and draw the sample figures for `adapters` into `out_dir`. Returns the files written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(ctx.cfg.eval_seeds[0]) if seed is None else seed
    names = [a.name for a in adapters]
    bg = Background.load(roads, ctx)
    files: List[Path] = []
    saved: Dict[str, np.ndarray] = {}
    if "recovery" in tasks and "test" in ctx.windows and any(RECOVERY in a.capabilities for a in adapters):
        rec = collect_recovery(ctx, adapters, n, seed=seed)
        files += plot_recovery(rec, bg, out_dir, names)
        saved.update({"recovery_lat": rec["lat"], "recovery_lon": rec["lon"], "recovery_t": rec["t"]})
        for scheme, s in rec["schemes"].items():
            saved[f"recovery_{scheme}_mask"] = s["mask"]
            for name, (la, lo) in s["preds"].items():
                saved[f"recovery_{scheme}_{name}_lat"], saved[f"recovery_{scheme}_{name}_lon"] = la, lo
    if "generation" in tasks:
        gen = collect_generation(ctx, adapters, n, seed)
        files += plot_generation(gen, bg, out_dir, names)
        r = gen.get("rollout")
        if r:
            for i, (la, lo) in enumerate(r["real"]):
                saved[f"rollout_real{i}_lat"], saved[f"rollout_real{i}_lon"] = la, lo
                for name, seqs in r["gen"].items():
                    saved[f"rollout_{name}{i}_lat"], saved[f"rollout_{name}{i}_lon"] = seqs[i]
    if saved:
        np.savez_compressed(out_dir / "samples.npz", **saved)
        files.append(out_dir / "samples.npz")
    log.info(f"visualize: {len(files)} files in {out_dir} (background: {bg.kind})")
    return files
