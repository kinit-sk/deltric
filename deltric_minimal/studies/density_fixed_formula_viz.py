#!/usr/bin/env python3
"""Visual + metric audit of the *fixed* Dijkstra reclamation formula, with density.

Scope: only the numbered study datasets carrying ground-truth anomalies
(``01_``, ``02_``, ``04_``, ``10_``, ``11_``, ``14_``  ``*__anom10.npz``).

--------------------------------------------------------------------------
WHAT "FIXED FORMULA" MEANS HERE
--------------------------------------------------------------------------
The shipped stage (``_assign_anomalies_dijkstra``) accumulates a penalized
cost into ``dist`` but never gates on it -- the only stop condition is a
*single-edge* test on raw length:

    weight = length * (length / avg_c) ** p     # accumulated, never bounded
    stop   if length > stop_ratio * avg_c       # raw, per-edge, unbounded reach

so ``p`` is inert (monotone transform of length -> same pop order, same
admissible set) and one wrongly claimed anomaly becomes a launchpad for the
next. The fix (from ``anomaly_cost_gate_study.py``) measures cost in
"typical hop" units and budgets the *whole path*:

    r    = (length * rho(v, c) ** q) / avg_c    # 1.0 == typical edge for c
    cost = r ** (1 + p)                         # this hop's price, in hops
    claim v iff sum(cost along path) <= reach

with a single global heap (true multi-source Dijkstra) instead of one claim
per cluster per round. The density term is

    rho(v, c) = core_distance[v] / median(core_distance over members of c)

``core_distance`` = mean distance to the k nearest neighbours in *original*
space. rho > 1 means "v sits in a void relative to the inside of c" -- the
signal UMAP destroys and projected length alone cannot see.

--------------------------------------------------------------------------
OUTPUTS (per dataset, into --out)
--------------------------------------------------------------------------
  <stem>__panels.png    6-panel projected-space comparison, incl. an error map
  <stem>__frontier.gif  the fixed+density gate claiming points in cost order
  <stem>__density.png   rho separation: true anomalies vs real points
  metrics.csv           every variant x every dataset
  summary__f1.png       per-dataset F1 / ARI_all bars across variants

Usage:
    python studies/density_fixed_formula_viz.py                 # everything
    python studies/density_fixed_formula_viz.py --no-gif        # plots only
    python studies/density_fixed_formula_viz.py --datasets 01 11
"""

from __future__ import annotations

import argparse
import csv
import heapq
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
from sklearn.metrics import (
    adjusted_rand_score, f1_score, precision_score, recall_score,
)
from sklearn.preprocessing import StandardScaler

_MINIMAL_DIR = Path(__file__).resolve().parents[1]
_DEFAULT_DATA = _MINIMAL_DIR / "data"
_DEFAULT_OUT = _MINIMAL_DIR / "studies" / "results" / "density_fixed_viz"
sys.path.insert(0, str(_MINIMAL_DIR.parent))

from deltric_minimal.utils_component_growth import (  # noqa: E402
    _assign_anomalies_dijkstra,
    _knn_core_distance,
    cluster_tri,
)
from deltric_minimal.studies.anomaly_cost_gate_study import (  # noqa: E402
    BASE_CONFIG,
    DENSITY_K,
    KIND_NAMES,
    _prep,
    _density_ratio_fn,
    cost_gate,
    edge_gate,
)

# p / reach / q defaults are the ones the cost-gate study landed on.
PENALTY_POWER = 3.0
REACH = 25.0
DENSITY_Q = 4.0

UNCLAIMED = (0.74, 0.74, 0.74, 0.85)


# ---------------------------------------------------------------------------
# frozen graph (with X_proj, which the cost-gate cache does not keep)
# ---------------------------------------------------------------------------

def numbered_anomaly_datasets(data_dir: Path) -> list[Path]:
    """``text_support_tickets__*.npz`` -- real-embedding stems with ground-truth noise."""
    out = []
    for path in sorted(data_dir.glob("text_support_tickets__*.npz")):
        with np.load(path, allow_pickle=False) as npz:
            if "y" in npz.files and np.any(np.asarray(npz["y"]) < 0):
                out.append(path)
    return out


def frozen_graph(path: Path, cache_dir: Path, force: bool = False) -> dict:
    """cluster_tri with reassignment OFF, cached. Every variant replays on this."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{path.stem}.npz"
    if cached.exists() and not force:
        with np.load(cached, allow_pickle=False) as npz:
            return {k: npz[k] for k in npz.files} | {"dataset": path.stem}

    npz = np.load(path, allow_pickle=False)
    X = StandardScaler().fit_transform(np.asarray(npz["X"], dtype=np.float64))
    y = np.asarray(npz["y"]).astype(np.int64)
    kind = (np.asarray(npz["anomaly_kind"]) if "anomaly_kind" in npz.files
            else np.zeros(len(y), dtype=np.int8))
    started = time.perf_counter()
    cluster_tri(X, **BASE_CONFIG, component_growth_anomaly_reassign=False)
    state = cluster_tri.last_component_growth
    payload = dict(
        y=y, kind=kind.astype(np.int8),
        n=np.int64(len(X)), d=np.int64(X.shape[1]),
        labels=np.asarray(state["merged_labels"], dtype=np.int64),
        edge_keys=np.asarray(state["edge_keys"], dtype=np.int64),
        sizes=np.asarray(state["projected_edge_sizes"], dtype=np.float64),
        X_proj=np.asarray(state["X_proj"], dtype=np.float64),
        core=_knn_core_distance(X, DENSITY_K),
    )
    np.savez_compressed(cached, **payload)
    print(f"    built frozen graph in {time.perf_counter() - started:.0f}s", flush=True)
    return payload | {"dataset": path.stem}


def entry_of(graph: dict) -> dict:
    """The dict shape the cost-gate study's gate functions expect."""
    return {"labels": graph["labels"], "edge_keys": graph["edge_keys"],
            "sizes": graph["sizes"], "core": graph["core"], "n": int(graph["n"])}


# ---------------------------------------------------------------------------
# traced cost gate (identical arithmetic to cost_gate, but records claim order)
# ---------------------------------------------------------------------------

def cost_gate_traced(entry, p=PENALTY_POWER, reach=REACH, q=DENSITY_Q, clip=False):
    """``cost_gate`` plus the pop order, so the frontier can be animated.

    Claims come out in increasing accumulated cost, which is exactly the
    quantity the reach budget bounds -- so playing them back in order *is* a
    playback of the budget being spent.
    """
    labels, edge_keys, sizes, n_edges, n_clusters, adjacency, avg = _prep(
        entry["labels"], entry["edge_keys"], entry["sizes"], entry["n"])
    if n_clusters == 0:
        return labels.copy(), []
    ratio = _density_ratio_fn(labels, n_clusters, entry.get("core"), q, clip)

    def relative(index, node, cluster):
        return (sizes[index] * ratio(node, cluster)) / avg[cluster]

    owner = labels.copy()
    heap = []
    for cluster in range(n_clusters):
        for node in np.flatnonzero(labels == cluster):
            for neighbor, index in adjacency[int(node)]:
                if labels[neighbor] == cluster:
                    continue
                cost = relative(index, int(neighbor), cluster) ** (1.0 + p)
                if cost <= reach:
                    heapq.heappush(heap, (cost, int(neighbor), cluster, index))
    trace = []
    while heap:
        dist, node, cluster, index = heapq.heappop(heap)
        if owner[node] != -1:
            continue
        owner[node] = cluster
        trace.append((int(node), int(cluster), float(dist), int(index)))
        for neighbor, next_index in adjacency[node]:
            if owner[neighbor] != -1:
                continue
            nxt = dist + relative(next_index, int(neighbor), cluster) ** (1.0 + p)
            if nxt <= reach:
                heapq.heappush(heap, (nxt, int(neighbor), cluster, next_index))
    return owner, trace


def edge_gate_traced(entry, p=2.0, stop_ratio=3.0):
    """The shipped gate, recording claim order so it can be animated beside the fix.

    Bit-exact with ``edge_gate`` (which ``anomaly_cost_gate_study --validate``
    proves is bit-exact with production); the only addition is the trace.
    Claims are emitted round by round, one per cluster per round -- the pacing
    that makes the shipped stage a hop-paced flood fill rather than a Dijkstra.
    """
    labels, edge_keys, sizes, n_edges, n_clusters, adjacency, avg = _prep(
        entry["labels"], entry["edge_keys"], entry["sizes"], entry["n"])
    if n_clusters == 0:
        return labels.copy(), []
    n_points = entry["n"]
    owner = labels.copy()
    heaps = [[] for _ in range(n_clusters)]
    for cluster in range(n_clusters):
        for node in np.flatnonzero(labels == cluster):
            for neighbor, index in adjacency[int(node)]:
                if labels[neighbor] == cluster:
                    continue
                length = sizes[index]
                heapq.heappush(
                    heaps[cluster],
                    (length * (length / avg[cluster]) ** p, int(neighbor), index))
    active = {c for c in range(n_clusters) if heaps[c]}
    trace, rounds = [], 0
    while rounds < 4 * n_points + 16 and active:
        candidate, dead = {}, []
        for root in list(active):
            heap = heaps[root]
            while heap:
                dist, node, index = heap[0]
                if owner[node] != -1:
                    heapq.heappop(heap)
                    continue
                if sizes[index] > stop_ratio * avg[root]:
                    heapq.heappop(heap)
                    continue
                candidate[root] = (dist, node, index)
                break
            if not heap:
                dead.append(root)
        for root in dead:
            active.discard(root)
        if not candidate:
            break
        proposals = {}
        for root, (dist, node, index) in candidate.items():
            current = proposals.get(node)
            if current is None or dist < current[0]:
                proposals[node] = (dist, root, index)
        for node, (dist, root, index) in proposals.items():
            if owner[node] != -1:
                continue
            heapq.heappop(heaps[root])
            owner[node] = root
            trace.append((int(node), int(root), float(dist), int(index)))
            for neighbor, next_index in adjacency[node]:
                if owner[neighbor] != -1:
                    continue
                length = sizes[next_index]
                heapq.heappush(
                    heaps[root],
                    (dist + length * (length / avg[root]) ** p, int(neighbor), next_index))
        rounds += 1
    return owner, trace


def old_gate_with_density(entry, p=2.0, stop_ratio=3.0, q=0.0, clip=False):
    """The shipped stage, optionally with the density term bolted on (production API)."""
    labels, _ = _assign_anomalies_dijkstra(
        entry["labels"], entry["edge_keys"], entry["sizes"], entry["n"],
        penalty_power=p, stop_ratio=stop_ratio,
        core_distance=entry["core"] if q else None,
        density_power=q, density_clip=clip,
    )
    return labels, {}


VARIANTS = [
    ("noreassign", lambda e: (np.asarray(e["labels"]).copy(), {}),
     "stage disabled (control)"),
    ("old_p2_s3", lambda e: (edge_gate(e, p=2.0, stop_ratio=3.0)[0], {}),
     "shipped: per-edge gate, unbounded reach"),
    ("old_p2_s3_q4", lambda e: old_gate_with_density(e, q=4.0),
     "shipped gate + density (old formula)"),
    ("fix_p3_r25_q0", lambda e: (cost_gate(e, p=3.0, reach=25.0, q=0.0)[0], {}),
     "fixed cost gate, no density"),
    ("fix_p3_r25_q1", lambda e: (cost_gate(e, p=3.0, reach=25.0, q=1.0)[0], {}),
     "fixed cost gate + density q=1"),
    ("fix_p3_r25_q2", lambda e: (cost_gate(e, p=3.0, reach=25.0, q=2.0)[0], {}),
     "fixed cost gate + density q=2"),
    ("fix_p3_r25_q4", lambda e: (cost_gate(e, p=3.0, reach=25.0, q=4.0)[0], {}),
     "fixed cost gate + density q=4"),
    ("fix_p3_r60_q4", lambda e: (cost_gate(e, p=3.0, reach=60.0, q=4.0)[0], {}),
     "fixed cost gate + density q=4, loose reach"),
]
HEADLINE = "fix_p3_r25_q4"
PANEL_VARIANTS = ["noreassign", "old_p2_s3", "fix_p3_r25_q0", HEADLINE]


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def metrics_for(graph, assigned) -> dict:
    y = graph["y"]
    true_anomaly = y < 0
    real = ~true_anomaly
    predicted = assigned < 0
    n_noise_in = int((graph["labels"] < 0).sum())
    claimed = (graph["labels"] < 0) & (assigned >= 0)
    out = {
        "precision": float(precision_score(true_anomaly, predicted, zero_division=0)),
        "recall": float(recall_score(true_anomaly, predicted, zero_division=0)),
        "f1": float(f1_score(true_anomaly, predicted, zero_division=0)),
        "ari_all": float(adjusted_rand_score(y, assigned)),
        "ari_real": float(adjusted_rand_score(y[real], assigned[real])),
        "claims": int(claimed.sum()),
        "claims_good": int((claimed & real).sum()),
        "claims_bad": int((claimed & true_anomaly).sum()),
        "claim_purity": float((claimed & real).sum() / max(int(claimed.sum()), 1)),
        "n_noise_in": n_noise_in,
        "n_pred_anomaly": int(predicted.sum()),
    }
    for code, name in KIND_NAMES.items():
        mask = graph["kind"] == code
        if mask.any():
            out[f"recall_{name}"] = float((assigned[mask] < 0).mean())
    return out


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

def _palette(labels, cmap):
    clusters = np.unique(labels[labels >= 0])
    return {int(c): cmap(i % 20) for i, c in enumerate(clusters)}


def _colors(owner, palette):
    colors = np.full((len(owner), 4), UNCLAIMED, dtype=float)
    for cluster, color in palette.items():
        colors[owner == cluster] = color
    return colors


def _style_axis(ax, X_proj):
    ax.set_xlim(X_proj[:, 0].min() - 1, X_proj[:, 0].max() + 1)
    ax.set_ylim(X_proj[:, 1].min() - 1, X_proj[:, 1].max() + 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])


def panel_figure(graph, results, out_path: Path) -> None:
    """6 panels: truth, then 4 variants, then the headline variant's error map."""
    X_proj = graph["X_proj"][:, :2]
    y = graph["y"]
    true_anomaly = y < 0
    cmap = plt.get_cmap("tab20")
    truth_palette = _palette(y, cmap)
    palette = _palette(graph["labels"], cmap)

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 11))
    axes = axes.ravel()

    ax = axes[0]
    ax.scatter(*X_proj[~true_anomaly].T, s=5, c=_colors(y, truth_palette)[~true_anomaly],
               linewidths=0, alpha=0.85)
    ax.scatter(*X_proj[true_anomaly].T, s=30, marker="*", c="black",
               linewidths=0, alpha=0.9)
    ax.set_title(f"ground truth  ({int(true_anomaly.sum())} anomalies, black stars)",
                 fontsize=10)
    _style_axis(ax, X_proj)

    for slot, variant in enumerate(PANEL_VARIANTS, start=1):
        assigned, m = results[variant]
        ax = axes[slot]
        colors = _colors(assigned, palette)
        ax.scatter(*X_proj[~true_anomaly].T, s=5, c=colors[~true_anomaly],
                   linewidths=0, alpha=0.85)
        # true anomalies drawn as stars, coloured by whoever (wrongly) claimed them
        ax.scatter(*X_proj[true_anomaly].T, s=34, marker="*", c=colors[true_anomaly],
                   edgecolors="black", linewidths=0.45, alpha=0.95)
        # ring every anomaly this variant reclaimed -- the errors, at a glance
        wrong = true_anomaly & (assigned >= 0) & (graph["labels"] < 0)
        if wrong.any():
            ax.scatter(*X_proj[wrong].T, s=95, marker="o", facecolors="none",
                       edgecolors="#d1495b", linewidths=1.0)
        ax.set_title(
            f"{variant}\nF1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}  "
            f"ARI_all={m['ari_all']:.3f}\nclaims={m['claims']} "
            f"({m['claims_good']} real / {m['claims_bad']} anomaly, "
            f"purity={m['claim_purity']:.2f})",
            fontsize=9)
        _style_axis(ax, X_proj)

    assigned, m = results[HEADLINE]
    was_noise = graph["labels"] < 0
    claimed = was_noise & (assigned >= 0)
    ax = axes[5]
    ax.scatter(*X_proj[~was_noise].T, s=4, c="#dddddd", linewidths=0, alpha=0.7)
    groups = [
        (was_noise & ~true_anomaly & ~claimed, "#e8a33d", 14, "o",
         "real pt left as noise (miss)"),
        (was_noise & true_anomaly & ~claimed, "#5b8dd6", 16, "*",
         "anomaly kept as noise (correct)"),
        (claimed & ~true_anomaly, "#3fa66a", 14, "o", "real pt reclaimed (correct)"),
        (claimed & true_anomaly, "#d1495b", 34, "X", "anomaly reclaimed (error)"),
    ]
    for mask, color, size, marker, _label in groups:
        if mask.any():
            ax.scatter(*X_proj[mask].T, s=size, marker=marker, c=color,
                       linewidths=0, alpha=0.9)
    ax.legend(handles=[Line2D([], [], marker=mk, color="none", markerfacecolor=c,
                              markeredgecolor="none", markersize=7, label=f"{lb} ({int(mask.sum())})")
                       for mask, c, _s, mk, lb in groups],
              fontsize=7.5, loc="best", framealpha=0.9)
    ax.set_title(f"{HEADLINE}: what the reclamation did to the noise set", fontsize=10)
    _style_axis(ax, X_proj)

    fig.suptitle(
        f"{graph['dataset']}   n={int(graph['n'])}  d={int(graph['d'])}   "
        f"fixed cost gate (p={PENALTY_POWER:g}, reach={REACH:g}) with original-space "
        f"density (q={DENSITY_Q:g}, k={DENSITY_K})",
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def density_figure(graph, out_path: Path) -> None:
    """Is rho actually separating? If not, no q can help on this dataset."""
    labels = graph["labels"]
    core = np.asarray(graph["core"], dtype=np.float64)
    n_clusters = int(labels.max()) + 1 if np.any(labels >= 0) else 0
    dref = np.full(max(n_clusters, 1), float(np.median(core)))
    for cluster in range(n_clusters):
        members = core[labels == cluster]
        if len(members):
            dref[cluster] = float(np.median(members))
    dref = np.maximum(dref, 1e-12)

    # For each still-noise point, rho against the cluster owning its nearest
    # labelled graph neighbour -- the cluster that would actually try to claim it.
    _l, edge_keys, sizes, _ne, _nc, adjacency, _avg = _prep(
        labels, graph["edge_keys"], graph["sizes"], int(graph["n"]))
    rho, is_anom = [], []
    true_anomaly = graph["y"] < 0
    for node in np.flatnonzero(labels < 0):
        best = None
        for neighbor, index in adjacency[int(node)]:
            if labels[neighbor] < 0:
                continue
            if best is None or sizes[index] < best[0]:
                best = (sizes[index], int(labels[neighbor]))
        if best is None:
            continue
        rho.append(core[node] / dref[best[1]])
        is_anom.append(bool(true_anomaly[node]))
    rho = np.asarray(rho)
    is_anom = np.asarray(is_anom, dtype=bool)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    if len(rho):
        bins = np.linspace(0, max(3.0, float(np.percentile(rho, 99))), 45)
        ax.hist(rho[~is_anom], bins=bins, alpha=0.65, color="#3fa66a",
                label=f"real points ({int((~is_anom).sum())})")
        ax.hist(rho[is_anom], bins=bins, alpha=0.65, color="#d1495b",
                label=f"true anomalies ({int(is_anom.sum())})")
        ax.axvline(1.0, color="black", lw=1, ls="--")
        # rho past which a single *typical-length* edge already blows the whole
        # budget: rho ** (q(1+p)) > reach. Everything right of the line is
        # unreachable at that q no matter how short its projected edge is.
        for q, color in ((1.0, "#5b8dd6"), (2.0, "#e8a33d"), (4.0, "#7a4fa3")):
            cut = REACH ** (1.0 / (q * (1.0 + PENALTY_POWER)))
            if cut <= bins[-1]:
                ax.axvline(cut, color=color, lw=1.4, ls=":",
                           label=fr"$q={q:g}$ cutoff $\rho={cut:.2f}$")
    ax.set_xlabel(r"$\rho$ = core_dist(v) / median core_dist inside claiming cluster")
    ax.set_ylabel("noise points")
    ax.set_title("density ratio of the points the stage may claim", fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[1]
    grid = np.linspace(0.3, 3.0, 200)
    for q in (0.0, 2.0, 4.0):
        ax.plot(grid, grid ** (q * (1.0 + PENALTY_POWER)),
                label=fr"$q={q:g}$  (exponent {q * (1 + PENALTY_POWER):g})")
    ax.axhline(REACH, color="black", lw=1, ls="--", label=f"reach={REACH:g}")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"cost multiplier  $\rho^{\,q(1+p)}$")
    ax.set_title(f"what q does to one hop's price (p={PENALTY_POWER:g})", fontsize=10)
    ax.legend(fontsize=8)

    fig.suptitle(f"{graph['dataset']}  --  density signal and its price", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


GIF_TRACKS = [
    ("old_p2_s3", "SHIPPED  --  per-edge gate, unbounded reach",
     lambda e: edge_gate_traced(e, p=2.0, stop_ratio=3.0)),
    ("fix_p3_r25_q0", f"FIXED cost gate  (p={PENALTY_POWER:g}, reach={REACH:g}, q=0)",
     lambda e: cost_gate_traced(e, p=PENALTY_POWER, reach=REACH, q=0.0)),
    (HEADLINE, f"FIXED + density  (q={DENSITY_Q:g})",
     lambda e: cost_gate_traced(e, p=PENALTY_POWER, reach=REACH, q=DENSITY_Q)),
]


def frontier_gif(graph, entry, out_path: Path, frames=48, fps=7) -> None:
    """Three gates claiming points side by side, replayed in their own order.

    Each panel advances one claim per step of a shared counter, so the panels
    are synchronised on *number of points claimed* -- the fair axis, since the
    gates disagree about how many points may be claimed at all. A panel that
    finishes early simply freezes, which is the point: that gate stopped.

    Red rings mark claims that took a ground-truth anomaly (the errors this
    stage exists to avoid); the counter under each panel tracks them live.
    """
    X_proj = graph["X_proj"][:, :2]
    labels = graph["labels"]
    edge_keys = graph["edge_keys"]
    true_anomaly = graph["y"] < 0
    was_noise = labels < 0
    cmap = plt.get_cmap("tab20")
    palette = _palette(labels, cmap)
    base_colors = _colors(labels, palette)
    noise_idx = np.flatnonzero(was_noise & ~true_anomaly)
    anom_idx = np.flatnonzero(was_noise & true_anomaly)

    tracks = []
    for name, caption, fn in GIF_TRACKS:
        _owner, trace = fn(entry)
        tracks.append({
            "name": name, "caption": caption,
            "nodes": np.array([t[0] for t in trace], dtype=np.int64),
            "clusters": np.array([t[1] for t in trace], dtype=np.int64),
            "costs": np.array([t[2] for t in trace], dtype=np.float64),
            "edges": np.array([t[3] for t in trace], dtype=np.int64),
            "n": len(trace),
        })
    max_claims = max(t["n"] for t in tracks) or 1
    steps = ([0] * 5
             + list(np.unique(np.linspace(0, max_claims, frames).astype(int)))
             + [max_claims] * 8)

    fig, axes = plt.subplots(1, 3, figsize=(19.5, 7.4))
    artists = []
    for ax, track in zip(axes, tracks):
        ax.add_collection(LineCollection(
            X_proj[edge_keys] if len(edge_keys) else np.empty((0, 2, 2)),
            colors="#c9c9c9", linewidths=0.3, alpha=0.25, zorder=1))
        claim_edges = LineCollection([], linewidths=1.2, alpha=0.85, zorder=2)
        ax.add_collection(claim_edges)
        ax.scatter(*X_proj[~was_noise].T, s=5, c=base_colors[~was_noise],
                   linewidths=0, alpha=0.8, zorder=3)
        scatter_noise = ax.scatter([], [], s=15, linewidths=0, alpha=0.9, zorder=4)
        scatter_anom = ax.scatter([], [], s=44, marker="*", edgecolors="black",
                                  linewidths=0.5, alpha=0.95, zorder=5)
        wrong = ax.scatter([], [], s=110, marker="o", facecolors="none",
                           edgecolors="#d1495b", linewidths=1.2, zorder=6)
        _style_axis(ax, X_proj)
        artists.append((claim_edges, scatter_noise, scatter_anom, wrong,
                        ax.set_title("", fontsize=9)))

    fig.suptitle(
        f"{graph['dataset']}   n={int(graph['n'])}  d={int(graph['d'])}   "
        f"anomaly reclamation, synchronised on claims made "
        f"(stars = ground-truth anomalies, red ring = wrongly reclaimed)",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    def render(step: int):
        for track, (claim_edges, scatter_noise, scatter_anom, wrong, title) in zip(tracks, artists):
            k = min(step, track["n"])
            owner = labels.copy()
            if k:
                owner[track["nodes"][:k]] = track["clusters"][:k]
            colors = _colors(owner, palette)
            scatter_noise.set_offsets(X_proj[noise_idx])
            scatter_noise.set_color(colors[noise_idx])
            scatter_anom.set_offsets(X_proj[anom_idx])
            scatter_anom.set_color(colors[anom_idx])
            claim_edges.set_segments(X_proj[edge_keys[track["edges"][:k]]] if k else [])
            claim_edges.set_color(colors[track["nodes"][:k]] if k else [])
            bad_nodes = track["nodes"][:k][true_anomaly[track["nodes"][:k]]] if k else np.empty(0, np.int64)
            wrong.set_offsets(X_proj[bad_nodes] if len(bad_nodes) else np.empty((0, 2)))
            done = " (stopped)" if k == track["n"] and step > track["n"] else ""
            title.set_text(
                f"{track['caption']}\n"
                f"claims {k}/{track['n']}{done}   "
                f"{k - len(bad_nodes)} real / {len(bad_nodes)} anomalies wrongly taken")
        return [a for group in artists for a in group]

    anim = FuncAnimation(fig, lambda i: render(steps[i]), frames=len(steps),
                         blit=False, interval=1000 / fps)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=fps), dpi=95)
    plt.close(fig)
    print("      " + "  |  ".join(f"{t['name']}: {t['n']} claims" for t in tracks),
          flush=True)


def validate_traces(entry) -> None:
    """The traced gates must reproduce the untraced ones exactly, or the GIFs lie.

    ``edge_gate`` itself is proven bit-exact against production by
    ``anomaly_cost_gate_study.py --validate``; this chains onto that.
    """
    for label, traced, plain in (
        ("edge_gate", lambda: edge_gate_traced(entry, p=2.0, stop_ratio=3.0)[0],
         lambda: edge_gate(entry, p=2.0, stop_ratio=3.0)[0]),
        ("cost_gate", lambda: cost_gate_traced(entry, p=PENALTY_POWER, reach=REACH,
                                               q=DENSITY_Q)[0],
         lambda: cost_gate(entry, p=PENALTY_POWER, reach=REACH, q=DENSITY_Q)[0]),
    ):
        a, b = traced(), plain()
        if not np.array_equal(a, b):
            raise SystemExit(
                f"traced {label} drifted from the plain one on this dataset "
                f"({int((a != b).sum())} points differ) -- the animations would "
                f"not match the metrics")


SINGLE_TRACKS = [
    ("old_p2_s3", "SHIPPED per-edge gate (p=2, stop_ratio=3)",
     lambda e: edge_gate_traced(e, p=2.0, stop_ratio=3.0)),
    ("fix_p3_r25_q0", f"FIXED cost gate (p={PENALTY_POWER:g}, reach={REACH:g}), no density",
     lambda e: cost_gate_traced(e, p=PENALTY_POWER, reach=REACH, q=0.0)),
    ("fix_p3_r25_q2", f"FIXED cost gate + density q=2",
     lambda e: cost_gate_traced(e, p=PENALTY_POWER, reach=REACH, q=2.0)),
    (HEADLINE, f"FIXED cost gate + density q={DENSITY_Q:g}",
     lambda e: cost_gate_traced(e, p=PENALTY_POWER, reach=REACH, q=DENSITY_Q)),
]


def single_frontier_gif(graph, entry, track, out_path: Path, frames=48, fps=6) -> None:
    """One gate, one big panel -- the layout ``animate_dijkstra_anomaly_assign.py`` uses.

    Grey web = the full Delaunay/growth graph; coloured segments = edges the
    stage actually used to claim a point; stars = the noise set entering the
    stage, coloured once claimed. Red rings mark claims that took a
    ground-truth anomaly (the old script had no labels to check against; these
    datasets do, and it is the whole question here).
    """
    name, caption, trace_fn = track
    X_proj = graph["X_proj"][:, :2]
    labels = graph["labels"]
    edge_keys = graph["edge_keys"]
    true_anomaly = graph["y"] < 0
    was_noise = labels < 0
    palette = _palette(labels, plt.get_cmap("tab20"))
    base_colors = _colors(labels, palette)

    _owner, trace = trace_fn(entry)
    nodes = np.array([t[0] for t in trace], dtype=np.int64)
    clusters = np.array([t[1] for t in trace], dtype=np.int64)
    costs = np.array([t[2] for t in trace], dtype=np.float64)
    edges = np.array([t[3] for t in trace], dtype=np.int64)
    n_claims = len(trace)
    steps = ([0] * 6
             + list(np.unique(np.linspace(0, n_claims, frames).astype(int)))
             + [n_claims] * 10)

    fig, ax = plt.subplots(figsize=(7.4, 7.6))
    ax.add_collection(LineCollection(
        X_proj[edge_keys] if len(edge_keys) else np.empty((0, 2, 2)),
        colors="#bdbdbd", linewidths=0.35, alpha=0.30, zorder=1))
    claim_edges = LineCollection([], linewidths=1.3, alpha=0.9, zorder=2)
    ax.add_collection(claim_edges)
    ax.scatter(*X_proj[~was_noise].T, s=6, c=base_colors[~was_noise],
               linewidths=0, alpha=0.9, zorder=3)
    scatter_anom = ax.scatter([], [], s=26, marker="*", edgecolors="black",
                              linewidths=0.5, alpha=0.95, zorder=4)
    wrong = ax.scatter([], [], s=100, marker="o", facecolors="none",
                       edgecolors="#d1495b", linewidths=1.1, zorder=5)
    title = ax.set_title("", fontsize=9)
    _style_axis(ax, X_proj)
    fig.suptitle(f"{graph['dataset']}\n{caption}", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.91))

    noise_idx = np.flatnonzero(was_noise)

    def render(k: int):
        owner = labels.copy()
        if k:
            owner[nodes[:k]] = clusters[:k]
        colors = _colors(owner, palette)
        scatter_anom.set_offsets(X_proj[noise_idx])
        scatter_anom.set_color(colors[noise_idx])
        claim_edges.set_segments(X_proj[edge_keys[edges[:k]]] if k else [])
        claim_edges.set_color(colors[nodes[:k]] if k else [])
        bad = nodes[:k][true_anomaly[nodes[:k]]] if k else np.empty(0, np.int64)
        wrong.set_offsets(X_proj[bad] if len(bad) else np.empty((0, 2)))
        spent = f"cost {costs[k - 1]:.2f}/{REACH:g}" if (k and name.startswith("fix")) else ""
        title.set_text(
            f"{k}/{int(was_noise.sum())} noise points claimed, "
            f"{int(was_noise.sum()) - k} remaining   {spent}\n"
            f"{k - len(bad)} real / {len(bad)} ground-truth anomalies wrongly taken")
        return claim_edges, scatter_anom, wrong, title

    anim = FuncAnimation(fig, lambda i: render(steps[i]), frames=len(steps),
                         blit=False, interval=1000 / fps)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=fps), dpi=120)
    plt.close(fig)
    return n_claims


def summary_figure(rows, out_path: Path) -> None:
    names = [v[0] for v in VARIANTS]
    datasets = [r["dataset"] for r in rows]
    short = [d.replace("__anom10", "") for d in datasets]
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    width = 0.8 / len(names)
    positions = np.arange(len(datasets))
    colors = plt.get_cmap("tab10")
    for metric, ax, label in ((("f1"), axes[0], "anomaly F1"),
                              (("ari_all"), axes[1], "ARI (noise as its own label)")):
        for i, name in enumerate(names):
            values = [r[f"{name}_{metric}"] for r in rows]
            ax.bar(positions + i * width, values, width, label=name, color=colors(i))
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8, ncol=len(names) // 2 + 1)
    axes[1].set_xticks(positions + 0.4 - width / 2)
    axes[1].set_xticklabels(short, rotation=18, ha="right", fontsize=8)
    fig.suptitle("Dijkstra reclamation: shipped gate vs fixed cost gate (+/- density)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------

def print_dataset_table(graph, results) -> None:
    print(f"\n  {graph['dataset']}   n={int(graph['n'])} d={int(graph['d'])} "
          f"true_anomalies={int((graph['y'] < 0).sum())} "
          f"noise_in={int((graph['labels'] < 0).sum())}")
    print(f"    {'variant':<16}{'P':>7}{'R':>7}{'F1':>7}{'dF1':>8}"
          f"{'ariA':>7}{'dariA':>8}{'ariR':>7}{'claims':>8}{'bad':>6}{'purity':>8}")
    base = results["noreassign"][1]
    for name, _fn, _doc in VARIANTS:
        m = results[name][1]
        print(f"    {name:<16}{m['precision']:>7.3f}{m['recall']:>7.3f}{m['f1']:>7.3f}"
              f"{m['f1'] - base['f1']:>+8.3f}{m['ari_all']:>7.3f}"
              f"{m['ari_all'] - base['ari_all']:>+8.3f}{m['ari_real']:>7.3f}"
              f"{m['claims']:>8}{m['claims_bad']:>6}{m['claim_purity']:>8.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=_DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="substring filters, e.g. 01 11")
    parser.add_argument("--force", action="store_true", help="rebuild frozen graphs")
    parser.add_argument("--no-gif", action="store_true",
                        help="skip the 3-panel comparison GIF")
    parser.add_argument("--single", action="store_true",
                        help="also write one-gate-per-GIF animations in the\n"
                             "layout animate_dijkstra_anomaly_assign.py uses")
    parser.add_argument("--single-variants", nargs="*", default=None,
                        help="which gates get a single-panel GIF "
                             "(default: all four)")
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--fps", type=float, default=7.0)
    args = parser.parse_args()

    files = numbered_anomaly_datasets(args.data)
    if args.datasets:
        files = [f for f in files if any(s in f.stem for s in args.datasets)]
    if not files:
        raise SystemExit("no numbered anomaly datasets matched")
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(files)} numbered datasets with ground-truth anomalies:")
    for f in files:
        print(f"  - {f.stem}")

    rows = []
    for path in files:
        print(f"\n[{path.stem}]", flush=True)
        graph = frozen_graph(path, args.out / "cache", force=args.force)
        entry = entry_of(graph)
        results = {}
        for name, fn, _doc in VARIANTS:
            started = time.perf_counter()
            assigned, _ = fn(entry)
            results[name] = (assigned, metrics_for(graph, assigned)
                             | {"seconds": round(time.perf_counter() - started, 3)})
        validate_traces(entry)
        print_dataset_table(graph, results)

        row = {"dataset": graph["dataset"], "n": int(graph["n"]), "d": int(graph["d"]),
               "n_true_anomaly": int((graph["y"] < 0).sum()),
               "n_noise_in": int((graph["labels"] < 0).sum())}
        for name, _fn, _doc in VARIANTS:
            for key, value in results[name][1].items():
                row[f"{name}_{key}"] = value
        rows.append(row)

        panel_figure(graph, results, args.out / f"{graph['dataset']}__panels.png")
        density_figure(graph, args.out / f"{graph['dataset']}__density.png")
        print(f"    wrote {graph['dataset']}__panels.png, __density.png", flush=True)
        if not args.no_gif:
            frontier_gif(graph, entry,
                         args.out / f"{graph['dataset']}__frontier.gif",
                         frames=args.frames, fps=args.fps)
            print(f"    wrote {graph['dataset']}__frontier.gif", flush=True)
        if args.single:
            wanted = args.single_variants
            for track in SINGLE_TRACKS:
                if wanted and track[0] not in wanted:
                    continue
                out = args.out / f"{graph['dataset']}__single_{track[0]}.gif"
                made = single_frontier_gif(graph, entry, track, out,
                                           frames=args.frames, fps=args.fps)
                print(f"    wrote {out.name}  ({made} claims)", flush=True)

    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (args.out / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary_figure(rows, args.out / "summary__f1.png")

    print(f"\n=== mean over {len(rows)} numbered datasets ===")
    print(f"{'variant':<16}{'P':>7}{'R':>7}{'F1':>7}{'dF1':>8}{'w/l':>7}"
          f"{'ariA':>7}{'dariA':>8}{'ariR':>7}{'claims':>8}{'purity':>8}")
    for name, _fn, doc in VARIANTS:
        d_f1 = [r[f"{name}_f1"] - r["noreassign_f1"] for r in rows]
        d_ari = [r[f"{name}_ari_all"] - r["noreassign_ari_all"] for r in rows]
        wl = f"{sum(x > 1e-9 for x in d_f1)}/{sum(x < -1e-9 for x in d_f1)}"
        print(f"{name:<16}"
              f"{np.mean([r[f'{name}_precision'] for r in rows]):>7.3f}"
              f"{np.mean([r[f'{name}_recall'] for r in rows]):>7.3f}"
              f"{np.mean([r[f'{name}_f1'] for r in rows]):>7.3f}"
              f"{np.mean(d_f1):>+8.3f}{wl:>7}"
              f"{np.mean([r[f'{name}_ari_all'] for r in rows]):>7.3f}"
              f"{np.mean(d_ari):>+8.3f}"
              f"{np.mean([r[f'{name}_ari_real'] for r in rows]):>7.3f}"
              f"{np.mean([r[f'{name}_claims'] for r in rows]):>8.1f}"
              f"{np.mean([r[f'{name}_claim_purity'] for r in rows]):>8.3f}   {doc}")

    names = [n for n in KIND_NAMES.values() if any(f"noreassign_recall_{n}" in r for r in rows)]
    if names:
        print("\n=== recall by anomaly kind (higher = anomaly correctly left as noise) ===")
        print(f"{'variant':<16}" + "".join(f"{n:>10}" for n in names))
        for name, _fn, _doc in VARIANTS:
            cells = []
            for kind_name in names:
                values = [r[f"{name}_recall_{kind_name}"] for r in rows
                          if f"{name}_recall_{kind_name}" in r]
                cells.append(f"{np.mean(values):>10.3f}" if values else f"{'--':>10}")
            print(f"{name:<16}" + "".join(cells))

    print(f"\nWrote {args.out}/metrics.csv and figures")


if __name__ == "__main__":
    main()
