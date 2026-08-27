#!/usr/bin/env python3
"""Animate the competitive-Dijkstra anomaly-reassignment stage.

Same mechanism as ``plot_growth_dijkstra_anomaly_assign.py`` (2x2 static
snapshots) but rendered as a frame-per-round animation (GIF or MP4) so the
multi-source frontier growth is visible in motion: every main cluster starts
a penalized-Dijkstra search at once, one accepted claim per cluster per
round, until each cluster's cheapest remaining option exceeds
``stop_ratio * avg_c`` and its frontier goes dead.

Usage:
    python studies/animate_dijkstra_anomaly_assign.py \\
        --data data/varied_2d_5c.npz \\
        --out studies/results/growth_dijkstra_anomaly_assign/varied_2d_5c.gif
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from growth_dijkstra_anomaly_assign import BASE_CONFIG  # noqa: E402

sys.path.insert(0, str(ROOT.parent.parent))
from deltric_minimal.utils_component_growth import (  # noqa: E402
    _assign_anomalies_dijkstra,
    _knn_core_distance,
    cluster_tri,
)


def _segments(X_proj: np.ndarray, edge_keys: np.ndarray) -> np.ndarray:
    display = np.asarray(X_proj[:, :2], dtype=float)
    if len(edge_keys) == 0:
        return np.empty((0, 2, 2), dtype=float)
    return display[edge_keys]


def _point_colors(owner: np.ndarray, main_labels: np.ndarray, cmap) -> np.ndarray:
    """Gray for unclaimed anomalies, else the claiming cluster's fixed color."""
    colors = np.full((len(owner), 4), (0.72, 0.72, 0.72, 0.85), dtype=float)
    clusters = np.unique(main_labels[main_labels >= 0])
    palette = {int(c): cmap(i % 20) for i, c in enumerate(clusters)}
    for cluster, color in palette.items():
        colors[owner == cluster] = color
    return colors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=ROOT.parent / "data" / "varied_2d_5c.npz", type=Path)
    parser.add_argument("--out", default=Path("studies/results/growth_dijkstra_anomaly_assign/anomaly_assign.gif"), type=Path)
    parser.add_argument("--penalty-power", type=float, default=2.0)
    parser.add_argument("--stop-ratio", type=float, default=3.0)
    parser.add_argument("--density-power", type=float, default=0.0,
                        help="0 = old behaviour (no original-space density term); "
                             "e.g. 2.0 = new density-correlated variant")
    parser.add_argument("--density-clip", action="store_true",
                        help="one-sided density correction (ratios below 1 clamped to 1)")
    parser.add_argument("--density-k", type=int, default=15,
                        help="k for the original-space kNN core distance")
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--hold-start", type=int, default=6, help="repeated frames on round 0")
    parser.add_argument("--hold-end", type=int, default=10, help="repeated frames on the final round")
    args = parser.parse_args()

    npz = np.load(args.data, allow_pickle=False)
    X = StandardScaler().fit_transform(np.asarray(npz["X"], dtype=np.float64))
    y = np.asarray(npz["y"]).astype(np.int64) if "y" in npz else None

    main_labels = cluster_tri(X, **BASE_CONFIG)
    baseline_ari = float(adjusted_rand_score(y, main_labels)) if y is not None else float("nan")
    state = cluster_tri.last_component_growth
    edge_keys = np.asarray(state["edge_keys"], dtype=np.int64)
    proj_sizes = np.asarray(state["projected_edge_sizes"], dtype=np.float64)
    X_proj = np.asarray(state["X_proj"], dtype=np.float64)

    use_density = args.density_power != 0.0
    core_distance = _knn_core_distance(X, args.density_k) if use_density else None
    density_kwargs = dict(
        core_distance=core_distance,
        density_power=args.density_power,
        density_clip=args.density_clip,
    )

    # Dry pass: learn the round count so every round can get its own frame.
    _, dry_diag = _assign_anomalies_dijkstra(
        main_labels, edge_keys, proj_sizes, len(X),
        penalty_power=args.penalty_power, stop_ratio=args.stop_ratio,
        **density_kwargs,
    )
    n_rounds = dry_diag["n_rounds"]
    assigned_labels, diag = _assign_anomalies_dijkstra(
        main_labels, edge_keys, proj_sizes, len(X),
        penalty_power=args.penalty_power, stop_ratio=args.stop_ratio,
        snapshot_rounds=set(range(0, n_rounds + 1)),
        **density_kwargs,
    )
    ari = float(adjusted_rand_score(y, assigned_labels)) if y is not None else float("nan")

    background = _segments(X_proj, edge_keys)
    cmap = plt.get_cmap("tab20")
    grown_mask = diag["grown_mask"]

    # One frame index per round, with extra held frames at the start/end so
    # the viewer has time to read the first and last states.
    round_sequence = [0] * max(args.hold_start, 1) + list(range(1, n_rounds + 1))
    round_sequence += [n_rounds] * max(args.hold_end - 1, 0)

    fig, ax = plt.subplots(figsize=(7, 7))
    bg_collection = LineCollection(background, colors="#bdbdbd", linewidths=0.35, alpha=0.30, zorder=1)
    edge_collection = LineCollection([], linewidths=1.3, alpha=0.9, zorder=2)
    ax.add_collection(bg_collection)
    ax.add_collection(edge_collection)
    scatter_main = ax.scatter([], [], s=6, alpha=0.9, linewidths=0, zorder=3)
    scatter_anom = ax.scatter([], [], s=26, edgecolors="black", linewidths=0.5, marker="*", alpha=0.95, zorder=4)
    title = ax.set_title("")
    ax.set_xlim(X_proj[:, 0].min() - 1, X_proj[:, 0].max() + 1)
    ax.set_ylim(X_proj[:, 1].min() - 1, X_proj[:, 1].max() + 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    variant_tag = (
        f"density-correlated (q={args.density_power:g}{', clip' if args.density_clip else ''})"
        if use_density else "old (no density term)"
    )
    fig.suptitle(
        f"{args.data.stem}  --  competitive-Dijkstra anomaly reassignment  --  {variant_tag}\n"
        f"(penalty_power={args.penalty_power}, stop_ratio={args.stop_ratio})",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    def render(round_number: int):
        owner = diag["snapshots"][round_number]
        colors = _point_colors(owner, main_labels, cmap)

        live_mask = owner >= 0
        used = grown_mask & live_mask[edge_keys[:, 0]] & live_mask[edge_keys[:, 1]]
        used = used & (owner[edge_keys[:, 0]] == owner[edge_keys[:, 1]])
        segs = _segments(X_proj, edge_keys[used])
        edge_collection.set_segments(segs)
        edge_collection.set_color(colors[edge_keys[used, 0]] if len(segs) else [])

        anomaly_mask = main_labels < 0
        scatter_main.set_offsets(X_proj[~anomaly_mask, :2])
        scatter_main.set_color(colors[~anomaly_mask])
        scatter_anom.set_offsets(X_proj[anomaly_mask, :2])
        scatter_anom.set_color(colors[anomaly_mask])

        n_claimed = int(np.count_nonzero((owner >= 0) & (main_labels < 0)))
        n_left = diag["n_anomalies_in"] - n_claimed
        title.set_text(
            f"round {round_number}/{n_rounds}  --  "
            f"{n_claimed}/{diag['n_anomalies_in']} anomalies claimed, {n_left} remaining"
        )
        return bg_collection, edge_collection, scatter_main, scatter_anom, title

    def update(frame_index: int):
        return render(round_sequence[frame_index])

    anim = FuncAnimation(fig, update, frames=len(round_sequence), blit=False, interval=1000 / args.fps)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix.lower() == ".mp4":
        anim.save(args.out, writer="ffmpeg", fps=args.fps, dpi=140)
    else:
        anim.save(args.out, writer=PillowWriter(fps=args.fps), dpi=140)
    plt.close(fig)

    print(
        f"wrote {args.out}  (baseline ARI={baseline_ari:.4f} -> assigned ARI={ari:.4f}, "
        f"rounds={n_rounds}, anomalies {diag['n_anomalies_in']}->{diag['n_anomalies_out']}, "
        f"{len(round_sequence)} frames)"
    )


if __name__ == "__main__":
    main()
