#!/usr/bin/env python3
"""Plot the anomaly-assignment stage from ``growth_dijkstra_anomaly_assign.py``.

2x2 layout, all four panels sharing the same projected-space point layout:

  1. main clusters from ``cluster_tri`` (production output), anomalies in gray
  2/3. assignment in progress at two intermediate rounds
  4. final: every reachable anomaly claimed by a cluster

Edges actually used to reach an anomaly (``grown_mask``) are drawn thicker
and colored by the claiming cluster; all other Delaunay edges are drawn as a
faint gray background so the underlying graph stays visible.

Usage:
    python studies/plot_growth_dijkstra_anomaly_assign.py \\
        --data data/14_v2_blobs_8d_3500_k7.npz \\
        --out studies/results/growth_dijkstra_anomaly_assign/14_v2_blobs_8d_3500_k7.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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
    cluster_tri,
)


def _segments(X_proj: np.ndarray, edge_keys: np.ndarray) -> np.ndarray:
    display = np.asarray(X_proj[:, :2], dtype=float)
    if len(edge_keys) == 0:
        return np.empty((0, 2, 2), dtype=float)
    return display[edge_keys]


def _point_colors(owner: np.ndarray, main_labels: np.ndarray) -> np.ndarray:
    """Gray for still-unclaimed anomalies, else the claiming cluster's color.

    Colors are keyed off ``main_labels`` cluster ids (fixed across panels)
    so a cluster keeps the same color in every snapshot, whether the point
    was a main-cluster member from round 0 or claimed later.
    """
    colors = np.full((len(owner), 4), (0.72, 0.72, 0.72, 0.78), dtype=float)
    cmap = plt.get_cmap("tab20")
    clusters = np.unique(main_labels[main_labels >= 0])
    palette = {int(c): cmap(i % 20) for i, c in enumerate(clusters)}
    for cluster, color in palette.items():
        colors[owner == cluster] = color
    return colors


def _panel(ax, X_proj, edge_keys, background_segments, owner, main_labels, grown_mask, title):
    ax.add_collection(LineCollection(
        background_segments, colors="#bdbdbd", linewidths=0.35, alpha=0.30, zorder=1,
    ))
    colors = _point_colors(owner, main_labels)
    if np.any(grown_mask):
        live_mask = owner >= 0
        used = grown_mask & live_mask[edge_keys[:, 0]] & live_mask[edge_keys[:, 1]]
        used = used & (owner[edge_keys[:, 0]] == owner[edge_keys[:, 1]])
        segments = _segments(X_proj, edge_keys[used])
        edge_colors = colors[edge_keys[used, 0]]
        if len(segments):
            ax.add_collection(LineCollection(
                segments, colors=edge_colors, linewidths=1.3, alpha=0.9, zorder=2,
            ))
    anomaly_mask = main_labels < 0
    ax.scatter(
        X_proj[~anomaly_mask, 0], X_proj[~anomaly_mask, 1], s=5,
        c=colors[~anomaly_mask], alpha=0.9, linewidths=0, zorder=3,
    )
    ax.scatter(
        X_proj[anomaly_mask, 0], X_proj[anomaly_mask, 1], s=22,
        c=colors[anomaly_mask], edgecolors="black", linewidths=0.5,
        marker="*", alpha=0.95, zorder=4,
    )
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.autoscale()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--penalty-power", type=float, default=2.0)
    parser.add_argument("--stop-ratio", type=float, default=3.0)
    args = parser.parse_args()

    npz = np.load(args.data, allow_pickle=False)
    X = StandardScaler().fit_transform(np.asarray(npz["X"], dtype=np.float64))
    y = np.asarray(npz["y"]).astype(np.int64) if "y" in npz else None

    main_labels = cluster_tri(X, **BASE_CONFIG)
    baseline_ari = float(adjusted_rand_score(y, main_labels)) if y is not None else float("nan")
    graph_state = cluster_tri.last_component_growth
    edge_keys = np.asarray(graph_state["edge_keys"], dtype=np.int64)
    proj_sizes = np.asarray(graph_state["projected_edge_sizes"], dtype=np.float64)
    X_proj = np.asarray(graph_state["X_proj"], dtype=np.float64)

    # Dry pass: learn the round count so snapshot rounds land at fixed
    # fractions of the actual run instead of guessing a round budget.
    _, dry_diag = _assign_anomalies_dijkstra(
        main_labels, edge_keys, proj_sizes, len(X),
        penalty_power=args.penalty_power, stop_ratio=args.stop_ratio,
    )
    n_rounds = max(dry_diag["n_rounds"], 1)
    checkpoints = sorted({0, n_rounds // 3, (2 * n_rounds) // 3, n_rounds})

    assigned_labels, diag = _assign_anomalies_dijkstra(
        main_labels, edge_keys, proj_sizes, len(X),
        penalty_power=args.penalty_power, stop_ratio=args.stop_ratio,
        snapshot_rounds=set(checkpoints) | {-1},
    )

    background = _segments(X_proj, edge_keys)
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    panel_rounds = checkpoints[:3] + [checkpoints[-1]]
    panel_titles = [
        f"round {panel_rounds[0]} (main clusters, {diag['n_anomalies_in']} anomalies)",
        f"round {panel_rounds[1]} (assigning)",
        f"round {panel_rounds[2]} (assigning)",
        f"round {panel_rounds[3]} (final, {diag['n_anomalies_out']} unreached)",
    ]
    for ax, round_number, title in zip(axes.flat, panel_rounds, panel_titles):
        _panel(
            ax, X_proj, edge_keys, background,
            diag["snapshots"][round_number], main_labels, diag["grown_mask"], title,
        )

    ari = float(adjusted_rand_score(y, assigned_labels)) if y is not None else float("nan")
    fig.suptitle(
        f"{args.data.stem}  --  anomaly_assign (power={args.penalty_power}, "
        f"stop_ratio={args.stop_ratio})  baseline ARI={baseline_ari:.4f} -> "
        f"assigned ARI={ari:.4f}  rounds={diag['n_rounds']}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    plt.close(fig)
    print(
        f"wrote {args.out}  (baseline={baseline_ari:.4f}, assigned={ari:.4f}, "
        f"rounds={diag['n_rounds']}, anomalies {diag['n_anomalies_in']}->{diag['n_anomalies_out']})"
    )


if __name__ == "__main__":
    main()
