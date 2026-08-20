#!/usr/bin/env python3
"""Plot competitive weighted-Dijkstra cluster growth stages.

Mirrors ``plot_prunning_stages_2x2_growth.py``'s 2x2 layout (mirrored by
``results/prune_stages_component_growth``), but for the algorithm in
``growth_dijkstra_penalty.py``: seed clusters -> early growth -> late growth
-> final clusters, all four panels sharing the projected-space point layout.
This is a thin visualization wrapper; all seeding/growth logic lives in
``growth_dijkstra_penalty.dijkstra_penalty_growth``.

Usage:
    python studies/plot_growth_dijkstra_penalty.py \\
        --data data/11_v2_moons_2d_2500.npz \\
        --out studies/results/growth_dijkstra_penalty/11_v2_moons_2d_2500.png
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

from growth_dijkstra_penalty import (  # noqa: E402
    BASE_CONFIG,
    DEFAULT_KNN_K,
    DEFAULT_SEED_HARD_LIMIT,
    _build_graph,
    dijkstra_penalty_growth,
)


def _segments(X_proj: np.ndarray, edge_keys: np.ndarray) -> np.ndarray:
    display = np.asarray(X_proj[:, :2], dtype=float)
    if len(edge_keys) == 0:
        return np.empty((0, 2, 2), dtype=float)
    return display[edge_keys]


def _point_colors(labels: np.ndarray) -> np.ndarray:
    colors = np.full((len(labels), 4), (0.72, 0.72, 0.72, 0.78), dtype=float)
    cmap = plt.get_cmap("tab20")
    clusters = np.unique(labels[labels >= 0])
    palette = {int(c): cmap(i % 20) for i, c in enumerate(clusters)}
    for cluster, color in palette.items():
        colors[labels == cluster] = color
    return colors


def _panel(ax, X_proj, edge_keys, background_segments, labels, title):
    ax.add_collection(LineCollection(
        background_segments, colors="#bdbdbd", linewidths=0.35, alpha=0.30, zorder=1,
    ))
    live_mask = labels >= 0
    if np.any(live_mask):
        live_edges = live_mask[edge_keys[:, 0]] & live_mask[edge_keys[:, 1]]
        same_cluster = labels[edge_keys[:, 0]] == labels[edge_keys[:, 1]]
        cluster_edges = live_edges & same_cluster
        segments = _segments(X_proj, edge_keys[cluster_edges])
        colors = _point_colors(labels)
        edge_colors = colors[edge_keys[cluster_edges, 0]]
        if len(segments):
            ax.add_collection(LineCollection(
                segments, colors=edge_colors, linewidths=0.9, alpha=0.85, zorder=2,
            ))
    else:
        colors = _point_colors(labels)
    ax.scatter(X_proj[:, 0], X_proj[:, 1], s=5, c=colors, alpha=0.9, linewidths=0, zorder=4)
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

    graph = _build_graph(X)
    edge_keys, orig_sizes, X_proj = graph["edge_keys"], graph["orig_sizes"], graph["X_proj"]

    # Dry pass: learn the round count so snapshot rounds land at fixed
    # fractions of the actual run instead of guessing a round budget.
    dry = dijkstra_penalty_growth(
        X, edge_keys, orig_sizes, knn_k=DEFAULT_KNN_K, seed_hard_limit=DEFAULT_SEED_HARD_LIMIT,
        neighbor_stat=BASE_CONFIG["neighbor_stat"],
        multiplicative_hard_limit=BASE_CONFIG["multiplicative_hard_limit"],
        penalty_power=args.penalty_power, stop_ratio=args.stop_ratio,
        min_cluster_size=BASE_CONFIG["min_cluster_size"],
    )
    n_rounds = max(dry["n_rounds"], 1)
    checkpoints = sorted({0, n_rounds // 3, (2 * n_rounds) // 3, n_rounds})

    state = dijkstra_penalty_growth(
        X, edge_keys, orig_sizes, knn_k=DEFAULT_KNN_K, seed_hard_limit=DEFAULT_SEED_HARD_LIMIT,
        neighbor_stat=BASE_CONFIG["neighbor_stat"],
        multiplicative_hard_limit=BASE_CONFIG["multiplicative_hard_limit"],
        penalty_power=args.penalty_power, stop_ratio=args.stop_ratio,
        min_cluster_size=BASE_CONFIG["min_cluster_size"],
        snapshot_rounds=set(checkpoints) | {-1},
    )

    background = _segments(X_proj, edge_keys)
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    panel_rounds = checkpoints[:3] + [checkpoints[-1]]
    panel_titles = [
        f"round {panel_rounds[0]} (seed clusters)",
        f"round {panel_rounds[1]} (early growth)",
        f"round {panel_rounds[2]} (late growth)",
        f"round {panel_rounds[3]} (converged, pre-min-size)",
    ]
    for ax, round_number, title in zip(axes.flat, panel_rounds, panel_titles):
        _panel(ax, X_proj, edge_keys, background, state["snapshots"][round_number], title)

    ari = float(adjusted_rand_score(y, state["labels"])) if y is not None else float("nan")
    n_final_clusters = int(len(np.unique(state["labels"][state["labels"] >= 0])))
    fig.suptitle(
        f"{args.data.stem}  --  dijkstra_penalty (power={args.penalty_power}, "
        f"stop_ratio={args.stop_ratio})  final ARI={ari:.4f}  "
        f"n_seed_clusters={state['n_clusters']}  n_final_clusters={n_final_clusters}  "
        f"rounds={state['n_rounds']}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    plt.close(fig)
    print(f"wrote {args.out}  (ari={ari:.4f}, rounds={state['n_rounds']}, "
          f"seed_clusters={state['n_clusters']}, final_clusters={n_final_clusters})")


if __name__ == "__main__":
    main()
