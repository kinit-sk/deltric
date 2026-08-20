#!/usr/bin/env python3
"""Before/after comparison: production noise-labeling vs. Dijkstra assignment.

1x3 panels sharing the same projected-space layout:

  1. ground truth
  2. "before" -- production ``cluster_tri`` output as it always has been:
     points that don't join a component get left as noise (label -1)
  3. "after" -- same main clusters, but every reachable noise point has been
     claimed by ``component_growth_anomaly_reassign=True`` (see
     ``utils_component_growth._assign_anomalies_dijkstra`` for the algorithm
     and its math)

This is a thin visualization wrapper; all logic lives in ``cluster_tri`` /
``_assign_anomalies_dijkstra``.

Usage:
    python studies/plot_anomaly_assign_before_after.py \\
        --data data/59_h2mg_64_50.npz \\
        --out studies/results/growth_dijkstra_anomaly_assign/59_h2mg_64_50_before_after.png
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
from deltric_minimal.utils_component_growth import cluster_tri  # noqa: E402


def _segments(X_proj: np.ndarray, edge_keys: np.ndarray) -> np.ndarray:
    display = np.asarray(X_proj[:, :2], dtype=float)
    if len(edge_keys) == 0:
        return np.empty((0, 2, 2), dtype=float)
    return display[edge_keys]


def _point_colors(labels: np.ndarray, palette: dict[int, tuple]) -> np.ndarray:
    colors = np.full((len(labels), 4), (0.15, 0.15, 0.15, 0.55), dtype=float)
    for cluster, color in palette.items():
        colors[labels == cluster] = color
    return colors


def _panel(ax, X_proj, background_segments, labels, palette, title, noise_label=None):
    ax.add_collection(LineCollection(
        background_segments, colors="#bdbdbd", linewidths=0.35, alpha=0.25, zorder=1,
    ))
    colors = _point_colors(labels, palette)
    anomaly_mask = labels < 0
    ax.scatter(
        X_proj[~anomaly_mask, 0], X_proj[~anomaly_mask, 1], s=6,
        c=colors[~anomaly_mask], alpha=0.9, linewidths=0, zorder=3,
    )
    ax.scatter(
        X_proj[anomaly_mask, 0], X_proj[anomaly_mask, 1], s=26,
        c=colors[anomaly_mask], edgecolors="black", linewidths=0.5,
        marker="*", alpha=0.95, zorder=4,
    )
    if noise_label is not None:
        n_noise = int(np.count_nonzero(anomaly_mask))
        title = f"{title}\n({n_noise} {noise_label})"
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

    assigned_labels = cluster_tri(
        X, **BASE_CONFIG,
        component_growth_anomaly_reassign=True,
        component_growth_anomaly_reassign_penalty_power=args.penalty_power,
        component_growth_anomaly_reassign_stop_ratio=args.stop_ratio,
    )
    graph_state = cluster_tri.last_component_growth
    main_labels = graph_state["merged_labels"]
    X_proj = np.asarray(graph_state["X_proj"], dtype=np.float64)
    edge_keys = np.asarray(graph_state["edge_keys"], dtype=np.int64)

    baseline_ari = float(adjusted_rand_score(y, main_labels)) if y is not None else float("nan")
    assigned_ari = float(adjusted_rand_score(y, assigned_labels)) if y is not None else float("nan")

    background = _segments(X_proj, edge_keys)
    cmap = plt.get_cmap("tab20")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    if y is not None:
        true_clusters = np.unique(y[y >= 0])
        true_palette = {int(c): cmap(i % 20) for i, c in enumerate(true_clusters)}
        _panel(axes[0], X_proj, background, y, true_palette, "ground truth")
    else:
        axes[0].axis("off")

    main_clusters = np.unique(main_labels[main_labels >= 0])
    main_palette = {int(c): cmap(i % 20) for i, c in enumerate(main_clusters)}
    _panel(
        axes[1], X_proj, background, main_labels, main_palette,
        f"before: production output\nARI={baseline_ari:.4f}", noise_label="left as noise",
    )
    # assigned_labels uses freshly-compacted ids, but every point that was
    # already in a main cluster keeps the same *point-to-point* grouping as
    # main_labels; reuse main_palette by remapping each assigned id to the
    # main-cluster id its owner set is a superset of, so colors stay stable
    # between the "before" and "after" panels.
    assigned_palette: dict[int, tuple] = {}
    for main_id in main_clusters:
        member = np.flatnonzero(main_labels == main_id)[0]
        assigned_palette[int(assigned_labels[member])] = main_palette[int(main_id)]
    _panel(
        axes[2], X_proj, background, assigned_labels, assigned_palette,
        f"after: + dijkstra assignment\nARI={assigned_ari:.4f}", noise_label="still unreached",
    )

    fig.suptitle(
        f"{args.data.stem}  --  anomaly assignment before/after  "
        f"(delta ARI = {assigned_ari - baseline_ari:+.4f})",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    plt.close(fig)
    print(
        f"wrote {args.out}  (baseline={baseline_ari:.4f}, assigned={assigned_ari:.4f}, "
        f"delta={assigned_ari - baseline_ari:+.4f})"
    )


if __name__ == "__main__":
    main()
