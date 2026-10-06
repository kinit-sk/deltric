#!/usr/bin/env python3
"""Plot seed components and component-local boundary diagnostics.

Seed edges come directly from ``component_growth_graph(...)["initial_mask"]``.
This is the graph after projection, Delaunay construction, original-kNN
relation, and the seed hard gate, but before any growth edge is accepted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils_component_growth import component_growth_graph  # noqa: E402


ORANGE = "#f28e2b"
BACKGROUND = "#bdbdbd"


def _seed_components(edge_keys: np.ndarray, initial_mask: np.ndarray, n_points: int,
                     min_edges: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return seed component labels, per-component edge counts, and large mask."""
    seed_edges = edge_keys[initial_mask]
    if len(seed_edges):
        graph = coo_matrix(
            (np.ones(2 * len(seed_edges), dtype=np.int8),
             (np.concatenate((seed_edges[:, 0], seed_edges[:, 1])),
              np.concatenate((seed_edges[:, 1], seed_edges[:, 0])))),
            shape=(n_points, n_points),
        ).tocsr()
        component_count, labels = connected_components(graph, directed=False)
    else:
        component_count = n_points
        labels = np.arange(n_points, dtype=np.int64)
    edge_counts = np.zeros(component_count, dtype=np.int64)
    if len(seed_edges):
        np.add.at(edge_counts, labels[seed_edges[:, 0]], 1)
    large = edge_counts > int(min_edges)
    return labels.astype(np.int64), edge_counts, large


def _component_boundary_scores(X: np.ndarray, X_proj: np.ndarray,
                               edge_keys: np.ndarray, lengths: np.ndarray,
                               labels: np.ndarray, large: np.ndarray) -> dict[str, np.ndarray]:
    """Calculate component-relative boundary length and outer-point metrics."""
    n_points = len(X)
    incident = [[] for _ in range(n_points)]
    for edge_index, (u, v) in enumerate(edge_keys):
        incident[int(u)].append(edge_index)
        incident[int(v)].append(edge_index)

    component_count = len(large)
    component_median = np.full(component_count, np.nan, dtype=np.float64)
    component_q75 = np.full(component_count, np.nan, dtype=np.float64)
    component_right_spread = np.full(component_count, np.nan, dtype=np.float64)
    for component in np.flatnonzero(large):
        members = np.flatnonzero(labels == component)
        edge_indices = np.unique(np.concatenate([incident[point] for point in members]))
        q25, median, q75 = np.percentile(lengths[edge_indices], [25.0, 50.0, 75.0])
        component_median[component] = median
        component_q75[component] = q75
        component_right_spread[component] = max(float(q75 - median), 1e-12)

    # This is genuine projected-neighbour recall, not Delaunay recall.  It is
    # therefore exactly one for a 2D input where no UMAP projection occurs.
    k = min(10, n_points - 1)
    original_knn = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(
        X, return_distance=False
    )[:, 1:]
    projected_knn = NearestNeighbors(n_neighbors=k + 1).fit(X_proj).kneighbors(
        X_proj, return_distance=False
    )[:, 1:]
    recall10_point = np.array([
        len(set(original_knn[point]) & set(projected_knn[point])) / k
        for point in range(n_points)
    ], dtype=np.float64)

    boundary_mask = np.zeros(len(edge_keys), dtype=bool)
    outer_q25 = np.array([
        np.percentile(lengths[incident[point]], 25.0)
        for point in range(n_points)
    ], dtype=np.float64)
    boundary_length_relative_raw = np.full(len(edge_keys), np.nan, dtype=np.float64)
    boundary_outer_q25_relative_raw = np.full(len(edge_keys), np.nan, dtype=np.float64)
    boundary_recall = np.full(len(edge_keys), np.nan, dtype=np.float64)
    boundary_component = np.full(len(edge_keys), -1, dtype=np.int64)
    for edge_index, (u, v) in enumerate(edge_keys):
        component_u, component_v = labels[u], labels[v]
        candidates: list[tuple[int, int]] = []
        if large[component_u] and component_u != component_v:
            candidates.append((int(component_u), int(v)))
        if large[component_v] and component_v != component_u:
            candidates.append((int(component_v), int(u)))
        if not candidates:
            continue
        boundary_mask[edge_index] = True
        # An edge can touch two distinct seed components.  Use the lower
        # component-relative score and lower outer recall, avoiding an
        # arbitrary endpoint ordering while keeping the display conservative.
        candidate_length_relative = np.array([
            (lengths[edge_index] - component_median[c]) / component_right_spread[c]
            for c, _ in candidates
        ])
        candidate_outer_q25_relative = np.array([
            (outer_q25[outer] - component_median[c]) / component_right_spread[c]
            for c, outer in candidates
        ])
        candidate_recall = np.array([recall10_point[outer] for _, outer in candidates])
        chosen = int(np.nanargmin(candidate_length_relative))
        boundary_length_relative_raw[edge_index] = candidate_length_relative[chosen]
        boundary_outer_q25_relative_raw[edge_index] = np.min(candidate_outer_q25_relative)
        boundary_recall[edge_index] = float(np.min(candidate_recall))
        boundary_component[edge_index] = candidates[chosen][0]

    return {
        "component_median": component_median,
        "component_q75": component_q75,
        "component_right_spread": component_right_spread,
        "recall10_projected_point": recall10_point,
        "outer_adjacent_edge_q25": outer_q25,
        "boundary_mask": boundary_mask,
        "boundary_length_relative_raw": boundary_length_relative_raw,
        "boundary_length_relative_display": np.clip(
            boundary_length_relative_raw, 0.0, 2.0,
        ),
        "boundary_outer_q25_relative_raw": boundary_outer_q25_relative_raw,
        "boundary_outer_q25_relative_display": np.clip(
            boundary_outer_q25_relative_raw, 0.0, 2.0,
        ),
        "boundary_outer_recall10": boundary_recall,
        "boundary_component": boundary_component,
    }


def _segments(X_proj: np.ndarray, edge_keys: np.ndarray) -> np.ndarray:
    return np.asarray(X_proj[:, :2], dtype=float)[edge_keys]


def _draw_base(ax, X_proj: np.ndarray, segments: np.ndarray, large_points: np.ndarray,
               seed_mask: np.ndarray, title: str):
    ax.add_collection(LineCollection(segments, colors=BACKGROUND, linewidths=0.35,
                                     alpha=0.34, zorder=1))
    if np.any(seed_mask):
        ax.add_collection(LineCollection(segments[seed_mask], colors=ORANGE,
                                         linewidths=1.0, alpha=0.95, zorder=3))
    ax.scatter(X_proj[:, 0], X_proj[:, 1], s=2.5, c="#4d4d4d", alpha=0.45,
               linewidths=0, zorder=2)
    if np.any(large_points):
        ax.scatter(X_proj[large_points, 0], X_proj[large_points, 1], s=5.5,
                   c=ORANGE, alpha=0.88, linewidths=0, zorder=4)
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    ax.autoscale()
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_seed_truth(ax, X_proj: np.ndarray, segments: np.ndarray,
                     seed_mask: np.ndarray, labels: np.ndarray | None, title: str):
    ax.add_collection(LineCollection(segments, colors=BACKGROUND, linewidths=0.35,
                                     alpha=0.34, zorder=1))
    if np.any(seed_mask):
        ax.add_collection(LineCollection(segments[seed_mask], colors=ORANGE,
                                         linewidths=1.0, alpha=0.92, zorder=2))
    if labels is None:
        ax.scatter(X_proj[:, 0], X_proj[:, 1], s=4, c="#4d4d4d", alpha=0.7,
                   linewidths=0, zorder=3)
    else:
        colors = np.full((len(labels), 4), (0.12, 0.12, 0.12, 0.9), dtype=float)
        cmap = plt.get_cmap("tab20")
        classes = np.unique(labels[labels >= 0])
        for index, label in enumerate(classes):
            colors[labels == label] = cmap(index % 20)
        ax.scatter(X_proj[:, 0], X_proj[:, 1], s=5.5, c=colors, alpha=0.86,
                   linewidths=0, zorder=3)
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    ax.autoscale()
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_boundary_metric(ax, X_proj: np.ndarray, segments: np.ndarray,
                          large_points: np.ndarray, seed_mask: np.ndarray,
                          boundary_mask: np.ndarray, values: np.ndarray,
                          title: str, color_label: str,
                          fixed_limits: tuple[float, float] | None = None):
    _draw_base(ax, X_proj, segments, large_points, seed_mask, title)
    boundary_values = np.asarray(values[boundary_mask], dtype=float)
    if len(boundary_values) == 0:
        ax.text(0.5, 0.5, "no large-seed boundary edges", transform=ax.transAxes,
                ha="center", va="center", color="#555555", fontsize=10)
        return
    if fixed_limits is None:
        lower = float(np.nanmin(boundary_values))
        upper = float(np.nanmax(boundary_values))
        # A constant metric still needs a valid colour range.
        if not np.isfinite(lower) or not np.isfinite(upper):
            lower, upper = 0.0, 1.0
        elif np.isclose(lower, upper):
            padding = max(abs(lower) * 0.05, 1e-6)
            lower, upper = lower - padding, upper + padding
    else:
        lower, upper = fixed_limits
    collection = LineCollection(
        segments[boundary_mask], array=values[boundary_mask], cmap="RdBu_r",
        clim=(lower, upper), linewidths=1.1, alpha=0.96, zorder=5,
    )
    ax.add_collection(collection)
    colorbar = ax.figure.colorbar(collection, ax=ax, fraction=0.046, pad=0.02)
    colorbar.set_label(color_label, fontsize=8)
    if fixed_limits is not None:
        colorbar.set_ticks([lower, 0.5 * (lower + upper), upper])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dim-reduction", default="umap")
    parser.add_argument("--project-dim", type=int, choices=(2, 3), default=2)
    parser.add_argument("--umap-n-epochs", default="100")
    parser.add_argument("--umap-n-neighbors", type=int, default=15)
    parser.add_argument("--seed-hard-limit", type=float, default=0.75)
    parser.add_argument("--hard-limit", type=float, default=0.9)
    parser.add_argument("--growth-hard-limit", type=float, default=None)
    parser.add_argument("--mild-limit", type=float, default=-0.5)
    parser.add_argument("--initial-relation", choices=("union", "mutual"), default="union")
    parser.add_argument("--growth-relation", choices=("union",), default="union")
    parser.add_argument("--component-growth-knn", type=int, default=50)
    parser.add_argument("--component-growth-min-edges", type=int, default=10)
    parser.add_argument("--hard-gate-mode", choices=(
        "strict", "separate_limits", "knn_relaxed", "knn_growth_relaxed", "knn_global"
    ), default="knn_relaxed")
    parser.add_argument("--seed-relaxed-hard-limit", type=float, default=0.75)
    parser.add_argument("--growth-relaxed-hard-limit", type=float, default=1.7)
    parser.add_argument("--local-knn-selectivity-threshold", type=float, default=0.1)
    parser.add_argument("--local-knn-selectivity-scope", choices=("edge", "edge_2hop", "component"), default="edge")
    parser.add_argument("--projected-hard-limit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--multiplicative-hard-limit", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    if args.umap_n_epochs != "auto":
        args.umap_n_epochs = int(args.umap_n_epochs)

    loaded = np.load(args.data, allow_pickle=False)
    X = StandardScaler().fit_transform(np.asarray(loaded["X"], dtype=np.float64))
    state = component_growth_graph(
        X, dim_reduction=args.dim_reduction, project_dim=args.project_dim,
        umap_n_epochs=args.umap_n_epochs, umap_n_neighbors=args.umap_n_neighbors,
        hard_limit=args.hard_limit, growth_hard_limit=args.growth_hard_limit,
        seed_hard_limit=args.seed_hard_limit, mild_limit=args.mild_limit,
        initial_relation=args.initial_relation, growth_relation=args.growth_relation,
        knn_k=args.component_growth_knn, component_min_edges=args.component_growth_min_edges,
        hard_gate_mode=args.hard_gate_mode,
        seed_relaxed_hard_limit=args.seed_relaxed_hard_limit,
        growth_relaxed_hard_limit=args.growth_relaxed_hard_limit,
        local_knn_selectivity_threshold=args.local_knn_selectivity_threshold,
        local_knn_selectivity_scope=args.local_knn_selectivity_scope,
        projected_hard_limit=args.projected_hard_limit,
        multiplicative_hard_limit=args.multiplicative_hard_limit,
    )
    edge_keys = np.asarray(state["edge_keys"], dtype=np.int64)
    initial_mask = np.asarray(state["initial_mask"], dtype=bool)
    labels, edge_counts, large = _seed_components(
        edge_keys, initial_mask, len(X), args.component_growth_min_edges,
    )
    large_points = large[labels]
    seed_mask = initial_mask & large_points[edge_keys[:, 0]] & large_points[edge_keys[:, 1]]
    scores = _component_boundary_scores(
        X, np.asarray(state["X_proj"], dtype=np.float64), edge_keys,
        np.asarray(state["orig_edge_sizes"], dtype=np.float64), labels, large,
    )
    segments = _segments(state["X_proj"], edge_keys)
    truth_labels = None
    if "y" in loaded.files:
        truth_labels = np.asarray(loaded["y"], dtype=np.int64)
    elif "y_clean" in loaded.files:
        truth_labels = np.asarray(loaded["y_clean"], dtype=np.int64)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14), constrained_layout=True)
    _draw_seed_truth(axes[0, 0], state["X_proj"], segments, seed_mask, truth_labels,
                     f"seed components (orange), original labels (points)")
    _draw_boundary_metric(
        axes[0, 1], state["X_proj"], segments, large_points, seed_mask,
        scores["boundary_mask"], scores["boundary_length_relative_display"],
        "boundary: component-relative edge length",
        "(edge − component median)/(q75 − median)",
        fixed_limits=(0.0, 2.0),
    )
    _draw_boundary_metric(
        axes[1, 0], state["X_proj"], segments, large_points, seed_mask,
        scores["boundary_mask"], scores["boundary_outer_recall10"],
        "boundary: outer-point projected recall@10",
        "outer-point projected recall@10",
        fixed_limits=(0.0, 1.0),
    )
    _draw_boundary_metric(
        axes[1, 1], state["X_proj"], segments, large_points, seed_mask,
        scores["boundary_mask"], scores["boundary_outer_q25_relative_display"],
        "boundary: outer-point adjacent-edge q25",
        "(outer q25 − component median)/(q75 − median)",
        fixed_limits=(0.0, 2.0),
    )
    fig.suptitle(
        f"Seed-component boundary diagnostics — {args.data.stem}", fontsize=14
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    np.savez_compressed(
        args.out.with_suffix(".npz"), X_proj=state["X_proj"], edge_keys=edge_keys,
        initial_mask=initial_mask, seed_labels=labels, seed_edge_counts=edge_counts,
        large_seed_components=large, **scores,
    )
    summary = {
        "dataset": str(args.data), "out": str(args.out),
        "seed_hard_limit": state["seed_hard_limit"],
        "initial_relation": state["initial_relation"], "knn_k": args.component_growth_knn,
        "hard_gate_mode": state["hard_gate_mode"],
        "component_min_edges": args.component_growth_min_edges,
        "seed_component_count": int(len(edge_counts)),
        "large_seed_component_count": int(np.count_nonzero(large)),
        "seed_edge_count": int(np.count_nonzero(initial_mask)),
        "boundary_edge_count": int(np.count_nonzero(scores["boundary_mask"])),
        "edge_relative_definition": "(boundary edge length − component median)/(component q75 − component median); component statistics use all incident original-space Delaunay edges; display is clipped to [0, 2], while raw values are preserved in the NPZ",
        "recall_definition": "outer endpoint original-10NN recovered in projected-space 10NN; no Delaunay restriction",
        "outer_q25_definition": "same component-relative formula as boundary edge length, replacing boundary edge length by q25 of original-space Delaunay edge lengths incident to the outer point",
    }
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
