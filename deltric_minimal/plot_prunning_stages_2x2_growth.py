#!/usr/bin/env python3
"""Plot the exact component-growth graph used by ``DelTriC``.

This is intentionally a thin visualization wrapper.  All projection,
thresholding, kNN filtering, and component growth come from
``utils_component_growth.component_growth_graph``; no pruning logic is
reimplemented here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from utils_component_growth import (  # noqa: E402
    _merge_component_outliers,
    _labels_from_components,
    component_growth_graph,
)


def _segments(X_proj: np.ndarray, edge_keys: np.ndarray) -> np.ndarray:
    display = np.asarray(X_proj[:, :2], dtype=float)
    if len(edge_keys) == 0:
        return np.empty((0, 2, 2), dtype=float)
    return display[edge_keys]


def _base_axes(ax, X_proj, edge_keys, title, point_colors=None):
    if point_colors is None:
        point_colors = "black"
    ax.scatter(X_proj[:, 0], X_proj[:, 1], s=5, c=point_colors, alpha=0.82,
               linewidths=0, zorder=4)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.autoscale()


def _draw_background(ax, segments, colors="#bdbdbd", alpha=0.30, lw=0.35):
    if len(segments):
        ax.add_collection(LineCollection(
            segments, colors=colors, linewidths=lw, alpha=alpha, zorder=1
        ))


def _component_point_colors(state):
    """Return one color per point for the final component assignment.

    ``merged_labels`` is populated by ``main`` even when outlier merging is
    disabled; in that case it is the large-component assignment and points
    outside sufficiently large components remain gray.
    """
    labels = np.asarray(
        state.get("merged_labels", state.get("large_component_point_labels")),
        dtype=np.int64,
    )
    colors = np.full((len(labels), 4), (0.72, 0.72, 0.72, 0.78), dtype=float)
    cmap = plt.get_cmap("tab20")
    components = np.unique(labels[labels >= 0])
    component_colors = {
        int(component): cmap(index % 20)
        for index, component in enumerate(components)
    }
    for component, color in component_colors.items():
        colors[labels == component] = color
    return colors


def _draw_hard(ax, X_proj, state):
    edges = state["edge_keys"]
    segments = _segments(X_proj, edges)
    _draw_background(ax, segments)
    allowed = state["hard_allowed"]
    original_hard = state["seed_original_hard_mask"]
    projected_hard = state["seed_projected_hard_mask"]
    both = original_hard & projected_hard
    original_only = original_hard & ~projected_hard
    projected_only = projected_hard & ~original_hard
    if np.any(original_only):
        ax.add_collection(LineCollection(segments[original_only], colors="#d73027",
                                          linewidths=0.8, alpha=0.75, zorder=2))
    if np.any(projected_only):
        ax.add_collection(LineCollection(segments[projected_only], colors="#762a83",
                                          linewidths=0.8, alpha=0.75, zorder=2))
    if np.any(both):
        ax.add_collection(LineCollection(segments[both], colors="#7f0000",
                                          linewidths=1.0, alpha=0.9, zorder=3))
    _base_axes(
        ax, X_proj, edges,
        "seed dual hard gate\n"
        f"original ≤ {state['seed_orig_hard_threshold']:.3g}, "
        f"projected ≤ {state['seed_projected_hard_threshold'] if state['seed_projected_hard_threshold'] is not None else 'off'}",
    )
    ax.legend(handles=[
        Line2D([0], [0], color="#d73027", lw=2, label="original-space hard reject"),
        Line2D([0], [0], color="#762a83", lw=2, label="projected-space hard reject"),
        Line2D([0], [0], color="#7f0000", lw=2, label="both hard rejects"),
        Line2D([0], [0], color="#bdbdbd", lw=1, label="Delaunay background"),
    ], loc="upper right", frameon=False, fontsize=8)


def _draw_initial(ax, X_proj, state):
    edges = state["edge_keys"]
    segments = _segments(X_proj, edges)
    _draw_background(ax, segments)
    initial = state["initial_mask"]
    # Initial component coloring is computed independently from the final
    # state, so this panel does not accidentally present grown components.
    from utils_component_growth import _component_state
    initial_labels, initial_counts = _component_state(edges, initial, len(X_proj))
    initial_large = initial_counts > state["component_min_edges"]
    small = initial & ~initial_large[initial_labels[edges[:, 0]]]
    large_edges = initial & initial_large[initial_labels[edges[:, 0]]]
    if np.any(small):
        ax.add_collection(LineCollection(segments[small], colors="#2166ac",
                                          linewidths=0.7, alpha=0.80, zorder=2))
    if np.any(large_edges):
        ax.add_collection(LineCollection(segments[large_edges], colors="#e66101",
                                          linewidths=1.0, alpha=0.90, zorder=3))
    _base_axes(ax, X_proj, edges,
               "initial graph after dual hard gate\n"
               f"{state['initial_relation']} kNN: {int(initial.sum())}/{len(initial)} edges")
    ax.legend(handles=[
        Line2D([0], [0], color="#2166ac", lw=2, label="small initial component"),
        Line2D([0], [0], color="#e66101", lw=2, label=">20-edge initial component"),
        Line2D([0], [0], color="#bdbdbd", lw=1, label="Delaunay background"),
    ], loc="upper right", frameon=False, fontsize=8)


def _draw_growth(ax, X_proj, state):
    edges = state["edge_keys"]
    segments = _segments(X_proj, edges)
    _draw_background(ax, segments)
    initial = state["initial_mask"]
    if np.any(initial):
        ax.add_collection(LineCollection(segments[initial], colors="#2166ac",
                                          linewidths=0.65, alpha=0.55, zorder=2))
    automatic = state["automatic_growth_mask"]
    blue = state["blue_growth_mask"]
    if np.any(automatic):
        ax.add_collection(LineCollection(segments[automatic], colors="#2ca25f",
                                          linewidths=1.0, alpha=0.9, zorder=3))
    open_space = state.get("open_space_growth_mask")
    if open_space is not None and np.any(open_space):
        ax.add_collection(LineCollection(segments[open_space], colors="#984ea3",
                                          linewidths=1.1, alpha=0.9, zorder=3))
    if np.any(blue):
        ax.add_collection(LineCollection(segments[blue], colors="#17becf",
                                          linewidths=1.0, alpha=0.9, zorder=3))
    bridge_pruned = state.get(
        "bridge_pruned_mask", np.zeros(len(edges), dtype=bool)
    )
    if np.any(bridge_pruned):
        ax.add_collection(LineCollection(
            segments[bridge_pruned], colors="#d73027", linewidths=1.2,
            linestyles="dashed", alpha=0.9, zorder=4,
        ))
    _base_axes(ax, X_proj, edges,
               "growth edges\n"
               f"automatic={int(automatic.sum())}, union-blue={int(blue.sum())}\n"
               f"bridge-pruned={int(bridge_pruned.sum())}",
               )
    ax.legend(handles=[
        Line2D([0], [0], color="#2166ac", lw=2, label="initial retained"),
        Line2D([0], [0], color="#2ca25f", lw=2, label="automatic mild-zone growth"),
        Line2D([0], [0], color="#984ea3", lw=2, label="open-space relaxation growth"),
        Line2D([0], [0], color="#17becf", lw=2, label="blue union-kNN growth"),
        Line2D([0], [0], color="#d73027", lw=2, linestyle="--", label="short-path bridge prune"),
    ], loc="upper right", frameon=False, fontsize=8)


def _draw_final(ax, X_proj, state):
    edges = state["edge_keys"]
    segments = _segments(X_proj, edges)
    _draw_background(ax, segments)
    final = np.asarray(state["final_mask"], dtype=bool)
    strict_edges = final & np.asarray(state["hard_allowed"], dtype=bool)
    relaxed_edges = final & ~np.asarray(state["hard_allowed"], dtype=bool)
    if np.any(strict_edges):
        ax.add_collection(LineCollection(
            segments[strict_edges], colors="#111111", linewidths=0.75,
            alpha=0.78, zorder=2,
        ))
    if np.any(relaxed_edges):
        ax.add_collection(LineCollection(
            segments[relaxed_edges], colors="#777777", linewidths=1.0,
            alpha=0.95, zorder=3,
        ))
    bridge_pruned = state.get(
        "bridge_pruned_mask", np.zeros(len(edges), dtype=bool)
    )
    if np.any(bridge_pruned):
        ax.add_collection(LineCollection(
            segments[bridge_pruned], colors="#d73027", linewidths=1.2,
            linestyles="--", alpha=0.9, zorder=4,
        ))
    final_title = (
        "final components after bridge pruning"
        if state.get("bridge_pruning", False)
        else "final grown components"
    )
    _base_axes(ax, X_proj, edges,
               final_title + "\n"
               f"{int(state['final_mask'].sum())} edges, "
               f"{int(np.count_nonzero(state['large_components']))} components >"
               f"{state['component_min_edges']} edges\n"
               f"large-component ARI={state.get('large_component_ari', float('nan')):.3f}",
               point_colors=_component_point_colors(state))
    ax.legend(handles=[
        Line2D([0], [0], color="#111111", lw=2,
               label="retained within strict hard limits"),
        Line2D([0], [0], color="#777777", lw=2,
               label="retained only by relaxed hard limit"),
        Line2D([0], [0], color="#d73027", lw=2, linestyle="--",
               label="short-path bridge prune"),
        Patch(facecolor="#4c78a8", label="component-colored points"),
        Patch(facecolor="#b8b8b8", label="small components / noise"),
    ], loc="upper right", frameon=False, fontsize=8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dim-reduction", default="umap")
    parser.add_argument("--project-dim", type=int, choices=(2, 3), default=2)
    parser.add_argument("--umap-n-epochs", default="100")
    parser.add_argument("--umap-n-neighbors", type=int, default=15)
    parser.add_argument("--hard-limit", type=float, default=1.0)
    parser.add_argument("--seed-hard-limit", type=float, default=None)
    parser.add_argument("--growth-hard-limit", type=float, default=None)
    parser.add_argument("--mild-limit", type=float, default=-0.5)
    parser.add_argument("--neighbor-stat", choices=("median", "mean"), default="median")
    parser.add_argument("--projected-hard-limit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--multiplicative-hard-limit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--initial-relation", choices=("union", "mutual"), default="mutual")
    parser.add_argument("--growth-relation", choices=("union",), default="union")
    parser.add_argument("--component-growth-knn", type=int, default=30)
    parser.add_argument("--component-growth-min-edges", type=int, default=20)
    parser.add_argument(
        "--hard-gate-mode", choices=(
            "strict", "separate_limits", "knn_relaxed", "knn_growth_relaxed", "knn_global"
        ),
        default="strict",
    )
    parser.add_argument("--seed-relaxed-hard-limit", type=float, default=None)
    parser.add_argument("--growth-relaxed-hard-limit", type=float, default=None)
    parser.add_argument(
        "--local-knn-selectivity-threshold", type=float, default=1.0,
    )
    parser.add_argument(
        "--local-knn-selectivity-scope", choices=("edge", "edge_2hop", "component"),
        default="edge",
    )
    parser.add_argument("--bridge-pruning", action="store_true", default=False)
    parser.add_argument("--bridge-max-hops", type=int, default=5)
    parser.add_argument(
        "--component-growth-outlier-limit", type=float, default=None,
        help="Enable post-growth outlier merging; None disables it",
    )
    parser.add_argument(
        "--open-space-relaxation", action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--outer-long-ratio", type=float, default=3.0)
    parser.add_argument("--outer-relaxation", type=float, default=0.25)
    parser.add_argument(
        "--outer-transition-width", type=float, default=0.4054651081
    )
    args = parser.parse_args()
    if args.umap_n_epochs != "auto":
        args.umap_n_epochs = int(args.umap_n_epochs)

    npz = np.load(args.data, allow_pickle=False)
    X = StandardScaler().fit_transform(np.asarray(npz["X"], dtype=np.float64))
    state = component_growth_graph(
        X, dim_reduction=args.dim_reduction, project_dim=args.project_dim,
        umap_n_epochs=args.umap_n_epochs, umap_n_neighbors=args.umap_n_neighbors,
        hard_limit=args.hard_limit, mild_limit=args.mild_limit,
        seed_hard_limit=args.seed_hard_limit,
        growth_hard_limit=args.growth_hard_limit,
        neighbor_stat=args.neighbor_stat,
        projected_hard_limit=args.projected_hard_limit,
        multiplicative_hard_limit=args.multiplicative_hard_limit,
        initial_relation=args.initial_relation,
        growth_relation=args.growth_relation,
        knn_k=args.component_growth_knn,
        component_min_edges=args.component_growth_min_edges,
        hard_gate_mode=args.hard_gate_mode,
        seed_relaxed_hard_limit=args.seed_relaxed_hard_limit,
        growth_relaxed_hard_limit=args.growth_relaxed_hard_limit,
        local_knn_selectivity_threshold=args.local_knn_selectivity_threshold,
        local_knn_selectivity_scope=args.local_knn_selectivity_scope,
        bridge_pruning=args.bridge_pruning,
        bridge_max_hops=args.bridge_max_hops,
        open_space_relaxation=args.open_space_relaxation,
        outer_long_ratio=args.outer_long_ratio,
        outer_relaxation=args.outer_relaxation,
        outer_transition_width=args.outer_transition_width,
    )

    pre_merge_labels = _labels_from_components(
        state["labels"], state["edge_counts"], args.component_growth_min_edges
    )
    merged_labels, merge_diagnostics = _merge_component_outliers(
        X, state["edge_keys"], state["orig_edge_sizes"],
        state["final_mask"], state["labels"], args.component_growth_min_edges,
        args.component_growth_outlier_limit,
    )
    state["pre_merge_labels"] = pre_merge_labels
    state["merged_labels"] = merged_labels
    state["outlier_merge"] = merge_diagnostics

    # Convert the component-level ``large_components`` mask into point labels:
    # each sufficiently large connected component keeps its component ID,
    # while points in smaller components are treated as noise (-1).  This is
    # the meaningful ARI comparison; comparing only the boolean mask would
    # collapse all large components into one class.
    y_true = None
    if "y" in npz.files:
        y_true = np.asarray(npz["y"], dtype=np.int64)
    elif "y_clean" in npz.files:
        y_true = np.asarray(npz["y_clean"], dtype=np.int64)
    large_point_labels = np.full(len(X), -1, dtype=np.int64)
    for component, is_large in enumerate(state["large_components"]):
        if is_large:
            large_point_labels[state["labels"] == component] = component
    if args.component_growth_outlier_limit is not None:
        # Report the post-merge labels when the optional merge is enabled.
        large_point_labels = np.asarray(state["merged_labels"], dtype=np.int64)
    if y_true is not None and len(y_true) == len(large_point_labels):
        state["large_component_ari"] = float(
            adjusted_rand_score(y_true, large_point_labels)
        )
    else:
        state["large_component_ari"] = float("nan")
    state["large_component_point_labels"] = large_point_labels

    fig, axes = plt.subplots(2, 2, figsize=(18, 15), constrained_layout=True)
    _draw_hard(axes[0, 0], state["X_proj"], state)
    _draw_initial(axes[0, 1], state["X_proj"], state)
    _draw_growth(axes[1, 0], state["X_proj"], state)
    _draw_final(axes[1, 1], state["X_proj"], state)
    fig.suptitle(
        f"DelTriC component growth — {args.data.stem} "
        f"({args.dim_reduction}, topology dimension={args.project_dim})",
        y=1.01,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "dataset": str(args.data),
        "dim_reduction": args.dim_reduction,
        "project_dim": args.project_dim,
        "umap_n_epochs": args.umap_n_epochs,
        "umap_n_neighbors": args.umap_n_neighbors,
        "hard_limit": args.hard_limit,
        "seed_hard_limit": state["seed_hard_limit"],
        "growth_hard_limit": state["growth_hard_limit"],
        "mild_limit": args.mild_limit,
        "projected_hard_limit": args.projected_hard_limit,
        "multiplicative_hard_limit": args.multiplicative_hard_limit,
        "initial_relation": args.initial_relation,
        "growth_relation": args.growth_relation,
        "component_growth_knn": args.component_growth_knn,
        "component_growth_min_edges": args.component_growth_min_edges,
        "hard_gate_mode": args.hard_gate_mode,
        "seed_relaxed_hard_limit": args.seed_relaxed_hard_limit,
        "growth_relaxed_hard_limit": args.growth_relaxed_hard_limit,
        "local_knn_selectivity_threshold": args.local_knn_selectivity_threshold,
        "local_knn_selectivity_scope": args.local_knn_selectivity_scope,
        "bridge_pruning": args.bridge_pruning,
        "bridge_max_hops": args.bridge_max_hops,
        "component_growth_outlier_limit": args.component_growth_outlier_limit,
        "outlier_merge_candidates": merge_diagnostics["candidate_count"],
        "outlier_merged_points": merge_diagnostics["merged_count"],
        "open_space_relaxation": args.open_space_relaxation,
        "outer_long_ratio": args.outer_long_ratio,
        "outer_relaxation": args.outer_relaxation,
        "outer_transition_width": args.outer_transition_width,
        "open_space_growth_edges": int(
            np.count_nonzero(state["open_space_growth_mask"])
            if state["open_space_growth_mask"] is not None else 0
        ),
        "original_hard_threshold": state["orig_hard_threshold"],
        "projected_hard_threshold": state["projected_hard_threshold"],
        "initial_edges": int(state["initial_mask"].sum()),
        "final_edges": int(state["final_mask"].sum()),
        "growth_edges": int(state["grown_mask"].sum()),
        "automatic_growth_edges": int(state["automatic_growth_mask"].sum()),
        "blue_growth_edges": int(state["blue_growth_mask"].sum()),
        "bridge_pruned_edges": int(state["bridge_pruned_mask"].sum()),
        "final_components": int(len(state["edge_counts"])),
        "final_large_components": int(np.count_nonzero(state["large_components"])),
        "large_component_ari": state["large_component_ari"],
        "large_component_point_coverage": float(np.mean(large_point_labels >= 0)),
    }
    summary_path = args.out.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
