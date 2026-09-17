#!/usr/bin/env python3
"""Grow seed components in rounds by a component-relative hard criterion.

This is a diagnostic experiment, separate from the production component-growth
rule.  It starts from the exact ``initial_mask`` seed graph.  In each round it
evaluates every current boundary edge using that round's component statistics,
then accepts every qualifying edge with

    (edge_length - component_median) / (component_q75 - component_median)

when that value is below ``--hard-growth-limit``.  Component statistics are
recomputed only after the complete round.  Boundary colors are frozen the first
time an edge is encountered, so the plot records the values used during growth.
"""

from __future__ import annotations

import argparse
from collections import deque
import heapq
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import adjusted_rand_score, f1_score, precision_score, recall_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils_component_growth import component_growth_graph  # noqa: E402
from hdbscan_comparison import fit_hdbscan, result_metrics  # noqa: E402
from plot_seed_component_metrics import (  # noqa: E402
    _draw_base,
    _draw_seed_truth,
    _seed_components,
    _segments,
)


# Selected by ``results/hdbscan_epsilon_mcs10_default`` on the 12 curated
# non-1000D data sets.  The epsilon-only sweep retained the conventional
# epsilon=0.0; min_samples=None preserves HDBSCAN's default behavior.
HDBSCAN_MIN_CLUSTER_SIZE = 10
HDBSCAN_MIN_SAMPLES: int | None = None
HDBSCAN_SELECTION_METHOD = "eom"
HDBSCAN_CLUSTER_SELECTION_EPSILON = 0.0


def _draw_growth_gate_classification(
    ax,
    X_proj: np.ndarray,
    segments: np.ndarray,
    large_points: np.ndarray,
    seed_mask: np.ndarray,
    seen_mask: np.ndarray,
    original_ratio: np.ndarray,
    projected_ratio: np.ndarray,
    *,
    hard_growth_limit: float,
    projected_hard_growth_limit: float,
) -> None:
    """Show which gate blocked each boundary edge at first evaluation."""
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    _draw_base(
        ax, X_proj, segments, large_points, seed_mask,
        "first boundary decision: original vs projected hard gate",
    )
    if not np.any(seen_mask):
        ax.text(0.5, 0.5, "no large-seed boundary edges", transform=ax.transAxes,
                ha="center", va="center", color="#555555", fontsize=10)
        return

    original_pass = original_ratio < float(hard_growth_limit)
    projected_pass = projected_ratio < float(projected_hard_growth_limit)
    categories = (
        (seen_mask & original_pass & projected_pass, "accepted", "#2ca25f"),
        (seen_mask & ~original_pass & projected_pass,
         "blocked: original-space", "#d73027"),
        (seen_mask & original_pass & ~projected_pass,
         "blocked: projected-space", "#2c7bb6"),
        (seen_mask & ~original_pass & ~projected_pass,
         "blocked: both", "#7b3294"),
    )
    handles = []
    for mask, label, colour in categories:
        count = int(np.count_nonzero(mask))
        if count:
            ax.add_collection(LineCollection(
                segments[mask], colors=colour, linewidths=1.15,
                alpha=0.95, zorder=5,
            ))
        handles.append(Line2D([0], [0], color=colour, lw=2,
                              label=f"{label} ({count})"))
    ax.legend(handles=handles, loc="lower right", fontsize=7.5,
              frameon=True, framealpha=0.88)


def _draw_hdbscan_final(
    ax,
    X_display: np.ndarray,
    labels: np.ndarray,
    *,
    ari: float,
    min_cluster_size: int,
    min_samples: int | None,
    selection_method: str,
    cluster_selection_epsilon: float,
) -> None:
    """Draw original-space HDBSCAN labels in DelTriC's exact display space."""
    import matplotlib.pyplot as plt

    colours = np.full((len(labels), 4), (0.62, 0.62, 0.62, 0.55), dtype=float)
    cluster_ids = np.unique(labels[labels >= 0])
    cmap = plt.get_cmap("turbo", max(2, len(cluster_ids)))
    for colour_index, cluster_id in enumerate(cluster_ids):
        colours[labels == cluster_id] = cmap(colour_index)
    ax.scatter(
        X_display[:, 0], X_display[:, 1], c=colours, s=5.2,
        linewidths=0, rasterized=True,
    )
    ax.set_title(
        "HDBSCAN, standardized original space\n"
        f"mcs={min_cluster_size}, ms={min_samples if min_samples is not None else 'default'}, "
        f"eps={cluster_selection_epsilon:g}, {selection_method}; "
        f"ARI={ari:.3f}",
        fontsize=11,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.autoscale()
    ax.set_xticks([])
    ax.set_yticks([])


def _projected_recall10(X: np.ndarray, X_proj: np.ndarray) -> np.ndarray:
    n_points = len(X)
    k = min(10, n_points - 1)
    original = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(
        X, return_distance=False
    )[:, 1:]
    projected = NearestNeighbors(n_neighbors=k + 1).fit(X_proj).kneighbors(
        X_proj, return_distance=False
    )[:, 1:]
    return np.array([
        len(set(original[point]) & set(projected[point])) / k
        for point in range(n_points)
    ], dtype=np.float64)


def _final_graph_components(edge_keys: np.ndarray, retained_mask: np.ndarray,
                            n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Return labels and an edge-bearing mask for the final retained graph."""
    retained_edges = edge_keys[retained_mask]
    if not len(retained_edges):
        return np.arange(n_points, dtype=np.int64), np.zeros(n_points, dtype=bool)
    graph = coo_matrix(
        (np.ones(2 * len(retained_edges), dtype=np.int8),
         (np.concatenate((retained_edges[:, 0], retained_edges[:, 1])),
          np.concatenate((retained_edges[:, 1], retained_edges[:, 0])))),
        shape=(n_points, n_points),
    ).tocsr()
    n_components, labels = connected_components(graph, directed=False)
    edge_counts = np.zeros(n_components, dtype=np.int64)
    np.add.at(edge_counts, labels[retained_edges[:, 0]], 1)
    return labels.astype(np.int64), edge_counts[labels] > 0


def _outer_hull_edge_mask(edge_keys: np.ndarray, retained_mask: np.ndarray,
                          triangles: np.ndarray) -> np.ndarray:
    """Return retained edges adjacent to the global exterior of the graph.

    Retained Delaunay edges are barriers in the triangle dual graph.  Flooding
    from the mesh exterior through absent edges excludes inner holes and
    nested islands from the hull.
    """
    triangles = np.asarray(triangles, dtype=np.int64)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(
            "outer-hull metrics require a 2D Delaunay triangulation "
            "with triangular simplices"
        )
    edge_index_by_key = {
        (int(u), int(v)): index for index, (u, v) in enumerate(edge_keys)
    }
    triangle_edge_indices = np.empty((len(triangles), 3), dtype=np.int64)
    for triangle_index, (a, b, c) in enumerate(triangles):
        triangle_edge_indices[triangle_index] = (
            edge_index_by_key[tuple(sorted((int(a), int(b))))],
            edge_index_by_key[tuple(sorted((int(a), int(c))))],
            edge_index_by_key[tuple(sorted((int(b), int(c))))],
        )
    edge_triangles: list[list[int]] = [[] for _ in range(len(edge_keys))]
    for triangle_index, triangle_edges in enumerate(triangle_edge_indices):
        for edge_index in triangle_edges:
            edge_triangles[int(edge_index)].append(triangle_index)

    exterior_faces = np.zeros(len(triangles), dtype=bool)
    queue: deque[int] = deque()
    for edge_index, triangle_indices in enumerate(edge_triangles):
        if len(triangle_indices) == 1 and not retained_mask[edge_index]:
            triangle_index = triangle_indices[0]
            exterior_faces[triangle_index] = True
            queue.append(triangle_index)
    while queue:
        triangle_index = queue.popleft()
        for edge_index in triangle_edge_indices[triangle_index]:
            if retained_mask[int(edge_index)]:
                continue
            for neighbour_triangle in edge_triangles[int(edge_index)]:
                if not exterior_faces[neighbour_triangle]:
                    exterior_faces[neighbour_triangle] = True
                    queue.append(neighbour_triangle)

    hull_edge_mask = np.zeros(len(edge_keys), dtype=bool)
    for edge_index in np.flatnonzero(retained_mask):
        incident_triangles = edge_triangles[int(edge_index)]
        hull_edge_mask[edge_index] = (
            len(incident_triangles) == 1
            or any(exterior_faces[triangle] for triangle in incident_triangles)
        )
    return hull_edge_mask


def _gomory_hu_joint_cut_mask(
    edge_keys: np.ndarray,
    retained_mask: np.ndarray,
    *,
    cut_size: int,
    min_component_points: int,
    removable_mask: np.ndarray | None = None,
    hull_ratio_skip_threshold: float = 0.0,
    triangles: np.ndarray | None = None,
    edge_sizes: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Return eligible fundamental Gomory--Hu cuts with size <= cut_size.

    Both sides must contain ``min_component_points`` graph points.  That
    excludes cuts which only detach outliers or small nested islands.  When
    ``removable_mask`` is supplied, only those retained edges may be cut;
    every other retained edge has capacity ``cut_size + 1`` and is therefore
    excluded from every eligible cut.
    """
    if min_component_points < 1:
        raise ValueError("min_component_points must be positive")
    if hull_ratio_skip_threshold < 0.0:
        raise ValueError("hull_ratio_skip_threshold must be non-negative")
    if hull_ratio_skip_threshold > 0.0 and (triangles is None or edge_sizes is None):
        raise ValueError(
            "triangles and original-space edge_sizes are required when "
            "hull_ratio_skip_threshold is enabled"
        )
    if cut_size <= 0:
        return np.zeros(len(edge_keys), dtype=bool), {
            "enabled": False, "cut_size": int(cut_size),
            "min_component_points": int(min_component_points),
            "hull_ratio_skip_threshold": float(hull_ratio_skip_threshold),
            "component_count": 0, "skipped_too_small_component_count": 0,
            "skipped_hull_ratio_component_count": 0,
            "eligible_tree_cut_count": 0,
            "rejected_small_side_tree_cut_count": 0, "pruned_edge_count": 0,
        }

    if removable_mask is not None:
        removable_mask = np.asarray(removable_mask, dtype=bool)
        if removable_mask.shape != (len(edge_keys),):
            raise ValueError("removable_mask must contain one value per edge")
        if np.any(removable_mask & ~retained_mask):
            raise ValueError("removable_mask may contain only retained edges")

    import networkx as nx

    labels, edge_bearing_points = _final_graph_components(
        edge_keys, retained_mask, int(edge_keys.max()) + 1,
    )
    retained_indices = np.flatnonzero(retained_mask)
    if hull_ratio_skip_threshold > 0.0:
        hull_edge_mask = _outer_hull_edge_mask(edge_keys, retained_mask, triangles)
        edge_sizes = np.asarray(edge_sizes, dtype=np.float64)
        if edge_sizes.shape != (len(edge_keys),):
            raise ValueError("edge_sizes must contain one value per edge")
    else:
        hull_edge_mask = None
    pruned_mask = np.zeros(len(edge_keys), dtype=bool)
    component_summaries: list[dict] = []
    for component in np.unique(labels[edge_bearing_points]):
        edge_indices = retained_indices[
            (labels[edge_keys[retained_indices, 0]] == component)
            & (labels[edge_keys[retained_indices, 1]] == component)
        ]
        if not len(edge_indices):
            continue
        nodes = np.unique(edge_keys[edge_indices].ravel())
        hull_ratio = None
        if hull_edge_mask is not None:
            component_median = max(float(np.median(edge_sizes[edge_indices])), 1e-12)
            component_hull = hull_edge_mask[edge_indices]
            non_hull_count = int(len(edge_indices) - np.count_nonzero(component_hull))
            hull_ratio = (
                float(np.sum(edge_sizes[edge_indices][component_hull] / component_median))
                / non_hull_count if non_hull_count else float("inf")
            )
        if len(nodes) < 2 * min_component_points:
            component_summaries.append({
                "edge_count": int(len(edge_indices)), "point_count": int(len(nodes)),
                "hull_to_non_hull_edge_ratio": hull_ratio,
                "skipped_too_small": True, "skipped_hull_ratio": False,
                "eligible_tree_cut_count": 0,
                "rejected_small_side_tree_cut_count": 0, "pruned_edge_count": 0,
            })
            continue
        if hull_ratio is not None and hull_ratio >= hull_ratio_skip_threshold:
            component_summaries.append({
                "edge_count": int(len(edge_indices)), "point_count": int(len(nodes)),
                "hull_to_non_hull_edge_ratio": hull_ratio,
                "skipped_too_small": False, "skipped_hull_ratio": True,
                "eligible_tree_cut_count": 0,
                "rejected_small_side_tree_cut_count": 0, "pruned_edge_count": 0,
            })
            continue
        graph = nx.Graph()
        graph.add_nodes_from(map(int, nodes))
        protected_capacity = int(cut_size) + 1
        graph.add_edges_from(
            (
                int(edge_keys[index, 0]), int(edge_keys[index, 1]),
                {"capacity": 1 if removable_mask is None or removable_mask[index]
                 else protected_capacity},
            )
            for index in edge_indices
        )
        tree = nx.gomory_hu_tree(graph, capacity="capacity")
        eligible_cuts = 0
        rejected_small_side_cuts = 0
        before = int(np.count_nonzero(pruned_mask[edge_indices]))
        for left, right, data in list(tree.edges(data=True)):
            if int(round(data["weight"])) > cut_size:
                continue
            tree.remove_edge(left, right)
            left_side = nx.node_connected_component(tree, left)
            tree.add_edge(left, right, **data)
            if min(len(left_side), len(nodes) - len(left_side)) < min_component_points:
                rejected_small_side_cuts += 1
                continue
            on_left_u = np.isin(edge_keys[edge_indices, 0], list(left_side))
            on_left_v = np.isin(edge_keys[edge_indices, 1], list(left_side))
            cut_edges = edge_indices[on_left_u != on_left_v]
            if removable_mask is not None:
                cut_edges = cut_edges[removable_mask[cut_edges]]
            if len(cut_edges):
                pruned_mask[cut_edges] = True
                eligible_cuts += 1
        component_summaries.append({
            "edge_count": int(len(edge_indices)), "point_count": int(len(nodes)),
            "hull_to_non_hull_edge_ratio": hull_ratio,
            "skipped_too_small": False, "skipped_hull_ratio": False,
            "eligible_tree_cut_count": eligible_cuts,
            "rejected_small_side_tree_cut_count": rejected_small_side_cuts,
            "pruned_edge_count": int(np.count_nonzero(pruned_mask[edge_indices])) - before,
        })

    component_summaries.sort(key=lambda item: item["edge_count"], reverse=True)
    return pruned_mask, {
        "enabled": True, "cut_size": int(cut_size),
        "min_component_points": int(min_component_points),
        "hull_ratio_skip_threshold": float(hull_ratio_skip_threshold),
        "component_count": len(component_summaries),
        "skipped_too_small_component_count": int(sum(
            item["skipped_too_small"] for item in component_summaries
        )),
        "skipped_hull_ratio_component_count": int(sum(
            item["skipped_hull_ratio"] for item in component_summaries
        )),
        "eligible_tree_cut_count": int(sum(
            item["eligible_tree_cut_count"] for item in component_summaries
        )),
        "rejected_small_side_tree_cut_count": int(sum(
            item["rejected_small_side_tree_cut_count"] for item in component_summaries
        )),
        "pruned_edge_count": int(np.count_nonzero(pruned_mask)),
        "largest_components": component_summaries[:5],
    }


def _gomory_hu_contracted_candidate_cut_mask(
    edge_keys: np.ndarray,
    candidate_mask: np.ndarray,
    seed_labels: np.ndarray,
    *,
    cut_size: int,
    min_component_points: int,
) -> tuple[np.ndarray, dict]:
    """Find low-cardinality GH cuts after contracting original seed components.

    The graph contains *only* candidate growth-related edges.  Every original
    seed component is contracted to one weighted vertex, so seed-internal
    edges never enter a max-flow calculation.  A Gomory--Hu tree cut is still
    evaluated using the number of original points on both sides.
    """
    if min_component_points < 1:
        raise ValueError("min_component_points must be positive")
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    if candidate_mask.shape != (len(edge_keys),):
        raise ValueError("candidate_mask must contain one value per edge")
    pruned_mask = np.zeros(len(edge_keys), dtype=bool)
    if cut_size <= 0 or not np.any(candidate_mask):
        return pruned_mask, {
            "enabled": bool(cut_size > 0), "graph_mode": "contracted_candidates",
            "cut_size": int(cut_size), "min_component_points": int(min_component_points),
            "component_count": 0, "contracted_node_count": 0,
            "skipped_too_small_component_count": 0,
            "eligible_tree_cut_count": 0,
            "rejected_small_side_tree_cut_count": 0, "pruned_edge_count": 0,
            "largest_components": [],
        }

    import networkx as nx

    seed_labels = np.asarray(seed_labels, dtype=np.int64)
    seed_sizes = np.bincount(seed_labels)
    candidate_indices = np.flatnonzero(candidate_mask)
    candidate_seed_u = seed_labels[edge_keys[candidate_indices, 0]]
    candidate_seed_v = seed_labels[edge_keys[candidate_indices, 1]]
    involved_seeds = np.unique(np.concatenate((candidate_seed_u, candidate_seed_v)))
    contracted_index = np.full(len(seed_sizes), -1, dtype=np.int64)
    contracted_index[involved_seeds] = np.arange(len(involved_seeds), dtype=np.int64)
    candidate_u = contracted_index[candidate_seed_u]
    candidate_v = contracted_index[candidate_seed_v]
    cross_seed = candidate_u != candidate_v
    candidate_indices = candidate_indices[cross_seed]
    candidate_u = candidate_u[cross_seed]
    candidate_v = candidate_v[cross_seed]
    node_weights = seed_sizes[involved_seeds].astype(np.int64, copy=False)
    if not len(candidate_indices):
        return pruned_mask, {
            "enabled": True, "graph_mode": "contracted_candidates",
            "cut_size": int(cut_size), "min_component_points": int(min_component_points),
            "component_count": 0, "contracted_node_count": int(len(involved_seeds)),
            "skipped_too_small_component_count": 0,
            "eligible_tree_cut_count": 0,
            "rejected_small_side_tree_cut_count": 0, "pruned_edge_count": 0,
            "largest_components": [],
        }

    graph = nx.Graph()
    graph.add_nodes_from(range(len(involved_seeds)))
    for left, right in zip(candidate_u, candidate_v):
        left, right = int(left), int(right)
        if graph.has_edge(left, right):
            graph[left][right]["capacity"] += 1
        else:
            graph.add_edge(left, right, capacity=1)

    component_summaries: list[dict] = []
    for nodes in nx.connected_components(graph):
        nodes = np.fromiter(nodes, dtype=np.int64)
        point_count = int(np.sum(node_weights[nodes]))
        component_edge_mask = np.isin(candidate_u, nodes) & np.isin(candidate_v, nodes)
        component_edge_indices = candidate_indices[component_edge_mask]
        if point_count < 2 * min_component_points:
            component_summaries.append({
                "edge_count": int(len(component_edge_indices)), "point_count": point_count,
                "contracted_point_count": int(len(nodes)), "skipped_too_small": True,
                "eligible_tree_cut_count": 0,
                "rejected_small_side_tree_cut_count": 0, "pruned_edge_count": 0,
            })
            continue
        tree = nx.gomory_hu_tree(graph.subgraph(nodes).copy(), capacity="capacity")
        eligible_cuts = 0
        rejected_small_side_cuts = 0
        before = int(np.count_nonzero(pruned_mask[component_edge_indices]))
        for left, right, data in list(tree.edges(data=True)):
            if int(round(data["weight"])) > cut_size:
                continue
            tree.remove_edge(left, right)
            left_side = nx.node_connected_component(tree, left)
            tree.add_edge(left, right, **data)
            left_side = np.fromiter(left_side, dtype=np.int64)
            left_points = int(np.sum(node_weights[left_side]))
            if min(left_points, point_count - left_points) < min_component_points:
                rejected_small_side_cuts += 1
                continue
            on_left = np.zeros(len(involved_seeds), dtype=bool)
            on_left[left_side] = True
            cut_edges = component_edge_indices[
                on_left[candidate_u[component_edge_mask]]
                != on_left[candidate_v[component_edge_mask]]
            ]
            if len(cut_edges):
                pruned_mask[cut_edges] = True
                eligible_cuts += 1
        component_summaries.append({
            "edge_count": int(len(component_edge_indices)), "point_count": point_count,
            "contracted_point_count": int(len(nodes)), "skipped_too_small": False,
            "eligible_tree_cut_count": eligible_cuts,
            "rejected_small_side_tree_cut_count": rejected_small_side_cuts,
            "pruned_edge_count": int(np.count_nonzero(pruned_mask[component_edge_indices])) - before,
        })

    component_summaries.sort(key=lambda item: item["edge_count"], reverse=True)
    return pruned_mask, {
        "enabled": True, "graph_mode": "contracted_candidates",
        "cut_size": int(cut_size), "min_component_points": int(min_component_points),
        "component_count": len(component_summaries),
        "contracted_node_count": int(len(involved_seeds)),
        "skipped_too_small_component_count": int(sum(
            item["skipped_too_small"] for item in component_summaries
        )),
        "eligible_tree_cut_count": int(sum(
            item["eligible_tree_cut_count"] for item in component_summaries
        )),
        "rejected_small_side_tree_cut_count": int(sum(
            item["rejected_small_side_tree_cut_count"] for item in component_summaries
        )),
        "pruned_edge_count": int(np.count_nonzero(pruned_mask)),
        "largest_components": component_summaries[:5],
    }


def _edge_redundancy_pruning(
    edge_keys: np.ndarray,
    final_mask: np.ndarray,
    *,
    redundancy_threshold: int,
    redundancy_prune_limit: int,
    low_redundancy_majority: float,
    max_hops: int | None = None,
    path_distinct_fraction: float = 1.0,
    triangles: np.ndarray | None = None,
    edge_sizes: np.ndarray | None = None,
) -> dict:
    """Final, simultaneous bridge pruning based on edge-disjoint paths.

    For an edge ``u--v``, its redundancy is the number of edge-disjoint
    alternatives between ``u`` and ``v`` after removing ``u--v``.  The count
    is capped at ``redundancy_threshold`` because larger values cannot affect
    this rule.  When ``max_hops`` is set, every augmenting-path search is
    restricted to paths with at most that many hops.  With
    ``path_distinct_fraction < 1``, each additionally counted path must add at
    least that fraction of previously unused edges relative to the union of
    already accepted paths.  This diversity mode is a greedy bounded-hop
    packing; the 1.0 edge-disjoint mode remains exact max-flow.  Each
    component is evaluated on the same pre-pruning graph.
    """
    if redundancy_threshold < 1 or redundancy_prune_limit < 1:
        raise ValueError("redundancy thresholds must be positive")
    if not 0.0 <= low_redundancy_majority <= 1.0:
        raise ValueError("low_redundancy_majority must be in [0, 1]")
    if max_hops is not None and int(max_hops) < 1:
        raise ValueError("max_hops must be at least 1 or None")
    if not 0.0 < float(path_distinct_fraction) <= 1.0:
        raise ValueError("path_distinct_fraction must be in (0, 1]")
    if path_distinct_fraction < 1.0 and max_hops is None:
        raise ValueError(
            "a finite max_hops is required when path_distinct_fraction < 1"
        )
    if triangles is not None:
        triangles = np.asarray(triangles, dtype=np.int64)
        if triangles.ndim != 2 or triangles.shape[1] != 3:
            raise ValueError(
                "hull-edge metrics require a 2D Delaunay triangulation "
                "with triangular simplices"
            )
    if edge_sizes is not None:
        edge_sizes = np.asarray(edge_sizes, dtype=np.float64)
        if edge_sizes.shape != (len(edge_keys),):
            raise ValueError("edge_sizes must contain one value per edge")

    n_points = int(edge_keys.max()) + 1
    labels, edge_bearing_points = _final_graph_components(
        edge_keys, final_mask, n_points,
    )
    retained_indices = np.flatnonzero(final_mask)
    redundancy = np.full(len(edge_keys), -1, dtype=np.int16)
    pruned_mask = np.zeros(len(edge_keys), dtype=bool)
    skipped_component_mask = np.zeros(len(edge_keys), dtype=bool)
    component_metrics: list[dict] = []

    # The residual search is exact for undirected unit-capacity edges.  Unlike
    # a generic max-flow call it stops as soon as the result reaches the
    # configured threshold, which is all the subsequent decision needs.
    adjacency = [[] for _ in range(n_points)]
    for edge_index in retained_indices:
        u, v = map(int, edge_keys[edge_index])
        adjacency[u].append((v, int(edge_index)))
        adjacency[v].append((u, int(edge_index)))

    hull_geometry_available = triangles is not None
    hull_edge_mask = (
        _outer_hull_edge_mask(edge_keys, final_mask, triangles)
        if hull_geometry_available else np.zeros(len(edge_keys), dtype=bool)
    )
    visited = np.zeros(n_points, dtype=np.int64)
    parent_node = np.empty(n_points, dtype=np.int64)
    parent_edge = np.empty(n_points, dtype=np.int64)
    search_id = 0

    def count_edge_disjoint_paths(start: int, target: int,
                                  excluded_edge: int, limit: int) -> int:
        nonlocal search_id
        # `flow[edge]` is -1, 0, or +1 with respect to the orientation stored
        # in edge_keys.  Residual reverse traversals undo a prior path, so the
        # repeated breadth-first searches implement Edmonds--Karp correctly.
        flow: dict[int, int] = {}
        count = 0
        while count < limit:
            search_id += 1
            visited[start] = search_id
            queue = deque([(start, 0)])
            found = False
            while queue and not found:
                node, hops = queue.popleft()
                if max_hops is not None and hops >= max_hops:
                    continue
                for neighbour, edge_index in adjacency[node]:
                    if edge_index == excluded_edge or visited[neighbour] == search_id:
                        continue
                    direction = 1 if edge_keys[edge_index, 0] == node else -1
                    current_flow = flow.get(edge_index, 0)
                    if ((direction == 1 and current_flow >= 1)
                            or (direction == -1 and current_flow <= -1)):
                        continue
                    visited[neighbour] = search_id
                    parent_node[neighbour] = node
                    parent_edge[neighbour] = edge_index
                    if neighbour == target:
                        found = True
                        break
                    queue.append((neighbour, hops + 1))
            if not found:
                break
            node = target
            while node != start:
                previous = int(parent_node[node])
                edge_index = int(parent_edge[node])
                direction = 1 if edge_keys[edge_index, 0] == previous else -1
                next_flow = flow.get(edge_index, 0) + direction
                if next_flow:
                    flow[edge_index] = next_flow
                else:
                    flow.pop(edge_index, None)
                node = previous
            count += 1
        return count

    def count_diverse_paths(start: int, target: int, excluded_edge: int,
                            limit: int) -> int:
        """Greedily pack short paths that each contribute enough new edges."""
        accepted_edges: set[int] = set()
        count = 0
        # This branch requires a finite limit above, so the state search is
        # bounded and never follows arbitrary long detours.
        assert max_hops is not None
        while count < limit:
            queue = deque([(start, 0, 0, -1)])
            seen_states = {(start, 0, 0, -1)}
            parents: dict[tuple[int, int, int, int], tuple[
                tuple[int, int, int, int], int
            ]] = {}
            accepted_path: list[int] | None = None
            while queue and accepted_path is None:
                node, hops, new_edges, previous_node = queue.popleft()
                if node == target and hops:
                    if new_edges / hops >= path_distinct_fraction:
                        state = (node, hops, new_edges, previous_node)
                        path: list[int] = []
                        while state in parents:
                            state, edge_index = parents[state]
                            path.append(edge_index)
                        accepted_path = path
                    continue
                if hops >= max_hops:
                    continue
                for neighbour, edge_index in adjacency[node]:
                    if edge_index == excluded_edge or neighbour == previous_node:
                        continue
                    next_hops = hops + 1
                    next_new_edges = new_edges + int(edge_index not in accepted_edges)
                    state = (neighbour, next_hops, next_new_edges, node)
                    if state in seen_states:
                        continue
                    seen_states.add(state)
                    parents[state] = ((node, hops, new_edges, previous_node), edge_index)
                    queue.append(state)
            if accepted_path is None:
                break
            accepted_edges.update(accepted_path)
            count += 1
        return count

    for component in np.unique(labels[edge_bearing_points]):
        edge_indices = retained_indices[
            (labels[edge_keys[retained_indices, 0]] == component)
            & (labels[edge_keys[retained_indices, 1]] == component)
        ]
        for edge_index in edge_indices:
            u, v = map(int, edge_keys[edge_index])
            if path_distinct_fraction >= 1.0:
                redundancy[edge_index] = count_edge_disjoint_paths(
                    u, v, int(edge_index), redundancy_threshold,
                )
            else:
                redundancy[edge_index] = count_diverse_paths(
                    u, v, int(edge_index), redundancy_threshold,
                )

        low_fraction = float(np.mean(
            redundancy[edge_indices] < redundancy_threshold,
        ))
        skipped = low_fraction > low_redundancy_majority
        if skipped:
            skipped_component_mask[edge_indices] = True
        else:
            pruned_mask[edge_indices] = (
                redundancy[edge_indices] < redundancy_prune_limit
            )
        component_metric = {
            "edge_count": int(len(edge_indices)),
            "low_redundancy_fraction": low_fraction,
            "skipped": skipped,
            "pruned_edge_count": int(np.count_nonzero(pruned_mask[edge_indices])),
        }
        if hull_geometry_available:
            hull_edge = hull_edge_mask[edge_indices]
            hull_edge_count = int(np.count_nonzero(hull_edge))
            non_hull_edge_count = int(len(edge_indices) - hull_edge_count)
            component_metric.update({
                "hull_edge_count": hull_edge_count,
                "non_hull_edge_count": non_hull_edge_count,
            })
            if edge_sizes is not None:
                component_median_edge_size = float(np.median(
                    edge_sizes[edge_indices],
                ))
                normalized_hull_length = float(np.sum(
                    edge_sizes[edge_indices][hull_edge]
                    / max(component_median_edge_size, 1e-12),
                ))
                component_metric.update({
                    "component_median_edge_size": component_median_edge_size,
                    "normalized_hull_length": normalized_hull_length,
                    "hull_to_non_hull_edge_ratio": (
                        normalized_hull_length / non_hull_edge_count
                        if non_hull_edge_count else None
                    ),
                })
            else:
                component_metric["hull_to_non_hull_edge_ratio"] = (
                    hull_edge_count / non_hull_edge_count
                    if non_hull_edge_count else None
                )
        component_metrics.append(component_metric)

    evaluated = redundancy >= 0
    histogram = {
        str(level): int(np.count_nonzero(redundancy[evaluated] == level))
        for level in range(redundancy_threshold + 1)
    }
    component_metrics.sort(key=lambda item: item["edge_count"], reverse=True)

    return {
        "redundancy": redundancy,
        "pruned_mask": pruned_mask,
        "skipped_component_mask": skipped_component_mask,
        "hull_edge_mask": hull_edge_mask,
        "metrics": {
            "evaluated_edge_count": int(np.count_nonzero(evaluated)),
            "capped_redundancy_histogram": histogram,
            "low_redundancy_edge_fraction": float(np.mean(
                redundancy[evaluated] < redundancy_threshold,
            )),
            "component_count": len(component_metrics),
            "max_hops": max_hops,
            "path_distinct_fraction": path_distinct_fraction,
            "hull_edge_definition": (
                "retained edge adjacent to the global exterior in the "
                "Delaunay triangle dual graph; enclosed-hole and nested "
                "island boundaries are excluded"
                if triangles is not None else None
            ),
            "hull_ratio_definition": (
                "sum(hull_edge_length / component_median_edge_length) / "
                "non_hull_edge_count"
                if triangles is not None and edge_sizes is not None else None
            ),
            "skipped_component_count": int(sum(
                item["skipped"] for item in component_metrics
            )),
            "pruned_component_count": int(sum(
                not item["skipped"] for item in component_metrics
            )),
            "largest_components": component_metrics[:5],
        },
    }


def _draw_final_components(ax, X_proj: np.ndarray, edge_keys: np.ndarray,
                           segments: np.ndarray, retained_mask: np.ndarray,
                           redundancy_pruned_mask: np.ndarray | None = None,
                           assigned_labels: np.ndarray | None = None,
                           added_outlier_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Draw final graph components and optional post-graph point assignments."""
    from matplotlib.collections import LineCollection
    import matplotlib.pyplot as plt

    ax.add_collection(LineCollection(
        segments, colors="#d0d0d0", linewidths=0.3, alpha=0.28, zorder=1,
    ))
    labels, component_point_mask = _final_graph_components(
        edge_keys, retained_mask, len(X_proj),
    )
    display_labels = labels if assigned_labels is None else np.asarray(
        assigned_labels, dtype=np.int64,
    )
    displayed = display_labels >= 0
    component_ids = np.unique(display_labels[displayed])
    if len(component_ids):
        cmap = plt.get_cmap(
            "tab20" if len(component_ids) <= 20 else "turbo", len(component_ids),
        )
        for colour_index, component in enumerate(component_ids):
            point_mask = displayed & (display_labels == component)
            graph_point_mask = component_point_mask & (labels == component)
            edge_mask = (
                retained_mask & graph_point_mask[edge_keys[:, 0]]
                & graph_point_mask[edge_keys[:, 1]]
            )
            colour = cmap(colour_index)
            if np.any(edge_mask):
                ax.add_collection(LineCollection(
                    segments[edge_mask], colors=[colour], linewidths=1.05,
                    alpha=0.93, zorder=3,
                ))
            ax.scatter(X_proj[point_mask, 0], X_proj[point_mask, 1], s=5.2,
                       color=[colour], alpha=0.9, linewidths=0, zorder=4)
    inactive = ~displayed
    if np.any(inactive):
        ax.scatter(X_proj[inactive, 0], X_proj[inactive, 1], s=2.5,
                   color="#9e9e9e", alpha=0.42, linewidths=0, zorder=2)
    if added_outlier_mask is not None and np.any(added_outlier_mask):
        ax.scatter(
            X_proj[added_outlier_mask, 0], X_proj[added_outlier_mask, 1],
            s=16, facecolors="none", edgecolors="#111111", linewidths=0.45,
            zorder=5,
        )
    if redundancy_pruned_mask is not None and np.any(redundancy_pruned_mask):
        ax.add_collection(LineCollection(
            segments[redundancy_pruned_mask], colors="#f781bf", linewidths=1.25,
            linestyles="--", alpha=0.98, zorder=5,
        ))
    ax.set_title("final retained connected components", fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    ax.autoscale()
    ax.set_xticks([])
    ax.set_yticks([])
    return labels, component_point_mask


def _final_component_prediction(edge_keys: np.ndarray, retained_mask: np.ndarray,
                                labels: np.ndarray,
                                edge_bearing_points: np.ndarray,
                                minimum_edges: int) -> np.ndarray:
    """Large final components keep their ID; all other points become noise."""
    retained_indices = np.flatnonzero(retained_mask)
    component_edge_counts = np.zeros(int(labels.max()) + 1, dtype=np.int64)
    if len(retained_indices):
        np.add.at(
            component_edge_counts,
            labels[edge_keys[retained_indices, 0]],
            1,
        )
    predicted = np.full(len(labels), -1, dtype=np.int64)
    large = edge_bearing_points & (
        component_edge_counts[labels] > int(minimum_edges)
    )
    predicted[large] = labels[large]
    return predicted


def _grow_frozen_components(
    edge_keys: np.ndarray,
    original_lengths: np.ndarray,
    projected_lengths: np.ndarray,
    initial_labels: np.ndarray,
    *,
    hard_limit: float,
    projected_hard_limit: float,
) -> dict[str, np.ndarray | dict]:
    """Extend completed components without ever merging them together.

    This is a separate, round-based continuation of component growth.  The
    input labels identify the completed components that are frozen: an edge
    can be accepted only from a frozen component to an unassigned point.  An
    edge between two completed components is always ignored, so they cannot
    be merged or "eat" one another.
    """
    edge_keys = np.asarray(edge_keys, dtype=np.int64)
    original_lengths = np.asarray(original_lengths, dtype=np.float64)
    projected_lengths = np.asarray(projected_lengths, dtype=np.float64)
    labels = np.asarray(initial_labels, dtype=np.int64).copy()
    n_points = len(labels)
    active_ids = np.unique(labels[labels >= 0])
    members = {
        int(component): set(np.flatnonzero(labels == component))
        for component in active_ids
    }
    adjacency = [[] for _ in range(n_points)]
    for edge_index, (u, v) in enumerate(edge_keys):
        adjacency[int(u)].append(edge_index)
        adjacency[int(v)].append(edge_index)
    accepted_edges = np.zeros(len(edge_keys), dtype=bool)
    added_points = np.zeros(n_points, dtype=bool)
    accepted_per_round: list[int] = []
    round_index = 0

    while True:
        # For a point adjoining more than one frozen component, retain only
        # its best accepted boundary edge for this round.  Thus components
        # stay distinct even when they surround the same outlier.
        proposals: dict[int, tuple[tuple[float, float, float, int, int], int, int]] = {}
        for component in sorted(members):
            component_incident = set()
            for point in members[component]:
                component_incident.update(adjacency[point])
            if not component_incident:
                continue
            incident_indices = np.fromiter(component_incident, dtype=np.int64)
            _, median, q75 = np.percentile(
                original_lengths[incident_indices], [25.0, 50.0, 75.0],
            )
            original_spread = max(float(q75 - median), 1e-12)
            _, projected_median, projected_q75 = np.percentile(
                projected_lengths[incident_indices], [25.0, 50.0, 75.0],
            )
            projected_spread = max(
                float(projected_q75 - projected_median), 1e-12,
            )
            for edge_index in incident_indices:
                u, v = edge_keys[edge_index]
                if labels[u] == component and labels[v] < 0:
                    outer = int(v)
                elif labels[v] == component and labels[u] < 0:
                    outer = int(u)
                else:
                    continue
                original_ratio = (
                    original_lengths[edge_index] - median
                ) / original_spread
                projected_ratio = (
                    projected_lengths[edge_index] - projected_median
                ) / projected_spread
                if (original_ratio >= float(hard_limit)
                        or projected_ratio >= float(projected_hard_limit)):
                    continue
                rank = (
                    max(
                        original_ratio / max(float(hard_limit), 1e-12),
                        projected_ratio / max(float(projected_hard_limit), 1e-12),
                    ),
                    float(original_ratio), float(projected_ratio),
                    int(edge_index), int(component),
                )
                previous = proposals.get(outer)
                if previous is None or rank < previous[0]:
                    proposals[outer] = (rank, component, int(edge_index))
        if not proposals:
            break
        for outer, (_, component, edge_index) in sorted(proposals.items()):
            labels[outer] = component
            members[component].add(outer)
            added_points[outer] = True
            accepted_edges[edge_index] = True
        accepted_per_round.append(len(proposals))
        round_index += 1
    return {
        "assigned_labels": labels,
        "accepted_edge_mask": accepted_edges,
        "added_point_mask": added_points,
        "metrics": {
            "initially_assigned_point_count": int(np.count_nonzero(initial_labels >= 0)),
            "added_outlier_count": int(np.count_nonzero(added_points)),
            "remaining_outlier_count": int(np.count_nonzero(labels < 0)),
            "round_count": int(round_index),
            "accepted_edges_per_round": accepted_per_round,
            "original_hard_limit": float(hard_limit),
            "projected_hard_limit": float(projected_hard_limit),
            "length_space": "standardized original space",
            "graph": "all original Delaunay edges",
            "mode": "round-based frozen-component growth",
        },
    }


def _knn_core_distance(X: np.ndarray, k: int) -> np.ndarray:
    """Mean original-space distance to the ``k`` nearest neighbours."""
    X = np.asarray(X, dtype=np.float64)
    n_points = len(X)
    if n_points < 2:
        return np.zeros(n_points, dtype=np.float64)
    k = int(max(1, min(k, n_points - 1)))
    distances, _ = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    return distances[:, 1:].mean(axis=1)


def _assign_anomalies_cost_gate(
    labels: np.ndarray,
    edge_keys: np.ndarray,
    orig_edge_sizes: np.ndarray,
    n_points: int,
    penalty_power: float,
    reach: float,
    core_distance: np.ndarray | None = None,
    density_power: float = 0.0,
    density_clip: bool = False,
) -> tuple[np.ndarray, dict[str, object]]:
    """Reclaim noise with original-space cost-gate density Dijkstra.

    Completed components are fixed sources. Only points currently labeled
    ``-1`` can be claimed, so no component merge is possible in this phase.
    Topology comes from the projected Delaunay graph, while every edge cost
    and density scale is computed in standardized original space.
    """
    labels = np.asarray(labels, dtype=np.int64)
    edge_keys = np.asarray(edge_keys, dtype=np.int64)
    orig_edge_sizes = np.asarray(orig_edge_sizes, dtype=np.float64)
    n_edges = len(edge_keys)
    n_clusters = int(labels.max()) + 1 if np.any(labels >= 0) else 0
    use_density = density_power != 0.0 and core_distance is not None
    diagnostics: dict[str, object] = {
        "mode": "original-space cost-gate density Dijkstra",
        "penalty_power": float(penalty_power),
        "reach": float(reach),
        "density_power": float(density_power) if use_density else 0.0,
        "density_clip": bool(density_clip) if use_density else False,
        "n_anomalies_in": int(np.count_nonzero(labels < 0)),
        "n_anomalies_out": 0,
        "n_claims": 0,
        "grown_mask": np.zeros(n_edges, dtype=bool),
    }
    if n_clusters == 0 or diagnostics["n_anomalies_in"] == 0:
        diagnostics["n_anomalies_out"] = diagnostics["n_anomalies_in"]
        return labels.copy(), diagnostics

    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(n_points)]
    for edge_index, (u, v) in enumerate(edge_keys):
        adjacency[int(u)].append((int(v), edge_index))
        adjacency[int(v)].append((int(u), edge_index))

    sums = np.zeros(n_clusters, dtype=np.float64)
    counts = np.zeros(n_clusters, dtype=np.int64)
    for edge_index, (u, v) in enumerate(edge_keys):
        cluster_u, cluster_v = int(labels[u]), int(labels[v])
        if cluster_u >= 0 and cluster_u == cluster_v:
            sums[cluster_u] += orig_edge_sizes[edge_index]
            counts[cluster_u] += 1
    global_median = float(np.median(orig_edge_sizes)) if n_edges else 1.0
    average = np.maximum(
        np.where(counts > 0, sums / np.maximum(counts, 1), global_median), 1e-12,
    )

    if use_density:
        core_distance = np.asarray(core_distance, dtype=np.float64)
        global_core = float(np.median(core_distance)) if len(core_distance) else 1.0
        density_reference = np.full(n_clusters, global_core, dtype=np.float64)
        for cluster in range(n_clusters):
            values = core_distance[labels == cluster]
            if len(values):
                density_reference[cluster] = float(np.median(values))
        density_reference = np.maximum(density_reference, 1e-12)

        def relative(edge_index: int, node: int, cluster: int) -> float:
            ratio = core_distance[node] / density_reference[cluster]
            if density_clip and ratio < 1.0:
                ratio = 1.0
            density_factor = ratio ** float(density_power)
            return (orig_edge_sizes[edge_index] * density_factor) / average[cluster]
    else:
        def relative(edge_index: int, node: int, cluster: int) -> float:
            return orig_edge_sizes[edge_index] / average[cluster]

    owner = labels.copy()
    exponent = 1.0 + float(penalty_power)
    heap: list[tuple[float, int, int, int]] = []
    for cluster in range(n_clusters):
        for node in np.flatnonzero(labels == cluster):
            for neighbour, edge_index in adjacency[int(node)]:
                if labels[neighbour] != cluster:
                    cost = relative(edge_index, neighbour, cluster) ** exponent
                    if cost <= reach:
                        heapq.heappush(heap, (cost, neighbour, cluster, edge_index))

    while heap:
        distance, node, cluster, edge_index = heapq.heappop(heap)
        if owner[node] != -1:
            continue
        owner[node] = cluster
        grown_mask = diagnostics["grown_mask"]
        assert isinstance(grown_mask, np.ndarray)
        grown_mask[edge_index] = True
        diagnostics["n_claims"] = int(diagnostics["n_claims"]) + 1
        for neighbour, next_edge_index in adjacency[node]:
            if owner[neighbour] != -1:
                continue
            candidate = distance + relative(
                next_edge_index, neighbour, cluster,
            ) ** exponent
            if candidate <= reach:
                heapq.heappush(
                    heap, (candidate, neighbour, cluster, next_edge_index),
                )

    diagnostics["n_anomalies_out"] = int(np.count_nonzero(owner < 0))
    return owner, diagnostics


def _sequential_growth(edge_keys: np.ndarray, lengths: np.ndarray,
                       projected_lengths: np.ndarray, initial_mask: np.ndarray,
                       seed_labels: np.ndarray,
                       seed_edge_counts: np.ndarray, min_edges: int,
                       recall10: np.ndarray, hard_growth_limit: float,
                       projected_hard_growth_limit: float) -> dict:
    """Grow only initially large seed components, one boundary round at a time."""
    n_points = len(seed_labels)
    n_components = len(seed_edge_counts)
    parent = np.arange(n_components, dtype=np.int64)
    members = [set(np.flatnonzero(seed_labels == component)) for component in range(n_components)]
    active = np.asarray(seed_edge_counts > int(min_edges), dtype=bool)
    adjacency = [[] for _ in range(n_points)]
    for edge_index, (u, v) in enumerate(edge_keys):
        adjacency[int(u)].append(edge_index)
        adjacency[int(v)].append(edge_index)
    outer_q25 = np.array([
        np.percentile(lengths[indices], 25.0)
        for indices in adjacency
    ], dtype=np.float64)

    def find(component: int) -> int:
        component = int(component)
        while parent[component] != component:
            parent[component] = parent[parent[component]]
            component = int(parent[component])
        return component

    def merge(left: int, right: int) -> int:
        left, right = find(left), find(right)
        if left == right:
            return left
        if len(members[left]) < len(members[right]):
            left, right = right, left
        parent[right] = left
        members[left].update(members[right])
        members[right].clear()
        active[left] = bool(active[left] or active[right])
        active[right] = False
        return left

    initially_large = active.copy()
    first_seen = np.zeros(len(edge_keys), dtype=bool)
    first_seen_step = np.full(len(edge_keys), -1, dtype=np.int64)
    first_seen_component = np.full(len(edge_keys), -1, dtype=np.int64)
    first_ratio = np.full(len(edge_keys), np.nan, dtype=np.float64)
    first_projected_ratio = np.full(len(edge_keys), np.nan, dtype=np.float64)
    first_outer_q25_ratio = np.full(len(edge_keys), np.nan, dtype=np.float64)
    first_outer_recall = np.full(len(edge_keys), np.nan, dtype=np.float64)
    accepted = np.zeros(len(edge_keys), dtype=bool)
    accepted_step = np.full(len(edge_keys), -1, dtype=np.int64)
    round_index = 0
    accepted_edges_per_round: list[int] = []

    while True:
        qualifying_edges: set[int] = set()
        for root in range(n_components):
            if find(root) != root or not active[root] or not members[root]:
                continue
            incident = set()
            for node in members[root]:
                incident.update(adjacency[node])
            incident_indices = np.fromiter(incident, dtype=np.int64)
            _, median, q75 = np.percentile(
                lengths[incident_indices], [25.0, 50.0, 75.0],
            )
            right_spread = max(float(q75 - median), 1e-12)
            _, projected_median, projected_q75 = np.percentile(
                projected_lengths[incident_indices], [25.0, 50.0, 75.0],
            )
            projected_right_spread = max(
                float(projected_q75 - projected_median), 1e-12,
            )
            for edge_index in incident_indices:
                u, v = edge_keys[edge_index]
                root_u = find(seed_labels[u])
                root_v = find(seed_labels[v])
                if root_u == root_v:
                    continue
                if root_u == root:
                    outer = int(v)
                elif root_v == root:
                    outer = int(u)
                else:
                    continue
                ratio = (lengths[edge_index] - median) / right_spread
                projected_ratio = (
                    (projected_lengths[edge_index] - projected_median)
                    / projected_right_spread
                )
                outer_q25_ratio = (outer_q25[outer] - median) / right_spread
                if not first_seen[edge_index]:
                    first_seen[edge_index] = True
                    first_seen_step[edge_index] = round_index
                    first_seen_component[edge_index] = root
                    first_ratio[edge_index] = ratio
                    first_projected_ratio[edge_index] = projected_ratio
                    first_outer_q25_ratio[edge_index] = outer_q25_ratio
                    first_outer_recall[edge_index] = recall10[outer]
                if (ratio < float(hard_growth_limit)
                        and projected_ratio < float(projected_hard_growth_limit)):
                    qualifying_edges.add(int(edge_index))

        if not qualifying_edges:
            break
        # Deliberately defer every merge until all current boundary edges have
        # been evaluated.  Thus no edge in a round sees statistics changed by
        # another accepted edge from that same round.
        for edge_index in sorted(qualifying_edges):
            u, v = edge_keys[edge_index]
            merge(seed_labels[u], seed_labels[v])
            accepted[edge_index] = True
            accepted_step[edge_index] = round_index
        accepted_edges_per_round.append(len(qualifying_edges))
        round_index += 1

    final_component = np.array([find(seed_labels[point]) for point in range(n_points)], dtype=np.int64)
    final_active_point = active[final_component]
    # A growth-added point was not part of an initially large seed component,
    # but became part of an active component through growth.  This excludes
    # points in the original large seed components even when two such seeds
    # are later joined by a growth edge.
    growth_added_points = (
        ~initially_large[seed_labels] & final_active_point
    )
    return {
        "initially_large_seed_components": initially_large,
        "first_seen_mask": first_seen,
        "first_seen_step": first_seen_step,
        "first_seen_component": first_seen_component,
        "first_seen_panel2_ratio_raw": first_ratio,
        "first_seen_panel2_display": np.clip(first_ratio, 0.0, 2.0),
        "first_seen_projected_ratio_raw": first_projected_ratio,
        "first_seen_panel3_outer_recall10": first_outer_recall,
        "first_seen_panel4_ratio_raw": first_outer_q25_ratio,
        "first_seen_panel4_display": np.clip(first_outer_q25_ratio, 0.0, 2.0),
        "accepted_growth_mask": accepted,
        "accepted_growth_step": accepted_step,
        "growth_added_point_mask": growth_added_points,
        "final_component": final_component,
        "final_active_point": final_active_point,
        "growth_round_count": int(round_index),
        "accepted_edges_per_round": np.asarray(accepted_edges_per_round, dtype=np.int64),
    }


def main() -> None:
    def _optional_positive_int(value: str) -> int | None:
        if value.lower() in {"none", "auto", "default"}:
            return None
        parsed = int(value)
        if parsed < 1:
            raise argparse.ArgumentTypeError("must be positive or 'default'")
        return parsed

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--quiet", action="store_true",
        help="Write plot/NPZ/JSON artifacts without printing the full JSON summary.",
    )
    parser.add_argument(
        "--metrics-only", action="store_true",
        help="Run the exact stage pipeline and write JSON only; skip all plotting and HDBSCAN.",
    )
    parser.add_argument(
        "--outlier-metrics", action="store_true",
        help="Also score final noise labels against binary y>0 outlier labels.",
    )
    parser.add_argument("--dim-reduction", default="umap")
    parser.add_argument("--project-dim", type=int, choices=(2, 3), default=2)
    parser.add_argument("--umap-n-epochs", default="100")
    parser.add_argument("--umap-n-neighbors", type=int, default=15)
    parser.add_argument("--hdbscan-min-cluster-size", type=int,
                        default=HDBSCAN_MIN_CLUSTER_SIZE)
    parser.add_argument("--hdbscan-min-samples", type=_optional_positive_int,
                        default=HDBSCAN_MIN_SAMPLES)
    parser.add_argument("--hdbscan-selection-method", choices=("eom", "leaf"),
                        default=HDBSCAN_SELECTION_METHOD)
    parser.add_argument("--hdbscan-cluster-selection-epsilon", type=float,
                        default=HDBSCAN_CLUSTER_SELECTION_EPSILON)
    parser.add_argument("--seed-hard-limit", type=float, default=0.5)
    parser.add_argument("--hard-growth-limit", type=float, default=0.05)
    parser.add_argument(
        "--projected-hard-growth-limit", default="auto",
        help="Projected-space guard; 'auto' uses two times --hard-growth-limit.",
    )
    parser.add_argument(
        "--redundancy-pruning", action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--redundancy-threshold", type=int, default=3)
    parser.add_argument("--redundancy-prune-limit", type=int, default=3)
    parser.add_argument("--redundancy-low-majority", type=float, default=0.5)
    parser.add_argument(
        "--gomory-hu-cut-size", type=int, default=0,
        help=(
            "Apply every eligible Gomory--Hu cut of this size or smaller "
            "after growth; 0 disables this final bridge-pruning stage."
        ),
    )
    parser.add_argument(
        "--gomory-hu-min-component-points", type=int, default=10,
        help="Require this many points on both sides of a Gomory--Hu cut.",
    )
    parser.add_argument(
        "--gomory-hu-hull-ratio-skip-threshold", type=float, default=0.0,
        help=(
            "Skip Gomory--Hu pruning for a component when its weighted "
            "outer-hull/non-hull ratio reaches this value; 0 disables the gate."
        ),
    )
    parser.add_argument(
        "--gomory-hu-growth-edges-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Contract original seed components, then build Gomory--Hu only "
            "from accepted growth edges and restored edges incident to a "
            "growth-added point. Disable to run GH on every final edge."
        ),
    )
    parser.add_argument(
        "--restore-intra-component-edges",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Before final bridge analysis, restore every original Delaunay "
            "edge whose endpoints are in the same completed growth component."
        ),
    )
    parser.add_argument(
        "--outlier-growth", action=argparse.BooleanOptionalAction, default=False,
        help=(
            "Experimental final reassignment phase. The selected method may "
            "only claim currently unassigned points and cannot merge components."
        ),
    )
    parser.add_argument(
        "--outlier-reassignment", choices=("cost_gate", "frozen_growth"),
        default="cost_gate",
        help=(
            "Post-growth outlier reassignment rule. cost_gate is the prototype "
            "cost-budgeted density-Dijkstra method; frozen_growth is the prior "
            "component-relative growth fallback."
        ),
    )
    parser.add_argument(
        "--outlier-cost-gate-penalty-power", type=float, default=0.5,
        help="Cost exponent is 1 + this value for cost-gate reassignment.",
    )
    parser.add_argument(
        "--outlier-cost-gate-reach", type=float, default=10.0,
        help="Maximum accumulated Dijkstra cost for cost-gate reassignment.",
    )
    parser.add_argument(
        "--outlier-cost-gate-density-power", type=float, default=0.5,
        help=(
            "Exponent for the original-space density ratio in cost-gate "
            "edge costs; zero disables density weighting."
        ),
    )
    parser.add_argument(
        "--outlier-cost-gate-density-k", type=int, default=15,
        help="Original-space kNN scale used by cost-gate density weighting.",
    )
    parser.add_argument(
        "--outlier-cost-gate-density-clip",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Clip cost-gate density ratios below one.",
    )
    parser.add_argument(
        "--outlier-growth-hard-limit", type=float, default=1.5,
        help="Original-space component-relative hard limit for frozen growth.",
    )
    parser.add_argument(
        "--projected-outlier-growth-hard-limit", type=float, default=1.0,
        help="Projected-space component-relative hard limit for frozen growth.",
    )
    parser.add_argument("--initial-relation", choices=("union", "mutual"), default="union")
    parser.add_argument("--component-growth-knn", type=int, default=50)
    parser.add_argument("--component-growth-min-edges", type=int, default=10)
    parser.add_argument("--hard-gate-mode", choices=(
        "strict", "separate_limits", "knn_relaxed", "knn_growth_relaxed", "knn_global"
    ), default="knn_relaxed")
    parser.add_argument("--seed-relaxed-hard-limit", type=float, default=0.5)
    parser.add_argument("--local-knn-selectivity-threshold", type=float, default=0.1)
    parser.add_argument("--local-knn-selectivity-scope", choices=("edge", "edge_2hop", "component"), default="edge")
    parser.add_argument("--projected-hard-limit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--multiplicative-hard-limit", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    if args.umap_n_epochs != "auto":
        args.umap_n_epochs = int(args.umap_n_epochs)
    if args.gomory_hu_cut_size < 0:
        raise ValueError("--gomory-hu-cut-size must be non-negative")
    if args.gomory_hu_min_component_points < 1:
        raise ValueError("--gomory-hu-min-component-points must be positive")
    if args.gomory_hu_hull_ratio_skip_threshold < 0.0:
        raise ValueError("--gomory-hu-hull-ratio-skip-threshold must be non-negative")
    if args.outlier_growth_hard_limit < 0.0:
        raise ValueError("--outlier-growth-hard-limit must be non-negative")
    if args.projected_outlier_growth_hard_limit < 0.0:
        raise ValueError("--projected-outlier-growth-hard-limit must be non-negative")
    if args.outlier_cost_gate_penalty_power < 0.0:
        raise ValueError("--outlier-cost-gate-penalty-power must be non-negative")
    if args.outlier_cost_gate_reach < 0.0:
        raise ValueError("--outlier-cost-gate-reach must be non-negative")
    if args.outlier_cost_gate_density_k < 1:
        raise ValueError("--outlier-cost-gate-density-k must be positive")
    if args.projected_hard_growth_limit == "auto":
        args.projected_hard_growth_limit = 2.0 * args.hard_growth_limit
    else:
        args.projected_hard_growth_limit = float(args.projected_hard_growth_limit)

    loaded = np.load(args.data, allow_pickle=False)
    X = StandardScaler().fit_transform(np.asarray(loaded["X"], dtype=np.float64))
    # Use the production seed-construction path, but disable its subsequent
    # growth: this experiment owns all growth after initial_mask is obtained.
    state = component_growth_graph(
        X, dim_reduction=args.dim_reduction, project_dim=args.project_dim,
        umap_n_epochs=args.umap_n_epochs, umap_n_neighbors=args.umap_n_neighbors,
        hard_limit=0.9, growth_hard_limit=0.9, seed_hard_limit=args.seed_hard_limit,
        mild_limit=-0.5, initial_relation=args.initial_relation, growth_relation="union",
        knn_k=args.component_growth_knn, component_min_edges=len(X) * len(X),
        hard_gate_mode=args.hard_gate_mode,
        seed_relaxed_hard_limit=args.seed_relaxed_hard_limit,
        growth_relaxed_hard_limit=0.9,
        local_knn_selectivity_threshold=args.local_knn_selectivity_threshold,
        local_knn_selectivity_scope=args.local_knn_selectivity_scope,
        projected_hard_limit=args.projected_hard_limit,
        multiplicative_hard_limit=args.multiplicative_hard_limit,
    )
    edge_keys = np.asarray(state["edge_keys"], dtype=np.int64)
    lengths = np.asarray(state["orig_edge_sizes"], dtype=np.float64)
    projected_lengths = np.linalg.norm(
        np.asarray(state["X_proj"], dtype=np.float64)[edge_keys[:, 0]]
        - np.asarray(state["X_proj"], dtype=np.float64)[edge_keys[:, 1]],
        axis=1,
    )
    initial_mask = np.asarray(state["initial_mask"], dtype=bool)
    seed_labels, seed_edge_counts, large_seed = _seed_components(
        edge_keys, initial_mask, len(X), args.component_growth_min_edges,
    )
    recall10 = _projected_recall10(X, np.asarray(state["X_proj"], dtype=np.float64))
    growth = _sequential_growth(
        edge_keys, lengths, projected_lengths, initial_mask, seed_labels,
        seed_edge_counts, args.component_growth_min_edges, recall10,
        args.hard_growth_limit, args.projected_hard_growth_limit,
    )
    large_points = large_seed[seed_labels]
    seed_mask = initial_mask & large_points[edge_keys[:, 0]] & large_points[edge_keys[:, 1]]
    # Restore intra-component Delaunay edges for final connectivity and the
    # optional redundancy analysis.  GH nevertheless protects seed-only
    # edges, so it can remove only growth-related links.
    pre_redundancy_mask = initial_mask | growth["accepted_growth_mask"]
    growth_component_labels, growth_component_points = _final_graph_components(
        edge_keys, pre_redundancy_mask, len(X),
    )
    same_completed_component = (
        growth_component_points[edge_keys[:, 0]]
        & growth_component_points[edge_keys[:, 1]]
        & (growth_component_labels[edge_keys[:, 0]]
           == growth_component_labels[edge_keys[:, 1]])
    )
    if args.restore_intra_component_edges:
        bridge_base_mask = pre_redundancy_mask | same_completed_component
    else:
        bridge_base_mask = pre_redundancy_mask.copy()
    restored_intra_component_mask = bridge_base_mask & ~pre_redundancy_mask
    if args.redundancy_pruning:
        redundancy_state = _edge_redundancy_pruning(
            edge_keys, bridge_base_mask,
            redundancy_threshold=args.redundancy_threshold,
            redundancy_prune_limit=args.redundancy_prune_limit,
            low_redundancy_majority=args.redundancy_low_majority,
        )
    else:
        redundancy_state = {
            "redundancy": np.full(len(edge_keys), -1, dtype=np.int16),
            "pruned_mask": np.zeros(len(edge_keys), dtype=bool),
            "skipped_component_mask": np.zeros(len(edge_keys), dtype=bool),
            "metrics": {},
        }
    gomory_graph_mask = bridge_base_mask
    growth_added_points = growth["growth_added_point_mask"]
    enriched_growth_incident_mask = (
        restored_intra_component_mask
        & (growth_added_points[edge_keys[:, 0]]
           | growth_added_points[edge_keys[:, 1]])
    )
    gomory_removable_mask = (
        growth["accepted_growth_mask"] | enriched_growth_incident_mask
        if args.gomory_hu_growth_edges_only else None
    )
    if args.gomory_hu_growth_edges_only:
        if args.gomory_hu_hull_ratio_skip_threshold > 0.0:
            raise ValueError(
                "the hull-ratio GH gate is unavailable with contracted "
                "growth-edge candidate cuts; set it to 0"
            )
        gomory_hu_pruned_mask, gomory_hu_metrics = (
            _gomory_hu_contracted_candidate_cut_mask(
                edge_keys, gomory_removable_mask, seed_labels,
                cut_size=args.gomory_hu_cut_size,
                min_component_points=args.gomory_hu_min_component_points,
            )
        )
    else:
        gomory_hu_pruned_mask, gomory_hu_metrics = _gomory_hu_joint_cut_mask(
            edge_keys, gomory_graph_mask,
            cut_size=args.gomory_hu_cut_size,
            min_component_points=args.gomory_hu_min_component_points,
            hull_ratio_skip_threshold=args.gomory_hu_hull_ratio_skip_threshold,
            triangles=np.asarray(state["triangles"], dtype=np.int64),
            edge_sizes=lengths,
        )
    removed_bridge_mask = redundancy_state["pruned_mask"] | gomory_hu_pruned_mask
    final_base_mask = bridge_base_mask
    final_retained_mask = final_base_mask & ~removed_bridge_mask
    seen = growth["first_seen_mask"]
    segments = _segments(state["X_proj"], edge_keys)
    truth_labels = (
        np.asarray(loaded["y"], dtype=np.int64) if "y" in loaded.files else
        np.asarray(loaded["y_clean"], dtype=np.int64) if "y_clean" in loaded.files else None
    )
    base_final_labels, base_final_component_points = _final_graph_components(
        edge_keys, final_retained_mask, len(X),
    )
    base_prediction = _final_component_prediction(
        edge_keys, final_retained_mask, base_final_labels,
        base_final_component_points, args.component_growth_min_edges,
    )
    if args.outlier_growth and args.outlier_reassignment == "cost_gate":
        core_distance = (
            _knn_core_distance(X, args.outlier_cost_gate_density_k)
            if args.outlier_cost_gate_density_power != 0.0 else None
        )
        reassigned_labels, cost_gate_metrics = _assign_anomalies_cost_gate(
            base_prediction, edge_keys, lengths, len(X),
            penalty_power=args.outlier_cost_gate_penalty_power,
            reach=args.outlier_cost_gate_reach,
            core_distance=core_distance,
            density_power=args.outlier_cost_gate_density_power,
            density_clip=args.outlier_cost_gate_density_clip,
        )
        accepted_edge_mask = np.asarray(cost_gate_metrics.pop("grown_mask"), dtype=bool)
        added_point_mask = (base_prediction < 0) & (reassigned_labels >= 0)
        cost_gate_metrics.update({
            "method": "cost_gate",
            "added_outlier_count": int(np.count_nonzero(added_point_mask)),
            "remaining_outlier_count": int(np.count_nonzero(reassigned_labels < 0)),
            "edge_length_space": "standardized original space",
            "density_space": (
                "standardized original space"
                if core_distance is not None else "disabled"
            ),
        })
        outlier_growth_state = {
            "assigned_labels": reassigned_labels,
            "accepted_edge_mask": accepted_edge_mask,
            "added_point_mask": added_point_mask,
            "metrics": cost_gate_metrics,
        }
    elif args.outlier_growth:
        outlier_growth_state = _grow_frozen_components(
            edge_keys, lengths, projected_lengths, base_prediction,
            hard_limit=args.outlier_growth_hard_limit,
            projected_hard_limit=args.projected_outlier_growth_hard_limit,
        )
        outlier_growth_state["metrics"]["method"] = "frozen_growth"
    else:
        outlier_growth_state = {
            "assigned_labels": base_prediction,
            "accepted_edge_mask": np.zeros(len(edge_keys), dtype=bool),
            "added_point_mask": np.zeros(len(X), dtype=bool),
            "metrics": {
                "mode": "disabled",
                "method": args.outlier_reassignment,
                "original_hard_limit": float(args.outlier_growth_hard_limit),
                "projected_hard_limit": float(args.projected_outlier_growth_hard_limit),
                "added_outlier_count": 0,
            },
        }
    final_prediction = np.asarray(outlier_growth_state["assigned_labels"], dtype=np.int64)
    final_assignment_mask = final_retained_mask | np.asarray(
        outlier_growth_state["accepted_edge_mask"], dtype=bool,
    )
    hdbscan_metrics = None
    final_labels, final_component_points = _final_graph_components(
        edge_keys, final_assignment_mask, len(X),
    )
    final_ari = None
    if truth_labels is not None:
        final_ari = float(adjusted_rand_score(truth_labels, final_prediction))
    outlier_metrics = None
    if args.outlier_metrics:
        if truth_labels is None:
            raise ValueError("--outlier-metrics requires y or y_clean in the input NPZ")
        y_true_outlier = np.asarray(truth_labels > 0, dtype=np.int8)
        y_pred_outlier = np.asarray(final_prediction < 0, dtype=np.int8)
        outlier_metrics = {
            "ari": float(adjusted_rand_score(y_true_outlier, y_pred_outlier)),
            "f1": float(f1_score(y_true_outlier, y_pred_outlier, zero_division=0)),
            "precision": float(precision_score(y_true_outlier, y_pred_outlier, zero_division=0)),
            "recall": float(recall_score(y_true_outlier, y_pred_outlier, zero_division=0)),
            "true_outlier_count": int(np.count_nonzero(y_true_outlier)),
            "predicted_outlier_count": int(np.count_nonzero(y_pred_outlier)),
        }
    if not args.metrics_only:
        # Fit only in standardized original space.  Reuse X_proj for the
        # picture, which means 2-D data stays unprojected and high-D data
        # shares the exact UMAP coordinates used for the DelTriC panels.
        hdbscan_labels = fit_hdbscan(
            X,
            min_cluster_size=args.hdbscan_min_cluster_size,
            min_samples=args.hdbscan_min_samples,
            cluster_selection_method=args.hdbscan_selection_method,
            cluster_selection_epsilon=args.hdbscan_cluster_selection_epsilon,
        )
        hdbscan_metrics = result_metrics(truth_labels, hdbscan_labels)
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(16, 14), constrained_layout=True)
        _draw_seed_truth(axes[0, 0], state["X_proj"], segments, seed_mask, truth_labels,
                         "initial seed components (orange), original labels (points)")
        _draw_hdbscan_final(
            axes[0, 1], state["X_proj"], hdbscan_labels,
            ari=float(hdbscan_metrics["ari"]),
            min_cluster_size=args.hdbscan_min_cluster_size,
            min_samples=args.hdbscan_min_samples,
            selection_method=args.hdbscan_selection_method,
            cluster_selection_epsilon=args.hdbscan_cluster_selection_epsilon,
        )
        _draw_growth_gate_classification(
            axes[1, 0], state["X_proj"], segments, large_points, seed_mask, seen,
            growth["first_seen_panel2_ratio_raw"],
            growth["first_seen_projected_ratio_raw"],
            hard_growth_limit=args.hard_growth_limit,
            projected_hard_growth_limit=args.projected_hard_growth_limit,
        )
        _draw_final_components(
            axes[1, 1], state["X_proj"], edge_keys, segments,
            final_assignment_mask,
            redundancy_pruned_mask=removed_bridge_mask,
            assigned_labels=final_prediction,
            added_outlier_mask=outlier_growth_state["added_point_mask"],
        )
        growth_note = (
            f"; added={outlier_growth_state['metrics']['added_outlier_count']}"
            if args.outlier_growth else ""
        )
        axes[1, 1].set_title(
            f"final assignments — ARI={final_ari:.3f}{growth_note}",
            fontsize=11,
        )
        fig.suptitle(
            f"Sequential component-relative growth — {args.data.stem}", fontsize=14
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=150, bbox_inches="tight")
        plt.close(fig)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not args.metrics_only:
        np.savez_compressed(
            args.out.with_suffix(".npz"), X_proj=state["X_proj"], edge_keys=edge_keys,
            initial_mask=initial_mask, seed_labels=seed_labels,
            seed_edge_counts=seed_edge_counts, pre_redundancy_mask=pre_redundancy_mask,
            bridge_base_mask=bridge_base_mask,
            gomory_graph_mask=gomory_graph_mask,
            gomory_removable_mask=(
                gomory_removable_mask if gomory_removable_mask is not None
                else np.zeros(len(edge_keys), dtype=bool)
            ),
            enriched_growth_incident_mask=enriched_growth_incident_mask,
            restored_intra_component_mask=restored_intra_component_mask,
            final_retained_mask=final_retained_mask,
            final_assignment_mask=final_assignment_mask,
            final_prediction=final_prediction,
            outlier_growth_accepted_edge_mask=outlier_growth_state["accepted_edge_mask"],
            outlier_growth_added_point_mask=outlier_growth_state["added_point_mask"],
            edge_redundancy=redundancy_state["redundancy"],
            redundancy_pruned_mask=redundancy_state["pruned_mask"],
            gomory_hu_pruned_mask=gomory_hu_pruned_mask,
            redundancy_skipped_component_mask=redundancy_state["skipped_component_mask"],
            **growth,
        )
    summary = {
        "dataset": str(args.data), "out": str(args.out),
        "seed_hard_limit": args.seed_hard_limit,
        "hard_growth_limit": args.hard_growth_limit,
        "projected_hard_growth_limit": args.projected_hard_growth_limit,
        "redundancy_pruning": args.redundancy_pruning,
        "redundancy_threshold": args.redundancy_threshold,
        "redundancy_prune_limit": args.redundancy_prune_limit,
        "redundancy_low_majority": args.redundancy_low_majority,
        "gomory_hu_cut_size": args.gomory_hu_cut_size,
        "gomory_hu_min_component_points": args.gomory_hu_min_component_points,
        "gomory_hu_hull_ratio_skip_threshold": args.gomory_hu_hull_ratio_skip_threshold,
        "gomory_hu_growth_edges_only": args.gomory_hu_growth_edges_only,
        "restore_intra_component_edges": args.restore_intra_component_edges,
        "outlier_growth": bool(args.outlier_growth),
        "outlier_reassignment": args.outlier_reassignment,
        "outlier_growth_metrics": outlier_growth_state["metrics"],
        "initial_relation": args.initial_relation, "knn_k": args.component_growth_knn,
        "hdbscan": None if hdbscan_metrics is None else {
            "fit_space": "standardized original space",
            "display_space": (
                "standardized original 2D coordinates"
                if X.shape[1] <= args.project_dim else
                "same UMAP projection as DelTriC panels"
            ),
            "min_cluster_size": args.hdbscan_min_cluster_size,
            "min_samples": args.hdbscan_min_samples,
            "cluster_selection_method": args.hdbscan_selection_method,
            "cluster_selection_epsilon": args.hdbscan_cluster_selection_epsilon,
            "ari_definition": "all-point ARI, including HDBSCAN noise label -1",
            **hdbscan_metrics,
        },
        "outlier_metrics": outlier_metrics,
        "ari": final_ari,
        "ari_definition": (
            "ARI after optional outlier reassignment; components with <= "
            f"{args.component_growth_min_edges} edges are otherwise mapped to noise (-1)"
            if final_ari is not None else None
        ),
        "seed_edge_count": int(np.count_nonzero(initial_mask)),
        "large_seed_component_count": int(np.count_nonzero(large_seed)),
        "historical_boundary_edge_count": int(np.count_nonzero(seen)),
        "accepted_growth_edge_count": int(np.count_nonzero(growth["accepted_growth_mask"])),
        "growth_added_point_count": int(np.count_nonzero(growth_added_points)),
        "restored_intra_component_edge_count": int(np.count_nonzero(
            restored_intra_component_mask,
        )),
        "gomory_removable_edge_count": int(np.count_nonzero(
            gomory_removable_mask if gomory_removable_mask is not None
            else gomory_graph_mask,
        )),
        "bridge_analysis_edge_count": int(np.count_nonzero(bridge_base_mask)),
        "final_retained_edge_count_before_outlier_growth": int(np.count_nonzero(final_retained_mask)),
        "final_retained_edge_count": int(np.count_nonzero(final_assignment_mask)),
        "redundancy_pruned_edge_count": int(np.count_nonzero(
            redundancy_state["pruned_mask"],
        )),
        "gomory_hu_pruned_edge_count": int(np.count_nonzero(gomory_hu_pruned_mask)),
        "gomory_hu_metrics": gomory_hu_metrics,
        "redundancy_skipped_component_edge_count": int(np.count_nonzero(
            redundancy_state["skipped_component_mask"],
        )),
        "redundancy_metrics": redundancy_state["metrics"],
        "final_component_count": int(np.unique(
            final_labels[final_component_points]
        ).size),
        "growth_round_count": growth["growth_round_count"],
        "accepted_edges_per_round": growth["accepted_edges_per_round"].tolist(),
        "panel2_definition": "recomputed after every complete boundary-growth round; colors frozen when an edge is first evaluated",
        "panel4_definition": "all edge-bearing components after final redundancy and Gomory--Hu bridge removals, each with a distinct colour",
        "redundancy_definition": "edge-disjoint alternative paths in the completed pre-pruning graph; all pruning decisions are simultaneous per component",
    }
    if args.redundancy_pruning and not args.quiet:
        metrics = redundancy_state["metrics"]
        top_five = "; ".join(
            f"{item['edge_count']}e/{item['low_redundancy_fraction']:.1%}"
            f"/{'skip' if item['skipped'] else 'prune'}"
            for item in metrics["largest_components"]
        )
        print(
            "Redundancy metrics: "
            f"edges={metrics['evaluated_edge_count']}, "
            f"hist={metrics['capped_redundancy_histogram']}, "
            f"low(<{args.redundancy_threshold})="
            f"{metrics['low_redundancy_edge_fraction']:.1%}, "
            f"components skipped/pruned="
            f"{metrics['skipped_component_count']}/{metrics['pruned_component_count']}, "
            f"edges pruned={int(np.count_nonzero(redundancy_state['pruned_mask']))}; "
            f"top components (edges/low-fraction/action): {top_five}"
        )
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    if not args.quiet:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
