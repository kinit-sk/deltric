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
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from utils_component_growth import component_growth_graph  # noqa: E402
from plot_seed_component_metrics import (  # noqa: E402
    _draw_boundary_metric,
    _draw_seed_truth,
    _seed_components,
    _segments,
)


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
    hull_ratio_skip_threshold: float = 0.0,
    triangles: np.ndarray | None = None,
    edge_sizes: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Return eligible fundamental Gomory--Hu cuts with size <= cut_size.

    Both sides must contain ``min_component_points`` graph points.  That
    excludes cuts which only detach outliers or small nested islands.
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
        graph.add_edges_from(
            (int(edge_keys[index, 0]), int(edge_keys[index, 1]), {"capacity": 1})
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
                           redundancy_pruned_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Draw every edge-bearing component of the final retained graph."""
    from matplotlib.collections import LineCollection
    import matplotlib.pyplot as plt

    ax.add_collection(LineCollection(
        segments, colors="#d0d0d0", linewidths=0.3, alpha=0.28, zorder=1,
    ))
    labels, component_point_mask = _final_graph_components(
        edge_keys, retained_mask, len(X_proj),
    )
    component_ids = np.unique(labels[component_point_mask])
    if len(component_ids):
        cmap = plt.get_cmap(
            "tab20" if len(component_ids) <= 20 else "turbo", len(component_ids),
        )
        for colour_index, component in enumerate(component_ids):
            point_mask = component_point_mask & (labels == component)
            edge_mask = retained_mask & point_mask[edge_keys[:, 0]] & point_mask[edge_keys[:, 1]]
            colour = cmap(colour_index)
            if np.any(edge_mask):
                ax.add_collection(LineCollection(
                    segments[edge_mask], colors=[colour], linewidths=1.05,
                    alpha=0.93, zorder=3,
                ))
            ax.scatter(X_proj[point_mask, 0], X_proj[point_mask, 1], s=5.2,
                       color=[colour], alpha=0.9, linewidths=0, zorder=4)
    inactive = ~component_point_mask
    if np.any(inactive):
        ax.scatter(X_proj[inactive, 0], X_proj[inactive, 1], s=2.5,
                   color="#9e9e9e", alpha=0.42, linewidths=0, zorder=2)
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
        "final_component": final_component,
        "final_active_point": final_active_point,
        "growth_round_count": int(round_index),
        "accepted_edges_per_round": np.asarray(accepted_edges_per_round, dtype=np.int64),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--quiet", action="store_true",
        help="Write plot/NPZ/JSON artifacts without printing the full JSON summary.",
    )
    parser.add_argument("--dim-reduction", default="umap")
    parser.add_argument("--project-dim", type=int, choices=(2, 3), default=2)
    parser.add_argument("--umap-n-epochs", default="100")
    parser.add_argument("--umap-n-neighbors", type=int, default=15)
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
        "--restore-intra-component-edges",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Before final bridge analysis, restore every original Delaunay "
            "edge whose endpoints are in the same completed growth component."
        ),
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
    # Panel 4 represents the actual final graph: every seed edge survives,
    # including components too small to be eligible for subsequent growth.
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
    gomory_hu_pruned_mask, gomory_hu_metrics = _gomory_hu_joint_cut_mask(
        edge_keys, bridge_base_mask,
        cut_size=args.gomory_hu_cut_size,
        min_component_points=args.gomory_hu_min_component_points,
        hull_ratio_skip_threshold=args.gomory_hu_hull_ratio_skip_threshold,
        triangles=np.asarray(state["triangles"], dtype=np.int64),
        edge_sizes=lengths,
    )
    removed_bridge_mask = redundancy_state["pruned_mask"] | gomory_hu_pruned_mask
    final_retained_mask = bridge_base_mask & ~removed_bridge_mask
    seen = growth["first_seen_mask"]
    segments = _segments(state["X_proj"], edge_keys)
    truth_labels = (
        np.asarray(loaded["y"], dtype=np.int64) if "y" in loaded.files else
        np.asarray(loaded["y_clean"], dtype=np.int64) if "y_clean" in loaded.files else None
    )

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), constrained_layout=True)
    _draw_seed_truth(axes[0, 0], state["X_proj"], segments, seed_mask, truth_labels,
                     "initial seed components (orange), original labels (points)")
    _draw_boundary_metric(
        axes[0, 1], state["X_proj"], segments, large_points, seed_mask, seen,
        growth["first_seen_panel2_display"],
        f"historical boundary edge ratio; accept if < {args.hard_growth_limit:g}",
        "(edge − component median)/(q75 − median), clipped [0, 2]",
        fixed_limits=(0.0, 2.0),
    )
    _draw_boundary_metric(
        axes[1, 0], state["X_proj"], segments, large_points, seed_mask, seen,
        growth["first_seen_panel3_outer_recall10"],
        "historical boundary outer-point projected recall@10",
        "outer-point projected recall@10", fixed_limits=(0.0, 1.0),
    )
    final_labels, final_component_points = _draw_final_components(
        axes[1, 1], state["X_proj"], edge_keys, segments, final_retained_mask,
        removed_bridge_mask,
    )
    fig.suptitle(
        f"Sequential component-relative growth — {args.data.stem}", fontsize=14
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    np.savez_compressed(
        args.out.with_suffix(".npz"), X_proj=state["X_proj"], edge_keys=edge_keys,
        initial_mask=initial_mask, seed_labels=seed_labels,
        seed_edge_counts=seed_edge_counts, pre_redundancy_mask=pre_redundancy_mask,
        bridge_base_mask=bridge_base_mask,
        restored_intra_component_mask=restored_intra_component_mask,
        final_retained_mask=final_retained_mask,
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
        "restore_intra_component_edges": args.restore_intra_component_edges,
        "initial_relation": args.initial_relation, "knn_k": args.component_growth_knn,
        "seed_edge_count": int(np.count_nonzero(initial_mask)),
        "large_seed_component_count": int(np.count_nonzero(large_seed)),
        "historical_boundary_edge_count": int(np.count_nonzero(seen)),
        "accepted_growth_edge_count": int(np.count_nonzero(growth["accepted_growth_mask"])),
        "restored_intra_component_edge_count": int(np.count_nonzero(
            restored_intra_component_mask,
        )),
        "bridge_analysis_edge_count": int(np.count_nonzero(bridge_base_mask)),
        "final_retained_edge_count": int(np.count_nonzero(final_retained_mask)),
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
