"""Component-growth pruning backend for DelTriC.

This module is deliberately separate from :mod:`utils_pruning`.  It implements the
component-growth experiment as a small, inspectable alternative backend while
reusing the canonical projection and Delaunay construction from
``utils_pruning.py``.

The graph is built in the projected space, but all edge-length statistics used
for the original-space decisions are computed from ``X``.  An edge survives
the global hard gate only when it passes both the original-space and (when
enabled) projected-space hard limits.
"""

from __future__ import annotations

import heapq
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors

try:  # package import
    from .utils_pruning import get_triangles_with_edges, _extract_edge_data
except ImportError:  # script/test import
    from utils_pruning import get_triangles_with_edges, _extract_edge_data


def _percentile_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    q25, median, q75 = np.percentile(values, [25.0, 50.0, 75.0])
    return {
        "q25": float(q25),
        "median": float(median),
        "q75": float(q75),
        "iqr": float(max(q75 - q25, 1e-12)),
    }


def _hard_threshold(values: np.ndarray, hard_limit: float,
                    neighbor_stat: str = "median",
                    multiplicative: bool = False) -> tuple[float, dict[str, float]]:
    stats = _percentile_stats(values)
    if neighbor_stat == "median":
        center = stats["median"]
    elif neighbor_stat == "mean":
        center = float(np.mean(values))
    else:
        raise ValueError("neighbor_stat must be 'median' or 'mean'")
    if multiplicative:
        threshold = center * (1.0 + float(hard_limit))
    else:
        threshold = center + float(hard_limit) * stats["iqr"]
    stats["center"] = float(center)
    stats["threshold"] = float(threshold)
    return float(threshold), stats


def _original_knn_relation(X: np.ndarray, edge_keys: np.ndarray, k: int,
                           relation: str) -> np.ndarray:
    """Return the union/mutual original-space kNN relation on graph edges."""
    uv, vu, _, _ = _original_knn_directed_relations(X, edge_keys, k)
    if relation == "union":
        return uv | vu
    if relation == "mutual":
        return uv & vu
    raise ValueError("relation must be 'union' or 'mutual'")


def _original_knn_directed_relations(
    X: np.ndarray, edge_keys: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return directed edge relations and local kNN selectivity.

    ``selectivity[node]`` is one minus the fraction of that node's incident
    Delaunay edges that also occur in its original-space kNN list.  It is a
    cheap local diagnostic: no second neighbor search is performed.
    """
    n_points = len(X)
    if n_points < 2 or len(edge_keys) == 0:
        empty_edges = np.zeros(len(edge_keys), dtype=bool)
        return empty_edges, empty_edges.copy(), np.zeros(n_points), np.zeros(n_points, dtype=np.int64)
    k_eff = min(max(1, int(k)), n_points - 1)
    indices = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean").fit(X).kneighbors(
        X, return_distance=False
    )[:, 1:]
    # A sparse relation avoids the dense N-by-N matrix used by the original
    # exploratory script, while preserving exactly the same edge predicate.
    rows = np.repeat(np.arange(n_points), k_eff)
    relation_graph = coo_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, indices.ravel())),
        shape=(n_points, n_points),
    ).tocsr()
    u, v = edge_keys[:, 0], edge_keys[:, 1]
    uv = np.asarray(relation_graph[u, v]).ravel().astype(bool)
    vu = np.asarray(relation_graph[v, u]).ravel().astype(bool)
    degree = np.zeros(n_points, dtype=np.int64)
    pass_count = np.zeros(n_points, dtype=np.int64)
    np.add.at(degree, u, 1)
    np.add.at(degree, v, 1)
    np.add.at(pass_count, u, uv.astype(np.int64))
    np.add.at(pass_count, v, vu.astype(np.int64))
    coverage = np.divide(
        pass_count, degree, out=np.zeros(n_points, dtype=np.float64),
        where=degree > 0,
    )
    selectivity = 1.0 - coverage
    return uv, vu, selectivity, degree


def _two_hop_edge_selectivity(
    edge_keys: np.ndarray,
    node_selectivity: np.ndarray,
    node_degree: np.ndarray,
    directed_uv: np.ndarray,
    directed_vu: np.ndarray,
    n_points: int,
) -> np.ndarray:
    """Measure local kNN disagreement in each candidate edge's 2-hop context.

    The context comprises all Delaunay vertices within at most two graph hops
    of either endpoint.  Its score is the degree-weighted fraction of
    directed Delaunay incidences that *do not* occur in the original-space
    kNN relation.  The candidate edge itself is removed from this aggregate,
    so the score describes its surrounding graph rather than directly
    revealing whether that candidate is a kNN edge.
    """
    edge_keys = np.asarray(edge_keys, dtype=np.int64)
    n_edges = len(edge_keys)
    adjacency = [[] for _ in range(n_points)]
    for edge_index, (u, v) in enumerate(edge_keys):
        adjacency[int(u)].append(int(v))
        adjacency[int(v)].append(int(u))

    degree = np.asarray(node_degree, dtype=np.float64)
    # The original point metric is exact up to normal floating-point
    # representation; rounding recovers its integral directed pass counts.
    pass_count = np.rint(degree * (1.0 - node_selectivity)).astype(np.int64)
    scores = np.zeros(n_edges, dtype=np.float64)
    for edge_index, (u_raw, v_raw) in enumerate(edge_keys):
        u, v = int(u_raw), int(v_raw)
        visited = {u, v}
        frontier = {u, v}
        for _ in range(2):
            next_frontier = set()
            for node in frontier:
                next_frontier.update(adjacency[node])
            next_frontier.difference_update(visited)
            visited.update(next_frontier)
            frontier = next_frontier
            if not frontier:
                break
        nodes = np.fromiter(visited, dtype=np.int64)
        total_degree = int(np.sum(degree[nodes])) - 2
        total_pass = int(np.sum(pass_count[nodes])) - int(directed_uv[edge_index]) - int(directed_vu[edge_index])
        scores[edge_index] = (
            1.0 - total_pass / total_degree if total_degree > 0 else 0.0
        )
    return scores


def _component_state(edge_keys: np.ndarray, edge_mask: np.ndarray,
                     n_points: int) -> tuple[np.ndarray, np.ndarray]:
    kept = edge_keys[np.asarray(edge_mask, dtype=bool)]
    if len(kept):
        rows = np.concatenate((kept[:, 0], kept[:, 1]))
        cols = np.concatenate((kept[:, 1], kept[:, 0]))
        graph = coo_matrix(
            (np.ones(len(rows), dtype=np.int8), (rows, cols)),
            shape=(n_points, n_points),
        ).tocsr()
    else:
        graph = coo_matrix((n_points, n_points), dtype=np.int8).tocsr()
    n_components, labels = connected_components(graph, directed=False)
    edge_counts = np.zeros(n_components, dtype=np.int64)
    if len(kept):
        np.add.at(edge_counts, labels[kept[:, 0]], 1)
    return labels, edge_counts


def _grow_components(edge_keys: np.ndarray, orig_edge_sizes: np.ndarray,
                     blue_mask: np.ndarray, hard_allowed: np.ndarray,
                     initial_mask: np.ndarray, mild_limit: float,
                     n_points: int, component_min_edges: int) -> dict[str, Any]:
    """Grow components using frozen component-local statistics per wave.

    The largest current component is processed first.  Within a wave the
    component median and IQR are fixed; accepted edges are then added as a
    batch, the component labels are recomputed, and the statistics are
    recomputed for the next wave.  Components smaller than
    ``component_min_edges`` are not excluded from growth; that parameter is a
    reporting/coloring threshold, matching the experiment script.
    """
    final_mask = np.asarray(initial_mask, dtype=bool).copy()
    grown_mask = np.zeros(len(edge_keys), dtype=bool)
    automatic_mask = np.zeros(len(edge_keys), dtype=bool)
    blue_growth_mask = np.zeros(len(edge_keys), dtype=bool)
    done_nodes = np.zeros(n_points, dtype=bool)
    iterations = 0
    growth_waves = []

    while True:
        labels, edge_counts = _component_state(edge_keys, final_mask, n_points)
        available = [
            c for c in range(len(edge_counts))
            if not np.any(done_nodes[labels == c])
        ]
        if not available:
            break
        component = max(
            available,
            key=lambda c: (int(edge_counts[c]), int(np.count_nonzero(labels == c))),
        )
        active_nodes = labels == component
        if edge_counts[component] == 0:
            done_nodes[active_nodes] = True
            continue

        while True:
            incident = active_nodes[edge_keys[:, 0]] | active_nodes[edge_keys[:, 1]]
            internal = active_nodes[edge_keys[:, 0]] & active_nodes[edge_keys[:, 1]]
            internal_lengths = orig_edge_sizes[final_mask & internal]
            if len(internal_lengths) == 0:
                break
            local = _percentile_stats(internal_lengths)
            mild_threshold = local["median"] + float(mild_limit) * local["iqr"]

            boundary = incident & ~final_mask
            automatic = boundary & hard_allowed & (orig_edge_sizes <= mild_threshold)
            gray_blue = boundary & hard_allowed & ~automatic & blue_mask
            accepted = automatic | gray_blue
            if not np.any(accepted):
                break

            final_mask[accepted] = True
            grown_mask[accepted] = True
            automatic_mask[automatic] = True
            blue_growth_mask[gray_blue] = True
            iterations += 1
            growth_waves.append({
                "component_before": int(component),
                "accepted": int(np.count_nonzero(accepted)),
                "automatic": int(np.count_nonzero(automatic)),
                "blue": int(np.count_nonzero(gray_blue)),
                "mild_threshold": float(mild_threshold),
                "component_median": local["median"],
                "component_iqr": local["iqr"],
            })

            labels, edge_counts = _component_state(edge_keys, final_mask, n_points)
            active_point = int(np.flatnonzero(active_nodes)[0])
            active_nodes = labels == labels[active_point]

        done_nodes[active_nodes] = True

    labels, edge_counts = _component_state(edge_keys, final_mask, n_points)
    return {
        "final_mask": final_mask,
        "grown_mask": grown_mask,
        "automatic_growth_mask": automatic_mask,
        "blue_growth_mask": blue_growth_mask,
        "labels": labels,
        "edge_counts": edge_counts,
        "large_components": edge_counts > int(component_min_edges),
        "iterations": int(iterations),
        "growth_waves": growth_waves,
    }


def _grow_components_fast(edge_keys: np.ndarray, orig_edge_sizes: np.ndarray,
                          blue_mask: np.ndarray, hard_allowed: np.ndarray,
                          initial_mask: np.ndarray, mild_limit: float,
                          n_points: int, component_min_edges: int,
                          original_hard_mask: np.ndarray | None = None,
                          projected_hard_mask: np.ndarray | None = None,
                          growth_relaxed_hard_allowed: np.ndarray | None = None,
                          component_selectivity: np.ndarray | None = None,
                          local_knn_selectivity_threshold: float = 1.0,
                          selectivity_scope: str = "edge",
                          open_space_relaxation: bool = False,
                          outer_long_ratio: float = 3.0,
                          outer_relaxation: float = 0.25,
                          outer_transition_width: float = 0.4054651081,
                          ) -> dict[str, Any]:
    """Equivalent growth traversal using a dynamic union-find graph.

    The reference implementation above intentionally mirrors the exploratory
    code and rebuilds connected components after every wave.  That becomes
    unnecessarily expensive when the mutual-kNN seed graph contains many
    small components.  This implementation maintains component members and
    retained-edge sets incrementally; thresholding and batch growth semantics
    are unchanged.
    """
    edge_keys = np.asarray(edge_keys, dtype=np.int64)
    final_mask = np.asarray(initial_mask, dtype=bool).copy()
    grown_mask = np.zeros(len(edge_keys), dtype=bool)
    automatic_mask = np.zeros(len(edge_keys), dtype=bool)
    blue_growth_mask = np.zeros(len(edge_keys), dtype=bool)
    boundary_seen_mask = np.zeros(len(edge_keys), dtype=bool)
    rejected_boundary_mask = np.zeros(len(edge_keys), dtype=bool)
    rejected_original_hard_mask = np.zeros(len(edge_keys), dtype=bool)
    rejected_projected_hard_mask = np.zeros(len(edge_keys), dtype=bool)
    rejected_component_rule_mask = np.zeros(len(edge_keys), dtype=bool)
    open_space_growth_mask = (
        np.zeros(len(edge_keys), dtype=bool)
        if open_space_relaxation else None
    )
    boundary_attempts = 0
    rejected_attempts = 0
    rejected_original_hard_attempts = 0
    rejected_projected_hard_attempts = 0
    rejected_component_rule_attempts = 0

    if original_hard_mask is None:
        original_hard_mask = ~np.asarray(hard_allowed, dtype=bool)
    if projected_hard_mask is None:
        projected_hard_mask = np.zeros(len(edge_keys), dtype=bool)
    if growth_relaxed_hard_allowed is None:
        growth_relaxed_hard_allowed = np.asarray(hard_allowed, dtype=bool)
    if selectivity_scope not in {"edge", "component"}:
        raise ValueError("selectivity_scope must be 'edge' or 'component'")
    if selectivity_scope == "component":
        if component_selectivity is None:
            raise ValueError(
                "component_selectivity is required for component selectivity"
            )
        point_selectivity = np.asarray(component_selectivity, dtype=np.float64)
        if point_selectivity.shape != (n_points,):
            raise ValueError("component_selectivity must have one value per point")
    else:
        point_selectivity = None

    outer_score_u = outer_score_v = None
    if open_space_relaxation:
        if outer_long_ratio <= 0:
            raise ValueError("outer_long_ratio must be positive")
        if outer_relaxation < 0:
            raise ValueError("outer_relaxation must be non-negative")
        if outer_transition_width <= 0:
            raise ValueError("outer_transition_width must be positive")

        # Keep only the three longest incident edges per point.  Three are
        # sufficient because the candidate edge itself must be excluded when
        # looking for the second-longest *other* edge.
        top_lengths = np.full((n_points, 3), -np.inf, dtype=np.float64)
        top_indices = np.full((n_points, 3), -1, dtype=np.int64)
        for edge_index, (u, v) in enumerate(edge_keys):
            length = float(orig_edge_sizes[edge_index])
            for node in (int(u), int(v)):
                for position in range(3):
                    if length > top_lengths[node, position]:
                        if position < 2:
                            top_lengths[node, position + 1:] = (
                                top_lengths[node, position:-1]
                            )
                            top_indices[node, position + 1:] = (
                                top_indices[node, position:-1]
                            )
                        top_lengths[node, position] = length
                        top_indices[node, position] = edge_index
                        break

        def second_other(node: int, edge_index: int) -> float:
            values = [
                top_lengths[node, position]
                for position in range(3)
                if top_indices[node, position] != edge_index
                and np.isfinite(top_lengths[node, position])
            ]
            return float(values[1]) if len(values) >= 2 else 0.0

        second_u = np.zeros(len(edge_keys), dtype=np.float64)
        second_v = np.zeros(len(edge_keys), dtype=np.float64)
        for edge_index, (u, v) in enumerate(edge_keys):
            second_u[edge_index] = second_other(int(u), edge_index)
            second_v[edge_index] = second_other(int(v), edge_index)

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            ratio_u = second_u / np.maximum(orig_edge_sizes, 1e-12)
            ratio_v = second_v / np.maximum(orig_edge_sizes, 1e-12)
            z_u = (
                np.log(np.maximum(ratio_u, 1e-12))
                - np.log(float(outer_long_ratio))
            ) / float(outer_transition_width)
            z_v = (
                np.log(np.maximum(ratio_v, 1e-12))
                - np.log(float(outer_long_ratio))
            ) / float(outer_transition_width)
            outer_score_u = 1.0 / (1.0 + np.exp(-np.clip(z_u, -60.0, 60.0)))
            outer_score_v = 1.0 / (1.0 + np.exp(-np.clip(z_v, -60.0, 60.0)))
        outer_score_u[second_u <= 0] = 0.0
        outer_score_v[second_v <= 0] = 0.0

    parent = np.arange(n_points, dtype=np.int64)
    size = np.ones(n_points, dtype=np.int64)
    component_selectivity_sum = (
        point_selectivity.copy() if point_selectivity is not None else None
    )
    members = [{i} for i in range(n_points)]
    component_edges = [set() for _ in range(n_points)]
    adjacency = [[] for _ in range(n_points)]
    for ei, (u, v) in enumerate(edge_keys):
        adjacency[int(u)].append(ei)
        adjacency[int(v)].append(ei)
    active_roots = set()

    def find(value):
        value = int(value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def merge(u, v, edge_index=None):
        ru, rv = find(u), find(v)
        if ru != rv:
            if size[ru] < size[rv]:
                ru, rv = rv, ru
            parent[rv] = ru
            size[ru] += size[rv]
            if component_selectivity_sum is not None:
                component_selectivity_sum[ru] += component_selectivity_sum[rv]
                component_selectivity_sum[rv] = 0.0
            members[ru].update(members[rv])
            members[rv].clear()
            if component_edges[rv]:
                component_edges[ru].update(component_edges[rv])
                component_edges[rv].clear()
            if active_roots:
                active_roots.discard(rv)
                active_roots.discard(ru)
                active_roots.add(ru)
        if edge_index is not None:
            component_edges[ru].add(int(edge_index))
        return ru

    for ei in np.flatnonzero(final_mask):
        u, v = edge_keys[ei]
        merge(u, v, ei)
    # Only sufficiently substantial seed components are grown independently.
    # Smaller components can still be absorbed when a growing component
    # reaches them, but they do not each trigger their own boundary walk.
    active_roots = {
        root for root in range(n_points)
        if (
            parent[root] == root
            and len(component_edges[root]) > int(component_min_edges)
        )
    }

    done_nodes = np.zeros(n_points, dtype=bool)
    iterations = 0
    growth_waves = []

    while True:
        available = [
            root for root in active_roots
            if not np.any(done_nodes[np.fromiter(members[root], dtype=np.int64)])
        ]
        if not available:
            break
        root = max(
            available,
            key=lambda value: (
                len(component_edges[value]),
                len(members[value]),
                -min(members[value]),
            ),
        )
        while True:
            root = find(next(iter(members[root])))
            internal_indices = np.fromiter(
                component_edges[root], dtype=np.int64
            )
            local = _percentile_stats(orig_edge_sizes[internal_indices])
            mild_threshold = local["median"] + float(mild_limit) * local["iqr"]

            boundary_set = set()
            for node in members[root]:
                boundary_set.update(adjacency[node])
            boundary_indices = [ei for ei in boundary_set if not final_mask[ei]]
            if not boundary_indices:
                break
            boundary_indices = np.asarray(boundary_indices, dtype=np.int64)
            boundary_seen_mask[boundary_indices] = True
            boundary_attempts += int(len(boundary_indices))
            automatic = boundary_indices[
                hard_allowed[boundary_indices]
                & (orig_edge_sizes[boundary_indices] <= mild_threshold)
            ]
            if open_space_relaxation:
                boundary_outer_scores = np.zeros(
                    len(boundary_indices), dtype=np.float64
                )
                for position, edge_index in enumerate(boundary_indices):
                    u, v = edge_keys[edge_index]
                    ru, rv = find(u), find(v)
                    if ru == root and rv != root:
                        boundary_outer_scores[position] = outer_score_v[edge_index]
                    elif rv == root and ru != root:
                        boundary_outer_scores[position] = outer_score_u[edge_index]
                    else:
                        boundary_outer_scores[position] = max(
                            outer_score_u[edge_index], outer_score_v[edge_index]
                        )
                relaxed_thresholds = (
                    mild_threshold
                    + float(outer_relaxation)
                    * boundary_outer_scores
                    * local["iqr"]
                )
                open_space = boundary_indices[
                    hard_allowed[boundary_indices]
                    & (orig_edge_sizes[boundary_indices] > mild_threshold)
                    & (orig_edge_sizes[boundary_indices] <= relaxed_thresholds)
                    & ~blue_mask[boundary_indices]
                ]
            else:
                open_space = None
            component_score = None
            component_is_selective = True
            if selectivity_scope == "component":
                component_score = float(
                    component_selectivity_sum[root] / max(size[root], 1)
                )
                component_is_selective = (
                    component_score > float(local_knn_selectivity_threshold)
                )
            gray_blue = boundary_indices[
                growth_relaxed_hard_allowed[boundary_indices]
                & (orig_edge_sizes[boundary_indices] > mild_threshold)
                & blue_mask[boundary_indices]
                & component_is_selective
            ]
            if open_space_relaxation:
                accepted = np.concatenate((automatic, open_space, gray_blue))
            else:
                accepted = np.concatenate((automatic, gray_blue))
            accepted_set = np.zeros(len(edge_keys), dtype=bool)
            accepted_set[accepted] = True
            rejected = boundary_indices[~accepted_set[boundary_indices]]
            if len(rejected):
                rejected_boundary_mask[rejected] = True
                rejected_attempts += int(len(rejected))
                original_rejected = rejected[original_hard_mask[rejected]]
                projected_rejected = rejected[projected_hard_mask[rejected]]
                component_rejected = rejected[
                    hard_allowed[rejected]
                    & (orig_edge_sizes[rejected] > mild_threshold)
                    & ~blue_mask[rejected]
                ]
                rejected_original_hard_mask[original_rejected] = True
                rejected_projected_hard_mask[projected_rejected] = True
                rejected_component_rule_mask[component_rejected] = True
                rejected_original_hard_attempts += int(len(original_rejected))
                rejected_projected_hard_attempts += int(len(projected_rejected))
                rejected_component_rule_attempts += int(len(component_rejected))
            if len(accepted) == 0:
                break

            active_point = next(iter(members[root]))
            final_mask[accepted] = True
            grown_mask[accepted] = True
            automatic_mask[automatic] = True
            if open_space_relaxation:
                open_space_growth_mask[open_space] = True
            blue_growth_mask[gray_blue] = True
            iterations += 1
            growth_waves.append({
                "component_before": int(root),
                "accepted": int(len(accepted)),
                "automatic": int(len(automatic)),
                "open_space": int(len(open_space)) if open_space is not None else 0,
                "blue": int(len(gray_blue)),
                "mild_threshold": float(mild_threshold),
                "component_median": local["median"],
                "component_iqr": local["iqr"],
                "component_selectivity": component_score,
            })
            for ei in accepted:
                u, v = edge_keys[ei]
                merge(u, v, ei)
            root = find(active_point)

        root = find(root)
        done_nodes[np.fromiter(members[root], dtype=np.int64)] = True
        active_roots.discard(root)

    roots = [root for root in range(n_points)
             if find(root) == root and members[root]]
    labels = np.full(n_points, -1, dtype=np.int64)
    edge_counts = np.zeros(len(roots), dtype=np.int64)
    component_selectivity_scores = np.full(len(roots), np.nan, dtype=np.float64)
    for component, root in enumerate(sorted(roots, key=lambda r: min(members[r]))):
        points = np.fromiter(members[root], dtype=np.int64)
        labels[points] = component
        edge_counts[component] = len(component_edges[root])
        if component_selectivity_sum is not None:
            component_selectivity_scores[component] = (
                component_selectivity_sum[root] / max(size[root], 1)
            )
    return {
        "final_mask": final_mask,
        "grown_mask": grown_mask,
        "automatic_growth_mask": automatic_mask,
        "open_space_growth_mask": open_space_growth_mask,
        "open_space_relaxation": bool(open_space_relaxation),
        "outer_long_ratio": float(outer_long_ratio),
        "outer_relaxation": float(outer_relaxation),
        "outer_transition_width": float(outer_transition_width),
        "blue_growth_mask": blue_growth_mask,
        "labels": labels,
        "edge_counts": edge_counts,
        "component_selectivity_scores": component_selectivity_scores,
        "large_components": edge_counts > int(component_min_edges),
        "iterations": int(iterations),
        "growth_waves": growth_waves,
        "boundary_seen_mask": boundary_seen_mask,
        "rejected_boundary_mask": rejected_boundary_mask,
        "rejected_original_hard_mask": rejected_original_hard_mask,
        "rejected_projected_hard_mask": rejected_projected_hard_mask,
        "rejected_component_rule_mask": rejected_component_rule_mask,
        "boundary_attempts": int(boundary_attempts),
        "rejected_boundary_attempts": int(rejected_attempts),
        "rejected_original_hard_attempts": int(rejected_original_hard_attempts),
        "rejected_projected_hard_attempts": int(rejected_projected_hard_attempts),
        "rejected_component_rule_attempts": int(rejected_component_rule_attempts),
    }


def _short_alternative_path_mask(edge_keys: np.ndarray, edge_mask: np.ndarray,
                                 max_hops: int) -> np.ndarray:
    """Mark retained edges that have no short alternative path.

    The candidate edge itself is excluded from the breadth-first search.  A
    path is considered an alternative only when its length is strictly less
    than ``max_hops``.  All decisions use the same pre-pruning graph, so
    removing one bridge cannot accidentally turn another edge into a bridge
    during this pass.
    """
    edge_keys = np.asarray(edge_keys, dtype=np.int64)
    edge_mask = np.asarray(edge_mask, dtype=bool)
    max_hops = int(max_hops)
    if max_hops < 2:
        return edge_mask.copy()

    adjacency = [[] for _ in range(int(edge_keys.max()) + 1)]
    for edge_index in np.flatnonzero(edge_mask):
        u, v = edge_keys[edge_index]
        adjacency[int(u)].append((int(v), int(edge_index)))
        adjacency[int(v)].append((int(u), int(edge_index)))

    bridge_mask = np.zeros(len(edge_keys), dtype=bool)
    for edge_index in np.flatnonzero(edge_mask):
        start, target = map(int, edge_keys[edge_index])
        queue = [(start, 0)]
        visited = {start}
        found = False
        cursor = 0
        while cursor < len(queue):
            node, depth = queue[cursor]
            cursor += 1
            if depth >= max_hops:
                continue
            for neighbour, neighbour_edge in adjacency[node]:
                if neighbour_edge == edge_index:
                    continue
                next_depth = depth + 1
                if neighbour == target and next_depth < max_hops:
                    found = True
                    break
                if neighbour not in visited and next_depth < max_hops:
                    visited.add(neighbour)
                    queue.append((neighbour, next_depth))
            if found:
                break
        if not found:
            bridge_mask[edge_index] = True
    return bridge_mask


def component_growth_graph(
    X: np.ndarray,
    *,
    dim_reduction: str = "umap",
    back_proj: bool = True,
    umap_n_epochs: int | str = 100,
    umap_n_neighbors: int = 15,
    project_dim: int = 2,
    hard_limit: float = 1.0,
    mild_limit: float = -0.5,
    neighbor_stat: str = "median",
    projected_hard_limit: bool = True,
    multiplicative_hard_limit: bool = False,
    initial_relation: str = "mutual",
    growth_relation: str = "union",
    knn_k: int = 30,
    component_min_edges: int = 10,
    seed_hard_limit: float | None = None,
    growth_hard_limit: float | None = None,
    hard_gate_mode: str = "strict",
    seed_relaxed_hard_limit: float | None = None,
    growth_relaxed_hard_limit: float | None = None,
    local_knn_selectivity_threshold: float = 1.0,
    local_knn_selectivity_scope: str = "edge",
    bridge_pruning: bool = False,
    bridge_max_hops: int = 5,
    open_space_relaxation: bool = False,
    outer_long_ratio: float = 3.0,
    outer_relaxation: float = 0.25,
    outer_transition_width: float = 0.4054651081,
    edge_rule: Any = None,
    edge_rule_scope: str = "both",
) -> dict[str, Any]:
    """Build and grow the component graph used by the growth backend.

    Projection and Delaunay topology are produced by the same helper as the
    legacy backend.  Edge lengths are then extracted twice: from ``X`` for
    original-space decisions and from ``X_proj`` for projected-space checks.
    In ``strict`` mode the hard gate is the intersection of the two checks.
    ``separate_limits`` uses ``hard_limit`` exclusively for seed construction
    and ``growth_hard_limit`` exclusively for the subsequent growth phase;
    it has no relaxed branch or selectivity gate.
    ``knn_relaxed`` additionally requires a local selectivity test, whereas
    ``knn_global`` applies the relaxed threshold to every eligible union-kNN
    edge without a selectivity gate.
    ``knn_growth_relaxed`` is the conservative variant: seeds always satisfy
    the strict seed gate and a relaxed *growth* edge must be selective at both
    of its endpoints.  It deliberately leaves the historical ``knn_relaxed``
    behavior unchanged for comparison.
    The selectivity test can be computed per candidate edge (the historical
    behavior) or as a cached mean over the currently growing component.
    """
    X = np.asarray(X, dtype=np.float64)
    if project_dim not in (2, 3):
        raise ValueError("project_dim must be 2 or 3")
    if growth_relation != "union":
        raise ValueError("component growth currently uses union kNN")
    if initial_relation not in {"union", "mutual"}:
        raise ValueError("initial_relation must be 'union' or 'mutual'")
    if hard_gate_mode not in {
        "strict", "separate_limits", "knn_relaxed", "knn_growth_relaxed",
        "knn_global"
    }:
        raise ValueError(
            "hard_gate_mode must be 'strict', 'separate_limits', "
            "'knn_relaxed', 'knn_growth_relaxed', or 'knn_global'"
        )
    if not -1.0 <= float(local_knn_selectivity_threshold) <= 1.0:
        raise ValueError(
            "local_knn_selectivity_threshold must be in [-1, 1]"
        )
    if local_knn_selectivity_scope not in {"edge", "edge_2hop", "component"}:
        raise ValueError(
            "local_knn_selectivity_scope must be 'edge', 'edge_2hop', "
            "or 'component'"
        )
    if (
        hard_gate_mode == "knn_growth_relaxed"
        and local_knn_selectivity_scope != "edge"
    ):
        raise ValueError(
            "knn_growth_relaxed requires local_knn_selectivity_scope='edge'"
        )
    if int(bridge_max_hops) < 2:
        raise ValueError("bridge_max_hops must be at least 2")
    bridge_pruning = bool(bridge_pruning)
    if seed_hard_limit is None:
        seed_hard_limit = hard_limit
    if growth_hard_limit is None:
        growth_hard_limit = hard_limit
    if hard_gate_mode == "separate_limits":
        # This mode deliberately has exactly two independent gates: the
        # public HARD_LIMIT for seed construction and GROWTH_HARD_LIMIT for
        # every later growth decision.  Historical seed/relaxed overrides do
        # not participate.
        seed_hard_limit = hard_limit
    if seed_relaxed_hard_limit is None:
        seed_relaxed_hard_limit = seed_hard_limit
    if growth_relaxed_hard_limit is None:
        growth_relaxed_hard_limit = hard_limit

    triangles, _, _, X_proj = get_triangles_with_edges(
        X, project_dim=project_dim, method=dim_reduction, back_proj=back_proj,
        umap_n_epochs=umap_n_epochs, umap_n_neighbors=umap_n_neighbors,
    )
    edge_keys_list, orig_sizes, _, _ = _extract_edge_data(
        triangles, X_proj, X_size=X
    )
    edge_keys = np.asarray(edge_keys_list, dtype=np.int64)
    _, proj_sizes, _, _ = _extract_edge_data(triangles, X_proj)

    growth_orig_hard_threshold, growth_orig_stats = _hard_threshold(
        orig_sizes, growth_hard_limit, neighbor_stat, multiplicative_hard_limit
    )
    seed_orig_hard_threshold, seed_orig_stats = _hard_threshold(
        orig_sizes, seed_hard_limit, neighbor_stat, multiplicative_hard_limit
    )
    if projected_hard_limit:
        growth_proj_hard_threshold, growth_proj_stats = _hard_threshold(
            proj_sizes, growth_hard_limit, neighbor_stat, multiplicative_hard_limit
        )
        projected_hard_mask = proj_sizes > growth_proj_hard_threshold
    else:
        growth_proj_hard_threshold, growth_proj_stats = None, None
        projected_hard_mask = np.zeros(len(edge_keys), dtype=bool)
    if projected_hard_limit:
        seed_proj_hard_threshold, seed_proj_stats = _hard_threshold(
            proj_sizes, seed_hard_limit, neighbor_stat, multiplicative_hard_limit
        )
        seed_projected_hard_mask = proj_sizes > seed_proj_hard_threshold
    else:
        seed_proj_hard_threshold, seed_proj_stats = None, None
        seed_projected_hard_mask = np.zeros(len(edge_keys), dtype=bool)
    original_hard_mask = orig_sizes > growth_orig_hard_threshold
    hard_allowed = ~(original_hard_mask | projected_hard_mask)
    seed_original_hard_mask = orig_sizes > seed_orig_hard_threshold
    seed_hard_allowed = ~(seed_original_hard_mask | seed_projected_hard_mask)

    # Optional learned edge rule (see ``edge_rule.py``).  It is an *additional*
    # veto layered on top of the length gates, never a relaxation: an edge the
    # rule prunes is removed, but an edge it keeps still has to pass the
    # existing thresholds.  ``edge_rule`` is None by default, so this block is
    # inert unless a caller opts in.
    edge_rule_prune_mask = None
    if edge_rule is not None:
        edge_rule_prune_mask = np.asarray(
            edge_rule(X, X_proj, edge_keys, orig_sizes, proj_sizes), dtype=bool
        )
        if edge_rule_prune_mask.shape != (len(edge_keys),):
            raise ValueError(
                f"edge_rule returned shape {edge_rule_prune_mask.shape}, "
                f"expected ({len(edge_keys)},)"
            )
        if edge_rule_scope not in {"both", "seed", "growth"}:
            raise ValueError(f"unknown edge_rule_scope: {edge_rule_scope!r}")

    directed_uv, directed_vu, local_selectivity, local_degree = (
        _original_knn_directed_relations(X, edge_keys, knn_k)
    )
    growth_blue = directed_uv | directed_vu
    initial_blue = (
        directed_uv | directed_vu
        if initial_relation == "union"
        else directed_uv & directed_vu
    )
    edge_u, edge_v = edge_keys[:, 0], edge_keys[:, 1]
    initial_selectivity = np.minimum(
        local_selectivity[edge_u], local_selectivity[edge_v]
    )
    # Historical ``knn_relaxed`` uses the more selective directed endpoint.
    # Keep that quantity intact for reproducibility, but also retain the
    # conservative both-endpoint value used by ``knn_growth_relaxed``.
    growth_selectivity = np.maximum(
        np.where(directed_uv, local_selectivity[edge_u], 0.0),
        np.where(directed_vu, local_selectivity[edge_v], 0.0),
    )
    growth_selectivity_both = np.minimum(
        local_selectivity[edge_u], local_selectivity[edge_v]
    )
    two_hop_edge_selectivity = None
    if local_knn_selectivity_scope == "edge_2hop":
        two_hop_edge_selectivity = _two_hop_edge_selectivity(
            edge_keys, local_selectivity, local_degree, directed_uv, directed_vu,
            len(X),
        )
        initial_selectivity = two_hop_edge_selectivity
        growth_selectivity = two_hop_edge_selectivity
    relaxed_mode_active = (
        hard_gate_mode == "knn_global"
        or (
            hard_gate_mode in {"knn_relaxed", "knn_growth_relaxed"}
            and float(local_knn_selectivity_threshold) < 1.0
        )
    )
    if hard_gate_mode == "separate_limits":
        seed_relaxed_orig_threshold = growth_relaxed_orig_threshold = None
        seed_relaxed_proj_threshold = growth_relaxed_proj_threshold = None
        seed_relaxed_orig_stats = growth_relaxed_orig_stats = None
        seed_relaxed_proj_stats = growth_relaxed_proj_stats = None
        seed_relaxed_allowed = seed_hard_allowed.copy()
        growth_relaxed_allowed = hard_allowed.copy()
        growth_allowed = hard_allowed.copy()
        initial_allowed = seed_hard_allowed
        seed_selective = np.zeros(len(edge_keys), dtype=bool)
        growth_selective = np.zeros(len(edge_keys), dtype=bool)
    elif relaxed_mode_active:
        seed_relaxed_orig_threshold, seed_relaxed_orig_stats = _hard_threshold(
            orig_sizes, seed_relaxed_hard_limit, neighbor_stat,
            multiplicative_hard_limit,
        )
        growth_relaxed_orig_threshold, growth_relaxed_orig_stats = _hard_threshold(
            orig_sizes, growth_relaxed_hard_limit, neighbor_stat,
            multiplicative_hard_limit,
        )
        if projected_hard_limit:
            seed_relaxed_proj_threshold, seed_relaxed_proj_stats = _hard_threshold(
                proj_sizes, seed_relaxed_hard_limit, neighbor_stat,
                multiplicative_hard_limit,
            )
            growth_relaxed_proj_threshold, growth_relaxed_proj_stats = _hard_threshold(
                proj_sizes, growth_relaxed_hard_limit, neighbor_stat,
                multiplicative_hard_limit,
            )
            seed_relaxed_projected_mask = proj_sizes > seed_relaxed_proj_threshold
            growth_relaxed_projected_mask = proj_sizes > growth_relaxed_proj_threshold
        else:
            seed_relaxed_proj_threshold = growth_relaxed_proj_threshold = None
            seed_relaxed_proj_stats = growth_relaxed_proj_stats = None
            seed_relaxed_projected_mask = growth_relaxed_projected_mask = np.zeros(
                len(edge_keys), dtype=bool
            )
        seed_relaxed_original_mask = orig_sizes > seed_relaxed_orig_threshold
        growth_relaxed_original_mask = orig_sizes > growth_relaxed_orig_threshold
        seed_relaxed_allowed = ~(
            seed_relaxed_original_mask | seed_relaxed_projected_mask
        )
        growth_relaxed_allowed = ~(
            growth_relaxed_original_mask | growth_relaxed_projected_mask
        )
        if hard_gate_mode == "knn_global":
            seed_selective = np.ones(len(edge_keys), dtype=bool)
            growth_selective = np.ones(len(edge_keys), dtype=bool)
            initial_allowed = seed_hard_allowed | (
                initial_blue & seed_relaxed_allowed
            )
            growth_allowed = hard_allowed | (
                growth_blue & growth_relaxed_allowed
            )
        else:
            seed_selective = (
                initial_selectivity > float(local_knn_selectivity_threshold)
            )
            growth_selective = (
                (
                    growth_selectivity_both
                    if hard_gate_mode == "knn_growth_relaxed"
                    else growth_selectivity
                )
                > float(local_knn_selectivity_threshold)
            )
            if hard_gate_mode == "knn_growth_relaxed":
                # Do not let a relaxed edge seed an erroneous component.  A
                # component must first be established by the strict gate.
                initial_allowed = seed_hard_allowed
            else:
                initial_allowed = seed_hard_allowed | (
                    initial_blue & seed_selective & seed_relaxed_allowed
                )
        if (
            hard_gate_mode in {"knn_relaxed", "knn_growth_relaxed"}
            and local_knn_selectivity_scope in {"edge", "edge_2hop"}
        ):
            growth_allowed = hard_allowed | (
                growth_blue & growth_selective & growth_relaxed_allowed
            )
        elif hard_gate_mode in {"knn_relaxed", "knn_growth_relaxed"}:
            # The component-level selectivity gate is applied dynamically in
            # _grow_components_fast after the current component is known.
            growth_allowed = hard_allowed | (
                growth_blue & growth_relaxed_allowed
            )
    else:
        seed_relaxed_orig_threshold = growth_relaxed_orig_threshold = None
        seed_relaxed_proj_threshold = growth_relaxed_proj_threshold = None
        seed_relaxed_orig_stats = growth_relaxed_orig_stats = None
        seed_relaxed_proj_stats = growth_relaxed_proj_stats = None
        seed_relaxed_allowed = seed_hard_allowed.copy()
        growth_relaxed_allowed = hard_allowed.copy()
        growth_allowed = hard_allowed.copy()
        initial_allowed = seed_hard_allowed
        seed_selective = np.zeros(len(edge_keys), dtype=bool)
        growth_selective = np.zeros(len(edge_keys), dtype=bool)
    # Apply the learned rule at the single chokepoint where every gate mode has
    # converged to its final allow-masks.  Vetoing earlier would miss the
    # relaxed branches, which rebuild their own masks from raw thresholds.
    if edge_rule_prune_mask is not None:
        if edge_rule_scope in {"both", "seed"}:
            initial_allowed = initial_allowed & ~edge_rule_prune_mask
        if edge_rule_scope in {"both", "growth"}:
            # Both are consumed by _grow_components: ``hard_allowed`` is the
            # strict gate, ``growth_allowed`` the relaxed one.
            hard_allowed = hard_allowed & ~edge_rule_prune_mask
            growth_allowed = growth_allowed & ~edge_rule_prune_mask

    initial_mask = initial_blue & initial_allowed
    state = _grow_components_fast(
        edge_keys, orig_sizes, growth_blue, hard_allowed, initial_mask,
        mild_limit, len(X), component_min_edges,
        original_hard_mask=original_hard_mask,
        projected_hard_mask=projected_hard_mask,
        growth_relaxed_hard_allowed=growth_allowed,
        component_selectivity=(
            local_selectivity
            if (
                hard_gate_mode == "knn_relaxed"
                and local_knn_selectivity_scope == "component"
            )
            else None
        ),
        local_knn_selectivity_threshold=local_knn_selectivity_threshold,
        selectivity_scope=(
            "component"
            if (
                hard_gate_mode == "knn_relaxed"
                and local_knn_selectivity_scope == "component"
            )
            else "edge"
        ),
        open_space_relaxation=open_space_relaxation,
        outer_long_ratio=outer_long_ratio,
        outer_relaxation=outer_relaxation,
        outer_transition_width=outer_transition_width,
    )
    growth_final_mask = state["final_mask"].copy()
    # Bridge pruning is deliberately the final stage. It sees exactly the
    # edge set produced after hard filtering, initialization, and complete
    # component growth.
    bridge_base_mask = growth_final_mask
    if bridge_pruning:
        bridge_pruned_mask = _short_alternative_path_mask(
            edge_keys, bridge_base_mask, bridge_max_hops
        )
        state["final_mask"] = bridge_base_mask & ~bridge_pruned_mask
        state["labels"], state["edge_counts"] = _component_state(
            edge_keys, state["final_mask"], len(X)
        )
        state["large_components"] = (
            state["edge_counts"] > int(component_min_edges)
        )
    else:
        bridge_pruned_mask = np.zeros(len(edge_keys), dtype=bool)
    state.update({
        "triangles": triangles,
        "X_proj": X_proj,
        "edge_keys": edge_keys,
        "orig_edge_sizes": orig_sizes,
        "projected_edge_sizes": proj_sizes,
        "edge_rule_prune_mask": edge_rule_prune_mask,
        "edge_rule_scope": edge_rule_scope if edge_rule is not None else None,
        "original_hard_mask": original_hard_mask,
        "projected_hard_mask": projected_hard_mask,
        "hard_allowed": hard_allowed,
        "seed_original_hard_mask": seed_original_hard_mask,
        "seed_projected_hard_mask": seed_projected_hard_mask,
        "seed_hard_allowed": seed_hard_allowed,
        "hard_gate_mode": hard_gate_mode,
        "relaxed_mode_active": bool(relaxed_mode_active),
        "seed_relaxed_hard_limit": float(seed_relaxed_hard_limit),
        "growth_relaxed_hard_limit": float(growth_relaxed_hard_limit),
        "local_knn_selectivity_threshold": float(local_knn_selectivity_threshold),
        "local_knn_selectivity_scope": local_knn_selectivity_scope,
        "local_knn_selectivity": local_selectivity,
        "local_knn_degree": local_degree,
        "initial_edge_selectivity": initial_selectivity,
        "growth_edge_selectivity": growth_selectivity,
        "growth_edge_selectivity_both": growth_selectivity_both,
        "two_hop_edge_selectivity": two_hop_edge_selectivity,
        "seed_relaxed_allowed": seed_relaxed_allowed,
        "growth_relaxed_allowed": growth_allowed,
        "growth_relaxed_threshold_allowed": growth_relaxed_allowed,
        "seed_relaxed_orig_threshold": None if seed_relaxed_orig_threshold is None else float(seed_relaxed_orig_threshold),
        "growth_relaxed_orig_threshold": None if growth_relaxed_orig_threshold is None else float(growth_relaxed_orig_threshold),
        "seed_relaxed_projected_threshold": None if seed_relaxed_proj_threshold is None else float(seed_relaxed_proj_threshold),
        "growth_relaxed_projected_threshold": None if growth_relaxed_proj_threshold is None else float(growth_relaxed_proj_threshold),
        "initial_blue_mask": initial_blue,
        "growth_blue_mask": growth_blue,
        "initial_mask": initial_mask,
        "growth_final_mask": growth_final_mask,
        "bridge_base_mask": bridge_base_mask,
        "bridge_pruned_mask": bridge_pruned_mask,
        "bridge_pruning": bridge_pruning,
        "bridge_max_hops": int(bridge_max_hops),
        "orig_hard_threshold": float(growth_orig_hard_threshold),
        "projected_hard_threshold": None if growth_proj_hard_threshold is None else float(growth_proj_hard_threshold),
        "seed_orig_hard_threshold": float(seed_orig_hard_threshold),
        "seed_projected_hard_threshold": None if seed_proj_hard_threshold is None else float(seed_proj_hard_threshold),
        "orig_stats": growth_orig_stats,
        "projected_stats": growth_proj_stats,
        "seed_orig_stats": seed_orig_stats,
        "seed_projected_stats": seed_proj_stats,
        "hard_limit": float(hard_limit),
        "growth_hard_limit": float(growth_hard_limit),
        "seed_hard_limit": float(seed_hard_limit),
        "mild_limit": float(mild_limit),
        "initial_relation": initial_relation,
        "growth_relation": growth_relation,
        "knn_k": int(knn_k),
        "component_min_edges": int(component_min_edges),
        "project_dim": int(project_dim),
        "projected_hard_limit": bool(projected_hard_limit),
        "open_space_relaxation": bool(open_space_relaxation),
        "outer_long_ratio": float(outer_long_ratio),
        "outer_relaxation": float(outer_relaxation),
        "outer_transition_width": float(outer_transition_width),
    })
    return state


def _labels_from_components(labels: np.ndarray, edge_counts: np.ndarray,
                            min_cluster_size: int) -> np.ndarray:
    result = np.full(len(labels), -1, dtype=int)
    label_id = 0
    for component, count in enumerate(np.bincount(labels)):
        points = np.flatnonzero(labels == component)
        if len(points) >= int(min_cluster_size):
            result[points] = label_id
            label_id += 1
    return result


def _merge_component_outliers(
    X: np.ndarray,
    edge_keys: np.ndarray,
    orig_edge_sizes: np.ndarray,
    final_mask: np.ndarray,
    component_labels: np.ndarray,
    min_cluster_size: int,
    outlier_limit: float | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Merge final small-component points into nearby dominant clusters.

    This is deliberately a post-processing step.  It does not alter growth
    or bridge decisions.  A point currently labelled ``-1`` is considered
    only when most of its Delaunay neighbours belonging to a retained
    cluster belong to the same target cluster.  Its nearest original-space
    distance to that target cluster must then satisfy::

        distance <= cluster_median + outlier_limit * cluster_iqr

    The cluster statistics are computed from retained intra-cluster graph
    edges in the original space.  ``None`` disables the operation.
    """
    labels = _labels_from_components(
        component_labels, np.zeros(0, dtype=np.int64), min_cluster_size
    )
    # The helper above only needs component labels for the point-size test;
    # reconstructing from components is done directly here because the
    # component edge counts are not needed for the merge rule.
    labels = np.asarray(labels, dtype=np.int64)
    merged = labels.copy()
    merged_mask = np.zeros(len(labels), dtype=bool)
    target_for_point = np.full(len(labels), -1, dtype=np.int64)
    distance_for_point = np.full(len(labels), np.nan, dtype=np.float64)
    threshold_for_point = np.full(len(labels), np.nan, dtype=np.float64)

    diagnostics = {
        "outlier_limit": None if outlier_limit is None else float(outlier_limit),
        "candidate_count": int(np.count_nonzero(labels < 0)),
        "dominant_connection_count": 0,
        "merged_count": 0,
        "merged_mask": merged_mask,
        "target_for_point": target_for_point,
        "distance_for_point": distance_for_point,
        "threshold_for_point": threshold_for_point,
    }
    if outlier_limit is None:
        return merged, diagnostics

    outlier_limit = float(outlier_limit)
    if not np.isfinite(outlier_limit):
        raise ValueError("component_growth_outlier_limit must be finite or None")

    edge_keys = np.asarray(edge_keys, dtype=np.int64)
    orig_edge_sizes = np.asarray(orig_edge_sizes, dtype=np.float64)
    final_mask = np.asarray(final_mask, dtype=bool)
    X = np.asarray(X, dtype=np.float64)
    valid = labels >= 0
    if not np.any(valid) or not np.any(labels < 0):
        return merged, diagnostics

    # Count all Delaunay connections from an outlier point to retained
    # clusters.  This deliberately uses the full triangulation, because the
    # point may be an outlier precisely because its connecting edge was
    # removed during pruning.
    connections: list[dict[int, int]] = [dict() for _ in range(len(labels))]
    for u, v in edge_keys:
        lu, lv = int(labels[u]), int(labels[v])
        if lu < 0 and lv >= 0:
            counts = connections[int(u)]
            counts[lv] = counts.get(lv, 0) + 1
        elif lv < 0 and lu >= 0:
            counts = connections[int(v)]
            counts[lu] = counts.get(lu, 0) + 1

    target_by_point: dict[int, int] = {}
    for point, counts in enumerate(connections):
        if not counts:
            continue
        target, count = max(counts.items(), key=lambda item: (item[1], -item[0]))
        total = sum(counts.values())
        if count > total / 2.0:
            target_by_point[point] = int(target)
            diagnostics["dominant_connection_count"] += 1
    if not target_by_point:
        return merged, diagnostics

    # Compute original-space median/IQR from retained intra-cluster edges.
    cluster_edge_lengths: dict[int, list[float]] = {}
    for edge_index in np.flatnonzero(final_mask):
        u, v = edge_keys[edge_index]
        lu, lv = int(labels[u]), int(labels[v])
        if lu >= 0 and lu == lv:
            cluster_edge_lengths.setdefault(lu, []).append(
                float(orig_edge_sizes[edge_index])
            )
    cluster_stats: dict[int, tuple[float, float]] = {}
    for cluster, values in cluster_edge_lengths.items():
        if values:
            q25, median, q75 = np.percentile(values, [25.0, 50.0, 75.0])
            cluster_stats[cluster] = (
                float(median),
                float(max(q75 - q25, 1e-12)),
            )

    # Fit one nearest-neighbour index per target cluster, rather than doing
    # an outlier-by-cluster all-pairs distance calculation.
    points_by_target: dict[int, list[int]] = {}
    for point, target in target_by_point.items():
        points_by_target.setdefault(target, []).append(point)
    for target, points in points_by_target.items():
        if target not in cluster_stats:
            continue
        cluster_points = np.flatnonzero(labels == target)
        if len(cluster_points) == 0:
            continue
        nearest = NearestNeighbors(n_neighbors=1, metric="euclidean")
        nearest.fit(X[cluster_points])
        distances, _ = nearest.kneighbors(X[np.asarray(points)])
        median, iqr = cluster_stats[target]
        threshold = median + outlier_limit * iqr
        for point, distance in zip(points, distances[:, 0]):
            distance = float(distance)
            distance_for_point[point] = distance
            threshold_for_point[point] = threshold
            if distance <= threshold:
                merged[point] = target
                merged_mask[point] = True
                target_for_point[point] = target

    diagnostics["merged_count"] = int(np.count_nonzero(merged_mask))
    return merged, diagnostics


def _assign_anomalies_dijkstra(
    labels: np.ndarray,
    edge_keys: np.ndarray,
    proj_edge_sizes: np.ndarray,
    n_points: int,
    penalty_power: float,
    stop_ratio: float,
    max_rounds: int | None = None,
    snapshot_rounds: set[int] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reclaim label ``-1`` points via competitive penalized-Dijkstra search.

    Ported from ``studies/growth_dijkstra_anomaly_assign.py`` (see that
    module's docstring for the full derivation and the study results that
    motivated it). This is deliberately a *post*-processing step, run after
    ``labels`` already reflects every other growth/merge/outlier decision:
    it never changes which points belong to which cluster among points that
    already have a cluster, and it never merges two clusters into one -- it
    only offers already-formed clusters a chance to claim points that were
    left as noise (``label == -1``).

    Each cluster ``c`` gets a fixed baseline ``avg_c``, the mean length (in
    *projected* space -- original-space distances concentrate in high
    dimensions and lose the ability to tell a normal edge from an outlier
    one) of its own internal edges. A candidate edge's cost when explored
    from cluster ``c`` is ``length * (length / avg_c) ** penalty_power``,
    penalizing longer-than-normal hops superlinearly so a chain of several
    short, in-scale hops beats one long, out-of-scale hop even when the raw
    lengths sum to the same total. Multi-source Dijkstra runs from every
    cluster at once, one accepted claim per cluster per round so no cluster
    races ahead of the others; contested points go to whichever cluster
    reaches them at lower cumulative penalized cost. A cluster stops
    reaching once its cheapest remaining option exceeds
    ``stop_ratio * avg_c`` -- points beyond that stay noise.

    ``stop_ratio=5.0`` was tried as a looser default and evaluated against
    ``stop_ratio=3.0`` (the current default) across 146 study datasets: mean
    assigned ARI was 0.4780 vs 0.4779 (essentially flat, delta -0.0002), with
    22 datasets improving and 19 regressing. The regressions were
    concentrated in genuinely noisy datasets (e.g. ``blobs_with_noise_*``,
    ``blobs_informative_subspace_*``), where the looser gate lets clusters
    claim true noise points via long low-scale-fit hops; the gains were
    mostly clean, well-separated 2D blobs already close to ARI 1.0. Net: not
    worth raising past 3.0.

    ``snapshot_rounds``, when given, records a copy of the per-point
    ``owner`` array immediately after each listed round completes (for
    plotting assignment progress -- see
    ``studies/plot_growth_dijkstra_anomaly_assign.py``); ``0`` captures the
    state before any round runs. ``diagnostics["grown_mask"]`` always marks
    which edges were actually used to claim a point, snapshots or not.
    """
    labels = np.asarray(labels, dtype=np.int64)
    edge_keys = np.asarray(edge_keys, dtype=np.int64)
    proj_edge_sizes = np.asarray(proj_edge_sizes, dtype=np.float64)
    n_edges = len(edge_keys)
    n_clusters = int(labels.max()) + 1 if np.any(labels >= 0) else 0
    snapshot_rounds = snapshot_rounds or set()
    diagnostics = {
        "penalty_power": float(penalty_power),
        "stop_ratio": float(stop_ratio),
        "n_anomalies_in": int(np.count_nonzero(labels < 0)),
        "n_anomalies_out": 0,
        "n_rounds": 0,
        "grown_mask": np.zeros(n_edges, dtype=bool),
        "snapshots": {},
    }
    if n_clusters == 0 or diagnostics["n_anomalies_in"] == 0:
        diagnostics["n_anomalies_out"] = diagnostics["n_anomalies_in"]
        if 0 in snapshot_rounds:
            diagnostics["snapshots"][0] = labels.copy()
        if -1 in snapshot_rounds:
            diagnostics["snapshots"][-1] = labels.copy()
        return labels.copy(), diagnostics
    if max_rounds is None:
        max_rounds = 4 * n_points + 16

    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(n_points)]
    for edge_index in range(n_edges):
        u, v = int(edge_keys[edge_index, 0]), int(edge_keys[edge_index, 1])
        adjacency[u].append((v, edge_index))
        adjacency[v].append((u, edge_index))

    # owner[point] is the cluster id that currently claims point (-1 =
    # anomaly, still unclaimed). Clusters never merge in this stage, so
    # owner[point] is always the point's final cluster directly.
    owner = labels.copy()

    # avg[c] is frozen once, from c's own edges, and never updated as c
    # absorbs anomalies during this stage.
    sum_len = np.zeros(max(n_clusters, 1), dtype=np.float64)
    count_len = np.zeros(max(n_clusters, 1), dtype=np.int64)
    for edge_index in range(n_edges):
        u, v = edge_keys[edge_index]
        cu, cv = int(labels[u]), int(labels[v])
        if cu >= 0 and cu == cv:
            sum_len[cu] += proj_edge_sizes[edge_index]
            count_len[cu] += 1
    global_median = float(np.median(proj_edge_sizes)) if n_edges else 1.0
    avg = np.where(count_len > 0, sum_len / np.maximum(count_len, 1), global_median)

    heaps: list[list[tuple[float, int, int]]] = [[] for _ in range(n_clusters)]
    for cluster in range(n_clusters):
        for node in np.flatnonzero(labels == cluster):
            for neighbor, edge_index in adjacency[int(node)]:
                if labels[neighbor] == cluster:
                    continue
                length = proj_edge_sizes[edge_index]
                weight = length * (length / avg[cluster]) ** penalty_power
                heapq.heappush(heaps[cluster], (weight, neighbor, edge_index))
    active_roots = {cluster for cluster in range(n_clusters) if heaps[cluster]}

    grown_mask = diagnostics["grown_mask"]
    snapshots = diagnostics["snapshots"]
    if 0 in snapshot_rounds:
        snapshots[0] = owner.copy()

    round_index = 0
    while round_index < max_rounds and active_roots:
        root_candidate: dict[int, tuple[float, int, int]] = {}
        dead_roots = []
        for root in list(active_roots):
            heap = heaps[root]
            while heap:
                dist, node, edge_index = heap[0]
                if owner[node] != -1:
                    heapq.heappop(heap)
                    continue
                length = proj_edge_sizes[edge_index]
                if length > stop_ratio * avg[root]:
                    heapq.heappop(heap)
                    continue
                root_candidate[root] = (dist, node, edge_index)
                break
            if not heap:
                dead_roots.append(root)
        for root in dead_roots:
            active_roots.discard(root)
        if not root_candidate:
            break

        proposals: dict[int, tuple[float, int, int]] = {}
        for root, (dist, node, edge_index) in root_candidate.items():
            current = proposals.get(node)
            if current is None or dist < current[0]:
                proposals[node] = (dist, root, edge_index)

        for node, (dist, root, edge_index) in proposals.items():
            if owner[node] != -1:
                continue
            heapq.heappop(heaps[root])
            owner[node] = root
            grown_mask[edge_index] = True
            for neighbor, next_edge_index in adjacency[node]:
                if owner[neighbor] != -1:
                    continue
                length = proj_edge_sizes[next_edge_index]
                weight = length * (length / avg[root]) ** penalty_power
                heapq.heappush(heaps[root], (dist + weight, neighbor, next_edge_index))

        round_index += 1
        if round_index in snapshot_rounds:
            snapshots[round_index] = owner.copy()

    diagnostics["n_anomalies_out"] = int(np.count_nonzero(owner < 0))
    diagnostics["n_rounds"] = round_index
    if -1 in snapshot_rounds:
        snapshots[-1] = owner.copy()
    return owner, diagnostics


def cluster_tri(
    X, prune_param=1.0, merge_param=None, min_cluster_size=10,
    dim_reduction="umap", back_proj=True, anomaly_sensitivity=0.0,
    prune_mode="component_growth", umap_n_epochs=100, umap_n_neighbors=15,
    hard_limit=1.0, mild_limit=-0.5, neighbor_stat="median", random_state=42,
    project_dim=2, profile_phases=False, projected_hard_limit=True,
    multiplicative_hard_limit=False, auto_threshold=False, prune_regime="none",
    pointwise_bridge_pruning=False, point_edge_ratio=1.5,
    point_edge_compare="ratio", point_edge_subset="all", point_edge_gap=0.5,
    bridge_chain_len=4, component_growth_initial_relation="mutual",
    component_growth_growth_relation="union", component_growth_knn=30,
    component_growth_min_edges=10,
    component_growth_seed_hard_limit=None,
    component_growth_growth_hard_limit=None,
    component_growth_hard_gate_mode="strict",
    component_growth_seed_relaxed_hard_limit=None,
    component_growth_growth_relaxed_hard_limit=None,
    component_growth_local_knn_selectivity_threshold=1.0,
    component_growth_local_knn_selectivity_scope="edge",
    component_growth_bridge_pruning=False,
    component_growth_bridge_max_hops=5,
    component_growth_outlier_limit=None,
    component_growth_open_space_relaxation=False,
    component_growth_outer_long_ratio=3.0,
    component_growth_outer_relaxation=0.25,
    component_growth_outer_transition_width=0.4054651081,
    component_growth_edge_rule=None,
    component_growth_edge_rule_scope="both",
    component_growth_anomaly_reassign=False,
    component_growth_anomaly_reassign_penalty_power=2.0,
    component_growth_anomaly_reassign_stop_ratio=3.0,
    **kwargs,
):
    """Cluster using the component-growth graph.

    Legacy-only arguments are accepted for estimator compatibility but are not
    used by this backend.  ``prune_param`` and merge/anomaly settings do not
    change the tested growth rule; ``hard_limit`` and ``mild_limit`` do.
    """
    state = component_growth_graph(
        X, dim_reduction=dim_reduction, back_proj=back_proj,
        umap_n_epochs=umap_n_epochs, umap_n_neighbors=umap_n_neighbors,
        project_dim=project_dim, hard_limit=hard_limit, mild_limit=mild_limit,
        neighbor_stat=neighbor_stat,
        projected_hard_limit=projected_hard_limit,
        multiplicative_hard_limit=multiplicative_hard_limit,
        initial_relation=component_growth_initial_relation,
        growth_relation=component_growth_growth_relation,
        knn_k=component_growth_knn,
        component_min_edges=component_growth_min_edges,
        seed_hard_limit=component_growth_seed_hard_limit,
        growth_hard_limit=component_growth_growth_hard_limit,
        hard_gate_mode=component_growth_hard_gate_mode,
        seed_relaxed_hard_limit=component_growth_seed_relaxed_hard_limit,
        growth_relaxed_hard_limit=component_growth_growth_relaxed_hard_limit,
        local_knn_selectivity_threshold=(
            component_growth_local_knn_selectivity_threshold
        ),
        local_knn_selectivity_scope=(
            component_growth_local_knn_selectivity_scope
        ),
        bridge_pruning=component_growth_bridge_pruning,
        bridge_max_hops=component_growth_bridge_max_hops,
        open_space_relaxation=component_growth_open_space_relaxation,
        outer_long_ratio=component_growth_outer_long_ratio,
        outer_relaxation=component_growth_outer_relaxation,
        outer_transition_width=component_growth_outer_transition_width,
        edge_rule=component_growth_edge_rule,
        edge_rule_scope=component_growth_edge_rule_scope,
    )
    labels = _labels_from_components(
        state["labels"], state["edge_counts"], min_cluster_size
    )
    pre_merge_labels = labels.copy()
    labels, merge_diagnostics = _merge_component_outliers(
        np.asarray(X), state["edge_keys"], state["orig_edge_sizes"],
        state["final_mask"], state["labels"], min_cluster_size,
        component_growth_outlier_limit,
    )
    state["pre_merge_labels"] = pre_merge_labels
    state["merged_labels"] = labels
    state["outlier_merge"] = merge_diagnostics
    state["component_growth_outlier_limit"] = (
        None if component_growth_outlier_limit is None
        else float(component_growth_outlier_limit)
    )
    if component_growth_anomaly_reassign:
        # ``state["merged_labels"]`` (set above) is the labeling as of right
        # before this stage runs -- the "before" side of the before/after
        # comparison, kept for diagnostics/plotting.
        labels, anomaly_diagnostics = _assign_anomalies_dijkstra(
            labels, state["edge_keys"], state["projected_edge_sizes"], len(X),
            penalty_power=component_growth_anomaly_reassign_penalty_power,
            stop_ratio=component_growth_anomaly_reassign_stop_ratio,
        )
        state["anomaly_reassign"] = anomaly_diagnostics
        state["anomaly_reassigned_labels"] = labels
    cluster_tri.last_component_growth = state
    cluster_tri.regime_used = "component_growth"
    cluster_tri.fallback_triggered = False
    cluster_tri.last_phase_times = {}
    return labels
