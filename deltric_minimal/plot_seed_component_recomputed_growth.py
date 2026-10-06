#!/usr/bin/env python3
"""Grow seed components in rounds by a component-relative hard criterion.

This is a diagnostic experiment, separate from the production component-growth
rule.  It starts from the exact ``initial_mask`` seed graph.  In each round it
evaluates every current boundary edge using statistics from the *accepted
internal graph* of that component (seed edges plus growth edges accepted in
earlier rounds),
then accepts every qualifying edge with

    (edge_length - component_median) / (component_q75 - component_median)

when that value is below ``--hard-growth-limit``.  Merely incident boundary
edges never enter the reference distribution.  Component statistics are
recomputed only after the complete round.  Boundary colors are frozen the first
time an edge is encountered, so the plot records the values used during growth.
"""

from __future__ import annotations

import argparse
from collections import deque
import heapq
from itertools import combinations
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import (
    adjusted_mutual_info_score, adjusted_rand_score, completeness_score,
    f1_score, homogeneity_score, normalized_mutual_info_score,
    precision_score, recall_score, v_measure_score,
)
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


def _first_mode_seed_mask(
    edge_lengths: np.ndarray,
    fallback_mask: np.ndarray,
    *,
    min_peak_fraction: float = 0.20,
    max_valley_fraction: float = 0.60,
    fallback_on_failure: bool = True,
) -> tuple[np.ndarray, dict]:
    """Use the first clearly separated log-length mode as a seed cutoff.

    This is deliberately a conservative special regime.  It accepts a first
    valley only when two non-trivial peaks surround a sufficiently deep valley.
    When ``fallback_on_failure`` is false, a failed mode test produces an empty
    seed mask instead of silently reverting to the median rule.
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    lengths = np.asarray(edge_lengths, dtype=np.float64)
    fallback = np.asarray(fallback_mask, dtype=bool)
    failure_mask = fallback if fallback_on_failure else np.zeros_like(fallback)
    positive = lengths[np.isfinite(lengths) & (lengths > 0.0)]
    diagnostics: dict[str, object] = {
        "mode": "median_fallback" if fallback_on_failure else "first_mode_no_fallback",
        "fallback_seed_edge_count": int(np.count_nonzero(fallback)),
        "fallback_on_failure": bool(fallback_on_failure),
    }
    if len(positive) < 40:
        diagnostics["reason"] = "too_few_positive_edges"
        return failure_mask, diagnostics

    log_lengths = np.log(positive)
    bins = int(np.clip(np.sqrt(len(log_lengths)) * 2.0, 32, 96))
    counts, edges = np.histogram(log_lengths, bins=bins)
    smooth = gaussian_filter1d(counts.astype(np.float64), sigma=1.25)
    peak_floor = max(2.0, float(np.max(smooth)) * float(min_peak_fraction))
    peaks, _ = find_peaks(smooth, height=peak_floor, prominence=peak_floor * 0.35)
    diagnostics.update({
        "log_histogram_bin_count": bins,
        "significant_peak_count": int(len(peaks)),
        "min_peak_fraction": float(min_peak_fraction),
        "max_valley_fraction": float(max_valley_fraction),
    })
    if len(peaks) < 2:
        diagnostics["reason"] = "fewer_than_two_significant_modes"
        return failure_mask, diagnostics

    first, second = int(peaks[0]), int(peaks[1])
    if second - first < 3:
        diagnostics["reason"] = "modes_not_separated"
        return failure_mask, diagnostics
    valley = first + int(np.argmin(smooth[first:second + 1]))
    valley_fraction = float(smooth[valley] / max(min(smooth[first], smooth[second]), 1e-12))
    diagnostics.update({
        "first_peak_log_length": float((edges[first] + edges[first + 1]) / 2.0),
        "second_peak_log_length": float((edges[second] + edges[second + 1]) / 2.0),
        "valley_log_length": float((edges[valley] + edges[valley + 1]) / 2.0),
        "valley_fraction_of_smaller_peak": valley_fraction,
    })
    if valley_fraction > max_valley_fraction:
        diagnostics["reason"] = "valley_not_deep_enough"
        return failure_mask, diagnostics

    cutoff = float(np.exp((edges[valley] + edges[valley + 1]) / 2.0))
    selected = lengths <= cutoff
    diagnostics.update({
        "mode": "first_mode_valley",
        "seed_cutoff": cutoff,
        "seed_edge_count": int(np.count_nonzero(selected)),
    })
    return selected, diagnostics


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


def _component_topology_metrics(edge_keys: np.ndarray, retained_mask: np.ndarray,
                                n_points: int, *, minimum_edges: int,
                                edge_lengths: np.ndarray | None = None,
                                X_proj: np.ndarray | None = None) -> dict:
    """Describe real pre-Gomory components with cheap graph topology metrics.

    E/V is near three for a compact planar Delaunay-like component, while an
    almost one-dimensional ribbon has fewer independent cycles and tends
    towards one or two.  These are diagnostics only, evaluated before any
    Gomory--Hu cut so they can later serve as a potential gate.
    """
    labels, edge_bearing = _final_graph_components(edge_keys, retained_mask, n_points)
    retained = np.flatnonzero(retained_mask)
    components: list[dict] = []
    for component in np.unique(labels[edge_bearing]):
        edge_indices = retained[
            (labels[edge_keys[retained, 0]] == component)
            & (labels[edge_keys[retained, 1]] == component)
        ]
        point_count = int(len(np.unique(edge_keys[edge_indices])))
        edge_count = int(len(edge_indices))
        if edge_count < minimum_edges or point_count == 0:
            continue
        cycle_rank = edge_count - point_count + 1
        components.append({
            "component_label": int(component),
            "edge_count": edge_count,
            "point_count": point_count,
            "edge_to_point_ratio": edge_count / point_count,
            "mean_degree": 2.0 * edge_count / point_count,
            "cycle_rank": cycle_rank,
            "cycle_rank_per_point": cycle_rank / point_count,
        })
        if edge_lengths is not None:
            local_points = np.unique(edge_keys[edge_indices])
            point_to_local = np.full(n_points, -1, dtype=np.int64)
            point_to_local[local_points] = np.arange(len(local_points))
            local_u = point_to_local[edge_keys[edge_indices, 0]]
            local_v = point_to_local[edge_keys[edge_indices, 1]]
            local_lengths = np.asarray(edge_lengths[edge_indices], dtype=np.float64)
            nearest = np.full(len(local_points), np.inf, dtype=np.float64)
            np.minimum.at(nearest, local_u, local_lengths)
            np.minimum.at(nearest, local_v, local_lengths)
            nearest_median = float(np.median(nearest))
            # The first count follows the proposed common component radius.
            # The second is density-normalized: number of neighbours no more
            # than 1.5 times each endpoint's own closest retained edge.
            common_counts = np.zeros(len(local_points), dtype=np.int64)
            np.add.at(common_counts, local_u, local_lengths <= nearest_median)
            np.add.at(common_counts, local_v, local_lengths <= nearest_median)
            comparable = local_lengths <= 1.5 * np.maximum(
                nearest[local_u], nearest[local_v],
            )
            relative_counts = np.zeros(len(local_points), dtype=np.int64)
            np.add.at(relative_counts, local_u, comparable)
            np.add.at(relative_counts, local_v, comparable)
            components[-1].update({
                "median_nearest_edge_length": nearest_median,
                "neighbors_at_or_below_component_median_nearest": {
                    "median": float(np.median(common_counts)),
                    "mean": float(np.mean(common_counts)),
                    "q25": float(np.quantile(common_counts, 0.25)),
                    "q75": float(np.quantile(common_counts, 0.75)),
                },
                "neighbors_within_1p5x_own_nearest": {
                    "median": float(np.median(relative_counts)),
                    "mean": float(np.mean(relative_counts)),
                    "q25": float(np.quantile(relative_counts, 0.25)),
                    "q75": float(np.quantile(relative_counts, 0.75)),
                },
            })
            if X_proj is not None:
                # Point-level M2 of comparably short neighbours.  Opposite
                # directions reinforce under exp(2 i theta), as desired for
                # a locally ribbon-like component; isotropic support cancels.
                delta = np.asarray(X_proj[edge_keys[edge_indices, 1]]
                                   - X_proj[edge_keys[edge_indices, 0]],
                                   dtype=np.float64)
                theta = np.arctan2(delta[:, 1], delta[:, 0])
                close_u = local_lengths <= 1.5 * nearest[local_u]
                close_v = local_lengths <= 1.5 * nearest[local_v]
                cos2 = np.cos(2.0 * theta)
                sin2 = np.sin(2.0 * theta)
                m2_real = np.zeros(len(local_points), dtype=np.float64)
                m2_imag = np.zeros(len(local_points), dtype=np.float64)
                m2_count = np.zeros(len(local_points), dtype=np.int64)
                np.add.at(m2_real, local_u[close_u], cos2[close_u])
                np.add.at(m2_imag, local_u[close_u], sin2[close_u])
                np.add.at(m2_count, local_u[close_u], 1)
                # theta + pi leaves the second moment unchanged.
                np.add.at(m2_real, local_v[close_v], cos2[close_v])
                np.add.at(m2_imag, local_v[close_v], sin2[close_v])
                np.add.at(m2_count, local_v[close_v], 1)
                valid_m2 = m2_count > 0
                point_m2 = np.hypot(m2_real[valid_m2], m2_imag[valid_m2]) / m2_count[valid_m2]
                components[-1]["short_neighbour_M2"] = {
                    "definition": (
                        "point-level |mean exp(2i theta)| over retained "
                        "neighbours within 1.5x that point's shortest "
                        "original-space retained edge; theta in projection"
                    ),
                    "median": float(np.median(point_m2)),
                    "mean": float(np.mean(point_m2)),
                    "q25": float(np.quantile(point_m2, 0.25)),
                    "q75": float(np.quantile(point_m2, 0.75)),
                }
                # A smooth alternative avoids reducing dense interiors to
                # one or two edges by a hard local-length cutoff.  The scale
                # remains point-local and original-space, while directions
                # remain in the triangulation plane.
                weight_u = np.exp(-np.square(local_lengths / np.maximum(
                    2.0 * nearest[local_u], 1e-12,
                )))
                weight_v = np.exp(-np.square(local_lengths / np.maximum(
                    2.0 * nearest[local_v], 1e-12,
                )))
                weighted_real = np.zeros(len(local_points), dtype=np.float64)
                weighted_imag = np.zeros(len(local_points), dtype=np.float64)
                weighted_mass = np.zeros(len(local_points), dtype=np.float64)
                np.add.at(weighted_real, local_u, weight_u * cos2)
                np.add.at(weighted_imag, local_u, weight_u * sin2)
                np.add.at(weighted_mass, local_u, weight_u)
                np.add.at(weighted_real, local_v, weight_v * cos2)
                np.add.at(weighted_imag, local_v, weight_v * sin2)
                np.add.at(weighted_mass, local_v, weight_v)
                weighted_m2 = np.hypot(weighted_real, weighted_imag) / np.maximum(
                    weighted_mass, 1e-12,
                )
                components[-1]["weighted_short_neighbour_M2"] = {
                    "definition": (
                        "point-level |sum w exp(2i theta)| / sum w over all "
                        "retained neighbours; w=exp(-(length/(2*own_nearest))^2), "
                        "length in original space and theta in projection"
                    ),
                    "median": float(np.median(weighted_m2)),
                    "mean": float(np.mean(weighted_m2)),
                    "q25": float(np.quantile(weighted_m2, 0.25)),
                    "q75": float(np.quantile(weighted_m2, 0.75)),
                }
    components.sort(key=lambda item: item["point_count"], reverse=True)
    return {
        "definition": "real pre-Gomory components; topology diagnostics only",
        "component_count": len(components),
        "weighted_m2_by_component": {
            str(item["component_label"]): item["weighted_short_neighbour_M2"]["median"]
            for item in components if "weighted_short_neighbour_M2" in item
        },
        "components": components,
        "largest_components": components[:5],
    }


def _draw_component_m2(ax, X_proj: np.ndarray, edge_keys: np.ndarray,
                       retained_mask: np.ndarray, topology: dict) -> None:
    """Draw the real pre-Gomory graph, coloured by component-level M2."""
    from matplotlib.collections import LineCollection

    labels, edge_bearing = _final_graph_components(
        edge_keys, retained_mask, len(X_proj),
    )
    values = {
        item["component_label"]: item["weighted_short_neighbour_M2"]["median"]
        for item in topology["largest_components"]
        if "weighted_short_neighbour_M2" in item
    }
    point_values = np.full(len(X_proj), np.nan, dtype=np.float64)
    for label, value in values.items():
        point_values[labels == label] = value
    segments = _segments(X_proj, edge_keys)
    ax.add_collection(LineCollection(
        segments[retained_mask], colors="#555555", linewidths=0.35,
        alpha=0.55, zorder=1,
    ))
    shown = edge_bearing & np.isfinite(point_values)
    scatter = ax.scatter(
        X_proj[shown, 0], X_proj[shown, 1], c=point_values[shown],
        cmap="viridis", vmin=0.0, vmax=1.0, s=7, linewidths=0, zorder=2,
    )
    for label, value in values.items():
        members = labels == label
        if not np.any(members):
            continue
        center = np.median(X_proj[members, :2], axis=0)
        ax.text(center[0], center[1], f"M2={value:.2f}", ha="center", va="center",
                fontsize=8, color="black", zorder=3,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7,
                      "pad": 1.5})
    ax.figure.colorbar(scatter, ax=ax, label="weighted component M2")
    ax.set_title("pre-Gomory components: weighted short-neighbour M2", fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    ax.autoscale()
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_component_loopiness(ax, X_proj: np.ndarray, edge_keys: np.ndarray,
                              retained_mask: np.ndarray, topology: dict) -> None:
    """Draw pre-Gomory components and annotate their cycle rank L=E-V+1."""
    from matplotlib.collections import LineCollection

    labels, edge_bearing = _final_graph_components(
        edge_keys, retained_mask, len(X_proj),
    )
    metrics = {
        item["component_label"]: item for item in topology["components"]
    }
    point_values = np.full(len(X_proj), np.nan, dtype=np.float64)
    for label, item in metrics.items():
        point_values[labels == label] = item["cycle_rank_per_point"]
    segments = _segments(X_proj, edge_keys)
    ax.add_collection(LineCollection(
        segments[retained_mask], colors="#555555", linewidths=0.35,
        alpha=0.6, zorder=1,
    ))
    shown = edge_bearing & np.isfinite(point_values)
    scatter = ax.scatter(
        X_proj[shown, 0], X_proj[shown, 1], c=point_values[shown],
        cmap="plasma", s=7, linewidths=0, zorder=2,
    )
    for label, item in metrics.items():
        members = labels == label
        center = np.median(X_proj[members, :2], axis=0)
        ax.text(
            center[0], center[1],
            f"L={item['cycle_rank']}\nE={item['edge_count']}, V={item['point_count']}",
            ha="center", va="center", fontsize=7, color="black", zorder=3,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75,
                  "pad": 1.3},
        )
    ax.figure.colorbar(scatter, ax=ax, label="cycle rank per point")
    ax.set_title("grown components: loopiness L = E − V + 1", fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    ax.autoscale()
    ax.set_xticks([])
    ax.set_yticks([])


def _add_component_area_perimeter_metrics(topology: dict, edge_keys: np.ndarray,
                                          retained_mask: np.ndarray,
                                          triangles: np.ndarray,
                                          X_proj: np.ndarray,
                                          X_original: np.ndarray | None = None) -> None:
    """Attach retained-triangle area / two perimeter variants per component.

    A triangle contributes only when all three of its Delaunay edges are
    retained and all its vertices belong to the same grown component.  The
    ``shape_score`` uses the primitive Gomory boundary: retained component
    edges with exactly one supported Delaunay triangle.  The prior
    exterior-hull perimeter remains in the output as an alternative
    diagnostic.
    """
    labels, edge_bearing = _final_graph_components(
        edge_keys, retained_mask, len(X_proj),
    )
    hull_edge_mask = _outer_hull_edge_mask(edge_keys, retained_mask, triangles)
    _, supported_triangle_counts = _component_supported_triangle_counts(
        edge_keys, triangles, labels, edge_bearing, retained_mask,
    )
    edge_index = {
        (int(u), int(v)): index for index, (u, v) in enumerate(edge_keys)
    }
    triangle_edges = np.array([
        [
            edge_index[tuple(sorted((int(a), int(b))))],
            edge_index[tuple(sorted((int(a), int(c))))],
            edge_index[tuple(sorted((int(b), int(c))))],
        ]
        for a, b, c in triangles
    ], dtype=np.int64)
    coords = np.asarray(X_proj, dtype=np.float64)
    triangle_points = coords[triangles]
    triangle_area = 0.5 * np.abs(
        (triangle_points[:, 1, 0] - triangle_points[:, 0, 0])
        * (triangle_points[:, 2, 1] - triangle_points[:, 0, 1])
        - (triangle_points[:, 2, 0] - triangle_points[:, 0, 0])
        * (triangle_points[:, 1, 1] - triangle_points[:, 0, 1])
    )
    projected_lengths = np.linalg.norm(
        coords[edge_keys[:, 1]] - coords[edge_keys[:, 0]], axis=1,
    )
    if X_original is None:
        original_triangle_area = triangle_area
        original_lengths = projected_lengths
    else:
        original_coords = np.asarray(X_original, dtype=np.float64)
        original_triangle_points = original_coords[triangles]
        # Heron's formula is unnecessarily fragile in high dimension.  The
        # Gram determinant gives the area of the three-point simplex in its
        # ambient original feature space.
        side_a = original_triangle_points[:, 1] - original_triangle_points[:, 0]
        side_b = original_triangle_points[:, 2] - original_triangle_points[:, 0]
        gram_det = (
            np.einsum("ij,ij->i", side_a, side_a)
            * np.einsum("ij,ij->i", side_b, side_b)
            - np.einsum("ij,ij->i", side_a, side_b) ** 2
        )
        original_triangle_area = 0.5 * np.sqrt(np.maximum(gram_det, 0.0))
        original_lengths = np.linalg.norm(
            original_coords[edge_keys[:, 1]] - original_coords[edge_keys[:, 0]],
            axis=1,
        )
    retained_degree = np.zeros(len(X_proj), dtype=np.int32)
    retained_edges = edge_keys[retained_mask]
    if len(retained_edges):
        np.add.at(retained_degree, retained_edges[:, 0], 1)
        np.add.at(retained_degree, retained_edges[:, 1], 1)
    for item in topology["components"]:
        component = item["component_label"]
        vertices = labels[triangles]
        internal_triangle = (
            np.all(vertices == component, axis=1)
            & np.all(retained_mask[triangle_edges], axis=1)
        )
        component_edges = (
            retained_mask
            & (labels[edge_keys[:, 0]] == component)
            & (labels[edge_keys[:, 1]] == component)
        )
        exterior_hull_edges = component_edges & hull_edge_mask
        gomory_primitive_boundary_edges = (
            component_edges
            & (supported_triangle_counts == 1)
            & (retained_degree[edge_keys[:, 0]] > 1)
            & (retained_degree[edge_keys[:, 1]] > 1)
        )
        area = float(np.sum(triangle_area[internal_triangle]))
        original_area = float(np.sum(original_triangle_area[internal_triangle]))
        exterior_perimeter = float(np.sum(projected_lengths[exterior_hull_edges]))
        gomory_perimeter = float(np.sum(
            projected_lengths[gomory_primitive_boundary_edges],
        ))
        original_gomory_perimeter = float(np.sum(
            original_lengths[gomory_primitive_boundary_edges],
        ))
        shape_score = (
            100.0 * np.sqrt(area) / gomory_perimeter
            if gomory_perimeter > 0.0 else None
        )
        original_shape_score = (
            100.0 * np.sqrt(original_area) / original_gomory_perimeter
            if original_gomory_perimeter > 0.0 else None
        )
        item["triangle_area_border_length"] = {
            "triangle_count": int(np.count_nonzero(internal_triangle)),
            "gomory_primitive_boundary_edge_count": int(np.count_nonzero(
                gomory_primitive_boundary_edges,
            )),
            "gomory_primitive_boundary_length": gomory_perimeter,
            "gomory_primitive_boundary_definition": (
                "retained component edge with exactly one supported "
                "Delaunay triangle and retained degree > 1 at both ends; "
                "no boundary expansion"
            ),
            "internal_triangle_area": area,
            "original_internal_triangle_area": original_area,
            "exterior_hull_border_edge_count": int(np.count_nonzero(
                exterior_hull_edges,
            )),
            "exterior_hull_border_length": exterior_perimeter,
            "area_over_gomory_boundary_length": (
                area / gomory_perimeter if gomory_perimeter > 0.0 else None
            ),
            "shape_score": shape_score,
            "original_gomory_primitive_boundary_length": original_gomory_perimeter,
            "original_shape_score": original_shape_score,
            "shape_score_definition": (
                "100 * sqrt(internal_triangle_area) / "
                "gomory_primitive_boundary_length"
            ),
            "geometry_space": "UMAP triangulation plane",
            "original_geometry_space": "standardized original feature space",
        }


def _draw_component_area_perimeter(ax, X_proj: np.ndarray, edge_keys: np.ndarray,
                                   retained_mask: np.ndarray, topology: dict) -> None:
    """Draw pre-Gomory components and annotate retained area/border length."""
    from matplotlib.collections import LineCollection

    labels, edge_bearing = _final_graph_components(
        edge_keys, retained_mask, len(X_proj),
    )
    metrics = {
        item["component_label"]: item["triangle_area_border_length"]
        for item in topology["components"]
    }
    point_values = np.full(len(X_proj), np.nan, dtype=np.float64)
    for label, item in metrics.items():
        score = item["shape_score"]
        if score is not None:
            point_values[labels == label] = score
    segments = _segments(X_proj, edge_keys)
    ax.add_collection(LineCollection(
        segments[retained_mask], colors="#555555", linewidths=0.35,
        alpha=0.6, zorder=1,
    ))
    shown = edge_bearing & np.isfinite(point_values)
    scatter = ax.scatter(
        X_proj[shown, 0], X_proj[shown, 1], c=point_values[shown],
        cmap="viridis", s=7, linewidths=0, zorder=2,
    )
    for label, item in metrics.items():
        members = labels == label
        score = item["shape_score"]
        perimeter = item["gomory_primitive_boundary_length"]
        if not np.any(members) or score is None:
            continue
        center = np.median(X_proj[members, :2], axis=0)
        ax.text(
            center[0], center[1], f"S={score:.2f}\nP={perimeter:.2f}",
            ha="center", va="center", fontsize=7, color="black", zorder=3,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75,
                  "pad": 1.3},
        )
    ax.figure.colorbar(
        scatter, ax=ax,
        label="shape score: 100 √area / one-support boundary length",
    )
    ax.set_title(
        "grown components: 100 √(retained triangle area) / one-support boundary",
        fontsize=11,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.autoscale()
    ax.set_xticks([])
    ax.set_yticks([])


def _save_component_shape_space_comparison(
    output_path: Path,
    X_umap: np.ndarray,
    X_original: np.ndarray,
    edge_keys: np.ndarray,
    retained_mask: np.ndarray,
    topology: dict,
) -> None:
    """Plot the same components in UMAP and PCA, labelled by both S scores."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from sklearn.decomposition import PCA

    labels, edge_bearing = _final_graph_components(
        edge_keys, retained_mask, len(X_umap),
    )
    metrics = {
        int(item["component_label"]): item["triangle_area_border_length"]
        for item in topology["components"]
    }
    X_pca = PCA(n_components=2, random_state=42).fit_transform(X_original)
    component_ids = sorted(metrics)
    cmap = plt.get_cmap("tab20" if len(component_ids) <= 20 else "turbo", max(len(component_ids), 1))
    colours = {component: cmap(index) for index, component in enumerate(component_ids)}
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=True)
    for ax, coords, title in (
        (axes[0], X_umap, "UMAP triangulation plane"),
        (axes[1], X_pca, "PCA of standardized original space"),
    ):
        segments = _segments(coords, edge_keys)
        ax.add_collection(LineCollection(
            segments[retained_mask], colors="#c7c7c7", linewidths=0.32,
            alpha=0.55, zorder=1,
        ))
        for component in component_ids:
            members = (labels == component) & edge_bearing
            if not np.any(members):
                continue
            edge_mask = (
                retained_mask
                & (labels[edge_keys[:, 0]] == component)
                & (labels[edge_keys[:, 1]] == component)
            )
            if np.any(edge_mask):
                ax.add_collection(LineCollection(
                    segments[edge_mask], colors=[colours[component]],
                    linewidths=0.65, alpha=0.9, zorder=2,
                ))
            ax.scatter(coords[members, 0], coords[members, 1], s=6,
                       color=[colours[component]], linewidths=0, zorder=3)
            metric = metrics[component]
            s_umap = metric.get("shape_score")
            s_original = metric.get("original_shape_score")
            p_umap = metric.get("gomory_primitive_boundary_length")
            p_original = metric.get("original_gomory_primitive_boundary_length")
            if s_umap is None or s_original is None:
                label = "S_U=—  P_U=—\nS_O=—  P_O=—"
            else:
                label = (
                    f"S_U={s_umap:.1f}  P_U={p_umap:.1f}\n"
                    f"S_O={s_original:.1f}  P_O={p_original:.1f}"
                )
            center = np.median(coords[members, :2], axis=0)
            ax.text(
                center[0], center[1], label, ha="center", va="center",
                fontsize=7, zorder=4,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78,
                      "pad": 1.1},
            )
        # Points outside components large enough for the topology report are
        # still informative background, but deliberately receive no S label.
        unlabelled = edge_bearing & ~np.isin(labels, component_ids)
        if np.any(unlabelled):
            ax.scatter(coords[unlabelled, 0], coords[unlabelled, 1], s=3,
                       color="#999999", alpha=0.45, linewidths=0, zorder=1)
        ax.set_title(title, fontsize=12)
        ax.set_aspect("equal", adjustable="box")
        ax.autoscale()
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "Pre-Gomory components: S_U = UMAP-plane score; S_O = original-space score\n"
        "S = 100 √(internal Delaunay-triangle area) / one-support boundary length",
        fontsize=13,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _draw_restoration_boundary(ax, X_proj: np.ndarray, edge_keys: np.ndarray,
                               retained_mask: np.ndarray,
                               boundary_mask: np.ndarray) -> None:
    """Show real component edges, separating core and one-layer boundary."""
    from matplotlib.collections import LineCollection

    segments = _segments(X_proj, edge_keys)
    internal = retained_mask & ~boundary_mask
    boundary = retained_mask & boundary_mask
    if np.any(internal):
        ax.add_collection(LineCollection(
            segments[internal], colors="#356aa0", linewidths=0.45,
            alpha=0.72, zorder=1, label="internal retained edges",
        ))
    if np.any(boundary):
        ax.add_collection(LineCollection(
            segments[boundary], colors="#e67e22", linewidths=0.62,
            alpha=0.9, zorder=2, label="boundary-layer edges",
        ))
    edge_points = np.zeros(len(X_proj), dtype=bool)
    if np.any(retained_mask):
        edge_points[edge_keys[retained_mask].ravel()] = True
    ax.scatter(
        X_proj[edge_points, 0], X_proj[edge_points, 1], s=4,
        color="#202020", alpha=0.6, linewidths=0, zorder=3,
    )
    ax.legend(loc="best", fontsize=8, frameon=True)
    ax.set_title(
        "real retained component graph: internal vs one-layer boundary",
        fontsize=11,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.autoscale()
    ax.set_xticks([])
    ax.set_yticks([])


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


def _component_supported_triangle_counts(
    edge_keys: np.ndarray,
    triangles: np.ndarray,
    component_labels: np.ndarray,
    component_points: np.ndarray,
    support_mask: np.ndarray,
 ) -> tuple[np.ndarray, np.ndarray]:
    """Return component-local edges and their supported-face counts.

    The classification uses full Delaunay faces, but a face supports one of
    its edges only when its *other two* sides are already real retained edges.
    Thus an absent restoration candidate can be tested hypothetically: adding
    it would close each such supported face.  Conversely, a visibly open
    triangle cannot make an edge look internal just because all three points
    share a component.

    """
    edge_keys = np.asarray(edge_keys, dtype=np.int64)
    triangles = np.asarray(triangles, dtype=np.int64)
    labels = np.asarray(component_labels, dtype=np.int64)
    eligible_points = np.asarray(component_points, dtype=bool)
    support = np.asarray(support_mask, dtype=bool)
    if support.shape != (len(edge_keys),):
        raise ValueError("support_mask must have one value per Delaunay edge")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("boundary restoration requires triangular Delaunay faces")

    u, v = edge_keys[:, 0], edge_keys[:, 1]
    same_component = (
        eligible_points[u] & eligible_points[v]
        & (labels[u] == labels[v])
    )
    edge_index = {
        (int(a), int(b)): index for index, (a, b) in enumerate(edge_keys)
    }
    triangle_edges = np.empty((len(triangles), 3), dtype=np.int64)
    for index, (a, b, c) in enumerate(triangles):
        triangle_edges[index] = (
            edge_index[tuple(sorted((int(a), int(b))))],
            edge_index[tuple(sorted((int(a), int(c))))],
            edge_index[tuple(sorted((int(b), int(c))))],
        )

    triangle_labels = labels[triangles]
    triangle_eligible = eligible_points[triangles]
    internal_triangles = (
        np.all(triangle_eligible, axis=1)
        & np.all(triangle_labels == triangle_labels[:, :1], axis=1)
    )
    triangle_counts = np.zeros(len(edge_keys), dtype=np.int16)
    if np.any(internal_triangles):
        local_edges = triangle_edges[internal_triangles]
        local_support = support[local_edges]
        # For edge 0, the triangle is available if edges 1 and 2 already
        # exist, and analogously for the other two sides.  This counts a
        # closed triangle for an existing edge and a hypothetically closed
        # triangle for the one candidate edge currently being considered.
        support_other_two = np.column_stack((
            local_support[:, 1] & local_support[:, 2],
            local_support[:, 0] & local_support[:, 2],
            local_support[:, 0] & local_support[:, 1],
        ))
        for position in range(3):
            np.add.at(
                triangle_counts, local_edges[:, position],
                support_other_two[:, position].astype(np.int16),
            )
    return same_component, triangle_counts


def _component_boundary_edge_mask(
    edge_keys: np.ndarray,
    triangles: np.ndarray,
    component_labels: np.ndarray,
    component_points: np.ndarray,
    support_mask: np.ndarray,
    expansion_hops: int = 1,
) -> np.ndarray:
    """Mark the expanded boundary of every eligible Delaunay component.

    Primitive boundary edges have zero or one supported Delaunay triangle.
    ``expansion_hops`` grows the primitive mask by incident-edge layers: one
    is the original rule and two includes its one-hop neighbouring edges too.
    """
    if expansion_hops < 0:
        raise ValueError("boundary expansion_hops must be non-negative")
    edge_keys = np.asarray(edge_keys, dtype=np.int64)
    labels = np.asarray(component_labels, dtype=np.int64)
    same_component, triangle_counts = _component_supported_triangle_counts(
        edge_keys, triangles, labels, component_points, support_mask,
    )
    u, v = edge_keys[:, 0], edge_keys[:, 1]
    primitive_boundary = same_component & (triangle_counts <= 1)
    boundary = primitive_boundary.copy()
    for _ in range(expansion_hops):
        boundary_points = np.zeros(len(labels), dtype=bool)
        boundary_points[u[boundary]] = True
        boundary_points[v[boundary]] = True
        boundary |= same_component & (boundary_points[u] | boundary_points[v])
    return boundary


def _gomory_hu_joint_cut_mask(
    edge_keys: np.ndarray,
    retained_mask: np.ndarray,
    *,
    cut_size: int,
    min_component_points: int,
    min_partition_fraction: float = 0.0,
    removable_mask: np.ndarray | None = None,
    hull_ratio_skip_threshold: float = 0.0,
    triangles: np.ndarray | None = None,
    edge_sizes: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Return eligible fundamental Gomory--Hu cuts with size <= cut_size.

    Both sides must contain ``min_component_points`` graph points and, when
    enabled, at least ``min_partition_fraction`` of the original component.
    That excludes cuts which only detach outliers, small nested islands, or
    a very small tail from a large component.  When
    ``removable_mask`` is supplied, only those retained edges may be cut;
    every other retained edge has capacity ``cut_size + 1`` and is therefore
    excluded from every eligible cut.
    """
    if min_component_points < 1:
        raise ValueError("min_component_points must be positive")
    if not 0.0 <= min_partition_fraction <= 0.5:
        raise ValueError("min_partition_fraction must be between 0 and 0.5")
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
            "min_partition_fraction": float(min_partition_fraction),
            "hull_ratio_skip_threshold": float(hull_ratio_skip_threshold),
            "component_count": 0, "skipped_too_small_component_count": 0,
            "skipped_hull_ratio_component_count": 0,
            "eligible_tree_cut_count": 0,
            "rejected_small_side_tree_cut_count": 0,
            "rejected_small_fraction_tree_cut_count": 0, "pruned_edge_count": 0,
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
                "rejected_small_side_tree_cut_count": 0,
                "rejected_small_fraction_tree_cut_count": 0,
                "pruned_edge_count": 0,
            })
            continue
        if hull_ratio is not None and hull_ratio >= hull_ratio_skip_threshold:
            component_summaries.append({
                "edge_count": int(len(edge_indices)), "point_count": int(len(nodes)),
                "hull_to_non_hull_edge_ratio": hull_ratio,
                "skipped_too_small": False, "skipped_hull_ratio": True,
                "eligible_tree_cut_count": 0,
                "rejected_small_side_tree_cut_count": 0,
                "rejected_small_fraction_tree_cut_count": 0,
                "pruned_edge_count": 0,
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
        rejected_small_fraction_cuts = 0
        before = int(np.count_nonzero(pruned_mask[edge_indices]))
        for left, right, data in list(tree.edges(data=True)):
            if int(round(data["weight"])) > cut_size:
                continue
            # A GH-tree edge weight is a valid min-cut value, but removing
            # that edge from the tree does not in general recover the actual
            # min-cut partition in the original graph.  Compute the latter
            # explicitly before marking any real Delaunay edge for removal.
            cut_value, partition = nx.minimum_cut(
                graph, left, right, capacity="capacity",
            )
            if int(round(cut_value)) > cut_size:
                continue
            left_side, _ = partition
            smaller_side_size = min(len(left_side), len(nodes) - len(left_side))
            if smaller_side_size < min_component_points:
                rejected_small_side_cuts += 1
                continue
            if smaller_side_size < min_partition_fraction * len(nodes):
                rejected_small_fraction_cuts += 1
                continue
            on_left_u = np.isin(edge_keys[edge_indices, 0], list(left_side))
            on_left_v = np.isin(edge_keys[edge_indices, 1], list(left_side))
            cut_edges = edge_indices[on_left_u != on_left_v]
            if removable_mask is not None:
                cut_edges = cut_edges[removable_mask[cut_edges]]
            if not len(cut_edges):
                continue
            # Only accept the *whole* set when deleting it truly separates
            # the completed component.  This prevents an isolated pink edge
            # from being displayed/pruned when another real path remains.
            verification = graph.copy()
            verification.remove_edges_from(
                (int(edge_keys[index, 0]), int(edge_keys[index, 1]))
                for index in cut_edges
            )
            if nx.has_path(verification, left, right):
                continue
            pruned_mask[cut_edges] = True
            eligible_cuts += 1
        component_summaries.append({
            "edge_count": int(len(edge_indices)), "point_count": int(len(nodes)),
            "hull_to_non_hull_edge_ratio": hull_ratio,
            "skipped_too_small": False, "skipped_hull_ratio": False,
            "eligible_tree_cut_count": eligible_cuts,
            "rejected_small_side_tree_cut_count": rejected_small_side_cuts,
            "rejected_small_fraction_tree_cut_count": rejected_small_fraction_cuts,
            "pruned_edge_count": int(np.count_nonzero(pruned_mask[edge_indices])) - before,
        })

    component_summaries.sort(key=lambda item: item["edge_count"], reverse=True)
    return pruned_mask, {
        "enabled": True, "cut_size": int(cut_size),
        "min_component_points": int(min_component_points),
        "min_partition_fraction": float(min_partition_fraction),
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
        "rejected_small_fraction_tree_cut_count": int(sum(
            item["rejected_small_fraction_tree_cut_count"] for item in component_summaries
        )),
        "pruned_edge_count": int(np.count_nonzero(pruned_mask)),
        "largest_components": component_summaries[:5],
    }


def _candidate_vertex_cut_mask(
    edge_keys: np.ndarray,
    retained_mask: np.ndarray,
    *,
    max_cut_points: int,
    min_component_points: int,
    min_partition_fraction: float,
    component_eligible_points: np.ndarray | None = None,
    candidate_count: int = 10,
    centrality_samples: int = 16,
) -> tuple[np.ndarray, dict]:
    """Find one small, balanced vertex separator per final component.

    This is a deliberately cheap alternative to the all-edge Gomory--Hu
    stage.  It ranks a bounded candidate pool using sampled betweenness plus
    exact articulation points, then tests every subset of one through
    ``max_cut_points`` candidates by plain graph connectivity.  It is a
    heuristic vertex-cut search, not an exhaustive min-node-cut algorithm.
    """
    if max_cut_points <= 0:
        return np.zeros(len(edge_keys), dtype=bool), {
            "enabled": False, "mode": "points",
            "max_cut_points": int(max_cut_points), "pruned_edge_count": 0,
        }
    if min_component_points < 1:
        raise ValueError("min_component_points must be positive")
    if not 0.0 <= min_partition_fraction <= 0.5:
        raise ValueError("min_partition_fraction must be between 0 and 0.5")

    import networkx as nx

    edge_keys = np.asarray(edge_keys, dtype=np.int64)
    labels, edge_bearing_points = _final_graph_components(
        edge_keys, retained_mask, int(edge_keys.max()) + 1,
    )
    if component_eligible_points is None:
        component_eligible_points = np.ones(len(labels), dtype=bool)
    else:
        component_eligible_points = np.asarray(component_eligible_points, dtype=bool)
    pruned_mask = np.zeros(len(edge_keys), dtype=bool)
    summaries: list[dict] = []

    for component in np.unique(labels[edge_bearing_points]):
        edge_indices = np.flatnonzero(
            retained_mask
            & (labels[edge_keys[:, 0]] == component)
            & (labels[edge_keys[:, 1]] == component)
        )
        nodes = np.unique(edge_keys[edge_indices].ravel())
        n_nodes = len(nodes)
        minimum_side = max(
            int(min_component_points),
            int(np.ceil(min_partition_fraction * n_nodes)),
        )
        summary = {
            "component_label": int(component), "point_count": int(n_nodes),
            "edge_count": int(len(edge_indices)), "minimum_side_points": minimum_side,
            "eligible": bool(np.all(component_eligible_points[nodes])),
            "candidate_point_count": 0, "tested_subset_count": 0,
            "accepted_cut_points": [], "pruned_edge_count": 0,
        }
        if (not summary["eligible"] or n_nodes < 2 * minimum_side):
            summaries.append(summary)
            continue
        graph = nx.Graph()
        graph.add_edges_from((int(edge_keys[i, 0]), int(edge_keys[i, 1])) for i in edge_indices)

        articulation = list(nx.articulation_points(graph))
        centrality = nx.betweenness_centrality(
            graph, k=min(int(centrality_samples), n_nodes), seed=42,
        )
        ranked = sorted(centrality, key=lambda node: (-centrality[node], int(node)))
        candidates: list[int] = []
        for node in [*articulation, *ranked]:
            if node not in candidates:
                candidates.append(int(node))
            if len(candidates) >= int(candidate_count):
                break
        summary["candidate_point_count"] = len(candidates)
        best: tuple[tuple[int, int, tuple[int, ...]], tuple[int, ...]] | None = None
        for size in range(1, min(int(max_cut_points), len(candidates)) + 1):
            for cut_points in combinations(candidates, size):
                summary["tested_subset_count"] += 1
                remainder = graph.copy()
                remainder.remove_nodes_from(cut_points)
                sizes = sorted((len(part) for part in nx.connected_components(remainder)), reverse=True)
                if len(sizes) < 2 or sizes[1] < minimum_side:
                    continue
                # Prefer fewer removed points, then the most balanced split.
                key = (len(cut_points), -sizes[1], tuple(sorted(map(int, cut_points))))
                if best is None or key < best[0]:
                    best = (key, cut_points)
        if best is not None:
            cut_points = np.asarray(best[1], dtype=np.int64)
            incident = np.isin(edge_keys[edge_indices, 0], cut_points) | np.isin(
                edge_keys[edge_indices, 1], cut_points,
            )
            removed_edges = edge_indices[incident]
            pruned_mask[removed_edges] = True
            summary["accepted_cut_points"] = sorted(map(int, best[1]))
            summary["pruned_edge_count"] = int(len(removed_edges))
        summaries.append(summary)

    summaries.sort(key=lambda item: item["edge_count"], reverse=True)
    return pruned_mask, {
        "enabled": True, "mode": "points",
        "max_cut_points": int(max_cut_points),
        "min_component_points": int(min_component_points),
        "min_partition_fraction": float(min_partition_fraction),
        "candidate_count": int(candidate_count),
        "centrality_samples": int(centrality_samples),
        "component_count": len(summaries),
        "cut_component_count": int(sum(bool(item["accepted_cut_points"]) for item in summaries)),
        "pruned_edge_count": int(np.count_nonzero(pruned_mask)),
        "largest_components": summaries[:5],
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
            edge_mask = (
                retained_mask
                & (display_labels[edge_keys[:, 0]] == component)
                & (display_labels[edge_keys[:, 1]] == component)
            )
            colour = cmap(colour_index)
            if np.any(edge_mask):
                ax.add_collection(LineCollection(
                    segments[edge_mask], colors=[colour], linewidths=1.05,
                    alpha=0.93, zorder=3,
                ))
            ax.scatter(X_proj[point_mask, 0], X_proj[point_mask, 1], s=5.2,
                       color=[colour], alpha=0.9, linewidths=0, zorder=4)
    # ``assigned_labels`` is the prediction used for the ARI.  In particular,
    # points still labelled -1 are genuine final outliers, not absent data.
    # They must remain visible in the diagnostic even though they own no
    # retained component edge.
    outlier_mask = ~displayed
    if np.any(outlier_mask):
        ax.scatter(
            X_proj[outlier_mask, 0], X_proj[outlier_mask, 1],
            s=7.0, color="#4d4d4d", alpha=0.78, linewidths=0,
            zorder=6, label="final outlier (-1)",
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
                       projected_hard_growth_limit: float,
                       restore_hard_limit: float | None = None,
                       restoration_mode: str = "real") -> dict:
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
    # Degenerate/collinear projections can leave an isolated vertex outside
    # every Delaunay simplex.  It has no local incident-edge distribution, so
    # fall back to the global q25 rather than attempting a percentile of an
    # empty array.  Such a point cannot create a boundary-growth edge anyway.
    global_q25 = float(np.percentile(lengths, 25.0)) if len(lengths) else 0.0
    outer_q25 = np.array([
        np.percentile(lengths[indices], 25.0) if indices else global_q25
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
    # An explicitly bounded restoration can enrich a seed before growth.  It
    # remains strictly intra-seed and is therefore a legitimate part of the
    # initial component reference.  An unbounded restoration is deliberately
    # kept as the existing post-growth connectivity step instead.
    initially_large_points = initially_large[seed_labels]
    pre_growth_restored = (
        _bounded_intra_component_restore_mask(
            edge_keys, lengths, initial_mask, seed_labels,
            initially_large_points, restore_hard_limit,
        )
        if restore_hard_limit is not None
        else np.zeros(len(edge_keys), dtype=bool)
    )
    accepted = np.zeros(len(edge_keys), dtype=bool)
    # ``reference_restored`` enriches the scale used by the next growth round.
    # In ``virtual`` mode those edges are deliberately absent from the real
    # component graph, so they cannot make a temporary growth merge permanent.
    reference_restored = pre_growth_restored.copy()
    restored_during_growth = pre_growth_restored.copy()
    accepted_step = np.full(len(edge_keys), -1, dtype=np.int64)
    round_index = 0
    accepted_edges_per_round: list[int] = []

    while True:
        qualifying_edges: set[int] = set()
        qualifying_restorations: set[int] = set()
        # Roots can change only after the full round has been evaluated.
        # Materialize them once, rather than repeatedly traversing union-find
        # parents for every component/edge combination below.
        point_roots = np.fromiter(
            (find(seed_labels[point]) for point in range(n_points)),
            dtype=np.int64,
            count=n_points,
        )
        internal_reference = initial_mask | accepted | reference_restored
        reference_u = point_roots[edge_keys[:, 0]]
        reference_v = point_roots[edge_keys[:, 1]]
        for root in range(n_components):
            if find(root) != root or not active[root] or not members[root]:
                continue

            # The reference scale must come only from the graph that the
            # component has already accepted.  In particular, do not use all
            # Delaunay edges incident to its vertices: long, rejected boundary
            # edges would otherwise inflate the median/IQR before they are
            # tested and can make separate seed components merge immediately.
            # At round zero this is precisely the seed graph; thereafter it
            # also includes edges accepted in prior complete rounds.
            reference_indices = np.flatnonzero(
                internal_reference
                & (reference_u == root)
                & (reference_v == root)
            )
            if not len(reference_indices):
                # Active components are seeded by edges, so this should only
                # be reachable for a malformed input.  Skipping is safer than
                # falling back to boundary edges and recreating the problem.
                continue
            q25, median, q75 = np.percentile(
                lengths[reference_indices], [25.0, 50.0, 75.0],
            )
            right_spread = max(float(q75 - median), 1e-12)
            _, projected_median, projected_q75 = np.percentile(
                projected_lengths[reference_indices], [25.0, 50.0, 75.0],
            )
            projected_right_spread = max(
                float(projected_q75 - projected_median), 1e-12,
            )

            # Restoration is an internal operation: endpoints already belong
            # to this component.  With an explicit bound, accept only edges
            # compatible with the *current accepted* scale.  They are not
            # allowed to affect this round's boundary decisions; both these
            # edges and newly accepted growth edges enter the reference only
            # at the next round.
            if restore_hard_limit is not None and restoration_mode != "seed_only":
                restore_threshold = float(
                    median + restore_hard_limit * max(float(q75 - q25), 1e-12)
                )
                restorable = np.flatnonzero(
                    ~internal_reference
                    & (reference_u == root)
                    & (reference_v == root)
                    & (lengths <= restore_threshold)
                )
                qualifying_restorations.update(map(int, restorable))

            incident = set()
            for node in members[root]:
                incident.update(adjacency[node])
            incident_indices = np.fromiter(incident, dtype=np.int64)
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

        if not qualifying_edges and not qualifying_restorations:
            break
        # Deliberately defer every merge until all current boundary edges have
        # been evaluated.  Thus no edge in a round sees statistics changed by
        # another accepted edge from that same round.
        for edge_index in sorted(qualifying_edges):
            u, v = edge_keys[edge_index]
            merge(seed_labels[u], seed_labels[v])
            accepted[edge_index] = True
            accepted_step[edge_index] = round_index
        restored_indices = list(qualifying_restorations)
        reference_restored[restored_indices] = True
        if restoration_mode == "real":
            restored_during_growth[restored_indices] = True
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
        "pre_growth_restored_mask": pre_growth_restored,
        "restored_during_growth_mask": restored_during_growth,
        "virtual_restored_mask": reference_restored & ~restored_during_growth,
        "growth_added_point_mask": growth_added_points,
        "final_component": final_component,
        "final_active_point": final_active_point,
        "growth_round_count": int(round_index),
        "accepted_edges_per_round": np.asarray(accepted_edges_per_round, dtype=np.int64),
    }


def _no_component_growth(edge_keys: np.ndarray, seed_labels: np.ndarray,
                         seed_edge_counts: np.ndarray, min_edges: int) -> dict:
    """Return the exact post-seed state with boundary growth disabled."""
    n_edges = len(edge_keys)
    initially_large = np.asarray(seed_edge_counts > int(min_edges), dtype=bool)
    return {
        "initially_large_seed_components": initially_large,
        "first_seen_mask": np.zeros(n_edges, dtype=bool),
        "first_seen_step": np.full(n_edges, -1, dtype=np.int64),
        "first_seen_component": np.full(n_edges, -1, dtype=np.int64),
        "first_seen_panel2_ratio_raw": np.full(n_edges, np.nan, dtype=np.float64),
        "first_seen_panel2_display": np.full(n_edges, np.nan, dtype=np.float64),
        "first_seen_projected_ratio_raw": np.full(n_edges, np.nan, dtype=np.float64),
        "first_seen_panel3_outer_recall10": np.full(n_edges, np.nan, dtype=np.float64),
        "first_seen_panel4_ratio_raw": np.full(n_edges, np.nan, dtype=np.float64),
        "first_seen_panel4_display": np.full(n_edges, np.nan, dtype=np.float64),
        "accepted_growth_mask": np.zeros(n_edges, dtype=bool),
        "accepted_growth_step": np.full(n_edges, -1, dtype=np.int64),
        "pre_growth_restored_mask": np.zeros(n_edges, dtype=bool),
        "restored_during_growth_mask": np.zeros(n_edges, dtype=bool),
        "virtual_restored_mask": np.zeros(n_edges, dtype=bool),
        "growth_added_point_mask": np.zeros(len(seed_labels), dtype=bool),
        "final_component": np.asarray(seed_labels, dtype=np.int64).copy(),
        "final_active_point": initially_large[seed_labels],
        "growth_round_count": 0,
        "accepted_edges_per_round": np.empty(0, dtype=np.int64),
    }


def _bounded_intra_component_restore_mask(
    edge_keys: np.ndarray,
    lengths: np.ndarray,
    initial_mask: np.ndarray,
    completed_labels: np.ndarray,
    completed_points: np.ndarray,
    restore_hard_limit: float | None,
) -> np.ndarray:
    """Select original intra-component edges eligible for restoration.

    The scale is deliberately derived only from original seed edges.  Using
    accepted growth or already-restored edges here would let a long boundary
    edge inflate the very distribution meant to reject it.
    """
    u, v = edge_keys[:, 0], edge_keys[:, 1]
    same_completed = (
        completed_points[u] & completed_points[v]
        & (completed_labels[u] == completed_labels[v])
    )
    if restore_hard_limit is None:
        return same_completed

    restored = np.zeros(len(edge_keys), dtype=bool)
    for component in np.unique(completed_labels[completed_points]):
        in_component = same_completed & (completed_labels[u] == component)
        reference = initial_mask & in_component
        if not np.any(reference):
            continue
        q25, median, q75 = np.percentile(lengths[reference], [25.0, 50.0, 75.0])
        threshold = float(median + restore_hard_limit * max(q75 - q25, 1e-12))
        restored[in_component] = lengths[in_component] <= threshold
    return restored


def main() -> None:
    run_started = time.perf_counter()
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
        "--component-m2-plot", action="store_true",
        help="Also write a pre-Gomory component-M2 diagnostic next to --out; works with --metrics-only.",
    )
    parser.add_argument(
        "--component-loopiness-plot", action="store_true",
        help="Also write pre-Gomory component cycle-rank labels next to --out; works with --metrics-only.",
    )
    parser.add_argument(
        "--component-area-perimeter-plot", action="store_true",
        help="Also write retained-triangle-area/exterior-border diagnostics next to --out.",
    )
    parser.add_argument(
        "--component-shape-space-comparison-plot", action="store_true",
        help=(
            "Also write matching UMAP/PCA component views labelled with "
            "UMAP-plane and original-space S/P geometry diagnostics."
        ),
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
    parser.add_argument("--seed-hard-limit-2d", type=float,
                        help="Native-2-D override for --seed-hard-limit.")
    parser.add_argument(
        "--seed-baseline", choices=("median", "first_mode"), default="median",
        help=(
            "Seed-threshold regime. first_mode is experimental: it uses the "
            "first significant log-length valley, with automatic fallback to "
            "the median-based seed mask when the separation is weak."
        ),
    )
    parser.add_argument(
        "--first-mode-no-fallback", action="store_true",
        help="When --seed-baseline=first_mode fails its mode test, use no seeds instead of the median mask.",
    )
    parser.add_argument("--hard-growth-limit", type=float, default=0.05)
    parser.add_argument(
        "--no-component-growth", action="store_true",
        help="Disable every post-seed boundary-growth edge; retain final pruning and outlier reassignment.",
    )
    parser.add_argument(
        "--projected-hard-growth-limit", default="auto",
        help="Projected-space guard; 'auto' uses two times --hard-growth-limit.",
    )
    parser.add_argument("--hard-growth-limit-2d", type=float,
                        help="Native-2-D override for --hard-growth-limit.")
    parser.add_argument("--projected-hard-growth-limit-2d", type=float,
                        help="Native-2-D override for --projected-hard-growth-limit.")
    parser.add_argument(
        "--redundancy-pruning", action=argparse.BooleanOptionalAction,
        # Optional legacy diagnostic; current-stage component growth keeps it
        # disabled unless explicitly requested.
        default=False,
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
        "--gomory-hu-cut-size-2d", type=int,
        help="Optional native-2-D override for the final Gomory cut size.",
    )
    parser.add_argument(
        "--gomory-pruning-mode", choices=("edges", "points"), default="edges",
        help=(
            "Use ordinary Gomory--Hu edge cuts, or the experimental fast "
            "balanced vertex-separator alternative."
        ),
    )
    parser.add_argument(
        "--gomory-point-cut-size", type=int, default=4,
        help="Maximum number of separator points in --gomory-pruning-mode points.",
    )
    parser.add_argument(
        "--gomory-hu-min-component-points", type=int, default=10,
        help="Require this many points on both sides of a Gomory--Hu cut.",
    )
    parser.add_argument(
        "--gomory-hu-min-partition-fraction", type=float, default=0.20,
        help=(
            "Require each side of an accepted Gomory--Hu cut to contain at "
            "least this fraction of its original component; 0 disables this gate."
        ),
    )
    parser.add_argument(
        "--gomory-hu-hull-ratio-skip-threshold", type=float, default=0.0,
        help=(
            "Skip Gomory--Hu pruning for a component when its weighted "
            "outer-hull/non-hull ratio reaches this value; 0 disables the gate."
        ),
    )
    parser.add_argument(
        "--gomory-hu-m2-max", type=float,
        help=(
            "Run contracted Gomory--Hu only in pre-Gomory components whose "
            "weighted short-neighbour M2 is below this value; omit to "
            "disable the component-M2 gate."
        ),
    )
    parser.add_argument(
        "--gomory-hu-shape-score-min", type=float,
        help=(
            "Run Gomory--Hu only in components with restored-reference "
            "shape score 100*sqrt(retained triangle area)/border length at "
            "least this value; omit to disable this gate."
        ),
    )
    parser.add_argument(
        "--gomory-hu-min-triangle-edge-ratio", type=float,
        help=(
            "Run Gomory--Hu only when a component has at least this ratio "
            "of fully retained internal Delaunay triangles to retained "
            "edges. Omit to disable this triangular-support gate."
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
        "--gomory-hu-boundary-edges-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Restrict final Gomory--Hu candidate cuts to the post-growth "
            "one-layer boundary classification. This has no effect on seed "
            "construction, restoration, or growth."
        ),
    )
    parser.add_argument(
        "--boundary-edge-expansion-hops", type=int, default=1,
        help=(
            "Number of incident-edge layers grown from primitive boundary "
            "edges for the final GH boundary mask; 1 is the original rule."
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
        "--seed-only-restoration", action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Restore bounded Delaunay edges only within the original seed "
            "components before growth. Disable restoration during and after "
            "growth, so a temporary growth merge cannot create protected "
            "restored bridges."
        ),
    )
    parser.add_argument(
        "--growth-restoration-mode", choices=("real", "virtual", "seed_only"),
        default="real",
        help=(
            "How bounded restoration behaves after seed construction: real "
            "adds edges to the component graph, virtual uses them only for "
            "the next growth scale, seed_only disables post-seed restoration."
        ),
    )
    parser.add_argument(
        "--interior-real-restoration", "--post-growth-boundary-classification",
        dest="interior_real_restoration", action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "After all real restoration and component growth, classify the "
            "completed graph's boundary layer for diagnostics: a primitive "
            "boundary edge has fewer than two supported Delaunay triangles, "
            "then all incident edges are included. This never affects growth "
            "or restoration."
        ),
    )
    parser.add_argument(
        "--restore-hard-limit", type=float,
        help=(
            "Optional seed-core restoration gate: restore only original "
            "edges <= median + limit*IQR inside each completed component."
        ),
    )
    parser.add_argument(
        "--restoration-boundary-plot", action="store_true",
        help="Write a sidecar plot colouring retained internal and boundary-layer edges.",
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
    parser.add_argument("--outlier-cost-gate-reach-2d", type=float,
                        help="Native-2-D override for --outlier-cost-gate-reach.")
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
    parser.add_argument(
        "--growth-seed-min-edges", type=int, default=10,
        help=(
            "Minimum seed-component edge count eligible for component growth. "
            "The separate --component-growth-min-edges still controls final "
            "cluster eligibility."
        ),
    )
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
    if (args.gomory_hu_cut_size_2d is not None
            and args.gomory_hu_cut_size_2d < 0):
        raise ValueError("--gomory-hu-cut-size-2d must be non-negative")
    if args.gomory_point_cut_size < 0:
        raise ValueError("--gomory-point-cut-size must be non-negative")
    if args.boundary_edge_expansion_hops < 0:
        raise ValueError("--boundary-edge-expansion-hops must be non-negative")
    if args.gomory_hu_min_component_points < 1:
        raise ValueError("--gomory-hu-min-component-points must be positive")
    if not 0.0 <= args.gomory_hu_min_partition_fraction <= 0.5:
        raise ValueError("--gomory-hu-min-partition-fraction must be in [0, 0.5]")
    if args.gomory_hu_hull_ratio_skip_threshold < 0.0:
        raise ValueError("--gomory-hu-hull-ratio-skip-threshold must be non-negative")
    if args.gomory_hu_m2_max is not None and not 0.0 <= args.gomory_hu_m2_max <= 1.0:
        raise ValueError("--gomory-hu-m2-max must be in [0, 1]")
    if args.gomory_hu_shape_score_min is not None and args.gomory_hu_shape_score_min < 0.0:
        raise ValueError("--gomory-hu-shape-score-min must be non-negative")
    if (args.gomory_hu_min_triangle_edge_ratio is not None
            and args.gomory_hu_min_triangle_edge_ratio < 0.0):
        raise ValueError("--gomory-hu-min-triangle-edge-ratio must be non-negative")
    if args.restore_hard_limit is not None and args.restore_hard_limit < 0.0:
        raise ValueError("--restore-hard-limit must be non-negative")
    if args.seed_only_restoration:
        args.growth_restoration_mode = "seed_only"
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
    loaded = np.load(args.data, allow_pickle=False)
    raw_X = np.asarray(loaded["X"], dtype=np.float64)
    native_2d = raw_X.shape[1] == 2
    if native_2d:
        if args.gomory_hu_cut_size_2d is not None:
            args.gomory_hu_cut_size = args.gomory_hu_cut_size_2d
        if args.seed_hard_limit_2d is not None:
            args.seed_hard_limit = args.seed_hard_limit_2d
        if args.hard_growth_limit_2d is not None:
            args.hard_growth_limit = args.hard_growth_limit_2d
        if args.projected_hard_growth_limit_2d is not None:
            args.projected_hard_growth_limit = args.projected_hard_growth_limit_2d
        if args.outlier_cost_gate_reach_2d is not None:
            args.outlier_cost_gate_reach = args.outlier_cost_gate_reach_2d
    if args.projected_hard_growth_limit == "auto":
        args.projected_hard_growth_limit = 2.0 * args.hard_growth_limit
    else:
        args.projected_hard_growth_limit = float(args.projected_hard_growth_limit)
    # Native 2-D has no topology projection: X_proj is the standardized input
    # itself. A second projected-space gate is redundant and must not cap the
    # original-space growth gate.
    if native_2d:
        args.projected_hard_growth_limit = float("inf")
    X = StandardScaler().fit_transform(raw_X)
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
        projected_hard_limit=(False if native_2d else args.projected_hard_limit),
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
    seed_baseline_diagnostics: dict[str, object] = {
        "mode": "median",
        "seed_edge_count": int(np.count_nonzero(initial_mask)),
    }
    if args.seed_baseline == "first_mode":
        initial_mask, seed_baseline_diagnostics = _first_mode_seed_mask(
            lengths, initial_mask,
            fallback_on_failure=not args.first_mode_no_fallback,
        )
    seed_labels, seed_edge_counts, large_seed = _seed_components(
        edge_keys, initial_mask, len(X), args.growth_seed_min_edges,
    )
    recall10 = _projected_recall10(X, np.asarray(state["X_proj"], dtype=np.float64))
    growth = (
        _no_component_growth(
            edge_keys, seed_labels, seed_edge_counts,
            args.growth_seed_min_edges,
        )
        if args.no_component_growth else _sequential_growth(
            edge_keys, lengths, projected_lengths, initial_mask, seed_labels,
            seed_edge_counts, args.growth_seed_min_edges, recall10,
            args.hard_growth_limit, args.projected_hard_growth_limit,
            args.restore_hard_limit, args.growth_restoration_mode,
        )
    )
    large_points = large_seed[seed_labels]
    seed_mask = initial_mask & large_points[edge_keys[:, 0]] & large_points[edge_keys[:, 1]]
    # Restore intra-component Delaunay edges for final connectivity and the
    # optional redundancy analysis.  GH nevertheless protects seed-only
    # edges, so it can remove only growth-related links.
    pre_redundancy_mask = (
        initial_mask
        | growth["restored_during_growth_mask"]
        | growth["accepted_growth_mask"]
    )
    growth_component_labels, growth_component_points = _final_graph_components(
        edge_keys, pre_redundancy_mask, len(X),
    )
    same_completed_component = (
        np.zeros(len(edge_keys), dtype=bool)
        if args.growth_restoration_mode != "real" else _bounded_intra_component_restore_mask(
            edge_keys, lengths, initial_mask, growth_component_labels,
            growth_component_points, args.restore_hard_limit,
        )
    )
    if args.restore_intra_component_edges and args.growth_restoration_mode == "real":
        bridge_base_mask = pre_redundancy_mask | same_completed_component
    else:
        bridge_base_mask = pre_redundancy_mask.copy()
    restored_intra_component_mask = bridge_base_mask & ~pre_redundancy_mask
    restoration_boundary_mask = (
        _component_boundary_edge_mask(
            edge_keys, np.asarray(state["triangles"], dtype=np.int64),
            growth_component_labels, growth_component_points,
            bridge_base_mask,
            args.boundary_edge_expansion_hops,
        )
        if ((args.interior_real_restoration or args.gomory_hu_boundary_edges_only)
            and args.growth_restoration_mode == "real")
        else np.zeros(len(edge_keys), dtype=bool)
    )
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
    if args.gomory_hu_boundary_edges_only:
        if gomory_removable_mask is None:
            gomory_removable_mask = gomory_graph_mask.copy()
        gomory_removable_mask &= restoration_boundary_mask
    pre_gomory_topology = _component_topology_metrics(
        edge_keys, bridge_base_mask, len(X),
        minimum_edges=args.component_growth_min_edges,
        edge_lengths=lengths,
        X_proj=np.asarray(state["X_proj"], dtype=np.float64),
    )
    # Geometry is evaluated on the restored reference graph in both modes.
    # In virtual mode these edges inform only the shape measurement: they do
    # not become real edges, nor candidates for Gomory--Hu cuts.
    shape_reference_mask = bridge_base_mask | growth["virtual_restored_mask"]
    shape_reference_labels, _ = _final_graph_components(
        edge_keys, shape_reference_mask, len(X),
    )
    shape_reference_topology = None
    if (args.component_area_perimeter_plot
            or args.component_shape_space_comparison_plot
            or args.gomory_hu_shape_score_min is not None
            or args.gomory_hu_min_triangle_edge_ratio is not None):
        shape_reference_topology = _component_topology_metrics(
            edge_keys, shape_reference_mask, len(X),
            minimum_edges=args.component_growth_min_edges,
            edge_lengths=lengths,
            X_proj=np.asarray(state["X_proj"], dtype=np.float64),
        )
        _add_component_area_perimeter_metrics(
            shape_reference_topology, edge_keys, shape_reference_mask,
            np.asarray(state["triangles"], dtype=np.int64),
            np.asarray(state["X_proj"], dtype=np.float64),
            X_original=X,
        )
    point_m2_eligible = np.ones(len(X), dtype=bool)
    point_shape_eligible = np.ones(len(X), dtype=bool)
    point_triangle_edge_eligible = np.ones(len(X), dtype=bool)
    m2_gate_eligible_edge_mask = np.ones(len(edge_keys), dtype=bool)
    if args.gomory_hu_m2_max is not None:
        component_m2 = {
            int(label): float(value)
            for label, value in pre_gomory_topology[
                "weighted_m2_by_component"
            ].items()
        }
        point_m2_eligible = np.array([
            component_m2.get(int(label), np.inf) < args.gomory_hu_m2_max
            for label in growth_component_labels
        ], dtype=bool)
        m2_gate_eligible_edge_mask = (
            point_m2_eligible[edge_keys[:, 0]]
            & point_m2_eligible[edge_keys[:, 1]]
        )
        if gomory_removable_mask is not None:
            gomory_removable_mask &= m2_gate_eligible_edge_mask
    shape_gate_eligible_edge_mask = np.ones(len(edge_keys), dtype=bool)
    if args.gomory_hu_shape_score_min is not None:
        # The UMAP graph defines component topology, but the shape gate itself
        # must follow the geometry that DelTriC is trying to preserve.  Use
        # the original-space triangle areas and edge lengths, not their often
        # substantially distorted UMAP counterparts.
        component_shape_score = {
            int(item["component_label"]): item["triangle_area_border_length"][
                "original_shape_score"
            ]
            for item in shape_reference_topology["components"]
        }
        point_shape_eligible = np.array([
            (score is not None and score >= args.gomory_hu_shape_score_min)
            for score in (
                component_shape_score.get(int(label))
                for label in shape_reference_labels
            )
        ], dtype=bool)
        shape_gate_eligible_edge_mask = (
            point_shape_eligible[edge_keys[:, 0]]
            & point_shape_eligible[edge_keys[:, 1]]
        )
        if gomory_removable_mask is not None:
            gomory_removable_mask &= shape_gate_eligible_edge_mask
    triangle_edge_gate_eligible_edge_mask = np.ones(len(edge_keys), dtype=bool)
    if args.gomory_hu_min_triangle_edge_ratio is not None:
        component_triangle_edge_ratio = {
            int(item["component_label"]): (
                item["triangle_area_border_length"]["triangle_count"]
                / max(int(item["edge_count"]), 1)
            )
            for item in shape_reference_topology["components"]
        }
        point_triangle_edge_eligible = np.array([
            component_triangle_edge_ratio.get(int(label), 0.0)
            >= args.gomory_hu_min_triangle_edge_ratio
            for label in shape_reference_labels
        ], dtype=bool)
        triangle_edge_gate_eligible_edge_mask = (
            point_triangle_edge_eligible[edge_keys[:, 0]]
            & point_triangle_edge_eligible[edge_keys[:, 1]]
        )
        if gomory_removable_mask is not None:
            gomory_removable_mask &= triangle_edge_gate_eligible_edge_mask
    gomory_component_eligible_points = (
        point_m2_eligible & point_shape_eligible & point_triangle_edge_eligible
    )
    if args.gomory_pruning_mode == "points":
        gomory_hu_pruned_mask, gomory_hu_metrics = _candidate_vertex_cut_mask(
            edge_keys, gomory_graph_mask,
            max_cut_points=args.gomory_point_cut_size,
            min_component_points=args.gomory_hu_min_component_points,
            min_partition_fraction=args.gomory_hu_min_partition_fraction,
            component_eligible_points=gomory_component_eligible_points,
        )
    elif args.gomory_hu_growth_edges_only:
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
            min_partition_fraction=args.gomory_hu_min_partition_fraction,
            removable_mask=(
                (gomory_removable_mask
                 if gomory_removable_mask is not None else gomory_graph_mask)
                & m2_gate_eligible_edge_mask & shape_gate_eligible_edge_mask
                & triangle_edge_gate_eligible_edge_mask
                if (gomory_removable_mask is not None
                    or args.gomory_hu_m2_max is not None
                    or args.gomory_hu_shape_score_min is not None
                    or args.gomory_hu_min_triangle_edge_ratio is not None) else None
            ),
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
    clustering_metrics = None
    final_ari = None
    if truth_labels is not None:
        clustering_metrics = {
            "ari": float(adjusted_rand_score(truth_labels, final_prediction)),
            "nmi": float(normalized_mutual_info_score(truth_labels, final_prediction)),
            "ami": float(adjusted_mutual_info_score(truth_labels, final_prediction)),
            "homogeneity": float(homogeneity_score(truth_labels, final_prediction)),
            "completeness": float(completeness_score(truth_labels, final_prediction)),
            "v_measure": float(v_measure_score(truth_labels, final_prediction)),
        }
        final_ari = clustering_metrics["ari"]
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
    deltric_pipeline_seconds = time.perf_counter() - run_started
    hdbscan_fit_seconds = None
    if not args.metrics_only:
        # Fit only in standardized original space.  Reuse X_proj for the
        # picture, which means 2-D data stays unprojected and high-D data
        # shares the exact UMAP coordinates used for the DelTriC panels.
        hdbscan_started = time.perf_counter()
        hdbscan_labels = fit_hdbscan(
            X,
            min_cluster_size=args.hdbscan_min_cluster_size,
            min_samples=args.hdbscan_min_samples,
            cluster_selection_method=args.hdbscan_selection_method,
            cluster_selection_epsilon=args.hdbscan_cluster_selection_epsilon,
        )
        hdbscan_fit_seconds = time.perf_counter() - hdbscan_started
        hdbscan_metrics = result_metrics(truth_labels, hdbscan_labels)
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(16, 14), constrained_layout=True)
        if seed_baseline_diagnostics["mode"] == "first_mode_valley":
            seed_title = (
                "initial seed components (orange), original labels (points)\n"
                "seed baseline: first-mode valley "
                f"({seed_baseline_diagnostics['seed_cutoff']:.4f})"
            )
        elif args.seed_baseline == "first_mode":
            seed_title = (
                "initial seed components (orange), original labels (points)\n"
                "seed baseline: median fallback "
                f"({seed_baseline_diagnostics.get('reason', 'no first-mode split')})"
            )
        else:
            seed_title = "initial seed components (orange), original labels (points)\nseed baseline: median"
        _draw_seed_truth(
            axes[0, 0], state["X_proj"], segments, seed_mask, truth_labels,
            seed_title,
        )
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
        final_outlier_count = int(np.count_nonzero(final_prediction < 0))
        axes[1, 1].set_title(
            f"final assignments — ARI={final_ari:.3f}; outliers={final_outlier_count}"
            f"{growth_note}",
            fontsize=11,
        )
        fig.suptitle(
            f"Sequential component-relative growth — {args.data.stem}", fontsize=14
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=150, bbox_inches="tight")
        plt.close(fig)
    if args.component_m2_plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
        _draw_component_m2(
            ax, np.asarray(state["X_proj"], dtype=np.float64), edge_keys,
            bridge_base_mask, pre_gomory_topology,
        )
        fig.suptitle(f"Component M2 diagnostic — {args.data.stem}", fontsize=13)
        m2_path = args.out.with_name(f"{args.out.stem}_component_m2.png")
        m2_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(m2_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
    if args.component_loopiness_plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
        _draw_component_loopiness(
            ax, np.asarray(state["X_proj"], dtype=np.float64), edge_keys,
            bridge_base_mask, pre_gomory_topology,
        )
        fig.suptitle(f"Component loopiness diagnostic — {args.data.stem}", fontsize=13)
        loopiness_path = args.out.with_name(
            f"{args.out.stem}_component_loopiness.png"
        )
        loopiness_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(loopiness_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
    if args.component_area_perimeter_plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
        _draw_component_area_perimeter(
            ax, np.asarray(state["X_proj"], dtype=np.float64), edge_keys,
            shape_reference_mask, shape_reference_topology,
        )
        fig.suptitle(f"Component area/border diagnostic — {args.data.stem}", fontsize=13)
        area_path = args.out.with_name(
            f"{args.out.stem}_component_area_perimeter.png"
        )
        area_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(area_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
    if args.component_shape_space_comparison_plot:
        comparison_path = args.out.with_name(
            f"{args.out.stem}_shape_space_comparison.png"
        )
        _save_component_shape_space_comparison(
            comparison_path,
            np.asarray(state["X_proj"], dtype=np.float64),
            X,
            edge_keys,
            shape_reference_mask,
            shape_reference_topology,
        )
    if args.restoration_boundary_plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
        _draw_restoration_boundary(
            ax, np.asarray(state["X_proj"], dtype=np.float64), edge_keys,
            bridge_base_mask, restoration_boundary_mask,
        )
        fig.suptitle(
            f"Real-restoration boundary diagnostic — {args.data.stem}",
            fontsize=13,
        )
        boundary_path = args.out.with_name(
            f"{args.out.stem}_restoration_boundary.png"
        )
        boundary_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(boundary_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Metrics-only skips figures and HDBSCAN, not the exact DelTriC result.
    # This artifact is also the programmatic result contract.
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
            final_restoration_boundary_mask=restoration_boundary_mask,
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
        "deltric_pipeline_seconds": float(deltric_pipeline_seconds),
        "hdbscan_fit_seconds": (
            None if hdbscan_fit_seconds is None else float(hdbscan_fit_seconds)
        ),
        "native_2d": bool(native_2d),
        "seed_hard_limit": args.seed_hard_limit,
        "seed_baseline": args.seed_baseline,
        "seed_baseline_diagnostics": seed_baseline_diagnostics,
        "hard_growth_limit": args.hard_growth_limit,
        "projected_hard_growth_limit": args.projected_hard_growth_limit,
        "component_growth_enabled": not args.no_component_growth,
        "redundancy_pruning": args.redundancy_pruning,
        "redundancy_threshold": args.redundancy_threshold,
        "redundancy_prune_limit": args.redundancy_prune_limit,
        "redundancy_low_majority": args.redundancy_low_majority,
        "gomory_hu_cut_size": args.gomory_hu_cut_size,
        "gomory_pruning_mode": args.gomory_pruning_mode,
        "gomory_point_cut_size": args.gomory_point_cut_size,
        "gomory_hu_min_component_points": args.gomory_hu_min_component_points,
        "gomory_hu_min_partition_fraction": args.gomory_hu_min_partition_fraction,
        "gomory_hu_hull_ratio_skip_threshold": args.gomory_hu_hull_ratio_skip_threshold,
        "gomory_hu_m2_max": args.gomory_hu_m2_max,
        "gomory_hu_shape_score_min": args.gomory_hu_shape_score_min,
        "gomory_hu_shape_score_space": "standardized original feature space",
        "gomory_hu_min_triangle_edge_ratio": args.gomory_hu_min_triangle_edge_ratio,
        "gomory_hu_shape_reference_edge_count": int(np.count_nonzero(
            shape_reference_mask,
        )),
        "gomory_hu_shape_eligible_edge_count": int(np.count_nonzero(
            (gomory_graph_mask & shape_gate_eligible_edge_mask),
        )),
        "gomory_hu_m2_eligible_edge_count": int(np.count_nonzero(
            gomory_removable_mask if gomory_removable_mask is not None
            else gomory_graph_mask & m2_gate_eligible_edge_mask,
        )),
        "gomory_hu_growth_edges_only": args.gomory_hu_growth_edges_only,
        "gomory_hu_boundary_edges_only": args.gomory_hu_boundary_edges_only,
        "boundary_edge_expansion_hops": args.boundary_edge_expansion_hops,
        "restore_intra_component_edges": args.restore_intra_component_edges,
        "restore_hard_limit": args.restore_hard_limit,
        "growth_restoration_mode": args.growth_restoration_mode,
        "interior_real_restoration": args.interior_real_restoration,
        "restoration_boundary_edge_count": int(np.count_nonzero(
            restoration_boundary_mask & bridge_base_mask
        )),
        "seed_only_restoration": args.growth_restoration_mode == "seed_only",
        "pre_growth_restored_edge_count": int(np.count_nonzero(
            growth["pre_growth_restored_mask"]
        )),
        "growth_restored_edge_count": int(np.count_nonzero(
            growth["restored_during_growth_mask"]
        )),
        "virtual_restored_edge_count": int(np.count_nonzero(
            growth["virtual_restored_mask"]
        )),
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
        "clustering_metrics": clustering_metrics,
        "ari": final_ari,
        "ari_definition": (
            "ARI after optional outlier reassignment; components with <= "
            f"{args.component_growth_min_edges} edges are otherwise mapped to noise (-1)"
            if final_ari is not None else None
        ),
        "seed_edge_count": int(np.count_nonzero(initial_mask)),
        "growth_seed_min_edges": int(args.growth_seed_min_edges),
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
        "shape_reference_topology": shape_reference_topology,
        "pre_gomory_component_topology": pre_gomory_topology,
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
