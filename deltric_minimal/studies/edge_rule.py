"""Learned edge-pruning rule for DelTriC (research add-on).

This module ports the depth-4 decision tree extracted in
``DELTRIC_RULE_EXTRACTION_REPORT.md`` into the component-growth pipeline so its
effect on clustering quality (ARI) can be measured.

The tree was fitted on shape-grouped holdout over 24 shapes / 1.34M Delaunay
edges with target ``cluster_label_bridge`` (endpoints carry different ground
truth labels).  It uses four of the nineteen candidate features:

    orig_knn_scale_ratio
    orig_length                          (run-median normalised)
    ambient_projected_density_log_ratio
    endpoint_density_asymmetry

Two conventions must match the study exactly or the thresholds are meaningless:

* ``k = 15`` for the ambient/projected kNN radii, regardless of the
  component-growth ``knn_k`` (which defaults to 30 and is set to 50 by
  ``run_plot_stages.sh``).  These are different quantities that happen to share
  a name.
* ``orig_length`` is divided by the *median* Delaunay edge length of the run.
  The backend keeps raw lengths, so the normalisation happens here.

The exported ``rules_tree_depth4_raw.py`` from the study emits bare feature
names rather than dict lookups and is not directly executable; the thresholds
below were transcribed from ``rules_tree_depth4_raw.json``.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

#: kNN neighbourhood size used by ``edge_pruning_study.py --ambient-knn-k``.
RULE_KNN_K = 15

#: Thresholds transcribed from ``rules_tree_depth4_raw.json``.
T_SCALE_RATIO_ROOT = 1.285209
T_ORIG_LENGTH = 0.665956
T_DENSITY_LOG_RATIO = 1.715939
T_SCALE_RATIO_INNER = 2.604881
T_DENSITY_ASYMMETRY = 0.274422

RULE_FEATURES = (
    "orig_knn_scale_ratio",
    "orig_length",
    "ambient_projected_density_log_ratio",
    "endpoint_density_asymmetry",
)


def _knn_radius(A: np.ndarray, k: int) -> np.ndarray:
    """Distance from each point to its ``k``-th nearest neighbour.

    Mirrors ``_build_knn_relation`` + the ``[:, -1]`` column take in
    ``edge_pruning_study._edge_feature_arrays``: self is excluded by asking for
    ``k + 1`` neighbours, and ``k`` is clipped to ``n - 1``.
    """
    n = len(A)
    k_eff = min(max(1, int(k)), n - 1)
    distances = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean").fit(A).kneighbors(
        A, return_distance=True
    )[0]
    return distances[:, -1]


def rule_features(
    X: np.ndarray,
    X_proj: np.ndarray,
    edge_keys: np.ndarray,
    orig_sizes: np.ndarray,
    proj_sizes: np.ndarray,
    k: int = RULE_KNN_K,
) -> dict[str, np.ndarray]:
    """Compute the four features the depth-4 rule reads.

    ``orig_sizes`` / ``proj_sizes`` are the per-edge lengths the backend
    already has; passing them in avoids recomputing norms.
    """
    u, v = np.asarray(edge_keys)[:, 0], np.asarray(edge_keys)[:, 1]
    orig_sizes = np.asarray(orig_sizes, dtype=np.float64)
    proj_sizes = np.asarray(proj_sizes, dtype=np.float64)

    ambient_distances = _knn_radius(np.asarray(X, dtype=np.float64), k)
    projected_distances = _knn_radius(np.asarray(X_proj, dtype=np.float64), k)

    ambient_edge_scale = np.sqrt(ambient_distances[u] * ambient_distances[v])
    projected_edge_scale = np.sqrt(projected_distances[u] * projected_distances[v])

    orig_median = float(np.median(orig_sizes)) if len(orig_sizes) else 1.0

    with np.errstate(divide="ignore", invalid="ignore"):
        features = {
            "orig_length": orig_sizes / max(orig_median, 1e-12),
            "orig_knn_scale_ratio": orig_sizes / np.maximum(ambient_edge_scale, 1e-12),
            "ambient_projected_density_log_ratio": np.log(
                np.maximum(ambient_edge_scale, 1e-12)
                / np.maximum(projected_edge_scale, 1e-12)
            ),
            "endpoint_density_asymmetry": np.abs(
                np.log(
                    np.maximum(ambient_distances[u], 1e-12)
                    / np.maximum(ambient_distances[v], 1e-12)
                )
            ),
        }
    for name, values in features.items():
        features[name] = np.nan_to_num(values, nan=0.0, posinf=1e6, neginf=-1e6)
    return features


def prune_mask_from_features(features: dict[str, np.ndarray]) -> np.ndarray:
    """Depth-4 tree, vectorised.  ``True`` means the rule votes to PRUNE.

    Leaf structure (support and observed bridge rate from the fitted tree):

        scale_ratio <= 1.285                       -> KEEP   (1051124)
        else orig_length <= 0.666                  -> KEEP   (  18490)
        else density_log_ratio <= 1.716
            scale_ratio <= 2.605                   -> KEEP   (  93005, rate 0.108)
            else                                   -> PRUNE  (  31058, rate 0.614)
        else
            density_asymmetry <= 0.274             -> KEEP   (  79586, rate 0.292)
            else                                   -> PRUNE  (  62274, rate 0.679)
    """
    scale_ratio = features["orig_knn_scale_ratio"]
    orig_length = features["orig_length"]
    density_log_ratio = features["ambient_projected_density_log_ratio"]
    density_asymmetry = features["endpoint_density_asymmetry"]

    reached = (scale_ratio > T_SCALE_RATIO_ROOT) & (orig_length > T_ORIG_LENGTH)
    low_density_branch = density_log_ratio <= T_DENSITY_LOG_RATIO

    prune = reached & (
        (low_density_branch & (scale_ratio > T_SCALE_RATIO_INNER))
        | (~low_density_branch & (density_asymmetry > T_DENSITY_ASYMMETRY))
    )
    return prune


def depth4_rule(
    X: np.ndarray,
    X_proj: np.ndarray,
    edge_keys: np.ndarray,
    orig_sizes: np.ndarray,
    proj_sizes: np.ndarray,
    k: int = RULE_KNN_K,
) -> np.ndarray:
    """Callable matching the ``edge_rule`` hook of ``component_growth_graph``."""
    return prune_mask_from_features(
        rule_features(X, X_proj, edge_keys, orig_sizes, proj_sizes, k)
    )
