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


# ---------------------------------------------------------------------------
# Depth-6 tree (``studies/edge_rule_extraction.py`` with ``(2, 3, 4, 6)``,
# exported as ``rules_tree_depth6_raw.py``).  Same raw feature variant and
# lineage as the depth-4 tree above, but with two more splits and two more
# features: ``orig_incident_min_ratio`` and ``projected_length``.  Grouped
# holdout AUC improves from 0.859 to 0.874 and r80 intra-cluster retention
# from 0.739 to 0.838 (pooled/raw), pending the ARI check that showed the
# depth-4 tree's edge-level gains did not survive contact with clustering.
# ---------------------------------------------------------------------------

T6_SCALE_RATIO_ROOT = 1.248494
T6_INCIDENT_MIN_RATIO = 1.324070
T6_DENSITY_ASYMMETRY_1 = 0.322869
T6_DENSITY_LOG_RATIO_1 = 0.624800
T6_PROJECTED_LENGTH = 11.198782
T6_DENSITY_LOG_RATIO_2 = 1.710659
T6_DENSITY_ASYMMETRY_2 = 0.126819
T6_SCALE_RATIO_INNER = 1.538848
T6_ORIG_LENGTH = 0.643018
T6_DENSITY_ASYMMETRY_3 = 0.273857

RULE_FEATURES_DEPTH6 = RULE_FEATURES + ("orig_incident_min_ratio", "projected_length")


def _incident_min_ratio(edge_keys: np.ndarray, orig_sizes: np.ndarray) -> np.ndarray:
    """``orig_length / min(orig_length of edges sharing an endpoint)``.

    Mirrors ``edge_pruning_study._edge_feature_arrays``'s ``orig_incident_min_ratio``:
    for each edge, the minimum raw length among all *other* edges incident to
    either endpoint (the union of both endpoints' incident-edge sets, self
    excluded).  A ratio near 1 means the edge is as short as its shortest
    neighbour; a large ratio means every locally-incident edge is shorter,
    which is the "sticks out of a dense cluster" signature.
    """
    edge_keys = np.asarray(edge_keys)
    orig_sizes = np.asarray(orig_sizes, dtype=np.float64)
    n_edges = len(edge_keys)
    n_points = int(edge_keys.max()) + 1 if n_edges else 0

    point_to_edges: list[list[int]] = [[] for _ in range(n_points)]
    for ei, (a, b) in enumerate(edge_keys):
        point_to_edges[int(a)].append(ei)
        point_to_edges[int(b)].append(ei)

    incident_min = np.empty(n_edges, dtype=np.float64)
    for ei, (a, b) in enumerate(edge_keys):
        incident = set(point_to_edges[int(a)]) | set(point_to_edges[int(b)])
        incident.discard(ei)
        if incident:
            incident_min[ei] = orig_sizes[np.fromiter(incident, dtype=np.int64)].min()
        else:
            incident_min[ei] = orig_sizes[ei]

    return orig_sizes / np.maximum(incident_min, 1e-12)


def rule_features_depth6(
    X: np.ndarray,
    X_proj: np.ndarray,
    edge_keys: np.ndarray,
    orig_sizes: np.ndarray,
    proj_sizes: np.ndarray,
    k: int = RULE_KNN_K,
) -> dict[str, np.ndarray]:
    """The four depth-4 features plus ``orig_incident_min_ratio`` and
    ``projected_length`` (run-median normalised, like ``orig_length``)."""
    features = rule_features(X, X_proj, edge_keys, orig_sizes, proj_sizes, k)

    proj_sizes = np.asarray(proj_sizes, dtype=np.float64)
    proj_median = float(np.median(proj_sizes)) if len(proj_sizes) else 1.0
    features["projected_length"] = proj_sizes / max(proj_median, 1e-12)
    features["orig_incident_min_ratio"] = _incident_min_ratio(edge_keys, orig_sizes)

    for name in ("projected_length", "orig_incident_min_ratio"):
        features[name] = np.nan_to_num(
            features[name], nan=0.0, posinf=1e6, neginf=-1e6
        )
    return features


def prune_mask_from_features_depth6(features: dict[str, np.ndarray]) -> np.ndarray:
    """Depth-6 tree, vectorised.  ``True`` means the rule votes to PRUNE.

    See ``rules_tree_depth6_raw.py`` (``studies/results/edge_rules_depth6/``)
    for the literal if/elif form this mirrors.
    """
    scale_ratio = features["orig_knn_scale_ratio"]
    orig_length = features["orig_length"]
    density_log_ratio = features["ambient_projected_density_log_ratio"]
    asymmetry = features["endpoint_density_asymmetry"]
    incident_min_ratio = features["orig_incident_min_ratio"]
    projected_length = features["projected_length"]

    low_scale = scale_ratio <= T6_SCALE_RATIO_ROOT
    prune_low_scale = (
        (incident_min_ratio > T6_INCIDENT_MIN_RATIO)
        & (asymmetry > T6_DENSITY_ASYMMETRY_1)
        & (density_log_ratio > T6_DENSITY_LOG_RATIO_1)
    )

    short_projected = projected_length <= T6_PROJECTED_LENGTH
    tight_asymmetry = asymmetry <= T6_DENSITY_ASYMMETRY_2
    prune_short_projected = (density_log_ratio > T6_DENSITY_LOG_RATIO_2) & (
        (tight_asymmetry & (scale_ratio > T6_SCALE_RATIO_INNER))
        | (
            ~tight_asymmetry
            & (orig_length > T6_ORIG_LENGTH)
            & (asymmetry > T6_DENSITY_ASYMMETRY_3)
        )
    )
    prune_high_scale = np.where(short_projected, prune_short_projected, True)

    return np.where(low_scale, prune_low_scale, prune_high_scale)


def depth6_rule(
    X: np.ndarray,
    X_proj: np.ndarray,
    edge_keys: np.ndarray,
    orig_sizes: np.ndarray,
    proj_sizes: np.ndarray,
    k: int = RULE_KNN_K,
) -> np.ndarray:
    """Callable matching the ``edge_rule`` hook of ``component_growth_graph``."""
    return prune_mask_from_features_depth6(
        rule_features_depth6(X, X_proj, edge_keys, orig_sizes, proj_sizes, k)
    )
