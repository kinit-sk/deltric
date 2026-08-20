"""Run-level "is this projection trustworthy" gate for the growth phase.

Ports the depth-3 tree from ``studies/growth_dr_trust_tree.py`` (fit on all
415 rows of kinit's ``variant_dr_metrics.csv``, target ``degenerate_graph``:
the projected Delaunay graph's mean degree drops below 2, or it has fewer
than half-n edges) into a callable ``component_growth_graph`` can consult
*before* building the graph, the way ``edge_rule.py`` supplies a per-edge
callable it consults while pruning.

Unlike the per-edge rules, this only uses metrics computable from ``X`` and
``X_proj`` alone -- no ground truth. The tree the study fit also offered
``centroid_corr`` (needs cluster labels) and ``manifold_corr`` (needs the
generative manifold coordinate); both were dropped before fitting the
version below because neither is available at gate time -- that is what the
clustering this gate feeds into is trying to produce.

The surviving rule collapses to a 3-condition AND (see
``studies/results/growth_dr_trust/rules_tree_depth3.txt``):

    recall_at_5 <= 0.2114 AND n_surprise <= 0.2531 AND dist_corr_spearman <= 0.8751
        -> DEGENERATE (fall back)
    else
        -> TRUST

``recall_at_5`` alone carries most of the tree's importance (0.80); the
other two conditions only matter in the low-recall band.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors

T_RECALL_AT_5 = 0.211400
T_N_SURPRISE = 0.253100
T_DIST_CORR_SPEARMAN = 0.875100

TRUST_GATE_FEATURES = ("recall_at_5", "n_surprise", "dist_corr_spearman")


def _recall_at_5(X: np.ndarray, X_proj: np.ndarray) -> float:
    n = len(X)
    k = min(5, n - 1)
    if k < 1:
        return 1.0
    nn_x = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(return_distance=False)[:, 1:]
    nn_z = NearestNeighbors(n_neighbors=k + 1).fit(X_proj).kneighbors(return_distance=False)[:, 1:]
    both = np.concatenate([nn_x, nn_z], axis=1)
    both.sort(axis=1)
    overlap = (both[:, 1:] == both[:, :-1]).sum(axis=1) / k
    return float(overlap.mean())


def dr_trust_features(X: np.ndarray, X_proj: np.ndarray) -> dict[str, float]:
    """The three run-level metrics the gate reads, computed from (X, X_proj)."""
    X = np.asarray(X, dtype=np.float64)
    X_proj = np.asarray(X_proj, dtype=np.float64)
    n = len(X)

    recall_at_5 = _recall_at_5(X, X_proj)

    k_tw = min(10, n - 1)
    n_surprise = (
        1.0 - float(trustworthiness(X, X_proj, n_neighbors=max(k_tw, 1)))
        if n > 2
        else 0.0
    )

    if n > 3:
        rho = spearmanr(pdist(X), pdist(X_proj)).statistic
        dist_corr_spearman = float(rho) if np.isfinite(rho) else 0.0
    else:
        dist_corr_spearman = 0.0

    return {
        "recall_at_5": recall_at_5,
        "n_surprise": n_surprise,
        "dist_corr_spearman": dist_corr_spearman,
    }


def predict_degenerate(features: dict[str, float]) -> bool:
    """``True`` means the gate does not trust this projection's graph."""
    return (
        features["recall_at_5"] <= T_RECALL_AT_5
        and features["n_surprise"] <= T_N_SURPRISE
        and features["dist_corr_spearman"] <= T_DIST_CORR_SPEARMAN
    )


def is_projection_trustworthy(X: np.ndarray, X_proj: np.ndarray) -> bool:
    """Convenience: compute the features and apply the gate in one call."""
    return not predict_degenerate(dr_trust_features(X, X_proj))
