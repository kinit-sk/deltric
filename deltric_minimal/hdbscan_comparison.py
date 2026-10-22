#!/usr/bin/env python3
"""Shared, fair HDBSCAN evaluation and cluster-colouring helpers.

HDBSCAN is fitted on standardized *original-space* coordinates, never on a
projection.  The ARI includes every sample, including HDBSCAN's ``-1`` noise
assignments.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score


def fit_hdbscan(
    X: np.ndarray,
    *,
    min_cluster_size: int,
    min_samples: int | None,
    cluster_selection_method: str,
    cluster_selection_epsilon: float = 0.0,
) -> np.ndarray:
    """Fit the standard original-space HDBSCAN baseline deterministically."""
    model = HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=None if min_samples is None else int(min_samples),
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=float(cluster_selection_epsilon),
        n_jobs=1,
    )
    return np.asarray(model.fit_predict(X), dtype=np.int64)


def result_metrics(y_true: np.ndarray, labels: np.ndarray) -> dict[str, float | int]:
    """Metrics deliberately include HDBSCAN noise labels in ARI."""
    non_noise = labels >= 0
    return {
        "ari": float(adjusted_rand_score(y_true, labels)),
        "n_pred_clusters": int(np.unique(labels[non_noise]).size),
        "noise_fraction": float(np.mean(~non_noise)),
    }


def _label_colours(labels: np.ndarray) -> np.ndarray:
    colours = np.full((len(labels), 4), (0.62, 0.62, 0.62, 0.55), dtype=float)
    unique = np.unique(labels[labels >= 0])
    if len(unique):
        cmap = plt.get_cmap("turbo", max(2, len(unique)))
        for index, label in enumerate(unique):
            colours[labels == label] = cmap(index)
    return colours

