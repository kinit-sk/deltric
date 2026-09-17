#!/usr/bin/env python3
"""Shared, fair HDBSCAN evaluation and final-cluster plotting helpers.

HDBSCAN is fitted on standardized *original-space* coordinates.  A UMAP
embedding is used only as a fixed 2-D display, never as the clustering input.
The ARI includes every sample, including HDBSCAN's ``-1`` noise assignments.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a curated data set and standardize its original-space features."""
    loaded = np.load(path, allow_pickle=False)
    label_key = "y" if "y" in loaded.files else "y_clean"
    X = StandardScaler().fit_transform(
        np.asarray(loaded["X"], dtype=np.float64)
    )
    return X, np.asarray(loaded[label_key], dtype=np.int64)


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


def umap_for_display(
    X: np.ndarray, *, n_neighbors: int, n_epochs: int, random_state: int = 42,
) -> np.ndarray:
    """A deterministic display embedding; it is not used by HDBSCAN."""
    # Keep UMAP (and therefore Numba) out of the parameter sweep entirely.
    from umap import UMAP

    return UMAP(
        n_components=2,
        n_neighbors=min(max(2, int(n_neighbors)), max(2, len(X) - 1)),
        n_epochs=int(n_epochs),
        random_state=int(random_state),
        transform_seed=int(random_state),
    ).fit_transform(X)


def _label_colours(labels: np.ndarray) -> np.ndarray:
    colours = np.full((len(labels), 4), (0.62, 0.62, 0.62, 0.55), dtype=float)
    unique = np.unique(labels[labels >= 0])
    if len(unique):
        cmap = plt.get_cmap("turbo", max(2, len(unique)))
        for index, label in enumerate(unique):
            colours[labels == label] = cmap(index)
    return colours


def plot_final_clusters(
    X_display: np.ndarray,
    labels: np.ndarray,
    *,
    title: str,
    out: Path,
) -> None:
    """Write the requested compact final HDBSCAN plot."""
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 7.0), constrained_layout=True)
    ax.scatter(
        X_display[:, 0], X_display[:, 1], c=_label_colours(labels),
        s=5.5, linewidths=0, rasterized=True,
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("UMAP 1 (display only)")
    ax.set_ylabel("UMAP 2 (display only)")
    ax.set_aspect("equal", adjustable="datalim")
    fig.savefig(out, dpi=220)
    plt.close(fig)
