"""Minimal projection and Delaunay helpers used by component-growth DelTriC.

The production component-growth pipeline needs only these helpers: UMAP/PCA
projection, Delaunay topology, and original- or projected-space edge lengths.
Legacy learned pruning and clustering implementations deliberately do not live
in this self-contained prototype.
"""

from __future__ import annotations

import time
import os
import sys
import tempfile
from pathlib import Path
from itertools import combinations

import numpy as np
from scipy.spatial import Delaunay
from sklearn.decomposition import PCA


def _configure_numba_cache() -> None:
    """Use a cache that cannot collide across Python minor versions.

    Numba cache artifacts are interpreter/ABI-specific.  Configure the cache
    before importing UMAP (and therefore Numba), while respecting an explicit
    user-provided ``NUMBA_CACHE_DIR``.
    """
    if os.environ.get("NUMBA_CACHE_DIR"):
        return
    tag = f"py{sys.version_info.major}{sys.version_info.minor}"
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    target = base / "deltric" / "numba" / tag
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError:
        target = Path(tempfile.gettempdir()) / "deltric" / "numba" / tag
        target.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(target)


_configure_numba_cache()
from umap import UMAP


def _resolve_umap_n_epochs(n_samples: int, umap_n_epochs: int | str | None) -> int | None:
    """Preserve the development pipeline's deterministic epoch policy."""
    if umap_n_epochs in (None, "auto"):
        if n_samples < 10_000:
            return None
        if n_samples <= 22_000:
            return 200
        return 100
    return int(umap_n_epochs)


def get_triangles_with_edges(
    X: np.ndarray,
    project_dim: int = 2,
    method: str = "umap",
    back_proj: bool = True,
    umap_n_epochs: int | str | None = "auto",
    umap_n_neighbors: int = 15,
    profile_phases: bool = False,
):
    """Project, triangulate, and return simplex lengths in both spaces.

    The triangulation is built in projected space.  With ``back_proj=True``
    (the active pipeline), simplex sizes are measured in original space.
    """
    t0 = time.perf_counter() if profile_phases else None
    if method == "pca" and X.shape[1] > project_dim:
        X_proj = PCA(n_components=project_dim, random_state=42).fit_transform(X)
    elif method == "umap" and X.shape[1] > project_dim:
        kwargs = {
            "n_components": project_dim,
            "random_state": 42,
            "n_neighbors": umap_n_neighbors,
        }
        resolved_epochs = _resolve_umap_n_epochs(len(X), umap_n_epochs)
        if resolved_epochs is not None:
            kwargs["n_epochs"] = resolved_epochs
        X_proj = UMAP(**kwargs).fit_transform(X)
    elif method == "none" or X.shape[1] <= project_dim:
        X_proj = X
    else:
        raise ValueError(
            f"Invalid projection method {method!r}; choose 'umap', 'pca', or 'none'."
        )

    projection_seconds = (
        time.perf_counter() - t0 if profile_phases and t0 is not None else None
    )
    t1 = time.perf_counter() if profile_phases else None
    triangles = Delaunay(X_proj).simplices

    size_space = X if back_proj else X_proj
    simplex_sizes = []
    simplex_sizes_proj = []
    for simplex in triangles:
        points = size_space[simplex]
        projected_points = X_proj[simplex]
        simplex_sizes.append(max(
            np.linalg.norm(points[i] - points[j])
            for i, j in combinations(range(len(simplex)), 2)
        ))
        simplex_sizes_proj.append(max(
            np.linalg.norm(projected_points[i] - projected_points[j])
            for i, j in combinations(range(len(simplex)), 2)
        ))

    if profile_phases:
        get_triangles_with_edges.last_phase_times = {
            "projection_seconds": projection_seconds,
            "triangulation_seconds": time.perf_counter() - t1,
        }
    return (
        triangles,
        np.asarray(simplex_sizes),
        np.asarray(simplex_sizes_proj),
        X_proj,
    )


def _extract_edge_data(
    triangles: np.ndarray, X_proj: np.ndarray, X_size: np.ndarray | None = None,
):
    """Extract unique simplex edges and their lengths.

    ``X_size`` controls the measurement space; edge topology and centers always
    remain in the projected triangulation space.
    """
    edge_to_tris: dict[tuple[int, int], list[int]] = {}
    for triangle_index, simplex in enumerate(triangles):
        for a, b in combinations(simplex, 2):
            key = (min(int(a), int(b)), max(int(a), int(b)))
            edge_to_tris.setdefault(key, []).append(triangle_index)

    edge_keys = list(edge_to_tris)
    length_space = X_proj if X_size is None else X_size
    starts = np.fromiter((edge[0] for edge in edge_keys), dtype=np.int64)
    ends = np.fromiter((edge[1] for edge in edge_keys), dtype=np.int64)
    edge_sizes = np.linalg.norm(length_space[starts] - length_space[ends], axis=1)
    edge_centers = (X_proj[starts] + X_proj[ends]) / 2.0
    return edge_keys, edge_sizes, edge_centers, edge_to_tris
