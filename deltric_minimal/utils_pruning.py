import time
import json
import numpy as np
import os
import sys
import multiprocessing as mp
from pathlib import Path
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor

from sklearn.decomposition import PCA
from scipy.spatial import Delaunay
from scipy.stats import kurtosis, skew
import networkx as nx
from sklearn.manifold import TSNE
from umap import UMAP
import pandas as pd

from statsmodels import robust

# Optional GNN support (imported lazily if needed)
_gnn_available = False
_gnn_model_cache = None
_mlp_model_cache = None
_mlp_bridge_model_cache = None
_mlp_parallel_context = None
_combined_baseline_weights_cache = None
try:
    import torch
    from torch import nn
    import warnings
    warnings.filterwarnings('ignore')
    _gnn_available = True
except ImportError:
    pass

ANOMALY_THRESH = 5

def _simplex_edges(simplex):
    """Return all unordered edge pairs from a simplex (triangle, tetrahedron, etc.)."""
    n = len(simplex)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((simplex[i], simplex[j]))
    return edges


def _resolve_umap_n_epochs(n_samples, umap_n_epochs):
    """Resolve UMAP n_epochs policy.

    Policy for ``umap_n_epochs`` in auto mode:
      - n < 10k: leave unspecified (use UMAP default)
      - 10k <= n <= 22k: 200 epochs
      - n > 22k: 100 epochs
    """
    if umap_n_epochs in (None, "auto"):
        if n_samples < 10_000:
            return None
        if n_samples <= 22_000:
            return 200
        return 100
    return int(umap_n_epochs)


def get_triangles_with_edges(X, project_dim=2, method='umap', back_proj=True,
                              umap_n_epochs='auto', umap_n_neighbors=15,
                              profile_phases=False):
    """Project data to low-D and compute Delaunay triangulation.

    Parameters
    ----------
    project_dim : int
        Target dimensionality for projection (2 or 3).
    method : str
        'umap', 'pca', 'isomap', 'tsne', 'lle', or 'none'
    umap_n_epochs : int or 'auto'
        UMAP n_epochs parameter (only used when method='umap').
        In 'auto' mode:
          - n < 10k: use UMAP default by not passing n_epochs
          - 10k <= n <= 22k: n_epochs = 200
          - n > 22k: n_epochs = 100
    umap_n_neighbors : int
        UMAP n_neighbors parameter (only used when method='umap')
    """
    from sklearn.manifold import Isomap, TSNE, LocallyLinearEmbedding

    t0 = time.perf_counter() if profile_phases else None
    if method == 'pca' and X.shape[1] > project_dim:
        projector = PCA(n_components=project_dim, random_state=42)
        X_proj = projector.fit_transform(X)
    elif method == 'umap' and X.shape[1] > project_dim:
        resolved_epochs = _resolve_umap_n_epochs(len(X), umap_n_epochs)
        umap_kwargs = dict(
            n_components=project_dim,
            random_state=42,
            n_neighbors=umap_n_neighbors,
        )
        if resolved_epochs is not None:
            umap_kwargs["n_epochs"] = resolved_epochs
        projector = UMAP(**umap_kwargs)
        X_proj = projector.fit_transform(X)
    elif method == 'isomap' and X.shape[1] > project_dim:
        projector = Isomap(n_components=project_dim, n_neighbors=umap_n_neighbors)
        X_proj = projector.fit_transform(X)
    elif method == 'tsne' and X.shape[1] > project_dim:
        projector = TSNE(n_components=project_dim, random_state=42, init='pca')
        X_proj = projector.fit_transform(X)
    elif method == 'lle' and X.shape[1] > project_dim:
        projector = LocallyLinearEmbedding(n_components=project_dim,
                                            n_neighbors=umap_n_neighbors,
                                            random_state=42)
        X_proj = projector.fit_transform(X)
    elif method == 'none' or X.shape[1] <= project_dim:
        X_proj = X
    else:
        raise ValueError(f"Invalid projection method '{method}'. "
                         "Choose from 'pca', 'umap', 'isomap', 'tsne', 'lle', 'none'.")

    t_proj = time.perf_counter() - t0 if profile_phases and t0 is not None else None
    t1 = time.perf_counter() if profile_phases else None
    tri = Delaunay(X_proj)
    triangles = tri.simplices  # shape (num_simplices, project_dim+1)

    # Calculate simplex sizes (max edge length)
    simplex_sizes = []
    simplex_sizes_proj = []
    for simplex in triangles:
        pts = X[list(simplex)]  # original space points for accurate edge lengths
        pts_proj = X_proj[list(simplex)]
        if back_proj==False:
            pts = pts_proj
        n_verts = len(simplex)
        edges = [np.linalg.norm(pts[i] - pts[j]) for i in range(n_verts) for j in range(i + 1, n_verts)]
        edges_proj = [np.linalg.norm(pts_proj[i] - pts_proj[j]) for i in range(n_verts) for j in range(i + 1, n_verts)]
        max_edge = max(edges)
        max_edge_proj = max(edges_proj)
        simplex_sizes.append(max_edge)
        simplex_sizes_proj.append(max_edge_proj)

    t_tri = time.perf_counter() - t1 if profile_phases and t1 is not None else None
    if profile_phases:
        get_triangles_with_edges.last_phase_times = {
            "projection_seconds": t_proj,
            "triangulation_seconds": t_tri,
        }
    return triangles, np.array(simplex_sizes), np.array(simplex_sizes_proj), X_proj


# 4️⃣ Get clusters from remaining edges
def get_clusters(edges, n_points):
    G = nx.Graph()
    G.add_nodes_from(range(n_points))
    G.add_edges_from(edges)
    clusters = [list(c) for c in nx.connected_components(G)]
    return clusters

def cluster_umap_representative(X, X_proj, clusters):
    reps = []
    for cluster in clusters:
        pts_proj = X_proj[cluster]
        center = pts_proj.mean(axis=0)
        # find the index in the original dataset
        idx = cluster[np.argmin(np.linalg.norm(pts_proj - center, axis=1))]
        reps.append(X_proj[idx])  # X would be actual point, if X_proj used, the merging effectively runs only in umap space. There is only few points after creating centroids, thus umap would create manifolds wrongly.
    return np.array(reps)

def _build_point_triangle_index(triangles, n_points):
    """Build a list of triangle indices for each point.

    Returns a list of length n_points where each entry is a 1-D int array
    of triangle indices that include that point.  This avoids the O(n_tri)
    ``np.any(triangles == p, axis=1)`` scan for every anomaly point.
    """
    point_to_tris = [[] for _ in range(n_points)]
    for ti, tri in enumerate(triangles):
        for p in tri:
            point_to_tris[p].append(ti)
    # Convert to arrays for efficient fancy indexing
    return [np.array(lst, dtype=np.intp) if lst else np.empty(0, dtype=np.intp)
            for lst in point_to_tris]


def merge_anomalies_triangles(X, X_proj, clusters, triangles, sizes, proj_sizes,
                              anomaly_thresh=1, max_iter=3, debug=False,
                              anomaly_sensitivity=0.5):
    """Merge tiny anomaly groups into nearby rich clusters.

    For each anomaly cluster, we compute its *affinity* to each nearby
    rich cluster as the sum of inverse distances from triangle-neighbor
    points.  The anomaly is merged into the richest-affinity cluster
    when the closest point-pair distance is within::

        merge_dist = median(proj_sizes) * anomaly_sensitivity

    Distances are computed in the same space as ``X``:
      - original space when back-projection is enabled
      - projected space otherwise
    This avoids projection distortion causing whole local groups to remain
    unmerged as noise.

    Parameters
    ----------
    anomaly_sensitivity : float in [0, 1]
        Fraction of the median projected triangle edge length used as the
        maximum merge distance.  0.0 = never merge; 1.0 = merge if within
        one full median edge.  Default 0.5.
    """
    n_points = X_proj.shape[0]
    point_to_tris = _build_point_triangle_index(triangles, n_points)

    # Distance space for anomaly merge
    merge_X = X if X is not None else X_proj
    median_edge = np.median(sizes) if X is not None else np.median(proj_sizes)
    merge_dist = median_edge * anomaly_sensitivity

    point_to_cluster = {}
    for ci, cluster in enumerate(clusters):
        for p in cluster:
            point_to_cluster[p] = ci

    merged_clusters = [list(c) for c in clusters]

    for it in range(max_iter):
        if debug:
            print(f"\nIteration {it+1}/{max_iter}")

        merged_any = False
        anomaly_count = 0
        merged_count = 0

        # Collect all anomaly clusters and their merge candidates
        # Process closest-first: sort anomalies by distance to nearest
        # rich-cluster neighbor, then merge in that order.
        anomaly_candidates = []

        for ci, cluster in enumerate(list(merged_clusters)):
            if len(cluster) > anomaly_thresh:
                continue
            if len(cluster) == 0:
                continue

            anomaly_count += 1

            # Collect unique neighbor points via pre-built triangle index
            neighbor_set = set()
            for p in cluster:
                tri_indices = point_to_tris[p]
                if len(tri_indices) == 0:
                    continue
                for tri in triangles[tri_indices]:
                    for q in tri:
                        if q != p and q in point_to_cluster and point_to_cluster[q] != ci:
                            neighbor_set.add(q)

            if not neighbor_set:
                continue

            # Group neighbors by their cluster for vectorized processing
            neigh_by_cluster = {}
            for q in neighbor_set:
                nci = point_to_cluster[q]
                if nci not in neigh_by_cluster:
                    neigh_by_cluster[nci] = []
                neigh_by_cluster[nci].append(q)

            # Vectorized distance computation in merge space
            affinity = {}
            min_dist_per_target = {}
            cluster_arr = np.array(cluster)
            X_cluster = merge_X[cluster_arr]

            for neigh_ci, neigh_points in neigh_by_cluster.items():
                neigh_arr = np.array(neigh_points)
                X_neigh = merge_X[neigh_arr]
                # All pairwise distances: (len(cluster), len(neigh)) in 2D
                diffs = X_cluster[:, None, :] - X_neigh[None, :, :]
                dists = np.linalg.norm(diffs, axis=2)

                with np.errstate(divide='ignore', invalid='ignore'):
                    inv_dists = np.where(dists > 0, 1.0 / dists, 0.0)
                affinity[neigh_ci] = float(inv_dists.sum())
                min_dist_per_target[neigh_ci] = float(dists.min())

            if not affinity:
                continue

            # Find the closest target cluster (not the richest-affinity)
            best_target = min(min_dist_per_target, key=min_dist_per_target.get)
            best_min_dist = min_dist_per_target[best_target]

            anomaly_candidates.append((best_min_dist, ci, best_target, affinity))

        # Sort by distance: merge closest anomaly pairs first
        anomaly_candidates.sort(key=lambda x: x[0])

        for best_min_dist, ci, best_target, affinity in anomaly_candidates:
            # Re-check: cluster may have been emptied by a previous merge
            if len(merged_clusters[ci]) == 0:
                continue
            # Re-check: target may have been merged into another cluster
            if len(merged_clusters[best_target]) == 0:
                continue

            if best_min_dist <= merge_dist:
                merged_clusters[best_target].extend(merged_clusters[ci])
                merged_clusters[ci] = []
                merged_any = True
                merged_count += 1

        if debug:
            print(f" - Anomalies processed: {anomaly_count}")
            print(f" - Merged this round: {merged_count}")

        if merged_any:
            point_to_cluster = {}
            for ci, cluster in enumerate(merged_clusters):
                for p in cluster:
                    point_to_cluster[p] = ci
        else:
            break

    merged_clusters = [sorted(c) for c in merged_clusters if len(c) > 0]
    return merged_clusters


def _extract_edge_data(triangles, X_proj, X_size=None):
    """Extract unique edges from triangulation with sizes and centers.

    Returns
    -------
    edge_keys : list of (int, int)
        Sorted vertex pairs for each unique edge.
    edge_sizes : ndarray of shape (n_edges,)
        Edge lengths in projected space.
    edge_centers : ndarray of shape (n_edges, n_dims)
        Midpoints in projected space.
    edge_to_tris : dict
        Maps edge key -> list of simplex indices containing that edge.
    """
    edge_to_tris = {}
    for ti, simplex in enumerate(triangles):
        for a, b in _simplex_edges(list(simplex)):
            key = (min(a, b), max(a, b))
            if key not in edge_to_tris:
                edge_to_tris[key] = []
            edge_to_tris[key].append(ti)

    edge_keys = list(edge_to_tris.keys())
    X_len = X_proj if X_size is None else X_size
    v0_len = X_len[[k[0] for k in edge_keys]]
    v1_len = X_len[[k[1] for k in edge_keys]]
    v0 = X_proj[[k[0] for k in edge_keys]]
    v1 = X_proj[[k[1] for k in edge_keys]]
    edge_sizes = np.linalg.norm(v1_len - v0_len, axis=1)
    edge_centers = (v0 + v1) / 2.0

    return edge_keys, edge_sizes, edge_centers, edge_to_tris


def _edge_center_density(edge_sizes, edge_centers, k=10, size_lo=0.3, size_hi=3.0):
    """For each edge, compute the fraction of its k nearest edge-center
    neighbours whose size is within [size_lo, size_hi] of the edge's own size.

    Edges in dense regions of similar-sized edges get a high fraction;
    isolated edges (e.g. bridges between clusters) get a low fraction.
    """
    from scipy.spatial import cKDTree

    n = len(edge_sizes)
    k_eff = min(k + 1, n)  # +1 because query includes self
    tree = cKDTree(edge_centers)
    _, idxs = tree.query(edge_centers, k=k_eff)

    frac = np.zeros(n)
    for i in range(n):
        neighbors = idxs[i]
        # Remove self-match
        if neighbors[0] == i:
            neighbors = neighbors[1:]
        else:
            neighbors = neighbors[:k_eff - 1]
        if len(neighbors) == 0:
            continue
        ns = edge_sizes[neighbors]
        lo = edge_sizes[i] * size_lo
        hi = edge_sizes[i] * size_hi
        frac[i] = np.sum((ns >= lo) & (ns <= hi)) / len(neighbors)
    return frac



def _neighbor_length_ratio(edge_keys, edge_sizes, k_neighbor=6):
    """For each edge, compute the ratio of its length to the average length
    of its neighbouring edges (edges sharing an endpoint).

    Returns
    -------
    ratio : ndarray of shape (n_edges,)
        edge_size / mean(neighbor_sizes).  High values indicate the edge
        is much longer than its neighbours -> likely a bridge.
    """
    from collections import defaultdict

    # Build adjacency: point -> list of edge indices
    point_to_edges = defaultdict(list)
    for ei, (a, b) in enumerate(edge_keys):
        point_to_edges[a].append(ei)
        point_to_edges[b].append(ei)

    n = len(edge_sizes)
    ratio = np.ones(n)

    for ei in range(n):
        a, b = edge_keys[ei]
        # Collect all edges incident to either endpoint, excluding self
        neighbor_idxs = set(point_to_edges[a]) | set(point_to_edges[b])
        neighbor_idxs.discard(ei)
        if not neighbor_idxs:
            ratio[ei] = 1.0
            continue
        neighbor_sizes = edge_sizes[list(neighbor_idxs)]
        mean_neighbor = np.mean(neighbor_sizes)
        if mean_neighbor == 0:
            ratio[ei] = 1.0
        else:
            ratio[ei] = edge_sizes[ei] / mean_neighbor
    return ratio


def _find_bridge_edges(edge_keys, X_proj):
    """Find bridge edges whose removal would disconnect clusters.

    Builds a graph from the given edges, finds articulation points
    (cut vertices), and identifies edges that connect two biconnected
    components through an articulation point.

    Parameters
    ----------
    edge_keys : list of (int, int)
        Edges in the graph.
    X_proj : ndarray of shape (n_points, 2)
        Projected coordinates.

    Returns
    -------
    bridge_edge_keys : set of (int, int)
        Edges identified as bridges between clusters.
    """
    if len(edge_keys) == 0:
        return set()

    G = nx.Graph()
    G.add_edges_from(edge_keys)

    # Find articulation points (cut vertices)
    try:
        articulation_pts = set(nx.articulation_points(G))
    except nx.NetworkXError:
        return set()

    if not articulation_pts:
        return set()

    # Get biconnected components (as sets of nodes)
    biconnected_comps = list(nx.biconnected_components(G))

    # Build edge -> biconnected component size mapping
    edge_to_bcc_size = {}
    for comp in biconnected_comps:
        sub = G.subgraph(comp)
        n_edges = sub.number_of_edges()
        for u, v in sub.edges():
            key = (min(u, v), max(u, v))
            edge_to_bcc_size[key] = n_edges

    bridge_edges = set()
    for a, b in edge_keys:
        key = (min(a, b), max(a, b))
        bcc_size = edge_to_bcc_size.get(key, 0)
        a_is_ap = a in articulation_pts
        b_is_ap = b in articulation_pts

        # Edge is a bridge if:
        # 1. Both endpoints are articulation points, OR
        # 2. One endpoint is an AP and the edge is not in a biconnected
        #    component with at least 3 edges (i.e., it's a thin bridge)
        if a_is_ap and b_is_ap:
            bridge_edges.add(key)
        elif (a_is_ap or b_is_ap) and bcc_size < 3:
            bridge_edges.add(key)

    return bridge_edges


# =============================================================================
# EXPERIMENTAL: KDE-based gap-threshold detection
# The goal is to automatically detect the valley between intra-cluster edges
# and inter-cluster bridge edges, avoiding the need to tune prune_param.
# This block can be removed entirely if the approach is abandoned.
# =============================================================================

def _kde_gap_threshold(edge_sizes, sensitivity=1.0):
    """Detect the first significant valley above the median in the edge-length
    KDE.  Returns (hard_threshold, mild_threshold) or (None, None) if no
    clear gap is found.

    Parameters
    ----------
    edge_sizes : ndarray
        Original-space edge lengths.
    sensitivity : float
        Minimum relative valley depth (fraction of surrounding peak height)
        needed to count as a real gap.  0.05 is very lenient; 0.30 requires
        a pronounced dip.  Default 0.15.

    Returns
    -------
    (hard_thr, mild_thr) : (float, float) or (None, None)
    """
    from scipy.stats import gaussian_kde

    if len(edge_sizes) < 40:
        return None, None

    q25, median, q75 = np.percentile(edge_sizes, [25, 50, 75])
    iqr = q75 - q25
    if iqr == 0 or median == 0:
        return None, None

    x_lo = median
    x_hi = min(np.percentile(edge_sizes, 99.5), median + 20 * iqr)
    if (x_hi - x_lo) < iqr * 0.2:
        return None, None

    x = np.linspace(x_lo, x_hi, 500)
    try:
        # Bandwidth ≈ 15 % of IQR — small enough to see a gap, robust to noise
        std_e = np.std(edge_sizes)
        bw = np.clip(iqr * 0.15 / (std_e + 1e-12), 0.01, 0.5)
        kde_fn = gaussian_kde(edge_sizes, bw_method=bw)
        y = kde_fn(x)
    except Exception:
        return None, None

    # Find local minima above the median
    valleys = []
    for i in range(1, len(y) - 1):
        if y[i] <= y[i - 1] and y[i] <= y[i + 1]:
            left_peak = float(y[:i].max()) if i > 0 else y[0]
            right_peak = float(y[i + 1:].max()) if i < len(y) - 1 else y[-1]
            min_peak = min(left_peak, right_peak)
            rel_depth = (min_peak - y[i]) / (min_peak + 1e-12)
            valleys.append((x[i], rel_depth))

    min_depth = 0.15 * sensitivity
    deep = [(pos, d) for pos, d in valleys if d >= min_depth]
    if not deep:
        return None, None

    hard_thr = deep[0][0]              # first significant gap position
    mild_thr = hard_thr - 0.5 * iqr   # half-IQR below the gap as soft zone
    mild_thr = max(mild_thr, median)   # never below median
    return hard_thr, mild_thr


def _select_prune_regime(edge_sizes, projected_edge_sizes=None, center=None):
    """Return 'low' or 'high' using the learned tiny decision tree.

    If projected edge sizes are available, the tree is:

      if projected_gap <= 0.809:
          low  if projected_pearson <= 0.681 else high
      else:
          high if projected_pearson <= 0.591 else low

    where projected_gap = (mean - median) / IQR and projected_pearson =
    3 * (mean - median) / std.

    If projected sizes are unavailable, fall back to the previous
    signed mean-median gap heuristic on ``edge_sizes``.
    """
    def _safe_metrics(values):
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values) & (values > 0)]
        if len(values) < 5:
            return None
        q25, median, q75 = np.percentile(values, [25, 50, 75])
        iqr = q75 - q25
        if iqr <= 0:
            return None
        mean = float(np.mean(values))
        std = float(np.std(values))
        gap = (mean - float(median)) / iqr
        pearson = 3.0 * (mean - float(median)) / std if std > 0 else float("inf")
        return gap, pearson

    proj_stats = _safe_metrics(projected_edge_sizes) if projected_edge_sizes is not None else None
    if proj_stats is not None:
        proj_gap, proj_pearson = proj_stats
        if proj_gap <= 0.809:
            return "low" if proj_pearson <= 0.681 else "high"
        return "high" if proj_pearson <= 0.591 else "low"

    values = np.asarray(edge_sizes, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) < 5:
        return "high"
    q25, median, q75 = np.percentile(values, [25, 50, 75])
    iqr = q75 - q25
    if iqr <= 0:
        return "high"
    scaled_gap = (float(np.mean(values)) - float(median)) / iqr
    return "low" if scaled_gap < 0.1 else "high"

# =============================================================================
# END EXPERIMENTAL
# =============================================================================

# Decision tree rules extracted from full-features tree (depth=4, 12,404 samples).
# These helper functions support _eligible_mask_tree() in pointwise_dual_space mode.

def _compute_tree_features(edge_idx, edge_keys, orig_edge_sizes, point_to_edges):
    """
    Compute 26 features for decision tree classification.
    
    Features include:
    - Edge-relative thresholds (10%, 25%)
    - Relative distance to median
    - First/second-hop neighborhood statistics
    """
    a, b = edge_keys[edge_idx]
    edge_length = orig_edge_sizes[edge_idx]
    
    # Get all edge lengths for statistics
    all_lengths = orig_edge_sizes
    edge_median = np.median(all_lengths)
    edge_std = np.std(all_lengths)
    if edge_std == 0:
        edge_std = 1e-10
    
    # First-hop edges
    first_hop_indices = (
        set(point_to_edges.get(a, []))
        | set(point_to_edges.get(b, []))
    )
    first_hop_indices.discard(edge_idx)
    first_hop_lengths = np.array([orig_edge_sizes[ei] for ei in first_hop_indices])
    
    # Second-hop edges
    second_hop_indices = set()
    for nei in first_hop_indices:
        na, nb = edge_keys[nei]
        second_hop_indices.update(point_to_edges.get(na, []))
        second_hop_indices.update(point_to_edges.get(nb, []))
    second_hop_indices -= first_hop_indices
    second_hop_indices.discard(edge_idx)
    second_hop_lengths = np.array([orig_edge_sizes[ei] for ei in second_hop_indices])
    
    # Compute features
    features = {}
    
    # Edge-relative counts
    features['second_short10'] = (
        np.sum(second_hop_lengths < edge_length * 0.9) / max(len(second_hop_lengths), 1)
    )
    features['second_long25'] = (
        np.sum(second_hop_lengths > edge_length * 1.25) / max(len(second_hop_lengths), 1)
    )
    features['first_long10'] = (
        np.sum(first_hop_lengths > edge_length * 1.1) / max(len(first_hop_lengths), 1)
    )
    features['first_long25'] = (
        np.sum(first_hop_lengths > edge_length * 1.25) / max(len(first_hop_lengths), 1)
    )
    
    # Relative distance to median
    features['relative_dist_to_median'] = (edge_length - edge_median) / edge_std
    
    # First-hop statistics
    if len(first_hop_lengths) > 0:
        first_rel_dists = np.abs(first_hop_lengths - edge_median) / edge_std
        features['first_avg_rel_dist_to_median'] = np.mean(first_rel_dists)
        features['first_std_rel_dist_to_median'] = np.std(first_rel_dists)
    else:
        features['first_avg_rel_dist_to_median'] = 0
        features['first_std_rel_dist_to_median'] = 0
    
    # Second-hop statistics
    if len(second_hop_lengths) > 0:
        second_rel_dists = np.abs(second_hop_lengths - edge_median) / edge_std
        features['second_avg_rel_dist_to_median'] = np.mean(second_rel_dists)
        features['second_std_rel_dist_to_median'] = np.std(second_rel_dists)
    else:
        features['second_avg_rel_dist_to_median'] = 0
        features['second_std_rel_dist_to_median'] = 0
    
    features['second_hop_count'] = len(second_hop_indices)
    
    return features

def _rule1_keep(f):
    """Rule 1: Short edge, uniform first-hop = valid (92.4% purity)"""
    return (f['second_short10'] <= 0.7090 and
            f['second_std_rel_dist_to_median'] <= 2.5997 and
            f['relative_dist_to_median'] <= -0.0690 and
            f['first_std_rel_dist_to_median'] <= 0.1371)

def _rule2_keep(f):
    """Rule 2: Sparse with high 2nd-hop diversity = valid (92.1% purity)"""
    return (f['second_short10'] <= 0.7090 and
            f['second_std_rel_dist_to_median'] > 2.5997 and
            f['relative_dist_to_median'] <= 5.0081 and
            f['second_std_rel_dist_to_median'] > 6.3693)

def _rule3_prune(f):
    """Rule 3: Dense, uniform 2nd-hop = bridge (85.8% purity)"""
    return (f['second_short10'] > 0.7090 and
            f['relative_dist_to_median'] <= 11.6427 and
            f['second_std_rel_dist_to_median'] <= 3.1058 and
            f['second_hop_count'] > 62.50)

def _rule4_keep(f):
    """Rule 4: Dense but diverse 2nd-hop = valid (88.7% purity)"""
    return (f['second_short10'] > 0.7090 and
            f['relative_dist_to_median'] <= 11.6427 and
            f['second_std_rel_dist_to_median'] > 3.1058 and
            f['relative_dist_to_median'] <= 4.8149)

def _rule5_prune(f):
    """Rule 5: Outlier edge, normal endpoints = bridge (93.1% purity)"""
    return (f['second_short10'] > 0.7090 and
            f['relative_dist_to_median'] > 11.6427 and
            f['first_avg_rel_dist_to_median'] <= 5.9037 and
            f['first_long10'] <= 0.0513)

def _rule6_prune(f):
    """Rule 6: Extreme outlier = bridge (98.7% purity) - STRONGEST"""
    return (f['second_short10'] > 0.7090 and
            f['relative_dist_to_median'] > 11.6427 and
            f['first_avg_rel_dist_to_median'] > 5.9037 and
            f['first_long25'] <= 0.2367)

# =============================================================================
# GNN-based intra/inter-cluster classification (Graph Neural Network inference)
# =============================================================================

if _gnn_available:
    class _MPLayer(nn.Module):
        """Mean-aggregation message passing on the line graph."""
        def __init__(self, dim):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(2 * dim, dim), nn.ReLU(), nn.Linear(dim, dim)
            )

        def forward(self, x, edge_index):
            if edge_index.shape[1] == 0:
                return x
            src, dst = edge_index[0], edge_index[1]
            agg   = torch.zeros_like(x)
            count = torch.zeros(x.shape[0], 1, device=x.device)
            agg.scatter_add_(0, dst.unsqueeze(1).expand(-1, x.shape[1]), x[src])
            count.scatter_add_(0, dst.unsqueeze(1),
                               torch.ones(src.shape[0], 1, device=x.device))
            agg = agg / count.clamp(min=1)
            return self.mlp(torch.cat([x, agg], 1))

    class EdgeGNN(nn.Module):
        """
        Edge-focused GNN operating on the line graph of the Delaunay subgraph.

        Line-graph nodes = Delaunay edges; line-graph edges = two Delaunay
        edges share an endpoint.  Message passing aggregates edge-level
        information.  The target edge-node is classified via a sigmoid head.

        Output: scalar probability in [0, 1].  > 0.5 → prune (bridge).
        """
        def __init__(self, n_feat=11, hidden=64):
            super().__init__()
            self.proj = nn.Sequential(nn.Linear(n_feat, hidden), nn.ReLU())
            self.mp1  = _MPLayer(hidden)
            self.mp2  = _MPLayer(hidden)
            self.head = nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Linear(hidden // 2, 1),
                nn.Sigmoid(),
            )

        def forward(self, batch):
            x          = batch['x']
            edge_index = batch['edge_index']
            batch_idx  = batch['batch']
            target_idx = batch['target_idx']   # list of per-sample local ints

            h = self.proj(x)
            h = self.mp1(h, edge_index)
            h = self.mp2(h, edge_index)

            tgt_embeds = []
            for bid, local_idx in enumerate(target_idx):
                node_mask  = (batch_idx == bid).nonzero(as_tuple=True)[0]
                tgt_embeds.append(h[node_mask[local_idx]])

            return self.head(torch.stack(tgt_embeds)).squeeze(1)  # (B,)
    
    def _load_gnn_model(benchmark_dir='benchmark_clustering_tuning'):
        """Load trained GNN model from checkpoint.
        
        The model expects batch inputs with format:
        {
            'x': (total_nodes, input_dim) tensor,
            'edge_index': (2, total_edges) tensor,
            'batch': (total_nodes,) tensor with batch assignment,
            'target_edges': list of (local_u, local_v) tuples,
            'y': (batch_size,) tensor with labels (inference-only)
        }
        
        Searches for model in multiple possible locations:
        1. benchmark_dir/gnn_experiments/gnn_results.pt
        2. gnn_experiments/gnn_results.pt (when running from benchmark_clustering_tuning/)
        3. Current directory relative path
        
        Returns
        -------
        model : EdgeClassifier or None
            Loaded model in eval mode, or None if load fails
        config : dict or None
            Training configuration
        """
        global _gnn_model_cache
        
        # Check cache first
        if _gnn_model_cache is not None:
            return _gnn_model_cache
        
        try:
            # Try multiple path candidates
            candidates = [
                Path(benchmark_dir) / 'gnn_experiments' / 'gnn_results.pt',
                Path('gnn_experiments') / 'gnn_results.pt',
                Path('.') / 'gnn_experiments' / 'gnn_results.pt',
            ]
            
            model_path = None
            for candidate in candidates:
                if candidate.exists():
                    model_path = candidate
                    break
            
            if model_path is None:
                return None, None
            
            checkpoint = torch.load(str(model_path), map_location='cpu', weights_only=False)
            config = checkpoint.get('config', {})
            model_state = checkpoint.get('model_state')
            
            if model_state is None:
                return None, None
            
            # Infer dimensions from saved weights.
            key = 'encoder.input_proj.0.weight'
            if key not in model_state:
                print(f"Warning: unexpected model weights (missing '{key}')")
                return None, None

            n_feat = model_state[key].shape[1]
            hidden_dim = model_state[key].shape[0]

            model = _GNNEdgeClassifier(n_feat, hidden_dim)
            model.load_state_dict(model_state)
            model.eval()

            config = dict(checkpoint.get('config', {}))
            config['model_type'] = 'edge_classifier'
            config['checkpoint_path'] = str(model_path)
            
            _gnn_model_cache = (model, config)
            return model, config
        except Exception as e:
            print(f"Warning: Could not load GNN model: {e}")
            return None, None
    
    def _build_delaunay_graph(triangles):
        """Build NetworkX graph from Delaunay triangles (one-time operation)."""
        edge_list = []
        for simplex in triangles:
            for i in range(len(simplex)):
                for j in range(i+1, len(simplex)):
                    edge_list.append((simplex[i], simplex[j]))
                    edge_list.append((simplex[j], simplex[i]))
        
        G = nx.Graph()
        G.add_edges_from(edge_list)
        return G
    
    def _compute_gnn_global_feats(edge_lengths):
        """5 dataset-level scalars describing the edge length distribution."""
        from scipy.stats import skew as _skew
        med  = float(np.median(edge_lengths))
        q25  = float(np.percentile(edge_lengths, 25))
        q75  = float(np.percentile(edge_lengths, 75))
        p10  = float(np.percentile(edge_lengths, 10))
        p90  = float(np.percentile(edge_lengths, 90))
        iqr  = q75 - q25
        sm   = max(med, 1e-10)
        sp10 = max(p10, 1e-10)
        return np.array([
            iqr / sm,
            p90 / sp10,
            float(_skew(edge_lengths)),
            float((edge_lengths > 2 * med).mean()),
            float((edge_lengths > 3 * med).mean()),
        ], dtype=np.float32), med

    def _extract_line_subgraph(target_ei, edge_keys, edge_lengths,
                                point_to_edges, global_median, gfeats,
                                max_degree, max_hops=2):
        """
        Build a 2-hop line-graph subgraph around the target edge.

        Line-graph nodes = Delaunay edges that are within 2 hops of the target.
        Line-graph edges = two Delaunay edges share a Delaunay endpoint.

        Per edge-node features (11-dim):
          Local (6):
            0: L / global_median
            1: log(L / global_median)
            2: degree of endpoint u (normalised)
            3: degree of endpoint v (normalised)
            4: is_target flag
            5: hop_dist (0 / 1 / 2)
          Global (5):
            6-10: IQR/median, p90/p10, skewness, frac>2x, frac>3x

        Returns
        -------
        features   : torch.FloatTensor (N_ln, 11)
        edge_index : torch.LongTensor  (2, 2*E_ln)  bidirectional
        target_idx : int
        """
        from collections import defaultdict as _dd
        u, v = edge_keys[target_ei]
        safe_med = max(float(global_median), 1e-10)

        # BFS on Delaunay nodes to collect all edges within 2 hops
        node_dist = {u: 0, v: 0}
        queue = [u, v]
        visited = set()
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            d = node_dist[node]
            if d < max_hops:
                for nei_ei in point_to_edges.get(node, []):
                    na, nb = edge_keys[nei_ei]
                    nb_ = nb if na == node else na
                    if nb_ not in node_dist:
                        node_dist[nb_] = d + 1
                        queue.append(nb_)

        subgraph_edges = set()
        for node in visited:
            for ei in point_to_edges.get(node, []):
                na, nb = edge_keys[ei]
                if na in visited and nb in visited:
                    subgraph_edges.add(ei)
        subgraph_edges.add(target_ei)
        subgraph_edges = sorted(subgraph_edges)
        ei_to_ln = {ei: i for i, ei in enumerate(subgraph_edges)}

        def edge_hop(ei):
            a, b = edge_keys[ei]
            if ei == target_ei:
                return 0
            if a in (u, v) or b in (u, v):
                return 1
            return 2

        n_ln = len(subgraph_edges)
        features = np.zeros((n_ln, 11), dtype=np.float32)
        for i, ei in enumerate(subgraph_edges):
            a, b = edge_keys[ei]
            L    = float(edge_lengths[ei])
            features[i, 0] = L / safe_med
            features[i, 1] = float(np.log(max(L / safe_med, 1e-10)))
            features[i, 2] = len(point_to_edges.get(a, [])) / max(max_degree, 1)
            features[i, 3] = len(point_to_edges.get(b, [])) / max(max_degree, 1)
            features[i, 4] = 1.0 if ei == target_ei else 0.0
            features[i, 5] = edge_hop(ei)
            features[i, 6:] = gfeats

        # Line-graph adjacency: edges sharing a Delaunay endpoint
        ep_to_edges = _dd(list)
        for ei in subgraph_edges:
            a, b = edge_keys[ei]
            ep_to_edges[a].append(ei)
            ep_to_edges[b].append(ei)

        lg_edge_set = set()
        for sharing in ep_to_edges.values():
            for ii in range(len(sharing)):
                for jj in range(ii + 1, len(sharing)):
                    p, q = ei_to_ln[sharing[ii]], ei_to_ln[sharing[jj]]
                    if p != q:
                        lg_edge_set.add((min(p, q), max(p, q)))

        if lg_edge_set:
            arr = np.array(sorted(lg_edge_set), dtype=np.int64)
            et  = torch.from_numpy(arr)
            edge_index = torch.cat([et, et.flip(1)], 0).T
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)

        return torch.from_numpy(features), edge_index, ei_to_ln[target_ei]


class _GNNMessagePassingLayer(nn.Module):
    """Single mean-aggregation message passing step for the edge GNN."""
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x, edge_index):
        if edge_index.shape[1] == 0:
            return x
        src, dst = edge_index[0], edge_index[1]
        agg = torch.zeros_like(x)
        count = torch.zeros(x.shape[0], 1, device=x.device)
        agg.scatter_add_(0, dst.unsqueeze(1).expand(-1, x.shape[1]), x[src])
        count.scatter_add_(0, dst.unsqueeze(1), torch.ones(src.shape[0], 1, device=x.device))
        agg = agg / count.clamp(min=1)
        return self.mlp(torch.cat([x, agg], dim=1))


class _GNNGraphEncoder(nn.Module):
    """Encodes node features with two rounds of message passing."""
    def __init__(self, node_dim, hidden_dim=64):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mp1 = _GNNMessagePassingLayer(hidden_dim)
        self.mp2 = _GNNMessagePassingLayer(hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x, edge_index):
        h = self.input_proj(x)
        h = self.mp1(h, edge_index)
        h = self.mp2(h, edge_index)
        return h


class _GNNEdgeClassifier(nn.Module):
    """Classifies edges as intra-cluster or inter-cluster."""
    def __init__(self, node_dim, hidden_dim=64):
        super().__init__()
        self.encoder = _GNNGraphEncoder(node_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, batch):
        x = batch['x']
        edge_index = batch['edge_index']
        batch_idx = batch['batch']
        target_edges = batch['target_edges']

        node_embed = self.encoder(x, edge_index)
        edge_embeds = []

        for batch_id in range(len(target_edges)):
            mask = (batch_idx == batch_id)
            sample_nodes = torch.where(mask)[0]

            a_local, b_local = target_edges[batch_id]
            a_global = sample_nodes[a_local]
            b_global = sample_nodes[b_local]

            edge_embeds.append(torch.cat([node_embed[a_global], node_embed[b_global]]))

        return self.classifier(torch.stack(edge_embeds))


def _compute_gnn_node_features(edge_lengths, point_to_edges, n_points):
    """Compute the base node features used by the trained GNN."""
    global_median = float(np.median(edge_lengths))
    q25, q75 = np.percentile(edge_lengths, [25, 75])
    global_iqr = float(q75 - q25)
    safe_med = max(global_median, 1e-10)
    safe_iqr = max(global_iqr, 1e-10)

    feat = np.zeros((n_points, 7), dtype=np.float32)
    for node in range(n_points):
        inc = point_to_edges.get(node, [])
        if not inc:
            continue
        lengths = np.array([edge_lengths[ei] for ei in inc], dtype=np.float32)
        deg = len(lengths)
        mean_l = float(lengths.mean())
        feat[node, 0] = deg
        feat[node, 1] = mean_l / safe_med
        feat[node, 2] = float(lengths.std()) / safe_iqr
        feat[node, 3] = float(lengths.min()) / safe_med
        feat[node, 4] = float(lengths.max()) / safe_med
        feat[node, 5] = (mean_l - global_median) / safe_iqr
        feat[node, 6] = float((lengths < global_median).sum()) / deg
    return feat, global_median, global_iqr


def _extract_gnn_subgraph(target_ei, edge_keys, node_feat_base,
                          point_to_edges, max_hops=2):
    """Extract the 2-hop point subgraph used by the trained GNN."""
    a, b = edge_keys[target_ei]

    node_to_dist = {a: 0, b: 0}
    queue = [a, b]
    visited = set()
    while queue:
        node = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        dist = node_to_dist[node]
        if dist < max_hops:
            for nei_ei in point_to_edges.get(node, []):
                na, nb = edge_keys[nei_ei]
                neighbor = nb if na == node else na
                if neighbor not in node_to_dist:
                    node_to_dist[neighbor] = dist + 1
                    queue.append(neighbor)

    subgraph_nodes = sorted(node_to_dist.keys())
    node_to_sub = {n: i for i, n in enumerate(subgraph_nodes)}

    features = np.zeros((len(subgraph_nodes), 9), dtype=np.float32)
    endpoints = {a, b}
    for i, node in enumerate(subgraph_nodes):
        features[i, :7] = node_feat_base[node]
        features[i, 7] = node_to_dist[node]
        features[i, 8] = 1.0 if node in endpoints else 0.0

    edge_set = set()
    for node in subgraph_nodes:
        for ei in point_to_edges.get(node, []):
            na, nb = edge_keys[ei]
            if na in node_to_sub and nb in node_to_sub:
                u, v = node_to_sub[na], node_to_sub[nb]
                if u > v:
                    u, v = v, u
                edge_set.add((u, v))

    a_sub = node_to_sub[a]
    b_sub = node_to_sub[b]
    target_edge = (a_sub, b_sub) if a_sub < b_sub else (b_sub, a_sub)

    return (
        features,
        np.array(sorted(edge_set), dtype=np.int64) if edge_set else np.zeros((0, 2), dtype=np.int64),
        target_edge,
    )

    def _compute_mlp_global_feats(edge_lengths):
        """Dataset-level statistics used by the scalar-feature MLP."""
        from scipy.stats import skew as _skew

        med = float(np.median(edge_lengths))
        q25 = float(np.percentile(edge_lengths, 25))
        q75 = float(np.percentile(edge_lengths, 75))
        p10 = float(np.percentile(edge_lengths, 10))
        p90 = float(np.percentile(edge_lengths, 90))
        iqr = q75 - q25
        safe_med = max(med, 1e-10)
        safe_p10 = max(p10, 1e-10)
        return {
            "median": med,
            "iqr": iqr,
            "p90_over_p10": float(p90) / safe_p10,
            "skew": float(_skew(edge_lengths)),
            "frac_gt_2med": float((edge_lengths > 2 * med).mean()),
            "frac_gt_3med": float((edge_lengths > 3 * med).mean()),
            "safe_median": safe_med,
            "safe_iqr": max(iqr, 1e-10),
        }

    def _build_mlp_trusted_adjacency(edge_keys, edge_lengths):
        """Adjacency of trusted edges only (edges shorter than median)."""
        med = float(np.median(edge_lengths))
        trusted = edge_lengths <= med
        trusted_adj = _defaultdict(list)
        for ei, (a, b) in enumerate(edge_keys):
            if trusted[ei]:
                trusted_adj[a].append((b, ei))
                trusted_adj[b].append((a, ei))
        return trusted_adj

    def _compute_mlp_trusted_path_features(target_ei, edge_keys, trusted_adj, max_hops=5):
        """Trusted alternative-path features for one edge."""
        u, v = edge_keys[target_ei]

        def shortest_hops(start, target):
            q = deque([(start, 0)])
            visited = {start}
            while q:
                node, dist = q.popleft()
                if dist >= max_hops:
                    continue
                for nei, ei in trusted_adj.get(node, []):
                    if ei == target_ei:
                        continue
                    if nei == target:
                        return dist + 1
                    if nei not in visited:
                        visited.add(nei)
                        q.append((nei, dist + 1))
            return max_hops + 1

        def branch_count(start, target, cap=3):
            count = 0
            for nei, ei in trusted_adj.get(start, []):
                if ei == target_ei:
                    continue
                if nei == target:
                    count += 1
                else:
                    q = deque([(nei, 1)])
                    visited = {start, nei}
                    found = False
                    while q and not found:
                        node, dist = q.popleft()
                        if dist >= max_hops:
                            continue
                        for nxt, nxt_ei in trusted_adj.get(node, []):
                            if nxt_ei == target_ei:
                                continue
                            if nxt == target:
                                found = True
                                break
                            if nxt not in visited:
                                visited.add(nxt)
                                q.append((nxt, dist + 1))
                    if found:
                        count += 1
                if count >= cap:
                    return cap
            return count

        return np.array([
            float(shortest_hops(u, v)),
            float(shortest_hops(v, u)),
            float(branch_count(u, v)),
            float(branch_count(v, u)),
        ], dtype=np.float32)

    class _MLPEdgeClassifier(nn.Module):
        """Scalar-feature MLP used as an alternative to the GNN."""

        def __init__(self, n_feat, hidden=192, dropout=0.15):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_feat, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
                nn.Linear(hidden // 2, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(1)

    def _load_mlp_model(benchmark_dir='benchmark_clustering_tuning'):
        """Load the scalar-feature MLP checkpoint."""
        global _mlp_model_cache

        if _mlp_model_cache is not None:
            return _mlp_model_cache

        try:
            candidates = [
                Path(benchmark_dir) / 'gnn_experiments' / 'mlp_dualspace_results.pt',
                Path(benchmark_dir) / 'gnn_experiments' / 'mlp_results.pt',
                Path('gnn_experiments') / 'mlp_dualspace_results.pt',
                Path('gnn_experiments') / 'mlp_results.pt',
                Path('.') / 'gnn_experiments' / 'mlp_dualspace_results.pt',
                Path('.') / 'gnn_experiments' / 'mlp_results.pt',
            ]

            model_path = None
            for candidate in candidates:
                if candidate.exists():
                    model_path = candidate
                    break

            if model_path is None:
                return None, None

            checkpoint = torch.load(str(model_path), map_location='cpu', weights_only=False)
            model_state = checkpoint.get('model_state')
            if model_state is None:
                return None, None

            key = 'net.0.weight'
            if key not in model_state:
                print(f"Warning: unexpected MLP weights (missing '{key}')")
                return None, None

            n_feat = model_state[key].shape[1]
            hidden_dim = model_state[key].shape[0]

            model = _MLPEdgeClassifier(n_feat, hidden_dim)
            model.load_state_dict(model_state)
            model.eval()

            config = dict(checkpoint.get('config', {}))
            config['model_type'] = 'edge_mlp'
            config['checkpoint_path'] = str(model_path)
            config['scaler_mean'] = np.asarray(config.get('scaler_mean', []), dtype=np.float32)
            config['scaler_scale'] = np.asarray(config.get('scaler_scale', []), dtype=np.float32)

            _mlp_model_cache = (model, config)
            return model, config
        except Exception as e:
            print(f"Warning: Could not load MLP model: {e}")
            return None, None

    def _compute_mlp_edge_features(edge_idx, edge_keys, edge_lengths_orig, edge_lengths_proj,
                                   point_to_edges, dim, max_hops=2):
        """Build the scalar feature vector used by the MLP."""
        u, v = edge_keys[edge_idx]
        orig_stats = _compute_mlp_global_feats(edge_lengths_orig)
        proj_stats = _compute_mlp_global_feats(edge_lengths_proj)
        max_degree = max((len(vs) for vs in point_to_edges.values()), default=1)

        orig_len = float(edge_lengths_orig[edge_idx])
        proj_len = float(edge_lengths_proj[edge_idx])
        orig_med = max(orig_stats["median"], 1e-10)
        proj_med = max(proj_stats["median"], 1e-10)

        node_to_dist = {u: 0, v: 0}
        queue = [u, v]
        visited = set()
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            dist = node_to_dist[node]
            if dist < max_hops:
                for nei_ei in point_to_edges.get(node, []):
                    na, nb = edge_keys[nei_ei]
                    neighbor = nb if na == node else na
                    if neighbor not in node_to_dist:
                        node_to_dist[neighbor] = dist + 1
                        queue.append(neighbor)

        subgraph_edges = set()
        for node in visited:
            for ei in point_to_edges.get(node, []):
                na, nb = edge_keys[ei]
                if na in visited and nb in visited:
                    subgraph_edges.add(ei)

        direct_edges = set(point_to_edges.get(u, [])) | set(point_to_edges.get(v, []))
        direct_edges.discard(edge_idx)
        ring2_edges = subgraph_edges - direct_edges - {edge_idx}

        def _summarize(edge_ids, lengths, ref_len):
            if not edge_ids:
                return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            vals = np.asarray([lengths[ei] for ei in edge_ids], dtype=np.float32)
            return np.array([
                float(len(edge_ids)),
                float(vals.mean()),
                float(vals.std()),
                float(vals.min()),
                float(vals.max()),
                float(np.median(vals)),
                float((vals < ref_len).mean()),
            ], dtype=np.float32)

        trusted_adj_orig = _build_mlp_trusted_adjacency(edge_keys, edge_lengths_orig)
        trusted_adj_proj = _build_mlp_trusted_adjacency(edge_keys, edge_lengths_proj)
        trusted_path_orig = _compute_mlp_trusted_path_features(edge_idx, edge_keys, trusted_adj_orig)
        trusted_path_proj = _compute_mlp_trusted_path_features(edge_idx, edge_keys, trusted_adj_proj)

        deg_u = len(point_to_edges.get(u, []))
        deg_v = len(point_to_edges.get(v, []))
        deg_min = min(deg_u, deg_v)
        deg_max = max(deg_u, deg_v)
        deg_diff = abs(deg_u - deg_v)

        feature_vec = np.concatenate([
            np.array([
                float(np.log1p(dim)),
                float(np.log1p(len(edge_keys))),
                orig_stats["median"],
                orig_stats["iqr"],
                proj_stats["median"],
                proj_stats["iqr"],
                orig_stats["p90_over_p10"],
                proj_stats["p90_over_p10"],
                orig_stats["skew"],
                proj_stats["skew"],
                orig_stats["frac_gt_2med"],
                proj_stats["frac_gt_2med"],
                orig_len / orig_med,
                proj_len / proj_med,
                float(np.log(max(orig_len / orig_med, 1e-10))),
                float(np.log(max(proj_len / proj_med, 1e-10))),
                float(orig_len / max(proj_len, 1e-10)),
                float(deg_min / max(max_degree, 1)),
                float(deg_max / max(max_degree, 1)),
                float(deg_diff / max(max_degree, 1)),
            ], dtype=np.float32),
            _summarize(direct_edges, edge_lengths_orig, orig_len),
            _summarize(direct_edges, edge_lengths_proj, proj_len),
            _summarize(ring2_edges, edge_lengths_orig, orig_len),
            _summarize(ring2_edges, edge_lengths_proj, proj_len),
            trusted_path_orig,
            trusted_path_proj,
            np.array([
                float(len(subgraph_edges)),
                float(len(direct_edges)),
                float(len(ring2_edges)),
            ], dtype=np.float32),
        ]).astype(np.float32)
        return feature_vec


def _compute_mlp_global_feats(edge_lengths):
    """Dataset-level statistics used by the scalar-feature MLP."""
    from scipy.stats import skew as _skew

    med = float(np.median(edge_lengths))
    q25 = float(np.percentile(edge_lengths, 25))
    q75 = float(np.percentile(edge_lengths, 75))
    p10 = float(np.percentile(edge_lengths, 10))
    p90 = float(np.percentile(edge_lengths, 90))
    iqr = q75 - q25
    safe_med = max(med, 1e-10)
    safe_p10 = max(p10, 1e-10)
    return {
        "median": med,
        "iqr": iqr,
        "p90_over_p10": float(p90) / safe_p10,
        "skew": float(_skew(edge_lengths)),
        "frac_gt_2med": float((edge_lengths > 2 * med).mean()),
        "frac_gt_3med": float((edge_lengths > 3 * med).mean()),
        "safe_median": safe_med,
        "safe_iqr": max(iqr, 1e-10),
    }


def _build_mlp_trusted_adjacency(edge_keys, edge_lengths):
    """Adjacency of trusted edges only (edges shorter than median)."""
    med = float(np.median(edge_lengths))
    trusted = edge_lengths <= med
    trusted_adj = defaultdict(list)
    for ei, (a, b) in enumerate(edge_keys):
        if trusted[ei]:
            trusted_adj[a].append((b, ei))
            trusted_adj[b].append((a, ei))
    return trusted_adj


def _compute_mlp_trusted_path_features(target_ei, edge_keys, trusted_adj, max_hops=5):
    """Trusted alternative-path features for one edge."""
    u, v = edge_keys[target_ei]

    def shortest_hops(start, target):
        q = deque([(start, 0)])
        visited = {start}
        while q:
            node, dist = q.popleft()
            if dist >= max_hops:
                continue
            for nei, ei in trusted_adj.get(node, []):
                if ei == target_ei:
                    continue
                if nei == target:
                    return dist + 1
                if nei not in visited:
                    visited.add(nei)
                    q.append((nei, dist + 1))
        return max_hops + 1

    def branch_count(start, target, cap=3):
        count = 0
        for nei, ei in trusted_adj.get(start, []):
            if ei == target_ei:
                continue
            if nei == target:
                count += 1
            else:
                q = deque([(nei, 1)])
                visited = {start, nei}
                found = False
                while q and not found:
                    node, dist = q.popleft()
                    if dist >= max_hops:
                        continue
                    for nxt, nxt_ei in trusted_adj.get(node, []):
                        if nxt_ei == target_ei:
                            continue
                        if nxt == target:
                            found = True
                            break
                        if nxt not in visited:
                            visited.add(nxt)
                            q.append((nxt, dist + 1))
                if found:
                    count += 1
            if count >= cap:
                return cap
        return count

    return np.array([
        float(shortest_hops(u, v)),
        float(shortest_hops(v, u)),
        float(branch_count(u, v)),
        float(branch_count(v, u)),
    ], dtype=np.float32)


def _init_mlp_parallel_context(context):
    """Initialize per-process state for parallel MLP feature extraction."""
    global _mlp_parallel_context
    _mlp_parallel_context = context


def _compute_mlp_feature_batch_parallel(edge_ids):
    """Compute MLP features for a batch of edge ids inside a worker process."""
    context = _mlp_parallel_context
    if context is None:
        raise RuntimeError("Parallel MLP context was not initialized")
    return [(ei, _compute_mlp_edge_features(ei, context, max_hops=2)) for ei in edge_ids]


def _compute_mlp_bridge_feature_batch_parallel(edge_ids):
    """Compute bridge-MLP features for a batch of edge ids inside a worker process."""
    context = _mlp_parallel_context
    if context is None:
        raise RuntimeError("Parallel MLP context was not initialized")
    return [(ei, _compute_mlp_bridge_edge_features(ei, context, max_hops=2)) for ei in edge_ids]


def _compute_mlp_bridge_global_feats(edge_lengths):
    """Dataset-level statistics used by the bridge-focused MLP."""
    edge_lengths = np.asarray(edge_lengths, dtype=np.float32)
    if len(edge_lengths) == 0:
        return {
            "median": 0.0,
            "iqr": 0.0,
            "p05": 0.0,
            "p95": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "skew": 0.0,
            "kurtosis": 0.0,
            "safe_median": 1e-10,
            "safe_iqr": 1e-10,
        }

    med = float(np.median(edge_lengths))
    q25 = float(np.percentile(edge_lengths, 25))
    q75 = float(np.percentile(edge_lengths, 75))
    p05 = float(np.percentile(edge_lengths, 5))
    p95 = float(np.percentile(edge_lengths, 95))
    iqr = q75 - q25
    mean_val = float(np.mean(edge_lengths))
    std_val = float(np.std(edge_lengths))
    skew_val = float(skew(edge_lengths))
    if not np.isfinite(skew_val):
        skew_val = 0.0
    kurt_val = float(kurtosis(edge_lengths, fisher=True, bias=False))
    if not np.isfinite(kurt_val):
        kurt_val = 0.0
    return {
        "median": med,
        "iqr": iqr,
        "p05": p05,
        "p95": p95,
        "mean": mean_val,
        "std": std_val,
        "skew": skew_val,
        "kurtosis": kurt_val,
        "safe_median": max(med, 1e-10),
        "safe_iqr": max(iqr, 1e-10),
    }


def _compute_mlp_bridge_context(edge_keys, edge_lengths_orig, point_to_edges, dim):
    edge_lengths_orig = np.asarray(edge_lengths_orig, dtype=np.float32)
    edge_length_scale = max(float(np.median(edge_lengths_orig)), 1e-10)
    edge_lengths_norm = edge_lengths_orig / edge_length_scale
    orig_stats = _compute_mlp_bridge_global_feats(edge_lengths_norm)
    return {
        "edge_keys": edge_keys,
        "edge_lengths_orig": edge_lengths_orig,
        "edge_lengths_norm": edge_lengths_norm,
        "edge_length_scale": edge_length_scale,
        "point_to_edges": point_to_edges,
        "dim": dim,
        "dim_log1p": float(np.log1p(dim)),
        "orig_stats": orig_stats,
    }


def _relative_length_hist(lengths, target_len, bins):
    hist = np.zeros(bins + 1, dtype=np.float32)
    if len(lengths) == 0:
        return hist
    arr = np.asarray(lengths, dtype=np.float32)
    inner = arr[arr <= target_len]
    if len(inner):
        edges = np.linspace(0.0, max(target_len, 1e-10), bins + 1, dtype=np.float32)
        hist[:-1], _ = np.histogram(inner, bins=edges)
    hist[-1] = float((arr > target_len).sum())
    hist /= max(float(len(arr)), 1.0)
    return hist.astype(np.float32)


def _compute_mlp_bridge_edge_features(edge_idx, context, max_hops=2):
    """Build the bridge-focused feature vector used by the MLP."""
    edge_keys = context["edge_keys"]
    edge_lengths_norm = context["edge_lengths_norm"]
    point_to_edges = context["point_to_edges"]
    orig_stats = context["orig_stats"]
    dim_log1p = float(context["dim_log1p"])

    u, v = edge_keys[edge_idx]
    norm_len = float(edge_lengths_norm[edge_idx])

    node_to_dist = {u: 0, v: 0}
    queue = deque([u, v])
    visited = set()
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        dist = node_to_dist[node]
        if dist < max_hops:
            for nei_ei in point_to_edges.get(node, []):
                na, nb = edge_keys[nei_ei]
                neighbor = nb if na == node else na
                if neighbor not in node_to_dist:
                    node_to_dist[neighbor] = dist + 1
                    queue.append(neighbor)

    subgraph_edges = set()
    for node in visited:
        for ei in point_to_edges.get(node, []):
            na, nb = edge_keys[ei]
            if na in visited and nb in visited:
                subgraph_edges.add(ei)

    direct_edges = set(point_to_edges.get(u, [])) | set(point_to_edges.get(v, []))
    direct_edges.discard(edge_idx)
    ring2_edges = subgraph_edges - direct_edges - {edge_idx}

    direct_lengths = [edge_lengths_norm[ei] for ei in direct_edges]
    ring2_lengths = [edge_lengths_norm[ei] for ei in ring2_edges]

    feature_vec = np.concatenate([
        _relative_length_hist(direct_lengths, norm_len, bins=4),
        _relative_length_hist(ring2_lengths, norm_len, bins=6),
        np.array([
            orig_stats["median"],
            orig_stats["iqr"],
            orig_stats["p05"],
            orig_stats["p95"],
            orig_stats["mean"],
            orig_stats["std"],
            orig_stats["skew"],
            orig_stats["kurtosis"],
            norm_len,
            dim_log1p,
        ], dtype=np.float32),
    ]).astype(np.float32)
    return feature_vec


def _precompute_mlp_bridge_features(candidate_edges, context, feature_workers, batch_size):
    """
    Precompute bridge-MLP features for all candidate edges once before scoring.

    This keeps the feature extraction logic separate from thresholding and makes
    the bridge mode easier to keep in sync with the trained checkpoint.
    """
    candidate_edges = np.asarray(candidate_edges, dtype=np.int64)
    if len(candidate_edges) == 0:
        return candidate_edges, np.zeros((0, 0), dtype=np.float32), 0

    if feature_workers <= 1 or len(candidate_edges) < 2 * batch_size:
        errors = 0
        feats = [
            _compute_mlp_bridge_edge_features(ei, context, max_hops=2)
            for ei in candidate_edges
        ]
        return candidate_edges, np.asarray(feats, dtype=np.float32), errors

    chunk_size = max(batch_size, 256)
    chunks = [
        candidate_edges[start:start + chunk_size]
        for start in range(0, len(candidate_edges), chunk_size)
    ]
    print(f"  Using {feature_workers} workers for MLP bridge feature extraction...")

    feat_rows = []
    feat_edge_ids = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=feature_workers,
        mp_context=ctx,
        initializer=_init_mlp_parallel_context,
        initargs=(context,),
    ) as executor:
        try:
            for feat_batch in executor.map(_compute_mlp_bridge_feature_batch_parallel, chunks, chunksize=1):
                if not feat_batch:
                    continue
                feat_edge_ids.extend(ei for ei, _ in feat_batch)
                feat_rows.extend(feat for _, feat in feat_batch)
            return np.asarray(feat_edge_ids, dtype=np.int64), np.asarray(feat_rows, dtype=np.float32), 0
        except Exception as exc:
            print(f"  Parallel MLP bridge feature extraction failed; falling back to sequential mode: {exc}")

    feat_edge_ids = []
    feat_rows = []
    errors = 0
    for ei in candidate_edges:
        try:
            feat_edge_ids.append(ei)
            feat_rows.append(_compute_mlp_bridge_edge_features(ei, context, max_hops=2))
        except Exception:
            errors += 1
    return np.asarray(feat_edge_ids, dtype=np.int64), np.asarray(feat_rows, dtype=np.float32), errors


class _MLPEdgeClassifier(nn.Module):
    """Scalar-feature MLP used as an alternative to the GNN."""

    def __init__(self, n_feat, hidden=192, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def _load_mlp_model(benchmark_dir='benchmark_clustering_tuning'):
    """Load the scalar-feature MLP checkpoint."""
    global _mlp_model_cache

    if _mlp_model_cache is not None:
        return _mlp_model_cache

    try:
        candidates = [
            Path(benchmark_dir) / 'gnn_experiments' / 'mlp_dualspace_results.pt',
            Path(benchmark_dir) / 'gnn_experiments' / 'mlp_results.pt',
            Path('gnn_experiments') / 'mlp_dualspace_results.pt',
            Path('gnn_experiments') / 'mlp_results.pt',
            Path('.') / 'gnn_experiments' / 'mlp_dualspace_results.pt',
            Path('.') / 'gnn_experiments' / 'mlp_results.pt',
        ]

        model_path = None
        for candidate in candidates:
            if candidate.exists():
                model_path = candidate
                break

        if model_path is None:
            return None, None

        checkpoint = torch.load(str(model_path), map_location='cpu', weights_only=False)
        model_state = checkpoint.get('model_state')
        if model_state is None:
            return None, None

        key = 'net.0.weight'
        if key not in model_state:
            print(f"Warning: unexpected MLP weights (missing '{key}')")
            return None, None

        n_feat = model_state[key].shape[1]
        hidden_dim = model_state[key].shape[0]

        model = _MLPEdgeClassifier(n_feat, hidden_dim)
        model.load_state_dict(model_state)
        model.eval()

        config = dict(checkpoint.get('config', {}))
        config['model_type'] = 'edge_mlp'
        config['checkpoint_path'] = str(model_path)
        config['scaler_mean'] = np.asarray(config.get('scaler_mean', []), dtype=np.float32)
        config['scaler_scale'] = np.asarray(config.get('scaler_scale', []), dtype=np.float32)

        _mlp_model_cache = (model, config)
        return model, config
    except Exception as e:
        print(f"Warning: Could not load MLP model: {e}")
        return None, None


def _compute_mlp_edge_features(edge_idx, context, max_hops=2):
    """Build the scalar feature vector used by the MLP."""
    edge_keys = context["edge_keys"]
    edge_lengths_orig = context["edge_lengths_orig"]
    edge_lengths_proj = context["edge_lengths_proj"]
    point_to_edges = context["point_to_edges"]
    dim = context["dim"]
    orig_stats = context["orig_stats"]
    proj_stats = context["proj_stats"]
    trusted_adj_orig = context["trusted_adj_orig"]
    trusted_adj_proj = context["trusted_adj_proj"]
    max_degree = context["max_degree"]

    u, v = edge_keys[edge_idx]
    orig_len = float(edge_lengths_orig[edge_idx])
    proj_len = float(edge_lengths_proj[edge_idx])
    orig_med = max(orig_stats["median"], 1e-10)
    proj_med = max(proj_stats["median"], 1e-10)

    node_to_dist = {u: 0, v: 0}
    queue = [u, v]
    visited = set()
    while queue:
        node = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        dist = node_to_dist[node]
        if dist < max_hops:
            for nei_ei in point_to_edges.get(node, []):
                na, nb = edge_keys[nei_ei]
                neighbor = nb if na == node else na
                if neighbor not in node_to_dist:
                    node_to_dist[neighbor] = dist + 1
                    queue.append(neighbor)

    subgraph_edges = set()
    for node in visited:
        for ei in point_to_edges.get(node, []):
            na, nb = edge_keys[ei]
            if na in visited and nb in visited:
                subgraph_edges.add(ei)

    direct_edges = set(point_to_edges.get(u, [])) | set(point_to_edges.get(v, []))
    direct_edges.discard(edge_idx)
    ring2_edges = subgraph_edges - direct_edges - {edge_idx}

    def _summarize(edge_ids, lengths, ref_len):
        if not edge_ids:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        vals = np.asarray([lengths[ei] for ei in edge_ids], dtype=np.float32)
        return np.array([
            float(len(edge_ids)),
            float(vals.mean()),
            float(vals.std()),
            float(vals.min()),
            float(vals.max()),
            float(np.median(vals)),
            float((vals < ref_len).mean()),
        ], dtype=np.float32)

    trusted_path_orig = _compute_mlp_trusted_path_features(edge_idx, edge_keys, trusted_adj_orig)
    trusted_path_proj = _compute_mlp_trusted_path_features(edge_idx, edge_keys, trusted_adj_proj)

    deg_u = len(point_to_edges.get(u, []))
    deg_v = len(point_to_edges.get(v, []))
    deg_min = min(deg_u, deg_v)
    deg_max = max(deg_u, deg_v)
    deg_diff = abs(deg_u - deg_v)

    feature_vec = np.concatenate([
        np.array([
            float(np.log1p(dim)),
            float(np.log1p(len(edge_keys))),
            orig_stats["median"],
            orig_stats["iqr"],
            proj_stats["median"],
            proj_stats["iqr"],
            orig_stats["p90_over_p10"],
            proj_stats["p90_over_p10"],
            orig_stats["skew"],
            proj_stats["skew"],
            orig_stats["frac_gt_2med"],
            proj_stats["frac_gt_2med"],
            orig_len / orig_med,
            proj_len / proj_med,
            float(np.log(max(orig_len / orig_med, 1e-10))),
            float(np.log(max(proj_len / proj_med, 1e-10))),
            float(orig_len / max(proj_len, 1e-10)),
            float(deg_min / max(max_degree, 1)),
            float(deg_max / max(max_degree, 1)),
            float(deg_diff / max(max_degree, 1)),
        ], dtype=np.float32),
        _summarize(direct_edges, edge_lengths_orig, orig_len),
        _summarize(direct_edges, edge_lengths_proj, proj_len),
        _summarize(ring2_edges, edge_lengths_orig, orig_len),
        _summarize(ring2_edges, edge_lengths_proj, proj_len),
        trusted_path_orig,
        trusted_path_proj,
        np.array([
            float(len(subgraph_edges)),
            float(len(direct_edges)),
            float(len(ring2_edges)),
        ], dtype=np.float32),
    ]).astype(np.float32)
    return feature_vec


def _load_combined_baseline_weights():
    """Load the saved interpretable baseline weights used by stage plots."""
    global _combined_baseline_weights_cache
    if _combined_baseline_weights_cache is not None:
        return _combined_baseline_weights_cache

    candidates = [
        Path(__file__).resolve().parent / "benchmark_clustering_tuning" / "results" / "feature_selection" / "combined_feature_analysis.json",
        Path("benchmark_clustering_tuning") / "results" / "feature_selection" / "combined_feature_analysis.json",
    ]
    weight_path = next((path for path in candidates if path.exists()), None)
    if weight_path is None:
        raise FileNotFoundError(
            "Combined baseline weights not found. Expected "
            "benchmark_clustering_tuning/results/feature_selection/combined_feature_analysis.json"
        )

    with weight_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    combined = payload["combined_features"]
    _combined_baseline_weights_cache = {
        "keep_risk": {str(k): float(v) for k, v in combined["keep_purity_score"]["weights"].items()},
        "prune_score": {str(k): float(v) for k, v in combined["prune_recall_score"]["weights"].items()},
        "path": str(weight_path),
    }
    return _combined_baseline_weights_cache


def _combined_baseline_mask(
    edge_keys,
    X,
    edge_lengths_orig,
    candidate_edges,
    score_name="prune_score",
    threshold=0.5,
    score_weights=None,
):
    """Score candidate edges with a configurable circular-feature combination.

    Parameters
    ----------
    score_name : str
        Saved default to use when ``score_weights`` is omitted.  ``prune_score``
        is the learned high-PRUNE-rate combination; ``keep_risk`` is the
        learned high-KEEP-purity combination.
    score_weights : dict, optional
        Explicit ``{feature_name: weight}`` combination.  This overrides the
        JSON defaults and may contain any supported short/long circular
        feature, including ``ring1_short_C3_real`` or ``ring2_long_M2``.
        Length terms ``candidate_L_rel_left`` and
        ``candidate_L_rel_right`` use the lower and upper semi-IQRs,
        respectively, instead of the full IQR.
        Features not present in this mapping have weight zero.

    Learned defaults (also stored in
    ``results/feature_selection/combined_feature_analysis.json``):

      * ``keep_risk``: 0.2585846966*ring1_short_M1
        + 0.0306646758*ring1_long_M3
        + 0.5137954794*ring2_long_M1
        + 0.1969551483*candidate_L_rel
      * ``prune_score``: 0.1797709680*ring1_short_M1
        + 0.8202290320*candidate_L_rel

    The score is additive; it is not normalized or passed through a softmax.
    """

    if score_weights is None and score_name not in {"keep_risk", "prune_score"}:
        raise ValueError("baseline_score must be 'keep_risk' or 'prune_score'")

    # Complete supported score vocabulary.  Zero defaults make it possible to
    # override only one or two terms without having to spell out all features.
    circular_terms = (
        "edge_count", "mass", "mean_weight",
        "C1_real", "C1_imag", "C2_real", "C2_imag",
        "C3_real", "C3_imag", "C4_real", "C4_imag",
    )
    supported_features = {
        "candidate_L_rel",
        "candidate_L_rel_left",
        "candidate_L_rel_right",
    }
    for ring in ("ring1", "ring2"):
        for mode in ("short", "long"):
            supported_features.update(
                f"{ring}_{mode}_{term}" for term in circular_terms
            )
            supported_features.update(
                f"{ring}_{mode}_M{moment}" for moment in range(1, 5)
            )

    if score_weights is None:
        weights = dict(_load_combined_baseline_weights()[score_name])
    else:
        if not hasattr(score_weights, "items"):
            raise TypeError("score_weights must be a mapping of feature names to weights")
        unknown = set(score_weights) - supported_features
        if unknown:
            raise ValueError(
                "Unknown combined baseline feature(s): "
                + ", ".join(sorted(map(str, unknown)))
            )
        weights = {str(name): float(weight) for name, weight in score_weights.items()}
        if not all(np.isfinite(weight) for weight in weights.values()):
            raise ValueError("score_weights must contain only finite numeric values")

    feature_dir = Path(__file__).resolve().parent / "benchmark_clustering_tuning" / "gnn_experiments"
    if str(feature_dir) not in sys.path:
        sys.path.insert(0, str(feature_dir))
    from mlp_feature_utils import (  # pylint: disable=import-outside-toplevel
        _circular_long_features,
        _true_edge_shells,
        build_point_to_edges,
        compute_edge_features,
        make_dataset_context,
    )

    edge_lengths_orig = np.asarray(edge_lengths_orig, dtype=np.float32)
    point_to_edges = build_point_to_edges(edge_keys)
    context = make_dataset_context(
        edge_keys=edge_keys,
        edge_lengths_orig=edge_lengths_orig,
        point_to_edges=point_to_edges,
        dim=X.shape[1],
        X_orig=X,
    )
    scores = np.zeros(len(edge_keys), dtype=np.float32)
    pruned = np.zeros(len(edge_keys), dtype=bool)
    errors = 0
    iqr_norm = float(context["orig_stats"]["iqr"])
    mild_limit = 1.0 - 0.5 * iqr_norm
    hard_limit = 1.0 + 1.5 * iqr_norm
    q25_norm, median_norm, q75_norm = np.percentile(
        context["edge_lengths_norm"], [25, 50, 75]
    )
    left_spread = max(float(median_norm - q25_norm), 1e-8)
    right_spread = max(float(q75_norm - median_norm), 1e-8)

    for edge_idx in np.asarray(candidate_edges, dtype=np.int64):
        try:
            short_features = compute_edge_features(
                int(edge_idx), context, circular_feature_mode="circular_v3"
            )
            ring1, ring2 = _true_edge_shells(int(edge_idx), edge_keys, point_to_edges)
            long_features = _circular_long_features(int(edge_idx), ring1, ring2, context)
            values = {}
            for ring_offset, ring in enumerate(("ring1", "ring2")):
                short_base = 22 + ring_offset * 11
                long_base = ring_offset * 11
                for local_offset, term in enumerate(circular_terms):
                    values[f"{ring}_short_{term}"] = float(
                        short_features[short_base + local_offset]
                    )
                    values[f"{ring}_long_{term}"] = float(
                        long_features[long_base + local_offset]
                    )
                for moment in range(1, 5):
                    short_real = values[f"{ring}_short_C{moment}_real"]
                    short_imag = values[f"{ring}_short_C{moment}_imag"]
                    long_real = values[f"{ring}_long_C{moment}_real"]
                    long_imag = values[f"{ring}_long_C{moment}_imag"]
                    values[f"{ring}_short_M{moment}"] = float(
                        np.hypot(short_real, short_imag)
                    )
                    values[f"{ring}_long_M{moment}"] = float(
                        np.hypot(long_real, long_imag)
                    )
            target_length = float(context["edge_lengths_norm"][edge_idx])
            values["candidate_L_rel"] = float(
                np.clip(
                    (target_length - mild_limit) / max(hard_limit - mild_limit, 1e-12),
                    0.0,
                    1.0,
                )
            )
            values["candidate_L_rel_left"] = float(
                np.clip(
                    (target_length - (median_norm - 0.5 * left_spread))
                    / max(2.0 * left_spread, 1e-12),
                    0.0,
                    1.0,
                )
            )
            values["candidate_L_rel_right"] = float(
                np.clip(
                    (target_length - (median_norm - 0.5 * right_spread))
                    / max(2.0 * right_spread, 1e-12),
                    0.0,
                    1.0,
                )
            )
            score = sum(values[name] * weight for name, weight in weights.items())
            scores[edge_idx] = score
            pruned[edge_idx] = score >= float(threshold)
        except Exception:
            errors += 1

    if errors:
        print(f"Combined baseline feature errors: {errors} (fallback: keep)")
    return pruned, scores, errors


def sigma_prune_triangles(triangles, sizes, proj_sizes, sigma_factor,
                          prune_mode='neighbor_bridge', X_proj=None, X=None,
                          hard_limit=None, mild_limit=None,
                          neighbor_stat='median', projected_hard_limit=True,
                          multiplicative_hard_limit=False,
                          auto_threshold=False,
                          prune_regime='none',
                          pointwise_bridge_pruning=False,
                          point_edge_ratio=1.5,
                          point_edge_compare='ratio',
                          point_edge_subset='all',
                          point_edge_gap=0.5,
                          bridge_chain_len=4,
                          mlp_feature_workers=1,
                          baseline=False,
                          baseline_score='prune_score',
                          baseline_threshold=0.5,
                          mild_power=2.0,
                          nuclei_rule_a=0.8,
                          nuclei_min_edges=20,
                          nuclei_quantile=0.8,
                          nuclei_type='red'):
    """Prune triangles based on edge lengths.

    Parameters
    ----------
    prune_mode : str
        'orig_only' — use only original-space distances (no projected-space check).
        'iqr_max'   — normalize both spaces to common IQR scale, then use
                       np.maximum as safety guard (default).
        'edge_center' — two-phase pruning: a hard limit removes all edges
                         above a high percentile, then a milder limit removes
                         edges whose centres are not surrounded by similarly
                         sized edges (bridge detection via edge-centre density).
        'pointwise_dual_space' — separate dual-space mode: hard/low thresholds
                         are applied independently in original and projected
                         space, then per-edge local comparisons are performed
                         in both spaces and the pruning marks are unioned.
        'tree_rules' — decision tree-based pruning using high-purity rules from
                         full-features tree. Applies 6 decision rules (85.8%-98.7%
                         purity) to classify edges. Set point_edge_compare to a
                         tuple of rule numbers (1-6) to select which rules apply.
                         Default: (1,2,3,4,5,6) uses all rules. Example:
                         point_edge_compare=(5,6) uses only highest-confidence
                         rules (98.7% and 93.1% purity).
    X_proj : ndarray of shape (n_points, 2) or None
        Projected coordinates.  Required when prune_mode='edge_center' or
        'tree_rules'.
    X : ndarray of shape (n_points, n_features) or None
        Original coordinates.  Used by prune_mode='neighbor_bridge' or 'tree_rules'
        so edge length pruning runs in original space while the Delaunay graph
        still comes from the projected geometry.
    hard_limit, mild_limit : float or None
        IQR multipliers for prune_mode='neighbor_bridge'.  The hard threshold
        is center + sigma_factor * hard_limit * IQR.  The mild threshold uses
        the relative-IQR power described below.  When omitted, the original
        hard-threshold defaults are used.
    mild_power : float
        Power applied to the dimensionless relative IQR when computing the
        mild threshold.  The default ``2.0`` uses
        ``center * (1 + sigma_factor * mild_limit *
        (IQR / center)**mild_power)``.  This keeps the threshold scale-aware
        while protecting narrow edge distributions more strongly.  The hard
        threshold remains linear in IQR.
    neighbor_stat : {'median', 'mean'}
        Center statistic for prune_mode='neighbor_bridge'.
    projected_hard_limit : bool
        If True, apply the same hard-limit thresholding to projected-space
        edge lengths as an additional pruning gate.
    multiplicative_hard_limit : bool
        If True, use a multiplicative threshold around the robust center
        instead of the default additive IQR-based threshold.
    auto_threshold : bool  [EXPERIMENTAL]
        If True (and prune_mode='neighbor_bridge'), attempt to detect the
        hard/mild thresholds automatically via KDE gap detection.  Falls
        back to the standard formula when no clear gap is found.
    prune_regime : {'none', 'low', 'high', 'auto'}
        Optional additive two-regime selector (ablation-friendly):
        - low  -> force prune_param=0.5
        - high -> force prune_param=1.5
        - auto -> infer low/high from the learned tree
        - none -> use sigma_factor as provided
    pointwise_bridge_pruning : bool
        If True, use the new per-point "largest edge vs shortest edge"
        pruning rule in phase 2. If False, preserve the original
        neighbor-length-ratio behavior.
    point_edge_ratio : float
        Phase-2 pruning threshold.  For each point, an incident edge becomes
        eligible for pruning if it is at least this factor larger than the
        shortest surviving incident edge at that point, and it is above the
        mild threshold.
    point_edge_compare : {'ratio', 'sigma', 'gnn', 'mlp', 'bridges', 'nuclei_rule'}
        Local phase-2 comparison rule.  'ratio' keeps the current
        average-multiplier test.  'sigma' prunes when
        current_edge > average + point_edge_ratio * std(neighborhood).
        'gnn' and 'bridges' select alternate edge-classification heuristics
        in pointwise_dual_space mode.
    point_edge_subset : {'all', 'short_half', 'shorter', 'trim2'}
        Neighborhood subset used for local average/std computation in
        pointwise_dual_space mode. 'all' uses the full 2-hop neighborhood;
        'short_half' uses only the shorter half of neighborhood edges;
        'shorter' uses only edges shorter than the current edge and preserves
        the edge when fewer than 2 such neighbors exist;
        'trim2' removes the two longest neighborhood edges and preserves the
        edge when fewer than 2 edges remain.
    bridge_chain_len : {3, 4}
        Minimum number of consecutive bridge edges in a roughly collinear
        chain required to protect the chain from pruning in bridge mode.
    mlp_feature_workers : int
        Optional number of worker processes to use for scalar-feature MLP
        edge feature extraction.  ``1`` keeps the current sequential path.
    nuclei_rule_a : float
        For ``point_edge_compare='nuclei_rule'``, the threshold ``A`` in
        ``L / median(edges_above_hard_limit) > A``.
    nuclei_min_edges : int
        Minimum number of mild-limit edges required for a connected nucleus
        used by ``point_edge_compare='nuclei_rule'``.
    nuclei_quantile : {0.70, 0.75, 0.80}
        Quantile used for the three-hop nucleus reference edge length.
    nuclei_type : {'red', 'blue'}
        Use mild-edge connected components (red) or local density peaks
        (blue) as nuclei.
    """
    mild_power = float(mild_power)
    if not np.isfinite(mild_power) or mild_power < 0.0:
        raise ValueError("mild_power must be a finite non-negative number")

    if prune_mode == 'orig_only':
        # Only original-space distances
        q25, q50, q75 = np.percentile(sizes, [25, 50, 75])
        iqr = q75 - q25
        if iqr == 0:
            iqr = 1e-10
        sizes_z = (sizes - q50) / iqr
        mean_size = np.mean(sizes_z)
        sigma = np.std(sizes_z)
        threshold = mean_size + sigma_factor * sigma
        kept_triangles = triangles[sizes_z <= threshold]
        rm_triangles = triangles[sizes_z > threshold]

    elif prune_mode == 'iqr_max':
        # IQR normalization brings both distributions to common scale
        # (median=0, IQR=1) so np.maximum is meaningful as a safety guard
        q25_o, q50_o, q75_o = np.percentile(sizes, [25, 50, 75])
        q25_p, q50_p, q75_p = np.percentile(proj_sizes, [25, 50, 75])
        iqr_o = q75_o - q25_o
        iqr_p = q75_p - q25_p
        if iqr_o == 0:
            iqr_o = 1e-10
        if iqr_p == 0:
            iqr_p = 1e-10
        sizes_z = (sizes - q50_o) / iqr_o
        proj_sizes_z = (proj_sizes - q50_p) / iqr_p

        mean_size = np.mean(sizes_z)
        sigma = np.std(sizes_z)
        threshold = mean_size + sigma_factor * sigma

        combined = np.maximum(sizes_z, proj_sizes_z)
        kept_triangles = triangles[combined <= threshold]
        rm_triangles = triangles[combined > threshold]

    elif prune_mode == 'edge_center':
        # Two-phase pruning with edge-centre density check.
        #
        # Phase 1 — hard limit: every edge whose length exceeds
        #   hard_limit = percentile(edge_sizes, hard_pct)
        # is unconditionally pruned.
        #
        # Phase 2 — contextual mild limit: edges whose length is between
        #   mild_limit and hard_limit are pruned only if their centre is
        #   *not* surrounded by other edges of similar size, i.e. they are
        #   likely bridge/gap edges rather than intra-cluster edges in a
        #   sparse region.
        #
        # sigma_factor controls overall aggressiveness:
        #   0.0 = very gentle, 1.0 = very aggressive.
        # We map it to the hard/mild percentiles and the density threshold.
        if X_proj is None:
            raise ValueError("prune_mode='edge_center' requires X_proj")

        edge_keys, edge_sizes, edge_centers, _ = _extract_edge_data(
            triangles, X_proj)

        # Map sigma_factor to percentiles.
        # Calibrated to give robust clustering across varying-density data:
        #   sf=0.3 → hard=93, mild=87 (gentle)
        #   sf=0.7 → hard=90, mild=82 (default, robust)
        #   sf=1.0 → hard=88, mild=79 (aggressive)
        #   sf=1.5 → hard=85, mild=74 (very aggressive)
        # Allow mildly negative prune factors for aggressive pruning sweeps.
        sf = float(sigma_factor)
        hard_pct = 95.0 - 7.0 * sf
        mild_pct = 90.0 - 11.0 * sf

        hard_limit = np.percentile(edge_sizes, hard_pct)
        mild_limit = np.percentile(edge_sizes, mild_pct)

        # Edge-centre density: fraction of k-NN with similar size
        k_nn = min(10, len(edge_sizes) - 1)
        if k_nn < 1:
            # Degenerate: too few edges
            pruned_edge_keys = set()
        else:
            frac_sim = _edge_center_density(edge_sizes, edge_centers, k=k_nn)

            # Density threshold: edges with frac_sim below this are
            # considered "isolated" and pruned (if above mild_limit).
            # Calibrated so that sf=0.7 gives frac_thresh≈0.24
            frac_thresh = 0.10 + 0.20 * sf

            pruned_mask = (
                (edge_sizes > hard_limit) |
                ((edge_sizes > mild_limit) & (edge_sizes <= hard_limit) &
                 (frac_sim < frac_thresh))
            )
            pruned_edge_keys = set(edge_keys[i] for i in
                                   np.where(pruned_mask)[0])

        # Build kept/pruned edge sets
        all_edge_set = set()
        for tri in triangles:
            for a, b in _simplex_edges(list(tri)):
                all_edge_set.add((min(a, b), max(a, b)))

        edges = all_edge_set - pruned_edge_keys
        rm_edges = pruned_edge_keys
        threshold = hard_limit

        return list(edges), list(rm_edges), threshold

    elif prune_mode == 'pointwise_dual_space':
        # Separate dual-space pointwise pruning with optional tree-based or GNN methods.
        #
        # Five sub-methods available via point_edge_compare parameter:
        #   point_edge_compare='sigma'     → Use _eligible_mask (original)
        #   point_edge_compare='stat'      → Use _eligible_mask_stat
        #   point_edge_compare=tuple       → Use _eligible_mask_tree with selected rules
        #   point_edge_compare='gnn'       → Use _eligible_mask_gnn (Graph Neural Network)
        #   point_edge_compare='mlp'       → Use _eligible_mask_mlp (scalar-feature MLP)
        #   point_edge_compare='bridges'   → Use _eligible_mask_bridges (heuristic bridge pruning)
        #   point_edge_compare='nuclei_rule' → Use _eligible_mask_nuclei_rule
        #                                      (global tail OR nearest-nucleus q80)
        #
        # For tree-based method (tuple), point_edge_compare can be:
        #   (1,2,3,4,5,6) - All 6 rules (balanced, 19.5% coverage, 91.8% purity)
        #   (5,6) - Only strongest rules (7.1% coverage, 95.9% purity)
        #   (3,5,6) - Only PRUNE rules
        #   (1,2,4) - Only KEEP rules
        #
        # For learned classifiers:
        #   point_edge_compare='gnn' - Uses trained GNN for gray-zone classification
        #   point_edge_compare='mlp' - Uses scalar-feature MLP for gray-zone classification
        #   Both build features from the full Delaunay graph, matching training
        #   Requires PyTorch and a checkpoint in benchmark_clustering_tuning/gnn_experiments/
        #
        # Candidate "next-gen default" reference setup (kept here so we do
        # not lose the exact working configuration from benchmark tuning):
        #   prune_mode='pointwise_dual_space'
        #   point_edge_compare='sigma'
        #   point_edge_ratio=0.7
        #   prune_param=1.1
        #   hard_limit=1.0
        #   mild_limit=-1.0
        #   neighbor_stat='median'
        #   prune_regime='none'
        #   projected_hard_limit=True
        #   anomaly_sensitivity=0.1
        #   point_edge_subset='all'
        #   project_dim=2
        #   umap_n_epochs=100
        #
        # Third alternative (trim2 with phase 3 bridge removal - best performer):
        #   prune_mode='pointwise_dual_space'
        #   point_edge_compare='sigma'
        #   point_edge_ratio=1.2
        #   prune_param=1.1
        #   hard_limit=1.0
        #   mild_limit=0.0
        #   neighbor_stat='median'
        #   prune_regime='none'
        #   projected_hard_limit=True
        #   anomaly_sensitivity=0.1
        #   point_edge_subset='trim2'
        #   project_dim=2
        #   umap_n_epochs=100
        #
        # Phase 1: compute independent hard/low thresholds in original and
        # projected space.  Edges above the hard threshold in either space are
        # removed immediately.  Edges at or below the low threshold in either
        # space are protected from pruning.
        #
        # Phase 2: for the remaining edges, inspect all incident edges at both
        # endpoints, then expand that neighborhood by one hop through those
        # neighbors.  For each edge, average the extended local set and prune
        # when the current edge exceeds a threshold adapted by the upper-tail
        # narrowness of the current edge-size distribution.
        #
        # Phase 3: Remove degree-2 bridges (edges where both endpoints have
        # exactly 1 other incident edge), which typically connect isolated clusters.
        #
        # The final prune set currently uses only the original-space result.
        # The projected-space branch is kept here for later reuse.
        from collections import defaultdict as _defaultdict
        if X_proj is None:
            raise ValueError("prune_mode='pointwise_dual_space' requires X_proj")
        if X is None:
            raise ValueError("prune_mode='pointwise_dual_space' requires X")

        edge_keys, orig_edge_sizes, _, _ = _extract_edge_data(triangles, X_proj, X_size=X)
        _, proj_edge_sizes, _, _ = _extract_edge_data(triangles, X_proj)
        n_edges = len(edge_keys)
        sf = float(sigma_factor)
        hard_mult = 1.5 if hard_limit in (None, "auto") else hard_limit
        mild_mult = 0.0 if mild_limit in (None, "auto") else mild_limit
        print(sf, hard_limit, mild_limit)

        def _thresholds(edge_sizes):
            q25, median, q75 = np.percentile(edge_sizes, [25, 50, 75])
            iqr = q75 - q25
            if iqr == 0:
                iqr = 1e-10
            if neighbor_stat == "median":
                center = median
            elif neighbor_stat == "mean":
                center = float(np.mean(edge_sizes))
            else:
                raise ValueError("neighbor_stat must be 'median' or 'mean'")
            if multiplicative_hard_limit:
                hard_thr = center * (1.0 + sf * hard_mult)
            else:
                hard_thr = center + sf * hard_mult * iqr
            relative_iqr = iqr / max(abs(center), 1e-10)
            mild_thr = center * (
                1.0 + sf * mild_mult * relative_iqr ** float(mild_power)
            )
            return hard_thr, mild_thr

        hard_o, mild_o = _thresholds(orig_edge_sizes)
        hard_p, mild_p = _thresholds(proj_edge_sizes)
        def _narrowness(edge_sizes):
            _, median, p90 = np.percentile(edge_sizes, [25, 50, 90])
            center = max(abs(float(median)), 1e-10)
            w = (float(p90) - float(median)) / center
            return max(w, 1e-3)

        point_to_edges = {}
        for ei, (a, b) in enumerate(edge_keys):
            point_to_edges.setdefault(a, []).append(ei)
            point_to_edges.setdefault(b, []).append(ei)

        def _eligible_mask(edge_sizes, hard_thr, mild_thr):
            pruned = np.zeros(n_edges, dtype=bool)

            POINT_EDGE_HOPS = 2

            def classify(length):
                if length <= mild_thr:
                    return 0      # SHORT
                elif length > hard_thr:
                    return 2      # LONG
                return 1          # MODERATE

            edge_class = np.array([classify(x) for x in edge_sizes], dtype=np.int8)

            for ei in range(n_edges):

                size = edge_sizes[ei]

                # only moderate edges participate
                if size <= mild_thr:
                    continue

                if size > hard_thr:
                    continue

                a, b = edge_keys[ei]

                first_hop = (
                    set(point_to_edges.get(a, []))
                    | set(point_to_edges.get(b, []))
                )
                first_hop.discard(ei)

                if not first_hop:
                    continue

                first_cls = edge_class[list(first_hop)]

                first_short = np.mean(first_cls == 0)
                first_long = np.mean(first_cls == 2)

                second_short = 0.0
                second_long = 0.0

                if POINT_EDGE_HOPS >= 2:

                    second_hop = set()

                    for nei in first_hop:
                        na, nb = edge_keys[nei]
                        second_hop.update(point_to_edges.get(na, []))
                        second_hop.update(point_to_edges.get(nb, []))

                    second_hop.difference_update(first_hop)
                    second_hop.discard(ei)

                    if second_hop:
                        second_cls = edge_class[list(second_hop)]
                        second_short = np.mean(second_cls == 0)
                        second_long = np.mean(second_cls == 2)

                #
                # Decision tree
                #
                
                FIRST_SHORT = 0.8
                FIRST_LONG = 0.30
                SECOND_SHORT = 0.40

                # clearly inside dense cluster
                if first_long >= 0.5:
                    pruned[ei] = True
                    continue

                if first_short >= 0.6:
                    continue

                # mostly surrounded by long edges
                if first_long >= 0.2 and second_long>=0.2:
                    pruned[ei] = True
                    continue

                # dense region exists nearby -> likely bridge
                if second_short >= 0.4 and first_short < 0.2:
                    pruned[ei] = True
                    continue

                # otherwise preserve
                continue

            return pruned

        def _eligible_mask_stat(edge_sizes, hard_thr, mild_thr):
            pruned = np.zeros(n_edges, dtype=bool)
            threshold_factor = float(point_edge_ratio)
            compare_mode = str(point_edge_compare).lower()
            subset_mode = str(point_edge_subset).lower()
            if subset_mode not in {"all", "short_half", "shorter", "trim2"}:
                raise ValueError("point_edge_subset must be 'all', 'short_half', 'shorter', or 'trim2'")
            for ei in range(n_edges):
                size = edge_sizes[ei]
                if size <= mild_thr:
                    continue
                if size > hard_thr:
                    continue

                a, b = edge_keys[ei]
                first_hop = set(point_to_edges.get(a, [])) | set(point_to_edges.get(b, []))
                first_hop.discard(ei)
                neighborhood = set(first_hop)

                POINT_EDGE_HOPS=1
                if POINT_EDGE_HOPS >= 2:
                    for nei in first_hop:
                        na, nb = edge_keys[nei]
                        neighborhood.update(point_to_edges.get(na, []))
                        neighborhood.update(point_to_edges.get(nb, []))
                neighborhood.discard(ei)
                if not neighborhood:
                    continue

                neighborhood_idx = np.array(sorted(neighborhood), dtype=np.intp)
                neighborhood_sizes = edge_sizes[neighborhood_idx]
                if len(neighborhood_sizes) == 0:
                    continue
                stats_base = neighborhood_sizes
                if subset_mode == "short_half":
                    sorted_sizes = np.sort(neighborhood_sizes)
                    half_n = max(1, len(sorted_sizes) // 2)
                    stats_base = sorted_sizes[:half_n]
                elif subset_mode == "shorter":
                    stats_base = neighborhood_sizes[neighborhood_sizes < size]
                    if len(stats_base) < 2:
                        continue
                elif subset_mode == "trim2":
                    sorted_sizes = np.sort(neighborhood_sizes)
                    if len(sorted_sizes) <= 2:
                        continue
                    stats_base = sorted_sizes[:-2]
                avg_neigh = float(np.mean(stats_base))
                spread = float(np.std(stats_base))
                if compare_mode == 'sigma':
                    should_prune = edge_sizes[ei] > avg_neigh + threshold_factor * spread
                else:
                    should_prune = edge_sizes[ei] >= avg_neigh * threshold_factor
                if should_prune:
                    pruned[ei] = True
            return pruned

        def _eligible_mask_tree(use_rules=(1, 2, 3, 4, 5, 6)):
            """
            Classify edges using high-purity decision tree rules.
            Tree trained on 12,404 edge samples from full Delaunay graphs.
            
            Returns boolean mask: True = keep, False = prune.
            Supports selective rule application via use_rules tuple.
            
            6 rules extracted:
              Rule 1: Short edge, uniform first-hop → 92.4% purity (KEEP)
              Rule 2: Sparse with high 2nd-hop diversity → 92.1% purity (KEEP)
              Rule 3: Dense, uniform 2nd-hop → 85.8% purity (PRUNE)
              Rule 4: Dense but diverse 2nd-hop → 88.7% purity (KEEP)
              Rule 5: Outlier edge, normal endpoints → 93.1% purity (PRUNE)
              Rule 6: Extreme outlier → 98.7% purity (PRUNE)
            """
            pruned = np.zeros(n_edges, dtype=bool)
            stats = [0,0,0,0,0,0]

            for ei in range(n_edges):
                features = _compute_tree_features(ei, edge_keys, orig_edge_sizes, point_to_edges)
                
                # Apply rules in priority order
                decision = 'PRUNE'
                
                # High-confidence prune rules
                if 6 in use_rules and _rule6_prune(features):
                    decision = 'PRUNE'
                    stats[5]+=1
                elif 5 in use_rules and _rule5_prune(features):
                    decision = 'PRUNE'
                    stats[4]+=1
                
                # High-confidence keep rules
                elif 1 in use_rules and _rule1_keep(features):
                    decision = 'KEEP'
                    stats[0]+=1
                elif 2 in use_rules and _rule2_keep(features):
                    decision = 'KEEP'
                    stats[1]+=1

                # Medium-confidence prune rule
                elif 3 in use_rules and _rule3_prune(features):
                    decision = 'PRUNE'
                    stats[2]+=1

                # Medium-confidence keep rule
                elif 4 in use_rules and _rule4_keep(features):
                    decision = 'KEEP'
                    stats[3]+=1

                if decision == 'PRUNE':
                    pruned[ei] = True
            print("Rule stats:", stats)
            return pruned  # True = keep, False = prune

        def _eligible_mask_gnn():
            """
            Classify edges using the trained Graph Neural Network.
            The GNN was trained on full-graph 2-hop point subgraphs with
            only lightweight node features, so inference must build the same
            subgraphs from the unfiltered Delaunay graph.
            
            Only gray-zone edges are scored here:
            edges outside both ``hard_union`` and ``protected_union``.
            
            Returns boolean mask: True = keep, False = prune.
            
            Performance:
              Accuracy: 81.04%
              F1-score: 0.8043
            """
            if not _gnn_available:
                raise RuntimeError("PyTorch not available. Cannot use GNN mode. Install: pip install torch")
            
            model, config = _load_gnn_model()
            if model is None:
                raise RuntimeError("Could not load trained GNN model from benchmark_clustering_tuning/gnn_experiments/gnn_results.pt")
            
            import time
            start_time = time.time()

            pruned = np.zeros(n_edges, dtype=bool)
            batch_size = 64

            # ------------------------------------------------------------------
            # Precompute once from the full graph: node features use all incident
            # edges, exactly as during training.
            # ------------------------------------------------------------------
        
            print("Precomputing GNN node features from full graph...")
            t0 = time.time()

            point_to_edges_gnn = _defaultdict(list)
            for ei, (a, b) in enumerate(edge_keys):
                point_to_edges_gnn[a].append(ei)
                point_to_edges_gnn[b].append(ei)

            edge_lengths_orig = np.asarray(orig_edge_sizes, dtype=np.float32)
            node_feat_base, global_median, global_iqr = _compute_gnn_node_features(
                edge_lengths_orig, point_to_edges_gnn, len(X)
            )
            candidate_edges = np.flatnonzero((~hard_union) & (~protected_union))
            print(f"  Node features ready in {time.time()-t0:.1f}s")
            print(f"  Candidate edges: {len(candidate_edges)} / {n_edges}")

            if len(candidate_edges) == 0:
                return pruned

            # ------------------------------------------------------------------
            # Batch inference on the 2-hop point subgraphs
            # ------------------------------------------------------------------
            device = next(model.parameters()).device

            batch_feats, batch_ei_local, batch_ids = [], [], []
            batch_target_edges, batch_orig_ei = [], []
            node_offset = 0
            processed = 0

            def _process_batch():
                nonlocal node_offset
                if not batch_orig_ei:
                    return

                all_x = torch.cat(batch_feats, 0)
                all_ei = (torch.cat(batch_ei_local, 0).T
                          if batch_ei_local else torch.zeros(2, 0, dtype=torch.long))
                all_b = torch.LongTensor(batch_ids)

                batch_dict = {
                    'x': all_x.to(device),
                    'edge_index': all_ei.to(device),
                    'batch': all_b.to(device),
                    'target_edges': list(batch_target_edges),
                    'y': torch.zeros(len(batch_orig_ei), dtype=torch.long, device=device),
                }

                with torch.no_grad():
                    logits = model(batch_dict)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()

                for k, orig_ei in enumerate(batch_orig_ei):
                    prune_prob = probs[k, 1]
                    if prune_prob >= 0.8:
                        pruned[orig_ei] = True

                if processed and processed % (batch_size * 20) == 0:
                    elapsed = time.time() - start_time
                    rate = processed / max(elapsed, 1e-3)
                    eta = (len(candidate_edges) - processed) / max(rate, 1)
                    print(f"  {processed}/{len(candidate_edges)} candidate edges ({rate:.0f}/s, ETA {eta:.1f}s)")

                batch_feats.clear()
                batch_ei_local.clear()
                batch_ids.clear()
                batch_target_edges.clear()
                batch_orig_ei.clear()
                node_offset = 0

            errors = 0
            for idx, ei in enumerate(candidate_edges):
                try:
                    f, eidx, tgt = _extract_gnn_subgraph(
                        ei, edge_keys, edge_lengths_orig,
                        point_to_edges_gnn, max_hops=2
                    )
                    n_ln = f.shape[0]
                    batch_feats.append(f)
                    if len(eidx) > 0:
                        eidx_t = torch.LongTensor(eidx) + node_offset
                        batch_ei_local.append(torch.cat([eidx_t, eidx_t.flip(1)], dim=0))
                    batch_ids.extend([len(batch_orig_ei)] * n_ln)
                    batch_target_edges.append(tgt)
                    batch_orig_ei.append(ei)
                    node_offset += n_ln
                    processed += 1

                    if len(batch_orig_ei) == batch_size or idx == len(candidate_edges) - 1:
                        _process_batch()
                except Exception:
                    errors += 1

            elapsed = time.time() - start_time
            print(f"GNN inference: {processed}/{len(candidate_edges)} candidate edges in {elapsed:.1f}s "
                  f"({processed/max(elapsed,1e-3):.0f} edges/s)")
            if errors:
                print(f"  {errors} errors (fallback: keep)")

            return pruned  # True = prune

        def _eligible_mask_mlp():
            """
            Classify gray-zone edges with the scalar-feature MLP.

            The MLP was trained on full-graph scalar features computed from
            both original and projected edge-length statistics, so inference
            must build those features from the unfiltered Delaunay graph.

            Returns boolean mask: True = prune, False = keep.
            """
            if not _gnn_available:
                raise RuntimeError("PyTorch not available. Cannot use MLP mode. Install: pip install torch")

            model, config = _load_mlp_model()
            if model is None:
                raise RuntimeError("Could not load trained MLP model from benchmark_clustering_tuning/gnn_experiments/mlp_dualspace_results.pt")

            import time
            start_time = time.time()

            pruned = np.zeros(n_edges, dtype=bool)
            candidate_edges = np.flatnonzero((~hard_union) & (~protected_union))
            print(f"Precomputing MLP edge features from full graph...")

            if len(candidate_edges) == 0:
                return pruned

            point_to_edges_mlp = _defaultdict(list)
            for ei, (a, b) in enumerate(edge_keys):
                point_to_edges_mlp[a].append(ei)
                point_to_edges_mlp[b].append(ei)

            edge_lengths_orig = np.asarray(orig_edge_sizes, dtype=np.float32)
            edge_lengths_proj = np.asarray(proj_edge_sizes, dtype=np.float32)
            context = {
                "edge_keys": edge_keys,
                "edge_lengths_orig": edge_lengths_orig,
                "edge_lengths_proj": edge_lengths_proj,
                "point_to_edges": point_to_edges_mlp,
                "dim": X.shape[1],
                "orig_stats": _compute_mlp_global_feats(edge_lengths_orig),
                "proj_stats": _compute_mlp_global_feats(edge_lengths_proj),
                "trusted_adj_orig": _build_mlp_trusted_adjacency(edge_keys, edge_lengths_orig),
                "trusted_adj_proj": _build_mlp_trusted_adjacency(edge_keys, edge_lengths_proj),
                "max_degree": max((len(vs) for vs in point_to_edges_mlp.values()), default=1),
            }
            batch_size = 2048
            feature_workers = max(1, int(mlp_feature_workers))
            device = next(model.parameters()).device
            scaler_mean = np.asarray(config.get('scaler_mean', []), dtype=np.float32)
            scaler_scale = np.asarray(config.get('scaler_scale', []), dtype=np.float32)

            def _scale_features(feats):
                if scaler_mean.size == 0 or scaler_scale.size == 0:
                    return feats
                return (feats - scaler_mean) / np.maximum(scaler_scale, 1e-8)

            processed = 0
            errors = 0
            def _score_feature_batch(feat_edge_ids, feats):
                nonlocal processed
                if len(feat_edge_ids) == 0:
                    return
                xb = torch.from_numpy(_scale_features(np.asarray(feats, dtype=np.float32))).to(device)
                with torch.no_grad():
                    logits = model(xb)
                    probs = torch.sigmoid(logits).cpu().numpy()
                for local_i, orig_ei in enumerate(feat_edge_ids):
                    if probs[local_i] >= 0.7:
                        pruned[orig_ei] = True
                processed += len(feat_edge_ids)

                if processed and processed % (batch_size * 10) == 0:
                    elapsed = time.time() - start_time
                    rate = processed / max(elapsed, 1e-3)
                    eta = (len(candidate_edges) - processed) / max(rate, 1)
                    print(f"  {processed}/{len(candidate_edges)} candidate edges ({rate:.0f}/s, ETA {eta:.1f}s)")

            if feature_workers <= 1 or len(candidate_edges) < 2 * batch_size:
                for start in range(0, len(candidate_edges), batch_size):
                    batch_ids = candidate_edges[start:start + batch_size]
                    feats = []
                    feat_edge_ids = []
                    for ei in batch_ids:
                        try:
                            feats.append(_compute_mlp_edge_features(ei, context, max_hops=2))
                            feat_edge_ids.append(ei)
                        except Exception:
                            errors += 1
                    _score_feature_batch(feat_edge_ids, feats)
            else:
                chunk_size = max(batch_size, 256)
                chunks = [
                    candidate_edges[start:start + chunk_size]
                    for start in range(0, len(candidate_edges), chunk_size)
                ]
                print(f"  Using {feature_workers} workers for MLP feature extraction...")
                try:
                    ctx = mp.get_context("spawn")
                    with ProcessPoolExecutor(
                        max_workers=feature_workers,
                        mp_context=ctx,
                        initializer=_init_mlp_parallel_context,
                        initargs=(context,),
                    ) as executor:
                        for feat_batch in executor.map(_compute_mlp_feature_batch_parallel, chunks, chunksize=1):
                            if not feat_batch:
                                continue
                            feat_edge_ids = [ei for ei, _ in feat_batch]
                            feats = [feat for _, feat in feat_batch]
                            _score_feature_batch(feat_edge_ids, feats)
                except Exception as exc:
                    print(f"  Parallel MLP feature extraction failed; falling back to sequential mode: {exc}")
                    pruned[:] = False
                    processed = 0
                    errors = 0
                    for start in range(0, len(candidate_edges), batch_size):
                        batch_ids = candidate_edges[start:start + batch_size]
                        feats = []
                        feat_edge_ids = []
                        for ei in batch_ids:
                            try:
                                feats.append(_compute_mlp_edge_features(ei, context, max_hops=2))
                                feat_edge_ids.append(ei)
                            except Exception:
                                errors += 1
                        _score_feature_batch(feat_edge_ids, feats)

            elapsed = time.time() - start_time
            print(f"MLP inference: {processed}/{len(candidate_edges)} candidate edges in {elapsed:.1f}s "
                  f"({processed/max(elapsed,1e-3):.0f} edges/s)")
            if errors:
                print(f"  {errors} feature errors (fallback: keep)")
            return pruned

        class _MLPBridgeEdgeClassifier(nn.Module):
            """Bridge-focused scalar-feature MLP used for pruning."""

            def __init__(self, n_feat, hidden=160, dropout=0.20):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(n_feat, hidden),
                    nn.LayerNorm(hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, hidden),
                    nn.LayerNorm(hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, hidden // 2),
                    nn.LayerNorm(hidden // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden // 2, 1),
                )

            def forward(self, x):
                return self.net(x).squeeze(1)

        def _load_mlp_bridge_model(benchmark_dir='benchmark_clustering_tuning'):
            """Load the bridge-focused scalar-feature MLP checkpoint."""
            global _mlp_bridge_model_cache

            if _mlp_bridge_model_cache is not None:
                try:
                    cached_model, cached_config, cached_path, cached_mtime = _mlp_bridge_model_cache
                    cached_path = Path(cached_path)
                    if cached_path.exists() and cached_path.stat().st_mtime_ns == cached_mtime:
                        return cached_model, cached_config
                except Exception:
                    _mlp_bridge_model_cache = None

            try:
                bridge_checkpoint = Path(benchmark_dir) / 'gnn_experiments' / 'mlp_originalspace_results.pt'
                candidates = [
                    bridge_checkpoint,
                    Path('gnn_experiments') / 'mlp_originalspace_results.pt',
                ]

                model_path = None
                for candidate in candidates:
                    if candidate.exists():
                        model_path = candidate
                        break

                if model_path is None:
                    return None, None

                checkpoint = torch.load(str(model_path), map_location='cpu', weights_only=False)
                model_state = checkpoint.get('model_state')
                if model_state is None:
                    return None, None

                key = 'net.0.weight'
                if key not in model_state:
                    print(f"Warning: unexpected MLP bridge weights (missing '{key}')")
                    return None, None

                n_feat = model_state[key].shape[1]
                hidden_dim = model_state[key].shape[0]

                model = _MLPBridgeEdgeClassifier(n_feat, hidden_dim)
                model.load_state_dict(model_state)
                model.eval()

                config = dict(checkpoint.get('config', {}))
                config['model_type'] = 'edge_mlp_bridge'
                config['checkpoint_path'] = str(model_path)
                config['scaler_mean'] = np.asarray(config.get('scaler_mean', []), dtype=np.float32)
                config['scaler_scale'] = np.asarray(config.get('scaler_scale', []), dtype=np.float32)

                print(f"Loaded MLP bridge checkpoint: {model_path}")
                _mlp_bridge_model_cache = (model, config, str(model_path), model_path.stat().st_mtime_ns)
                return model, config
            except Exception as e:
                print(f"Warning: Could not load MLP bridge model: {e}")
                return None, None

        def _eligible_mask_mlp_bridge():
            """
            Classify gray-zone edges with the bridge-focused MLP.

            The model uses original-space edge features only:
              - 1-hop and 2-hop relative length histograms
              - global normalized edge stats
              - target edge length normalized by dataset median
              - original-space dimension

            Returns boolean mask: True = prune, False = keep.
            """
            if not _gnn_available:
                raise RuntimeError("PyTorch not available. Cannot use MLP mode. Install: pip install torch")

            model, config = _load_mlp_bridge_model()
            if model is None:
                raise RuntimeError(
                    "Could not load trained MLP bridge model from "
                    "benchmark_clustering_tuning/gnn_experiments/mlp_originalspace_results.pt"
                )

            import time
            start_time = time.time()

            pruned = np.zeros(n_edges, dtype=bool)
            candidate_edges = np.flatnonzero((~hard_union) & (~protected_union))
            print(f"Precomputing MLP bridge edge features from full graph...")

            if len(candidate_edges) == 0:
                return pruned

            edge_lengths_orig = np.asarray(orig_edge_sizes, dtype=np.float32)
            point_to_edges_mlp = _defaultdict(list)
            for ei, (a, b) in enumerate(edge_keys):
                point_to_edges_mlp[a].append(ei)
                point_to_edges_mlp[b].append(ei)
            context = _compute_mlp_bridge_context(
                edge_keys=edge_keys,
                edge_lengths_orig=edge_lengths_orig,
                point_to_edges=point_to_edges_mlp,
                dim=X.shape[1],
            )

            batch_size = 2048
            feature_workers = max(1, int(mlp_feature_workers))
            device = next(model.parameters()).device
            scaler_mean = np.asarray(config.get('scaler_mean', []), dtype=np.float32)
            scaler_scale = np.asarray(config.get('scaler_scale', []), dtype=np.float32)

            def _scale_features(feats):
                if scaler_mean.size == 0 or scaler_scale.size == 0:
                    return feats
                return (feats - scaler_mean) / np.maximum(scaler_scale, 1e-8)

            feat_edge_ids, feat_matrix, precompute_errors = _precompute_mlp_bridge_features(
                candidate_edges=candidate_edges,
                context=context,
                feature_workers=feature_workers,
                batch_size=batch_size,
            )
            feat_matrix = _scale_features(feat_matrix)

            processed = 0
            errors = int(precompute_errors)

            def _score_feature_batch(feat_edge_ids, feats):
                nonlocal processed
                if len(feat_edge_ids) == 0:
                    return
                xb = torch.from_numpy(np.asarray(feats, dtype=np.float32)).to(device)
                with torch.no_grad():
                    logits = model(xb)
                    probs = torch.sigmoid(logits).cpu().numpy()
                for local_i, orig_ei in enumerate(feat_edge_ids):
                    if probs[local_i] >= 0.6:
                        pruned[orig_ei] = True
                processed += len(feat_edge_ids)

                if processed and processed % (batch_size * 10) == 0:
                    elapsed = time.time() - start_time
                    rate = processed / max(elapsed, 1e-3)
                    eta = (len(candidate_edges) - processed) / max(rate, 1)
                    print(f"  {processed}/{len(candidate_edges)} candidate edges ({rate:.0f}/s, ETA {eta:.1f}s)")

            for start in range(0, len(feat_edge_ids), batch_size):
                batch_ids = feat_edge_ids[start:start + batch_size]
                feats = feat_matrix[start:start + batch_size]
                _score_feature_batch(batch_ids, feats)

            elapsed = time.time() - start_time
            print(f"MLP bridge inference: {processed}/{len(candidate_edges)} candidate edges in {elapsed:.1f}s "
                  f"({processed/max(elapsed,1e-3):.0f} edges/s)")
            if errors:
                print(f"  {errors} feature errors (fallback: keep)")
            return pruned

        def _eligible_mask_bridges(pre_pruned=None):
            """
            Heuristic bridge pruning for long-thin clusters.

            Returns a boolean mask where True means the edge should be pruned.
            An edge becomes a bridge candidate when it has fewer than two
            distinct trusted alternative paths of length at most four hops.
            Candidate edges that belong to a nearly collinear chain of
            bridge_chain_len bridges are then protected from pruning.
            """
            if bridge_chain_len not in (3, 4):
                raise ValueError("bridge_chain_len must be 3 or 4")

            pruned = np.zeros(n_edges, dtype=bool)

            # When this selector is chained after another selector (for
            # example the saved baseline), those edges must no longer be
            # available as either bridge candidates or alternate paths.
            excluded = hard_union.copy()
            if pre_pruned is not None:
                excluded |= np.asarray(pre_pruned, dtype=bool)

            trusted_adj = _defaultdict(list)
            for ei, (a, b) in enumerate(edge_keys):
                if not excluded[ei]:
                    trusted_adj[a].append((b, ei))
                    trusted_adj[b].append((a, ei))

            def _count_short_alt_paths(u, v, skip_ei, max_hops=2, limit=2):
                if max_hops <= 0:
                    return 0

                count = 0
                stack = [(u, (u,))]
                while stack and count < limit:
                    node, path = stack.pop()
                    depth = len(path) - 1
                    if depth >= max_hops:
                        continue

                    for nei, ei in trusted_adj.get(node, []):
                        if ei == skip_ei or nei in path:
                            continue
                        if nei == v:
                            count += 1
                            if count >= limit:
                                return count
                        else:
                            stack.append((nei, path + (nei,)))
                return count

            bridge_candidate = np.zeros(n_edges, dtype=bool)
            # for ei, (a, b) in enumerate(edge_keys):
            #     size = orig_edge_sizes[ei]
            #     if size > hard_o:
            #         continue
            #     if _count_short_alt_paths(a, b, ei, max_hops=3, limit=2) < 2:
            #         bridge_candidate[ei] = True

            for ei, (a, b) in enumerate(edge_keys):
                # Hard edges, and edges already pruned by a preceding
                # selector, are handled outside this helper.
                if excluded[ei]:
                    continue

                # Protect only edges that are short in BOTH spaces.
                if (
                    orig_edge_sizes[ei] <= mild_o
                    and proj_edge_sizes[ei] <= mild_p
                ):
                    continue

                if _count_short_alt_paths(
                    a, b, ei,
                    max_hops=3,
                    limit=2,
                ) < 2:
                    bridge_candidate[ei] = True

            pos = X if X is not None else X_proj
            if pos is None:
                return bridge_candidate

            candidate_adj = _defaultdict(list)
            candidate_edges = np.flatnonzero(bridge_candidate)
            for ei in candidate_edges:
                a, b = edge_keys[ei]
                candidate_adj[a].append((b, ei))
                candidate_adj[b].append((a, ei))


            pruned[:] = bridge_candidate
            return pruned
        
        hard_union = (orig_edge_sizes > hard_o) | (proj_edge_sizes > hard_p)
        protected_union = (orig_edge_sizes <= mild_o) & (proj_edge_sizes <= mild_p)

        def _eligible_mask_baseline():
            """Classify gray-zone edges with a saved combined feature score."""
            candidate_edges = np.flatnonzero((~hard_union) & (~protected_union))

            # Learned defaults from feature_selection/combined_feature_analysis.json:
            # keep_risk   = 0.2585846965728894*ring1_short_M1
            #             + 0.030664675772955174*ring1_long_M3
            #             + 0.5137954793508874*ring2_long_M1
            #             + 0.19695514830326802*candidate_L_rel
            # prune_score = 0.17977096801432518*ring1_short_M1
            #             + 0.8202290319856749*candidate_L_rel

            # Best score:

            # score =
            #     0.445999 * ring2_long_M1
            #     + 2.259154 * candidate_L_rel


            score_weights={
                "ring1_short_M3": 0.0,
                "ring2_long_M1": 0.45,
                "candidate_L_rel": 2.25,
            }
            score_weights = {
                "ring2_long_M1": 0.4956024751,
                "candidate_L_rel_right": 1.7611342343,
            }
            pruned, scores, errors = _combined_baseline_mask(
                edge_keys=edge_keys,
                X=X,
                edge_lengths_orig=orig_edge_sizes,
                candidate_edges=candidate_edges,
                score_name=baseline_score,
                threshold=baseline_threshold,
                score_weights=score_weights
            )
            print(
                f"Baseline selector: score={baseline_score} "
                f"threshold={baseline_threshold:.3f} "
                f"pruned={int(pruned.sum())} candidates={len(candidate_edges)} "
                f"errors={errors}"
            )
            return pruned

        def _eligible_mask_nuclei_rule():
            """Use the global-tail and nearest-nucleus quantile conditions."""
            if not np.isfinite(nuclei_rule_a):
                raise ValueError("nuclei_rule_a must be finite")
            if int(nuclei_min_edges) < 1:
                raise ValueError("nuclei_min_edges must be positive")
            if not any(np.isclose(float(nuclei_quantile), option) for option in (0.70, 0.75, 0.80)):
                raise ValueError("nuclei_quantile must be one of 0.70, 0.75, or 0.80")
            nuclei_type_local = str(nuclei_type).lower()
            if nuclei_type_local not in {"red", "blue"}:
                raise ValueError("nuclei_type must be 'red' or 'blue'")

            candidate_edges = np.flatnonzero((~hard_union) & (~protected_union))
            pruned = np.zeros(n_edges, dtype=bool)
            if len(candidate_edges) == 0:
                return pruned

            # Condition 1: compare with the median of the global upper tail.
            hard_values = orig_edge_sizes[orig_edge_sizes > hard_o]
            global_median_edges_over_hard = (
                float(np.median(hard_values))
                if len(hard_values) else float(np.median(orig_edge_sizes))
            )

            # Build the edge graph in original space.  All nucleus statistics
            # below use an inclusive three-hop edge subgraph.
            edge_keys_array = np.asarray(edge_keys, dtype=np.int64)
            X_original = np.asarray(X, dtype=np.float64)
            edge_centres = 0.5 * (
                X_original[edge_keys_array[:, 0]]
                + X_original[edge_keys_array[:, 1]]
            )
            point_to_edges_nuclei = _defaultdict(list)
            point_to_points_nuclei = _defaultdict(set)
            for edge_index, (a, b) in enumerate(edge_keys_array):
                point_to_edges_nuclei[int(a)].append(edge_index)
                point_to_edges_nuclei[int(b)].append(edge_index)
                point_to_points_nuclei[int(a)].add(int(b))
                point_to_points_nuclei[int(b)].add(int(a))

            def three_hop_edges(start_points):
                edge_indices = set()
                seen_points = {int(point) for point in start_points}
                frontier = set(seen_points)
                for _ in range(3):
                    next_frontier = set()
                    for point in frontier:
                        for edge_index in point_to_edges_nuclei.get(point, []):
                            edge_indices.add(int(edge_index))
                        for neighbour in point_to_points_nuclei.get(point, ()):
                            if neighbour not in seen_points:
                                seen_points.add(neighbour)
                                next_frontier.add(neighbour)
                    frontier = next_frontier
                    if not frontier:
                        break
                return np.asarray(sorted(edge_indices), dtype=np.int64)

            nucleus_centres = []
            nucleus_starts = []
            if nuclei_type_local == "red":
                mild_indices = np.flatnonzero(orig_edge_sizes <= mild_o)
                mild_set = set(int(index) for index in mild_indices)
                components = []
                visited = set()
                for start in mild_indices:
                    start = int(start)
                    if start in visited:
                        continue
                    stack = [start]
                    visited.add(start)
                    component = []
                    while stack:
                        edge_index = stack.pop()
                        component.append(edge_index)
                        a, b = edge_keys_array[edge_index]
                        for point in (int(a), int(b)):
                            for neighbour in point_to_edges_nuclei.get(point, []):
                                if neighbour in mild_set and neighbour not in visited:
                                    visited.add(neighbour)
                                    stack.append(neighbour)
                    components.append(np.asarray(component, dtype=np.int64))
                nuclei = [component for component in components if len(component) >= int(nuclei_min_edges)]
                if not nuclei and components:
                    nuclei = [np.concatenate(components)]
                for component in nuclei:
                    nucleus_centres.append(np.mean(edge_centres[component], axis=0))
                    nucleus_starts.append(np.unique(edge_keys_array[component].ravel()))
            else:
                # Match the blue peak setup used by run_find_nuclei:
                # original-space support, peak percentile 70, NMS radius 0.9
                # times the global median edge length, density floor 30.
                from scipy.spatial import cKDTree
                finite = orig_edge_sizes[np.isfinite(orig_edge_sizes) & (orig_edge_sizes > 0)]
                support_scale = max(float(np.median(finite)) if len(finite) else 1.0, 1e-12)
                weights = np.exp(-np.square(orig_edge_sizes / support_scale))
                support = np.zeros(len(X_original), dtype=np.float64)
                for edge_index, (a, b) in enumerate(edge_keys_array):
                    support[int(a)] += weights[edge_index]
                    support[int(b)] += weights[edge_index]
                smoothed = support.copy()
                for point, neighbours in point_to_points_nuclei.items():
                    if neighbours:
                        smoothed[point] = 0.5 * (support[point] + np.mean([support[n] for n in neighbours]))
                k = min(10, max(len(X_original) - 1, 1))
                if len(X_original) > 2:
                    distances = cKDTree(X_original).query(X_original, k=k + 1)[0][:, -1]
                    density = 1.0 / np.maximum(distances, 1e-12)
                else:
                    density = np.ones(len(X_original), dtype=np.float64)

                def robust_unit(values):
                    lo, hi = np.percentile(values[np.isfinite(values)], [10.0, 90.0])
                    if hi <= lo + 1e-12:
                        return np.ones_like(values)
                    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)

                score = np.sqrt(
                    (0.15 + 0.85 * robust_unit(smoothed))
                    * (0.15 + 0.85 * robust_unit(density))
                )
                density_unit = robust_unit(density)
                minimum_score = np.percentile(score, 70.0)
                minimum_density = np.percentile(density_unit, 30.0)
                candidates = []
                for point, neighbours in point_to_points_nuclei.items():
                    neighbour_scores = [score[n] for n in neighbours]
                    if (
                        score[point] >= minimum_score
                        and density_unit[point] >= minimum_density
                        and (not neighbour_scores or score[point] >= max(neighbour_scores))
                    ):
                        candidates.append(int(point))
                candidates.sort(key=lambda point: score[point], reverse=True)
                suppression_radius = 0.9 * support_scale
                accepted = []
                for point in candidates:
                    if all(np.linalg.norm(X_original[point] - X_original[other]) >= suppression_radius for other in accepted):
                        accepted.append(point)
                for point in accepted:
                    nucleus_centres.append(X_original[point])
                    nucleus_starts.append(np.asarray([point], dtype=np.int64))

            nucleus_centres = np.asarray(nucleus_centres, dtype=np.float64).reshape((-1, X_original.shape[1]))
            nucleus_rings = [three_hop_edges(starts) for starts in nucleus_starts]
            q_percent = 100.0 * float(nuclei_quantile)
            global_q = float(np.percentile(orig_edge_sizes, q_percent))
            nucleus_q = np.asarray(
                [np.percentile(orig_edge_sizes[ring], q_percent) if len(ring) else global_q for ring in nucleus_rings],
                dtype=np.float64,
            )
            if len(nucleus_centres):
                from scipy.spatial import cKDTree
                _, nearest = cKDTree(nucleus_centres).query(
                    edge_centres, k=min(2, len(nucleus_centres))
                )
                nearest = np.asarray(nearest, dtype=np.int64)
                if nearest.ndim == 1:
                    nearest = nearest[:, None]
                nearest_q = np.min(nucleus_q[nearest], axis=1)
            else:
                nearest_q = np.full(n_edges, global_q, dtype=np.float64)

            global_ratio = orig_edge_sizes / max(global_median_edges_over_hard, 1e-12)
            nuclei_ratio = orig_edge_sizes / np.maximum(nearest_q, 1e-12)
            global_condition = global_ratio > float(nuclei_rule_a)
            nuclei_condition = nuclei_ratio > 1.0
            eligible = global_condition | nuclei_condition
            pruned[candidate_edges] = eligible[candidate_edges]
            print(
                "Nuclei rule: "
                f"A={float(nuclei_rule_a):.3f}, type={nuclei_type_local}, "
                f"q={float(nuclei_quantile):.2f}, nuclei={len(nucleus_centres)}, "
                f"global={int(global_condition[candidate_edges].sum())}, "
                f"nuclei={int(nuclei_condition[candidate_edges].sum())}, "
                f"union={int(pruned.sum())}/{len(candidate_edges)} candidates"
            )
            sigma_prune_triangles.last_nuclei_rule_stats = {
                "A": float(nuclei_rule_a),
                "nuclei_type": nuclei_type_local,
                "nuclei_quantile": float(nuclei_quantile),
                "nuclei_hops": 3,
                "global_median_edges_over_hard": global_median_edges_over_hard,
                "nuclei_count": int(len(nucleus_centres)),
                "global_condition_count": int(global_condition.sum()),
                "nuclei_condition_count": int(nuclei_condition.sum()),
                "union_candidate_count": int(pruned.sum()),
            }
            return pruned

        # Dispatcher: Select which masking method to use based on point_edge_compare.
        # The saved combined baselines take precedence when explicitly enabled.
        if baseline:
            orig_pruned = _eligible_mask_baseline()
            # Detect bridges on the graph left after the baseline decision.
            # In particular, baseline-pruned edges must not make another
            # edge appear to have an alternative path.
            # bridge_pruned = _eligible_mask_bridges(pre_pruned=orig_pruned)
            # baseline_count = int(orig_pruned.sum())
            # bridge_count = int(bridge_pruned.sum())
            # orig_pruned = orig_pruned | bridge_pruned  # union: prune if either flags it
            # print(
            #     f"Baseline+bridges: baseline={baseline_count} "
            #     f"bridges={bridge_count} total={int(orig_pruned.sum())}"
            # )
        elif point_edge_compare == 'nuclei_rule':
            orig_pruned = _eligible_mask_nuclei_rule()
        elif point_edge_compare == 'gnn':
            orig_pruned = _eligible_mask_gnn()
            print('GNN pruned: ', len([e for e in orig_pruned if e==True]), 'from', len(orig_pruned))
        elif point_edge_compare == 'mlp':
            orig_pruned = _eligible_mask_mlp()
            print('MLP pruned: ', len([e for e in orig_pruned if e==True]), 'from', len(orig_pruned))
        elif point_edge_compare == 'mlp_bridge':
            orig_pruned = _eligible_mask_mlp_bridge()
            print('MLP bridge pruned: ', len([e for e in orig_pruned if e==True]), 'from', len(orig_pruned))
        elif point_edge_compare == 'bridges':
            orig_pruned = _eligible_mask_bridges()
            print('bridges pruned: ', len([e for e in orig_pruned if e==True]), 'from', len(orig_pruned))
        elif isinstance(point_edge_compare, tuple):
            orig_pruned = _eligible_mask_tree(use_rules=point_edge_compare)
            print('tree pruned: ', len([e for e in orig_pruned if e==True]), 'from', len(orig_pruned))
        elif point_edge_compare == 'stat':
            orig_pruned = _eligible_mask_stat(orig_edge_sizes, hard_o, mild_o)
            print('stat pruned: ', len([e for e in orig_pruned if e==True]), 'from', len(orig_pruned))
        elif point_edge_compare == 'mlp_bridges':
            mlp_pruned = _eligible_mask_mlp()
            bridge_pruned = _eligible_mask_bridges()
            orig_pruned = mlp_pruned | bridge_pruned  # union: prune if either flags it
            print('MLP pruned:', int(mlp_pruned.sum()),
                  ' bridges pruned:', int(bridge_pruned.sum()),
                  ' combined (OR):', int(orig_pruned.sum()), 'from', len(orig_pruned))
        else:
            # Default: sigma method (point_edge_compare='sigma' or other)
            orig_pruned = _eligible_mask(orig_edge_sizes, hard_o, mild_o)
            print('sigma pruned: ', len([e for e in orig_pruned if e==True]), 'from', len(orig_pruned))

        local_union = orig_pruned
        # local_union = orig_pruned | proj_pruned
        pruned_mask = hard_union | ((~protected_union) & local_union)

        LOCAL_BRIDGE_MAX_HOPS = 5
        phase2_surv = {ei for ei in range(n_edges) if not pruned_mask[ei]}
        # Adjacency restricted to edges that survived phase 1+2.

        n_hard = int(hard_union.sum())
        n_local_mask = int(((~protected_union) & local_union).sum())
        n_after_phase2_local = int(pruned_mask.sum())
        print(f"hard={n_hard}  local={n_local_mask}  after_hard+local={n_after_phase2_local}")


        adj = _defaultdict(list)  # node -> list of (neighbor, edge_idx)
        for ei in phase2_surv:
            a, b = edge_keys[ei]
            adj[a].append((b, ei))
            adj[b].append((a, ei))

        # Trusted edges: surviving edges at or below the mild threshold.
        # Only these may be used as alternate-path hops in the bridge test.
        trusted_edge_set = {
            ei for ei in phase2_surv if orig_edge_sizes[ei] <= mild_o
        }

        # def _is_local_bridge(u, v, skip_ei, max_hops):
        #     """True if u,v have no path of length <= max_hops through
        #     trusted edges only (excluding the edge under test itself)."""
        #     if max_hops <= 0:
        #         return True
        #     frontier = {u}
        #     visited = {u}
        #     for _ in range(max_hops):
        #         next_frontier = set()
        #         for node in frontier:
        #             for nei, ei in adj.get(node, []):
        #                 if ei == skip_ei or ei not in trusted_edge_set:
        #                     continue
        #                 if nei == v:
        #                     return False
        #                 if nei not in visited:
        #                     visited.add(nei)
        #                     next_frontier.add(nei)
        #         frontier = next_frontier
        #         if not frontier:
        #             break
        #     return True

        # phase34_prune = np.zeros(n_edges, dtype=bool)
        # for ei in phase2_surv:
        #     size = orig_edge_sizes[ei]
        #     if size <= mild_o or size > hard_o:
        #         continue  # only "moderate" edges are candidates, as before
        #     a, b = edge_keys[ei]
        #     if _is_local_bridge(a, b, ei, LOCAL_BRIDGE_MAX_HOPS):
        #         phase34_prune[ei] = True

        # pruned_mask = pruned_mask | phase34_prune

        # ... after phase34_prune is computed ...
        # n_phase34 = int(phase34_prune.sum())
        # n_final = int((pruned_mask | phase34_prune).sum())
        # print(f"phase34_bridge={n_phase34}  final_total={n_final}")
        
        # Phase 3: Remove degree-2 bridges (edges where both endpoints have exactly 1 other incident edge)
        # Build map of surviving edges first
        
        # point_to_phase2_edges = {}
        # for ei in phase2_surv:
        #     a, b = edge_keys[ei]
        #     point_to_phase2_edges.setdefault(a, []).append(ei)
        #     point_to_phase2_edges.setdefault(b, []).append(ei)
        # phase3_surv = set()
        # for ei in phase2_surv:
        #     a, b = edge_keys[ei]
        #     deg_a = len(point_to_phase2_edges.get(a, [])) - 1
        #     deg_b = len(point_to_phase2_edges.get(b, [])) - 1
        #     if deg_a == 1 or deg_b == 1:
        #         continue
        #     phase3_surv.add(ei)

        # # Phase 4
        # surviving_keys = [edge_keys[i] for i in phase3_surv]
        # bridge_keys = _find_bridge_edges(surviving_keys, X_proj)

        # phase4_surv = set()
        # for ei in phase3_surv:
        #     key = edge_keys[ei]
        #     if key not in bridge_keys:
        #         phase4_surv.add(ei)
        
        # # Update pruned_mask to include degree-2 bridges
        # bridge_prune = np.zeros(n_edges, dtype=bool)
        # for ei in phase2_surv:
        #     if ei not in phase3_surv:
        #         bridge_prune[ei] = True
        # pruned_mask = pruned_mask | bridge_prune

        # phase4_prune = np.zeros(n_edges, dtype=bool)
        # for ei in phase3_surv:
        #     if ei not in phase4_surv:
        #         phase4_prune[ei] = True

        # pruned_mask |= phase4_prune
        
        pruned_edge_keys = set(edge_keys[i] for i in np.where(pruned_mask)[0])

        all_edge_set = set()
        for tri in triangles:
            for a, b in _simplex_edges(list(tri)):
                all_edge_set.add((min(a, b), max(a, b)))

        edges = all_edge_set - pruned_edge_keys
        rm_edges = pruned_edge_keys
        threshold = max(hard_o, hard_p)
        return list(edges), list(rm_edges), threshold

    elif prune_mode == 'neighbor_bridge':
        # Three-phase pruning combining neighbor-length ratio and
        # articulation-point bridge detection.
        #
        # Phase 1 - hard limit: unconditionally remove edges above a
        #   median + sf * hard_limit * IQR original-space threshold.
        #
        # Phase 2 - pointwise imbalance check: for each point, compare the
        #   incident surviving edges and mark edges that are much longer than
        #   the shortest incident edge at that point.  Only edges above the
        #   mild threshold are eligible for this pruning step.
        #
        # Phase 3 - articulation-point bridge detection: build a graph
        #   from the surviving edges, find articulation points, and
        #   remove edges that connect distinct biconnected components
        #   through a single point.
        if X_proj is None:
            raise ValueError("prune_mode='neighbor_bridge' requires X_proj")
        if X is None:
            X = X_proj

        edge_keys, edge_sizes, edge_centers, _ = _extract_edge_data(
            triangles, X_proj, X_size=X)

        if projected_hard_limit:
            _, projected_edge_sizes, _, _ = _extract_edge_data(triangles, X_proj)
        else:
            projected_edge_sizes = None

        n_edges = len(edge_keys)
        sf = float(sigma_factor)

        # Phase 1: robust original-space edge thresholds.
        # The previous percentile thresholds moved too much across datasets.
        # Median/mean + x*IQR keeps the tuning surface interpretable.
        q25, median, q75 = np.percentile(edge_sizes, [25, 50, 75])
        iqr = q75 - q25
        if iqr == 0:
            iqr = 1e-10
        if neighbor_stat == 'median':
            center = median
        elif neighbor_stat == 'mean':
            center = float(np.mean(edge_sizes))
        else:
            raise ValueError("neighbor_stat must be 'median' or 'mean'")
        hard_mult = 1.5 if hard_limit in (None, "auto") else hard_limit
        mild_mult = 1.0 if mild_limit in (None, "auto") else mild_limit

        # Optional additive two-regime selector (for ablation and auto mode)
        selected_regime = None
        if prune_regime in ("low", "high"):
            selected_regime = prune_regime
        elif prune_regime == "auto":
            selected_regime = _select_prune_regime(edge_sizes, projected_edge_sizes)
        if selected_regime is not None:
            sf = 0.5 if selected_regime == "low" else 1.5
            # Regime mode is intentionally additive for clean ablation.
            multiplicative_hard_limit = False
        # Side-channel: expose which regime was actually used.
        sigma_prune_triangles.last_regime_used = selected_regime if selected_regime is not None else "none"

        if multiplicative_hard_limit:
            hard_threshold = center * (1.0 + sf * hard_mult)
            relative_iqr = iqr / max(abs(center), 1e-10)
            mild_threshold = center * (
                1.0 + sf * mild_mult * relative_iqr ** float(mild_power)
            )
        else:
            hard_threshold = center + sf * hard_mult * iqr
            relative_iqr = iqr / max(abs(center), 1e-10)
            mild_threshold = center * (
                1.0 + sf * mild_mult * relative_iqr ** float(mild_power)
            )

        # EXPERIMENTAL: override thresholds with auto gap detection
        if auto_threshold:
            auto_hard, auto_mild = _kde_gap_threshold(edge_sizes)
            if auto_hard is not None:
                hard_threshold = auto_hard
                mild_threshold = auto_mild

        if projected_hard_limit:
            q25_p, median_p, q75_p = np.percentile(projected_edge_sizes, [25, 50, 75])
            iqr_p = q75_p - q25_p
            if iqr_p == 0:
                iqr_p = 1e-10
            if neighbor_stat == 'median':
                center_p = median_p
            else:
                center_p = float(np.mean(proj_sizes))
            if multiplicative_hard_limit:
                hard_threshold_proj = center_p * (1.0 + sf * hard_mult)
            else:
                hard_threshold_proj = center_p + sf * hard_mult * iqr_p
        else:
            hard_threshold_proj = None

        # Start with all edges
        surviving = set(range(n_edges))

        # Phase 1: remove edges above hard limit
        for ei in range(n_edges):
            if edge_sizes[ei] > hard_threshold:
                surviving.discard(ei)
            elif projected_hard_limit and hard_threshold_proj is not None and projected_edge_sizes is not None and projected_edge_sizes[ei] > hard_threshold_proj:
                surviving.discard(ei)

        # Phase 2: choose between the original neighbor-ratio check and the
        # new pointwise imbalance rule.
        if len(surviving) > 0:
            surviving_list = sorted(surviving)
            if pointwise_bridge_pruning:
                point_to_edges = {}
                for ei in surviving_list:
                    a, b = edge_keys[ei]
                    point_to_edges.setdefault(a, []).append(ei)
                    point_to_edges.setdefault(b, []).append(ei)

                eligible = set()
                ratio_thr = max(1.0, float(point_edge_ratio))
                for _, incident in point_to_edges.items():
                    if len(incident) < 2:
                        continue
                    incident_sizes = edge_sizes[incident]
                    shortest = float(np.min(incident_sizes))
                    if shortest <= 0:
                        continue
                    for ei in incident:
                        if edge_sizes[ei] > mild_threshold and edge_sizes[ei] >= shortest * ratio_thr:
                            eligible.add(ei)

                for ei in eligible:
                    surviving.discard(ei)
            else:
                surviving_keys = [edge_keys[i] for i in surviving_list]
                surviving_sizes = edge_sizes[surviving_list]
                ratio = _neighbor_length_ratio(surviving_keys, surviving_sizes)
                for idx_in_surv, ei in enumerate(surviving_list):
                    if edge_sizes[ei] > mild_threshold:
                        if ratio[idx_in_surv] > 2.0:
                            surviving.discard(ei)

        # Phase 3: articulation-point bridge detection
        if len(surviving) > 0:
            surviving_keys = [edge_keys[i] for i in surviving]
            bridge_keys = _find_bridge_edges(surviving_keys, X_proj)
            for ei in list(surviving):
                a, b = edge_keys[ei]
                key = (min(a, b), max(a, b))
                if key in bridge_keys:
                    surviving.discard(ei)

        # Build output
        all_edge_set = set()
        for tri in triangles:
            for a, b in _simplex_edges(list(tri)):
                all_edge_set.add((min(a, b), max(a, b)))

        pruned_edge_keys = set(edge_keys[i] for i in range(n_edges)
                               if i not in surviving)
        edges = all_edge_set - pruned_edge_keys
        rm_edges = pruned_edge_keys
        threshold = hard_threshold

        return list(edges), list(rm_edges), threshold

    else:
        raise ValueError(f"Unknown prune_mode '{prune_mode}'. "
                         "Use 'orig_only', 'iqr_max', 'edge_center', "
                         "'pointwise_dual_space', or 'neighbor_bridge'.")

    # --- shared post-processing for orig_only and iqr_max ---
    edges = set()
    rm_edges = set()
    for tri in rm_triangles:
        for a, b in _simplex_edges(list(tri)):
            rm_edges.add((min(a, b), max(a, b)))
    for tri in kept_triangles:
        for a, b in _simplex_edges(list(tri)):
            edges.add((min(a, b), max(a, b)))
    return list(edges), list(rm_edges), threshold

def merge_clusters(X, X_proj, clusters, method, merge_param=None, back_proj=True,
                     anomaly_thresh=2, min_centroids=2):
    """Merge over-segmented rich clusters using directional neighborhood test.

    For each small cluster, finds the direction to the nearest larger cluster,
    samples outermost points in that direction, and checks if their nearest
    neighbor (in original high-D space) belongs to the target cluster.
    Merge if the overlap fraction >= merge_param.

    Operates in original high-D space as a cleanup step after pruning.

    Parameters
    ----------
    merge_param : float in [0, 1] or None
        Minimum fraction of outermost points that must have their nearest
        neighbor in the target cluster to trigger a merge.
        0.0 = always merge, 1.0 = only if ALL outer points face the target.
        None = skip merging entirely (clusters returned unchanged).
    """
    if merge_param is None:
        return clusters

    from scipy.spatial import cKDTree

    anomalies = [list(cl) for cl in clusters if len(cl) <= anomaly_thresh]
    rich = [list(cl) for cl in clusters if len(cl) > anomaly_thresh]
    if len(rich) < 2:
        return clusters

    try:
        tree = cKDTree(X)

        changed = True
        while changed:
            changed = False
            n = len(rich)
            if n < 2:
                break

            order = np.argsort([len(c) for c in rich])
            rich = [rich[i] for i in order]

            ptocl = np.full(X.shape[0], -1, dtype=int)
            for ci, cl in enumerate(rich):
                for p in cl:
                    ptocl[p] = ci

            centroids = np.array([np.mean(X[c], axis=0) for c in rich])
            merge_target = {}

            for i in range(n):
                ci = rich[i]
                larger = np.array([len(rich[j]) > len(ci) for j in range(n)])
                larger[i] = False
                if not np.any(larger):
                    continue

                dists = np.linalg.norm(centroids - centroids[i], axis=1)
                dists[~larger] = np.inf
                j = np.argmin(dists)
                if dists[j] == np.inf:
                    continue

                direction = centroids[j] - centroids[i]
                dn = np.linalg.norm(direction)
                if dn == 0:
                    continue
                direction /= dn

                ci_arr = np.array(ci)
                projections = (X[ci_arr] - centroids[i]) @ direction
                ns = min(20, len(ci))
                outer_pts = ci_arr[np.argsort(projections)[-ns:]]

                overlap = 0
                for p in outer_pts:
                    d_nn, idx_nn = tree.query(X[p], k=2)
                    nn = idx_nn[1] if idx_nn[0] == p else idx_nn[0]
                    if ptocl[nn] == j:
                        overlap += 1

                if overlap / ns >= merge_param:
                    merge_target[i] = j

            if not merge_target:
                break

            removed = set()
            new_rich = []
            for i in range(n):
                if i in removed:
                    continue
                if i in merge_target:
                    j = merge_target[i]
                    if j not in removed:
                        new_rich.append(sorted(rich[i] + rich[j]))
                        removed.add(i)
                        removed.add(j)
                        changed = True
                    else:
                        new_rich.append(rich[i])
                else:
                    new_rich.append(rich[i])
            rich = new_rich

        return rich + anomalies
    except Exception:
        return clusters




def tri_cluster_sigma_merge(X, prune_param=1.3, merge_param=None, method='umap',
                       anomaly_sensitivity=0.5, back_proj=True, debug=False,
                       prune_mode='neighbor_bridge', umap_n_epochs='auto', umap_n_neighbors=15,
                       hard_limit=1.0, mild_limit=0.5,
                       neighbor_stat='median', project_dim=2,
                       profile_phases=False, anomaly_thresh=ANOMALY_THRESH,
                       projected_hard_limit=True,
                       multiplicative_hard_limit=False,
                       auto_threshold=False,
                       prune_regime='none',
                       pointwise_bridge_pruning=False,
                       point_edge_ratio=1.5,
                       point_edge_compare='ratio',
                       point_edge_subset='all',
                       point_edge_gap=0.5,
                       bridge_chain_len=4,
                       mlp_feature_workers=4,
                       baseline=False,
                       baseline_score='prune_score',
                       baseline_threshold=0.5,
                       nuclei_rule_a=0.8,
                       nuclei_min_edges=20,
                       nuclei_quantile=0.8,
                       nuclei_type='red'):
    phase_times = {}
    # Step 1: Triangles & sigma pruning
    t0 = time.perf_counter() if profile_phases else None
    triangles, sizes, proj_sizes, X_proj = get_triangles_with_edges(
        X, project_dim=project_dim, method=method, back_proj=back_proj,
        umap_n_epochs=umap_n_epochs, umap_n_neighbors=umap_n_neighbors,
        profile_phases=profile_phases)
    if profile_phases:
        phase_times.update(getattr(get_triangles_with_edges, "last_phase_times", {}))
    t1 = time.perf_counter() if profile_phases else None
    pruned_edges, rm_edges, threshold = sigma_prune_triangles(
        triangles, sizes, proj_sizes, sigma_factor=prune_param,
        prune_mode=prune_mode, X_proj=X_proj, X=X,
        hard_limit=hard_limit, mild_limit=mild_limit,
        neighbor_stat=neighbor_stat, projected_hard_limit=projected_hard_limit,
        multiplicative_hard_limit=multiplicative_hard_limit,
        auto_threshold=auto_threshold,
        prune_regime=prune_regime,
        pointwise_bridge_pruning=pointwise_bridge_pruning,
        point_edge_ratio=point_edge_ratio,
        point_edge_compare=point_edge_compare,
        point_edge_subset=point_edge_subset,
        point_edge_gap=point_edge_gap,
        bridge_chain_len=bridge_chain_len,
        mlp_feature_workers=mlp_feature_workers,
        baseline=baseline,
        baseline_score=baseline_score,
        baseline_threshold=baseline_threshold,
        nuclei_rule_a=nuclei_rule_a,
        nuclei_min_edges=nuclei_min_edges,
        nuclei_quantile=nuclei_quantile,
        nuclei_type=nuclei_type)
    if profile_phases and t1 is not None:
        phase_times["pruning_seconds"] = time.perf_counter() - t1
    t2 = time.perf_counter() if profile_phases else None
    clusters = get_clusters(pruned_edges, len(X))
    if debug:
        print(f"After pruning: {len(clusters)} clusters")
        print(f"Anomalies: {len([cl for cl in clusters if len(cl) <= anomaly_thresh])}")

    # Step 2: Merge pass on centroids
    merged_clusters = merge_clusters(
        X, X_proj, clusters, method,
        merge_param=merge_param, back_proj=back_proj,
        anomaly_thresh=anomaly_thresh,
    )
    if profile_phases and t2 is not None:
        phase_times["merge_clusters_seconds"] = time.perf_counter() - t2
    if debug:
        print(f"After merging clusters: {len(merged_clusters)} clusters")

    if anomaly_sensitivity > 0.0:
        # anomaly_sensitivity controls outlier recall.
        # Higher -> larger merge radius -> fewer points kept as outliers.
        _X = X
        if back_proj==False:
            _X = X_proj
        t3 = time.perf_counter() if profile_phases else None
        merged_clusters = merge_anomalies_triangles(
            _X, X_proj, merged_clusters, triangles, sizes, proj_sizes,
            anomaly_thresh=anomaly_thresh,
            max_iter=3, debug=debug,
            anomaly_sensitivity=anomaly_sensitivity,
        )
        if profile_phases and t3 is not None:
            phase_times["merge_anomalies_seconds"] = time.perf_counter() - t3

    if debug:
        print(f"After merging anomalies: {len(merged_clusters)} clusters")

    # Keeping same output structure, but adding merged_clusters as extra
    if profile_phases:
        tri_cluster_sigma_merge.last_phase_times = phase_times
    return triangles, sizes, pruned_edges, rm_edges, merged_clusters


def _clusters_to_labels(clusters, n_points, anomaly_thresh):
    """Convert cluster list to dense integer label array (-1 = noise)."""
    labels = np.full(n_points, -1, dtype=int)
    label_id = 0
    for cluster in clusters:
        if len(cluster) > anomaly_thresh:
            for idx in cluster:
                labels[idx] = label_id
            label_id += 1
    return labels


def _is_degenerate(labels, n_points):
    """Return True if clustering result is degenerate.

    Degenerate conditions (any one is sufficient):
      - No real clusters (all points are noise / single cluster only).
      - One cluster absorbs ≥ 90 % of labelled points.
      - More than 50 % of points are marked as noise (-1).
    """
    unique = np.unique(labels[labels >= 0])
    n_clusters = len(unique)
    if n_clusters <= 1:
        return True
    n_noise = int(np.sum(labels == -1))
    if n_noise / n_points > 0.5:
        return True
    largest = int(max(np.sum(labels == lbl) for lbl in unique))
    if largest / n_points > 0.9:
        return True
    return False


def cluster_tri(X, prune_param=1.3, merge_param=None, min_cluster_size=10,
                dim_reduction='umap', back_proj=True, anomaly_sensitivity=0.5,
                prune_mode='neighbor_bridge', umap_n_epochs='auto', umap_n_neighbors=15,
                hard_limit=1.0, mild_limit=0.5,
                neighbor_stat='median', random_state=42, project_dim=2,
                profile_phases=False, projected_hard_limit=True,
                multiplicative_hard_limit=False,
                auto_threshold=False,
                prune_regime='none',
                pointwise_bridge_pruning=False,
                point_edge_ratio=1.5,
                point_edge_compare='ratio',
                point_edge_subset='all',
                point_edge_gap=0.5,
                bridge_chain_len=4,
                mlp_feature_workers=1,
                baseline=False,
                baseline_score='prune_score',
                baseline_threshold=0.5,
                nuclei_rule_a=0.8,
                nuclei_min_edges=20,
                nuclei_quantile=0.8,
                nuclei_type='red'):
    anomaly_thresh = max(1, int(min_cluster_size)) if min_cluster_size is not None else ANOMALY_THRESH

    def _run(regime):
        _, _, pe, _, cl = tri_cluster_sigma_merge(
            X, prune_param=prune_param, merge_param=merge_param,
            method=dim_reduction, back_proj=back_proj,
            anomaly_sensitivity=anomaly_sensitivity, prune_mode=prune_mode,
            umap_n_epochs=umap_n_epochs, umap_n_neighbors=umap_n_neighbors,
            hard_limit=hard_limit, mild_limit=mild_limit,
            neighbor_stat=neighbor_stat, project_dim=project_dim,
            profile_phases=profile_phases, anomaly_thresh=anomaly_thresh,
            projected_hard_limit=projected_hard_limit,
            multiplicative_hard_limit=multiplicative_hard_limit,
            auto_threshold=auto_threshold,
            prune_regime=regime,
            pointwise_bridge_pruning=pointwise_bridge_pruning,
            point_edge_ratio=point_edge_ratio,
            point_edge_compare=point_edge_compare,
            point_edge_subset=point_edge_subset,
            point_edge_gap=point_edge_gap,
            bridge_chain_len=bridge_chain_len,
            mlp_feature_workers=mlp_feature_workers,
            baseline=baseline,
            baseline_score=baseline_score,
            baseline_threshold=baseline_threshold,
            nuclei_rule_a=nuclei_rule_a,
            nuclei_min_edges=nuclei_min_edges,
            nuclei_quantile=nuclei_quantile,
            nuclei_type=nuclei_type)
        used = getattr(sigma_prune_triangles, "last_regime_used", "none")
        return cl, used

    clusters, regime_used = _run(prune_regime)
    labels = _clusters_to_labels(clusters, len(X), anomaly_thresh)
    fallback_triggered = False

    # Guarded fallback: if auto-regime produced a degenerate result, retry
    # with the opposite regime.
    if prune_regime == "auto" and _is_degenerate(labels, len(X)):
        opposite = "high" if regime_used == "low" else "low"
        print(f"Fallback to {opposite} regime.")
        fb_clusters, fb_regime_used = _run(opposite)
        fb_labels = _clusters_to_labels(fb_clusters, len(X), anomaly_thresh)
        clusters, labels, regime_used = fb_clusters, fb_labels, fb_regime_used
        fallback_triggered = True

    # Expose side-channel attributes for callers (e.g. DelTriC, plot scripts).
    cluster_tri.regime_used = regime_used
    cluster_tri.fallback_triggered = fallback_triggered
    if profile_phases:
        cluster_tri.last_phase_times = getattr(tri_cluster_sigma_merge, "last_phase_times", {})
    return labels
