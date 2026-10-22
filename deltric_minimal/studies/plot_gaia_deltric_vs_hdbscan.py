#!/usr/bin/env python3
"""Side-by-side diagnostic plots: ground truth vs. DelTriC vs. HDBSCAN, per Gaia cluster.

Companion to ``eval_gaia_anomaly.py``. That script found DelTriC wins
anomaly-detection F1 by a large margin while HDBSCAN wins ARI by a large
margin on the same 5 Gaia open-cluster datasets. This renders the actual
partitions on shared UMAP, PCA, or t-SNE display coordinates so the reason is visible:
HDBSCAN tends to fold the surrounding field-star population into its own
legitimate dense cluster (never labelled ``-1``) rather than calling it
noise, so it recovers the true 2-group structure (high ARI) but predicts
almost no anomalies (low F1); DelTriC's component-growth + cost-gate
Dijkstra stage is built to leave/reclaim points as noise explicitly.

Each output PNG has three panels sharing one x/y scale:
  1. ground truth (member = gray, field/noise = red)
  2. DelTriC final clusters (production cost-gate config, p=3/reach=25/q=4)
  3. HDBSCAN final clusters (original-space, min_cluster_size=10, eom)

Usage:
    uv run studies/plot_gaia_deltric_vs_hdbscan.py
    uv run studies/plot_gaia_deltric_vs_hdbscan.py --out-dir studies/results/gaia_diagnostic
    uv run studies/plot_gaia_deltric_vs_hdbscan.py --projection pca --dataset gaia_melotte_22
    uv run studies/plot_gaia_deltric_vs_hdbscan.py --projection tsne --dataset gaia_melotte_22

The display projection never changes the clustering: DelTriC still uses
UMAP internally and HDBSCAN still runs on the standardized original features.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

_DELTRIC_MINIMAL_DIR = Path(__file__).resolve().parents[1]
_DEFAULT_DATA_DIR = _DELTRIC_MINIMAL_DIR / "data"
sys.path.insert(0, str(_DELTRIC_MINIMAL_DIR.parent))

from deltric_minimal.utils_component_growth import cluster_tri  # noqa: E402
from deltric_minimal.studies.anomaly_cost_gate_study import BASE_CONFIG  # noqa: E402
from deltric_minimal.hdbscan_comparison import fit_hdbscan, _label_colours  # noqa: E402

PENALTY_POWER = 3.0
REACH = 25.0
DENSITY_POWER = 4.0

HDBSCAN_MIN_CLUSTER_SIZE = 10
HDBSCAN_MIN_SAMPLES = None
HDBSCAN_SELECTION_METHOD = "eom"
HDBSCAN_CLUSTER_SELECTION_EPSILON = 0.0

GAIA_STEMS = [
    "gaia_ic_2391", "gaia_melotte_22", "gaia_ngc_2516",
    "gaia_ngc_2632", "gaia_ngc_6705",
]


def _truth_colours(y: np.ndarray) -> np.ndarray:
    colours = np.full((len(y), 4), (0.55, 0.55, 0.55, 0.55), dtype=float)
    colours[y < 0] = (0.85, 0.1, 0.1, 0.9)
    return colours


def plot_one(path: Path, out_dir: Path, projection: str = "umap") -> Path:
    if projection not in {"umap", "pca", "tsne"}:
        raise ValueError(f"Unknown display projection: {projection}")
    npz = np.load(path, allow_pickle=False)
    X = StandardScaler().fit_transform(np.asarray(npz["X"], dtype=np.float64))
    y = np.asarray(npz["y"]).astype(np.int64)

    deltric_labels = cluster_tri(
        X, **BASE_CONFIG,
        component_growth_anomaly_reassign=True,
        component_growth_anomaly_reassign_penalty_power=PENALTY_POWER,
        component_growth_anomaly_reassign_reach=REACH,
        component_growth_anomaly_reassign_density_power=DENSITY_POWER,
    )
    X_proj = cluster_tri.last_component_growth["X_proj"]
    axis_labels = ["UMAP 1 (display only)", "UMAP 2 (display only)"]
    projection_note = ""
    if projection == "pca":
        pca = PCA(n_components=2, svd_solver="full")
        X_proj = pca.fit_transform(X)
        axis_labels = [
            f"PC {i + 1} ({ratio:.1%} variance; display only)"
            for i, ratio in enumerate(pca.explained_variance_ratio_)
        ]
        projection_note = (
            f"\nPCA display: {pca.explained_variance_ratio_.sum():.1%} variance retained"
            " | DelTriC clustering: UMAP"
        )
    elif projection == "tsne":
        X_proj = TSNE(
            n_components=2, perplexity=30.0, init="pca",
            learning_rate="auto", random_state=42, max_iter=1000,
        ).fit_transform(X)
        axis_labels = ["t-SNE 1 (display only)", "t-SNE 2 (display only)"]
        projection_note = (
            "\nt-SNE display: perplexity=30, seed=42 | DelTriC clustering: UMAP"
        )

    hdb_labels = fit_hdbscan(
        X,
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        cluster_selection_method=HDBSCAN_SELECTION_METHOD,
        cluster_selection_epsilon=HDBSCAN_CLUSTER_SELECTION_EPSILON,
    )

    n_true = int((y < 0).sum())
    n_deltric_pred = int((deltric_labels < 0).sum())
    n_hdb_pred = int((hdb_labels < 0).sum())
    n_deltric_clusters = int(np.unique(deltric_labels[deltric_labels >= 0]).size)
    n_hdb_clusters = int(np.unique(hdb_labels[hdb_labels >= 0]).size)

    fig, axes = plt.subplots(1, 3, figsize=(19, 3.2 if projection == "pca" else 6.2),
                             constrained_layout=True,
                             sharex=True, sharey=True)

    axes[0].scatter(X_proj[:, 0], X_proj[:, 1], c=_truth_colours(y), s=6, linewidths=0)
    axes[0].set_title(f"ground truth\n{n_true} field stars (red) / {len(y) - n_true} members (gray)",
                       fontsize=10)

    axes[1].scatter(X_proj[:, 0], X_proj[:, 1], c=_label_colours(deltric_labels), s=6, linewidths=0)
    axes[1].set_title(f"DelTriC (cost-gate)\n{n_deltric_clusters} clusters, "
                       f"{n_deltric_pred} predicted noise", fontsize=10)

    axes[2].scatter(X_proj[:, 0], X_proj[:, 1], c=_label_colours(hdb_labels), s=6, linewidths=0)
    axes[2].set_title(f"HDBSCAN (mcs=10, eom)\n{n_hdb_clusters} clusters, "
                       f"{n_hdb_pred} predicted noise", fontsize=10)

    for ax in axes:
        ax.set_xlabel(axis_labels[0])
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel(axis_labels[1])

    fig.suptitle(path.stem + projection_note, fontsize=12,
                 y=1.06 if projection == "tsne" else 0.98)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{projection}" if projection != "umap" else ""
    out_path = out_dir / f"{path.stem}_compare{suffix}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=_DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path,
                         default=_DELTRIC_MINIMAL_DIR / "studies/results/gaia_diagnostic")
    parser.add_argument("--projection", choices=("umap", "pca", "tsne"), default="umap",
                        help="Display projection only; DelTriC always clusters using UMAP.")
    parser.add_argument("--dataset", choices=GAIA_STEMS, nargs="+", default=GAIA_STEMS)
    args = parser.parse_args()

    for stem in args.dataset:
        path = args.data / f"{stem}.npz"
        if not path.exists():
            print(f"skip (missing): {path}")
            continue
        print(f"plotting {stem} ...", flush=True)
        out_path = plot_one(path, args.out_dir, args.projection)
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
