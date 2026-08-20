#!/usr/bin/env python3
"""Greedy ARI-oracle bridge pruning: an upper-bound test for edge pruning.

Every other edge-pruning policy in this repo (the depth-4 tree in
``edge_rule.py``, the KDE-gap heuristic, etc.) decides prune-or-keep from
per-edge features it can compute without ground truth.  This script asks a
different question: if a pruning policy had a perfect, direct read on
clustering quality, how much could bridge pruning possibly buy?

It runs the production ``component_growth_graph`` pipeline once per dataset
to get the same surviving-edge graph the real algorithm would grow (the
``final_mask`` edges), then walks that graph's bridge edges one at a time --
longest first, since long edges are the usual bridge suspects -- and, for
each one, tentatively removes it, recomputes global ARI against the ground
truth labels, and *keeps the removal only if ARI strictly increased*.  ARI
computed against ground truth is the "loss": a bridge is pruned iff doing so
is loss-improving, kept otherwise.

This is a single-pass oracle: the bridge set is computed once up front from
the starting graph and not recomputed after each accepted removal (pruning
one bridge can occasionally turn a previously non-bridge edge into a bridge
elsewhere in a large biconnected component). That keeps the algorithm to one
connected-components call per bridge candidate rather than a full bridge
re-scan, which is what makes it tractable on datasets up to ~5k points /
~30k edges. Treat the resulting ARI as a ceiling estimate, not a tight one.

Because it consumes ``y`` (ground truth) while pruning, this is a ceiling /
diagnostic experiment, not a deployable rule -- there is no ground truth at
inference time. It measures how much headroom exists above the current
heuristic (``depth4_rule``) and above doing nothing (``baseline``).

Usage:
    python studies/edge_ari_oracle_pruning.py --out studies/results/edge_ari_oracle
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

_DELTRIC_MINIMAL_DIR = Path(__file__).resolve().parents[1]
_DEFAULT_DATA_DIR = _DELTRIC_MINIMAL_DIR / "data"
sys.path.insert(0, str(_DELTRIC_MINIMAL_DIR.parent))

from deltric_minimal.utils_component_growth import cluster_tri  # noqa: E402
from deltric_minimal.studies.edge_rule import depth4_rule  # noqa: E402

# Growth-mode defaults, matching eval_edge_rule_ari.py / run_plot_stages.sh.
BASE_CONFIG = dict(
    dim_reduction="umap",
    back_proj=True,
    project_dim=2,
    umap_n_epochs=100,
    umap_n_neighbors=15,
    hard_limit=0.9,
    mild_limit=0.5,
    neighbor_stat="median",
    projected_hard_limit=True,
    multiplicative_hard_limit=False,
    component_growth_initial_relation="union",
    component_growth_growth_relation="union",
    component_growth_knn=50,
    component_growth_min_edges=10,
    component_growth_seed_hard_limit=0.25,
    component_growth_growth_hard_limit=0.9,
    component_growth_hard_gate_mode="knn_relaxed",
    component_growth_seed_relaxed_hard_limit=0.75,
    component_growth_growth_relaxed_hard_limit=1.7,
    component_growth_local_knn_selectivity_threshold=0.1,
    component_growth_local_knn_selectivity_scope="edge",
    component_growth_bridge_pruning=False,
    component_growth_outlier_limit=1.0,
    component_growth_open_space_relaxation=False,
    min_cluster_size=10,
)


def _component_labels(edge_keys: np.ndarray, mask: np.ndarray, n: int) -> np.ndarray:
    """Connected-component id per point for the surviving (masked) edges."""
    if not np.any(mask):
        return np.arange(n, dtype=np.int64)
    kept = edge_keys[mask]
    u, v = kept[:, 0], kept[:, 1]
    data = np.ones(len(u), dtype=np.int8)
    graph = coo_matrix((data, (u, v)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    return labels.astype(np.int64)


def _find_bridges(edge_keys: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Indices (into ``edge_keys``) of bridge edges among the masked edges."""
    import networkx as nx

    kept_idx = np.flatnonzero(mask)
    graph = nx.Graph()
    graph.add_edges_from(
        (int(edge_keys[i, 0]), int(edge_keys[i, 1]), {"idx": int(i)})
        for i in kept_idx
    )
    bridge_idx = [
        graph[u][v]["idx"] for u, v in nx.bridges(graph)
    ]
    return np.asarray(bridge_idx, dtype=np.int64)


def oracle_bridge_prune(
    edge_keys: np.ndarray,
    orig_sizes: np.ndarray,
    mask: np.ndarray,
    y_true: np.ndarray,
    n: int,
) -> dict:
    """Greedily remove bridge edges, longest first, whenever ARI strictly improves.

    Returns a dict with the final mask, the ARI trajectory, and per-bridge
    accept/reject decisions.
    """
    working_mask = mask.copy()
    bridge_idx = _find_bridges(edge_keys, working_mask)
    order = bridge_idx[np.argsort(-orig_sizes[bridge_idx])]

    labels = _component_labels(edge_keys, working_mask, n)
    current_ari = float(adjusted_rand_score(y_true, labels))
    start_ari = current_ari

    n_pruned = 0
    n_tested = 0
    for idx in order:
        n_tested += 1
        candidate_mask = working_mask.copy()
        candidate_mask[idx] = False
        candidate_labels = _component_labels(edge_keys, candidate_mask, n)
        candidate_ari = float(adjusted_rand_score(y_true, candidate_labels))
        if candidate_ari > current_ari:
            working_mask = candidate_mask
            current_ari = candidate_ari
            n_pruned += 1

    return {
        "final_mask": working_mask,
        "start_ari": start_ari,
        "final_ari": current_ari,
        "n_bridges_tested": n_tested,
        "n_bridges_pruned": n_pruned,
    }


def run_one(path: Path) -> dict:
    npz = np.load(path, allow_pickle=False)
    X = StandardScaler().fit_transform(np.asarray(npz["X"], dtype=np.float64))
    y = np.asarray(npz["y"]).astype(np.int64)
    n = len(X)

    started = time.perf_counter()
    baseline_labels = cluster_tri(X, **BASE_CONFIG)
    state = cluster_tri.last_component_growth
    edge_keys = np.asarray(state["edge_keys"])
    orig_sizes = np.asarray(state["orig_edge_sizes"])
    final_mask = np.asarray(state["final_mask"], dtype=bool)
    baseline_ari = float(adjusted_rand_score(y, baseline_labels))
    baseline_graph_ari = float(
        adjusted_rand_score(y, _component_labels(edge_keys, final_mask, n))
    )

    rule_labels = cluster_tri(
        X, component_growth_edge_rule=depth4_rule,
        component_growth_edge_rule_scope="both", **BASE_CONFIG,
    )
    rule_ari = float(adjusted_rand_score(y, rule_labels))

    oracle = oracle_bridge_prune(edge_keys, orig_sizes, final_mask, y, n)
    elapsed = time.perf_counter() - started

    row = {
        "dataset": path.stem,
        "n": n,
        "n_true_clusters": int(len(np.unique(y[y >= 0]))),
        "n_delaunay_edges": int(len(edge_keys)),
        "n_final_mask_edges": int(final_mask.sum()),
        "baseline_ari": baseline_ari,
        "baseline_graph_ari": baseline_graph_ari,
        "depth4_rule_ari": rule_ari,
        "oracle_start_ari": oracle["start_ari"],
        "oracle_final_ari": oracle["final_ari"],
        "oracle_delta_vs_baseline_graph": oracle["final_ari"] - baseline_graph_ari,
        "oracle_delta_vs_rule": oracle["final_ari"] - rule_ari,
        "n_bridges_tested": oracle["n_bridges_tested"],
        "n_bridges_pruned": oracle["n_bridges_pruned"],
        "seconds": round(elapsed, 2),
    }
    print(
        f"  {path.stem:<38} baseline={baseline_ari:.4f}  rule={rule_ari:.4f}  "
        f"oracle={oracle['final_ari']:.4f}  "
        f"(bridges {oracle['n_bridges_pruned']}/{oracle['n_bridges_tested']} pruned, "
        f"{elapsed:.1f}s)",
        flush=True,
    )
    return row


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=_DEFAULT_DATA_DIR, type=Path)
    parser.add_argument("--out", default=Path("results/edge_ari_oracle"), type=Path)
    parser.add_argument(
        "--datasets", default="",
        help="optional comma-separated filename stems; default is all .npz files",
    )
    args = parser.parse_args()

    files = sorted(args.data.glob("*.npz"))
    requested = {stem.strip() for stem in args.datasets.split(",") if stem.strip()}
    if requested:
        files = [path for path in files if path.stem in requested]
    if not files:
        raise SystemExit(f"no .npz datasets under {args.data}")

    print(f"{len(files)} datasets", flush=True)
    rows = [run_one(path) for path in files]

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out / "edge_ari_oracle.csv")

    print("\n=== mean over datasets ===")
    for key in (
        "baseline_ari", "baseline_graph_ari", "depth4_rule_ari", "oracle_final_ari",
    ):
        print(f"{key:<20} {np.mean([r[key] for r in rows]):.4f}")
    print(f"\nWrote {args.out}/edge_ari_oracle.csv")


if __name__ == "__main__":
    main()
