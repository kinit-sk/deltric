#!/usr/bin/env python3
"""Audit DelTriC's component-growth backend on its curated datasets.

This deliberately calls the production backend in ``deltric_minimal`` rather
than reimplementing its graph logic.  It records graph counts and the same
large-component ARI used by the plotting wrapper, without producing plots.

This repo's optional dependency set (torch, umap-learn, etc.) is required to
import ``utils_pruning.py``; see the repo README for the environment setup.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

_DELTRIC_MINIMAL_DIR = Path(__file__).resolve().parents[1]
_DEFAULT_DATA_DIR = _DELTRIC_MINIMAL_DIR / "data"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA_DIR)
    parser.add_argument("--out", type=Path, default=Path("results/deltric_growth_audit.csv"))
    parser.add_argument("--methods", default="umap,pca")
    parser.add_argument("--scopes", default="edge")
    parser.add_argument(
        "--datasets", default="",
        help="optional comma-separated filename stems; default is all .npz files",
    )
    parser.add_argument("--umap-neighbors", type=int, default=15)
    parser.add_argument("--umap-epochs", type=int, default=100)
    return parser.parse_args()


def _component_labels(component_labels: np.ndarray, min_edges: int) -> np.ndarray:
    """Map component IDs to retained cluster IDs; small components become -1."""
    component_labels = np.asarray(component_labels, dtype=np.int64)
    result = np.full(len(component_labels), -1, dtype=np.int64)
    next_id = 0
    for component in np.unique(component_labels):
        points = component_labels == component
        if int(points.sum()) >= int(min_edges):
            result[points] = next_id
            next_id += 1
    return result


def main() -> None:
    args = _parse_args()
    data_dir = args.data_dir.resolve()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # This import is intentionally delayed so --help works without DelTriC's
    # optional dependency set and so the module path is explicit in the audit.
    sys.path.insert(0, str(_DELTRIC_MINIMAL_DIR.parent))
    from deltric_minimal.utils_component_growth import component_growth_graph

    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    scopes = [scope.strip() for scope in args.scopes.split(",") if scope.strip()]
    rows: list[dict[str, object]] = []
    files = sorted(data_dir.glob("*.npz"))
    requested = {stem.strip() for stem in args.datasets.split(",") if stem.strip()}
    if requested:
        files = [path for path in files if path.stem in requested]
    for path in files:
        npz = np.load(path, allow_pickle=False)
        X = StandardScaler().fit_transform(np.asarray(npz["X"], dtype=np.float64))
        y = None
        if "y" in npz.files:
            y = np.asarray(npz["y"], dtype=np.int64)
        elif "y_clean" in npz.files:
            y = np.asarray(npz["y_clean"], dtype=np.int64)

        for method, scope in (
            (method, scope) for method in methods for scope in scopes
        ):
            started = time.perf_counter()
            try:
                state = component_growth_graph(
                    X,
                    dim_reduction=method,
                    project_dim=2,
                    umap_n_epochs=args.umap_epochs,
                    umap_n_neighbors=args.umap_neighbors,
                    hard_limit=0.9,
                    seed_hard_limit=0.25,
                    growth_hard_limit=0.9,
                    mild_limit=0.5,
                    initial_relation="union",
                    growth_relation="union",
                    knn_k=50,
                    component_min_edges=10,
                    hard_gate_mode="knn_relaxed",
                    seed_relaxed_hard_limit=0.75,
                    growth_relaxed_hard_limit=1.7,
                    local_knn_selectivity_threshold=0.1,
                    local_knn_selectivity_scope=scope,
                    bridge_pruning=False,
                    bridge_max_hops=10,
                    open_space_relaxation=False,
                    outer_long_ratio=3.0,
                    outer_relaxation=2.0,
                    outer_transition_width=0.4054651081,
                    projected_hard_limit=True,
                    multiplicative_hard_limit=False,
                )
                labels = _component_labels(state["labels"], 10)
                ari = float("nan")
                if y is not None and len(y) == len(labels):
                    ari = float(adjusted_rand_score(y, labels))
                rows.append({
                    "dataset": path.stem,
                    "method": method,
                    "scope": scope,
                    "status": "ok",
                    "n": len(X),
                    "dimensions": X.shape[1],
                    "delaunay_edges": len(state["edge_keys"]),
                    "hard_allowed": int(np.count_nonzero(state["hard_allowed"])),
                    "initial_edges": int(np.count_nonzero(state["initial_mask"])),
                    "final_edges": int(np.count_nonzero(state["final_mask"])),
                    "initial_components": int(len(np.unique(state["labels"]))),
                    "large_components": int(np.count_nonzero(state["large_components"])),
                    "growth_waves": len(state["growth_waves"]),
                    "automatic_growth_edges": int(np.count_nonzero(state["automatic_growth_mask"])),
                    "blue_growth_edges": int(np.count_nonzero(state["blue_growth_mask"])),
                    "local_selectivity_mean": float(np.mean(state["local_knn_selectivity"])),
                    "large_component_ari": ari,
                    "seconds": time.perf_counter() - started,
                    "error": "",
                })
                print(
                    f"{path.stem} {method:8s} {scope:8s} edges={len(state['edge_keys']):5d} "
                    f"initial={int(np.count_nonzero(state['initial_mask'])):5d} "
                    f"final={int(np.count_nonzero(state['final_mask'])):5d} "
                    f"components={int(np.count_nonzero(state['large_components'])):3d} "
                    f"ARI={ari:.3f}"
                )
            except Exception as exc:  # record algorithm failures, keep the suite going
                rows.append({
                    "dataset": path.stem,
                    "method": method,
                    "scope": scope,
                    "status": "failed",
                    "n": len(X),
                    "dimensions": X.shape[1],
                    "delaunay_edges": "",
                    "hard_allowed": "",
                    "initial_edges": "",
                    "final_edges": "",
                    "initial_components": "",
                    "large_components": "",
                    "growth_waves": "",
                    "automatic_growth_edges": "",
                    "blue_growth_edges": "",
                    "local_selectivity_mean": "",
                    "large_component_ari": "",
                    "seconds": time.perf_counter() - started,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                print(f"{path.stem} {method:8s} {scope:8s} FAILED: {type(exc).__name__}: {exc}")

    fields = list(rows[0]) if rows else []
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
