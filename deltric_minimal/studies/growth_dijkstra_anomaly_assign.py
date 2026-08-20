#!/usr/bin/env python3
"""Eval harness for the anomaly-reassignment stage now built into ``cluster_tri``.

The algorithm itself (competitive penalized-Dijkstra reclamation of points
left as noise) is no longer implemented here -- it lives in
``utils_component_growth._assign_anomalies_dijkstra`` and is wired into the
production pipeline as ``component_growth_anomaly_reassign`` (default
``False``, so existing callers/baselines are unaffected). This script just
runs ``cluster_tri`` with the flag on and off across the curated datasets
and reports the ARI delta, the way every other eval script in this repo
compares a pipeline variant against the baseline.

The core idea (from the square-with-corner-clusters mental model): if every
cluster expands outward at the "same rate" and a noise point sits between
two clusters, the cluster that can reach it via a chain of short edges
should win over a cluster that can only reach it via a couple of long
edges -- even if the two paths sum to the same total length. Each edge's
weight is penalized superlinearly the further it sits above the claiming
cluster's own typical edge length::

    weight(e) = length(e) * (length(e) / avg_c) ** penalty_power

Two choices deliberately differ from ``studies/growth_dijkstra_penalty.py``
(the main-growth version of this same competitive-Dijkstra idea, which
collapses whole high-dimensional datasets into one cluster -- see that
file's docstring): lengths are taken in projected (UMAP) space, not
original high-dim space (original-space distances concentrate in high
dimensions and lose their power to discriminate); and each cluster's
``avg_c`` is fixed once from its own edges and never updated as it absorbs
noise points, since main clusters are already trusted and letting a noisy
early claim drag the yardstick around would cascade into more bad claims.
Clusters also never merge into each other here -- an earlier version
allowed it and a handful of short frontier edges cascade-fused 17
well-separated main clusters down to 3-9 on ``blobs_8d``, dropping ARI from
0.24 to 0.01-0.19.

Usage:
    python studies/growth_dijkstra_anomaly_assign.py --out studies/results/growth_dijkstra_anomaly_assign
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
sys.path.insert(0, str(_DELTRIC_MINIMAL_DIR.parent))

from deltric_minimal.utils_component_growth import cluster_tri  # noqa: E402

# Same production config used throughout this repo's studies
# (eval_edge_rule_ari.py, edge_ari_oracle_pruning.py, growth_dijkstra_penalty.py).
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


def run_one(path: Path, penalty_power: float, stop_ratio: float) -> dict:
    npz = np.load(path, allow_pickle=False)
    X = StandardScaler().fit_transform(np.asarray(npz["X"], dtype=np.float64))
    y = np.asarray(npz["y"]).astype(np.int64)
    n = len(X)

    started = time.perf_counter()
    assigned_labels = cluster_tri(
        X, **BASE_CONFIG,
        component_growth_anomaly_reassign=True,
        component_growth_anomaly_reassign_penalty_power=penalty_power,
        component_growth_anomaly_reassign_stop_ratio=stop_ratio,
    )
    elapsed = time.perf_counter() - started
    state = cluster_tri.last_component_growth
    diag = state["anomaly_reassign"]

    baseline_labels = state["merged_labels"]
    baseline_ari = float(adjusted_rand_score(y, baseline_labels))
    assigned_ari = float(adjusted_rand_score(y, assigned_labels))

    row = {
        "dataset": path.stem,
        "n": n,
        "n_true_clusters": int(len(np.unique(y[y >= 0]))),
        "n_main_clusters": int(len(np.unique(baseline_labels[baseline_labels >= 0]))),
        "n_anomalies_in": diag["n_anomalies_in"],
        "n_anomalies_out": diag["n_anomalies_out"],
        "n_rounds": diag["n_rounds"],
        "penalty_power": penalty_power,
        "stop_ratio": stop_ratio,
        "baseline_ari": baseline_ari,
        "assigned_ari": assigned_ari,
        "delta_vs_baseline": assigned_ari - baseline_ari,
        "seconds": round(elapsed, 2),
    }
    print(
        f"  {path.stem:<38} baseline={baseline_ari:.4f}  "
        f"assigned={assigned_ari:.4f}  (delta={row['delta_vs_baseline']:+.4f}, "
        f"{diag['n_anomalies_in']}->{diag['n_anomalies_out']} anomalies, "
        f"{diag['n_rounds']} rounds, {elapsed:.1f}s)",
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
    parser.add_argument("--out", default=Path("results/growth_dijkstra_anomaly_assign"), type=Path)
    parser.add_argument("--penalty-power", type=float, default=2.0)
    parser.add_argument("--stop-ratio", type=float, default=3.0)
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

    print(f"{len(files)} datasets  penalty_power={args.penalty_power}  stop_ratio={args.stop_ratio}", flush=True)
    rows = [run_one(path, args.penalty_power, args.stop_ratio) for path in files]

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out / "growth_dijkstra_anomaly_assign.csv")

    print("\n=== mean over datasets ===")
    for key in ("baseline_ari", "assigned_ari", "delta_vs_baseline"):
        print(f"{key:<20} {np.mean([r[key] for r in rows]):.4f}")
    print(f"\nWrote {args.out}/growth_dijkstra_anomaly_assign.csv")


if __name__ == "__main__":
    main()
