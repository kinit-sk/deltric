#!/usr/bin/env python3
"""Measure the wall-clock cost the Dijkstra anomaly-reassignment stage adds.

The "old" pipeline is ``cluster_tri`` with
``component_growth_anomaly_reassign=False``: the component-growth graph is
built, seeded, grown and merged, and whatever is left with label ``-1``
simply stays noise. The "new" pipeline adds one more stage on top --
competitive penalized-Dijkstra reclamation of those noise points, living in
``utils_component_growth._assign_anomalies_dijkstra`` (see
``studies/growth_dijkstra_anomaly_assign.py`` for the ARI-side evaluation of
the same stage; this script is the time-side counterpart).

Rather than running ``cluster_tri`` twice (once per flag value) -- which
would also re-pay UMAP/growth twice and mix its run-to-run jitter into the
comparison -- this script runs the graph-building part exactly once per
repeat and then calls ``_assign_anomalies_dijkstra`` directly on its output.
That isolates the added stage's own cost from the (identical, deterministic
given ``random_state``) cost of everything before it, so
``dijkstra_seconds`` is a direct measurement of "how much does Dijkstra
add", not a noisy difference of two large numbers.

Usage:
    python studies/dijkstra_timing_study.py --out studies/results/dijkstra_timing
    python studies/dijkstra_timing_study.py --out studies/results/dijkstra_timing --repeats 5 --jobs 8
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

_DELTRIC_MINIMAL_DIR = Path(__file__).resolve().parents[1]
_DEFAULT_DATA_DIR = _DELTRIC_MINIMAL_DIR / "data"
sys.path.insert(0, str(_DELTRIC_MINIMAL_DIR.parent))

from deltric_minimal.utils_component_growth import (  # noqa: E402
    _assign_anomalies_dijkstra,
    cluster_tri,
)

# Same production config used throughout this repo's studies
# (eval_edge_rule_ari.py, edge_ari_oracle_pruning.py, growth_dijkstra_anomaly_assign.py).
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


def run_one(path: Path, penalty_power: float, stop_ratio: float, repeats: int) -> dict:
    npz = np.load(path, allow_pickle=False)
    X = StandardScaler().fit_transform(np.asarray(npz["X"], dtype=np.float64))
    n = len(X)

    base_times: list[float] = []
    dijkstra_times: list[float] = []
    n_anomalies_in = n_anomalies_out = n_rounds = None

    for _ in range(repeats):
        t0 = time.perf_counter()
        cluster_tri(X, **BASE_CONFIG, component_growth_anomaly_reassign=False)
        t1 = time.perf_counter()
        state = cluster_tri.last_component_growth
        merged_labels = state["merged_labels"]

        t2 = time.perf_counter()
        _, diag = _assign_anomalies_dijkstra(
            merged_labels, state["edge_keys"], state["projected_edge_sizes"], n,
            penalty_power=penalty_power, stop_ratio=stop_ratio,
        )
        t3 = time.perf_counter()

        base_times.append(t1 - t0)
        dijkstra_times.append(t3 - t2)
        n_anomalies_in, n_anomalies_out, n_rounds = (
            diag["n_anomalies_in"], diag["n_anomalies_out"], diag["n_rounds"]
        )

    base_med = statistics.median(base_times)
    dij_med = statistics.median(dijkstra_times)
    row = {
        "dataset": path.stem,
        "n": n,
        "dim": int(X.shape[1]),
        "repeats": repeats,
        "base_seconds_median": round(base_med, 4),
        "base_seconds_min": round(min(base_times), 4),
        "base_seconds_max": round(max(base_times), 4),
        "dijkstra_seconds_median": round(dij_med, 4),
        "dijkstra_seconds_min": round(min(dijkstra_times), 4),
        "dijkstra_seconds_max": round(max(dijkstra_times), 4),
        "total_seconds_median": round(base_med + dij_med, 4),
        "overhead_pct": round(100.0 * dij_med / base_med, 2) if base_med > 0 else float("nan"),
        "n_anomalies_in": n_anomalies_in,
        "n_anomalies_out": n_anomalies_out,
        "n_rounds": n_rounds,
        "penalty_power": penalty_power,
        "stop_ratio": stop_ratio,
    }
    print(
        f"  {path.stem:<38} n={n:<7} base={base_med:.3f}s  "
        f"dijkstra={dij_med:.3f}s (+{row['overhead_pct']:.1f}%)  "
        f"{n_anomalies_in}->{n_anomalies_out} anomalies, {n_rounds} rounds",
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
    parser.add_argument("--out", default=Path("results/dijkstra_timing"), type=Path)
    parser.add_argument("--penalty-power", type=float, default=2.0)
    parser.add_argument("--stop-ratio", type=float, default=3.0)
    parser.add_argument("--repeats", type=int, default=3, help="repeats per dataset; median is reported")
    parser.add_argument("--jobs", type=int, default=1, help="parallel datasets via joblib")
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

    print(
        f"{len(files)} datasets  penalty_power={args.penalty_power}  "
        f"stop_ratio={args.stop_ratio}  repeats={args.repeats}  jobs={args.jobs}",
        flush=True,
    )

    if args.jobs == 1:
        rows = [run_one(path, args.penalty_power, args.stop_ratio, args.repeats) for path in files]
    else:
        from joblib import Parallel, delayed

        rows = Parallel(n_jobs=args.jobs, backend="loky", verbose=5)(
            delayed(run_one)(path, args.penalty_power, args.stop_ratio, args.repeats) for path in files
        )
    rows = sorted(rows, key=lambda r: r["n"])

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out / "dijkstra_timing.csv")
    (args.out / "config.json").write_text(json.dumps({
        "base_config": BASE_CONFIG,
        "penalty_power": args.penalty_power,
        "stop_ratio": args.stop_ratio,
        "repeats": args.repeats,
    }, indent=2))

    overheads = [r["overhead_pct"] for r in rows if not np.isnan(r["overhead_pct"])]
    print("\n=== summary ===")
    print(f"mean base_seconds_median      {np.mean([r['base_seconds_median'] for r in rows]):.4f}")
    print(f"mean dijkstra_seconds_median  {np.mean([r['dijkstra_seconds_median'] for r in rows]):.4f}")
    print(f"mean overhead_pct             {np.mean(overheads):.2f}%")
    print(f"median overhead_pct           {np.median(overheads):.2f}%")
    print(f"max overhead_pct              {np.max(overheads):.2f}%  ({rows[np.argmax(overheads)]['dataset']})")
    print(f"\nWrote {args.out}/dijkstra_timing.csv")


if __name__ == "__main__":
    main()
