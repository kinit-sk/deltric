#!/usr/bin/env python3
"""Measure ARI and anomaly-detection F1 on the real Gaia open-cluster datasets.

Unlike the synthetic ``blobs_*``/``__anom10`` sets used elsewhere in
``studies/``, ``data/gaia_*.npz`` (built by ``make_gaia_cluster_datasets.py``)
label real Gaia DR3 field stars against an independent HDBSCAN-based
membership catalogue (Hunt & Reffert 2023): ``y == 0`` is a cluster member,
``y == -1`` is field/noise. Ground truth is binary per file (one true
cluster + noise), so ARI here scores "did the pipeline separate the real
members from the field" rather than multi-cluster structure recovery.

Runs ``cluster_tri`` with the same ``BASE_CONFIG`` used by
``anomaly_cost_gate_study.py`` / ``anomaly_f1_dijkstra_vs_baseline.py``, once
with the production cost-gate Dijkstra reassignment on (p=3, reach=25, q=4 --
the config wired in as default, see commit 135dd83) and once with
reassignment off, so the reassignment stage's effect on real (non-synthetic)
data is visible alongside the raw numbers. Also fits the repo's standard
original-space HDBSCAN baseline (``hdbscan_comparison.fit_hdbscan``, defaults
min_cluster_size=10, eom) as a reference point.

Usage:
    uv run studies/eval_gaia_anomaly.py
    uv run studies/eval_gaia_anomaly.py --out studies/results/gaia_anomaly.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

_DELTRIC_MINIMAL_DIR = Path(__file__).resolve().parents[1]
_DEFAULT_DATA_DIR = _DELTRIC_MINIMAL_DIR / "data"
sys.path.insert(0, str(_DELTRIC_MINIMAL_DIR.parent))

from deltric_minimal.utils_component_growth import cluster_tri  # noqa: E402
from deltric_minimal.studies.anomaly_cost_gate_study import BASE_CONFIG  # noqa: E402
from deltric_minimal.hdbscan_comparison import fit_hdbscan  # noqa: E402

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


def scores(y: np.ndarray, assigned: np.ndarray) -> dict:
    true_anomaly = y < 0
    predicted = assigned < 0
    real = ~true_anomaly
    return {
        "precision": float(precision_score(true_anomaly, predicted, zero_division=0)),
        "recall": float(recall_score(true_anomaly, predicted, zero_division=0)),
        "f1": float(f1_score(true_anomaly, predicted, zero_division=0)),
        "ari_real": float(adjusted_rand_score(y[real], assigned[real])),
        "ari_all": float(adjusted_rand_score(y, assigned)),
    }


def run_one(path: Path) -> dict:
    npz = np.load(path, allow_pickle=False)
    X = StandardScaler().fit_transform(np.asarray(npz["X"], dtype=np.float64))
    y = np.asarray(npz["y"]).astype(np.int64)
    n, d = X.shape

    started = time.perf_counter()
    reassigned = cluster_tri(
        X, **BASE_CONFIG,
        component_growth_anomaly_reassign=True,
        component_growth_anomaly_reassign_penalty_power=PENALTY_POWER,
        component_growth_anomaly_reassign_reach=REACH,
        component_growth_anomaly_reassign_density_power=DENSITY_POWER,
    )
    elapsed = time.perf_counter() - started
    baseline = cluster_tri.last_component_growth["merged_labels"]

    hdb_labels = fit_hdbscan(
        X,
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        cluster_selection_method=HDBSCAN_SELECTION_METHOD,
        cluster_selection_epsilon=HDBSCAN_CLUSTER_SELECTION_EPSILON,
    )

    row = {"dataset": path.stem, "n": n, "d": d,
           "n_true_anomaly": int((y < 0).sum()), "elapsed_s": elapsed}
    for label, assigned in (("baseline", baseline), ("cost_gate", reassigned),
                             ("hdbscan", hdb_labels)):
        for k, v in scores(y, assigned).items():
            row[f"{label}_{k}"] = v
    return row


def report(rows: list[dict]) -> None:
    header = (f"{'dataset':<16}{'n':>6}{'n_anom':>8}"
              f"{'base_f1':>9}{'cg_f1':>8}{'hdb_f1':>8}"
              f"{'base_ari':>10}{'cg_ari':>9}{'hdb_ari':>9}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['dataset']:<16}{r['n']:>6}{r['n_true_anomaly']:>8}"
              f"{r['baseline_f1']:>9.3f}{r['cost_gate_f1']:>8.3f}{r['hdbscan_f1']:>8.3f}"
              f"{r['baseline_ari_all']:>10.3f}{r['cost_gate_ari_all']:>9.3f}{r['hdbscan_ari_all']:>9.3f}")
    print("-" * len(header))
    mean = lambda key: np.mean([r[key] for r in rows])
    print(f"{'mean':<16}{'':>6}{'':>8}"
          f"{mean('baseline_f1'):>9.3f}{mean('cost_gate_f1'):>8.3f}{mean('hdbscan_f1'):>8.3f}"
          f"{mean('baseline_ari_all'):>10.3f}{mean('cost_gate_ari_all'):>9.3f}{mean('hdbscan_ari_all'):>9.3f}")


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=_DEFAULT_DATA_DIR)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for stem in GAIA_STEMS:
        path = args.data / f"{stem}.npz"
        if not path.exists():
            print(f"skip (missing): {path}")
            continue
        print(f"running {stem} ...", flush=True)
        rows.append(run_one(path))

    if not rows:
        print("no gaia datasets found")
        return

    report(rows)
    if args.out:
        write_csv(rows, args.out)


if __name__ == "__main__":
    main()
