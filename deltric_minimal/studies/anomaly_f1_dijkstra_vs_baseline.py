#!/usr/bin/env python3
"""Compare anomaly-detection F1 with vs. without the Dijkstra reclamation stage.

Restricted to the datasets that actually carry ground-truth noise labels
(``y == -1``): ``blobs_with_noise__*`` and ``blobs_meeples__*`` (16 files).
``02_v3_overlap_blobs_10d_4000_k6.npz`` is excluded -- its "outliers" are
mislabeled points inside real clusters (``outlier_mask`` / ``y_clean``), not
spatial noise, so it isn't the kind of point this stage tries to reclaim.

For each dataset, runs ``cluster_tri`` once with
``component_growth_anomaly_reassign=False`` (baseline: whatever the
component-growth/pruning stages left as ``label == -1`` is the final
noise prediction) and once with it ``True`` (Dijkstra: those noise points
get a chance to be reclaimed by a reachable cluster; whatever is still
``-1`` afterwards is the final noise prediction). Anomaly detection is
scored as a binary task (positive class = "is noise") against the true
``y == -1`` mask, reporting precision/recall/F1 for both variants.

Usage:
    python studies/anomaly_f1_dijkstra_vs_baseline.py --out studies/results/anomaly_f1
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

_DELTRIC_MINIMAL_DIR = Path(__file__).resolve().parents[1]
_DEFAULT_DATA_DIR = _DELTRIC_MINIMAL_DIR / "data"
sys.path.insert(0, str(_DELTRIC_MINIMAL_DIR.parent))

from deltric_minimal.utils_component_growth import cluster_tri  # noqa: E402

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

DATASET_STEMS = [
    "blobs_with_noise__d2__n500", "blobs_with_noise__d2__n2000",
    "blobs_with_noise__d5__n500", "blobs_with_noise__d5__n2000",
    "blobs_with_noise__d50__n500", "blobs_with_noise__d50__n2000",
    "blobs_with_noise__d100__n500", "blobs_with_noise__d100__n2000",
    "blobs_meeples__d2__n500", "blobs_meeples__d2__n2000",
    "blobs_meeples__d5__n500", "blobs_meeples__d5__n2000",
    "blobs_meeples__d50__n500", "blobs_meeples__d50__n2000",
    "blobs_meeples__d100__n500", "blobs_meeples__d100__n2000",
]

# Kind codes written by studies/make_anomaly_datasets.py into ``anomaly_kind``.
KIND_NAMES = {1: "uniform", 2: "far", 3: "bridge", 4: "micro", 5: "subspace"}


def scores(y_true_anomaly: np.ndarray, pred_labels: np.ndarray) -> dict:
    pred_anomaly = pred_labels < 0
    return {
        "precision": float(precision_score(y_true_anomaly, pred_anomaly, zero_division=0)),
        "recall": float(recall_score(y_true_anomaly, pred_anomaly, zero_division=0)),
        "f1": float(f1_score(y_true_anomaly, pred_anomaly, zero_division=0)),
        "n_pred_anomaly": int(pred_anomaly.sum()),
    }


def run_one(path: Path, penalty_power: float, stop_ratio: float) -> dict:
    npz = np.load(path, allow_pickle=False)
    X = StandardScaler().fit_transform(np.asarray(npz["X"], dtype=np.float64))
    y = np.asarray(npz["y"]).astype(np.int64)
    y_true_anomaly = y < 0
    # Generated datasets record which kind of anomaly each point is, so a
    # single run also yields recall broken down by kind.
    kind = np.asarray(npz["anomaly_kind"]) if "anomaly_kind" in npz.files else None
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
    baseline_labels = state["merged_labels"]

    base = scores(y_true_anomaly, baseline_labels)
    dijk = scores(y_true_anomaly, assigned_labels)

    row = {
        "dataset": path.stem,
        "n": n,
        "n_true_anomaly": int(y_true_anomaly.sum()),
        "baseline_precision": base["precision"],
        "baseline_recall": base["recall"],
        "baseline_f1": base["f1"],
        "baseline_n_pred_anomaly": base["n_pred_anomaly"],
        "dijkstra_precision": dijk["precision"],
        "dijkstra_recall": dijk["recall"],
        "dijkstra_f1": dijk["f1"],
        "dijkstra_n_pred_anomaly": dijk["n_pred_anomaly"],
        "delta_f1": dijk["f1"] - base["f1"],
        "seconds": round(elapsed, 2),
    }
    if kind is not None:
        for code, name in KIND_NAMES.items():
            mask = kind == code
            if not mask.any():
                continue
            row[f"baseline_recall_{name}"] = float((baseline_labels[mask] < 0).mean())
            row[f"dijkstra_recall_{name}"] = float((assigned_labels[mask] < 0).mean())
    print(
        f"  {path.stem:<30} true={row['n_true_anomaly']:<4} "
        f"baseline F1={base['f1']:.4f} (P={base['precision']:.3f} R={base['recall']:.3f})  "
        f"dijkstra F1={dijk['f1']:.4f} (P={dijk['precision']:.3f} R={dijk['recall']:.3f})  "
        f"delta={row['delta_f1']:+.4f}  {elapsed:.1f}s",
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
    parser.add_argument("--glob", nargs="*", default=None,
                        help="glob pattern(s) under --data to score instead of the "
                             "built-in list, e.g. --glob '*__anom10.npz'")
    parser.add_argument("--out", default=Path("studies/results/anomaly_f1"), type=Path)
    parser.add_argument("--penalty-power", type=float, default=2.0)
    parser.add_argument("--stop-ratio", type=float, default=3.0)
    args = parser.parse_args()

    if args.glob:
        files = sorted({f for pat in args.glob for f in args.data.glob(pat)})
        if not files:
            raise SystemExit(f"no datasets matched {args.glob} under {args.data}")
    else:
        files = [args.data / f"{stem}.npz" for stem in DATASET_STEMS]
        missing = [f for f in files if not f.exists()]
        if missing:
            raise SystemExit(f"missing datasets: {missing}")

    print(f"{len(files)} noise-labeled datasets  penalty_power={args.penalty_power}  stop_ratio={args.stop_ratio}", flush=True)
    rows = [run_one(path, args.penalty_power, args.stop_ratio) for path in files]

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out / "anomaly_f1_dijkstra_vs_baseline.csv")

    print("\n=== mean over datasets ===")
    keys = ["baseline_precision", "baseline_recall", "baseline_f1",
            "dijkstra_precision", "dijkstra_recall", "dijkstra_f1", "delta_f1"]
    keys += [k for k in rows[0] if k.startswith(("baseline_recall_", "dijkstra_recall_"))]
    for key in keys:
        vals = [r[key] for r in rows if key in r]
        print(f"{key:<26} {np.mean(vals):.4f}  (n={len(vals)})")
    print(f"\nWrote {args.out}/anomaly_f1_dijkstra_vs_baseline.csv")


if __name__ == "__main__":
    main()
