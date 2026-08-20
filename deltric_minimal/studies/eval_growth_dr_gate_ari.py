#!/usr/bin/env python3
"""Measure the clustering-quality effect of gating the growth phase's
projection choice with ``growth_dr_gate``.

kinit's per-run DR-quality metrics show ``degenerate_graph`` (projected
Delaunay graph collapses: mean degree < 2, or fewer than half-n edges) firing
for 11/31 ``lle15`` runs and 12/32 ``spectral15`` runs, but 0/32 for every
UMAP variant and 0/32 for PCA (see ``studies/growth_dr_trust_tree.py``).
DelTriC's default growth config uses UMAP, so ``growth_dr_gate`` would never
fire there -- there is nothing to gate. To actually exercise the gate this
script deliberately runs the *risky* method (LLE, which produced most of the
degenerate runs in kinit's data) and asks: does checking the gate before
committing to LLE, and falling back to PCA when it fires, recover ARI lost to
degenerate LLE graphs, without giving up the cases where LLE was fine?

Four variants per dataset:

    baseline_umap  -- current default config (dim_reduction="umap")
    risky_lle      -- dim_reduction="lle", no gate
    safe_pca       -- dim_reduction="pca", no gate
    gated_lle      -- try "lle"; if growth_dr_gate flags the LLE projection
                      as degenerate, fall back to "pca" instead

``gated_lle`` is the one under test: it should track ``risky_lle`` where LLE
is fine and ``safe_pca`` where LLE is degenerate.

This repo's optional dependency set (torch, umap-learn, etc.) is required to
import ``utils_pruning.py``; see the repo README for the environment setup.

Usage:
    python eval_growth_dr_gate_ari.py --out results/growth_dr_gate_ari
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from growth_dr_gate import dr_trust_features, predict_degenerate

_DELTRIC_MINIMAL_DIR = Path(__file__).resolve().parents[1]
_DEFAULT_DATA_DIR = _DELTRIC_MINIMAL_DIR / "data"

# Delayed via sys.path so --help works without DelTriC's optional dependency
# set, matching deltric_growth_audit.py's convention.
sys.path.insert(0, str(_DELTRIC_MINIMAL_DIR.parent))
from deltric_minimal.utils_component_growth import cluster_tri
from deltric_minimal.utils_pruning import get_triangles_with_edges

# Growth-mode defaults from run_plot_stages.sh (same base as eval_edge_rule_ari.py).
BASE_CONFIG = dict(
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

VARIANTS = ("baseline_umap", "risky_lle", "safe_pca", "gated_lle")


def _metrics(y_true: np.ndarray, labels: np.ndarray) -> dict:
    labels = np.asarray(labels)
    assigned = labels >= 0
    out = {
        "ari": float(adjusted_rand_score(y_true, labels)),
        "ami": float(adjusted_mutual_info_score(y_true, labels)),
        "n_clusters": int(len(np.unique(labels[assigned]))),
        "noise_fraction": float(np.mean(~assigned)),
    }
    if assigned.sum() > 1:
        out["ari_assigned_only"] = float(adjusted_rand_score(y_true[assigned], labels[assigned]))
    else:
        out["ari_assigned_only"] = float("nan")
    return out


def run_one(path: Path, variant: str) -> dict:
    npz = np.load(path, allow_pickle=False)
    X = StandardScaler().fit_transform(np.asarray(npz["X"], dtype=np.float64))
    y = np.asarray(npz["y"]).astype(np.int64)

    gate_info = {}
    if variant == "baseline_umap":
        dim_reduction = "umap"
    elif variant == "risky_lle":
        dim_reduction = "lle"
    elif variant == "safe_pca":
        dim_reduction = "pca"
    elif variant == "gated_lle":
        _, _, _, X_proj_lle = get_triangles_with_edges(
            X, project_dim=BASE_CONFIG["project_dim"], method="lle",
            back_proj=BASE_CONFIG["back_proj"],
            umap_n_neighbors=BASE_CONFIG["umap_n_neighbors"],
        )
        features = dr_trust_features(X, X_proj_lle)
        degenerate = predict_degenerate(features)
        dim_reduction = "pca" if degenerate else "lle"
        gate_info = {"gate_fired": degenerate, **{f"gate_{k}": v for k, v in features.items()}}
    else:
        raise ValueError(variant)

    started = time.perf_counter()
    labels = cluster_tri(X, dim_reduction=dim_reduction, **BASE_CONFIG)
    elapsed = time.perf_counter() - started

    row = {
        "dataset": path.stem,
        "n": int(len(X)),
        "dim": int(X.shape[1]),
        "n_true_clusters": int(len(np.unique(y[y >= 0]))),
        "variant": variant,
        "dim_reduction_used": dim_reduction,
        "seconds": round(elapsed, 2),
        **gate_info,
    }
    row.update(_metrics(y, labels))
    print(
        f"  {path.stem:<38} {variant:<14} used={dim_reduction:<5} "
        f"ARI={row['ari']:.4f}  AMI={row['ami']:.4f}  k={row['n_clusters']:<3} "
        f"noise={row['noise_fraction']:.3f}  ({elapsed:.1f}s)"
        + (f"  gate_fired={gate_info['gate_fired']}" if gate_info else ""),
        flush=True,
    )
    return row


def write_csv(rows: list[dict], path: Path) -> None:
    import csv

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=_DEFAULT_DATA_DIR, type=Path)
    parser.add_argument("--out", default=Path("results/growth_dr_gate_ari"), type=Path)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    files = sorted(args.data.glob("*.npz"))
    if not files:
        raise SystemExit(f"no .npz datasets under {args.data}")

    tasks = [(path, variant) for path in files for variant in VARIANTS]
    print(f"{len(files)} datasets x {len(VARIANTS)} variants = {len(tasks)} runs, jobs={args.jobs}",
          flush=True)

    if args.jobs == 1:
        rows = [run_one(path, variant) for path, variant in tasks]
    else:
        from joblib import Parallel, delayed

        rows = Parallel(n_jobs=args.jobs, backend="loky", verbose=5)(
            delayed(run_one)(path, variant) for path, variant in tasks
        )
    rows = sorted(rows, key=lambda r: (r["dataset"], r["variant"]))
    write_csv(rows, args.out / "growth_dr_gate_ari.csv")

    by_dataset: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], {})[row["variant"]] = row

    summary = []
    for variant in VARIANTS:
        aris = [d[variant]["ari"] for d in by_dataset.values() if variant in d]
        deltas = [
            d[variant]["ari"] - d["baseline_umap"]["ari"]
            for d in by_dataset.values()
            if variant in d and "baseline_umap" in d
        ]
        summary.append({
            "variant": variant,
            "mean_ari": float(np.mean(aris)),
            "median_ari": float(np.median(aris)),
            "mean_delta_vs_baseline_umap": float(np.mean(deltas)),
            "n_datasets_improved": int(sum(d > 1e-9 for d in deltas)),
            "n_datasets_worsened": int(sum(d < -1e-9 for d in deltas)),
            "worst_delta": float(np.min(deltas)) if deltas else float("nan"),
            "best_delta": float(np.max(deltas)) if deltas else float("nan"),
        })
    write_csv(summary, args.out / "growth_dr_gate_ari_summary.csv")

    n_fired = sum(
        1 for d in by_dataset.values()
        if "gated_lle" in d and d["gated_lle"].get("gate_fired")
    )
    lle_vs_gated = [
        (name, d["risky_lle"]["ari"], d["gated_lle"]["ari"], d["gated_lle"].get("gate_fired"))
        for name, d in by_dataset.items()
        if "risky_lle" in d and "gated_lle" in d
    ]

    (args.out / "config.json").write_text(json.dumps({"base_config": BASE_CONFIG}, indent=2))

    print("\n=== summary (mean over datasets) ===")
    for row in summary:
        print(
            f"{row['variant']:<14} mean ARI {row['mean_ari']:.4f}  "
            f"delta-vs-umap {row['mean_delta_vs_baseline_umap']:+.4f}  "
            f"improved {row['n_datasets_improved']}  worsened {row['n_datasets_worsened']}"
        )
    print(f"\ngate fired on {n_fired}/{len(by_dataset)} datasets")
    print("\n=== risky_lle vs gated_lle, per dataset ===")
    for name, lle_ari, gated_ari, fired in sorted(lle_vs_gated):
        flag = "FIRED->PCA" if fired else "kept LLE"
        print(f"  {name:<38} lle={lle_ari:.4f}  gated={gated_ari:.4f}  "
              f"delta={gated_ari - lle_ari:+.4f}  [{flag}]")

    print(f"\nWrote {args.out}/growth_dr_gate_ari.csv")


if __name__ == "__main__":
    main()
