#!/usr/bin/env python3
"""Measure the clustering-quality effect of the learned depth-4 edge rule.

Edge-level AUC is not cluster quality.  The rule extracted in
``DELTRIC_RULE_EXTRACTION_REPORT.md`` scores 0.822 grouped-holdout AUC and
retains 97.3% of intra-cluster edges, but that says nothing about whether
applying it inside component growth produces better clusters.  This script runs
the 12 curated datasets through ``cluster_tri`` with and without the rule and
compares ARI.

Defaults mirror ``run_plot_stages.sh`` growth mode so the baseline is the
configuration actually in use.

Usage:
    python eval_edge_rule_ari.py --out results/edge_rule_ari
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from edge_rule import RULE_KNN_K, depth4_rule, prune_mask_from_features, rule_features
from utils_component_growth import cluster_tri

# Growth-mode defaults from run_plot_stages.sh.
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

VARIANTS = (
    ("baseline", None, "both"),
    ("rule_both", depth4_rule, "both"),
    ("rule_seed_only", depth4_rule, "seed"),
    ("rule_growth_only", depth4_rule, "growth"),
)


def _metrics(y_true: np.ndarray, labels: np.ndarray) -> dict:
    labels = np.asarray(labels)
    assigned = labels >= 0
    out = {
        "ari": float(adjusted_rand_score(y_true, labels)),
        "ami": float(adjusted_mutual_info_score(y_true, labels)),
        "n_clusters": int(len(np.unique(labels[assigned]))),
        "noise_fraction": float(np.mean(~assigned)),
    }
    # ARI computed only over points the algorithm actually assigned, so a run
    # that wins by declaring everything noise cannot hide behind the headline.
    if assigned.sum() > 1:
        out["ari_assigned_only"] = float(
            adjusted_rand_score(y_true[assigned], labels[assigned])
        )
    else:
        out["ari_assigned_only"] = float("nan")
    return out


def _rule_edge_diagnostics(state: dict, y_true: np.ndarray) -> dict:
    """Score the rule against ground truth on this dataset's own graph."""
    edge_keys = np.asarray(state["edge_keys"])
    u, v = edge_keys[:, 0], edge_keys[:, 1]
    valid = (y_true[u] >= 0) & (y_true[v] >= 0)
    bridge = valid & (y_true[u] != y_true[v])
    intra = valid & (y_true[u] == y_true[v])

    mask = state.get("edge_rule_prune_mask")
    if mask is None:
        features = rule_features(
            state["_X"], state["_X_proj"], edge_keys,
            state["orig_edge_sizes"], state["projected_edge_sizes"], RULE_KNN_K,
        )
        mask = prune_mask_from_features(features)
    mask = np.asarray(mask, dtype=bool)

    pruned = int(mask.sum())
    return {
        "n_edges": int(len(edge_keys)),
        "rule_pruned": pruned,
        "rule_prune_fraction": float(pruned / max(len(edge_keys), 1)),
        "bridge_rate": float(bridge.sum() / max(valid.sum(), 1)),
        "rule_bridge_precision": float((mask & bridge).sum() / max(pruned, 1)),
        "rule_bridge_recall": float((mask & bridge).sum() / max(bridge.sum(), 1)),
        "rule_intra_retention": float((~mask & intra).sum() / max(intra.sum(), 1)),
    }


def run_one(path: Path, variant: str) -> dict:
    """One (dataset, variant) cell.  Self-contained so it can be dispatched."""
    rule, scope = next((r, s) for name, r, s in VARIANTS if name == variant)

    npz = np.load(path, allow_pickle=False)
    X = StandardScaler().fit_transform(np.asarray(npz["X"], dtype=np.float64))
    y = np.asarray(npz["y"]).astype(np.int64)

    started = time.perf_counter()
    labels = cluster_tri(
        X,
        component_growth_edge_rule=rule,
        component_growth_edge_rule_scope=scope,
        **BASE_CONFIG,
    )
    elapsed = time.perf_counter() - started
    state = cluster_tri.last_component_growth
    state["_X"], state["_X_proj"] = X, state.get("X_proj")

    row = {
        "dataset": path.stem,
        "n": int(len(X)),
        "dim": int(X.shape[1]),
        "n_true_clusters": int(len(np.unique(y[y >= 0]))),
        "variant": variant,
        "seconds": round(elapsed, 2),
    }
    row.update(_metrics(y, labels))
    try:
        row.update(_rule_edge_diagnostics(state, y))
    except Exception as exc:  # diagnostics must never sink the measurement
        row["diagnostics_error"] = repr(exc)
    print(
        f"  {path.stem:<38} {variant:<17} ARI={row['ari']:.4f}  "
        f"AMI={row['ami']:.4f}  k={row['n_clusters']:<3} "
        f"noise={row['noise_fraction']:.3f}  "
        f"pruned={row.get('rule_prune_fraction', float('nan')):.3f}  "
        f"({elapsed:.1f}s)",
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
    parser.add_argument("--data", default="data", type=Path)
    parser.add_argument("--out", default="results/edge_rule_ari", type=Path)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    files = sorted(args.data.glob("*.npz"))
    if not files:
        raise SystemExit(f"no .npz datasets under {args.data}")

    tasks = [(path, variant) for path in files for variant, _, _ in VARIANTS]
    print(f"{len(files)} datasets x {len(VARIANTS)} variants = {len(tasks)} runs, "
          f"jobs={args.jobs}", flush=True)

    if args.jobs == 1:
        rows = [run_one(path, variant) for path, variant in tasks]
    else:
        from joblib import Parallel, delayed

        rows = Parallel(n_jobs=args.jobs, backend="loky", verbose=5)(
            delayed(run_one)(path, variant) for path, variant in tasks
        )
    rows = sorted(rows, key=lambda r: (r["dataset"], r["variant"]))
    write_csv(rows, args.out / "edge_rule_ari.csv")

    # Paired summary: the per-dataset delta is the number that matters, since
    # datasets differ far more from each other than variants do.
    summary = []
    by_dataset: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], {})[row["variant"]] = row
    for variant, _, _ in VARIANTS:
        aris = [d[variant]["ari"] for d in by_dataset.values() if variant in d]
        deltas = [
            d[variant]["ari"] - d["baseline"]["ari"]
            for d in by_dataset.values()
            if variant in d and "baseline" in d
        ]
        summary.append({
            "variant": variant,
            "mean_ari": float(np.mean(aris)),
            "median_ari": float(np.median(aris)),
            "mean_delta_vs_baseline": float(np.mean(deltas)),
            "n_datasets_improved": int(sum(d > 1e-9 for d in deltas)),
            "n_datasets_worsened": int(sum(d < -1e-9 for d in deltas)),
            "worst_delta": float(np.min(deltas)),
            "best_delta": float(np.max(deltas)),
        })
    write_csv(summary, args.out / "edge_rule_ari_summary.csv")
    (args.out / "config.json").write_text(
        json.dumps({"base_config": BASE_CONFIG, "rule_knn_k": RULE_KNN_K}, indent=2)
    )

    print("\n=== summary (mean over datasets) ===")
    for row in summary:
        print(
            f"{row['variant']:<17} mean ARI {row['mean_ari']:.4f}  "
            f"delta {row['mean_delta_vs_baseline']:+.4f}  "
            f"improved {row['n_datasets_improved']}  "
            f"worsened {row['n_datasets_worsened']}  "
            f"worst {row['worst_delta']:+.4f}"
        )
    print(f"\nWrote {args.out}/edge_rule_ari.csv")


if __name__ == "__main__":
    main()
