"""Extension of edge_pruning_study.py: does a point's ambient-preservation
trustworthiness (point_preserve / point_recall, already computed by the main
benchmark) help predict which Delaunay *edges* are bridges (spuriously join
two different clusters), on top of the existing geometric/density features?

Background: a companion study found that points near a true cluster boundary
tend to have lower point_preserve than interior points (weak-to-moderate,
consistently one-sided). That is a per-POINT result. It does not by itself
say whether a specific EDGE between two low-preserve points is actually a
bridge that should be pruned -- an edge between two boundary points of the
SAME real cluster would also have low-preserve endpoints without being a
bridge at all. This script tests the per-EDGE question directly using the
same bridge target and cross-shape evaluation edge_pruning_study.py already
built, so the answer is directly comparable to its existing baseline.

Reuses edge_pruning_study.py's edge/feature/target machinery by importing it
as a module -- does not modify that file, which may still be in use.  Writes
only to results/edge_preserve_extension/ (new directory).

Three comparisons are reported for the cross-shape logistic model:
  - baseline: edge_pruning_study.py's existing MODEL_FEATURES (geometric +
    density signals only) -- reproduced here (same seeds, same sampling) so
    the comparison is apples-to-apples, not read off the old CSV.
  - preserve_only: just the new endpoint-preserve/recall features, nothing
    geometric -- answers "is preserve alone a usable bridge signal".
  - baseline_plus_preserve: MODEL_FEATURES + the new features together --
    answers "does preserve add anything once geometry already knows".

Also reports standalone AUC for each new feature against the bridge target,
the same way edge_pruning_study.py reports it for its own features, so the
new features slot into the same feature_summary table for comparison.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import _kinit_path  # noqa: F401  (adds the sibling kinit repo to sys.path)
import edge_pruning_study as eps

DATA_DIR = _kinit_path.KINIT_DIR / "data"
EMBEDDINGS_DIR = eps.DEFAULT_EMBEDDINGS
OUT_DIR = Path("results/edge_preserve_extension")
METHODS = eps.METHODS
MAX_N = 5000
K = 15

NEW_FEATURES = (
    "endpoint_preserve_min",
    "endpoint_preserve_mean",
    "endpoint_preserve_max",
    "endpoint_preserve_asymmetry",
    "endpoint_recall_min",
    "endpoint_recall_mean",
)


def _add_preserve_features(features: dict, edges: np.ndarray, preserve: np.ndarray, recall: np.ndarray) -> None:
    u, v = edges.T
    pu, pv = preserve[u], preserve[v]
    ru, rv = recall[u], recall[v]
    features["endpoint_preserve_min"] = np.minimum(pu, pv)
    features["endpoint_preserve_mean"] = (pu + pv) / 2.0
    features["endpoint_preserve_max"] = np.maximum(pu, pv)
    features["endpoint_preserve_asymmetry"] = np.abs(pu - pv)
    features["endpoint_recall_min"] = np.minimum(ru, rv)
    features["endpoint_recall_mean"] = (ru + rv) / 2.0


def _fit_grouped_model_named(model_rows, feature_count: int, label: str) -> list[dict]:
    """Same cross-shape GroupKFold logistic evaluation as
    edge_pruning_study._fit_grouped_model, just tagged with which feature
    set was used so multiple feature sets can be compared side by side."""
    if not model_rows:
        return []
    X = np.concatenate([row[0] for row in model_rows], axis=0)
    y = np.concatenate([row[1] for row in model_rows], axis=0)
    groups = np.concatenate([row[2] for row in model_rows], axis=0)
    methods = np.concatenate([np.full(len(row[1]), row[3], dtype=object) for row in model_rows])
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    unique_groups = np.unique(groups)
    n_splits = min(5, len(unique_groups))
    if n_splits < 2:
        return []

    rows = []
    splitter = GroupKFold(n_splits=n_splits)

    def eval_mask(mask, method_label):
        scores = []
        for train_idx, test_idx in splitter.split(X[mask], y[mask], groups[mask]):
            y_train, y_test = y[mask][train_idx], y[mask][test_idx]
            if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
                continue
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0),
            )
            model.fit(X[mask][train_idx], y_train)
            scores.append(float(roc_auc_score(y_test, model.predict_proba(X[mask][test_idx])[:, 1])))
        if scores:
            rows.append({
                "feature_set": label, "method": method_label,
                "n_features": feature_count, "groups": int(len(unique_groups)),
                "rows": int(np.count_nonzero(mask)), "folds": len(scores),
                "mean_auc": float(np.mean(scores)), "std_auc": float(np.std(scores)),
            })

    for method in sorted(set(methods.tolist())):
        eval_mask(methods == method, method)
    eval_mask(np.ones(len(y), dtype=bool), "pooled")
    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_paths = sorted(DATA_DIR.glob("*.npz"))
    candidates = []
    for dataset_path in dataset_paths:
        shape, dim, n = eps._parse_dataset_path(dataset_path)
        if n > MAX_N:
            continue
        for method in METHODS:
            embedding_path = eps._embedding_path(EMBEDDINGS_DIR, shape, dim, n, method)
            if embedding_path.exists():
                candidates.append((dataset_path, embedding_path, method))

    print(f"{len(candidates)} candidate (dataset, method) runs")

    feature_rows = []
    baseline_rows, preserve_only_rows, combined_rows = [], [], []
    failures = []

    for index, (dataset_path, embedding_path, method) in enumerate(candidates):
        try:
            started = time.perf_counter()
            shape, dim, n = eps._parse_dataset_path(dataset_path)
            dataset = np.load(dataset_path, allow_pickle=False)
            embedding = np.load(embedding_path, allow_pickle=False)
            X = np.asarray(dataset["X"], dtype=np.float64)
            y = np.asarray(dataset["y"], dtype=np.int64)
            Z = np.asarray(embedding["X"], dtype=np.float64)
            preserve = np.asarray(embedding["point_preserve"], dtype=np.float64)
            recall = np.asarray(embedding["point_recall"], dtype=np.float64)
            if len(X) != len(Z):
                raise ValueError("point count mismatch")
            if not np.array_equal(y, np.asarray(embedding["y"], dtype=np.int64)):
                raise ValueError("label order mismatch between ambient and embedding files")

            edges, features, diagnostics = eps._edge_feature_arrays(X, Z, k=K)
            _add_preserve_features(features, edges, preserve, recall)
            target, valid, target_kind = eps._target_for_edges(X, y, edges, diagnostics)
            finite_valid = valid & np.isfinite(target)
            y_valid = target[finite_valid].astype(bool)

            for feature_name in NEW_FEATURES:
                values = features[feature_name][finite_valid]
                raw_auc, oriented_auc, direction = eps._safe_auc(y_valid, values)
                precision, recall_at_budget = eps._top_budget_metrics(y_valid, values, direction)
                feature_rows.append({
                    "dataset": dataset_path.stem, "shape": shape, "dim": dim, "n": n, "method": method,
                    "target_kind": target_kind, "feature": feature_name,
                    "n_edges": int(len(edges)), "n_valid": int(np.count_nonzero(finite_valid)),
                    "positive_rate": float(np.mean(y_valid)) if len(y_valid) else float("nan"),
                    "raw_auc": raw_auc, "oriented_auc": oriented_auc, "direction": direction,
                    "top10_precision": precision, "top10_recall": recall_at_budget,
                })

            if target_kind == "cluster_label_bridge" and len(y_valid) and np.unique(y_valid).size == 2:
                valid_indices = np.flatnonzero(finite_valid)
                pos = valid_indices[target[valid_indices].astype(bool)]
                neg = valid_indices[~target[valid_indices].astype(bool)]
                take = min(400, len(pos), len(neg))
                if take:
                    rng = np.random.default_rng(17_003 + index)  # same seed formula as edge_pruning_study
                    pos_s = rng.choice(pos, size=take, replace=False)
                    neg_s = rng.choice(neg, size=take, replace=False)
                    sample = np.concatenate((pos_s, neg_s))
                    groups = np.full(len(sample), shape, dtype=object)
                    labels = target[sample].astype(np.int8)

                    baseline_rows.append((
                        np.column_stack([features[name][sample] for name in eps.MODEL_FEATURES]),
                        labels, groups, method, target_kind,
                    ))
                    preserve_only_rows.append((
                        np.column_stack([features[name][sample] for name in NEW_FEATURES]),
                        labels, groups, method, target_kind,
                    ))
                    combined_rows.append((
                        np.column_stack([features[name][sample] for name in (*eps.MODEL_FEATURES, *NEW_FEATURES)]),
                        labels, groups, method, target_kind,
                    ))

            print(f"{index + 1:>4}/{len(candidates)} {dataset_path.stem} {method:<9} "
                  f"target={target_kind:<28} edges={len(edges):<6} ({time.perf_counter()-started:.2f}s)")
        except Exception as exc:
            failures.append({"dataset": str(dataset_path), "method": method, "error": f"{type(exc).__name__}: {exc}"})
            print(f"FAILED {dataset_path.name} {method}: {type(exc).__name__}: {exc}")

    eps._write_csv(OUT_DIR / "preserve_feature_scores.csv", feature_rows)
    eps._write_csv(OUT_DIR / "preserve_feature_summary.csv", eps._aggregate_feature_rows(feature_rows))
    eps._write_csv(OUT_DIR / "edge_preserve_failures.csv", failures)

    model_summary = []
    model_summary += _fit_grouped_model_named(baseline_rows, len(eps.MODEL_FEATURES), "baseline_geometric")
    model_summary += _fit_grouped_model_named(preserve_only_rows, len(NEW_FEATURES), "preserve_only")
    model_summary += _fit_grouped_model_named(combined_rows, len(eps.MODEL_FEATURES) + len(NEW_FEATURES), "baseline_plus_preserve")
    eps._write_csv(OUT_DIR / "model_comparison.csv", model_summary)

    print("\n=== cross-shape logistic AUC by feature set ===")
    for row in model_summary:
        print(f"  {row['feature_set']:22s} {row['method']:9s} n_features={row['n_features']:2d} "
              f"mean_auc={row['mean_auc']:.4f} +/- {row['std_auc']:.4f}  (rows={row['rows']}, groups={row['groups']})")


if __name__ == "__main__":
    main()
