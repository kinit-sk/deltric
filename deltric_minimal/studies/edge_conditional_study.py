"""Test whether local context adds signal after global edge length is known."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import _kinit_path  # noqa: F401  (adds the sibling kinit repo to sys.path)
from edge_pruning_study import (
    DEFAULT_EMBEDDINGS,
    FEATURES,
    MODEL_FEATURES,
    _edge_feature_arrays,
    _safe_auc,
    _target_for_edges,
)


METHODS = ("pca", "umap", "pca_umap")
REPRESENTATIVE_SHAPES = {
    "blobs_separated", "blobs_two", "blobs_varied_density", "blobs_hierarchical",
    "blobs_informative_subspace", "blobs_hearts", "blobs_meeples", "nested_spheres",
    "uniform_hypercube", "moons", "s_curve", "swiss_roll", "swiss_roll_hole",
    "severed_sphere", "torus", "clusters_on_swiss_roll",
}

ABLATIONS = {
    "length_only": ("orig_length", "orig_knn_scale_ratio"),
    "length_plus_projection": (
        "orig_length", "projected_length", "orig_knn_scale_ratio", "projected_knn_scale_ratio",
    ),
    "context_only": (
        "orig_incident_min_ratio", "orig_local_contrast", "projected_local_contrast",
        "edge_center_density", "endpoint_density_asymmetry", "endpoint_context_disagreement",
        "two_hop_context_disagreement", "endpoint_precision_min", "common_neighbors",
        "triangle_count", "boundary_edge",
    ),
    "length_plus_context": MODEL_FEATURES,
}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fit_grouped(samples: dict[str, list[tuple[np.ndarray, np.ndarray, str]]]) -> list[dict]:
    output = []
    for method, chunks in sorted(samples.items()):
        X = np.concatenate([c[0] for c in chunks], axis=0)
        y = np.concatenate([c[1] for c in chunks], axis=0)
        groups = np.concatenate([np.full(len(c[1]), c[2], dtype=object) for c in chunks])
        n_splits = min(5, len(np.unique(groups)))
        if n_splits < 2:
            continue
        splitter = GroupKFold(n_splits=n_splits)
        for ablation, columns in ABLATIONS.items():
            indices = [MODEL_FEATURES.index(name) for name in columns]
            scores = []
            for train, test in splitter.split(X, y, groups):
                if np.unique(y[train]).size < 2 or np.unique(y[test]).size < 2:
                    continue
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0),
                )
                model.fit(X[train][:, indices], y[train])
                scores.append(float(roc_auc_score(y[test], model.predict_proba(X[test][:, indices])[:, 1])))
            if scores:
                output.append({
                    "method": method,
                    "ablation": ablation,
                    "features": len(indices),
                    "groups": len(np.unique(groups)),
                    "folds": len(scores),
                    "mean_auc": float(np.mean(scores)),
                    "std_auc": float(np.std(scores)),
                    "rows": len(y),
                })

    # Pooled across projection methods, with shape held out as the group.
    pooled = [chunk for chunks in samples.values() for chunk in chunks]
    if pooled:
        X = np.concatenate([c[0] for c in pooled], axis=0)
        y = np.concatenate([c[1] for c in pooled], axis=0)
        groups = np.concatenate([np.full(len(c[1]), c[2], dtype=object) for c in pooled])
        splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
        for ablation, columns in ABLATIONS.items():
            indices = [MODEL_FEATURES.index(name) for name in columns]
            scores = []
            for train, test in splitter.split(X, y, groups):
                if np.unique(y[train]).size < 2 or np.unique(y[test]).size < 2:
                    continue
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0),
                )
                model.fit(X[train][:, indices], y[train])
                scores.append(float(roc_auc_score(y[test], model.predict_proba(X[test][:, indices])[:, 1])))
            if scores:
                output.append({
                    "method": "pooled",
                    "ablation": ablation,
                    "features": len(indices),
                    "groups": len(np.unique(groups)),
                    "folds": len(scores),
                    "mean_auc": float(np.mean(scores)),
                    "std_auc": float(np.std(scores)),
                    "rows": len(y),
                })
    return output


def run(data_dir: Path, embeddings_dir: Path, out_dir: Path, k: int) -> None:
    conditional_rows = []
    samples: dict[str, list[tuple[np.ndarray, np.ndarray, str]]] = defaultdict(list)
    datasets = []
    for path in sorted(data_dir.glob("*.npz")):
        stem = path.stem
        if "__n500" not in stem:
            continue
        shape = stem.split("__d", 1)[0]
        if shape in REPRESENTATIVE_SHAPES:
            datasets.append(path)

    run_count = 0
    for dataset_path in datasets:
        stem = dataset_path.stem
        dataset = np.load(dataset_path, allow_pickle=False)
        X = np.asarray(dataset["X"], dtype=np.float64)
        y = np.asarray(dataset["y"], dtype=np.int64)
        for method in METHODS:
            embedding_path = embeddings_dir / f"{stem}__{method}.npz"
            if not embedding_path.exists():
                continue
            Z = np.asarray(np.load(embedding_path, allow_pickle=False)["X"], dtype=np.float64)
            edges, features, diagnostics = _edge_feature_arrays(X, Z, k=k)
            target, valid, target_kind = _target_for_edges(X, y, edges, diagnostics)
            if target_kind != "cluster_label_bridge":
                continue
            run_count += 1
            base = features["orig_length"]
            zones = {
                "all": np.ones(len(edges), dtype=bool),
                "p25_p75_orig_length": (base >= np.quantile(base, 0.25)) & (base <= np.quantile(base, 0.75)),
                "p50_p90_orig_length": (base >= np.quantile(base, 0.50)) & (base <= np.quantile(base, 0.90)),
                "below_p90_orig_length": base <= np.quantile(base, 0.90),
            }
            for zone_name, zone in zones.items():
                usable = valid & zone
                yt = target[usable].astype(bool)
                if len(yt) < 20 or np.unique(yt).size < 2:
                    continue
                for feature in FEATURES:
                    raw, oriented, direction = _safe_auc(yt, features[feature][usable])
                    conditional_rows.append({
                        "dataset": stem,
                        "shape": stem.split("__d", 1)[0],
                        "method": method,
                        "target_kind": target_kind,
                        "zone": zone_name,
                        "feature": feature,
                        "n_edges": int(len(yt)),
                        "positive_rate": float(np.mean(yt)),
                        "raw_auc": raw,
                        "oriented_auc": oriented,
                        "direction": direction,
                    })

            valid_indices = np.flatnonzero(valid)
            positive = valid_indices[target[valid_indices].astype(bool)]
            negative = valid_indices[~target[valid_indices].astype(bool)]
            take = min(400, len(positive), len(negative))
            if take:
                rng = np.random.default_rng(30_000 + run_count)
                chosen = np.concatenate((
                    rng.choice(positive, size=take, replace=False),
                    rng.choice(negative, size=take, replace=False),
                ))
                samples[method].append((
                    np.column_stack([features[name][chosen] for name in MODEL_FEATURES]),
                    target[chosen].astype(np.int8),
                    stem.split("__d", 1)[0],
                ))
            print(run_count, stem, method)

    grouped = defaultdict(list)
    for row in conditional_rows:
        grouped[(row["method"], row["zone"], row["feature"])].append(row)
    summary_rows = []
    for (method, zone, feature), group in sorted(grouped.items()):
        values = np.asarray([float(r["oriented_auc"]) for r in group])
        summary_rows.append({
            "method": method,
            "zone": zone,
            "feature": feature,
            "runs": len(group),
            "mean_oriented_auc": float(np.mean(values)),
            "median_oriented_auc": float(np.median(values)),
            "fraction_auc_ge_060": float(np.mean(values >= 0.60)),
            "fraction_auc_ge_070": float(np.mean(values >= 0.70)),
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "conditional_feature_scores.csv", conditional_rows)
    write_csv(out_dir / "conditional_feature_summary.csv", summary_rows)
    write_csv(out_dir / "conditional_model_ablation.csv", _fit_grouped(samples))
    (out_dir / "conditional_manifest.json").write_text(
        f'{{"runs": {run_count}, "datasets": {len(datasets)}, "ambient_knn_k": {k}}}\n'
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=_kinit_path.KINIT_DIR / "data")
    parser.add_argument("--embeddings-dir", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--out", type=Path, default=Path("results/edge_conditional"))
    parser.add_argument("--ambient-knn-k", type=int, default=15)
    args = parser.parse_args()
    run(args.data_dir, args.embeddings_dir, args.out, args.ambient_knn_k)


if __name__ == "__main__":
    main()
