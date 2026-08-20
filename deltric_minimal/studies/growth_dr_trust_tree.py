"""Shallow decision tree: can run-level DR-quality metrics flag a degenerate
projected graph before the growth phase touches it?

``kinit``'s ``dimred_variant_study.py`` fits 13 dimensionality-reduction
variants (pca, several umap settings, isomap15, lle15, spectral15, tsne30,
mds, ...) over 2-40 shapes/dims and records, per run, both neighbourhood-
quality metrics (``recall_at_5/10/15``, ``n_preserve``, ``n_surprise``,
``dist_corr_pearson/spearman``, ``density_corr``, ``centroid_corr``,
``manifold_corr``) and a ``degenerate_graph`` flag: 1 when the projected
Delaunay graph itself collapses (mean degree < 2, or fewer than half-n
edges) -- i.e. the projection is structurally unusable for a Delaunay-based
growth pass, independent of whether individual edges look prunable.

DelTriC's growth phase (``utils_component_growth.py``) has no signal for
this today: it builds the Delaunay graph on ``X_proj`` and starts pruning
edges regardless of whether that graph was viable to begin with. This study
asks whether the *run-level* metrics kinit already computes -- available
before growth touches a single edge, since they only need ``X`` and
``X_proj`` -- predict ``degenerate_graph`` well enough to gate the phase: a
shallow tree that fires "this projection's graph is likely degenerate, warn
or fall back" before the per-edge rule in ``edge_rule.py`` ever runs.

As requested, the tree is fit on *all* examples (no train/test split) --
this is rule extraction for a human to read, matching how
``edge_rule_extraction.py`` reports its depth-4/6 rules, not a generalisation
estimate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

import _kinit_path  # noqa: F401  (adds the sibling kinit repo to sys.path)

DEFAULT_METRICS_CSV = (
    _kinit_path.KINIT_DIR / "results" / "dimred_variants" / "variant_dr_metrics.csv"
)

#: Run-level metrics kinit computes from (X, X_proj) alone -- no per-edge
#: pruning features. ``dim``/``n`` describe the dataset; ``method`` is
#: one-hot encoded since it is categorical.
#:
#: ``centroid_corr`` and ``manifold_corr`` are deliberately excluded even
#: though kinit's CSV has them: both need ground truth the growth phase does
#: not have at gate time (``centroid_corr`` needs cluster labels ``y``,
#: ``manifold_corr`` needs the generative manifold coordinate ``t`` --
#: kinit.metrics.cluster_centroid_distance_correlation /
#: manifold_coordinate_recovery). A gate that needs the answer to decide
#: whether to trust the question is not deployable.
NUMERIC_FEATURES = (
    "dim",
    "n",
    "recall_at_5",
    "recall_at_10",
    "recall_at_15",
    "n_preserve",
    "n_surprise",
    "dist_corr_pearson",
    "dist_corr_spearman",
    "density_corr",
)

TARGET = "degenerate_graph"


def load_dataset(csv_path: Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(csv_path)
    method_dummies = pd.get_dummies(df["method"], prefix="method")
    feature_names = list(NUMERIC_FEATURES) + list(method_dummies.columns)
    X = pd.concat([df[list(NUMERIC_FEATURES)], method_dummies], axis=1)
    # density_corr carries NaNs for shapes without a meaningful density
    # axis; the missingness is itself informative (deterministic given
    # shape/target_kind), so encode "value" and "was missing" separately
    # rather than imputing a number the tree would treat as real.
    X["density_corr_missing"] = X["density_corr"].isna().astype(int)
    X["density_corr"] = X["density_corr"].fillna(0.0)
    feature_names += ["density_corr_missing"]
    return df, X[feature_names], feature_names


def fit_tree(X: pd.DataFrame, y: pd.Series, max_depth: int, seed: int) -> DecisionTreeClassifier:
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(X, y)
    return model


def summarize(model: DecisionTreeClassifier, X: pd.DataFrame, y: pd.Series) -> dict:
    pred = model.predict(X)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    return {
        "n": int(len(y)),
        "n_positive": int(y.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "accuracy": float((pred == y).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS_CSV)
    parser.add_argument("--out", type=Path, default=Path("results/growth_dr_trust"))
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20_260_819)
    args = parser.parse_args()

    df, X, feature_names = load_dataset(args.metrics_csv)
    y = df[TARGET]

    model = fit_tree(X, y, args.max_depth, args.seed)
    stats = summarize(model, X, y)

    importances = sorted(
        zip(feature_names, model.feature_importances_), key=lambda kv: -kv[1]
    )
    rules_text = export_text(model, feature_names=feature_names, decimals=4)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"rules_tree_depth{args.max_depth}.txt").write_text(rules_text + "\n")
    (args.out / f"rules_tree_depth{args.max_depth}.json").write_text(
        json.dumps(
            {
                "target": TARGET,
                "max_depth": args.max_depth,
                "seed": args.seed,
                "metrics_csv": str(args.metrics_csv),
                "feature_names": feature_names,
                "feature_importances": [
                    {"feature": name, "importance": float(v)} for name, v in importances
                ],
                "fit_on_all_examples": True,
                "eval_stats": stats,
            },
            indent=2,
        )
        + "\n"
    )

    print(f"metrics csv: {args.metrics_csv}")
    print(f"n runs: {stats['n']}  degenerate: {stats['n_positive']}")
    print(f"train==eval accuracy: {stats['accuracy']:.4f}  "
          f"precision: {stats['precision']:.4f}  recall: {stats['recall']:.4f}")
    print()
    print("feature importances (nonzero):")
    for name, v in importances:
        if v > 0:
            print(f"  {name:<28} {v:.4f}")
    print()
    print(rules_text)
    print(f"\nwrote {args.out}/rules_tree_depth{args.max_depth}.{{txt,json}}")


if __name__ == "__main__":
    main()
