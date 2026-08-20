"""Extract interpretable keep/prune rules from the per-edge feature dump.

``edge_pruning_study.py --dump-edges`` writes the full per-edge feature matrix
for every run.  This module fits two *interpretable* model families on that
dump and compares them head to head under the same shape-grouped holdout used
by the study's logistic baseline:

* a depth-limited decision tree, exported as literal ``if/elif/else``
  thresholds;
* a linear score with an explicit threshold and abstain band.

Both are scored against three references: ``orig_length`` alone,
``orig_knn_scale_ratio`` alone, and the full 19-feature logistic model.  The
point of the comparison is to price interpretability, not to assume it is free.

Raw feature values are not comparable across runs: ambient dimension and the
choice of projection both move the distributions.  Every model is therefore fit
twice, once on the dump's raw features and once after a within-run robust
transform (log for ratio features, then median/IQR normalisation).  The
transform spec is exported alongside the rules, because a threshold without its
normalisation is meaningless.

AUC is reported but is not the deciding metric.  A wrongly pruned edge can
split a cluster, while a wrongly kept edge is usually repaired downstream, so
the tables also carry bridge precision at fixed recall and the intra-cluster
edge retention that results.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

import _kinit_path  # noqa: F401  (adds the sibling kinit repo to sys.path)
from edge_pruning_study import MODEL_FEATURES


TARGET_KIND = "cluster_label_bridge"

# Ratio-valued features: log first, then normalise within the run.  These are
# the ones whose spread changes with ambient dimension.
LOG_RATIO_FEATURES = (
    "orig_length",
    "projected_length",
    "orig_incident_mean_ratio",
    "orig_incident_median_ratio",
    "orig_incident_min_ratio",
    "projected_incident_mean_ratio",
    "orig_local_contrast",
    "projected_local_contrast",
    "orig_knn_scale_ratio",
    "projected_knn_scale_ratio",
)

# Already on a log scale in the study; only the within-run normalisation
# applies.
ALREADY_LOG_FEATURES = ("ambient_projected_density_log_ratio",)

# Counts: log1p, then normalise.
LOG1P_FEATURES = ("common_neighbors",)

# Bounded or categorical.  Normalising these would destroy the very property
# that makes them readable in a rule ("context disagreement above 0.6").
RAW_FEATURES = (
    "endpoint_density_asymmetry",
    "edge_center_density",
    "endpoint_context_disagreement",
    "two_hop_context_disagreement",
    "endpoint_precision_min",
    "triangle_count",
    "boundary_edge",
)

# Whether DelTriC's component-growth backend already computes the quantity a
# feature needs.  Consulted only for features the exported rules actually use,
# so the checklist stays short and actionable.
BACKEND_AVAILABILITY = {
    "orig_length": ("present", "orig_edge_sizes, utils_component_growth.py:785"),
    "projected_length": ("present", "projected_edge_sizes, utils_component_growth.py:789"),
    "two_hop_context_disagreement": (
        "present",
        "_two_hop_edge_selectivity, utils_component_growth.py:111",
    ),
    "endpoint_context_disagreement": (
        "present",
        "local_knn_selectivity, utils_component_growth.py:70-108",
    ),
    "endpoint_precision_min": (
        "present",
        "derivable from local_knn_selectivity, utils_component_growth.py:70-108",
    ),
    "orig_knn_scale_ratio": ("absent", "needs ambient kNN radius per point (cKDTree query)"),
    "projected_knn_scale_ratio": ("absent", "needs projected kNN radius per point"),
    "orig_incident_mean_ratio": ("absent", "utils_pruning.py:393 _neighbor_length_ratio is close"),
    "orig_incident_median_ratio": ("absent", "utils_pruning.py:393 _neighbor_length_ratio is close"),
    "orig_incident_min_ratio": ("absent", "incident-edge scan over point_to_edges"),
    "projected_incident_mean_ratio": ("absent", "incident-edge scan in projected space"),
    "orig_local_contrast": ("absent", "needs edge-centre kNN median length"),
    "projected_local_contrast": ("absent", "needs edge-centre kNN median length"),
    "edge_center_density": ("absent", "utils_pruning.py:361 _edge_center_density already exists"),
    "endpoint_density_asymmetry": ("absent", "needs ambient kNN radius per point"),
    "ambient_projected_density_log_ratio": ("absent", "needs both kNN radii"),
    "common_neighbors": ("absent", "adjacency intersection per edge"),
    "triangle_count": ("absent", "available from edge_to_tris in _extract_edge_data"),
    "boundary_edge": ("absent", "available from edge_to_tris in _extract_edge_data"),
}

RECALL_TARGETS = (0.5, 0.8)

# L1 strengths, expressed as C*n rather than C.  liblinear minimises
# ``C * sum(loss) + ||w||_1``, so the penalty's effective strength scales with
# the row count: a C that yields two non-zero coefficients on 2,000 rows is
# numerically irrelevant on 1.36M.  Fixing C*n keeps the sparsity comparable
# across subsets of very different sizes, which is the only way the per-method
# and pooled formulas can be read side by side.  The small end is there to
# force a genuinely short formula: a 19-term weighted sum is not more auditable
# than the logistic model it was meant to replace.
L1_CN_GRID = (20.0, 200.0, 2000.0, 20000.0)

# liblinear's default tol=1e-4 costs several times the runtime of 1e-3 on this
# corpus and moves the fourth decimal of the AUC.  The comparison is between
# model families, so that precision buys nothing.
L1_TOL = 1e-3


def assert_l1_is_sparse(n_rows: int) -> None:
    """Fail loudly if the L1 models are not actually sparse.

    Two distinct failures are caught here.  First, scikit-learn 1.8 deprecated
    ``penalty="l1"`` in favour of ``l1_ratio`` and silently fits an L2 model if
    the old argument is passed.  Second, ``C`` multiplies the summed loss, so a
    fixed ``C`` that looks sparse on a toy sample is numerically irrelevant on a
    million rows.  Both failures are invisible in the metrics -- the AUCs look
    fine and the "sparse" formula quietly keeps all 19 terms -- so the check
    runs at ``n_rows``, the size actually being fitted.
    """
    rng = np.random.default_rng(0)
    n = max(int(n_rows), 1000)
    X = rng.normal(size=(n, len(MODEL_FEATURES)))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(size=n) > 0).astype(int)
    model = LogisticRegression(
        l1_ratio=1.0,
        C=min(L1_CN_GRID) / n,
        solver="liblinear",
        tol=L1_TOL,
        max_iter=2000,
        random_state=0,
    ).fit(X, y)
    nonzero = int(np.count_nonzero(model.coef_[0]))
    if nonzero > len(MODEL_FEATURES) // 2:
        raise SystemExit(
            f"L1 sanity check failed: {nonzero}/{len(MODEL_FEATURES)} non-zero coefficients at "
            f"C*n={min(L1_CN_GRID)} on {n} rows. Either the L1 penalty is not being applied "
            "(check the scikit-learn API) or the C*n grid is too weak for this sample size."
        )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_dump(
    dump_dir: Path, max_edges_per_run: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y, shape_groups, run_ids, methods) over label-bridge runs.

    ``run_ids`` matter as much as the feature values: the within-run transform
    has to be applied per run, exactly as it would be at inference time on a
    single dataset.
    """
    files = sorted(p for p in dump_dir.glob("*.npz"))
    if not files:
        raise SystemExit(f"no .npz files found in {dump_dir}")

    feature_blocks: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    shapes: list[np.ndarray] = []
    runs: list[np.ndarray] = []
    methods: list[np.ndarray] = []
    rng = np.random.default_rng(seed)
    skipped = 0

    for run_index, path in enumerate(files):
        with np.load(path) as payload:
            if str(payload["target_kind"]) != TARGET_KIND:
                skipped += 1
                continue
            names = [str(name) for name in payload["feature_names"]]
            columns = [names.index(name) for name in MODEL_FEATURES]
            matrix = payload["features"][:, columns].astype(np.float64)
            target = payload["target"].astype(np.int8)
            valid = payload["valid"].astype(bool)
            shape = str(payload["shape"])
            method = str(payload["method"])

        keep = np.flatnonzero(valid & np.all(np.isfinite(matrix), axis=1))
        if len(keep) == 0 or np.unique(target[keep]).size < 2:
            skipped += 1
            continue
        if 0 < max_edges_per_run < len(keep):
            # Uniform, not balanced: the natural ~9% positive rate is what a
            # deployed threshold will actually face.
            keep = np.sort(rng.choice(keep, size=max_edges_per_run, replace=False))

        feature_blocks.append(matrix[keep])
        targets.append(target[keep])
        shapes.append(np.full(len(keep), shape, dtype=object))
        runs.append(np.full(len(keep), run_index, dtype=np.int32))
        methods.append(np.full(len(keep), method, dtype=object))

    if not feature_blocks:
        raise SystemExit(f"no usable {TARGET_KIND} runs in {dump_dir}")

    print(
        f"loaded {len(feature_blocks)} runs "
        f"({skipped} skipped: wrong target, degenerate, or single-class)"
    )
    return (
        np.concatenate(feature_blocks, axis=0),
        np.concatenate(targets).astype(np.int8),
        np.concatenate(shapes),
        np.concatenate(runs),
        np.concatenate(methods),
    )


def transform_spec() -> dict:
    spec = {}
    for name in MODEL_FEATURES:
        if name in LOG_RATIO_FEATURES:
            spec[name] = "log_then_within_run_median_iqr"
        elif name in ALREADY_LOG_FEATURES:
            spec[name] = "within_run_median_iqr"
        elif name in LOG1P_FEATURES:
            spec[name] = "log1p_then_within_run_median_iqr"
        else:
            spec[name] = "raw"
    return spec


def normalise_within_run(X: np.ndarray, runs: np.ndarray) -> np.ndarray:
    """Apply the transform spec, normalising per run.

    A run is a single (dataset, projection) pair, which is exactly the scope a
    deployed rule would see: one point cloud, one embedding, one Delaunay
    graph.  Normalising over the pooled corpus instead would leak information
    the rule cannot have at inference time.
    """
    out = X.copy()
    spec = transform_spec()
    needs_run_norm = np.zeros(len(MODEL_FEATURES), dtype=bool)

    for column, name in enumerate(MODEL_FEATURES):
        kind = spec[name]
        if kind.startswith("log_then"):
            out[:, column] = np.log(np.clip(out[:, column], 1e-6, 1e6))
        elif kind.startswith("log1p_then"):
            out[:, column] = np.log1p(np.clip(out[:, column], 0.0, 1e6))
        needs_run_norm[column] = kind != "raw"

    columns = np.flatnonzero(needs_run_norm)
    if len(columns):
        for run in np.unique(runs):
            mask = runs == run
            block = out[np.ix_(mask, columns)]
            median = np.median(block, axis=0)
            iqr = np.percentile(block, 75, axis=0) - np.percentile(block, 25, axis=0)
            out[np.ix_(mask, columns)] = (block - median) / np.maximum(iqr, 1e-6)

    return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)


def _threshold_for_recall(y: np.ndarray, scores: np.ndarray, recall: float) -> float:
    """Lowest score threshold that still reaches ``recall`` on bridges."""
    positive = scores[y == 1]
    if len(positive) == 0:
        return float("inf")
    return float(np.quantile(positive, 1.0 - recall))


def _operating_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predicted_prune = scores >= threshold
    pruned = int(np.count_nonzero(predicted_prune))
    true_bridges = int(np.count_nonzero(y == 1))
    hits = int(np.count_nonzero(predicted_prune & (y == 1)))
    intra_total = int(np.count_nonzero(y == 0))
    intra_kept = int(np.count_nonzero(~predicted_prune & (y == 0)))
    return {
        "bridge_precision": hits / pruned if pruned else float("nan"),
        "bridge_recall": hits / true_bridges if true_bridges else float("nan"),
        # The safety metric: how much of the true intra-cluster graph survives.
        "intra_retention": intra_kept / intra_total if intra_total else float("nan"),
    }


def _score_fold(
    y_train: np.ndarray,
    score_train: np.ndarray,
    y_test: np.ndarray,
    score_test: np.ndarray,
    abstain_fraction: float,
) -> dict:
    row = {
        "auc": float(roc_auc_score(y_test, score_test)),
        "average_precision": float(average_precision_score(y_test, score_test)),
    }
    for recall in RECALL_TARGETS:
        # The threshold is chosen on training scores only, then applied to the
        # held-out fold.  Picking it on the test fold would flatter every model.
        threshold = _threshold_for_recall(y_train, score_train, recall)
        metrics = _operating_metrics(y_test, score_test, threshold)
        tag = f"r{int(recall * 100)}"
        row[f"{tag}_bridge_precision"] = metrics["bridge_precision"]
        row[f"{tag}_bridge_recall"] = metrics["bridge_recall"]
        row[f"{tag}_intra_retention"] = metrics["intra_retention"]

    if abstain_fraction > 0:
        low, high = np.quantile(
            score_train, [0.5 - abstain_fraction / 2.0, 0.5 + abstain_fraction / 2.0]
        )
        confident = (score_test <= low) | (score_test >= high)
        row["abstain_rate"] = float(1.0 - np.mean(confident))
        if np.count_nonzero(confident) and np.unique(y_test[confident]).size == 2:
            row["confident_auc"] = float(
                roc_auc_score(y_test[confident], score_test[confident])
            )
            threshold = _threshold_for_recall(y_train, score_train, 0.8)
            row["confident_r80_bridge_precision"] = _operating_metrics(
                y_test[confident], score_test[confident], threshold
            )["bridge_precision"]
        else:
            row["confident_auc"] = float("nan")
            row["confident_r80_bridge_precision"] = float("nan")
    return row


def build_models(n_rows: int) -> dict:
    """Model zoo: interpretable candidates plus the references that price them.

    Specs are plain dicts of constructor keyword arguments rather than
    closures, so they survive pickling to joblib workers.
    """
    leaf = max(50, int(0.005 * n_rows))
    models: dict[str, dict] = {
        "baseline_orig_length": {
            "kind": "single_feature",
            "feature": "orig_length",
        },
        "baseline_orig_knn_scale_ratio": {
            "kind": "single_feature",
            "feature": "orig_knn_scale_ratio",
        },
        "logistic_19": {
            "kind": "linear",
            "params": {"max_iter": 1000, "class_weight": "balanced", "random_state": 0},
        },
    }
    for cn in L1_CN_GRID:
        models[f"logistic_l1_cn{cn:g}"] = {
            "kind": "linear",
            "params": {
                "l1_ratio": 1.0,
                "C": cn / max(n_rows, 1),
                "solver": "liblinear",
                "tol": L1_TOL,
                "max_iter": 2000,
                "class_weight": "balanced",
                "random_state": 0,
            },
        }
    for depth in (2, 3, 4, 6):
        models[f"tree_depth{depth}"] = {
            "kind": "tree",
            "params": {
                "max_depth": depth,
                "min_samples_leaf": leaf,
                "class_weight": "balanced",
                "random_state": 0,
            },
        }
    return models


def _make_estimator(spec: dict):
    if spec["kind"] == "linear":
        return LogisticRegression(**spec["params"])
    return DecisionTreeClassifier(**spec["params"])


def _fit_scores(spec: dict, X_train, y_train, X_test):
    """Fit one model and return (train_scores, test_scores, fitted_object)."""
    if spec["kind"] == "single_feature":
        column = MODEL_FEATURES.index(spec["feature"])
        return X_train[:, column], X_test[:, column], None

    if spec["kind"] == "linear":
        scaler = StandardScaler().fit(X_train)
        model = _make_estimator(spec).fit(scaler.transform(X_train), y_train)
        return (
            model.decision_function(scaler.transform(X_train)),
            model.decision_function(scaler.transform(X_test)),
            (model, scaler),
        )

    # Trees are scale-invariant, so they are fit on the untouched feature
    # values.  That is what makes their thresholds directly readable.
    model = _make_estimator(spec).fit(X_train, y_train)
    return (
        model.predict_proba(X_train)[:, 1],
        model.predict_proba(X_test)[:, 1],
        model,
    )


def _fold_job(
    spec: dict,
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    abstain_fraction: float,
) -> dict | None:
    """One (model, fold) unit of work.  Must stay picklable for joblib."""
    if np.unique(y[train_idx]).size < 2 or np.unique(y[test_idx]).size < 2:
        return None
    train_scores, test_scores, _ = _fit_scores(spec, X[train_idx], y[train_idx], X[test_idx])
    return _score_fold(y[train_idx], train_scores, y[test_idx], test_scores, abstain_fraction)


def evaluate(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    methods: np.ndarray,
    variant: str,
    abstain_fraction: float,
    jobs: int = 1,
) -> list[dict]:
    models = build_models(len(y))
    unique_groups = np.unique(groups)
    n_splits = min(5, len(unique_groups))
    if n_splits < 2:
        print(f"  {variant}: only {len(unique_groups)} shape groups, skipping")
        return []

    splitter = GroupKFold(n_splits=n_splits)
    rows: list[dict] = []
    subsets = [(method, methods == method) for method in sorted(set(methods.tolist()))]
    subsets.append(("pooled", np.ones(len(y), dtype=bool)))

    # Flatten (subset, model, fold) into one task list.  The units are wildly
    # uneven -- a single-feature baseline is instant, an L1 fit on the pooled
    # subset is minutes -- so one flat list keeps every core busy instead of
    # stalling at a per-model barrier.
    tasks = []
    for method, subset in subsets:
        Xs, ys, gs = X[subset], y[subset], groups[subset]
        # Rebuild per subset so the C*n scaling and the min_samples_leaf floor
        # track the rows this subset actually has; pooled is ~3x each method.
        for name, spec in build_models(len(ys)).items():
            for train_idx, test_idx in splitter.split(Xs, ys, gs):
                tasks.append((method, name, spec, Xs, ys, train_idx, test_idx))

    print(f"  {variant}: {len(tasks)} (subset, model, fold) tasks on {jobs} workers", flush=True)
    results = Parallel(n_jobs=jobs, prefer="processes", verbose=0)(
        delayed(_fold_job)(spec, Xs, ys, train_idx, test_idx, abstain_fraction)
        for _, _, spec, Xs, ys, train_idx, test_idx in tasks
    )

    grouped: dict[tuple[str, str], list[dict]] = {}
    for (method, name, *_), result in zip(tasks, results):
        if result is not None:
            grouped.setdefault((method, name), []).append(result)

    for method, subset in subsets:
        for name in models:
            folds = grouped.get((method, name), [])
            if not folds:
                continue
            row = {
                "variant": variant,
                "method": method,
                "model": name,
                "groups": int(len(unique_groups)),
                "folds": len(folds),
                "rows": int(np.count_nonzero(subset)),
                "positive_rate": float(np.mean(y[subset])),
            }
            for key in folds[0]:
                values = np.asarray([fold[key] for fold in folds], dtype=float)
                row[f"mean_{key}"] = float(np.nanmean(values))
                if key == "auc":
                    row["std_auc"] = float(np.nanstd(values))
            rows.append(row)
            print(
                f"  {variant:<10} {method:<9} {name:<28} "
                f"auc={row['mean_auc']:.4f} "
                f"r80_precision={row.get('mean_r80_bridge_precision', float('nan')):.3f} "
                f"r80_intra_retention={row.get('mean_r80_intra_retention', float('nan')):.3f}",
                flush=True,
            )
    return rows


def _tree_to_json(
    model: DecisionTreeClassifier, X: np.ndarray, y: np.ndarray, leaf_prune_rate: float
) -> dict:
    """Serialise the tree, carrying both weighted and observed leaf statistics.

    ``tree_.value`` reflects ``class_weight="balanced"``, which is right for
    fitting -- bridges are only ~9% of edges, so unweighted splits would ignore
    them -- but catastrophic as a decision rule.  Balanced weighting flips a
    leaf to PRUNE at roughly an 8-9% observed bridge rate, so a leaf that is
    89% good intra-cluster edges gets pruned wholesale.  Deleting those edges
    can disconnect a cluster, which is the one failure DelTriC cannot repair
    downstream.

    Leaves are therefore labelled by their *observed* bridge rate against an
    explicit ``leaf_prune_rate``, with the weighted purity retained alongside
    for reference.  This changes only the labels, never the split structure or
    the probability ranking the reported AUC is computed from.
    """
    tree = model.tree_
    leaf_of_row = model.apply(X)

    def node(index: int) -> dict:
        if tree.children_left[index] == -1:
            weighted = tree.value[index][0]
            total = float(weighted.sum())
            weighted_purity = float(weighted[1] / total) if total else float("nan")
            rows = y[leaf_of_row == index]
            observed = float(np.mean(rows)) if len(rows) else float("nan")
            return {
                "leaf": True,
                "decision": "PRUNE" if observed >= leaf_prune_rate else "KEEP",
                "weighted_bridge_purity": weighted_purity,
                "observed_bridge_rate": observed,
                "support": int(len(rows)),
            }
        return {
            "leaf": False,
            "feature": MODEL_FEATURES[tree.feature[index]],
            "threshold": float(tree.threshold[index]),
            "support": int(tree.n_node_samples[index]),
            "if_below_or_equal": node(int(tree.children_left[index])),
            "if_above": node(int(tree.children_right[index])),
        }

    return node(0)


def _same_decision(node: dict) -> str | None:
    """Return the shared decision of a subtree, or None if it is not uniform."""
    if node["leaf"]:
        return node["decision"]
    left = _same_decision(node["if_below_or_equal"])
    right = _same_decision(node["if_above"])
    return left if left is not None and left == right else None


def _tree_to_python(node: dict, indent: int = 1) -> list[str]:
    pad = "    " * indent
    if node["leaf"]:
        return [
            f"{pad}return \"{node['decision']}\"  "
            f"# observed bridge rate {node['observed_bridge_rate']:.3f}, "
            f"weighted purity {node['weighted_bridge_purity']:.3f}, "
            f"support {node['support']}"
        ]
    # A split whose whole subtree agrees is noise in the printed rule: it costs
    # the reader a branch and changes no decision.
    uniform = _same_decision(node)
    if uniform is not None:
        return [f"{pad}return \"{uniform}\"  # subtree collapsed, support {node['support']}"]
    lines = [f"{pad}if {node['feature']} <= {node['threshold']:.6f}:"]
    lines += _tree_to_python(node["if_below_or_equal"], indent + 1)
    lines.append(f"{pad}else:")
    lines += _tree_to_python(node["if_above"], indent + 1)
    return lines


def _tree_features(node: dict, seen: set[str]) -> set[str]:
    """Features on splits that actually change a decision, matching the export."""
    if node["leaf"] or _same_decision(node) is not None:
        return seen
    seen.add(node["feature"])
    _tree_features(node["if_below_or_equal"], seen)
    _tree_features(node["if_above"], seen)
    return seen


def export_rules(
    X: np.ndarray,
    y: np.ndarray,
    variant: str,
    out_dir: Path,
    abstain_fraction: float,
    leaf_prune_rate: float,
) -> dict[str, set[str]]:
    """Refit on all rows and write the human-readable artifacts.

    Refitting on everything is deliberate: the grouped CV above is what
    estimates generalisation, so the shipped rule should use all the evidence
    rather than an arbitrary fold's training split.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    used: dict[str, set[str]] = {}
    leaf = max(50, int(0.005 * len(y)))
    spec = transform_spec()

    for depth in (2, 3, 4, 6):
        model = DecisionTreeClassifier(
            max_depth=depth, min_samples_leaf=leaf, class_weight="balanced", random_state=0
        ).fit(X, y)
        as_json = _tree_to_json(model, X, y, leaf_prune_rate)
        used[f"{variant}/tree_depth{depth}"] = _tree_features(as_json, set())

        (out_dir / f"rules_tree_depth{depth}_{variant}.txt").write_text(
            export_text(model, feature_names=list(MODEL_FEATURES), decimals=6) + "\n"
        )
        (out_dir / f"rules_tree_depth{depth}_{variant}.json").write_text(
            json.dumps(
                {
                    "variant": variant,
                    "max_depth": depth,
                    "leaf_prune_rate": leaf_prune_rate,
                    "transform": spec,
                    "tree": as_json,
                },
                indent=2,
            )
            + "\n"
        )
        snippet = [
            f"# Depth-{depth} DelTriC edge rule, feature variant: {variant}",
            f"# Feature transform: see transform spec in rules_tree_depth{depth}_{variant}.json",
            "def edge_decision(**features):",
        ]
        snippet += _tree_to_python(as_json)
        (out_dir / f"rules_tree_depth{depth}_{variant}.py").write_text("\n".join(snippet) + "\n")

    linear_models = {
        "logistic_19": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0),
    }
    for cn in L1_CN_GRID:
        linear_models[f"logistic_l1_cn{cn:g}"] = LogisticRegression(
            l1_ratio=1.0,
            C=cn / max(len(y), 1),
            solver="liblinear",
            tol=L1_TOL,
            max_iter=2000,
            class_weight="balanced",
            random_state=0,
        )

    linear_export = {"variant": variant, "transform": spec, "models": {}}
    for name, estimator in linear_models.items():
        scaler = StandardScaler().fit(X)
        model = estimator.fit(scaler.transform(X), y)
        # Fold the scaler into the coefficients so the exported formula applies
        # directly to transformed feature values, with no fitted object needed.
        weights = model.coef_[0] / scaler.scale_
        intercept = float(model.intercept_[0] - np.dot(model.coef_[0], scaler.mean_ / scaler.scale_))
        scores = X @ weights + intercept
        low, high = np.quantile(
            scores, [0.5 - abstain_fraction / 2.0, 0.5 + abstain_fraction / 2.0]
        )
        nonzero = {
            feature: float(weight)
            for feature, weight in zip(MODEL_FEATURES, weights)
            if abs(weight) > 1e-9
        }
        used[f"{variant}/{name}"] = set(nonzero)
        linear_export["models"][name] = {
            "intercept": intercept,
            "weights": nonzero,
            "n_nonzero": len(nonzero),
            "prune_threshold_r80": _threshold_for_recall(y, scores, 0.8),
            "prune_threshold_r50": _threshold_for_recall(y, scores, 0.5),
            "abstain_band": [float(low), float(high)],
        }
    (out_dir / f"rules_linear_{variant}.json").write_text(
        json.dumps(linear_export, indent=2) + "\n"
    )

    lines = [f"# Linear DelTriC edge rules, feature variant: {variant}", ""]
    for name, payload in linear_export["models"].items():
        lines.append(f"## {name}  ({payload['n_nonzero']} non-zero of {len(MODEL_FEATURES)})")
        terms = sorted(payload["weights"].items(), key=lambda kv: -abs(kv[1]))
        lines.append("S(e) = " + f"{payload['intercept']:+.4f}")
        for feature, weight in terms:
            lines.append(f"       {weight:+.4f} * {feature}")
        lines.append(f"PRUNE if S(e) >= {payload['prune_threshold_r80']:.4f}   (80% bridge recall)")
        lines.append(f"PRUNE if S(e) >= {payload['prune_threshold_r50']:.4f}   (50% bridge recall)")
        lines.append(
            f"ABSTAIN if {payload['abstain_band'][0]:.4f} < S(e) < {payload['abstain_band'][1]:.4f}"
        )
        lines.append("")
    (out_dir / f"rules_linear_{variant}.txt").write_text("\n".join(lines) + "\n")

    return used


def porting_checklist(used: dict[str, set[str]]) -> list[dict]:
    """One row per feature, attributed to the rules that need it.

    A union over every exported rule would list all 19 features and say
    nothing.  Attributing each feature to the rules that use it makes the
    porting cost of choosing a particular rule visible.
    """
    by_feature: dict[str, list[str]] = {}
    for rule, features in used.items():
        for feature in features:
            by_feature.setdefault(feature, []).append(rule)

    rows = []
    for feature in sorted(by_feature, key=lambda f: (-len(by_feature[f]), f)):
        status, note = BACKEND_AVAILABILITY.get(feature, ("unknown", ""))
        rules = sorted(by_feature[feature])
        rows.append(
            {
                "feature": feature,
                "backend_status": status,
                "n_rules_using": len(rules),
                "used_by": " ".join(rules),
                "note": note,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, default=_kinit_path.KINIT_DIR / "results" / "edge_dump")
    parser.add_argument("--out", type=Path, default=Path("results/edge_rules"))
    parser.add_argument(
        "--max-edges-per-run",
        type=int,
        default=800,
        help=(
            "uniform per-run subsample to keep the fits tractable; 0 uses every edge. "
            "The default yields ~1.7x the pooled sample the study's own grouped logistic "
            "baseline used, which is ample for 24 shape groups."
        ),
    )
    parser.add_argument("--abstain-fraction", type=float, default=0.2)
    parser.add_argument(
        "--leaf-prune-rate",
        type=float,
        default=0.5,
        help=(
            "observed bridge rate at which a tree leaf is labelled PRUNE. The default "
            "requires a leaf to be majority-bridge; lowering it toward the ~0.09 base "
            "rate trades intra-cluster edge retention for bridge recall."
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="parallel workers for the cross-validated fits; -1 uses every core",
    )
    parser.add_argument("--seed", type=int, default=20_250_818)
    args = parser.parse_args()

    X_raw, y, groups, runs, methods = load_dump(args.dump, args.max_edges_per_run, args.seed)
    assert_l1_is_sparse(len(y))
    print(
        f"rows={len(y)} positive_rate={np.mean(y):.4f} "
        f"shapes={len(np.unique(groups))} runs={len(np.unique(runs))}"
    )

    X_norm = normalise_within_run(X_raw, runs)

    metric_rows: list[dict] = []
    used_features: dict[str, set[str]] = {}
    for variant, X in (("raw", X_raw), ("normalized", X_norm)):
        print(f"\n=== variant: {variant} ===")
        metric_rows.extend(
            evaluate(X, y, groups, methods, variant, args.abstain_fraction, jobs=args.jobs)
        )
        used_features.update(export_rules(X, y, variant, args.out, args.abstain_fraction, args.leaf_prune_rate))

    write_csv(args.out / "rule_model_comparison.csv", metric_rows)
    write_csv(args.out / "porting_checklist.csv", porting_checklist(used_features))

    manifest = {
        "dump": str(args.dump),
        "out": str(args.out),
        "rows": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "shapes": int(len(np.unique(groups))),
        "runs": int(len(np.unique(runs))),
        "max_edges_per_run": args.max_edges_per_run,
        "abstain_fraction": args.abstain_fraction,
        "leaf_prune_rate": args.leaf_prune_rate,
        "seed": args.seed,
        "model_features": list(MODEL_FEATURES),
        "transform": transform_spec(),
        "features_used_by_exported_rules": {
            rule: sorted(features) for rule, features in sorted(used_features.items())
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("\n" + json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
