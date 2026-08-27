#!/usr/bin/env python3
"""Reworked competitive-Dijkstra anomaly reclamation: gate/yardstick ablations.

Replays every variant against an identical frozen graph per dataset (pipeline
run once with ``component_growth_anomaly_reassign=False``), so differences are
attributable to the reassignment rule alone and not to UMAP seed variance.

Variants (each isolates one change against the production rule):

``base``      no reassignment stage (whatever growth/merge left as -1 stays).
``legacy``    production ``_assign_anomalies_dijkstra``, penalty_power=2,
              stop_ratio=3 (validated to reproduce the published regression).
``fixB``      gate semantics only: a claim needs cumulative *raw* projected
              length <= stop_ratio * scale_c AND every hop <= hop_cap_ratio *
              scale_c.  Penalized weights still only order contests.  The
              production rule gates the last hop exclusively, which permits
              arbitrarily long chains of sub-gate hops (single-linkage
              connectivity through voids).
``fixA``      yardstick only: scale_c = median over *retained* intra-cluster
              projected edges instead of the mean over ALL Delaunay edges whose
              endpoints share a label (the production definition includes
              pruned gap-crossing edges, inflating avg_c and loosening the
              gate).
``fixC``      eligibility only: only clusters with more than
              ``component_min_edges`` retained internal edges may claim, so
              accidentally-surviving anomaly clumps cannot compete as
              claimants (mirrors the trust rule the growth phase already
              applies).
``fixAB``     gate + yardstick.
``v2``        gate + yardstick + eligibility (full rework).

Usage:
    python studies/dijkstra_rework_study.py --limit 4
    python studies/dijkstra_rework_study.py --out studies/results/dijkstra_rework --jobs 6
"""

from __future__ import annotations

import argparse
import csv
import heapq
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

_DELTRIC_MINIMAL_DIR = Path(__file__).resolve().parents[1]
_DEFAULT_DATA_DIR = _DELTRIC_MINIMAL_DIR / "data"
sys.path.insert(0, str(_DELTRIC_MINIMAL_DIR.parent))

from deltric_minimal.utils_component_growth import (  # noqa: E402
    _assign_anomalies_dijkstra,
    cluster_tri,
)
from deltric_minimal.studies.anomaly_f1_dijkstra_vs_baseline import (  # noqa: E402
    BASE_CONFIG,
    DATASET_STEMS,
)

PENALTY_POWER = 2.0
STOP_RATIO = 3.0
HOP_CAP_RATIO = 2.0
COMPONENT_MIN_EDGES = 10

VARIANTS = ["base", "legacy", "fixB", "fixA", "fixC", "fixAB", "v2", "v2_orig"]


def assign_v2(
    labels: np.ndarray,
    edge_keys: np.ndarray,
    proj_edge_sizes: np.ndarray,
    final_mask: np.ndarray,
    n_points: int,
    penalty_power: float = PENALTY_POWER,
    stop_ratio: float = STOP_RATIO,
    hop_cap_ratio: float | None = None,
    scale_mode: str = "mean_all",
    trusted_only: bool = False,
    component_min_edges: int = COMPONENT_MIN_EDGES,
    orig_edge_sizes: np.ndarray | None = None,
    orig_hop_cap_ratio: float | None = None,
):
    """Flag-selectable reimplementation of the reclamation stage.

    ``hop_cap_ratio=None, scale_mode='mean_all', trusted_only=False``
    reproduces the production traversal (verified against
    ``_assign_anomalies_dijkstra`` in ``validate_legacy``).
    """
    labels = np.asarray(labels, dtype=np.int64)
    edge_keys = np.asarray(edge_keys, dtype=np.int64)
    proj_edge_sizes = np.asarray(proj_edge_sizes, dtype=np.float64)
    final_mask = np.asarray(final_mask, dtype=bool)
    n_clusters = int(labels.max()) + 1 if np.any(labels >= 0) else 0
    diag = {
        "n_anomalies_in": int(np.count_nonzero(labels < 0)),
        "n_anomalies_out": 0,
        "n_rounds": 0,
        "n_claimants": 0,
    }
    if n_clusters == 0 or diag["n_anomalies_in"] == 0:
        diag["n_anomalies_out"] = diag["n_anomalies_in"]
        return labels.copy(), diag

    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(n_points)]
    for edge_index in range(len(edge_keys)):
        u, v = int(edge_keys[edge_index, 0]), int(edge_keys[edge_index, 1])
        adjacency[u].append((v, edge_index))
        adjacency[v].append((u, edge_index))

    global_median = float(np.median(proj_edge_sizes)) if len(proj_edge_sizes) else 1.0
    scale = np.full(n_clusters, global_median, dtype=np.float64)
    retained_internal_count = np.zeros(n_clusters, dtype=np.int64)
    sum_all = np.zeros(n_clusters, dtype=np.float64)
    count_all = np.zeros(n_clusters, dtype=np.int64)
    retained_vals: list[list[float]] = [[] for _ in range(n_clusters)]
    u_arr, v_arr = edge_keys[:, 0], edge_keys[:, 1]
    lu, lv = labels[u_arr], labels[v_arr]
    same = (lu >= 0) & (lu == lv)
    np.add.at(sum_all, lu[same], proj_edge_sizes[same])
    np.add.at(count_all, lu[same], 1)
    for edge_index in np.flatnonzero(final_mask & same):
        retained_vals[int(lu[edge_index])].append(float(proj_edge_sizes[edge_index]))
    for cluster in range(n_clusters):
        if scale_mode == "median_retained" and retained_vals[cluster]:
            scale[cluster] = float(np.median(retained_vals[cluster]))
        elif count_all[cluster] > 0:
            scale[cluster] = float(sum_all[cluster] / count_all[cluster])
        retained_internal_count[cluster] = len(retained_vals[cluster])
    scale = np.maximum(scale, 1e-12)
    eligible = (
        (retained_internal_count > int(component_min_edges))
        if trusted_only
        else np.ones(n_clusters, dtype=bool)
    )

    orig_scale = None
    if orig_edge_sizes is not None and orig_hop_cap_ratio is not None:
        orig_edge_sizes = np.asarray(orig_edge_sizes, dtype=np.float64)
        global_orig = float(np.median(orig_edge_sizes))
        orig_scale = np.full(n_clusters, global_orig, dtype=np.float64)
        orig_retained: list[list[float]] = [[] for _ in range(n_clusters)]
        orig_all_count = np.zeros(n_clusters, dtype=np.int64)
        orig_all_sum = np.zeros(n_clusters, dtype=np.float64)
        for i in np.flatnonzero(same):
            orig_all_sum[lu[i]] += orig_edge_sizes[i]
            orig_all_count[lu[i]] += 1
        for edge_index in np.flatnonzero(final_mask & same):
            orig_retained[int(lu[edge_index])].append(float(orig_edge_sizes[edge_index]))
        for cluster in range(n_clusters):
            if scale_mode == "median_retained" and orig_retained[cluster]:
                orig_scale[cluster] = float(np.median(orig_retained[cluster]))
            elif orig_all_count[cluster] > 0:
                orig_scale[cluster] = float(orig_all_sum[cluster] / orig_all_count[cluster])

    owner = labels.copy()
    heaps: list[list[tuple[float, int, int, float]]] = [[] for _ in range(n_clusters)]
    for cluster in range(n_clusters):
        if not eligible[cluster]:
            continue
        for node in np.flatnonzero(labels == cluster):
            for neighbor, edge_index in adjacency[int(node)]:
                if labels[neighbor] == cluster:
                    continue
                length = float(proj_edge_sizes[edge_index])
                weight = length * (length / scale[cluster]) ** penalty_power
                heapq.heappush(heaps[cluster], (weight, neighbor, edge_index, length))
    active_roots = {cluster for cluster in range(n_clusters) if heaps[cluster]}
    diag["n_claimants"] = len(active_roots)

    max_rounds = 4 * n_points + 16
    round_index = 0
    while round_index < max_rounds and active_roots:
        root_candidate: dict[int, tuple[float, int, int, float]] = {}
        dead_roots = []
        for root in list(active_roots):
            heap = heaps[root]
            while heap:
                dist, node, edge_index, raw_sum = heap[0]
                if owner[node] != -1:
                    heapq.heappop(heap)
                    continue
                length = float(proj_edge_sizes[edge_index])
                if hop_cap_ratio is not None and length > hop_cap_ratio * scale[root]:
                    heapq.heappop(heap)
                    continue
                if hop_cap_ratio is not None and raw_sum > stop_ratio * scale[root]:
                    heapq.heappop(heap)
                    continue
                if hop_cap_ratio is None and length > stop_ratio * scale[root]:
                    heapq.heappop(heap)
                    continue
                if orig_scale is not None and float(
                    orig_edge_sizes[edge_index]
                ) > orig_hop_cap_ratio * orig_scale[root]:
                    heapq.heappop(heap)
                    continue
                root_candidate[root] = (dist, node, edge_index, raw_sum)
                break
            if not heap:
                dead_roots.append(root)
        for root in dead_roots:
            active_roots.discard(root)
        if not root_candidate:
            break

        proposals: dict[int, tuple[float, int, int, float]] = {}
        for root, (dist, node, edge_index, raw_sum) in root_candidate.items():
            current = proposals.get(node)
            if current is None or dist < current[0]:
                proposals[node] = (dist, root, edge_index, raw_sum)

        for node, (dist, root, edge_index, raw_sum) in proposals.items():
            if owner[node] != -1:
                continue
            heapq.heappop(heaps[root])
            owner[node] = root
            for neighbor, next_edge_index in adjacency[node]:
                if owner[neighbor] != -1:
                    continue
                length = float(proj_edge_sizes[next_edge_index])
                weight = length * (length / scale[root]) ** penalty_power
                heapq.heappush(
                    heaps[root],
                    (dist + weight, neighbor, next_edge_index, raw_sum + length),
                )
        round_index += 1

    diag["n_anomalies_out"] = int(np.count_nonzero(owner < 0))
    diag["n_rounds"] = round_index
    return owner, diag


def validate_legacy(labels, edge_keys, proj_sizes, n) -> bool:
    ref, _ = _assign_anomalies_dijkstra(
        labels, edge_keys, proj_sizes, n,
        penalty_power=PENALTY_POWER, stop_ratio=STOP_RATIO,
    )
    mine, _ = assign_v2(
        labels, edge_keys, proj_sizes,
        np.ones(len(edge_keys), dtype=bool), n,
        penalty_power=PENALTY_POWER, stop_ratio=STOP_RATIO,
        hop_cap_ratio=None, scale_mode="mean_all", trusted_only=False,
    )
    return bool(np.array_equal(ref, mine))


def scores(y_true_anomaly: np.ndarray, pred_labels: np.ndarray) -> dict:
    pred_anomaly = pred_labels < 0
    return {
        "precision": float(precision_score(y_true_anomaly, pred_anomaly, zero_division=0)),
        "recall": float(recall_score(y_true_anomaly, pred_anomaly, zero_division=0)),
        "f1": float(f1_score(y_true_anomaly, pred_anomaly, zero_division=0)),
        "n_pred_anomaly": int(pred_anomaly.sum()),
    }


def run_one(path: Path) -> list[dict]:
    npz = np.load(path, allow_pickle=False)
    X = StandardScaler().fit_transform(np.asarray(npz["X"], dtype=np.float64))
    y = np.asarray(npz["y"]).astype(np.int64)
    y_true_anomaly = y < 0
    real = ~y_true_anomaly

    cluster_tri(X, **BASE_CONFIG, component_growth_anomaly_reassign=False)
    state = cluster_tri.last_component_growth
    merged = state["merged_labels"]
    edge_keys = state["edge_keys"]
    proj_sizes = state["projected_edge_sizes"]
    final_mask = state["final_mask"]
    n = len(X)

    base_scores = scores(y_true_anomaly, merged)
    base_ari_all = float(adjusted_rand_score(y, merged))
    base_ari_real = float(adjusted_rand_score(y[real], merged[real])) if real.any() else float("nan")

    legacy_ok = validate_legacy(merged, edge_keys, proj_sizes, n)
    cfgs = {
        "legacy": dict(hop_cap_ratio=None, scale_mode="mean_all", trusted_only=False),
        "fixB": dict(hop_cap_ratio=HOP_CAP_RATIO, scale_mode="mean_all", trusted_only=False),
        "fixA": dict(hop_cap_ratio=None, scale_mode="median_retained", trusted_only=False),
        "fixC": dict(hop_cap_ratio=None, scale_mode="mean_all", trusted_only=True),
        "fixAB": dict(hop_cap_ratio=HOP_CAP_RATIO, scale_mode="median_retained", trusted_only=False),
        "v2": dict(hop_cap_ratio=HOP_CAP_RATIO, scale_mode="median_retained", trusted_only=True),
        "v2_orig": dict(hop_cap_ratio=HOP_CAP_RATIO, scale_mode="median_retained",
                        trusted_only=True, orig_hop_cap_ratio=HOP_CAP_RATIO),
    }

    def row_for(variant, sc, ari_all, ari_real, frac, diag, seconds):
        return {
            "dataset": path.stem,
            "d": int(X.shape[1]),
            "n": n,
            "n_true_anomaly": int(y_true_anomaly.sum()),
            "variant": variant,
            "legacy_replay_exact": "",
            "precision": sc["precision"],
            "recall": sc["recall"],
            "f1": sc["f1"],
            "delta_f1_vs_base": sc["f1"] - base_scores["f1"],
            "ari_all": ari_all,
            "delta_ari_all_vs_base": ari_all - base_ari_all,
            "ari_real": ari_real,
            "delta_ari_real_vs_base": ari_real - base_ari_real,
            "frac_assigned": frac,
            "n_pred_anomaly": sc["n_pred_anomaly"],
            "n_anomalies_in": diag.get("n_anomalies_in", "") if diag else "",
            "n_anomalies_out": diag.get("n_anomalies_out", "") if diag else "",
            "n_claimants": diag.get("n_claimants", "") if diag else "",
            "stage_seconds": round(seconds, 4),
        }

    rows = [
        row_for("base", base_scores, base_ari_all, base_ari_real,
                float((merged >= 0).mean()), None, 0.0)
    ]

    started = time.perf_counter()
    leg_labels, leg_diag = _assign_anomalies_dijkstra(
        merged, edge_keys, proj_sizes, n,
        penalty_power=PENALTY_POWER, stop_ratio=STOP_RATIO,
    )
    leg_seconds = time.perf_counter() - started
    leg_sc = scores(y_true_anomaly, leg_labels)
    rows.append(row_for(
        "legacy", leg_sc,
        float(adjusted_rand_score(y, leg_labels)),
        float(adjusted_rand_score(y[real], leg_labels[real])) if real.any() else float("nan"),
        float((leg_labels >= 0).mean()), leg_diag, leg_seconds,
    ))
    rows[-1]["legacy_replay_exact"] = str(legacy_ok)

    for variant, cfg in cfgs.items():
        if variant == "legacy":
            continue
        started = time.perf_counter()
        got, diag = assign_v2(
            merged, edge_keys, proj_sizes, final_mask, n,
            penalty_power=PENALTY_POWER, stop_ratio=STOP_RATIO,
            orig_edge_sizes=state["orig_edge_sizes"], **cfg,
        )
        seconds = time.perf_counter() - started
        sc = scores(y_true_anomaly, got)
        rows.append(row_for(
            variant, sc,
            float(adjusted_rand_score(y, got)),
            float(adjusted_rand_score(y[real], got[real])) if real.any() else float("nan"),
            float((got >= 0).mean()), diag, seconds,
        ))

    print(f"  {path.stem:<44} d={X.shape[1]:<4} "
          + "  ".join(f"{r['variant']}={r['f1']:.3f}" for r in rows)
          + f"  replay_ok={legacy_ok}",
          flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=_DEFAULT_DATA_DIR, type=Path)
    parser.add_argument("--out", default=Path("studies/results/dijkstra_rework"), type=Path)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None,
                        help="cap the number of datasets (smoke runs)")
    args = parser.parse_args()

    files = [args.data / f"{stem}.npz" for stem in DATASET_STEMS]
    files += sorted(args.data.glob("*__anom10.npz"))
    unique = {}
    for path in files:
        if path.exists():
            unique[path.stem] = path
    files = sorted(unique.values())
    if args.limit:
        idx = np.linspace(0, len(files) - 1, min(args.limit, len(files))).astype(int)
        files = [files[i] for i in dict.fromkeys(idx)]
    print(f"{len(files)} datasets, variants={VARIANTS}", flush=True)

    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            nested = list(pool.map(run_one, files))
    else:
        nested = [run_one(path) for path in files]
    rows = [row for group in nested for row in group]

    args.out.mkdir(parents=True, exist_ok=True)
    out_csv = args.out / "dijkstra_rework.csv"
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== mean over datasets ===")
    print(f"{'variant':<10} {'F1':>7} {'dF1':>8} {'P':>7} {'R(noise)':>9} "
          f"{'ARI_all':>8} {'ARI_real':>9} {'frac_asg':>9} {'W/L vs base':>12}")
    base_f1 = {r["dataset"]: r["f1"] for r in rows if r["variant"] == "base"}
    base_ar = {r["dataset"]: r["ari_all"] for r in rows if r["variant"] == "base"}
    base_arr = {r["dataset"]: r["ari_real"] for r in rows if r["variant"] == "base"}
    for variant in VARIANTS:
        sel = [r for r in rows if r["variant"] == variant]
        d = [r["f1"] - base_f1[r["dataset"]] for r in sel]
        da = [r["ari_all"] - base_ar[r["dataset"]] for r in sel]
        dar = [r["ari_real"] - base_arr[r["dataset"]] for r in sel]
        print(f"{variant:<10} {np.mean([r['f1'] for r in sel]):>7.4f} "
              f"{np.mean(d):>+8.4f} {np.mean([r['precision'] for r in sel]):>7.4f} "
              f"{np.mean([r['recall'] for r in sel]):>9.4f} "
              f"{np.mean(da):>+8.4f} {np.mean(dar):>+9.4f} "
              f"{np.mean([r['frac_assigned'] for r in sel]):>9.4f} "
              f"{sum(x > 1e-9 for x in d):>5}/{sum(x < -1e-9 for x in d)}")
        if variant != "base":
            print(f"{'':<10} ARI_real W/L={sum(x > 1e-9 for x in dar)}/"
                  f"{sum(x < -1e-9 for x in dar)}   "
                  f"ARI_all W/L={sum(x > 1e-9 for x in da)}/{sum(x < -1e-9 for x in da)}")

    dims = sorted({r["d"] for r in rows})
    print("\n=== delta F1 stratified by dimensionality ===")
    print(f"{'variant':<10}" + "".join(f"{'d=' + str(dd):>10}" for dd in dims))
    for variant in VARIANTS:
        cells = []
        for dd in dims:
            deltas = [r["delta_f1_vs_base"] for r in rows
                      if r["variant"] == variant and r["d"] == dd]
            cells.append(f"{np.mean(deltas):>+10.4f}" if deltas else f"{'--':>10}")
        print(f"{variant:<10}" + "".join(cells))

    bad = {r["dataset"] for r in rows if r["legacy_replay_exact"] == "False"}
    if bad:
        print(f"\nWARNING legacy replay mismatch on: {sorted(bad)}")
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
