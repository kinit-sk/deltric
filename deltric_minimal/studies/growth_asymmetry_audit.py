"""Audit growth asymmetry without modifying clustering code or reading holdouts.

Inspect saved full-analysis graphs, test point-order invariance, and exercise
the actual growth/cost-gate functions on controlled weighted graphs.
Run with deltric_minimal/.venv/bin/python deltric_minimal/studies/growth_asymmetry_audit.py.
"""
from __future__ import annotations

import csv
import heapq
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import plot_seed_component_recomputed_growth as stage
from utils_component_growth import _assign_anomalies_cost_gate, _knn_core_distance

OUT = ROOT / "studies/results/growth_asymmetry_audit"
SAVED = ROOT / "results/full_analysis/stages"
NAMES = ["varied_2d_5c", "11_v2_moons_2d_2500",
         "04_v3_vardensity_blobs_20d_4500_k6", "14_v2_blobs_8d_3500_k7"]


def components(edges, mask, n):
    e = edges[mask]
    g = coo_matrix((np.ones(2 * len(e)),
                    (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n))
    return connected_components(g, directed=False)[1]


def coverage_history(a):
    edges, sl = a["edge_keys"], a["seed_labels"]
    initially_active = a["initially_large_seed_components"][sl].astype(bool)
    history = [initially_active]
    for r in range(int(a["growth_round_count"])):
        cc = components(edges, a["initial_mask"] | (
            a["accepted_growth_mask"] & (a["accepted_growth_step"] <= r)), len(sl))
        history.append(np.isin(cc, np.unique(cc[initially_active])))
    assert np.array_equal(history[-1], a["final_active_point"])
    return np.array(history)


def independent_cost_paths(labels, edges, lengths, core, p=.5, q=.5, reach=10.):
    """Comparator: keep separate (cluster, node) distances through initial noise.

    Existing assigned vertices remain barriers to other sources. Traversal of
    initial noise is independent of which cluster eventually wins it. This is
    a different intended assignment semantics, not a claimed drop-in repair.
    """
    clusters = np.unique(labels[labels >= 0])
    adjacency = [[] for _ in labels]
    for i, (u, v) in enumerate(edges):
        adjacency[u].append((v, i)); adjacency[v].append((u, i))
    all_distances = np.full((len(clusters), len(labels)), np.inf)
    for ci, c in enumerate(clusters):
        internal = (labels[edges[:, 0]] == c) & (labels[edges[:, 1]] == c)
        avg = max(float(np.mean(lengths[internal])) if internal.any()
                  else float(np.median(lengths)), 1e-12)
        ref = max(float(np.median(core[labels == c])), 1e-12)
        dist = all_distances[ci]
        sources = np.flatnonzero(labels == c)
        dist[sources] = 0.
        heap = [(0., int(node)) for node in sources]
        heapq.heapify(heap)
        while heap:
            d, u = heapq.heappop(heap)
            if d != dist[u]: continue
            for v, i in adjacency[u]:
                if labels[v] >= 0: continue
                cand = d + (lengths[i] * (core[v] / ref) ** q / avg) ** (1 + p)
                if cand <= reach and cand < dist[v]:
                    dist[v] = cand
                    heapq.heappush(heap, (cand, int(v)))
    result = labels.copy()
    if len(clusters):
        best = np.argmin(all_distances, axis=0)
        ok = (labels < 0) & np.isfinite(all_distances[best, np.arange(len(labels))])
        result[ok] = clusters[best[ok]]
    return result, all_distances


def controlled_growth(right_length):
    # A 12-point seed with 11 internal edges and two 8-hop branches.
    edges = [(i, i + 1) for i in range(11)]
    lengths = list(np.linspace(.7, 1.3, 11))
    for start, source, length in [(12, 0, 1.), (20, 11, right_length)]:
        edges += [(source, start)] + [(i, i + 1) for i in range(start, start + 7)]
        lengths += [length] * 8
    edges, lengths = np.array(edges), np.array(lengths)
    sl = np.r_[np.zeros(12, dtype=int), np.arange(1, 17)]
    counts = np.r_[11, np.zeros(16, dtype=int)]
    initial = np.arange(len(edges)) < 11
    result = stage._sequential_growth(edges, lengths, lengths, initial, sl, counts,
                                      10, np.ones(28), 1.5, 1.5)
    return dict(right_length=right_length,
                left_added=int(result["growth_added_point_mask"][12:20].sum()),
                right_added=int(result["growth_added_point_mask"][20:28].sum()),
                rounds=int(result["growth_round_count"]),
                left_steps=result["accepted_growth_step"][11:19].tolist(),
                right_steps=result["accepted_growth_step"][19:].tolist())


def controlled_cost_gate():
    # Equal hop lengths; only right-side local neighbor spacing is changed.
    edges = [(0, 1), (0, 2)] + [(i, i + 1) for i in range(2, 9)]
    edges += [(1, 10)] + [(i, i + 1) for i in range(10, 17)]
    edges = np.array(edges)
    labels = np.r_[0, 0, np.full(16, -1)]
    lengths = np.ones(len(edges))
    results = []
    for right_core in [1., 4.]:
        core = np.ones(18); core[10:] = right_core
        pred, diag = _assign_anomalies_cost_gate(labels, edges, lengths, 18,
            .5, 10., core_distance=core, density_power=.5, density_clip=False)
        results.append(dict(right_core=right_core, left_claimed=int((pred[2:10]>=0).sum()),
                            right_claimed=int((pred[10:]>=0).sum()), claims=diag["n_claims"]))
    # Cluster-dependent scales allow the best source to change after a junction.
    scale = 100 ** (1 / 1.5)
    edges2 = np.array([(0, 1), (2, 3), (0, 4), (2, 4), (4, 5)])
    lengths2 = np.array([1., scale, 1., scale * 2 ** (1/1.5), scale])
    labels2 = np.array([0, 0, 1, 1, -1, -1])
    pred2, _ = _assign_anomalies_cost_gate(labels2, edges2, lengths2, 6, .5, 5.,
                                         core_distance=np.ones(6), density_power=.5)
    oracle, distances = independent_cost_paths(labels2, edges2, lengths2, np.ones(6), reach=5.)
    # Exact symmetry: heap ties use cluster ID, with permanent downstream ownership.
    tie_lengths = np.ones(5)
    tie, _ = _assign_anomalies_cost_gate(labels2, edges2, tie_lengths, 6, .5, 5.)
    swapped = labels2.copy(); swapped[labels2 == 0] = 1; swapped[labels2 == 1] = 0
    tie_swapped, _ = _assign_anomalies_cost_gate(swapped, edges2, tie_lengths, 6, .5, 5.)
    mapped_back = tie_swapped.copy()
    mapped_back[tie_swapped == 0] = 1; mapped_back[tie_swapped == 1] = 0
    assert results[0]["left_claimed"] == results[0]["right_claimed"] == 8
    assert results[1]["left_claimed"] == 8 and results[1]["right_claimed"] == 3
    assert pred2[5] == -1 and oracle[5] == 1
    return dict(density_control=results,
                bottleneck=dict(actual_labels=pred2.tolist(), independent_labels=oracle.tolist(),
                    junction_cost_A=1., junction_cost_B=2., tail_cost_A=100., tail_cost_B=1.,
                    independent_tail_cost_B=float(distances[1, 5]), reach=5.),
                exact_tie=dict(original=tie.tolist(), after_label_swap_mapped_back=mapped_back.tolist()))


def audit_saved(name, directory=SAVED, tag="historical"):
    a = np.load(directory / (name + ".npz"))
    meta = json.loads((directory / (name + ".json")).read_text())
    data = np.load(ROOT / "data" / (name + ".npz"))
    X, y = StandardScaler().fit_transform(data["X"]), data["y"]
    e, P = a["edge_keys"], a["X_proj"]
    L = np.linalg.norm(X[e[:, 0]] - X[e[:, 1]], axis=1)
    LP = np.linalg.norm(P[e[:, 0]] - P[e[:, 1]], axis=1)
    restore = meta.get("restore_hard_limit")
    mode = meta.get("growth_restoration_mode", "real")
    args = (e, L, LP, a["initial_mask"], a["seed_labels"], a["seed_edge_counts"],
            10, np.ones(len(X)), meta["hard_growth_limit"], meta["projected_hard_growth_limit"], restore, mode)
    replay = stage._sequential_growth(*args)
    checks = dict(accepted_mask=np.array_equal(replay["accepted_growth_mask"], a["accepted_growth_mask"]),
                  accepted_steps=np.array_equal(replay["accepted_growth_step"], a["accepted_growth_step"]),
                  active_points=np.array_equal(replay["final_active_point"], a["final_active_point"]))
    # Historical artifacts may predate changes in the growth reference rule.
    # Report mismatches instead of presenting them as a current-code replay.
    checks["accepted_mask_mismatch_count"] = int((replay["accepted_growth_mask"] != a["accepted_growth_mask"]).sum())
    if tag == "current":
        assert checks["accepted_mask"] and checks["accepted_steps"] and checks["active_points"]
    # Renumber all points and seed components; preserve the weighted graph.
    rng = np.random.default_rng(20261008)
    order = rng.permutation(len(X)); old_to_new = np.argsort(order)
    cs = rng.permutation(len(a["seed_edge_counts"]))
    oldc_to_new = np.argsort(cs)
    shuffled = stage._sequential_growth(old_to_new[e], L, LP, a["initial_mask"],
        oldc_to_new[a["seed_labels"][order]], a["seed_edge_counts"][cs], 10,
        np.ones(len(X)), meta["hard_growth_limit"], meta["projected_hard_growth_limit"], restore, mode)
    permutation = dict(accepted_mask=np.array_equal(replay["accepted_growth_mask"], shuffled["accepted_growth_mask"]),
        accepted_steps=np.array_equal(replay["accepted_growth_step"], shuffled["accepted_growth_step"]),
        partition_ari=float(adjusted_rand_score(replay["final_component"], shuffled["final_component"][old_to_new])))
    assert permutation["accepted_mask"] and permutation["accepted_steps"] and permutation["partition_ari"] == 1.
    history = coverage_history(a)
    rows = []
    u, v = e.T
    for c in np.unique(y[y >= 0]):
        points = y == c
        internal = (y[u] == c) & (y[v] == c)
        row = dict(dataset=name, run=tag, truth_class=int(c), n=int(points.sum()),
            median_internal_length=float(np.median(L[internal])),
            fraction_internal_edges_seeded=float(a["initial_mask"][internal].mean()),
            active_seed_coverage=float(history[0, points].mean()),
            active_growth_coverage=float(history[-1, points].mean()),
            final_assigned_coverage=float((a["final_prediction"][points]>=0).mean()))
        current_active = replay["final_active_point"]
        row["current_rule_on_saved_seeds_coverage"] = float(current_active[points].mean())
        rows.append(row)
    # Reconstruct pre-cost-gate labels from the saved assignment mask.
    retained = a["final_retained_mask"].astype(bool)
    cc, bearing = stage._final_graph_components(e, retained, len(X))
    base = stage._final_component_prediction(e, retained, cc, bearing, 10)
    core = _knn_core_distance(X, 15)
    reach = meta["outlier_growth_metrics"]["reach"]
    pred, diag = _assign_anomalies_cost_gate(base, e, L, len(X), .5, reach,
                                            core_distance=core, density_power=.5)
    cost_replay = np.array_equal(pred, a["final_prediction"])
    assert cost_replay
    independent, _ = independent_cost_paths(base, e, L, core, reach=reach)
    swap = base.copy()
    n_clusters = int(base.max()) + 1
    assigned = base >= 0
    swap[assigned] = n_clusters - 1 - base[assigned]
    reversed_pred, _ = _assign_anomalies_cost_gate(swap, e, L, len(X), .5, reach,
                                                 core_distance=core, density_power=.5)
    reverse_assigned = reversed_pred >= 0
    reversed_pred[reverse_assigned] = n_clusters - 1 - reversed_pred[reverse_assigned]
    projected_limit = meta["projected_hard_growth_limit"]
    result = dict(dataset=name, run=tag, saved_config=dict(seed=meta["seed_hard_limit"],
                  original=meta["hard_growth_limit"],
                  projected=projected_limit if np.isfinite(projected_limit) else "disabled", reach=reach,
                  restore_hard_limit=restore, restoration_mode=mode),
        growth_replay=checks, point_and_component_permutation=permutation,
        cost_replay=bool(cost_replay), cost_claims=diag["n_claims"],
        reverse_cluster_ids_changed_points=int((pred != reversed_pred).sum()),
        independent_paths_changed_points=int((pred != independent).sum()),
        independent_paths_extra_assigned=int(((pred < 0) & (independent >= 0)).sum()),
        rounds=int(a["growth_round_count"]), classes=rows)
    np.savez_compressed(OUT / (tag + "__" + name + ".npz"), X_proj=P, y=y, coverage_history=history,
                        actual_cost=pred, independent_cost=independent)
    return result


def plot(results, controlled, tag="historical"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, name, title in zip(axes[:2], NAMES[:2], ["Unequal-density blobs", "Two moons"]):
        a = np.load(OUT / (tag + "__" + name + ".npz"))
        h, y = a["coverage_history"], a["y"]
        for c in np.unique(y[y >= 0]):
            ax.plot(np.arange(len(h)), 100 * h[:, y == c].mean(axis=1), marker=".", label=f"Class {c}")
        ax.set(title=title, xlabel="Completed growth rounds (0 = seeds)", ylabel="Points in active components (%)", ylim=(0, 105))
        ax.legend(fontsize=8); ax.grid(alpha=.2)
    controls = controlled["cost_gate"]["density_control"]
    x = np.arange(2)
    axes[2].bar(x-.18, [c["left_claimed"] for c in controls], .36, label="Left branch")
    axes[2].bar(x+.18, [c["right_claimed"] for c in controls], .36, label="Right branch")
    axes[2].set(xticks=x, xticklabels=["Equal spacing", "Right spacing ×4"],
                ylabel="Claimed points out of 8 per side", title="Controlled cost-gate test", ylim=(0,9))
    axes[2].legend(); axes[2].grid(axis="y",alpha=.2)
    fig.suptitle("Uneven growth is measurable; symmetry alone produces balanced growth", fontsize=14)
    caption = ("Left: reconstructed historical animation states." if tag == "historical" else
               "Left: fresh runs with current run_plot_stages.sh defaults, verified against exact function replays.")
    fig.text(.5, .01, caption + " Right: controlled cost-gate test, p=q=0.5, reach=10.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.93))
    fig.savefig(OUT / ("growth_asymmetry_" + tag + ".png"), dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    controlled = dict(growth=[controlled_growth(1.), controlled_growth(1.4)],
                      cost_gate=controlled_cost_gate())
    assert controlled["growth"][0]["left_steps"] == controlled["growth"][0]["right_steps"]
    assert controlled["growth"][1]["left_added"] == 8 and controlled["growth"][1]["right_added"] == 0
    results = []
    for name in NAMES:
        result = audit_saved(name)
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if k != "classes"}), flush=True)
    for name, folder in zip(NAMES[:2], ["current_varied", "current_moons"]):
        directory = OUT / folder
        if (directory / (name + ".npz")).exists():
            result = audit_saved(name, directory, "current")
            results.append(result)
            print(json.dumps({k: v for k, v in result.items() if k != "classes"}), flush=True)
    (OUT / "summary.json").write_text(json.dumps(dict(controlled=controlled, saved=results), indent=2, allow_nan=False))
    rows = [row for result in results for row in result["classes"]]
    with (OUT / "class_coverage.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    plot(results, controlled)
    if len(results) == 6:
        plot(results, controlled, "current")
    print(json.dumps(controlled), flush=True)


if __name__ == "__main__":
    main()
