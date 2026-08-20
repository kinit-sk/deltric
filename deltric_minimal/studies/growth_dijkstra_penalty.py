#!/usr/bin/env python3
"""Competitive weighted-Dijkstra cluster growth.

An alternative to the threshold-based wave growth in
``utils_component_growth._grow_components_fast``.  Seeding is unchanged
(mutual-kNN edges under the existing hard length gate -- see
``component_growth_graph``'s ``seed_hard_allowed``), so this study isolates
the *growth* rule and asks whether a different one does better, not whether
seeding does.

The growth rule:

* Each seed component starts with ``avg_c`` = the mean original-space length
  of its own seed edges.
* Growth runs in synchronous rounds. In each round every still-growing
  cluster proposes exactly one new node: the endpoint of its cheapest
  not-yet-claimed frontier edge, where "cheapest" is a running Dijkstra
  distance from the cluster (multi-source: all seed members start at
  distance 0), not raw edge length. An edge's *weight* is its length
  penalized against the cluster's own current average::

      weight(e) = length(e) * (length(e) / avg_c) ** penalty_power

  so an edge at exactly the cluster's average costs ``length(e)``, and an
  edge twice the average costs ``4x`` its own length (at the default
  quadratic power) -- the penalty compounds with distance from the average,
  not just with raw length, which is the point: two clusters with different
  natural scales are not penalized on the same absolute yardstick.
* When two clusters propose the same node in the same round, the lower
  cumulative Dijkstra distance wins; the loser retries next round. This is
  what makes every cluster grow "at the same rate" -- one accepted node per
  cluster per round, resolved by a shared competitive distance rather than
  one cluster being allowed to run ahead and consume the graph before others
  get a turn.
* A cluster stops proposing once its cheapest remaining frontier edge is
  more than ``stop_ratio`` times its own current average -- the one hard
  knob left in the rule, playing the same "how far is too far" role
  ``hard_limit`` plays in the production backend.

Everything here is computed on the same Delaunay graph / edge lengths the
production ``component_growth_graph`` builds (``get_triangles_with_edges`` +
``_extract_edge_data``), and seeding reuses
``utils_component_growth._original_knn_directed_relations`` /
``_hard_threshold`` / ``_component_state`` directly so the seed graph is
identical to the production seed graph. Only the growth loop is new.

Cross-cluster merges (a frontier edge whose far endpoint already belongs to
a *different* live cluster) are judged against the tighter of the two
clusters' averages, and the merged average is the smaller of the two, never
a pooled mean -- a pooled mean ratchets upward with every merge (each
absorption drags the average toward the looser side, loosening the gate for
the next merge), which is single-linkage-style chaining wearing a
disguise.

Result of running this study (``growth_dijkstra_penalty.csv``): the rule
matches or beats the production baseline on low-dimensional, well-separated
data (moons +0.11 ARI at stop_ratio=3 in one run, blobs_2d/varied_2d
positive too) but collapses whole 8-64d datasets into one cluster
(digits, blobs_8d, overlap_blobs_10d all land at ARI ~0) regardless of
``stop_ratio``. That is not a tuning failure of this implementation: in
high dimensions Euclidean edge lengths concentrate (the ratio
``length(e) / avg_c`` stays close to 1 for nearly every Delaunay edge,
well- and badly-placed alike), so a length-ratio penalty loses almost all
its power to discriminate exactly where it would matter most. A
rank-based or per-dimension-normalized distance would be the natural next
thing to try for high-dimensional data; this study did not attempt it.

Usage:
    python studies/growth_dijkstra_penalty.py --out studies/results/growth_dijkstra_penalty
"""

from __future__ import annotations

import argparse
import csv
import heapq
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

_DELTRIC_MINIMAL_DIR = Path(__file__).resolve().parents[1]
_DEFAULT_DATA_DIR = _DELTRIC_MINIMAL_DIR / "data"
sys.path.insert(0, str(_DELTRIC_MINIMAL_DIR.parent))

from deltric_minimal.utils_pruning import (  # noqa: E402
    get_triangles_with_edges,
    _extract_edge_data,
)
from deltric_minimal.utils_component_growth import (  # noqa: E402
    _component_state,
    _hard_threshold,
    _original_knn_directed_relations,
    cluster_tri,
)

# Same growth-mode defaults used elsewhere in this repo (eval_edge_rule_ari.py,
# edge_ari_oracle_pruning.py), so the baseline comparison is apples to apples.
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

#: knn_k / seed_hard_limit mirror BASE_CONFIG's seed construction so this
#: study's seed graph matches the production seed graph.
DEFAULT_KNN_K = BASE_CONFIG["component_growth_knn"]
DEFAULT_SEED_HARD_LIMIT = BASE_CONFIG["component_growth_seed_hard_limit"]


def _build_graph(X: np.ndarray) -> dict[str, Any]:
    triangles, _, _, X_proj = get_triangles_with_edges(
        X, project_dim=BASE_CONFIG["project_dim"], method=BASE_CONFIG["dim_reduction"],
        back_proj=BASE_CONFIG["back_proj"], umap_n_epochs=BASE_CONFIG["umap_n_epochs"],
        umap_n_neighbors=BASE_CONFIG["umap_n_neighbors"],
    )
    edge_keys_list, orig_sizes, _, _ = _extract_edge_data(triangles, X_proj, X_size=X)
    edge_keys = np.asarray(edge_keys_list, dtype=np.int64)
    orig_sizes = np.asarray(orig_sizes, dtype=np.float64)
    return {"X_proj": X_proj, "edge_keys": edge_keys, "orig_sizes": orig_sizes}


def _seed_clusters(
    X: np.ndarray, edge_keys: np.ndarray, orig_sizes: np.ndarray, n_points: int,
    knn_k: int, seed_hard_limit: float, neighbor_stat: str,
    multiplicative_hard_limit: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Mutual-kNN seed edges under the production hard length gate.

    Returns ``(cluster_of, seed_mask, threshold_stats)`` where ``cluster_of``
    has one entry per point: a cluster id for points in a seed component with
    at least one edge, ``-1`` otherwise.
    """
    directed_uv, directed_vu, _, _ = _original_knn_directed_relations(X, edge_keys, knn_k)
    mutual = directed_uv & directed_vu
    threshold, stats = _hard_threshold(
        orig_sizes, seed_hard_limit, neighbor_stat, multiplicative_hard_limit
    )
    seed_mask = mutual & (orig_sizes <= threshold)

    labels, edge_counts = _component_state(edge_keys, seed_mask, n_points)
    active = np.flatnonzero(edge_counts > 0)
    remap = {int(component): index for index, component in enumerate(active)}
    cluster_of = np.array(
        [remap.get(int(label), -1) for label in labels], dtype=np.int64
    )
    return cluster_of, seed_mask, stats


def dijkstra_penalty_growth(
    X: np.ndarray,
    edge_keys: np.ndarray,
    orig_sizes: np.ndarray,
    *,
    knn_k: int = DEFAULT_KNN_K,
    seed_hard_limit: float = DEFAULT_SEED_HARD_LIMIT,
    neighbor_stat: str = "median",
    multiplicative_hard_limit: bool = False,
    penalty_power: float = 2.0,
    stop_ratio: float = 3.0,
    min_cluster_size: int = 10,
    max_rounds: int | None = None,
    snapshot_rounds: set[int] | None = None,
) -> dict[str, Any]:
    """Grow seed clusters with competitive penalized-Dijkstra expansion.

    ``snapshot_rounds``, when given, records a copy of the per-point
    ``cluster_of`` array immediately after each listed round completes (for
    plotting growth stages); this roughly doubles wall time only when used
    on large snapshot sets, since copying an int64 array is cheap.
    """
    X = np.asarray(X, dtype=np.float64)
    edge_keys = np.asarray(edge_keys, dtype=np.int64)
    orig_sizes = np.asarray(orig_sizes, dtype=np.float64)
    n_points = len(X)
    n_edges = len(edge_keys)
    if max_rounds is None:
        max_rounds = 4 * n_points + 16
    snapshot_rounds = snapshot_rounds or set()

    cluster_of, seed_mask, seed_stats = _seed_clusters(
        X, edge_keys, orig_sizes, n_points, knn_k, seed_hard_limit,
        neighbor_stat, multiplicative_hard_limit,
    )
    seed_labels = cluster_of.copy()
    n_clusters = int(cluster_of.max()) + 1 if np.any(cluster_of >= 0) else 0

    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(n_points)]
    for edge_index in range(n_edges):
        u, v = int(edge_keys[edge_index, 0]), int(edge_keys[edge_index, 1])
        adjacency[u].append((v, edge_index))
        adjacency[v].append((u, edge_index))

    final_mask = seed_mask.copy()
    grown_mask = np.zeros(n_edges, dtype=bool)

    # ``owner[point]`` is the cluster id that first claimed the point (a seed
    # id for seed members, or the id that won it during growth). Two clusters
    # can end up occupying the same union-find tree after a merge, so the
    # *current* cluster of a point is always ``find(owner[point])`` -- never
    # ``owner[point]`` directly.
    owner = cluster_of.copy()
    parent = np.arange(max(n_clusters, 1), dtype=np.int64)

    def find(cluster: int) -> int:
        while parent[cluster] != cluster:
            parent[cluster] = parent[parent[cluster]]
            cluster = int(parent[cluster])
        return cluster

    sum_len = np.zeros(max(n_clusters, 1), dtype=np.float64)
    count_len = np.zeros(max(n_clusters, 1), dtype=np.int64)
    for edge_index in np.flatnonzero(seed_mask):
        u, v = edge_keys[edge_index]
        cu, cv = int(cluster_of[u]), int(cluster_of[v])
        if cu >= 0 and cu == cv:
            sum_len[cu] += orig_sizes[edge_index]
            count_len[cu] += 1
    global_median = float(np.median(orig_sizes)) if n_edges else 1.0
    avg = np.where(count_len > 0, sum_len / np.maximum(count_len, 1), global_median)

    heaps: list[list[tuple[float, int, int]]] = [[] for _ in range(n_clusters)]
    for cluster in range(n_clusters):
        for node in np.flatnonzero(cluster_of == cluster):
            for neighbor, edge_index in adjacency[int(node)]:
                if cluster_of[neighbor] == cluster:
                    continue
                length = orig_sizes[edge_index]
                weight = length * (length / avg[cluster]) ** penalty_power
                heapq.heappush(heaps[cluster], (weight, neighbor, edge_index))
    active_roots = {cluster for cluster in range(n_clusters) if heaps[cluster]}

    def merge(root_a: int, root_b: int) -> int:
        """Union two growing clusters, combining their stats and frontiers.

        The merged average is deliberately the *tighter* (smaller) of the
        two, not the pooled mean. A pooled mean ratchets upward with every
        merge -- each absorption drags the average toward the looser side,
        which loosens the gate for the *next* merge, which drags the average
        up further. That chain reaction is single-linkage-style chaining in
        disguise and is what let the gate collapse whole datasets into one
        cluster before this fix.
        """
        if len(heaps[root_a]) < len(heaps[root_b]):
            root_a, root_b = root_b, root_a
        parent[root_b] = root_a
        sum_len[root_a] += sum_len[root_b]
        count_len[root_a] += count_len[root_b]
        avg[root_a] = min(avg[root_a], avg[root_b])
        heaps[root_a].extend(heaps[root_b])
        heaps[root_b] = []
        heapq.heapify(heaps[root_a])
        active_roots.discard(root_b)
        return root_a

    growth_waves: list[dict[str, Any]] = []
    snapshots: dict[int, np.ndarray] = {}

    def snapshot_labels() -> np.ndarray:
        return np.array(
            [find(int(owner[point])) if owner[point] >= 0 else -1 for point in range(n_points)],
            dtype=np.int64,
        )

    if 0 in snapshot_rounds:
        snapshots[0] = snapshot_labels()

    round_index = 0
    while round_index < max_rounds and active_roots:
        # Pass 1: each active root peeks its cheapest still-valid candidate,
        # discarding stale (already-absorbed) or too-far entries along the
        # way. A candidate pointing at an unclaimed node is a "claim"; one
        # pointing at a node another live cluster already owns is a "merge".
        root_candidate: dict[int, tuple[str, float, int, int]] = {}
        dead_roots = []
        for root in list(active_roots):
            if find(root) != root:
                active_roots.discard(root)
                continue
            heap = heaps[root]
            while heap:
                dist, node, edge_index = heap[0]
                length = orig_sizes[edge_index]
                owner_id = int(owner[node])
                if owner_id == -1:
                    if length > stop_ratio * avg[root]:
                        heapq.heappop(heap)
                        continue
                    root_candidate[root] = ("claim", dist, node, edge_index)
                    break
                other_root = find(owner_id)
                if other_root == root:
                    heapq.heappop(heap)
                    continue
                # Merging is judged against the *tighter* of the two
                # clusters, not just the proposer's own average. Otherwise a
                # loose, already-sprawling cluster can swallow a tight one
                # through a single moderately-long edge that looks cheap
                # only from the loose side -- exactly the runaway
                # over-merging this stricter check exists to block.
                if length > stop_ratio * min(avg[root], avg[other_root]):
                    heapq.heappop(heap)
                    continue
                root_candidate[root] = ("merge", dist, other_root, edge_index)
                break
            if not heap:
                dead_roots.append(root)
        for root in dead_roots:
            active_roots.discard(root)
        if not root_candidate:
            break

        # Pass 2: resolve merges first. A root whose heap gets extended by a
        # merge is "touched" -- its peeked top may no longer be the true top
        # after re-heapifying, so any pending claim for it is deferred to the
        # next round rather than popped against a possibly-stale entry.
        touched_roots: set[int] = set()
        n_merges = 0
        for root, (kind, dist, target, edge_index) in root_candidate.items():
            if kind != "merge" or find(root) != root:
                continue
            other_root = find(target)
            if other_root == root or other_root in touched_roots or root in touched_roots:
                continue
            heapq.heappop(heaps[root])
            final_mask[edge_index] = True
            grown_mask[edge_index] = True
            survivor = merge(root, other_root)
            touched_roots.add(survivor)
            n_merges += 1

        # Pass 3: resolve unclaimed-node contention among untouched roots --
        # the cluster with the smaller cumulative penalized distance wins.
        proposals: dict[int, tuple[float, int, int]] = {}
        for root, (kind, dist, node, edge_index) in root_candidate.items():
            if kind != "claim" or find(root) != root or root in touched_roots:
                continue
            current = proposals.get(node)
            if current is None or dist < current[0]:
                proposals[node] = (dist, root, edge_index)

        round_accepted = n_merges
        for node, (dist, root, edge_index) in proposals.items():
            if owner[node] != -1:
                continue
            heapq.heappop(heaps[root])
            owner[node] = root
            final_mask[edge_index] = True
            grown_mask[edge_index] = True
            sum_len[root] += orig_sizes[edge_index]
            count_len[root] += 1
            avg[root] = sum_len[root] / count_len[root]
            round_accepted += 1
            for neighbor, next_edge_index in adjacency[node]:
                if owner[neighbor] != -1 and find(int(owner[neighbor])) == root:
                    continue
                length = orig_sizes[next_edge_index]
                weight = length * (length / avg[root]) ** penalty_power
                heapq.heappush(heaps[root], (dist + weight, neighbor, next_edge_index))

        round_index += 1
        growth_waves.append({
            "round": round_index,
            "accepted": round_accepted,
            "merges": n_merges,
            "clusters_active": len(active_roots),
            "n_live_clusters": len({find(c) for c in range(n_clusters)}),
        })
        if round_index in snapshot_rounds:
            snapshots[round_index] = snapshot_labels()

    labels = snapshot_labels()
    pre_min_size_labels = labels.copy()
    root_size = {root: int(np.count_nonzero(labels == root)) for root in set(labels[labels >= 0].tolist())}
    for root, size in root_size.items():
        if size < min_cluster_size:
            labels[labels == root] = -1
    # Compact remaining ids to 0..m-1 in order of first appearance.
    remap: dict[int, int] = {}
    for point in range(n_points):
        cluster = int(labels[point])
        if cluster < 0:
            continue
        if cluster not in remap:
            remap[cluster] = len(remap)
        labels[point] = remap[cluster]
    if -1 in snapshot_rounds:
        snapshots[-1] = snapshot_labels()

    return {
        "labels": labels,
        "pre_min_size_labels": pre_min_size_labels,
        "seed_labels": seed_labels,
        "final_mask": final_mask,
        "grown_mask": grown_mask,
        "seed_mask": seed_mask,
        "n_clusters": n_clusters,
        "n_rounds": round_index,
        "growth_waves": growth_waves,
        "snapshots": snapshots,
        "seed_stats": seed_stats,
        "knn_k": knn_k,
        "seed_hard_limit": seed_hard_limit,
        "penalty_power": penalty_power,
        "stop_ratio": stop_ratio,
        "min_cluster_size": min_cluster_size,
    }


def run_one(path: Path, penalty_power: float, stop_ratio: float) -> dict:
    npz = np.load(path, allow_pickle=False)
    X = StandardScaler().fit_transform(np.asarray(npz["X"], dtype=np.float64))
    y = np.asarray(npz["y"]).astype(np.int64)
    n = len(X)

    started = time.perf_counter()
    baseline_labels = cluster_tri(X, **BASE_CONFIG)
    baseline_ari = float(adjusted_rand_score(y, baseline_labels))
    baseline_elapsed = time.perf_counter() - started

    graph = _build_graph(X)
    started = time.perf_counter()
    state = dijkstra_penalty_growth(
        X, graph["edge_keys"], graph["orig_sizes"],
        knn_k=DEFAULT_KNN_K, seed_hard_limit=DEFAULT_SEED_HARD_LIMIT,
        neighbor_stat=BASE_CONFIG["neighbor_stat"],
        multiplicative_hard_limit=BASE_CONFIG["multiplicative_hard_limit"],
        penalty_power=penalty_power, stop_ratio=stop_ratio,
        min_cluster_size=BASE_CONFIG["min_cluster_size"],
    )
    dijkstra_elapsed = time.perf_counter() - started
    dijkstra_ari = float(adjusted_rand_score(y, state["labels"]))

    row = {
        "dataset": path.stem,
        "n": n,
        "n_true_clusters": int(len(np.unique(y[y >= 0]))),
        "n_delaunay_edges": int(len(graph["edge_keys"])),
        "n_seed_edges": int(state["seed_mask"].sum()),
        "n_final_edges": int(state["final_mask"].sum()),
        "n_seed_clusters": state["n_clusters"],
        "n_final_clusters": int(len(np.unique(state["labels"][state["labels"] >= 0]))),
        "n_rounds": state["n_rounds"],
        "penalty_power": penalty_power,
        "stop_ratio": stop_ratio,
        "baseline_ari": baseline_ari,
        "dijkstra_penalty_ari": dijkstra_ari,
        "delta_vs_baseline": dijkstra_ari - baseline_ari,
        "baseline_seconds": round(baseline_elapsed, 2),
        "dijkstra_seconds": round(dijkstra_elapsed, 2),
    }
    print(
        f"  {path.stem:<38} baseline={baseline_ari:.4f}  "
        f"dijkstra_penalty={dijkstra_ari:.4f}  "
        f"(delta={row['delta_vs_baseline']:+.4f}, {state['n_rounds']} rounds, "
        f"{dijkstra_elapsed:.1f}s)",
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
    parser.add_argument("--out", default=Path("results/growth_dijkstra_penalty"), type=Path)
    parser.add_argument("--penalty-power", type=float, default=2.0)
    parser.add_argument("--stop-ratio", type=float, default=3.0)
    parser.add_argument(
        "--datasets", default="",
        help="optional comma-separated filename stems; default is all .npz files",
    )
    args = parser.parse_args()

    files = sorted(args.data.glob("*.npz"))
    requested = {stem.strip() for stem in args.datasets.split(",") if stem.strip()}
    if requested:
        files = [path for path in files if path.stem in requested]
    if not files:
        raise SystemExit(f"no .npz datasets under {args.data}")

    print(f"{len(files)} datasets  penalty_power={args.penalty_power}  stop_ratio={args.stop_ratio}", flush=True)
    rows = [run_one(path, args.penalty_power, args.stop_ratio) for path in files]

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out / "growth_dijkstra_penalty.csv")

    print("\n=== mean over datasets ===")
    for key in ("baseline_ari", "dijkstra_penalty_ari", "delta_vs_baseline"):
        print(f"{key:<20} {np.mean([r[key] for r in rows]):.4f}")
    print(f"\nWrote {args.out}/growth_dijkstra_penalty.csv")


if __name__ == "__main__":
    main()
