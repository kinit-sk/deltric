#!/usr/bin/env python3
"""Does gating the anomaly-reclamation stage on *accumulated path cost* fix it?

SELF-CONTAINED. This script does not import from, or modify, any other study
module or the production pipeline -- every gate variant is reimplemented here
and replayed against a frozen graph. ``--validate`` proves the reimplementation
of the current production stage is bit-exact against
``_assign_anomalies_dijkstra``, which is what licenses the comparison.

--------------------------------------------------------------------------
THE DIAGNOSIS
--------------------------------------------------------------------------
The production stage (``utils_component_growth._assign_anomalies_dijkstra``)
computes a penalized edge weight

    weight = length * (length / avg_c) ** penalty_power

accumulates it into ``dist``, and pushes ``(dist, node, edge)`` onto a
per-cluster heap. But the only thing that ever *stops* a cluster is

    if length > stop_ratio * avg[root]:   # raw, unpenalized, single edge

``dist`` is never compared against any limit -- it only orders the heap.
Consequences, all measured over 100 anomaly-labelled datasets:

  1. ``penalty_power`` is inert. Sweeping p in {0,1,2,3,4,8} leaves the number
     of reclaimed points *identical* (e.g. 38/331, 60/240, 83/326, 46/46 on
     four datasets) and the labels identical on 4 of 5. Reason: for a fixed
     cluster, length**(1+p)/avg**p is a strictly monotone transform of length,
     so the pop order never changes; and the gate reads raw length, so the
     admissible edge set never changes either.
  2. The cost comparison only fires when two clusters name the same point in
     the same round -- measured at 0 occurrences on 4 of 5 datasets. Ownership
     is therefore decided by hop count, not cost: the stage is a hop-paced
     flood fill, and matches a naive 3-hop greedy flood bit-for-bit on 6 of 8
     datasets.
  3. Reach is unbounded. With no cumulative limit, the only bound is
     ``max_rounds = 4 * n_points + 16``. One wrongly-claimed anomaly becomes a
     launchpad for the next.

Net effect of the stage as shipped: anomaly F1 0.696 -> 0.466 (-0.230, 18
datasets improved / 77 regressed) in exchange for ARI_all 0.604 -> 0.594.

--------------------------------------------------------------------------
THE FIX (what this script tests)
--------------------------------------------------------------------------
Measure cost in "typical hop" units and budget the whole path:

    r    = length / avg_c        # 1.0 == an edge of typical length for c
    cost = r ** (1 + p)          # this hop's price, in typical hops
    claim v iff  sum(cost along path) <= reach

A typical edge costs 1; an edge twice as long costs 2**(1+p) (= 16 at p=3), so
it burns 16 hops of budget. ``penalty_power`` becomes the exchange rate between
"many short hops" and "one long hop" -- the original design intent -- and reach
is finally bounded by cost rather than by round count.

Second change: a single global heap instead of one-claim-per-cluster-per-round,
so the genuinely cheapest cluster wins each point (true multi-source Dijkstra)
rather than whichever cluster is fewest hops away.

Density plugs into the same budget, applied to the length *before* the penalty:

    r = (length * (core[v] / dref_c) ** q) / avg_c

where ``core`` is the original-space kNN core distance and ``dref_c`` the median
core distance inside c. A point in an original-space void gets its edge
inflated, the penalty amplifies that, and it burns through the budget fast.

--------------------------------------------------------------------------
RESULTS (100 datasets, single UMAP seed -- see CAVEATS)
--------------------------------------------------------------------------
penalty_power becomes live, shown at matched claim budgets (F1 / mean claims):

    reach     p=0          p=1          p=2          p=3
    1      0.574/56     0.571/64     0.569/73     0.568/81
    3      0.516/118    0.532/112    0.543/117    0.550/126
    10     0.390/208    0.498/194    0.522/199    0.530/203

+0.140 F1 at reach=10 while reclaiming the same number of points -- p is not
being more conservative, it is choosing better points. Invisible at reach=1
(no reach to govern) and growing with reach, exactly as the mechanism predicts.

Headline comparison:

    variant           P      R     F1     dF1     w/l    ariA   dariA  claims
    noreassign      0.767  0.745  0.696  +0.000    0/0   0.604  +0.000     0
    edge_p2_s3      0.853  0.387  0.466  -0.230  18/77   0.594  -0.010   247   <- shipped
    cost_p3_r25     0.768  0.515  0.550  -0.147  11/63   0.596  -0.008   126
    cd_q4_r25       0.876  0.710  0.752  +0.056  32/12   0.635  +0.031   178   <- best

The cost gate alone is a large improvement over what ships (0.466 -> 0.574) but
still loses to not running the stage. Only cost gate *plus* density beats
``noreassign`` on both metrics -- and unlike the density term under the old edge
gate (which "worked" by reclaiming nothing, coming out bit-identical to
noreassign on several datasets), this one reclaims ~178 points per dataset
while *raising* precision to 0.876.

By dimensionality the pattern inverts -- the shipped stage does its damage at
d>=5, the fix does its work there:

    d      n   noreassign  edge_p2_s3  cd_q4_r25
    2     25      0.823       0.827      0.823    (no-op)
    5     24      0.774       0.429      0.774    (damage exactly repaired)
    20     1      0.233       0.027      0.629
    50    25      0.635       0.303      0.727
    100   24      0.571       0.334      0.689

Control -- the graph earns its place. A pure per-point density threshold (claim
v iff core[v]/dref_c <= T, nearest cluster by one graph hop, no propagation, no
penalty) tops out at F1 0.713 / ariA 0.614 with 41 claims, versus 0.752 / 0.635
with 178 claims for the full variant, at higher precision. The Dijkstra
propagation is contributing, not decorating.

--------------------------------------------------------------------------
CAVEATS (do not skip these when writing this up)
--------------------------------------------------------------------------
  * q=4 / reach=25 was selected on the same 100 datasets the +0.056 is reported
    on. That is benchmark overfitting; treat +0.056 as an optimistic bound
    until confirmed on a held-out split. ``--holdout`` does that split.
  * Single UMAP seed, no error bars -- inherited from the existing studies.
    Deltas below ~0.04 F1 are not distinguishable from seed variance.
  * At q=4 with p=3 the density ratio carries effective exponent 16. F1 is flat
    across reach 25->150, i.e. density does most of the gating and reach has
    saturated. Measures well, looks fragile.
  * 56 of 100 datasets are unchanged by every variant (mostly d=2), so the
    32/12 win record is really over ~44 live datasets.
  * Only the 100 anomaly-labelled datasets are covered. The ARI-only study
    datasets are untested here.

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
    # one-time: run cluster_tri per dataset and cache the frozen graphs (~6 min)
    python studies/anomaly_cost_gate_study.py --build-cache

    # prove the reimplementation matches production, then sweep
    python studies/anomaly_cost_gate_study.py --validate
    python studies/anomaly_cost_gate_study.py --sweep headline
    python studies/anomaly_cost_gate_study.py --sweep penalty   # matched-budget p
    python studies/anomaly_cost_gate_study.py --sweep density
    python studies/anomaly_cost_gate_study.py --sweep control
    python studies/anomaly_cost_gate_study.py --sweep all --holdout

    # custom grid
    python studies/anomaly_cost_gate_study.py --sweep grid \
        --powers 2 3 4 --reaches 10 25 60 --qs 0 2 4
"""

from __future__ import annotations

import argparse
import csv
import heapq
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score, f1_score, precision_score, recall_score,
)
from sklearn.preprocessing import StandardScaler

_MINIMAL_DIR = Path(__file__).resolve().parents[1]
_DEFAULT_DATA = _MINIMAL_DIR / "data"
_DEFAULT_CACHE = _MINIMAL_DIR / "studies" / "results" / "anomaly_cost_gate" / "cache"
_DEFAULT_OUT = _MINIMAL_DIR / "studies" / "results" / "anomaly_cost_gate"
sys.path.insert(0, str(_MINIMAL_DIR.parent))

# Inlined rather than imported so this file has no dependency on the other
# study modules (they are being edited independently). Kept identical to the
# config used by eval_edge_rule_ari.py / anomaly_f1_dijkstra_vs_baseline.py --
# if those drift, this comment is the place to reconcile.
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

DENSITY_K = 15
KIND_NAMES = {1: "uniform", 2: "far", 3: "bridge", 4: "micro", 5: "subspace"}


# ---------------------------------------------------------------------------
# frozen-graph cache
# ---------------------------------------------------------------------------

def eligible_datasets(data_dir: Path) -> list[Path]:
    """Every dataset carrying ground-truth noise (``y == -1``).

    That is the ``__anom*`` files produced by make_anomaly_datasets.py plus the
    two families that shipped with noise labels. Selecting by content rather
    than by filename so newly generated datasets are picked up automatically.
    """
    out = []
    for path in sorted(data_dir.glob("*.npz")):
        try:
            with np.load(path, allow_pickle=False) as npz:
                if "y" in npz.files and np.any(np.asarray(npz["y"]) < 0):
                    out.append(path)
        except Exception:
            continue
    return out


def build_cache(data_dir: Path, cache_dir: Path, force: bool = False) -> None:
    """Run cluster_tri once per dataset with reassignment OFF and cache the graph.

    Every variant then replays against byte-identical UMAP output, so all
    differences are attributable to the gate and not to projection variance.
    """
    from deltric_minimal.utils_component_growth import cluster_tri, _knn_core_distance

    cache_dir.mkdir(parents=True, exist_ok=True)
    files = eligible_datasets(data_dir)
    print(f"{len(files)} datasets with ground-truth anomalies", flush=True)
    started = time.perf_counter()
    for index, path in enumerate(files, 1):
        out = cache_dir / f"{path.stem}.npz"
        if out.exists() and not force:
            continue
        npz = np.load(path, allow_pickle=False)
        X = StandardScaler().fit_transform(np.asarray(npz["X"], dtype=np.float64))
        y = np.asarray(npz["y"]).astype(np.int64)
        kind = (np.asarray(npz["anomaly_kind"]) if "anomaly_kind" in npz.files
                else np.zeros(len(y), dtype=np.int8))
        cluster_tri(X, **BASE_CONFIG, component_growth_anomaly_reassign=False)
        state = cluster_tri.last_component_growth
        np.savez_compressed(
            out, y=y, kind=kind, n=len(X), d=X.shape[1],
            merged_labels=state["merged_labels"],
            edge_keys=state["edge_keys"],
            proj_sizes=state["projected_edge_sizes"],
            core=_knn_core_distance(X, DENSITY_K),
        )
        print(f"  [{index}/{len(files)}] {path.stem}  "
              f"{time.perf_counter() - started:.0f}s", flush=True)
    print(f"cache ready in {cache_dir} ({time.perf_counter() - started:.0f}s)", flush=True)


def load_cache(cache_dir: Path) -> list[dict]:
    entries = []
    for path in sorted(cache_dir.glob("*.npz")):
        npz = np.load(path, allow_pickle=False)
        entries.append({
            "dataset": path.stem,
            "y": npz["y"], "kind": npz["kind"],
            "n": int(npz["n"]), "d": int(npz["d"]),
            "labels": npz["merged_labels"],
            "edge_keys": npz["edge_keys"],
            "sizes": npz["proj_sizes"],
            "core": npz["core"],
        })
    if not entries:
        raise SystemExit("empty cache -- run with --build-cache first")
    return entries


# ---------------------------------------------------------------------------
# gate variants
# ---------------------------------------------------------------------------

def _prep(labels, edge_keys, sizes, n_points):
    """Shared setup: adjacency, and the frozen per-cluster mean edge length."""
    labels = np.asarray(labels, dtype=np.int64)
    edge_keys = np.asarray(edge_keys, dtype=np.int64)
    sizes = np.asarray(sizes, dtype=np.float64)
    n_edges = len(edge_keys)
    n_clusters = int(labels.max()) + 1 if np.any(labels >= 0) else 0
    adjacency = [[] for _ in range(n_points)]
    for index in range(n_edges):
        u, v = int(edge_keys[index, 0]), int(edge_keys[index, 1])
        adjacency[u].append((v, index))
        adjacency[v].append((u, index))
    total = np.zeros(max(n_clusters, 1))
    count = np.zeros(max(n_clusters, 1), dtype=np.int64)
    for index in range(n_edges):
        u, v = edge_keys[index]
        if labels[u] >= 0 and labels[u] == labels[v]:
            total[labels[u]] += sizes[index]
            count[labels[u]] += 1
    fallback = float(np.median(sizes)) if n_edges else 1.0
    avg = np.maximum(np.where(count > 0, total / np.maximum(count, 1), fallback), 1e-12)
    return labels, edge_keys, sizes, n_edges, n_clusters, adjacency, avg


def _density_ratio_fn(labels, n_clusters, core, q, clip):
    """rho(v, c) ** q, or a no-op when q == 0."""
    if q == 0.0 or core is None:
        return lambda node, cluster: 1.0
    core = np.asarray(core, dtype=np.float64)
    fallback = float(np.median(core)) if len(core) else 1.0
    dref = np.full(max(n_clusters, 1), fallback)
    for cluster in range(n_clusters):
        members = core[labels == cluster]
        if len(members):
            dref[cluster] = float(np.median(members))
    dref = np.maximum(dref, 1e-12)

    def ratio(node, cluster):
        value = core[node] / dref[cluster]
        if clip and value < 1.0:
            value = 1.0
        return value ** q
    return ratio


def edge_gate(entry, p=2.0, stop_ratio=3.0, **_):
    """CURRENT PRODUCTION BEHAVIOUR, reimplemented (see --validate).

    Per-edge gate on raw length, per-cluster heaps, one claim per cluster per
    round. Reported as the baseline everything else is measured against.
    """
    labels, edge_keys, sizes, n_edges, n_clusters, adjacency, avg = _prep(
        entry["labels"], entry["edge_keys"], entry["sizes"], entry["n"])
    if n_clusters == 0:
        return labels.copy(), {"claims": 0, "contested": 0}
    n_points = entry["n"]
    owner = labels.copy()
    heaps = [[] for _ in range(n_clusters)]
    for cluster in range(n_clusters):
        for node in np.flatnonzero(labels == cluster):
            for neighbor, index in adjacency[int(node)]:
                if labels[neighbor] == cluster:
                    continue
                length = sizes[index]
                heapq.heappush(
                    heaps[cluster],
                    (length * (length / avg[cluster]) ** p, neighbor, index))
    active = {c for c in range(n_clusters) if heaps[c]}
    claims = contested = rounds = 0
    while rounds < 4 * n_points + 16 and active:
        candidate, dead = {}, []
        for root in list(active):
            heap = heaps[root]
            while heap:
                dist, node, index = heap[0]
                if owner[node] != -1:
                    heapq.heappop(heap)
                    continue
                if sizes[index] > stop_ratio * avg[root]:
                    heapq.heappop(heap)
                    continue
                candidate[root] = (dist, node, index)
                break
            if not heap:
                dead.append(root)
        for root in dead:
            active.discard(root)
        if not candidate:
            break
        tally = Counter(node for _, (_, node, _) in candidate.items())
        contested += sum(1 for _, k in tally.items() if k > 1)
        proposals = {}
        for root, (dist, node, index) in candidate.items():
            current = proposals.get(node)
            if current is None or dist < current[0]:
                proposals[node] = (dist, root, index)
        for node, (dist, root, index) in proposals.items():
            if owner[node] != -1:
                continue
            heapq.heappop(heaps[root])
            owner[node] = root
            claims += 1
            for neighbor, next_index in adjacency[node]:
                if owner[neighbor] != -1:
                    continue
                length = sizes[next_index]
                heapq.heappush(
                    heaps[root],
                    (dist + length * (length / avg[root]) ** p, neighbor, next_index))
        rounds += 1
    return owner, {"claims": claims, "contested": contested, "rounds": rounds}


def cost_gate(entry, p=3.0, reach=25.0, q=0.0, clip=False, edge_cap=None, **_):
    """THE FIX. Cumulative-cost budget in typical-hop units, single global heap.

        r    = (length * rho(v, c) ** q) / avg_c
        cost = r ** (1 + p)
        claim v iff sum(cost along path) <= reach

    q == 0 gives the cost gate alone; q > 0 adds the original-space density
    term. ``edge_cap`` optionally retains the old per-edge gate on top, for
    ablation.
    """
    labels, edge_keys, sizes, n_edges, n_clusters, adjacency, avg = _prep(
        entry["labels"], entry["edge_keys"], entry["sizes"], entry["n"])
    if n_clusters == 0:
        return labels.copy(), {"claims": 0, "contested": -1}
    ratio = _density_ratio_fn(labels, n_clusters, entry.get("core"), q, clip)

    def relative(index, node, cluster):
        return (sizes[index] * ratio(node, cluster)) / avg[cluster]

    owner = labels.copy()
    heap = []
    for cluster in range(n_clusters):
        for node in np.flatnonzero(labels == cluster):
            for neighbor, index in adjacency[int(node)]:
                if labels[neighbor] == cluster:
                    continue
                r = relative(index, int(neighbor), cluster)
                if edge_cap is not None and sizes[index] > edge_cap * avg[cluster]:
                    continue
                cost = r ** (1.0 + p)
                if cost <= reach:
                    heapq.heappush(heap, (cost, int(neighbor), cluster))
    claims = 0
    while heap:
        dist, node, cluster = heapq.heappop(heap)
        if owner[node] != -1:
            continue
        owner[node] = cluster
        claims += 1
        for neighbor, index in adjacency[node]:
            if owner[neighbor] != -1:
                continue
            if edge_cap is not None and sizes[index] > edge_cap * avg[cluster]:
                continue
            nxt = dist + relative(index, int(neighbor), cluster) ** (1.0 + p)
            if nxt <= reach:
                heapq.heappush(heap, (nxt, int(neighbor), cluster))
    return owner, {"claims": claims, "contested": -1}


def hop_gate(entry, p=2.0, stop_ratio=3.0, max_hops=1, **_):
    """Current per-edge gate but reach capped at ``max_hops`` graph hops.

    Isolates how much of the shipped stage's damage is propagation: on 6 of 8
    spot-checked datasets ``max_hops=3`` reproduces the full stage exactly.
    """
    labels, edge_keys, sizes, n_edges, n_clusters, adjacency, avg = _prep(
        entry["labels"], entry["edge_keys"], entry["sizes"], entry["n"])
    if n_clusters == 0:
        return labels.copy(), {"claims": 0, "contested": -1}
    owner = labels.copy()
    claims = 0
    for _hop in range(max_hops):
        best = {}
        for node in range(entry["n"]):
            if owner[node] != -1:
                continue
            for neighbor, index in adjacency[node]:
                cluster = owner[neighbor]
                if cluster < 0:
                    continue
                length = sizes[index]
                if length > stop_ratio * avg[cluster]:
                    continue
                weight = length * (length / avg[cluster]) ** p
                if node not in best or weight < best[node][0]:
                    best[node] = (weight, cluster)
        if not best:
            break
        for node, (_w, cluster) in best.items():
            owner[node] = cluster
            claims += 1
    return owner, {"claims": claims, "contested": -1}


def density_only(entry, threshold=1.25, **_):
    """CONTROL: no propagation, no penalty, no cost -- just a density test.

    Claim v iff ``core[v] / dref_c <= threshold``, where c owns v's nearest
    labelled graph neighbour. If this matches the full variant, the Dijkstra
    machinery is not earning its place.
    """
    labels, edge_keys, sizes, n_edges, n_clusters, adjacency, avg = _prep(
        entry["labels"], entry["edge_keys"], entry["sizes"], entry["n"])
    if n_clusters == 0:
        return labels.copy(), {"claims": 0, "contested": -1}
    core = np.asarray(entry["core"], dtype=np.float64)
    fallback = float(np.median(core)) if len(core) else 1.0
    dref = np.full(max(n_clusters, 1), fallback)
    for cluster in range(n_clusters):
        members = core[labels == cluster]
        if len(members):
            dref[cluster] = float(np.median(members))
    dref = np.maximum(dref, 1e-12)
    owner = labels.copy()
    claims = 0
    for node in range(entry["n"]):
        if owner[node] != -1:
            continue
        best = None
        for neighbor, index in adjacency[node]:
            if labels[neighbor] < 0:
                continue
            if best is None or sizes[index] < best[0]:
                best = (sizes[index], int(labels[neighbor]))
        if best is None:
            continue
        cluster = best[1]
        if core[node] / dref[cluster] <= threshold:
            owner[node] = cluster
            claims += 1
    return owner, {"claims": claims, "contested": -1}


def no_reassign(entry, **_):
    return np.asarray(entry["labels"]).copy(), {"claims": 0, "contested": 0}


# ---------------------------------------------------------------------------
# sweeps
# ---------------------------------------------------------------------------

def build_variants(name: str, args) -> list[tuple[str, object, dict]]:
    """(label, fn, kwargs) triples. 'noreassign' must stay first -- deltas key off it."""
    variants = [("noreassign", no_reassign, {})]
    if name in ("headline", "all"):
        variants += [
            ("edge_p2_s3", edge_gate, dict(p=2.0, stop_ratio=3.0)),   # shipped
            ("hop1", hop_gate, dict(p=2.0, stop_ratio=3.0, max_hops=1)),
            ("hop3", hop_gate, dict(p=2.0, stop_ratio=3.0, max_hops=3)),
            ("cost_p3_r25", cost_gate, dict(p=3.0, reach=25.0, q=0.0)),
            ("cd_q2_r25", cost_gate, dict(p=3.0, reach=25.0, q=2.0)),
            ("cd_q4_r25", cost_gate, dict(p=3.0, reach=25.0, q=4.0)),
            ("cd_q4_r60", cost_gate, dict(p=3.0, reach=60.0, q=4.0)),
        ]
    if name in ("penalty", "all"):
        variants += [("edge_p2_s3", edge_gate, dict(p=2.0, stop_ratio=3.0))]
        for p in (0.0, 1.0, 2.0, 3.0):
            for reach in (1.0, 3.0, 10.0):
                variants.append((f"cost_p{p:g}_r{reach:g}",
                                 cost_gate, dict(p=p, reach=reach, q=0.0)))
    if name in ("density", "all"):
        for q in (0.0, 1.0, 2.0, 4.0):
            for reach in (10.0, 25.0, 60.0):
                variants.append((f"cd_q{q:g}_r{reach:g}",
                                 cost_gate, dict(p=3.0, reach=reach, q=q)))
    if name in ("control", "all"):
        for threshold in (1.0, 1.25, 1.5, 2.0, 3.0):
            variants.append((f"dens_only_T{threshold:g}",
                             density_only, dict(threshold=threshold)))
        variants.append(("cd_q4_r25", cost_gate, dict(p=3.0, reach=25.0, q=4.0)))
    if name == "grid":
        variants += [("edge_p2_s3", edge_gate, dict(p=2.0, stop_ratio=3.0))]
        for p in args.powers:
            for reach in args.reaches:
                for q in args.qs:
                    variants.append((f"g_p{p:g}_r{reach:g}_q{q:g}", cost_gate,
                                     dict(p=p, reach=reach, q=q, clip=args.clip)))
    seen, unique = set(), []
    for label, fn, kwargs in variants:
        if label in seen:
            continue
        seen.add(label)
        unique.append((label, fn, kwargs))
    return unique


def evaluate(entries, variants) -> list[dict]:
    rows = []
    for index, entry in enumerate(entries, 1):
        y = entry["y"]
        true_anomaly = y < 0
        real = ~true_anomaly
        kind = entry["kind"]
        n_noise_in = int((entry["labels"] < 0).sum())
        row = {"dataset": entry["dataset"], "n": entry["n"], "d": entry["d"],
               "n_true_anomaly": int(true_anomaly.sum()), "n_noise_in": n_noise_in}
        for label, fn, kwargs in variants:
            assigned, diag = fn(entry, **kwargs)
            predicted = assigned < 0
            row[f"{label}_precision"] = precision_score(true_anomaly, predicted, zero_division=0)
            row[f"{label}_recall"] = recall_score(true_anomaly, predicted, zero_division=0)
            row[f"{label}_f1"] = f1_score(true_anomaly, predicted, zero_division=0)
            # ari_real excludes the true anomalies and is therefore blind to the
            # main failure mode of this stage; ari_all keeps -1 as its own label
            # and is the metric that can actually see a wrongly-reclaimed point.
            row[f"{label}_ari_real"] = float(adjusted_rand_score(y[real], assigned[real]))
            row[f"{label}_ari_all"] = float(adjusted_rand_score(y, assigned))
            row[f"{label}_claims"] = int(n_noise_in - (assigned < 0).sum())
            row[f"{label}_contested"] = diag.get("contested", -1)
            for code, kind_name in KIND_NAMES.items():
                mask = kind == code
                if mask.any():
                    row[f"{label}_recall_{kind_name}"] = float((assigned[mask] < 0).mean())
        rows.append(row)
        print(f"  [{index}/{len(entries)}] {entry['dataset']}", flush=True)
    return rows


def mean_of(rows, key):
    values = [r[key] for r in rows if key in r and not np.isnan(r[key])]
    return float(np.mean(values)) if values else float("nan")


def report(rows, variants, title="") -> None:
    if title:
        print(f"\n=== {title} (n={len(rows)}) ===")
    header = (f"{'variant':<16}{'P':>7}{'R':>7}{'F1':>7}{'dF1':>8}{'w/l':>9}"
              f"{'ariR':>7}{'ariA':>7}{'dariA':>8}{'claims':>8}")
    print(header)
    for label, _fn, _kw in variants:
        d_f1 = [r[f"{label}_f1"] - r["noreassign_f1"] for r in rows]
        d_ari = [r[f"{label}_ari_all"] - r["noreassign_ari_all"] for r in rows]
        win_loss = f"{sum(x > 1e-9 for x in d_f1)}/{sum(x < -1e-9 for x in d_f1)}"
        print(f"{label:<16}{mean_of(rows, label+'_precision'):>7.3f}"
              f"{mean_of(rows, label+'_recall'):>7.3f}{mean_of(rows, label+'_f1'):>7.3f}"
              f"{np.mean(d_f1):>+8.3f}{win_loss:>9}"
              f"{mean_of(rows, label+'_ari_real'):>7.3f}"
              f"{mean_of(rows, label+'_ari_all'):>7.3f}{np.mean(d_ari):>+8.3f}"
              f"{mean_of(rows, label+'_claims'):>8.1f}")


def report_by_dim(rows, variants) -> None:
    print("\n=== F1 by dimensionality ===")
    labels = [v[0] for v in variants]
    print(f"{'d':<6}{'n':>4}" + "".join(f"{l:>14}" for l in labels))
    for d in sorted({r["d"] for r in rows}):
        subset = [r for r in rows if r["d"] == d]
        print(f"{d:<6}{len(subset):>4}"
              + "".join(f"{mean_of(subset, l+'_f1'):>14.3f}" for l in labels))


def report_by_kind(rows, variants) -> None:
    names = [n for n in KIND_NAMES.values()
             if any(f"noreassign_recall_{n}" in r for r in rows)]
    if not names:
        return
    print("\n=== recall by anomaly kind (higher = anomaly correctly left as noise) ===")
    print(f"{'variant':<16}" + "".join(f"{n:>10}" for n in names))
    for label, _fn, _kw in variants:
        print(f"{label:<16}"
              + "".join(f"{mean_of(rows, f'{label}_recall_{n}'):>10.3f}" for n in names))


def report_matched_budget(rows, variants) -> None:
    """The table that shows penalty_power is live: F1 at equal claim counts."""
    cost = {}
    for label, _fn, kwargs in variants:
        if label.startswith("cost_p") and kwargs.get("q", 0.0) == 0.0:
            cost[(kwargs["p"], kwargs["reach"])] = label
    if not cost:
        return
    powers = sorted({p for p, _ in cost})
    reaches = sorted({r for _, r in cost})
    print("\n=== penalty_power at matched claim budget (F1 / mean claims) ===")
    print(f"{'reach':<8}" + "".join(f"p={p:<11g}" for p in powers))
    for reach in reaches:
        cells = []
        for p in powers:
            label = cost.get((p, reach))
            cells.append("-".ljust(13) if label is None else
                         f"{mean_of(rows, label+'_f1'):.3f}/{mean_of(rows, label+'_claims'):.0f}".ljust(13))
        print(f"{reach:<8}" + "".join(cells))
    print("  Equal claim counts across a row => p is choosing better points,")
    print("  not simply being more conservative. Flat row => p is inert.")


def write_csv(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate(entries) -> int:
    """Prove edge_gate reproduces production exactly, else nothing here is comparable."""
    from deltric_minimal.utils_component_growth import _assign_anomalies_dijkstra
    mismatches = 0
    for entry in entries:
        reference, _ = _assign_anomalies_dijkstra(
            entry["labels"], entry["edge_keys"], entry["sizes"], entry["n"],
            penalty_power=2.0, stop_ratio=3.0)
        mine, _ = edge_gate(entry, p=2.0, stop_ratio=3.0)
        ok = np.array_equal(reference, mine)
        mismatches += not ok
        if not ok:
            print(f"  MISMATCH {entry['dataset']}  "
                  f"({int((reference != mine).sum())} points differ)")
    print(f"validate: {len(entries) - mismatches}/{len(entries)} exact, "
          f"{mismatches} mismatches")
    return mismatches


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=_DEFAULT_DATA)
    parser.add_argument("--cache", type=Path, default=_DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--build-cache", action="store_true")
    parser.add_argument("--force", action="store_true", help="rebuild cached graphs")
    parser.add_argument("--validate", action="store_true",
                        help="check edge_gate is bit-exact against production")
    parser.add_argument("--sweep", default=None,
                        choices=["headline", "penalty", "density", "control", "grid", "all"])
    parser.add_argument("--datasets", default="",
                        help="comma-separated stems; default is the whole cache")
    parser.add_argument("--holdout", action="store_true",
                        help="split datasets alternately and report both halves, so "
                             "params picked on one can be checked on the other")
    parser.add_argument("--powers", nargs="*", type=float, default=[2.0, 3.0, 4.0])
    parser.add_argument("--reaches", nargs="*", type=float, default=[10.0, 25.0, 60.0])
    parser.add_argument("--qs", nargs="*", type=float, default=[0.0, 2.0, 4.0])
    parser.add_argument("--clip", action="store_true",
                        help="one-sided density (rho < 1 clamped to 1)")
    args = parser.parse_args()

    if args.build_cache:
        build_cache(args.data, args.cache, force=args.force)
        if not (args.validate or args.sweep):
            return

    entries = load_cache(args.cache)
    requested = {s.strip() for s in args.datasets.split(",") if s.strip()}
    if requested:
        entries = [e for e in entries if e["dataset"] in requested]
        if not entries:
            raise SystemExit(f"no cached datasets matched {sorted(requested)}")

    if args.validate:
        if validate(entries):
            raise SystemExit("reimplementation drifted from production -- fix before trusting a sweep")
        if not args.sweep:
            return

    if not args.sweep:
        parser.print_help()
        return

    variants = build_variants(args.sweep, args)
    print(f"{len(entries)} datasets  {len(variants)} variants  sweep={args.sweep}", flush=True)
    rows = evaluate(entries, variants)

    out_csv = args.out / f"anomaly_cost_gate_{args.sweep}.csv"
    write_csv(rows, out_csv)

    report(rows, variants, "mean over all datasets")
    report_matched_budget(rows, variants)
    report_by_kind(rows, variants)
    report_by_dim(rows, variants)

    if args.holdout:
        # Alternating split by sorted dataset name: crude, but it keeps the
        # d/family mix roughly balanced and it is deterministic. Params chosen
        # on split A must survive on split B to be believable.
        split_a = [r for i, r in enumerate(rows) if i % 2 == 0]
        split_b = [r for i, r in enumerate(rows) if i % 2 == 1]
        report(split_a, variants, "HOLDOUT split A (tune here)")
        report(split_b, variants, "HOLDOUT split B (confirm here)")

    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
