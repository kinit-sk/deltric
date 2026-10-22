"""Independent verification of DIMENSION_CAUSE_AUDIT.md proposals.

Development inputs only; nothing under data/holdout is read. Production code is
imported but never modified: each intervention is a local replacement of one
stage. Expensive geometry (UMAP, Delaunay, kNN) is cached per dataset so every
variant of a dataset shares the exact same edges.

Run: OMP_NUM_THREADS=1 .venv/bin/python studies/verify_audit_fixes.py --datasets ...
"""
from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'studies'))
import utils_component_growth as cg
import plot_seed_component_recomputed_growth as stage
from plot_seed_component_metrics import _seed_components
from dimension_anomaly_metrics import anomaly_metrics

OUT = ROOT / 'studies/results/verify_audit_fixes'
CACHE = OUT / 'cache'


def resolve(name):
    for sub in ('', 'other', 'testing/real_world', 'testing/additional_anomaly'):
        path = ROOT / 'data' / sub / f'{name}.npz'
        if path.exists():
            assert 'holdout' not in path.parts
            return path
    raise FileNotFoundError(name)


def geometry(name):
    """Production launcher geometry (UMAP seed 42 for d>2), cached."""
    cache = CACHE / f'{name}.npz'
    if cache.exists():
        return dict(np.load(cache))
    data = np.load(resolve(name))
    X = StandardScaler().fit_transform(np.asarray(data['X'], dtype=float))
    y = np.asarray(data['y'] if 'y' in data else data['y_clean'], dtype=int)
    state = cg.component_growth_graph(
        X, dim_reduction='umap', umap_n_epochs=100, umap_n_neighbors=15, project_dim=2,
        hard_limit=.9, growth_hard_limit=.9, seed_hard_limit=0, mild_limit=-.5,
        initial_relation='union', growth_relation='union', knn_k=50,
        component_min_edges=len(X)**2, hard_gate_mode='strict',
        projected_hard_limit=True, multiplicative_hard_limit=False)
    edges = state['edge_keys']
    Xp = state['X_proj']
    geo = dict(X=X, y=y, X_proj=Xp, edges=edges, lengths=state['orig_edge_sizes'],
               projected=np.linalg.norm(Xp[edges[:, 0]] - Xp[edges[:, 1]], axis=1),
               initial=state['initial_mask'],
               knn=cg._original_knn_relation(X, edges, 50, 'union'),
               core=stage._knn_core_distance(X, 15))
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **geo)
    return geo


def growth(edges, lengths, projected, initial, seed_labels, seed_counts, core,
           limit=1.05, plimit=1.35, join='either', tau=1.5, min_edges=10):
    """Copy of stage._sequential_growth with optional join criteria.

    join='either'  production: one component's statistics accept the edge.
    join='mutual'  an edge between two active components must pass both.
    join='valley'  an edge between two active components additionally needs
                   both endpoint core distances <= tau * that side's median core.
    join='density' every growth edge needs the outer endpoint's core distance
                   <= tau * the absorbing component's median core distance.
    """
    n_points, n_comp = len(seed_labels), len(seed_counts)
    parent = np.arange(n_comp)
    members = [set(np.flatnonzero(seed_labels == c)) for c in range(n_comp)]
    active = np.asarray(seed_counts > min_edges)
    adjacency = [[] for _ in range(n_points)]
    for i, (u, v) in enumerate(edges):
        adjacency[u].append(i)
        adjacency[v].append(i)

    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return int(c)

    def merge(a, b):
        a, b = find(a), find(b)
        if a == b:
            return
        if len(members[a]) < len(members[b]):
            a, b = b, a
        parent[b] = a
        members[a].update(members[b])
        members[b].clear()
        active[a] = active[a] or active[b]
        active[b] = False

    accepted = np.zeros(len(edges), dtype=bool)
    while True:
        stats = {}
        for root in range(n_comp):
            if find(root) != root or not active[root] or not members[root]:
                continue
            nodes = np.fromiter(members[root], dtype=np.int64)
            inc = np.unique(np.concatenate([adjacency[n] for n in nodes]))
            _, med, q75 = np.percentile(lengths[inc], [25, 50, 75])
            _, pmed, pq75 = np.percentile(projected[inc], [25, 50, 75])
            stats[root] = (inc, med, max(q75 - med, 1e-12), pmed, max(pq75 - pmed, 1e-12),
                           float(np.median(core[nodes])))

        def passes(root, i):
            _, med, spr, pmed, pspr, _ = stats[root]
            return ((lengths[i] - med) / spr < limit
                    and (projected[i] - pmed) / pspr < plimit)

        qualifying = set()
        for root, (inc, *_rest) in stats.items():
            for i in inc:
                u, v = edges[i]
                ru, rv = find(seed_labels[u]), find(seed_labels[v])
                if ru == rv or root not in (ru, rv):
                    continue
                other = rv if ru == root else ru
                outer = v if ru == root else u
                if not passes(root, i):
                    continue
                if join == 'density' and core[outer] > tau * stats[root][5]:
                    continue
                if other in stats:
                    if join == 'mutual' and not passes(other, i):
                        continue
                    if join == 'valley':
                        cu = stats[ru][5]
                        cv = stats[rv][5]
                        if core[u] > tau * cu or core[v] > tau * cv:
                            continue
                qualifying.add(int(i))
        if not qualifying:
            break
        for i in sorted(qualifying):
            u, v = edges[i]
            merge(seed_labels[u], seed_labels[v])
            accepted[i] = True
    return accepted


def prediction(edges, mask, n):
    labels, points = stage._final_graph_components(edges, mask, n)
    return stage._final_component_prediction(edges, mask, labels, points, 10)


def run_variant(geo, v, stages=None):
    """Return (before, final) labels; optionally fill `stages` with intermediate labels."""
    X, y, edges, lengths, projected = geo['X'], geo['y'], geo['edges'], geo['lengths'], geo['projected']
    n = len(X)
    u, w = edges.T
    initial = geo['initial'].copy()
    if v.get('seed') == 'local':
        # Scale-free seed length: divide by the endpoints' kNN core distances.
        core = geo['core']
        norm = lengths / np.sqrt(np.maximum(core[u] * core[w], 1e-24))
        # Replace the original-space global median gate; keep the production
        # projected gate only when a distinct projection exists (d > 2).
        initial = geo['knn'] & (norm <= np.median(norm))
        if X.shape[1] > 2:
            initial &= projected <= np.median(projected)
    seed_labels, seed_counts, _ = _seed_components(edges, initial, n, 10)
    acc = growth(edges, lengths, projected, initial, seed_labels, seed_counts, geo['core'],
                 join=v.get('join', 'either'), tau=v.get('tau', 1.5))
    pre = initial | acc
    labels, points = stage._final_graph_components(edges, pre, n)
    restore = points[u] & points[w] & (labels[u] == labels[w])
    base = pre if v.get('restore', True) is False else pre | restore
    # Production growth_added_point_mask: not in an initially large seed, but in
    # a grown component that contains one.
    large_point = np.isin(seed_labels, np.flatnonzero(seed_counts > 10))
    active_comp = np.zeros(labels.max() + 1, dtype=bool)
    active_comp[labels[large_point]] = True
    added = ~large_point & active_comp[labels]
    removable = acc | ((base & ~pre) & (added[u] | added[w]))
    cut, _ = stage._gomory_hu_contracted_candidate_cut_mask(
        edges, removable, seed_labels, cut_size=v.get('cut', 6),
        min_component_points=v.get('min_side', 10))
    retained = base & ~cut
    before = prediction(edges, retained, n)
    if stages is not None:
        stages['seeds'] = prediction(edges, initial, n)
        stages['growth'] = prediction(edges, base, n)
        stages['cut'] = before
    reach = v.get('reach', 10.)
    if reach is None:
        final = before
    else:
        final, _ = stage._assign_anomalies_cost_gate(
            before, edges, lengths, n, penalty_power=.5, reach=reach,
            core_distance=geo['core'], density_power=.5, density_clip=False)
    return before, final


def score(y, pred):
    keep = y >= 0
    k = len(set(y[keep].tolist()))
    m = anomaly_metrics(y, pred)
    return dict(ari_clean=float(adjusted_rand_score(y[keep], pred[keep])) if k > 1 else None,
                ari_all=float(adjusted_rand_score(y, pred)),
                clusters=int(len(set(pred.tolist()) - {-1})),
                f1=m['anomaly_f1'], precision=m['anomaly_precision'],
                recall=m['anomaly_recall'], fpr=m['false_positive_rate'])


VARIANTS = {
    'baseline': {},
    'no_restore': dict(restore=False),
    'no_restore_cut2': dict(restore=False, cut=2),
    'no_restore_side50': dict(restore=False, min_side=50),
    'no_restore_cut2_side50': dict(restore=False, cut=2, min_side=50),
    'join_mutual': dict(join='mutual'),
    'join_valley': dict(join='valley'),
    'join_density': dict(join='density'),
    'join_density2': dict(join='density', tau=2.0),
    'seed_local': dict(seed='local'),
    'reach5': dict(reach=5.),
    'reach2': dict(reach=2.),
    'no_reassign': dict(reach=None),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', nargs='+', required=True)
    p.add_argument('--variants', nargs='+', default=list(VARIANTS))
    p.add_argument('--combo', action='append', default=[],
                   help='name=json, e.g. dens_nr=\'{"join":"density","restore":false}\'')
    args = p.parse_args()
    variants = dict(VARIANTS)
    for c in args.combo:
        name, spec = c.split('=', 1)
        variants[name] = json.loads(spec)
        args.variants.append(name)
    OUT.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        t = time.perf_counter()
        geo = geometry(name)
        tg = time.perf_counter() - t
        for vn in args.variants:
            path = OUT / f'{name}__{vn}.json'
            if path.exists():
                continue
            t = time.perf_counter()
            before, final = run_variant(geo, variants[vn])
            row = dict(dataset=name, variant=vn, spec=variants[vn], n=len(geo['X']),
                       d=int(geo['X'].shape[1]), before=score(geo['y'], before),
                       final=score(geo['y'], final), seconds=time.perf_counter() - t,
                       geometry_seconds=tg)
            path.write_text(json.dumps(row, indent=1))
            f = row['final']
            print(f"{name:45s} {vn:24s} ARI={f['ari_clean']} F1={f['f1']} k={f['clusters']} "
                  f"({row['seconds']:.1f}s)", flush=True)


if __name__ == '__main__':
    main()
