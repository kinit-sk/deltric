"""Reproducible causal audit of the current launcher, using production functions.

No holdout inputs. Oracle variants use labels deliberately for diagnosis only.
Run with OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import sys
import time

import numpy as np
from scipy.spatial import Delaunay
from sklearn.metrics import adjusted_rand_score, pair_confusion_matrix
from sklearn.preprocessing import StandardScaler
from umap import UMAP

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import utils_component_growth as cg
import plot_seed_component_recomputed_growth as stage
from plot_seed_component_metrics import _seed_components
from dimension_anomaly_metrics import anomaly_metrics

OUT = ROOT / 'studies/results/dimension_cause_audit'
CURATED = sorted(p.stem for p in (ROOT / 'results/prune_stages_component_growth').glob('*.json'))


def score(y, pred):
    keep = y >= 0
    tn, fp, fn, tp = pair_confusion_matrix(y[keep], pred[keep]).ravel()
    return dict(ari=float(adjusted_rand_score(y, pred)),
                ari_clean=float(adjusted_rand_score(y[keep], pred[keep])),
                clusters=int(len(set(pred) - {-1})), noise=int((pred < 0).sum()),
                false_merged_pairs=int(fp // 2), false_split_pairs=int(fn // 2),
                **anomaly_metrics(y, pred))


def prediction(edges, mask, n):
    labels, points = stage._final_graph_components(edges, mask, n)
    return stage._final_component_prediction(edges, mask, labels, points, 10)


def run(path, variant='baseline', umap_seed=42):
    start = time.perf_counter()
    data = np.load(path)
    X = StandardScaler().fit_transform(np.asarray(data['X'], dtype=float))
    y = np.asarray(data['y'] if 'y' in data else data['y_clean'], dtype=int)
    # Zero padding preserves every original distance and merely crosses the
    # production dimension branch. It adds no information or signal.
    if variant == 'zero_pad_umap':
        assert X.shape[1] == 2
        X = np.column_stack([X, np.zeros(len(X))])
    elif variant == 'zero_pad_none':
        assert X.shape[1] == 2
        X = np.column_stack([X, np.zeros(len(X))])
    original_builder = cg.get_triangles_with_edges
    def builder(values, **kwargs):
        if variant == 'zero_pad_none':
            z = values[:, :2]
        elif variant == 'force_umap' or (variant == 'zero_pad_umap' and umap_seed != 42):
            z = UMAP(n_components=2, n_neighbors=15, n_epochs=100,
                     random_state=umap_seed).fit_transform(values)
        else:
            return original_builder(values, **kwargs)
        tri = Delaunay(z).simplices
        return tri, None, None, z
    cg.get_triangles_with_edges = builder
    try:
        state = cg.component_growth_graph(
            X, dim_reduction='pca' if variant == 'pca' else 'umap',
            umap_n_epochs=100, umap_n_neighbors=15, project_dim=2,
            hard_limit=.9, growth_hard_limit=.9, seed_hard_limit=0,
            mild_limit=-.5, initial_relation='union', growth_relation='union',
            knn_k=50, component_min_edges=len(X)**2, hard_gate_mode='strict',
            projected_hard_limit=True, multiplicative_hard_limit=False)
    finally:
        cg.get_triangles_with_edges = original_builder
    edges = state['edge_keys']
    lengths = state['orig_edge_sizes']
    projected = np.linalg.norm(state['X_proj'][edges[:, 0]] - state['X_proj'][edges[:, 1]], axis=1)
    initial = state['initial_mask'].copy()
    u, v = edges.T
    cross = (y[u] != y[v]) & (y[u] >= 0) & (y[v] >= 0)
    # For noisy data also eliminate cluster/noise bridges in this diagnostic.
    oracle_bad = (y[u] != y[v]) | (y[u] < 0) | (y[v] < 0)
    if variant == 'oracle_seed':
        initial &= ~oracle_bad
    seed_labels, seed_counts, large_seed = _seed_components(edges, initial, len(X), 10)
    seed_mixed = []
    forced_pairs = 0
    for label in np.unique(seed_labels):
        counts = np.unique(y[(seed_labels == label) & (y >= 0)], return_counts=True)[1]
        forced_pairs += int((counts.sum()**2 - (counts**2).sum()) // 2)
        if len(counts) > 1:
            seed_mixed.append(dict(seed=int(label), size=int((seed_labels == label).sum()),
                                   truth_counts={str(int(c)): int((y[seed_labels == label] == c).sum())
                                                 for c in np.unique(y[seed_labels == label])}))
    growth = stage._sequential_growth(edges, lengths, projected, initial,
                                     seed_labels, seed_counts, 10, np.ones(len(X)), 1.05, 1.35)
    pre = initial | growth['accepted_growth_mask']
    labels, points = stage._final_graph_components(edges, pre, len(X))
    restore = points[u] & points[v] & (labels[u] == labels[v])
    base = pre | restore
    if variant == 'no_restore':
        base = pre.copy()
    added = growth['growth_added_point_mask']
    removable = growth['accepted_growth_mask'] | ((base & ~pre) & (added[u] | added[v]))
    if variant == 'all_cross_seed_removable':
        removable = base & (seed_labels[u] != seed_labels[v])
    cut, gh = stage._gomory_hu_contracted_candidate_cut_mask(
        edges, removable, seed_labels, cut_size=6, min_component_points=10)
    retained = base & ~cut
    if variant == 'oracle_final_cut':
        retained &= ~oracle_bad
    before = prediction(edges, retained, len(X))
    final, costs = stage._assign_anomalies_cost_gate(
        before, edges, lengths, len(X), penalty_power=.5, reach=10.,
        core_distance=stage._knn_core_distance(X, 15), density_power=.5, density_clip=False)
    union = cg._original_knn_relation(X, edges, 50, 'union')
    seen = growth['first_seen_mask']
    ro = growth['first_seen_panel2_ratio_raw']
    rp = growth['first_seen_projected_ratio_raw']
    row = dict(dataset=path.stem, variant=variant, umap_seed=umap_seed, n=len(X), d=X.shape[1],
               truth_clusters=len(set(y)-{-1}), edges=len(edges),
               seed_original_threshold=float(np.median(lengths)),
               seed_projected_threshold=float(np.median(projected)),
               length_correlation=float(np.corrcoef(lengths, projected)[0, 1]),
               knn_fraction=float(union.mean()), cross_edges=int(cross.sum()),
               cross_knn_fraction=float(union[cross].mean()) if cross.any() else None,
               seed_edges=int(initial.sum()), cross_seed_edges=int((initial & cross).sum()),
               mixed_seed_count=len(seed_mixed), forced_wrong_pairs=forced_pairs,
               largest_mixed_seeds=sorted(seed_mixed, key=lambda x: -x['size'])[:5],
               cross_growth_edges=int((cross & growth['accepted_growth_mask']).sum()),
               growth_edges=int(growth['accepted_growth_mask'].sum()),
               cross_removed_gh=int((cross & cut).sum()),
               cross_retained=int((cross & retained).sum()),
               restored_cross_seed_protected=int(((base & ~pre) & ~removable & (seed_labels[u] != seed_labels[v])).sum()),
               restored_cross_truth_protected=int(((base & ~pre) & ~removable & cross).sum()),
               projected_unique_veto=int((seen & (ro < 1.05) & (rp >= 1.35)).sum()),
               projected_unique_cross_veto=int((seen & cross & (ro < 1.05) & (rp >= 1.35)).sum()),
               stages={name: score(y, pred) for name, pred in (
                   ('seed', prediction(edges, initial, len(X))),
                   ('growth', prediction(edges, pre, len(X))),
                   ('gh', before), ('final', final))},
               gh=gh, seconds=time.perf_counter()-start)
    if variant == 'baseline' and path.parent.name == 'other':
        saved = np.load(ROOT / 'studies/results/clustbench_other' / (path.stem+'.npz'))
        row['saved_prediction_ari'] = float(adjusted_rand_score(saved['final_prediction'], final))
    key = f'{path.stem}__{variant}__s{umap_seed}'
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / (key+'.json')).write_text(json.dumps(row, indent=2)+'\n')
    np.savez_compressed(OUT / (key+'.npz'), X=X, X_proj=state['X_proj'], y=y,
                        edges=edges, lengths=lengths, projected=projected, initial=initial,
                        seed_labels=seed_labels, pre=pre, retained=retained, before=before,
                        final=final, cross=cross, cut=cut, ro=ro, rp=rp,
                        accepted=growth['accepted_growth_mask'])
    print(json.dumps({k: row[k] for k in ('dataset','variant','forced_wrong_pairs','cross_seed_edges','cross_growth_edges','stages','seconds')}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='*', default=CURATED)
    parser.add_argument('--variants', nargs='+', default=['baseline'])
    parser.add_argument('--umap-seeds', nargs='+', type=int, default=[42])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    provenance = dict(python=sys.version, packages={p: importlib.metadata.version(p) for p in
                     ['numpy','scipy','scikit-learn','umap-learn','networkx']},
                     code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                         [Path(__file__), ROOT/'utils_component_growth.py', ROOT/'utils_pruning.py',
                          ROOT/'plot_seed_component_recomputed_growth.py']},
                     argv=sys.argv, inputs={})
    for name in args.datasets:
        path = ROOT/'data'/f'{name}.npz'
        if not path.exists():
            path = ROOT/'data/other'/f'{name}.npz'
        provenance['inputs'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for variant in args.variants:
            for seed in args.umap_seeds:
                run(path, variant, seed)
    stamp = str(time.time_ns())
    (OUT/f'provenance_{stamp}.json').write_text(json.dumps(provenance, indent=2)+'\n')


if __name__ == '__main__':
    main()
