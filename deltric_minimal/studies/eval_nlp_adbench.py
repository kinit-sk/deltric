#!/usr/bin/env python3
"""Evaluate BBC test embeddings using the existing Gaia anomaly study configuration."""
import csv
import json
import sys
import time
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from deltric_minimal.studies.eval_gaia_anomaly import scores
from deltric_minimal.studies.anomaly_cost_gate_study import BASE_CONFIG
from deltric_minimal.utils_component_growth import cluster_tri
from deltric_minimal.hdbscan_comparison import fit_hdbscan


def main():
    source = ROOT / 'data/nlp_adbench'
    out = ROOT / 'studies/results/nlp_adbench_bbc'
    out.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in (source / 'datasets/bbc/bbc_test_data.jsonl').read_text().splitlines()]
    names = sorted({r['original_label'] for r in records if r['label'] == 0})
    mapping = {name: i for i, name in enumerate(names)}
    y = np.array([-1 if r['label'] else mapping[r['original_label']] for r in records])
    rows = []
    for embedding, slug in [('bert', 'bert_base_uncased'), ('openai', 'gpt_text-embedding-3-large')]:
        raw = np.load(source / f'embeddings/bbc/bbc_test_data_{slug}_feature.npy', allow_pickle=False)
        assert len(raw) == len(y) and np.isfinite(raw).all()
        X = StandardScaler().fit_transform(raw.astype(np.float64))
        print(f'Running {embedding}: {X.shape}, true anomalies={(y < 0).sum()}', flush=True)
        start = time.perf_counter()
        labels = cluster_tri(X, **BASE_CONFIG, random_state=42,
                             component_growth_anomaly_reassign=True,
                             component_growth_anomaly_reassign_penalty_power=3.,
                             component_growth_anomaly_reassign_reach=25.,
                             component_growth_anomaly_reassign_density_power=4.)
        seconds = time.perf_counter() - start
        baseline = np.array(cluster_tri.last_component_growth['merged_labels'], copy=True)
        start = time.perf_counter()
        hdb = fit_hdbscan(X, min_cluster_size=10, min_samples=None, cluster_selection_method='eom')
        hdb_seconds = time.perf_counter() - start
        np.savez_compressed(out / f'{embedding}_predictions.npz', y=y, deltric=labels, baseline=baseline, hdbscan=hdb)
        for method, assigned, elapsed in [('deltric_no_reassign', baseline, seconds), ('deltric_cost_density', labels, seconds), ('hdbscan', hdb, hdb_seconds)]:
            row = dict(embedding=embedding, method=method, n=len(y), dimensions=X.shape[1],
                       n_true_anomaly=int((y < 0).sum()), n_pred_anomaly=int((assigned < 0).sum()),
                       n_clusters=len(set(assigned[assigned >= 0].tolist())), seconds=elapsed, **scores(y, assigned))
            rows.append(row)
            print(json.dumps(row), flush=True)
        with (out / 'metrics.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    (out / 'protocol.json').write_text(json.dumps(dict(dataset='kendx/NLP-ADBench', split='bbc/test',
        preprocessing='StandardScaler fit on test embedding batch; transductive clustering', seed=42,
        category_mapping=mapping, anomaly_label=-1, configuration=BASE_CONFIG,
        reassignment=dict(penalty_power=3, reach=25, density_power=4),
        ari_real='All ground-truth normal points, including predicted noise',
        timing='Deltric timing includes base graph and reassignment for both reported variants',
        source_manifest=json.loads((source / 'download_manifest.json').read_text())), indent=2)+'\n')
    print(f'Results: {out}', flush=True)

if __name__ == '__main__':
    main()
