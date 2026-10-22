"""Score anomaly detection alongside clustering, without rerunning any fits.

These audit inputs use reference y < 0 for noise. Do not apply that convention
blindly to external binary anomaly datasets, which may instead use y == 1.
"""
from pathlib import Path
import csv
import json
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import precision_recall_fscore_support

OUT = Path(__file__).resolve().parent / 'results/dimension_cause_audit'


def anomaly_metrics(y, pred):
    true, predicted = np.asarray(y) < 0, np.asarray(pred) < 0
    tp = int(np.sum(true & predicted))
    fp = int(np.sum(~true & predicted))
    fn = int(np.sum(true & ~predicted))
    tn = int(np.sum(~true & ~predicted))
    labeled_noise = bool(true.any())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    if labeled_noise:
        independent = precision_recall_fscore_support(true, predicted, average='binary', zero_division=0)[:3]
        assert np.allclose(independent, (precision, recall, f1))
    return dict(anomaly_precision=precision if labeled_noise else None,
                anomaly_recall=recall if labeled_noise else None,
                anomaly_f1=f1 if labeled_noise else None,
                false_positive_rate=fp / (fp + tn) if fp + tn else None,
                true_noise=int(true.sum()), predicted_noise=int(predicted.sum()),
                tp=tp, fp=fp, fn=fn, tn=tn)


def graph_prediction(edges, mask, n):
    e = edges[mask]
    graph = coo_matrix((np.ones(2*len(e)),
        (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    counts = np.bincount(labels[e[:, 0]], minlength=n)
    return np.where(counts[labels] > 10, labels, -1)


def main():
    rows, transitions = [], []
    for p in sorted(OUT.glob('*__*.npz')):
        a = np.load(p)
        meta = json.loads(p.with_suffix('.json').read_text())
        y, e = a['y'], a['edges']
        stages = dict(seed=graph_prediction(e, a['initial'], len(y)),
                      growth=graph_prediction(e, a['pre'], len(y)),
                      gh=a['before'], final=a['final'])
        reconstructed = graph_prediction(e, a['retained'], len(y))
        assert np.array_equal(reconstructed < 0, a['before'] < 0)
        identity = dict(dataset=meta['dataset'], variant=meta['variant'], seed=meta['umap_seed'])
        for stage, pred in stages.items():
            rows.append(dict(**identity, stage=stage,
                ari_clean=meta['stages'][stage]['ari_clean'], **anomaly_metrics(y, pred)))
        reclaimed = (a['before'] < 0) & (a['final'] >= 0)
        assert not np.any((a['before'] >= 0) & (a['final'] != a['before']))
        transitions.append(dict(**identity, reclaimed_true_noise=int((reclaimed & (y < 0)).sum()),
                                reclaimed_true_inliers=int((reclaimed & (y >= 0)).sum())))
    for name, values in [('anomaly_metrics.csv', rows), ('anomaly_reassignment.csv', transitions)]:
        with (OUT/name).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    (OUT/'anomaly_metrics.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps([r for r in rows if r['true_noise'] and r['seed']==42 and
        r['variant'] in ('baseline','no_restore','zero_pad_umap')],indent=2))


if __name__ == '__main__':
    main()
