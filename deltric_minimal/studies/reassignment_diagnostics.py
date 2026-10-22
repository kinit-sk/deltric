#!/usr/bin/env python3
"""Bounded laptop evaluation and exact cost-gate diagnostics, no pipeline edits.

  .venv/bin/python studies/reassignment_diagnostics.py evaluate
  .venv/bin/python studies/reassignment_diagnostics.py export --top 4
  .venv/bin/python studies/reassignment_diagnostics_app.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/deltric-diagnostics-mpl")
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(key, "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

OUT = ROOT / "studies/results/reassignment_diagnostics"
DEFAULT_CASES = [
    "nested_spheres__d2__n2000", "other/square", "blobs_meeples__d5__n500",
    "blobs_meeples__d50__n500", "blobs_meeples__d5__n2000",
    "nested_spheres__d5__n2000", "blobs_informative_subspace__d50__n500__anom10",
    "blobs_informative_subspace__d100__n500__anom10", "text_support_tickets__qwen3emb06b",
    "testing/additional_anomaly/adbench_waveform", "varied_2d_5c",
    "11_v2_moons_2d_2500__anom10", "blobs_with_noise__d2__n500",
    "blobs_varied_density__d2__n2000__anom10", "adbench_speech",
]


def scores(y, pred):
    from sklearn.metrics import adjusted_rand_score
    true, flagged = y < 0, pred < 0
    tp, fp = int((true & flagged).sum()), int((~true & flagged).sum())
    fn, tn = int((true & ~flagged).sum()), int((~true & ~flagged).sum())
    normals = ~true
    return dict(ari_all=float(adjusted_rand_score(y, pred)),
                ari_clean=float(adjusted_rand_score(y[normals], pred[normals])) if normals.any() else None,
                f1=2 * tp / (2 * tp + fp + fn) if true.any() else None,
                precision=tp / (tp + fp) if tp + fp else 0.,
                recall=tp / (tp + fn) if true.any() else None,
                TP=tp, FP=fp, FN=fn, TN=tn)


def trace_gate(base, edges, lengths, core, reach=15., penalty=.5, density=.5):
    """Observe the production function; never reproduce its queue logic."""
    from utils_component_growth import _assign_anomalies_cost_gate as fn
    claims, observed = [], 0

    def trace(frame, event, arg):
        nonlocal observed
        if event == "line":
            z = frame.f_locals
            count = z.get("diagnostics", {}).get("n_claims", 0)
            if count > observed:
                observed = count
                u, v = edges[z["edge_index"]]
                node = int(z["node"])
                claims.append((node, int(z["cluster"]), float(z["distance"]),
                               int(z["edge_index"]), int(v if u == node else u)))
        return trace

    previous = sys.gettrace()
    try:
        sys.settrace(lambda f, e, a: trace if f.f_code is fn.__code__ else None)
        pred, diag = fn(base, edges, lengths, len(base), penalty, reach,
                        core_distance=core, density_power=density, density_clip=False)
    finally:
        sys.settrace(previous)
    claims = np.array(claims, dtype=float).reshape(-1, 5)
    assert len(claims) == diag["n_claims"]
    replay = base.copy()
    for node, cluster, *_ in claims:
        assert replay[int(node)] == -1
        replay[int(node)] = int(cluster)
    assert np.array_equal(replay, pred), "Trace does not reproduce production"
    assert np.array_equal(pred[base >= 0], base[base >= 0])
    assert np.all(np.diff(claims[:, 2]) >= -1e-12)
    return pred, claims


def timeline(y, base, claims, frames=41):
    stops = np.unique(np.linspace(0, len(claims), min(frames, len(claims) + 1)).astype(int))
    rows, pred, done = [], base.copy(), 0
    for stop in stops:
        batch = claims[done:stop]
        pred[batch[:, 0].astype(int)] = batch[:, 1].astype(int)
        s = scores(y, pred)
        rows.append([stop, s['ari_all'], s['ari_clean'] if s['ari_clean'] is not None else np.nan,
                     s['f1'] if s['f1'] is not None else np.nan,
                     s['recall'] if s['recall'] is not None else np.nan,
                     int((y[claims[:stop, 0].astype(int)] < 0).sum())])
        done = stop
    return np.asarray(rows)


def fingerprint(path):
    h = hashlib.sha256(path.read_bytes())
    for source in ("deltric.py", "utils_component_growth.py", "utils_pruning.py",
                   "plot_seed_component_recomputed_growth.py", "plot_seed_component_metrics.py"):
        h.update((ROOT / source).read_bytes())
    h.update(b"diagnostics-v2-default-estimator-stage-history")
    return h.hexdigest()


def finite_json(value):
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite_json(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def evaluate_one(name, out):
    from deltric import DelTriC
    import plot_seed_component_recomputed_growth as stage
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    path = ROOT / "data" / (name + ".npz")
    if any(p in {"holdout", "source", "raw"} for p in path.parts):
        raise ValueError("Only development datasets are eligible")
    key = name.replace("/", "__")
    dest = out / "cases" / key
    dest.parent.mkdir(parents=True, exist_ok=True)
    digest = fingerprint(path)
    if dest.with_suffix('.json').exists() and dest.with_suffix('.npz').exists():
        old = json.loads(dest.with_suffix('.json').read_text())
        if old.get('fingerprint') == digest:
            print(f"CACHE {name}", flush=True)
            return
    started = time.monotonic()
    with np.load(path, allow_pickle=False) as a:
        X, y = np.asarray(a['X'], float), np.asarray(a['y'], int)
        kind = np.asarray(a['anomaly_kind'], int) if 'anomaly_kind' in a else np.zeros(len(y), int)
    captured = {}
    real_gate, real_main = stage._assign_anomalies_cost_gate, stage.main

    def gate(labels, edges, lengths, n, penalty_power, reach, **kw):
        result = real_gate(labels, edges, lengths, n, penalty_power, reach, **kw)
        captured.update(base=labels.copy(), edges=edges.copy(), lengths=lengths.copy(),
                        core=kw['core_distance'].copy(), reach=reach, penalty=penalty_power,
                        density=kw['density_power'])
        return result

    def main():
        real_main()
        result = Path(sys.argv[sys.argv.index('--out') + 1]).with_suffix('.npz')
        with np.load(result, allow_pickle=False) as a:
            captured['xy'] = a['X_proj'].copy()
            edges = a['edge_keys']
            masks = [('Seeds', a['initial_mask'])]
            for r in range(int(a['growth_round_count'])):
                masks.append((f'Growth round {r+1}', a['initial_mask'] | (a['accepted_growth_mask'] & (a['accepted_growth_step'] <= r))))
            masks += [('Restoration', a['bridge_base_mask']), ('Pruning', a['final_retained_mask'])]
            stage_labels = []
            for label, mask in masks:
                cc, points = stage._final_graph_components(edges, mask, len(X))
                stage_labels.append(stage._final_component_prediction(edges, mask, cc, points, 10))
            captured['stage_labels'] = np.asarray(stage_labels)
            captured['stage_names'] = np.asarray([label for label, _ in masks])
        captured['pipeline'] = json.loads(result.with_suffix('.json').read_text())

    stage._assign_anomalies_cost_gate, stage.main = gate, main
    try:
        final = DelTriC().fit_predict(X)
    finally:
        stage._assign_anomalies_cost_gate, stage.main = real_gate, real_main
    replay, claims = trace_gate(captured['base'], captured['edges'], captured['lengths'],
                                captured['core'], captured['reach'], captured['penalty'], captured['density'])
    assert np.array_equal(final, replay)
    base = captured['base']
    before, after = scores(y, base), scores(y, final)
    delta = {k: after[k] - before[k] if before[k] is not None else None
             for k in ('ari_all', 'ari_clean', 'f1', 'recall')}
    harmful = int(((base < 0) & (final >= 0) & (y < 0)).sum())
    normal = int(((base < 0) & (final >= 0) & (y >= 0)).sum())
    majority = {int(c): int(np.unique(y[(base == c) & (y >= 0)], return_counts=True)[0][
        np.argmax(np.unique(y[(base == c) & (y >= 0)], return_counts=True)[1])])
        for c in np.unique(base[base >= 0]) if ((base == c) & (y >= 0)).any()}
    mismatch = int(sum(y[int(node)] >= 0 and int(y[int(node)]) != majority.get(int(c), -999)
                       for node, c, *_ in claims))
    row = dict(dataset=name, n=len(X), d=X.shape[1], before=before, after=after, delta=delta,
               claims=len(claims), swallowed_anomalies=harmful, recovered_normals=normal,
               majority_mismatch_normals=mismatch, majority_mapping=majority,
               seconds=round(time.monotonic() - started, 3), replay_verified=True,
               fingerprint=digest, settings={k:captured[k] for k in ('reach', 'penalty', 'density')},
               source=str(path), pipeline=captured['pipeline'])
    np.savez_compressed(dest.with_suffix('.npz'), X=X, y=y, base=base, final=final,
                        edges=captured['edges'], lengths=captured['lengths'], core=captured['core'],
                        xy=captured['xy'], pca=PCA(2, svd_solver='randomized', random_state=42).fit_transform(StandardScaler().fit_transform(X)),
                        claims=claims, timeline=timeline(y, base, claims), kind=kind,
                        stage_labels=captured['stage_labels'], stage_names=captured['stage_names'])
    dest.with_suffix('.json').write_text(json.dumps(finite_json(row), indent=2, allow_nan=False))
    print(f"DONE {name}: ARI {delta['ari_all']:+.4f} F1 {delta['f1']} swallowed={harmful} {row['seconds']:.1f}s", flush=True)


def load_rows(out):
    return [json.loads(p.read_text()) for p in sorted((out / 'cases').glob('*.json'))]


def severity(row):
    return min(v for v in row['delta'].values() if v is not None)


def evaluate(args):
    args.out.mkdir(parents=True, exist_ok=True)
    started, attempts = time.monotonic(), []
    for name in args.datasets or DEFAULT_CASES:
        remaining = args.budget - (time.monotonic() - started)
        if remaining < 5:
            break
        path = ROOT / 'data' / (name + '.npz')
        if not path.exists():
            attempts.append(dict(dataset=name, status='missing'))
            continue
        print(f"RUN {name} (case limit {min(args.timeout, remaining):.0f}s)", flush=True)
        log = args.out / (name.replace('/', '__') + '.log')
        with log.open('w') as handle:
            try:
                p = subprocess.run([sys.executable, __file__, '_one', '--dataset', name,
                                    '--out', str(args.out)], stdout=handle, stderr=subprocess.STDOUT,
                                   timeout=min(args.timeout, remaining))
                status = 'ok' if p.returncode == 0 else 'error'
            except subprocess.TimeoutExpired:
                status = 'timeout'
        attempts.append(dict(dataset=name, status=status, log=str(log)))
        print(f"  {status}: {log}", flush=True)
    rows = sorted(load_rows(args.out), key=severity)
    run = json.dumps(dict(seconds=time.monotonic() - started,
        budget=args.budget, timeout=args.timeout, attempts=attempts, pipeline='Current DelTriC() estimator defaults'), indent=2)
    history=args.out/'runs';history.mkdir(exist_ok=True)
    (history/f'{time.time_ns()}.json').write_text(run)
    (args.out / 'run.json').write_text(run)
    with (args.out / 'ranking.csv').open('w', newline='') as h:
        keys = ['dataset','n','d','delta_ari_all','delta_ari_clean','delta_f1','delta_recall',
                'swallowed_anomalies','recovered_normals','majority_mismatch_normals','seconds']
        w = csv.DictWriter(h, fieldnames=keys); w.writeheader()
        for row in rows:
            w.writerow({**{k:row[k] for k in keys if k in row},
                        **{'delta_'+k:v for k,v in row['delta'].items()}})
    print(f"{len(rows)} completed cases; {time.monotonic()-started:.1f}s wall time", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    e = sub.add_parser('evaluate'); e.add_argument('--datasets', nargs='+')
    e.add_argument('--budget', type=float, default=300); e.add_argument('--timeout', type=float, default=45)
    one = sub.add_parser('_one'); one.add_argument('--dataset', required=True)
    export = sub.add_parser('export'); export.add_argument('--top', type=int, default=4)
    for s in (e, one, export): s.add_argument('--out', type=Path, default=OUT)
    args = p.parse_args()
    if args.command == 'evaluate': evaluate(args)
    elif args.command == '_one': evaluate_one(args.dataset, args.out)
    else:
        from reassignment_diagnostics_app import export_cases
        export_cases(args.out, args.top)


if __name__ == '__main__':
    main()
