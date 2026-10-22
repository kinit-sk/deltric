#!/usr/bin/env python3
"""Loopback-only, dependency-free HTTP server for DelTriC Lab."""
from __future__ import annotations

import argparse
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import subprocess
import sys
import threading
import time
from urllib.parse import parse_qs, urlparse
import uuid
import webbrowser

import numpy as np
import engine

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
LEGACY = engine.diag.OUT
STATIC = ROOT / 'static'
LOCK = threading.RLock()
TRACE_LOCK = threading.Lock()
POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix='lab-job')
CACHE = OrderedDict()
JOBS = {}
for folder in ('cases', 'experiments', 'snapshots', 'notes', 'workspaces', 'logs', 'exports'):
    (DATA / folder).mkdir(parents=True, exist_ok=True)


def now():
    return datetime.now(timezone.utc).isoformat()


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def identifier(value):
    value = str(value)
    if not value or len(value) > 150 or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.' for c in value):
        raise ValueError('Invalid record ID')
    if value in ('.', '..'):
        raise ValueError('Invalid record ID')
    return value


def dataset_path(name):
    path = (engine.ROOT / 'deltric_minimal' / 'data' / (str(name) + '.npz')).resolve()
    source_root = (engine.ROOT / 'deltric_minimal' / 'data').resolve()
    if not path.is_relative_to(source_root) or not path.is_file():
        raise ValueError('Unknown dataset')
    if any(p in {'holdout', 'source', 'raw'} for p in path.relative_to(source_root).parts):
        raise ValueError('This runner uses development datasets; holdout/source/raw directories are excluded')
    return path


def case_index():
    rows = {}
    for out in (LEGACY, DATA):
        for path in sorted((out / 'cases').glob('*.json')):
            try:
                row = json.loads(path.read_text())
                if path.with_suffix('.npz').exists():
                    rows[row['dataset']] = (out, row)
            except (ValueError, KeyError):
                continue
    return rows


def record(name, run='default'):
    key = (name, run)
    with TRACE_LOCK:
        if key in CACHE:
            CACHE.move_to_end(key)
            return CACHE[key]
        cases = case_index()
        if name not in cases:
            raise ValueError('Dataset has no recorded case. Run it first.')
        out, row = cases[name]
        base = engine.load_case(out, row)
        saved = None
        if run != 'default':
            saved = json.loads((DATA / 'experiments' / (identifier(run) + '.json')).read_text())
            if saved['dataset'] != name:
                raise ValueError('Experiment belongs to a different dataset')
            snapshot = DATA / 'snapshots' / identifier(saved['fingerprint'])
            if snapshot.with_suffix('.json').exists() and snapshot.with_suffix('.npz').exists():
                row = json.loads(snapshot.with_suffix('.json').read_text())
                with np.load(snapshot.with_suffix('.npz'), allow_pickle=False) as a:
                    base = {k: a[k].copy() for k in a.files}
        sidecar = Path(row['source']).with_suffix('.json')
        metadata = json.loads(sidecar.read_text()) if sidecar.exists() else {}
        row = {**row, 'feature_names': metadata.get('features', []), 'run_id': 'default',
               'name': 'Recorded default', 'settings': {**row['settings'], 'density_clip': False}}
        source = dataset_path(name)
        row['stale'] = row.get('fingerprint') != engine.diag.fingerprint(source)
        if run == 'default':
            data = engine.replay(base, row['settings'])
            if not np.array_equal(data['final'], base['final']):
                raise ValueError('Saved case no longer replays against current algorithm. Run a fresh case from Datasets.')
        else:
            default_meta = row.copy()
            if saved.get('fingerprint') != row.get('fingerprint'):
                raise ValueError('Experiment uses an older case. Re-run its settings against the current recorded case.')
            row = {**row, **saved, 'stale': row['stale']}
            data = engine.replay(base, row['settings'])
            data['_recorded_meta'] = default_meta
            with np.load(DATA / 'experiments' / (identifier(run) + '.npz'), allow_pickle=False) as a:
                if not np.array_equal(data['final'], a['final']):
                    raise ValueError('Experiment no longer replays. Re-run its settings.')
        result = (data, row, base)
        CACHE[key] = result
        while len(CACHE) > 5:
            CACHE.popitem(last=False)
        return result


def settings_from(body):
    source = body.get('settings', body)
    result = {k: float(source[k]) for k in ('reach', 'penalty', 'density')}
    if not all(np.isfinite(v) and 0 <= v <= 10000 for v in result.values()):
        raise ValueError('Settings must be finite numbers between 0 and 10,000')
    result['density_clip'] = bool(source.get('density_clip', False))
    return result


def experiment(name, settings, title):
    _, default, base = record(name)
    with TRACE_LOCK:
        data = engine.replay(base, settings)
        data['_recorded_meta'] = default
    run = uuid.uuid4().hex
    after = engine.diag.scores(base['y'], data['final'])
    meta = {**default, 'dataset': name, 'run_id': run, 'name': title or f"Reach {settings['reach']:g} · penalty {settings['penalty']:g} · density {settings['density']:g}",
        'created': now(), 'settings': settings, 'after': after,
        'claims': len(data['claims']),
        'swallowed_anomalies': int((data['y'][data['claims'][:, 0].astype(int)] < 0).sum()),
        'recovered_normals': int((data['y'][data['claims'][:, 0].astype(int)] >= 0).sum()),
        'delta': {k: after[k] - default['before'][k] if default['before'][k] is not None else None
                  for k in ('ari_all', 'ari_clean', 'f1', 'recall')}}
    with LOCK:
        snapshot = DATA / 'snapshots' / identifier(default['fingerprint'])
        if not snapshot.with_suffix('.npz').exists():
            np.savez_compressed(snapshot.with_suffix('.npz'), **{k: v for k, v in base.items() if isinstance(v, np.ndarray)})
            write_json(snapshot.with_suffix('.json'), default)
        np.savez_compressed(DATA / 'experiments' / (run + '.npz'), final=data['final'], claims=data['claims'])
        write_json(DATA / 'experiments' / (run + '.json'), meta)
    with TRACE_LOCK:
        CACHE[(name, run)] = (data, meta, base)
        while len(CACHE) > 5:
            CACHE.popitem(last=False)
    return meta


def job(kind, fn):
    job_id = uuid.uuid4().hex
    JOBS[job_id] = dict(id=job_id, kind=kind, status='running', created=now(), progress='Starting…')
    def execute():
        try:
            result = fn(JOBS[job_id])
            JOBS[job_id].update(status='complete', result=clean(result), finished=now())
        except Exception as error:
            JOBS[job_id].update(status='error', error=str(error), finished=now())
    POOL.submit(execute)
    return JOBS[job_id]


def experiments(name):
    rows = [json.loads(p.read_text()) for p in (DATA / 'experiments').glob('*.json')]
    return sorted([r for r in rows if r['dataset'] == name], key=lambda r: r['created'], reverse=True)


def catalog():
    evaluated = case_index()
    rows = []
    source_root = engine.ROOT / 'deltric_minimal' / 'data'
    for path in sorted(source_root.rglob('*.npz')):
        relative = path.relative_to(source_root)
        if any(p in {'holdout', 'source', 'raw'} for p in relative.parts):
            continue
        name = relative.with_suffix('').as_posix()
        row = dict(dataset=name, ready=name in evaluated, bytes=path.stat().st_size)
        if name in evaluated:
            meta = evaluated[name][1]
            row.update(n=meta['n'], d=meta['d'], delta=meta['delta'], after=meta['after'],
                swallowed_anomalies=meta['swallowed_anomalies'], recovered_normals=meta['recovered_normals'])
        else:
            # Metadata is cheap; opening every compressed X array is not.
            sidecar = path.with_suffix('.json')
            if sidecar.exists():
                try:
                    meta = json.loads(sidecar.read_text())
                    row.update(n=meta.get('n_samples', meta.get('n')), d=meta.get('n_features', meta.get('d')))
                except (ValueError, AttributeError):
                    pass
        rows.append(row)
    return rows


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        if args and str(args[0]).split(' ')[0] not in ('GET', 'POST'):
            super().log_message(format, *args)

    def respond(self, value, code=200):
        payload = json.dumps(clean(value), ensure_ascii=False, allow_nan=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(payload)

    def valid_host(self):
        host = self.headers.get('Host', '').split(':')[0]
        origin = self.headers.get('Origin')
        return host in ('127.0.0.1', 'localhost') and (not origin or
            urlparse(origin).netloc == self.headers.get('Host'))

    def do_GET(self):
        if not self.valid_host():
            return self.respond({'error': 'Local origin required'}, 403)
        parsed = urlparse(self.path)
        query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        name, run = query.get('dataset'), query.get('run', 'default')
        try:
            if parsed.path == '/api/catalog':
                return self.respond(catalog())
            if parsed.path == '/api/case':
                data, meta, recorded = record(name, run)
                return self.respond(engine.serialize_case(data, meta, recorded))
            if parsed.path == '/api/point':
                data, meta, _ = record(name, run)
                candidate = int(query['cluster']) if 'cluster' in query else None
                return self.respond(engine.point_evidence(data, meta, int(query['node']), int(query.get('count', len(data['claims']))), candidate))
            if parsed.path == '/api/experiments':
                return self.respond(experiments(name))
            if parsed.path == '/api/notes':
                rows = [json.loads(p.read_text()) for p in (DATA / 'notes').glob('*.json')]
                return self.respond(sorted([r for r in rows if r['dataset'] == name], key=lambda r: r['created'], reverse=True))
            if parsed.path == '/api/workspace':
                path = DATA / 'workspaces' / (name.replace('/', '__') + '.json')
                dataset_path(name)
                return self.respond(json.loads(path.read_text()) if path.exists() else {})
            if parsed.path == '/api/job':
                return self.respond(JOBS[query['id']])
            if parsed.path == '/api/export':
                data, meta, recorded = record(name, run)
                notes = [json.loads(p.read_text()) for p in (DATA / 'notes').glob('*.json')]
                notes = [r for r in notes if r['dataset'] == name]
                payload = dict(meta=meta, notes=notes, claims=data['claims'],
                    timeline=data['timeline'], proposals=data['_events'])
                if 'node' in query:
                    payload['point'] = engine.point_evidence(data, meta, int(query['node']), int(query.get('count', len(data['claims']))))
                with LOCK:
                    write_json(DATA / 'exports' / (name.replace('/', '__') + '_' + identifier(run) + '.json'), payload)
                return self.respond(payload)
            files = {'/': 'index.html', '/app.js': 'app.js', '/style.css': 'style.css'}
            if parsed.path not in files:
                return self.respond({'error': 'Not found'}, 404)
            path = STATIC / files[parsed.path]
            payload = path.read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(payload)
        except (ValueError, KeyError, FileNotFoundError) as error:
            self.respond({'error': str(error)}, 400)
        except Exception as error:
            self.respond({'error': f'{type(error).__name__}: {error}'}, 500)

    def do_POST(self):
        if not self.valid_host():
            return self.respond({'error': 'Local origin required'}, 403)
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if length < 1 or length > 200000:
                raise ValueError('Invalid request size')
            body = json.loads(self.rfile.read(length))
            name = body.get('dataset')
            dataset_path(name)
            route = urlparse(self.path).path
            if route == '/api/experiment':
                settings = settings_from(body)
                return self.respond(job('experiment', lambda state: experiment(name, settings, str(body.get('name', ''))[:200])))
            if route == '/api/sweep':
                settings = settings_from(body)
                values = [float(v) for v in body['reaches']]
                if not 1 <= len(values) <= 12 or not all(np.isfinite(v) and 0 <= v <= 10000 for v in values):
                    raise ValueError('Use 1–12 finite reach values between 0 and 10,000')
                def sweep(state):
                    result = []
                    for i, reach in enumerate(values):
                        state['progress'] = f'Run {i + 1} of {len(values)} · reach {reach:g}'
                        result.append(experiment(name, {**settings, 'reach': reach}, f'Sweep · reach {reach:g}'))
                    return result
                return self.respond(job('sweep', sweep))
            if route == '/api/evaluate':
                timeout = max(10, min(900, int(body.get('timeout', 180))))
                def evaluate(state):
                    log = DATA / 'logs' / (state['id'] + '.log')
                    state['progress'] = f'Clustering {name} · time limit {timeout}s'
                    with log.open('w') as handle:
                        try:
                            process = subprocess.run([sys.executable, str(engine.ROOT / 'deltric_minimal' / 'studies' / 'reassignment_diagnostics.py'),
                                '_one', '--dataset', name, '--out', str(DATA)], stdout=handle,
                                stderr=subprocess.STDOUT, timeout=timeout)
                        except subprocess.TimeoutExpired:
                            raise ValueError(f'Run exceeded {timeout}s. Log: {log}')
                    if process.returncode:
                        raise ValueError(f'Run failed. {log.read_text()[-1800:]}')
                    with TRACE_LOCK:
                        for key in list(CACHE):
                            if key[0] == name:
                                CACHE.pop(key)
                    return dict(dataset=name, log=str(log))
                return self.respond(job('evaluate', evaluate))
            if route == '/api/notes':
                text = str(body.get('text', '')).strip()
                if not text or len(text) > 20000:
                    raise ValueError('Write a note of 1–20,000 characters')
                note_id = identifier(body.get('id', uuid.uuid4().hex))
                path = DATA / 'notes' / (note_id + '.json')
                with LOCK:
                    existing = json.loads(path.read_text()) if path.exists() else {}
                    if existing and existing['dataset'] != name:
                        raise ValueError('Note belongs to another dataset')
                    note = dict(id=note_id, dataset=name, run=identifier(body.get('run', 'default')),
                        node=body.get('node'), frame=body.get('frame'), text=text,
                        created=existing.get('created', now()), updated=now())
                    write_json(path, note)
                return self.respond(note)
            if route == '/api/workspace':
                path = DATA / 'workspaces' / (name.replace('/', '__') + '.json')
                with LOCK:
                    write_json(path, body.get('state', {}))
                return self.respond({'saved': True})
            raise ValueError('Unknown action')
        except (ValueError, KeyError, TypeError) as error:
            self.respond({'error': str(error)}, 400)
        except Exception as error:
            self.respond({'error': f'{type(error).__name__}: {error}'}, 500)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    url = f'http://127.0.0.1:{server.server_port}'
    print(f'DelTriC Lab → {url}\nSaved work → {DATA}', flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        POOL.shutdown(wait=False, cancel_futures=True)


if __name__ == '__main__':
    main()
