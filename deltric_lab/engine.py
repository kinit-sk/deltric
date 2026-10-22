"""Reassignment evidence from the production queue; no clustering-code edits."""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'deltric_minimal' / 'studies'))
import reassignment_diagnostics as diag
from reassignment_diagnostics_app import alternative_path, load_case
from utils_component_growth import _assign_anomalies_cost_gate as gate


def replay(data, settings):
    """Record queue pops and over-budget attempts at actual production branches.

    An event's `at` is the number of accepted claims BEFORE this attempt.
    The GUI uses that clock to distinguish future evidence from visible evidence.
    """
    source, start = inspect.getsourcelines(gate)
    locations = {}
    for key, needle in [('pop', 'if owner[node] != -1:'),
                        ('seed', 'if cost <= reach:'),
                        ('expand', 'if candidate <= reach:')]:
        locations[key] = start + next(i for i, line in enumerate(source) if needle in line)
    events, claims = [], []
    edges = data['edges']

    def trace(frame, event, arg):
        if event != 'line':
            return trace
        z = frame.f_locals
        line = frame.f_lineno
        at = int(z.get('diagnostics', {}).get('n_claims', 0))
        if line == locations['pop']:
            node, cluster, edge = int(z['node']), int(z['cluster']), int(z['edge_index'])
            u, v = map(int, edges[edge])
            parent = v if u == node else u
            owner = int(z['owner'][node])
            status = 'accepted' if owner == -1 else 'owned'
            events.append(dict(node=node, cluster=cluster, parent=parent, edge=edge,
                               cost=float(z['distance']), owner=owner, status=status, at=at))
            if owner == -1:
                claims.append((node, cluster, float(z['distance']), edge, parent))
        elif line in (locations['seed'], locations['expand']):
            seed = line == locations['seed']
            cost = float(z['cost'] if seed else z['candidate'])
            if cost > settings['reach']:
                node, parent = int(z['neighbour']), int(z['node'])
                events.append(dict(node=node, cluster=int(z['cluster']), parent=parent,
                    edge=int(z['edge_index'] if seed else z['next_edge_index']), cost=cost,
                    owner=int(z['owner'][node]), status='budget', at=at))
        return trace

    previous = sys.gettrace()
    try:
        sys.settrace(lambda f, e, a: trace if f.f_code is gate.__code__ else None)
        final, info = gate(data['base'], edges, data['lengths'], len(data['base']),
            settings['penalty'], settings['reach'], core_distance=data['core'],
            density_power=settings['density'], density_clip=settings.get('density_clip', False))
    finally:
        sys.settrace(previous)
    claims = np.asarray(claims, float).reshape(-1, 5)
    labels = data['base'].copy()
    for node, cluster, *_ in claims:
        assert labels[int(node)] == -1
        labels[int(node)] = int(cluster)
    assert np.array_equal(labels, final)
    assert len(claims) == info['n_claims']
    assert np.array_equal(final[data['base'] >= 0], data['base'][data['base'] >= 0])
    return {**data, 'final': final, 'claims': claims,
            'timeline': diag.timeline(data['y'], data['base'], claims), '_events': events}


def references(data):
    base, edges, lengths, core = (data[k] for k in ('base', 'edges', 'lengths', 'core'))
    result = {}
    for c in np.unique(base[base >= 0]):
        internal = (base[edges[:, 0]] == c) & (base[edges[:, 1]] == c)
        mask = base == c
        result[int(c)] = dict(size=int(mask.sum()), anomalies=int((mask & (data['y'] < 0)).sum()),
            mean_edge=max(float(lengths[internal].mean() if internal.any() else
                                np.median(lengths) if len(lengths) else 1), 1e-12),
            core_reference=max(float(np.median(core[mask])), 1e-12))
    return result


def path_to(data, node):
    parents = {int(c[0]): int(c[4]) for c in data['claims']}
    path, seen = [node], {node}
    while path[-1] in parents:
        parent = parents[path[-1]]
        if parent in seen:
            raise ValueError('Cycle in winning trace')
        path.append(parent)
        seen.add(parent)
    return path[::-1]


def point_evidence(data, meta, node, count, candidate=None):
    if not 0 <= node < len(data['base']):
        raise ValueError('Point ID out of range')
    count = max(0, min(count, len(data['claims'])))
    refs = references(data)
    by_node = {int(c[0]): (i + 1, c) for i, c in enumerate(data['claims'])}
    selected = by_node.get(node)
    path = path_to(data, node)
    hops = []
    for target in path[1:]:
        number, claim = by_node[target]
        cluster, edge, parent = int(claim[1]), int(claim[3]), int(claim[4])
        ref = refs[cluster]
        rho = float(data['core'][target]) / ref['core_reference']
        effective_rho = max(1, rho) if meta['settings'].get('density_clip') else rho
        ratio = float(data['lengths'][edge]) / ref['mean_edge']
        hop_cost = (ratio * effective_rho ** meta['settings']['density']) ** (1 + meta['settings']['penalty'])
        hops.append(dict(parent=parent, node=target, edge=edge, cluster=cluster, claim=number,
            length=float(data['lengths'][edge]), mean_edge=ref['mean_edge'], edge_ratio=ratio,
            core=float(data['core'][target]), core_reference=ref['core_reference'], density_ratio=rho,
            hop_cost=hop_cost, cumulative=float(claim[2]), visible=number <= count))
    events = [{**e, 'visible': (e['at'] < count if e['status'] == 'accepted' else e['at'] <= count)}
              for e in data['_events'] if e['node'] == node]
    events.sort(key=lambda e: (e['at'], e['cost'], e['cluster']))
    base, truth, final = int(data['base'][node]), int(data['y'][node]), int(data['final'][node])
    visible = int(selected[1][1]) if selected and selected[0] <= count else base
    if base >= 0:
        explanation = 'Already assigned before reassignment. The cost gate cannot change this point.'
    elif selected and selected[0] <= count:
        explanation = f"Claim {selected[0]} assigned this point to cluster {final}. " + (
            'A true anomaly was absorbed.' if truth < 0 else
            'A true normal was assigned; check its destination cluster against ground truth.')
        sources = data['y'][(data['base'] == final) & (data['y'] >= 0)]
        if truth >= 0 and len(sources):
            values, counts = np.unique(sources, return_counts=True)
            majority = int(values[counts.argmax()])
            explanation += (f" Most of cluster {final}'s fixed normal sources have truth {majority}"
                            + (', matching this point.' if majority == truth else f', not {truth}.'))
    elif selected:
        explanation = f"Still unassigned at this frame. It will be claimed at claim {selected[0]} by cluster {final}."
    elif any(e['status'] == 'budget' for e in events):
        explanation = 'Remained unassigned. Recorded incoming attempts exceeded the reach budget.'
    else:
        explanation = 'Remained unassigned. No proposal reached it in the production queue. Inspect its graph neighbors and a hypothetical candidate route.'
    neighbors = []
    for i in np.flatnonzero((data['edges'] == node).any(axis=1)):
        u, v = map(int, data['edges'][i])
        other = v if u == node else u
        entry = by_node.get(other)
        neighbors.append(dict(node=other, edge=int(i), length=float(data['lengths'][i]),
            base=int(data['base'][other]), final=int(data['final'][other]),
            visible=int(entry[1][1]) if entry and entry[0] <= count else int(data['base'][other])))
    route = None
    if candidate is not None:
        if candidate not in refs:
            raise ValueError('Unknown source cluster')
        if meta['settings'].get('density_clip'):
            route = dict(cluster=candidate, path=[], reason='Optimistic route comparison is available with density floor off. Actual proposals above use the active floor setting.')
        elif base >= 0:
            route = dict(cluster=candidate, path=[], reason='Already assigned points are fixed sources/barriers; they cannot be reassigned.')
        else:
            cost, nodes, blockers, avg, ref = alternative_path(data, node, candidate, meta['settings'])
            route = dict(cluster=candidate, cost=cost, path=nodes, blockers=blockers,
                mean_edge=avg, core_reference=ref, within_budget=bool(cost <= meta['settings']['reach']),
                reason='Hypothetical independent route; ignores ownership competition. A cheap route does not prove this cluster should win.')
    names = meta.get('feature_names', [])
    features = [dict(name=names[i] if i < len(names) else f'Feature {i}', value=float(value))
                for i, value in enumerate(data['X'][node])]
    return dict(node=node, truth=truth, base=base, final=final, visible=visible,
        claim=selected[0] if selected else None, core=float(data['core'][node]),
        explanation=explanation, path=path, hops=hops, proposals=events, neighbors=neighbors,
        source=refs.get(final), features=features, route=route,
        visible_path=[n for n in path if n not in by_node or by_node[n][0] <= count])


def serialize_case(data, meta, recorded):
    names = meta.get('feature_names', [])
    projections = {'graph': data['xy'], 'pca': data['pca']}
    for label in ('ra / dec', 'pmra / pmdec', 'parallax / pmra'):
        if label in data:
            projections[label] = data[label]
    recorded_meta = {**meta, 'run_id': 'default', 'name': 'Recorded default',
                     'after': diag.scores(data['y'], recorded['final'])}
    # Experiments carry active settings; retain recorded settings explicitly.
    if '_recorded_meta' in data:
        recorded_meta = data['_recorded_meta']
    return dict(meta=meta, recorded_meta=recorded_meta, projections=projections, y=data['y'], base=data['base'],
        final=data['final'], recorded_final=recorded['final'], recorded_claims=recorded['claims'], core=data['core'],
        edges=data['edges'], lengths=data['lengths'], claims=data['claims'], timeline=data['timeline'],
        stage_labels=data.get('stage_labels', []), stage_names=data.get('stage_names', []),
        stage_scores=[diag.scores(data['y'], labels) for labels in data.get('stage_labels', [])],
        clusters=references(data), feature_names=names,
        event_summary={s: sum(e['status'] == s for e in data['_events']) for s in ('accepted', 'owned', 'budget')})
