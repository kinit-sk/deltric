#!/usr/bin/env python3
"""Native Tk diagnostics window; reads bounded current-pipeline evaluations.

Run with deltric_minimal/.venv/bin/python. Use --preview FILE for headless QA.
"""
from __future__ import annotations

import argparse
import heapq
import inspect
import json
from pathlib import Path
import queue
import sys
import threading

import reassignment_diagnostics as diag
import numpy as np

COLORS = ['#2563eb', '#16a34a', '#9333ea', '#d97706', '#0891b2', '#db2777', '#4f46e5', '#65a30d']
NOISE, HARM, GOOD = '#9ca3af', '#dc2626', '#059669'


def load_case(out, row):
    path = out / 'cases' / (row['dataset'].replace('/', '__') + '.npz')
    with np.load(path, allow_pickle=False) as a:
        data = {k:a[k].copy() for k in a.files}
    data['Feature 0 / 1'] = data['X'][:,:2]
    sidecar = Path(row['source']).with_suffix('.json')
    names = json.loads(sidecar.read_text()).get('features',[]) if sidecar.exists() else []
    for left,right in [('ra','dec'),('pmra','pmdec'),('parallax','pmra')]:
        if left in names and right in names:
            data[f'{left} / {right}'] = data['X'][:,[names.index(left),names.index(right)]]
    return data


def label_colors(labels):
    return [NOISE if c < 0 else COLORS[int(c) % len(COLORS)] for c in labels]


def fmt(value):
    return 'n/a' if value is None or not np.isfinite(value) else f'{value:.3f}'


def scatter(ax, xy, labels, title):
    artist = ax.scatter(*xy.T, c=label_colors(labels), s=9, alpha=.8, linewidths=0)
    ax.set(title=title, aspect='equal'); ax.tick_params(labelsize=7)
    return artist


def draw_dashboard(fig, data, row, count=None, projection='xy', selected=None, route=None, stage_index=None, full_pipeline=False, reference_data=None):
    """Same dashboard for the native window, PNGs, and exact-claim GIFs."""
    fig.clear()
    xy, y, base, claims = data[projection], data['y'], data['base'], data['claims']
    count = len(claims) if count is None else min(int(count), len(claims))
    pred = base.copy()
    batch = claims[:count]
    nodes = batch[:, 0].astype(int)
    pred[nodes] = batch[:, 1].astype(int)
    if stage_index is not None:
        pred = data['stage_labels'][stage_index]
    axes = fig.subplots(2, 3)
    scatter(axes[0, 0], xy, y, 'Ground truth (gray = true anomaly)')
    scatter(axes[0, 1], xy, base if reference_data is None else reference_data['final'],
            'Before reassignment (gray = unassigned)' if reference_data is None else 'Recorded default result')
    visible_title = str(data['stage_names'][stage_index]) if stage_index is not None else f'Reassignment: {count:,} / {len(claims):,} claims'
    scatter(axes[0, 2], xy, pred, visible_title)
    harmful = nodes[y[nodes] < 0]
    useful = nodes[y[nodes] >= 0]
    for ax in (axes[0, 2], axes[1, 0]):
        if ax == axes[1, 0]:
            ax.scatter(*xy.T, c='#e2e8f0', s=7, linewidths=0)
            ax.set(title='Changed points: green normals / red anomalies', aspect='equal')
            ax.tick_params(labelsize=7)
        ax.scatter(*xy[useful].T, facecolors='none', edgecolors=GOOD, s=22, linewidths=.7)
        ax.scatter(*xy[harmful].T, c=HARM, marker='x', s=30, linewidths=1)
    for ax in (axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0]):
        if route:
            ax.plot(*xy[route].T, color='#d97706', lw=2, alpha=.85, ls='--')
        if selected is not None:
            actual = winning_path(data, selected)
            if len(actual)>1:ax.plot(*xy[actual].T,color='#111827',lw=2,alpha=.8)
            ax.scatter(*xy[selected], s=125, facecolors='none', edgecolors='#111827', linewidths=2)
    t = data['timeline'].copy()
    marker = count
    if full_pipeline and 'stage_labels' in data:
        offset = len(data['stage_labels'])
        t[:,0] += offset
        earlier = []
        for i, labels in enumerate(data['stage_labels']):
            s = diag.scores(y, labels)
            earlier.append([i,s['ari_all'],s['ari_clean'],s['f1'] if s['f1'] is not None else np.nan,
                            s['recall'] if s['recall'] is not None else np.nan,0])
        t = np.vstack([earlier,t])
        marker = stage_index if stage_index is not None else count+offset
        axes[1,1].axvline(offset,color='#64748b',ls=':',lw=1,label='Reassignment starts')
    for col, label, color in ((1,'ARI all','#2563eb'), (2,'ARI clean','#9333ea'), (3,'Anomaly F1','#dc2626')):
        if np.isfinite(t[:,col]).any():
            axes[1, 1].plot(t[:,0], t[:,col], label=label, color=color)
    axes[1, 1].axvline(marker, color='#111827', ls='--', lw=1)
    axes[1, 1].set(title='Quality through pipeline' if full_pipeline else 'Quality through actual claim order',
                  xlabel='Stage frames, then claims' if full_pipeline else 'Claims', ylabel='Score')
    axes[1, 1].legend(fontsize=7); axes[1, 1].grid(alpha=.15)
    axes[1, 2].plot(np.arange(len(claims)) + 1, claims[:,2], color='#475569', lw=1)
    if len(claims):
        bad = y[claims[:,0].astype(int)] < 0
        axes[1, 2].scatter(np.flatnonzero(bad) + 1, claims[bad,2], c=HARM, s=10, label='True anomaly claimed')
    axes[1, 2].axhline(row['settings']['reach'], color=HARM, ls='--', label='Reach budget')
    axes[1, 2].axvline(count, color='#111827', ls='--', lw=1)
    axes[1, 2].set(title='Winning cumulative cost', xlabel='Claims', ylabel='Cost')
    axes[1, 2].legend(fontsize=7); axes[1, 2].grid(alpha=.15)
    current = diag.scores(y, pred)
    before = row['before']
    if reference_data is not None:before=diag.scores(y,reference_data['final'])
    fig.suptitle(row['dataset'], fontsize=12, fontweight='bold', y=.985)
    fig.text(.5,.945, f"{'Default' if reference_data is not None else 'Pre-reassign'} → visible: ARI {fmt(before['ari_all'])} → {fmt(current['ari_all'])}   |   "
             f"F1 {fmt(before['f1'])} → {fmt(current['f1'])}   |   "
             f"Recovered normals {len(useful)}   |   Swallowed anomalies {len(harmful)}", ha='center', fontsize=9)
    fig.text(.5,.012, f"{row['n']:,} points · {row['d']} features · "
             f"{'Native 2D' if projection=='xy' and row['d']==2 else 'Graph projection' if projection == 'xy' else 'PCA display only' if projection=='pca' else projection+' (raw features)'} · "
             f"reach={row['settings']['reach']:g}, penalty={row['settings']['penalty']:g}, density={row['settings']['density']:g} · "
             'Costs use standardized original features. Black path: actual; orange dashed: hypothetical. Truth/cluster colors are independent.',
             ha='center', fontsize=7)
    fig.subplots_adjust(left=.055, right=.98, bottom=.09, top=.875, hspace=.42, wspace=.3)
    return axes


def winning_path(data, node):
    parents = {int(c[0]):int(c[4]) for c in data['claims']}
    path, seen = [node], {node}
    while path[-1] in parents:
        parent = parents[path[-1]]
        if parent in seen: raise ValueError('Claim trace contains a cycle')
        path.append(parent); seen.add(parent)
    return path[::-1]


def alternative_path(data, target, cluster, settings):
    """Optimistic single-source-cluster route ignoring other noise ownership.

    Existing assigned points of other clusters remain barriers. This comparator
    is deliberately different from the production permanent-ownership queue.
    """
    base, edges, lengths, core = (data[k] for k in ('base','edges','lengths','core'))
    internal = (base[edges[:,0]] == cluster) & (base[edges[:,1]] == cluster)
    average = max(float(lengths[internal].mean() if internal.any() else np.median(lengths)), 1e-12)
    reference = max(float(np.median(core[base == cluster])), 1e-12)
    adj = [[] for _ in base]
    for i,(u,v) in enumerate(edges):
        adj[u].append((int(v), i)); adj[v].append((int(u), i))
    distances = np.full(len(base), np.inf); parents = np.full(len(base), -1, int)
    sources = np.flatnonzero(base == cluster); distances[sources] = 0
    heap = [(0., int(n)) for n in sources]; heapq.heapify(heap)
    while heap:
        cost, node = heapq.heappop(heap)
        if cost != distances[node]: continue
        if node == target: break
        for neighbour, i in adj[node]:
            if base[neighbour] >= 0: continue
            hop = (lengths[i] * (core[neighbour]/reference)**settings['density']/average)**(1+settings['penalty'])
            candidate = cost + hop
            if candidate < distances[neighbour]:
                distances[neighbour], parents[neighbour] = candidate, node
                heapq.heappush(heap, (candidate, neighbour))
    route = [target]
    if np.isfinite(distances[target]):
        while parents[route[-1]] >= 0: route.append(int(parents[route[-1]]))
        route.reverse()
    else: route = []
    blockers = [n for n in route if base[n] < 0 and data['final'][n] >= 0 and data['final'][n] != cluster]
    return float(distances[target]), route, blockers, average, reference


def inspect_point(data, row, node, cluster=None):
    base, final, y = (data[k] for k in ('base','final','y'))
    lines = [f"POINT {node}   truth={y[node]}   before={base[node]}   after={final[node]}",
             f"Original feature values: {np.array2string(data['X'][node,:12], precision=3)}",
             f"15-NN core distance (standardized space): {data['core'][node]:.6g}"]
    path = winning_path(data, node)
    match = np.flatnonzero(data['claims'][:,0] == node)
    if len(match):
        c = data['claims'][match[0]]
        source_mask=base==int(c[1]);source_anomalies=int((source_mask&(y<0)).sum())
        lines += [f"Claim #{match[0]+1}, winner cluster {int(c[1])}, cumulative cost {c[2]:.6g}",
                  f"Fixed source cluster: {int(source_mask.sum())} points, {source_anomalies} are ground-truth anomalies",
                  f"Incoming edge #{int(c[3])}, length={data['lengths'][int(c[3])]:.6g}",
                  f"Actual winning path: {' → '.join(map(str,path))}"]
        if source_anomalies==int(source_mask.sum()):
            lines.append('UPSTREAM FAILURE: every point in this fixed source cluster is a true anomaly.')
        if y[node] < 0: lines.append('HARM: a ground-truth anomaly was absorbed into a cluster.')
        else:
            majority = row.get('majority_mapping', {}).get(str(final[node]))
            lines.append(f"Destination seed majority truth={majority}; this is a heuristic, not a pointwise ARI label.")
    elif base[node] >= 0:
        lines.append('This point was already assigned; reassignment never changes it.')
    else: lines.append('This point remained unassigned: compare a candidate cluster below.')
    if cluster is not None:
        cost, route, blockers, avg, ref = alternative_path(data, node, cluster, row['settings'])
        actual = target_proposals(data, node, row['settings'])
        popped = [item for item in actual if item[1] == cluster]
        lines += [f"\nALTERNATIVE CLUSTER {cluster} (hypothetical independent route)",
                  f"Mean internal edge length={avg:.6g}; median seed core distance={ref:.6g}",
                  f"Optimistic cumulative cost={cost:.6g}; reach={row['settings']['reach']:g}",
                  'Hop cost = (length × (destination core / seed median core)^density / mean edge)^(1+penalty)']
        if popped:
            queued = min(popped, key=lambda item:item[0])
            lines.append(f"Actual queue proposed this cluster at cost {queued[0]:.6g}; owner at pop={queued[2]}.")
            if queued[2] >= 0 and queued[2] != cluster:
                lines.append('PROVEN REJECTION: the point was already owned when this proposal was popped.')
        else:
            lines.append('No proposal from this cluster reached the point in the actual queue.')
        if not np.isfinite(cost): lines.append('No graph route exists through initially unassigned points.')
        elif cost > row['settings']['reach']: lines.append('Even the optimistic route exceeds the reach budget.')
        elif final[node] == cluster: lines.append('This candidate is the actual winner.')
        elif blockers:
            lines.append(f"Route crosses points owned by other clusters: {blockers[:12]}. "
                         'This route cannot establish that this candidate should have won; ownership timing matters.')
        else: lines.append('This optimistic route is within budget. Inspect actual proposals above; equal-cost ties use node and cluster IDs.')
        lines.append('Actual popped proposals (cost, cluster, owner at pop): ' + str(actual[:15]))
        lines.append('Optimistic route: ' + ' → '.join(map(str,route)))
        path = route or path
    return '\n'.join(lines), path


def target_proposals(data, target, settings):
    """Capture actual pop attempts, including proposals skipped after ownership."""
    from utils_component_growth import _assign_anomalies_cost_gate as fn
    source,start = inspect.getsourcelines(fn)
    check = start + next(i for i,line in enumerate(source) if 'if owner[node] != -1:' in line)
    proposals = []
    def trace(frame,event,arg):
        if event=='line' and frame.f_lineno==check and frame.f_locals['node']==target:
            z=frame.f_locals
            proposals.append((float(z['distance']),int(z['cluster']),int(z['owner'][target])))
        return trace
    previous=sys.gettrace()
    try:
        sys.settrace(lambda f,e,a:trace if f.f_code is fn.__code__ else None)
        result,_=fn(data['base'],data['edges'],data['lengths'],len(data['base']),
                    settings['penalty'],settings['reach'],core_distance=data['core'],density_power=settings['density'],density_clip=False)
    finally:sys.settrace(previous)
    assert np.array_equal(result,data['final'])
    return proposals


def experiment(data, row, settings):
    result = {**data}
    result['final'], result['claims'] = diag.trace_gate(data['base'], data['edges'], data['lengths'],
                         data['core'], settings['reach'], settings['penalty'], settings['density'])
    result['timeline'] = diag.timeline(data['y'], data['base'], result['claims'])
    after = diag.scores(data['y'], result['final'])
    meta = {**row, 'settings':settings, 'after':after,
            'delta':{k: after[k]-row['before'][k] if row['before'][k] is not None else None
                     for k in row['delta']}}
    return result, meta


def claim_ratios(data):
    base,edges,lengths,core=(data[k] for k in ('base','edges','lengths','core'))
    ratios=np.empty((len(data['claims']),2))
    for cluster in np.unique(data['claims'][:,1].astype(int)):
        mask=data['claims'][:,1]==cluster
        internal=(base[edges[:,0]]==cluster)&(base[edges[:,1]]==cluster)
        average=max(float(lengths[internal].mean() if internal.any() else np.median(lengths)),1e-12)
        reference=max(float(np.median(core[base==cluster])),1e-12)
        ratios[mask,0]=lengths[data['claims'][mask,3].astype(int)]/average
        ratios[mask,1]=core[data['claims'][mask,0].astype(int)]/reference
    return ratios


def draw_anatomy(fig,data,row,selected=None):
    fig.clear();axes=fig.subplots(2,2)
    claims=data['claims'];nodes=claims[:,0].astype(int);bad=data['y'][nodes]<0
    ratios=claim_ratios(data)
    cost_bins=np.linspace(0,max(1e-12,row['settings']['reach'],float(claims[:,2].max()) if len(claims) else 0),21)
    for mask,color,label in [(~bad,GOOD,'Normal claimed'),(bad,HARM,'True anomaly claimed')]:
        axes[0,0].scatter(ratios[mask,0],ratios[mask,1],c=color,s=15,alpha=.75,label=label)
        axes[0,1].hist(claims[mask,2],bins=cost_bins,histtype='step',color=color,label=label,lw=2)
    if selected is not None:
        chosen=nodes==selected
        axes[0,0].scatter(ratios[chosen,0],ratios[chosen,1],facecolors='none',edgecolors='#111827',s=130,lw=2)
    axes[0,0].axvline(1,color='#64748b',ls=':');axes[0,0].axhline(1,color='#64748b',ls=':')
    axes[0,0].set(title='Actual incoming hops: scale and density',xlabel='Edge length / destination cluster mean edge',
                  ylabel='Point core distance / destination cluster median core')
    axes[0,0].legend(fontsize=8)
    axes[0,1].axvline(row['settings']['reach'],color='#111827',ls='--',label='Reach')
    axes[0,1].set(title='Winning costs: does cost distinguish harm?',xlabel='Cumulative path cost',ylabel='Claim count')
    axes[0,1].legend(fontsize=8)
    positions=np.arange(3)
    axes[1,0].bar(positions-.18,[row['before'][k] for k in ('TP','FP','FN')],width=.36,label='Before',color='#94a3b8')
    axes[1,0].bar(positions+.18,[row['after'][k] for k in ('TP','FP','FN')],width=.36,label='After',color='#2563eb')
    axes[1,0].set_xticks(positions,['TP: detected anomaly','FP: flagged normal','FN: missed anomaly'],fontsize=8)
    axes[1,0].set(title='Anomaly detection confusion counts',ylabel='Points');axes[1,0].legend(fontsize=8)
    if np.any(data['kind']>0):
        kinds=[k for k in np.unique(data['kind']) if k>0];pos=np.arange(len(kinds))
        labels={1:'uniform',2:'far',3:'bridge',4:'micro',5:'subspace'}
        for shift,key,color in [(-.18,'base','#94a3b8'),(.18,'final','#2563eb')]:
            rates=[float((data[key][(data['kind']==k)&(data['y']<0)]<0).mean())
                   if ((data['kind']==k)&(data['y']<0)).any() else np.nan for k in kinds]
            axes[1,1].bar(pos+shift,rates,width=.36,label='Before' if key=='base' else 'After',color=color)
        axes[1,1].set_xticks(pos,[labels.get(int(k),str(k)) for k in kinds],fontsize=8)
        axes[1,1].set(title='Anomaly recall by recorded anomaly kind',ylabel='Recall',ylim=(0,1.05));axes[1,1].legend(fontsize=8)
    else:
        clusters,counts=np.unique(claims[bad,1].astype(int),return_counts=True)
        order=np.argsort(-counts)[:12]
        axes[1,1].bar(np.arange(len(order)),counts[order],color=HARM)
        axes[1,1].set_xticks(np.arange(len(order)),clusters[order],fontsize=8)
        axes[1,1].set(title='Which clusters swallowed true anomalies?',xlabel='Destination cluster ID',ylabel='Newly absorbed true anomalies')
    for ax in axes.flat:ax.grid(alpha=.12)
    fig.suptitle(row['dataset']+' · Cost anatomy',fontweight='bold',fontsize=12)
    fig.text(.5,.012,'These are successful claims only. Inspect an unassigned point to compare routes that did not win. '
             'Ratios use the fixed pre-reassignment cluster references.',ha='center',fontsize=8)
    fig.subplots_adjust(left=.09,right=.97,top=.90,bottom=.10,hspace=.4,wspace=.28)
    return axes,ratios


def export_cases(out, top):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.animation import FuncAnimation, PillowWriter
    rows = sorted(diag.load_rows(out), key=diag.severity)
    export = out / 'exports'; export.mkdir(parents=True, exist_ok=True)
    selected=[]
    def include(candidates,limit):
        for row in candidates[:limit]:
            if row not in selected:selected.append(row)
    include(sorted([r for r in rows if r['delta']['f1'] is not None], key=lambda r:r['delta']['f1']), max(1,top//2))
    include(sorted(rows,key=lambda r:min(r['delta']['ari_all'],r['delta']['ari_clean'] or 0)),max(1,top//2))
    include(sorted([r for r in rows if 'gaia' in r['dataset']],key=diag.severity),2)
    include(sorted([r for r in rows if 'gaia' in r['dataset']],key=lambda r:r['delta']['ari_all']),1)
    include(rows,top)
    selected=selected[:max(top,4)]
    (export/'selection.json').write_text(json.dumps([r['dataset'] for r in selected],indent=2))
    for row in selected:
        data = load_case(out, row); name = row['dataset'].replace('/','__')
        fig = Figure(figsize=(15,9), dpi=95); FigureCanvasAgg(fig)
        draw_dashboard(fig, data, row)
        fig.savefig(export / (name + '.png'), dpi=140)
        draw_anatomy(fig,data,row);fig.savefig(export/(name+'_cost_anatomy.png'),dpi=140)
        # Every visible state is an exact prefix, batched for a short playback.
        frames = np.unique(np.linspace(0,len(data['claims']),31).astype(int))
        animation = FuncAnimation(fig, lambda count:draw_dashboard(fig,data,row,count), frames=frames, interval=150)
        animation.save(export / (name + '.gif'), writer=PillowWriter(fps=7))
        if 'stage_labels' in data:
            stages = len(data['stage_labels'])
            full_frames = [(0,i) for i in range(stages) for _ in range(3)] + [(count,None) for count in frames]
            animation = FuncAnimation(fig,lambda item:draw_dashboard(fig,data,row,item[0],stage_index=item[1],full_pipeline=True),
                                      frames=full_frames,interval=150)
            animation.save(export / (name + '_full_pipeline.gif'),writer=PillowWriter(fps=7))
        print(f"EXPORTED {name}", flush=True)
    fig = Figure(figsize=(12,max(4,len(rows)*.3)), dpi=130); FigureCanvasAgg(fig)
    ax = fig.subplots(); pos = np.arange(len(rows))
    for shift, metric, label, color in ((-.23,'ari_all','ARI all','#2563eb'),(0,'ari_clean','ARI clean','#9333ea'),(.23,'f1','Anomaly F1','#dc2626')):
        ax.barh(pos+shift, [r['delta'][metric] if r['delta'][metric] is not None else np.nan for r in rows],
                height=.22, label=label, color=color)
    ax.set_yticks(pos, [r['dataset'] for r in rows], fontsize=8); ax.invert_yaxis()
    ax.axvline(0,color='#111827',lw=1); ax.legend(fontsize=8)
    ax.set(title='Current reassignment: score change on identical frozen graph', xlabel='After − before (negative = regression)')
    fig.tight_layout(); fig.savefig(export/'ranking.png')


class App:
    def __init__(self, window, out):
        import tkinter as tk
        from tkinter import ttk
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        self.tk, self.ttk, self.window, self.out = tk, ttk, window, out
        self.rows = diag.load_rows(out); self.messages = queue.Queue()
        self.busy = False; self.playing = False; self.setting_scale = False
        self.selected = None; self.route = None; self.revision = 0
        self.anatomy=False
        window.title('DelTriC · Reassignment diagnostics')
        window.geometry(f'{min(1550,window.winfo_screenwidth()-80)}x{min(1000,window.winfo_screenheight()-80)}')
        window.minsize(1050,700)
        shell = ttk.Panedwindow(window, orient='horizontal'); shell.pack(fill='both',expand=True)
        sidebar = ttk.Frame(shell, padding=10); body = ttk.Frame(shell); shell.add(sidebar,weight=0); shell.add(body,weight=1)
        ttk.Label(sidebar,text='FAILURE EXPLORER',font=('sans',12,'bold')).pack(anchor='w')
        ttk.Label(sidebar,text='Rank by score change; negative is worse').pack(anchor='w',pady=4)
        self.metric = tk.StringVar(value='f1')
        sort = ttk.Combobox(sidebar,textvariable=self.metric,values=['f1','ari_all','ari_clean','recall','worst'],state='readonly',width=28)
        sort.pack(fill='x'); sort.bind('<<ComboboxSelected>>',lambda e:self.rank())
        self.filter=tk.StringVar(value='All evaluated cases')
        filter_box=ttk.Combobox(sidebar,textvariable=self.filter,values=['All evaluated cases','F1 regressions','ARI regressions','Gaia'],state='readonly')
        filter_box.pack(fill='x',pady=(5,0));filter_box.bind('<<ComboboxSelected>>',lambda e:self.rank())
        self.listbox = tk.Listbox(sidebar,width=43,height=10,exportselection=False,font=('monospace',9))
        self.listbox.pack(fill='both',expand=True,pady=6); self.listbox.bind('<<ListboxSelect>>',self.choose)
        horizontal=ttk.Scrollbar(sidebar,orient='horizontal',command=self.listbox.xview)
        horizontal.pack(fill='x');self.listbox.configure(xscrollcommand=horizontal.set)
        ttk.Label(sidebar,text='EXPERIMENT ON SAME GRAPH',font=('sans',10,'bold')).pack(anchor='w',pady=(8,4))
        self.entries = {}
        for key, default in [('reach',15),('penalty',.5),('density',.5)]:
            line=ttk.Frame(sidebar); line.pack(fill='x',pady=2)
            ttk.Label(line,text=key.title(),width=12).pack(side='left')
            value=tk.StringVar(value=str(default)); self.entries[key]=value
            ttk.Entry(line,textvariable=value,width=12).pack(side='left')
        self.run_button=ttk.Button(sidebar,text='Run reassignment experiment',command=self.run_experiment); self.run_button.pack(fill='x',pady=5)
        ttk.Button(sidebar,text='Restore recorded default',command=self.restore).pack(fill='x')
        self.comparison=tk.StringVar(); ttk.Label(sidebar,textvariable=self.comparison,wraplength=320).pack(anchor='w',pady=6)
        ttk.Label(sidebar,text='CANDIDATE CLUSTER',font=('sans',10,'bold')).pack(anchor='w',pady=(8,4))
        self.cluster=tk.StringVar()
        self.candidates=ttk.Combobox(sidebar,textvariable=self.cluster,state='readonly',width=24)
        self.candidates.pack(fill='x'); self.candidates.bind('<<ComboboxSelected>>',lambda e:self.explain())
        ttk.Button(sidebar,text='Explain selected point / route',command=self.explain).pack(fill='x',pady=4)
        ttk.Button(sidebar,text='Save current chart + point details',command=self.save).pack(fill='x',pady=4)
        ttk.Button(sidebar,text='Toggle cost / density diagnostics',command=self.toggle_anatomy).pack(fill='x',pady=4)
        controls=ttk.Frame(body,padding=6); controls.pack(fill='x')
        ttk.Button(controls,text='Play / Pause',command=self.toggle_play).pack(side='left')
        ttk.Button(controls,text='Before',command=lambda:self.scrub(0)).pack(side='left')
        ttk.Button(controls,text='After',command=lambda:self.scrub(self.frame_count())).pack(side='left')
        self.projection=tk.StringVar(value='xy')
        selector=ttk.Combobox(controls,textvariable=self.projection,values=['xy','pca'],state='readonly',width=15)
        self.projection_box=selector
        selector.pack(side='left',padx=8); selector.bind('<<ComboboxSelected>>',lambda e:self.redraw())
        self.mode=tk.StringVar(value='Full pipeline')
        mode=ttk.Combobox(controls,textvariable=self.mode,values=['Full pipeline','Reassignment'],state='readonly',width=14)
        mode.pack(side='left'); mode.bind('<<ComboboxSelected>>',lambda e:self.change_mode())
        self.compare=tk.BooleanVar(value=False)
        ttk.Checkbutton(controls,text='Compare default',variable=self.compare,command=self.redraw).pack(side='left',padx=6)
        self.count=tk.DoubleVar(value=0)
        self.scale=ttk.Scale(controls,variable=self.count,from_=0,to=1,command=self.slider)
        self.scale.pack(side='left',fill='x',expand=True,padx=8)
        self.claim_label=tk.StringVar(); ttk.Label(controls,textvariable=self.claim_label,width=20).pack(side='left')
        self.fig=Figure(figsize=(12,7),dpi=95)
        self.canvas=FigureCanvasTkAgg(self.fig,master=body); self.canvas.get_tk_widget().pack(fill='both',expand=True)
        toolbar=NavigationToolbar2Tk(self.canvas,body,pack_toolbar=False); toolbar.pack(fill='x')
        self.toolbar=toolbar; self.canvas.mpl_connect('button_press_event',self.click)
        details_frame=ttk.Frame(body);details_frame.pack(fill='x',padx=7,pady=4)
        self.details=tk.Text(details_frame,height=9,font=('monospace',9),wrap='word',bg='#f8fafc',fg='#0f172a')
        self.details.pack(side='left',fill='x',expand=True)
        scroll=ttk.Scrollbar(details_frame,orient='vertical',command=self.details.yview)
        scroll.pack(side='right',fill='y');self.details.configure(yscrollcommand=scroll.set)
        self.status=tk.StringVar(value='Click a point in any scatter plot to inspect its path. Zoom and pan with the toolbar.')
        ttk.Label(window,textvariable=self.status,padding=6).pack(fill='x')
        self.rank(); window.after(100,self.poll)

    def rank(self):
        metric=self.metric.get()
        selected_filter=self.filter.get()
        rows=[r for r in self.rows if selected_filter=='All evaluated cases'
              or (selected_filter=='Gaia' and 'gaia' in r['dataset'])
              or (selected_filter=='F1 regressions' and r['delta']['f1'] is not None and r['delta']['f1']<0)
              or (selected_filter=='ARI regressions' and min(r['delta']['ari_all'],r['delta']['ari_clean'] or 0)<0)]
        self.ordered=sorted(rows,key=lambda r:diag.severity(r) if metric=='worst' else
                            r['delta'][metric] if r['delta'][metric] is not None else float('inf'))
        self.listbox.delete(0,'end')
        for row in self.ordered:
            val=diag.severity(row) if metric=='worst' else row['delta'][metric]
            prefix='   n/a' if val is None else f'{val:+.3f}'
            self.listbox.insert('end',f"{prefix}  {row['dataset']}")
        if self.ordered:
            self.listbox.selection_set(0); self.choose()

    def choose(self,event=None):
        selection=self.listbox.curselection()
        if not selection:return
        self.revision+=1; self.playing=False
        self.recorded_row=self.ordered[selection[0]]
        self.recorded_data=load_case(self.out,self.recorded_row)
        self.restore()

    def restore(self):
        self.revision+=1
        self.data,self.row=self.recorded_data,self.recorded_row
        projections=['xy','pca']+[k for k in self.data if ' / ' in k]
        self.projection_box['values']=projections
        if self.projection.get() not in projections:self.projection.set('xy')
        self.selected,self.route=None,None
        self.anatomy=False
        self.compare.set(False)
        for key,value in self.row['settings'].items():self.entries[key].set(str(value))
        self.candidates['values']=[str(int(c)) for c in np.unique(self.data['base'][self.data['base']>=0])]
        if len(self.candidates['values']):self.cluster.set(self.candidates['values'][0])
        else:self.cluster.set('')
        self.scale.configure(to=max(1,self.frame_count()))
        self.comparison.set(self.metric_details(self.data,self.row))
        self.set_details('Click a point. Candidate cluster selects an alternative route to compare.\n'
                         'Each plotted coordinate refers to the same point across all four scatter views.\n'
                         'ARI is permutation invariant; seed-majority mismatch is only a diagnostic heuristic.')
        self.scrub(self.frame_count())

    def frame_count(self):
        return len(self.data['claims']) + (len(self.data.get('stage_labels',[])) if self.mode.get()=='Full pipeline' else 0)

    def metric_details(self,data,row):
        before,after=row['before'],row['after']
        lines=[f"{row['n']:,} points / {row['d']} features / {int((data['y']<0).sum())} true anomalies",
               'Before reassignment → visible run final']
        for name,key in [('ARI all','ari_all'),('ARI clean','ari_clean'),('Anomaly F1','f1'),('Precision','precision'),('Recall','recall')]:
            lines.append(f"{name}: {fmt(before[key])} → {fmt(after[key])}")
        lines.append(f"Anomaly TP/FP/FN: {before['TP']}/{before['FP']}/{before['FN']} → {after['TP']}/{after['FP']}/{after['FN']}")
        lines.append(f"Clusters: {len(np.unique(data['base'][data['base']>=0]))} → {len(np.unique(data['final'][data['final']>=0]))}")
        bad=data['y'][data['claims'][:,0].astype(int)]<0
        if bad.any():lines.append(f"First swallowed anomaly: claim {int(np.flatnonzero(bad)[0])+1}, cost {data['claims'][bad,2].min():.3g}")
        return '\n'.join(lines)

    def change_mode(self):
        self.playing=False;self.scale.configure(to=max(1,self.frame_count()));self.scrub(self.frame_count())

    def slider(self,value):
        if not self.setting_scale:self.anatomy=False;self.redraw()

    def scrub(self,value):
        self.setting_scale=True;self.count.set(value);self.setting_scale=False;self.redraw()

    def redraw(self):
        if not hasattr(self,'data'):return
        if self.anatomy:
            self.axes,self.ratios=draw_anatomy(self.fig,self.data,self.row,self.selected)
            self.canvas.draw_idle();return
        frame=int(self.count.get());full=self.mode.get()=='Full pipeline'
        stages=len(self.data.get('stage_labels',[])) if full else 0
        stage_index=frame if frame<stages else None
        count=max(0,frame-stages)
        self.claim_label.set(str(self.data['stage_names'][frame]) if stage_index is not None else f"{count:,}/{len(self.data['claims']):,} claims")
        reference=self.recorded_data if self.compare.get() else None
        self.axes=draw_dashboard(self.fig,self.data,self.row,count,self.projection.get(),self.selected,self.route,stage_index,full,reference)
        self.canvas.draw_idle()

    def toggle_play(self):
        self.anatomy=False
        self.playing=not self.playing
        if self.playing and self.count.get()>=self.frame_count():self.scrub(0)

    def toggle_anatomy(self):
        self.playing=False;self.anatomy=not self.anatomy;self.redraw()

    def poll(self):
        while not self.messages.empty():
            revision,kind,result=self.messages.get()
            self.busy=False;self.run_button.configure(state='normal')
            self.candidates.configure(state='readonly')
            if revision!=self.revision:continue
            if kind=='error':self.status.set(result)
            elif kind=='explain':
                text,self.route=result;self.set_details(text);self.status.set('Black = actual winning path. Orange dashed = hypothetical candidate route, ignoring competing ownership.');self.redraw()
            else:
                self.data,self.row=result
                self.anatomy=False
                self.compare.set(True)
                self.selected,self.route=None,None
                a,b=self.recorded_row['after'],self.row['after']
                self.comparison.set(f"Default → experiment\nARI: {fmt(a['ari_all'])} → {fmt(b['ari_all'])}\n"
                                    f"F1: {fmt(a['f1'])} → {fmt(b['f1'])}\nRecall: {fmt(a['recall'])} → {fmt(b['recall'])}\n\n"+
                                    self.metric_details(self.data,self.row))
                self.scale.configure(to=max(1,self.frame_count()));self.scrub(self.frame_count())
                self.status.set('Experiment complete; pipeline graph and source clusters were held fixed.')
        if self.playing and hasattr(self,'data'):
            frame=int(self.count.get());stages=len(self.data.get('stage_labels',[])) if self.mode.get()=='Full pipeline' else 0
            count=min(self.frame_count(),frame+(1 if frame<stages else max(1,len(self.data['claims'])//80)))
            self.scrub(count)
            if count==self.frame_count():self.playing=False
        self.window.after(120,self.poll)

    def job(self,kind,fn):
        if self.busy:return
        self.busy=True;self.playing=False;self.run_button.configure(state='disabled')
        self.candidates.configure(state='disabled')
        revision=self.revision
        def worker():
            try:self.messages.put((revision,kind,fn()))
            except Exception as e:self.messages.put((revision,'error',f'{type(e).__name__}: {e}'))
        threading.Thread(target=worker,daemon=True).start()

    def run_experiment(self):
        try:
            settings={k:float(v.get()) for k,v in self.entries.items()}
            if not all(np.isfinite(v) and v>=0 for v in settings.values()):raise ValueError('Use finite, nonnegative settings')
        except ValueError as e:self.status.set(str(e));return
        data,row=self.recorded_data,self.recorded_row
        self.status.set('Running only reassignment; the window remains responsive…')
        self.job('experiment',lambda:experiment(data,row,settings))

    def click(self,event):
        if self.busy:return
        if self.anatomy:
            if self.toolbar.mode or event.inaxes!=self.axes[0,0] or not len(self.data['claims']):return
            screen=event.inaxes.transData.transform(self.ratios)
            distances=np.sum((screen-[event.x,event.y])**2,axis=1);claim=int(np.argmin(distances))
            if distances[claim]>225:return
            self.selected=int(self.data['claims'][claim,0]);self.cluster.set(str(int(self.data['final'][self.selected])))
            self.explain();return
        if self.toolbar.mode or event.inaxes not in [self.axes[0,0],self.axes[0,1],self.axes[0,2],self.axes[1,0]]:return
        xy=self.data[self.projection.get()]
        screen=event.inaxes.transData.transform(xy)
        distances=np.sum((screen-[event.x,event.y])**2,axis=1);node=int(np.argmin(distances))
        if distances[node]>225:return
        self.selected=node
        if self.data['final'][node]>=0:self.cluster.set(str(int(self.data['final'][node])))
        self.explain()

    def explain(self):
        if self.selected is None:self.status.set('Select a point on a scatter plot first.');return
        node=self.selected;data,row=self.data,self.row
        cluster=int(self.cluster.get()) if self.cluster.get() else None
        self.status.set('Inspecting actual path and hypothetical alternative…')
        self.job('explain',lambda:inspect_point(data,row,node,cluster))

    def set_details(self,text):
        self.details.configure(state='normal');self.details.delete('1.0','end');self.details.insert('1.0',text);self.details.configure(state='disabled')

    def save(self):
        folder=self.out/'exports';folder.mkdir(exist_ok=True)
        name=self.row['dataset'].replace('/','__')+'_inspection'
        self.fig.savefig(folder/(name+'.png'),dpi=150)
        (folder/(name+'.txt')).write_text(self.details.get('1.0','end'))
        (folder/(name+'.json')).write_text(json.dumps(diag.finite_json(self.row),indent=2))
        self.status.set(f'Saved chart, settings, and point details to {folder}')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=diag.OUT)
    p.add_argument('--preview',type=Path);p.add_argument('--smoke',action='store_true')
    p.add_argument('--screenshot',type=Path)
    args=p.parse_args()
    rows=diag.load_rows(args.out)
    if not rows:raise SystemExit('No completed evaluations. Run reassignment_diagnostics.py evaluate first.')
    if args.preview:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        fig=Figure(figsize=(15,9));FigureCanvasAgg(fig)
        row=min(rows,key=diag.severity);draw_dashboard(fig,load_case(args.out,row),row)
        fig.savefig(args.preview,dpi=120);return
    import tkinter as tk
    root=tk.Tk();app=App(root,args.out)
    if args.smoke:
        import time
        def settle():
            until=time.monotonic()+20
            while app.busy and time.monotonic()<until:
                root.update();time.sleep(.03)
            root.update()
            assert not app.busy, 'UI background job exceeded smoke-test time limit'
        root.update();app.scrub(0);root.update();app.scrub(app.frame_count());root.update()
        if len(app.data['claims']):
            app.selected=int(app.data['claims'][0,0]);app.cluster.set(str(int(app.data['final'][app.selected])))
            app.explain();settle();assert 'Actual winning path' in app.details.get('1.0','end')
        app.entries['reach'].set('0');app.run_experiment();settle()
        assert len(app.data['claims'])==0
        app.restore();root.update()
        if args.screenshot:
            from PIL import ImageGrab
            x,y=root.winfo_rootx(),root.winfo_rooty()
            try:
                ImageGrab.grab(bbox=(x,y,x+root.winfo_width(),y+root.winfo_height())).save(args.screenshot)
            except OSError as e:
                print(f'Window capture unavailable ({e}); saving its dashboard canvas instead.',flush=True)
                app.fig.savefig(args.screenshot,dpi=120)
        print('PASS native GUI: stage scrub, actual point inspection, asynchronous zero-reach experiment, restore',flush=True)
        root.destroy();return
    root.mainloop()


if __name__=='__main__':main()
