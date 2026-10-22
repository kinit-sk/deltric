"""Build additional proof and figures from dimension_cause_audit outputs."""
from pathlib import Path
from collections import deque
import csv
import inspect
import json
import sys
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import plot_seed_component_recomputed_growth as stage
from plot_seed_component_metrics import _seed_components
OUT = ROOT/'studies/results/dimension_cause_audit'


def components(e, m, n):
    ee=e[m]
    g=coo_matrix((np.ones(2*len(ee)), (np.r_[ee[:,0],ee[:,1]],np.r_[ee[:,1],ee[:,0]])),shape=(n,n))
    return connected_components(g, directed=False)


def diagnostics(a):
    e,y,L,initial,pre,retained = (a[k] for k in ('edges','y','lengths','initial','pre','retained'))
    u,v=e.T
    label_density=[]
    for label in np.unique(y[y>=0]):
        inside=(y[u]==label)&(y[v]==label)
        points=np.flatnonzero(y==label)
        covered=np.unique(e[initial & inside])
        label_density.append(dict(label=int(label), n=len(points),
            within_edge_median=float(np.median(L[inside])),
            fraction_within_edges_seeded=float(initial[inside].mean()),
            seed_point_coverage=float(len(covered)/len(points))))
    ncomp, lab=components(e, pre, len(y))
    mixed=[]
    for c in range(ncomp):
        vals, counts=np.unique(y[(lab==c)&(y>=0)],return_counts=True)
        if len(vals)>1:
            mixed.append(dict(component=c, size=int((lab==c).sum()), noise=int(((lab==c)&(y<0)).sum()),
                truth_counts={str(v):int(n) for v,n in zip(vals,counts)}))
    # Enumerate noise components that connect at least two reference classes.
    _, nl=components(e,pre&(y[u]<0)&(y[v]<0),len(y))
    adjacency=[[] for _ in y]
    for i,(left,right) in enumerate(e):
        if pre[i]:
            adjacency[left].append((int(right),i)); adjacency[right].append((int(left),i))
    bridges=[]
    for c in np.unique(nl[y<0]):
        nodes=np.flatnonzero((nl==c)&(y<0))
        boundary={other for node in nodes for other,_ in adjacency[node] if y[other]>=0}
        if len({int(y[x]) for x in boundary})<2:
            continue
        start=min(boundary); parent={start:None}; q=deque([start]); end=None
        while q and end is None:
            node=q.popleft()
            for other,ei in adjacency[node]:
                if other in parent: continue
                if y[other]>=0 and y[other]==y[start]: continue
                if y[other]<0 and nl[other]!=c: continue
                parent[other]=(node,ei)
                if y[other]>=0:
                    end=other; break
                q.append(other)
        if end is None: continue
        path=[end]; ids=[]
        while parent[path[-1]] is not None:
            node,ei=parent[path[-1]]; path.append(node); ids.append(ei)
        path.reverse(); ids.reverse()
        bridges.append(dict(noise_component_size=len(nodes),
            attached_truth_labels=sorted({int(y[x]) for x in boundary}),
            path=path, path_labels=y[path].tolist(), edge_ids=ids,
            lengths=L[ids].tolist(), seed_edges=initial[ids].tolist()))
    protected=(a['seed_labels'][u]!=a['seed_labels'][v]) & (retained & ~pre)
    return dict(label_density=label_density, mixed_growth_components=sorted(mixed,key=lambda z:-z['size']),
                noise_bridges=sorted(bridges,key=lambda z:len(z['path'])),
                retained_restored_cross_seed=int(protected.sum()))


def trace_growth(name):
    a=np.load(OUT/(name+'__baseline__s42.npz'))
    e,L,P,initial,y=(a[k] for k in ('edges','lengths','projected','initial','y'))
    sl,sc,_=_seed_components(e,initial,len(y),10)
    source,offset=inspect.getsourcelines(stage._sequential_growth)
    accept_line=offset+next(i for i,s in enumerate(source) if 'qualifying_edges.add(int(edge_index))' in s)
    merge_line=offset+next(i for i,s in enumerate(source) if 'merge(seed_labels[u], seed_labels[v])' in s)
    accepted=[]; mergers=[]
    def trace(frame,event,arg):
        if frame.f_code is not stage._sequential_growth.__code__: return None
        if event=='line':
            z=frame.f_locals
            if frame.f_lineno==accept_line:
                accepted.append(dict(edge=int(z['edge_index']),round=int(z['round_index']),
                    root=int(z['root']),ratio=float(z['ratio']),projected_ratio=float(z['projected_ratio'])))
            elif frame.f_lineno==merge_line:
                u,v=int(z['u']),int(z['v']); f=z['find']; left,right=f(sl[u]),f(sl[v])
                if left!=right:
                    ly=set(y[list(z['members'][left])]) - {-1}
                    ry=set(y[list(z['members'][right])]) - {-1}
                    if ly and ry and len(ly|ry)>max(len(ly),len(ry)):
                        mergers.append(dict(edge=int(z['edge_index']),round=int(z['round_index']),
                            endpoints=[u,v],endpoint_truth=y[[u,v]].tolist(),
                            left_truth=sorted(int(x) for x in ly),right_truth=sorted(int(x) for x in ry),
                            left_size=len(z['members'][left]),right_size=len(z['members'][right])))
        return trace
    sys.settrace(trace)
    try:
        result=stage._sequential_growth(e,L,P,initial,sl,sc,10,np.ones(len(y)),1.05,1.35)
    finally:
        sys.settrace(None)
    assert np.array_equal(result['accepted_growth_mask'],a['accepted'])
    for m in mergers:
        m['acceptance_checks']=[r for r in accepted if r['edge']==m['edge'] and r['round']==m['round']]
    return dict(exact_growth_replay=True, wrong_cluster_mergers=mergers,accepted_checks=accepted)


def restoration_counterexample():
    # Two valid planar triangulated strips, each a large 12-point seed.
    seed_edges=[]
    for offset in (0,12):
        seed_edges.extend((offset+i,offset+i+d) for d in (1,2) for i in range(12-d))
    e=np.array(seed_edges+[(11,12),(10,13)])
    sl=np.repeat([0,1],12); candidate=np.zeros(len(e),bool);candidate[-2]=True
    cut,metrics=stage._gomory_hu_contracted_candidate_cut_mask(e,candidate,sl,cut_size=6,min_component_points=10)
    all_cross=sl[e[:,0]]!=sl[e[:,1]]
    cut2,_=stage._gomory_hu_contracted_candidate_cut_mask(e,all_cross,sl,cut_size=6,min_component_points=10)
    return dict(default_removed_edges=e[cut].tolist(), default_final_components=components(e,~cut,24)[0],
                all_cross_removed_edges=e[cut2].tolist(),all_cross_final_components=components(e,~cut2,24)[0])


def cut_inflation(name):
    import networkx as nx
    a=np.load(OUT/(name+'__baseline__s42.npz'))
    e,y,sl,pre,initial=(a[k] for k in ('edges','y','seed_labels','pre','initial'))
    _,sc,large=_seed_components(e,initial,len(y),10)
    _,pc=components(e,pre,len(y));u,v=e.T
    active=np.isin(pc,np.unique(pc[large[sl]]))
    added=~large[sl]&active
    restored=(pc[u]==pc[v])&~pre
    candidates={'before':a['accepted'],'after':a['accepted']|(restored&(added[u]|added[v]))}
    graphs={}
    for key,mask in candidates.items():
        g=nx.Graph();g.add_nodes_from(np.unique(sl))
        for left,right in e[mask]:
            left,right=int(sl[left]),int(sl[right])
            if left==right:continue
            if g.has_edge(left,right):g[left][right]['capacity']+=1
            else:g.add_edge(left,right,capacity=1)
        graphs[key]=g
    reps={}
    for label in np.unique(y[y>=0]):
        counts=np.bincount(sl[y==label]);reps[int(label)]=int(counts.argmax())
    rows=[]
    for l in reps:
        for r in reps:
            if l>=r or reps[l]==reps[r]:continue
            row=dict(left_truth=l,right_truth=r,left_seed=reps[l],right_seed=reps[r])
            for key,g in graphs.items():
                row[key]=float(nx.minimum_cut_value(g,reps[l],reps[r],capacity='capacity'))
            rows.append(row)
    return rows


def main():
    rows=[]; detail={}
    for p in sorted(OUT.glob('*__*.json')):
        r=json.loads(p.read_text()); rows.append(dict(dataset=r['dataset'],variant=r['variant'],seed=r['umap_seed'],
            n=r['n'],d=r['d'],ari=r['stages']['final']['ari'],ari_clean=r['stages']['final']['ari_clean'],
            seed_ari=r['stages']['seed']['ari_clean'],growth_ari=r['stages']['growth']['ari_clean'],
            gh_ari=r['stages']['gh']['ari_clean'],knn_fraction=r['knn_fraction'],
            forced_wrong_pairs=r['forced_wrong_pairs'],projected_unique_veto=r['projected_unique_veto']))
        if r['variant']=='baseline':
            detail[r['dataset']]=diagnostics(np.load(p.with_suffix('.npz')))
    with (OUT/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (OUT/'graph_diagnostics.json').write_text(json.dumps(detail,indent=2)+'\n')
    for name in ('hdbscan','chameleon_t5_8k'):
        (OUT/(name+'_growth_trace.json')).write_text(json.dumps(trace_growth(name),indent=2)+'\n')
    (OUT/'restoration_counterexample.json').write_text(json.dumps(restoration_counterexample(),indent=2)+'\n')
    (OUT/'cut_inflation.json').write_text(json.dumps(cut_inflation('hdbscan'),indent=2)+'\n')
    # Verify zero-padding invariance independently of aggregate score.
    padding={}
    for p in OUT.glob('*__zero_pad_none__s42.npz'):
        b=np.load(str(p).replace('__zero_pad_none__','__baseline__'));a=np.load(p)
        padding[p.stem.split('__')[0]]={k:bool(np.array_equal(a[k],b[k])) for k in
                                     ('edges','lengths','initial','pre','retained','final')}
        assert all(padding[p.stem.split('__')[0]].values())
    (OUT/'padding_invariance.json').write_text(json.dumps(padding,indent=2)+'\n')
    print('Wrote summaries, graph witnesses, traced growth and counterexample.',flush=True)


if __name__=='__main__':main()
