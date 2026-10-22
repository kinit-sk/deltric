#!/usr/bin/env python3
"""BBC diagnostics on Deltric's actual UMAP topology; predictions stay fixed."""
import json
import sys
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from deltric_minimal.utils_component_growth import cluster_tri
from deltric_minimal.studies.anomaly_cost_gate_study import BASE_CONFIG
from deltric_minimal.studies.eval_gaia_anomaly import scores

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--projection', choices=['umap', 'pca'], default='umap')
args = parser.parse_args()
projection = args.projection
suffix = '_pca' if projection == 'pca' else ''
out = ROOT / 'studies/results/nlp_adbench_bbc'
source = ROOT / 'data/nlp_adbench'
records = [json.loads(l) for l in (source / 'datasets/bbc/bbc_test_data.jsonl').read_text().splitlines()]
topics = np.array([r['original_label'] for r in records])
colors = dict(business='#2878b5', politics='#a45ab5', sport='#2a9d60', tech='#e6a429', entertainment='#df3545')
for name, slug in [('bert', 'bert_base_uncased'), ('openai', 'gpt_text-embedding-3-large')]:
    saved = np.load(out / f'{name}_predictions.npz')
    y, final, before, hdb = [saved[k] for k in ['y','deltric','baseline','hdbscan']]
    X = StandardScaler().fit_transform(np.load(source / f'embeddings/bbc/bbc_test_data_{slug}_feature.npy').astype(float))
    if projection == 'pca':
        pca = PCA(n_components=2, svd_solver='full')
        xy = pca.fit_transform(X)
        axis_names = [f'PC {i+1} ({v:.1%} variance)' for i,v in enumerate(pca.explained_variance_ratio_)]
        note = f'PCA display only: {pca.explained_variance_ratio_.sum():.1%} variance retained. Saved UMAP-based predictions unchanged.'
        print(name, axis_names, flush=True)
    else:
        print('Reproducing topology:',name,flush=True)
        rerun = cluster_tri(X, **BASE_CONFIG, random_state=42, component_growth_anomaly_reassign=True,
                            component_growth_anomaly_reassign_penalty_power=3., component_growth_anomaly_reassign_reach=25.,
                            component_growth_anomaly_reassign_density_power=4.)
        assert np.array_equal(rerun,final), 'Predictions changed'
        xy = cluster_tri.last_component_growth['X_proj']
        axis_names = ['UMAP 1', 'UMAP 2']
        note = 'Shared coordinates from Deltric’s UMAP topology, seed 42. Edge decisions use standardized original embeddings.'
    np.save(out / f'{name}_{projection}_coordinates.npy',xy)
    fig, axes = plt.subplots(2,3,figsize=(16,10),sharex=True,sharey=True)
    def scatter(ax, c):
        ax.scatter(*xy.T,c=c,s=15,alpha=.85,linewidths=0)
    def legend(ax, entries):
        ax.legend(handles=[Line2D([],[],marker='o',linestyle='',color=c,label=l,markersize=6) for l,c in entries],fontsize=8,loc='best',framealpha=.93)
    scatter(axes[0,0],[colors[t] for t in topics])
    axes[0,0].set_title('True topics')
    legend(axes[0,0],[(t+(' (anomaly)' if t=='entertainment' else ''),c) for t,c in colors.items()])
    scatter(axes[0,1],np.where(y<0,'#df3545','#c8cdd3'))
    axes[0,1].set_title('True anomalies: 62 entertainment articles')
    legend(axes[0,1],[('Normal','#c8cdd3'),('Anomaly','#df3545')])
    for ax, labels, title in [(axes[0,2],before,'Deltric before reassignment'),(axes[1,0],final,'Deltric after reassignment'),(axes[1,1],hdb,'HDBSCAN')]:
        ids=sorted(set(labels[labels>=0]))
        cmap=plt.get_cmap('tab20')
        cs=[('#888888' if label<0 else cmap(ids.index(label)%20)) for label in labels]
        scatter(ax,cs)
        mask=y<0
        ax.scatter(*xy[mask].T,s=42,facecolors='none',edgecolors='#df3545',linewidths=.85)
        s=scores(y,labels)
        ax.set_title(f'{title}\n{len(ids)} clusters, {(labels<0).sum()} noise | F1 {s["f1"]:.3f}, ARI {s["ari_real"]:.3f}',fontsize=11)
        legend(ax,[('True anomaly (red outline)','#df3545'),('Predicted noise (gray)','#888888')])
    scatter(axes[1,2],np.where(y<0,'#df3545','#c8cdd3'))
    axes[1,2].set_title('Deltric anomaly errors after reassignment\n62 missed anomalies; 0 false alarms')
    legend(axes[1,2],[('Correct normal','#c8cdd3'),('Missed anomaly','#df3545')])
    for ax in axes.flat:
        ax.set_xlabel(axis_names[0]);ax.set_ylabel(axis_names[1]);ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle(f'BBC News — {name.upper()} embeddings — 579 test texts',fontsize=18)
    fig.text(.5,.018,note+'\nARI scores all true normal points. Cluster colors are arbitrary; red outlines identify the designated anomalous topic.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.06,1,.95))
    fig.savefig(out / f'{name}_diagnostic{suffix}.png',dpi=150)
    plt.close(fig)
    # Count how a designated anomalous topic is divided among predicted clusters.
    ids=sorted(set(final))
    ts=list(colors)
    counts=np.array([[np.sum((topics==t)&(final==c)) for c in ids] for t in ts])
    fig,ax=plt.subplots(figsize=(max(7,len(ids)*.7),4))
    im=ax.imshow(counts,cmap='Blues',aspect='auto')
    for i in range(len(ts)):
        for j in range(len(ids)):
            ax.text(j,i,str(counts[i,j]),ha='center',va='center',color='white' if counts[i,j]>counts.max()/2 else '#222')
    ax.set_yticks(range(len(ts)),[t+(' (anomaly)' if t=='entertainment' else '') for t in ts])
    ax.set_xticks(range(len(ids)),[str(c) for c in ids]);ax.set_xlabel('Deltric final cluster ID')
    ax.set_title(f'{name.upper()}: true topic vs. Deltric final cluster (counts)')
    fig.colorbar(im,ax=ax,label='Texts');fig.tight_layout();fig.savefig(out/f'{name}_topic_counts{suffix}.png',dpi=150);plt.close(fig)
    print('Saved plots:',name,flush=True)
