"""Render standalone figures from the causal audit's saved arrays."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent/'results/dimension_cause_audit'
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})


def plot_points(ax,x,labels):
    cluster_ids=list(np.unique(labels[labels>=0]))
    for label in np.unique(labels):
        m=labels==label
        color='#c5c8cb' if label<0 else plt.get_cmap('turbo')(cluster_ids.index(label)/max(1,len(cluster_ids)-1))
        ax.scatter(x[m,0],x[m,1],s=3,c=[color],linewidths=0,rasterized=True)
    ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])


fig,axes=plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
for row,name in enumerate(['hdbscan','chameleon_t5_8k']):
    base=np.load(OUT/(name+'__baseline__s42.npz'))
    fixed=np.load(OUT/(name+'__no_restore__s42.npz'))
    b=json.loads((OUT/(name+'__baseline__s42.json')).read_text())
    f=json.loads((OUT/(name+'__no_restore__s42.json')).read_text())
    x=base['X']
    for col,labels in enumerate([base['y'],base['final'],fixed['final']]):
        plot_points(axes[row,col],x,labels)
    axes[row,0].set_title(name+' — reference labels')
    axes[row,1].set_title(f"Current pipeline — ARI {b['stages']['final']['ari_clean']:.3f}\n{b['stages']['final']['clusters']} predicted clusters")
    axes[row,2].set_title(f"No edge restoration — ARI {f['stages']['final']['ari_clean']:.3f}\n{f['stages']['final']['clusters']} predicted clusters")
fig.suptitle('Restoring edges before pruning reinforces mistaken merges',fontsize=17)
fig.supxlabel('Same data, seeds, growth thresholds and reassignment. ARI excludes reference noise; gray points are noise.\nCluster colors are arbitrary within each panel. No parameter search.',fontsize=10)
fig.savefig(OUT/'restoration_ablation.png',dpi=170)
plt.close(fig)

names=['01_shapes_2d_compound_easy_3600_k6','02_shapes_2d_compound_harder_4800_k6',
       '11_v2_moons_2d_2500','varied_2d_5c','blobs_large_2d_20c','chameleon_t5_8k','hdbscan']
labels=['Compound shapes: easy','Compound shapes: harder','Moons','Varied density: 5 clusters',
        '20 blobs','Chameleon t5','HDBSCAN demonstration']
fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
for i,(name,label) in enumerate(zip(names,labels)):
    b=json.loads((OUT/(name+'__baseline__s42.json')).read_text())['stages']['final']['ari_clean']
    p=json.loads((OUT/(name+'__zero_pad_umap__s42.json')).read_text())['stages']['final']['ari_clean']
    ax.plot([b,p],[i,i],color='#c2c7ce',lw=3)
    ax.scatter(b,i,c='#147d92',s=65,label='Native 2D' if i==0 else None,zorder=3)
    ax.scatter(p,i,c='#d97732',s=65,marker='s',label='Same distances, zero coordinate + UMAP' if i==0 else None,zorder=3)
ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis();ax.set_xlim(0,1.04)
ax.set_xlabel('Adjusted Rand index (reference non-noise points)');ax.grid(axis='x',alpha=.2)
ax.set_title('The dimension branch changes results without adding information',fontsize=15,pad=15)
ax.legend(loc='lower center',bbox_to_anchor=(.5,-.3),frameon=False,ncol=2,fontsize=9)
fig.savefig(OUT/'dimension_branch.png',dpi=170)
plt.close(fig)
