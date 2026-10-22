"""Pictures for verify_audit_fixes.py: baseline vs the two useful join fixes.

For each dataset writes
  {name}_compare.png  truth + baseline / join_valley / join_density2 final labels
  {name}_stages.gif   truth + baseline and join_density2 stepping through
                      seeds -> growth -> GH cut -> cost-gate reassignment
With --mode cluster writes the same pair under clustering/, coloured for
clustering diagnosis: every predicted cluster takes the colour of the truth
class it is matched to (Hungarian on the contingency table), so a merge shows
as two truth colours sharing one colour, a split as a hatched "extra" cluster.
Uses the cached geometry from verify_audit_fixes.py (never reads holdout).

Run: OMP_NUM_THREADS=1 .venv/bin/python studies/plot_audit_fixes.py --datasets hcv ...
"""
from pathlib import Path
import argparse
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_audit_fixes import VARIANTS, geometry, run_variant, score

OUT = Path(__file__).resolve().parent / 'results/verify_audit_fixes_viz'
SHOW = ['baseline', 'join_valley', 'join_density2']
ANIMATE = ['baseline', 'join_density2']
STAGES = [('seeds', 'Seeds'), ('growth', 'Growth + restore'),
          ('cut', 'Gomory-Hu cut'), ('final', 'Cost-gate reassignment')]
ERRORS = [('TP', '#2563eb', '*', 55, 'TP: anomaly detected'),
          ('FP', '#dc2626', 'x', 30, 'FP: normal flagged'),
          ('FN', '#f59e0b', '^', 38, 'FN: anomaly missed')]


def colors(labels):
    cmap = plt.get_cmap('tab20')
    # Stable, label-derived colours; large label ids wrap around the palette.
    return np.array([cmap((c * 7) % 20) for c in labels]).reshape(-1, 4)


def draw(ax, P, y, pred, title):
    ax.clear()
    ax.set_xticks([]); ax.set_yticks([])
    anom = y < 0
    flagged = pred < 0
    ax.scatter(*P[~flagged].T, s=4, c=colors(pred[~flagged]), linewidths=0)
    ax.scatter(*P[flagged & ~anom].T, s=4, c='#d4d4d4', linewidths=0)
    masks = dict(TP=anom & flagged, FP=~anom & flagged, FN=anom & ~flagged)
    for key, color, marker, size, _ in ERRORS:
        ax.scatter(*P[masks[key]].T, s=size if key != 'FP' else 12, c=color, marker=marker,
                   linewidths=.7 if marker == 'x' else 0, zorder=3)
    ax.set_title(title, fontsize=9)


def draw_truth(ax, P, y):
    ax.set_xticks([]); ax.set_yticks([])
    normal = y >= 0
    ax.scatter(*P[normal].T, s=4, c=colors(y[normal]) if len(set(y[normal])) > 1 else '#9ca3af',
               linewidths=0)
    ax.scatter(*P[~normal].T, s=55, c='#2563eb', marker='*', linewidths=0, zorder=3)
    ax.set_title(f'Ground truth\n{int(normal.sum())} normal, {int((~normal).sum())} anomalies '
                 f'({len(set(y[normal].tolist()))} classes)', fontsize=9)


def fmt(s):
    ari = '-' if s['ari_clean'] is None else f"{s['ari_clean']:.3f}"
    f1 = '-' if s['f1'] is None else f"{s['f1']:.3f}"
    return (f"ARI {ari}  F1 {f1}  k={s['clusters']}\n"
            f"precision {s['precision'] or 0:.2f}  recall {s['recall'] or 0:.2f}")


def legend(fig):
    handles = [Line2D([], [], marker='*', color='#2563eb', ls='', label='Truth anomaly'),
               Line2D([], [], marker='o', color='#7c3aed', ls='', label='Colour = predicted cluster')]
    handles += [Line2D([], [], marker=m, color=c, ls='', label=l) for _, c, m, _, l in ERRORS]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=8, frameon=False)


def plot(name):
    geo = geometry(name)
    P, y = geo['X_proj'][:, :2], geo['y']
    runs = {}
    for v in SHOW:
        stages = {}
        _, final = run_variant(geo, VARIANTS[v], stages)
        stages['final'] = final
        runs[v] = stages
    d = geo['X'].shape[1]
    coords = '2D UMAP of standardized features' if d > 2 else 'native 2D coordinates'

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.8))
    draw_truth(axes[0], P, y)
    for ax, v in zip(axes[1:], SHOW):
        draw(ax, P, y, runs[v]['final'], f"{v}\n{fmt(score(y, runs[v]['final']))}")
    fig.suptitle(f'{name} (n={len(y)}, d={d}) - final predictions; display: {coords}', fontsize=11)
    legend(fig)
    fig.tight_layout(rect=(0, .08, 1, .95))
    fig.savefig(OUT / f'{name}_compare.png', dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(1, 1 + len(ANIMATE), figsize=(6 * (1 + len(ANIMATE)), 6.2))
    draw_truth(axes[0], P, y)
    legend(fig)
    frames = [i for i in range(len(STAGES)) for _ in range(3)] + [len(STAGES) - 1] * 3

    def update(f):
        key, label = STAGES[frames[f]]
        for ax, v in zip(axes[1:], ANIMATE):
            pred = runs[v][key]
            draw(ax, P, y, pred, f"{v} - step {frames[f] + 1}/4: {label}\n{fmt(score(y, pred))}")
        fig.suptitle(f'{name} - pipeline stages ({coords})', fontsize=11)
        return []

    fig.tight_layout(rect=(0, .08, 1, .94))
    FuncAnimation(fig, update, frames=len(frames)).save(
        OUT / f'{name}_stages.gif', writer=PillowWriter(fps=1.5), dpi=80)
    plt.close(fig)
    print('wrote', name, flush=True)


CLASS = plt.get_cmap('tab20')
EXTRA = plt.get_cmap('Set3')


def match(y, pred):
    """Map predicted cluster -> matched truth class (or None) via Hungarian."""
    keep = (y >= 0) & (pred >= 0)
    classes, clusters = np.unique(y[y >= 0]), np.unique(pred[pred >= 0])
    table = np.array([[np.sum((y[keep] == c) & (pred[keep] == k)) for k in clusters]
                      for c in classes])
    rows, cols = linear_sum_assignment(-table) if table.size else ([], [])
    mapping = {int(clusters[j]): int(classes[i]) for i, j in zip(rows, cols) if table[i, j] > 0}
    # A cluster is a merge when a second truth class holds >= 20% of it.
    merges = sum(1 for j in range(len(clusters))
                 if (np.sort(table[:, j])[-2] if len(classes) > 1 else 0) >= .2 * table[:, j].sum() > 0)
    return mapping, classes, merges


def class_color(c, classes):
    return CLASS(int(np.searchsorted(classes, c)) % 20)


def draw_truth_clusters(ax, P, y):
    ax.clear(); ax.set_xticks([]); ax.set_yticks([])
    classes = np.unique(y[y >= 0])
    ok = y >= 0
    ax.scatter(*P[ok].T, s=4, linewidths=0,
               c=np.array([class_color(c, classes) for c in y[ok]]).reshape(-1, 4))
    ax.scatter(*P[~ok].T, s=10, c='black', marker='x', linewidths=.5)
    ax.set_title(f'Ground truth: {len(classes)} classes'
                 + (f', {int((~ok).sum())} noise (black x)' if (~ok).any() else ''), fontsize=9)


def draw_clusters(ax, P, y, pred, title):
    ax.clear(); ax.set_xticks([]); ax.set_yticks([])
    mapping, classes, merges = match(y, pred)
    clustered = pred >= 0
    extra_ids = sorted(set(np.unique(pred[clustered]).tolist()) - set(mapping))
    col = np.array([class_color(mapping[k], classes) if k in mapping
                    else EXTRA(extra_ids.index(k) % 12) for k in pred[clustered]]).reshape(-1, 4)
    extra = np.isin(pred, extra_ids)
    matched = clustered & ~extra
    ax.scatter(*P[matched].T, s=4, c=col[~extra[clustered]], linewidths=0)
    ax.scatter(*P[extra].T, s=9, c=col[extra[clustered]], marker='s', linewidths=.3,
               edgecolors='black')
    # Wrong cluster: a normal point sitting in a cluster matched to another class.
    wrong = np.array([k in mapping and mapping[k] != t for k, t in zip(pred, y)]) & (y >= 0)
    ax.scatter(*P[wrong].T, s=6, c='black', linewidths=0, zorder=3)
    ax.scatter(*P[~clustered].T, s=10, c='#9ca3af', marker='x', linewidths=.5)
    s = score(y, pred)
    ari = '-' if s['ari_clean'] is None else f"{s['ari_clean']:.3f}"
    ax.set_title(f"{title}\nARI {ari}  k={s['clusters']} (truth {len(classes)})\n"
                 f"merged clusters {merges}, extra clusters {len(extra_ids)}, "
                 f"wrong-cluster pts {int(wrong.sum())}, unassigned {int((~clustered).sum())}",
                 fontsize=8.5)


def cluster_legend(fig):
    fig.legend(handles=[
        Line2D([], [], marker='o', color='#1f77b4', ls='', label='Colour = matched truth class'),
        Line2D([], [], marker='s', color='#8dd3c7', mec='black', ls='', label='Extra cluster (split, unmatched)'),
        Line2D([], [], marker='o', color='black', ls='', label='Point in the wrong class\'s cluster (merge victim)'),
        Line2D([], [], marker='x', color='#9ca3af', ls='', label='Unassigned (-1)')],
        loc='lower center', ncol=4, fontsize=8, frameon=False)


CLUSTER_SHOW = ['baseline', 'join_valley', 'join_mutual']
CLUSTER_ANIMATE = ['baseline', 'join_valley']


def plot_clusters(name):
    out = OUT / 'clustering'
    out.mkdir(parents=True, exist_ok=True)
    geo = geometry(name)
    P, y = geo['X_proj'][:, :2], geo['y']
    runs = {}
    for v in CLUSTER_SHOW:
        stages = {}
        _, stages['final'] = run_variant(geo, VARIANTS[v], stages)
        runs[v] = stages
    d = geo['X'].shape[1]
    coords = '2D UMAP of standardized features' if d > 2 else 'native 2D coordinates'

    fig, axes = plt.subplots(1, 4, figsize=(21, 6))
    draw_truth_clusters(axes[0], P, y)
    for ax, v in zip(axes[1:], CLUSTER_SHOW):
        draw_clusters(ax, P, y, runs[v]['final'], v)
    fig.suptitle(f'{name} (n={len(y)}, d={d}) - final clustering; display: {coords}', fontsize=11)
    cluster_legend(fig)
    fig.tight_layout(rect=(0, .06, 1, .95))
    fig.savefig(out / f'{name}_compare.png', dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(1, 1 + len(CLUSTER_ANIMATE), figsize=(6.5 * (1 + len(CLUSTER_ANIMATE)), 6.4))
    draw_truth_clusters(axes[0], P, y)
    cluster_legend(fig)
    frames = [i for i in range(len(STAGES)) for _ in range(3)] + [len(STAGES) - 1] * 3

    def update(f):
        key, label = STAGES[frames[f]]
        for ax, v in zip(axes[1:], CLUSTER_ANIMATE):
            draw_clusters(ax, P, y, runs[v][key], f"{v} - step {frames[f] + 1}/4: {label}")
        fig.suptitle(f'{name} - pipeline stages ({coords})', fontsize=11)
        return []

    fig.tight_layout(rect=(0, .06, 1, .94))
    FuncAnimation(fig, update, frames=len(frames)).save(
        out / f'{name}_stages.gif', writer=PillowWriter(fps=1.5), dpi=80)
    plt.close(fig)
    print('wrote clustering', name, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', nargs='+', required=True)
    p.add_argument('--mode', choices=['anomaly', 'cluster'], default='anomaly')
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        plot(name) if args.mode == 'anomaly' else plot_clusters(name)


if __name__ == '__main__':
    main()
