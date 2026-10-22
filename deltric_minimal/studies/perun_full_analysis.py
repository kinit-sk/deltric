#!/usr/bin/env python3
"""Run DelTriC over every non-holdout dataset and render everything (Perun job).

Per dataset (``<name>`` = path under ``data/`` with ``/`` -> ``__``):

  stages/<name>.png|.npz|.json   the current default pipeline, i.e.
                                 ``plot_seed_component_recomputed_growth.py``
                                 with the ``run_plot_stages.sh`` defaults,
                                 run unchanged in a subprocess
  summary/<name>.png             truth | DelTriC final | anomaly TP/FP/FN/TN |
                                 HDBSCAN
  gifs/<name>_growth.gif         seeds -> every growth round -> Gomory-Hu cuts
                                 -> final components, next to ground truth
  gifs/<name>_anomaly.gif        cost-gate reclamation claim by claim against
                                 ground truth (as ``animate_anomaly_truth.py``)
  per_dataset/<name>.json        all metrics below

Metrics (y < 0 = anomaly, repository convention), for DelTriC after and
before the cost-gate reassignment, for the production ``cluster_tri`` preset
and for HDBSCAN: ARI over all points (``ari_all``), ARI over true normals only
(``ari_real``), anomaly precision / recall / F1, predicted anomaly and cluster
counts.  The top level gets ``metrics.csv``, ``failures.csv``,
``overview.png`` and a browsable ``index.html``.

The anomaly GIF replays the pipeline's own ``_assign_anomalies_cost_gate``
under a trace and checks that its final labels equal the saved
``final_prediction``; ``anomaly_replay_verified`` records the result.

Holdout data is never loaded: any path with a ``holdout`` component is
excluded before anything else happens.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "plot_seed_component_recomputed_growth.py"
MIN_EDGES = 10
COST_GATE = dict(penalty_power=0.5, reach=10.0, density_power=0.5, density_k=15,
                 density_clip=False)

# Mirrors run_plot_stages.sh defaults exactly; keep them in sync.
PIPELINE_ARGS = [
    "--dim-reduction", "umap", "--project-dim", "2",
    "--umap-n-epochs", "100", "--umap-n-neighbors", "15",
    "--seed-hard-limit", "0.0",
    "--hard-growth-limit", "1.05", "--projected-hard-growth-limit", "1.35",
    "--initial-relation", "union", "--component-growth-knn", "50",
    "--component-growth-min-edges", str(MIN_EDGES),
    "--hard-gate-mode", "strict", "--no-redundancy-pruning",
    "--gomory-hu-cut-size", "6", "--gomory-hu-min-component-points", "10",
    "--gomory-hu-hull-ratio-skip-threshold", "0.0",
    "--outlier-reassignment", "cost_gate",
    "--outlier-cost-gate-penalty-power", str(COST_GATE["penalty_power"]),
    "--outlier-cost-gate-reach", str(COST_GATE["reach"]),
    "--outlier-cost-gate-density-power", str(COST_GATE["density_power"]),
    "--outlier-cost-gate-density-k", str(COST_GATE["density_k"]),
    "--outlier-growth-hard-limit", "3.0",
    "--projected-outlier-growth-hard-limit", "3.0",
    "--gomory-hu-growth-edges-only", "--outlier-growth",
    "--no-outlier-cost-gate-density-clip",
    "--quiet", "--restore-intra-component-edges",
    "--projected-hard-limit", "--no-multiplicative-hard-limit",
]

NOISE = "#9e9e9e"
CONFUSION = [  # key, colour, marker, size, legend
    ("TN", "#c5c9ce", "o", 5, "TN: normal, accepted"),
    ("TP", "#2563eb", "*", 45, "TP: anomaly, detected"),
    ("FP", "#dc2626", "x", 36, "FP: normal, flagged"),
    ("FN", "#f59e0b", "^", 36, "FN: anomaly, missed"),
]


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------

def discover(data_dir: Path, derived_dir: Path) -> list[tuple[str, Path]]:
    items = []
    for path in sorted(data_dir.rglob("*.npz")):
        rel = path.relative_to(data_dir)
        if "holdout" in rel.parts or "source" in rel.parts:
            continue
        items.append(("__".join(rel.with_suffix("").parts), path))
    items += materialize_bbc(data_dir / "nlp_adbench", derived_dir)
    assert not any("holdout" in p.parts for _, p in items)
    names = [name for name, _ in items]
    assert len(set(names)) == len(names), "dataset names must be unique"
    return items


def materialize_bbc(source: Path, derived_dir: Path) -> list[tuple[str, Path]]:
    """BBC test embeddings -> repo-format NPZ (same labels as eval_nlp_adbench)."""
    jsonl = source / "datasets/bbc/bbc_test_data.jsonl"
    if not jsonl.exists():
        return []
    records = [json.loads(line) for line in jsonl.read_text().splitlines() if line]
    names = sorted({r["original_label"] for r in records if r["label"] == 0})
    mapping = {name: i for i, name in enumerate(names)}
    y = np.array([-1 if r["label"] else mapping[r["original_label"]] for r in records])
    out = []
    for tag, slug in [("bert", "bert_base_uncased"), ("openai", "gpt_text-embedding-3-large")]:
        feature = source / f"embeddings/bbc/bbc_test_data_{slug}_feature.npy"
        if not feature.exists():
            continue
        target = derived_dir / f"nlp_adbench_bbc_{tag}.npz"
        if not target.exists():
            X = np.load(feature, allow_pickle=False).astype(np.float64)
            assert len(X) == len(y)
            derived_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, X=X, y=y)
        out.append((f"nlp_adbench__bbc_test_{tag}", target))
    return out


def family(name: str) -> str:
    base = name.split("__")[0]
    if name.startswith(("testing__", "other__", "nlp_adbench__")):
        return "__".join(name.split("__")[:2])
    for prefix in ("adbench", "odds", "gaia", "text_support_tickets"):
        if name.startswith(prefix):
            return prefix
    return base


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def scores(y: np.ndarray, pred: np.ndarray) -> dict:
    from sklearn.metrics import adjusted_rand_score
    true_anom, pred_anom = y < 0, pred < 0
    tp = int(np.count_nonzero(true_anom & pred_anom))
    fp = int(np.count_nonzero(~true_anom & pred_anom))
    fn = int(np.count_nonzero(true_anom & ~pred_anom))
    tn = int(np.count_nonzero(~true_anom & ~pred_anom))
    real = ~true_anom
    return dict(
        ari_all=float(adjusted_rand_score(y, pred)),
        ari_real=float(adjusted_rand_score(y[real], pred[real])) if real.any() else float("nan"),
        precision=tp / (tp + fp) if tp + fp else 0.0,
        recall=tp / (tp + fn) if tp + fn else float("nan"),
        f1=2 * tp / (2 * tp + fp + fn) if tp + fn else float("nan"),
        TP=tp, FP=fp, FN=fn, TN=tn,
        n_pred_anomaly=int(pred_anom.sum()),
        n_clusters=int(len(np.unique(pred[pred >= 0]))),
    )


# ---------------------------------------------------------------------------
# replay helpers
# ---------------------------------------------------------------------------

def components(n: int, edge_keys: np.ndarray, mask: np.ndarray) -> np.ndarray:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    e = edge_keys[mask]
    graph = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def trace_cost_gate(module, base, edge_keys, lengths, core, max_frames):
    """Snapshots of accepted claims from the pipeline's own cost-gate function."""
    snapshots = [(0, base.copy())]
    stride = max(1, math.ceil(np.count_nonzero(base < 0) / max(1, max_frames - 1)))
    observed = 0
    code = module._assign_anomalies_cost_gate.__code__

    def on_line(frame, event, arg):
        nonlocal observed
        if event == "line":
            diag = frame.f_locals.get("diagnostics")
            count = diag.get("n_claims", 0) if diag is not None else 0
            if count > observed:
                observed = count
                if count % stride == 0:
                    snapshots.append((count, frame.f_locals["owner"].copy()))
        return on_line

    previous = sys.gettrace()
    try:
        sys.settrace(lambda f, e, a: on_line if f.f_code is code else None)
        final, diag = module._assign_anomalies_cost_gate(
            base, edge_keys, lengths, len(base), COST_GATE["penalty_power"],
            COST_GATE["reach"], core_distance=core,
            density_power=COST_GATE["density_power"],
            density_clip=COST_GATE["density_clip"],
        )
    finally:
        sys.settrace(previous)
    if snapshots[-1][0] != diag["n_claims"]:
        snapshots.append((diag["n_claims"], final.copy()))
    return snapshots, final, diag


def pick(n_items: int, limit: int) -> list[int]:
    if n_items <= limit:
        return list(range(n_items))
    return sorted(set(np.linspace(0, n_items - 1, limit).round().astype(int).tolist()))


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------

def cluster_colours(labels: np.ndarray) -> dict:
    import matplotlib.pyplot as plt
    ids = np.unique(labels[labels >= 0])
    cmap = plt.get_cmap("tab20" if len(ids) <= 20 else "turbo", max(len(ids), 1))
    return {int(c): cmap(i) for i, c in enumerate(ids)}


def setup_ax(ax, P):
    span = np.maximum(np.ptp(P, axis=0), 1e-9)
    ax.set_xlim(P[:, 0].min() - .04 * span[0], P[:, 0].max() + .04 * span[0])
    ax.set_ylim(P[:, 1].min() - .04 * span[1], P[:, 1].max() + .04 * span[1])
    ax.set_aspect("auto")  # UMAP axes carry no metric meaning
    ax.set_xticks([]); ax.set_yticks([])


def draw_labels(ax, P, labels, title, size=5):
    colours = cluster_colours(labels)
    noise = labels < 0
    ax.scatter(*P[noise].T, s=size * 1.6, c=NOISE, marker="x", linewidths=.6, alpha=.8)
    if (~noise).any():
        ax.scatter(*P[~noise].T, s=size, linewidths=0,
                   c=np.array([colours[int(c)] for c in labels[~noise]]).reshape(-1, 4))
    ax.set_title(title, fontsize=10)


def draw_confusion(ax, P, y, pred, title_prefix=""):
    s = scores(y, pred)
    masks = dict(TN=(y >= 0) & (pred >= 0), TP=(y < 0) & (pred < 0),
                 FP=(y >= 0) & (pred < 0), FN=(y < 0) & (pred >= 0))
    for z, (key, colour, marker, size, _) in enumerate(CONFUSION):
        ax.scatter(*P[masks[key]].T, s=size, c=colour, marker=marker,
                   linewidths=.8 if marker == "x" else 0, zorder=z + 1)
    ax.set_title(f"{title_prefix}TP {s['TP']}  FP {s['FP']}  FN {s['FN']}  TN {s['TN']}\n"
                 f"P {s['precision']:.3f}  R {s['recall']:.3f}  F1 {s['f1']:.3f}", fontsize=10)


def confusion_legend(fig):
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker=m, color=c, linestyle="", label=l, markersize=8)
                        for _, c, m, _, l in CONFUSION],
               loc="lower center", ncol=4, fontsize=9, frameon=False)


def render_summary(out, name, P, y, final, hdb, m):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 13))
    for ax in axes.flat:
        setup_ax(ax, P)
    draw_labels(axes[0, 0], P, y, f"ground truth ({int((y < 0).sum())} anomalies, x)")
    d = m["deltric"]
    draw_labels(axes[0, 1], P, final,
                f"DelTriC final: {d['n_clusters']} clusters, {d['n_pred_anomaly']} noise\n"
                f"ARI all {d['ari_all']:.3f}  ARI normals {d['ari_real']:.3f}")
    draw_confusion(axes[1, 0], P, y, final, "DelTriC anomalies: ")
    h = m["hdbscan"]
    draw_labels(axes[1, 1], P, hdb,
                f"HDBSCAN (mcs=10): {h['n_clusters']} clusters, {h['n_pred_anomaly']} noise\n"
                f"ARI all {h['ari_all']:.3f}  F1 {h['f1']:.3f}")
    confusion_legend(fig)
    fig.suptitle(f"{name}   n={m['n']} d={m['d']}   (2-D UMAP display; decisions in "
                 "standardized original space)", fontsize=12)
    fig.tight_layout(rect=(0, .03, 1, .97))
    fig.savefig(out, dpi=110)
    plt.close(fig)


def render_growth_gif(out, name, P, y, st, bridge_base, gh_pruned, base_pred, final,
                      max_frames, fps, dpi):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.collections import LineCollection

    edge_keys = st["edge_keys"]
    n = len(P)
    seg = np.stack([P[edge_keys[:, 0]], P[edge_keys[:, 1]]], axis=1)
    seed_labels, seed_edges = st["seed_labels"], st["seed_edge_counts"]
    large = st["initially_large_seed_components"].astype(bool)
    initial = st["initial_mask"].astype(bool)
    accepted, step = st["accepted_growth_mask"].astype(bool), st["accepted_growth_step"]
    seen_step = st["first_seen_step"]
    n_rounds = int(st["growth_round_count"])

    # Stable colour per growing component: that of the largest initially large
    # seed it contains.
    large_ids = np.flatnonzero(large)
    order = large_ids[np.argsort(-seed_edges[large_ids], kind="stable")]
    cmap = plt.get_cmap("tab20")
    seed_colour = {int(s): cmap(i % 20) for i, s in enumerate(order)}
    point_large = large[seed_labels]

    def growth_state(r):
        mask = initial | (accepted & (step <= r))
        cc = components(n, edge_keys, mask)
        colour = np.tile(np.array([.62, .62, .62, .5]), (n, 1))
        idx = np.flatnonzero(point_large)
        if len(idx):
            weight = seed_edges[seed_labels[idx]]
            o = idx[np.argsort(-weight, kind="stable")]
            ccs, first = np.unique(cc[o], return_index=True)
            key = dict(zip(ccs.tolist(), seed_labels[o[first]].tolist()))
            active = np.isin(cc, ccs)
            colour[active] = [seed_colour[key[c]] for c in cc[active]]
        edge_colour = colour[edge_keys[:, 0]]
        same = cc[edge_keys[:, 0]] == cc[edge_keys[:, 1]]
        active_edge = mask & same & (edge_colour[:, 3] == 1.0)
        return colour, edge_colour, active_edge

    frames = [("seed", -1)]
    frames += [("round", r) for r in (np.array(pick(n_rounds, max(1, max_frames - 4))) if n_rounds else [])]
    frames += [("gh", None), ("final", None)]
    sequence = [frames[0]] * 6 + frames[1:-2] + [frames[-2]] * 8 + [frames[-1]] * 12

    fig, (ax_t, ax) = plt.subplots(1, 2, figsize=(15, 7.6))
    for a in (ax_t, ax):
        setup_ax(a, P)
    draw_labels(ax_t, P, y, "ground truth (fixed reference)")
    fig.suptitle(f"{name} — component growth (n={n})", fontsize=12)
    fig.text(.5, .02, "grey lines: Delaunay graph   colour: growing components (seeded by large "
             "seed components)   black: accepted this round   red: evaluated & rejected so far   "
             "magenta dashed: Gomory-Hu cut", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, .95))
    background = LineCollection(seg, colors="#d8d8d8", linewidths=.25, alpha=.35, zorder=1)
    ax.add_collection(background)
    dynamic = []

    def update(i):
        for artist in dynamic:
            artist.remove()
        dynamic.clear()
        kind, r = sequence[i]
        if kind in ("seed", "round"):
            colour, edge_colour, active_edge = growth_state(r)
            rejected = (seen_step >= 0) & (seen_step <= r) & ~(accepted & (step <= r))
            dynamic.append(ax.add_collection(LineCollection(
                seg[rejected], colors="#dc2626", linewidths=.5, alpha=.45, zorder=2)))
            dynamic.append(ax.add_collection(LineCollection(
                seg[active_edge], colors=edge_colour[active_edge], linewidths=.8, zorder=3)))
            new = accepted & (step == r)
            dynamic.append(ax.add_collection(LineCollection(
                seg[new], colors="black", linewidths=1.3, zorder=4)))
            dynamic.append(ax.scatter(*P.T, s=5, c=colour, linewidths=0, zorder=5))
            n_active = len(np.unique(colour[colour[:, 3] == 1.0], axis=0))
            title = ("initial seed graph" if kind == "seed" else
                     f"growth round {r + 1}/{n_rounds}: +{int(new.sum())} edges")
            ax.set_title(f"{title}\n{n_active} growing components, "
                         f"{int((colour[:, 3] == 1.0).sum())}/{n} points covered", fontsize=10)
        else:
            labels = base_pred if kind == "gh" else final
            colours = cluster_colours(labels)
            colour = np.array([colours.get(int(c), (.62, .62, .62, .5)) for c in labels])
            keep = bridge_base & ~gh_pruned
            same = (labels[edge_keys[:, 0]] == labels[edge_keys[:, 1]]) & (labels[edge_keys[:, 0]] >= 0)
            m = keep & same
            dynamic.append(ax.add_collection(LineCollection(
                seg[m], colors=colour[edge_keys[m, 0]], linewidths=.8, zorder=3)))
            if kind == "gh":
                dynamic.append(ax.add_collection(LineCollection(
                    seg[gh_pruned], colors="#d946ef", linewidths=1.6, linestyles="--", zorder=4)))
            dynamic.append(ax.scatter(*P.T, s=5, c=colour, linewidths=0, zorder=5))
            k = len(colours)
            ax.set_title(
                f"Gomory-Hu: {int(gh_pruned.sum())} edges cut -> {k} clusters, "
                f"{int((labels < 0).sum())} noise" if kind == "gh" else
                f"final after cost-gate reassignment: {k} clusters, {int((labels < 0).sum())} noise",
                fontsize=10)
        return dynamic

    try:
        FuncAnimation(fig, update, frames=len(sequence), blit=False).save(
            out, writer=PillowWriter(fps=fps), dpi=dpi)
    finally:
        plt.close(fig)
    return len(sequence)


def render_anomaly_gif(out, name, P, y, snapshots, base, n_claims, fps, dpi):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.8))
    for ax in axes:
        setup_ax(ax, P)
    truth = y < 0
    axes[0].scatter(*P[~truth].T, s=5, c="#c5c9ce", linewidths=0)
    axes[0].scatter(*P[truth].T, s=40, c="#2563eb", marker="*", linewidths=0, zorder=3)
    axes[0].set_title(f"ground truth: {int(truth.sum())} anomalies (blue stars)", fontsize=10)
    colours = cluster_colours(base)
    confusion_legend(fig)
    fig.suptitle(f"{name} — cost-gate anomaly reclamation against truth\n"
                 f"p={COST_GATE['penalty_power']:g}, reach={COST_GATE['reach']:g}, "
                 f"density power={COST_GATE['density_power']:g}, k={COST_GATE['density_k']}",
                 fontsize=12)
    fig.tight_layout(rect=(0, .06, 1, .9))
    sequence = [0] * 6 + list(range(1, len(snapshots))) + [len(snapshots) - 1] * 14
    dynamic = []

    def update(i):
        for artist in dynamic:
            artist.remove()
        dynamic.clear()
        count, labels = snapshots[sequence[i]]
        noise = labels < 0
        reclaimed = (base < 0) & ~noise
        if (~noise).any():
            dynamic.append(axes[1].scatter(
                *P[~noise].T, s=6, linewidths=0,
                c=np.array([colours[int(c)] for c in labels[~noise]]).reshape(-1, 4)))
        dynamic.append(axes[1].scatter(*P[noise].T, s=18, c="#737373", marker="x", linewidths=.7))
        dynamic.append(axes[1].scatter(*P[reclaimed].T, s=28, facecolors="none",
                                       edgecolors="black", linewidths=.6, zorder=3))
        axes[1].set_title(f"DelTriC clusters ({len(colours)}); rings = reclaimed\n"
                          f"reclaimed {count}/{n_claims}, {int(noise.sum())} still noise",
                          fontsize=10)
        before = list(axes[2].collections)
        draw_confusion(axes[2], P, y, labels)
        dynamic.extend(c for c in axes[2].collections if c not in before)
        return dynamic

    try:
        FuncAnimation(fig, update, frames=len(sequence), blit=False).save(
            out, writer=PillowWriter(fps=fps), dpi=dpi)
    finally:
        plt.close(fig)
    return len(sequence)


# ---------------------------------------------------------------------------
# one dataset
# ---------------------------------------------------------------------------

def run_one(job):
    name, path, args = job
    out = Path(args.out)
    result_path = out / "per_dataset" / f"{name}.json"
    if args.resume and result_path.exists():
        return json.loads(result_path.read_text())
    import matplotlib
    matplotlib.use("Agg")
    from sklearn.preprocessing import StandardScaler
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT.parent))
    import plot_seed_component_recomputed_growth as pipeline
    from hdbscan_comparison import fit_hdbscan

    record = dict(dataset=name, family=family(name), path=str(path), status="ok")
    started = time.perf_counter()
    try:
        stage_png = out / "stages" / f"{name}.png"
        if not (args.resume and stage_png.with_suffix(".json").exists()):
            t0 = time.perf_counter()
            pipeline_args = list(PIPELINE_ARGS)
            pipeline_args[pipeline_args.index("--umap-n-epochs") + 1] = args.umap_n_epochs
            proc = subprocess.run(
                [sys.executable, "-u", str(PIPELINE), "--data", str(path),
                 "--out", str(stage_png), *pipeline_args],
                cwd=ROOT, capture_output=True, text=True, timeout=args.timeout)
            (out / "logs").mkdir(parents=True, exist_ok=True)
            (out / "logs" / f"{name}.log").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
            if proc.returncode:
                raise RuntimeError(f"pipeline exit {proc.returncode}: {proc.stderr[-2000:]}")
            record["pipeline_seconds"] = time.perf_counter() - t0
        stage_json = json.loads(stage_png.with_suffix(".json").read_text())
        st = dict(np.load(stage_png.with_suffix(".npz"), allow_pickle=False))

        with np.load(path, allow_pickle=False) as data:
            X = StandardScaler().fit_transform(np.asarray(data["X"], dtype=np.float64))
            y = np.asarray(data["y"], dtype=np.int64)
        n, d = X.shape
        P = np.asarray(st["X_proj"], dtype=np.float64)[:, :2]
        edge_keys = st["edge_keys"].astype(np.int64)
        final = st["final_prediction"].astype(np.int64)
        retained = st["final_retained_mask"].astype(bool)
        record.update(n=n, d=d, k_true=int(len(np.unique(y[y >= 0]))),
                      n_true_anomaly=int((y < 0).sum()))

        # Pre-reassignment labels exactly as the pipeline derives them, then a
        # traced replay of its cost gate, verified against the saved result.
        comp, comp_points = pipeline._final_graph_components(edge_keys, retained, n)
        base = pipeline._final_component_prediction(edge_keys, retained, comp, comp_points, MIN_EDGES)
        lengths = np.linalg.norm(X[edge_keys[:, 0]] - X[edge_keys[:, 1]], axis=1)
        core = pipeline._knn_core_distance(X, COST_GATE["density_k"])
        snapshots, replay, diag = trace_cost_gate(pipeline, base, edge_keys, lengths, core,
                                                  args.max_frames)
        record["anomaly_replay_verified"] = bool(np.array_equal(replay, final))

        t0 = time.perf_counter()
        hdb = np.asarray(fit_hdbscan(X, min_cluster_size=10, min_samples=None,
                                     cluster_selection_method="eom"), dtype=np.int64)
        record["hdbscan_seconds"] = time.perf_counter() - t0
        record["deltric"] = scores(y, final)
        record["deltric_pre_reassign"] = scores(y, base)
        record["hdbscan"] = scores(y, hdb)
        preds = dict(y=y, deltric=final, deltric_pre_reassign=base, hdbscan=hdb, X_proj=P)

        if args.production:
            from deltric_minimal.studies.anomaly_cost_gate_study import BASE_CONFIG
            from utils_component_growth import cluster_tri
            t0 = time.perf_counter()
            prod_config = dict(BASE_CONFIG, umap_n_epochs=(
                args.umap_n_epochs if args.umap_n_epochs == "auto" else int(args.umap_n_epochs)))
            prod = np.asarray(cluster_tri(
                X, **prod_config, random_state=42,
                component_growth_anomaly_reassign=True,
                component_growth_anomaly_reassign_penalty_power=3.0,
                component_growth_anomaly_reassign_reach=25.0,
                component_growth_anomaly_reassign_density_power=4.0,
            ), dtype=np.int64)
            record["production_seconds"] = time.perf_counter() - t0
            record["production"] = scores(y, prod)
            preds["production"] = prod

        record.update(
            growth_rounds=int(st["growth_round_count"]),
            accepted_growth_edges=stage_json["accepted_growth_edge_count"],
            growth_added_points=stage_json["growth_added_point_count"],
            gomory_hu_cut_edges=stage_json["gomory_hu_pruned_edge_count"],
            large_seed_components=stage_json["large_seed_component_count"],
            cost_gate_claims=int(diag["n_claims"]),
            n_edges=int(len(edge_keys)),
        )
        (out / "predictions").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "predictions" / f"{name}.npz", **preds)

        for sub in ("summary", "gifs"):
            (out / sub).mkdir(parents=True, exist_ok=True)
        render_summary(out / "summary" / f"{name}.png", name, P, y, final, hdb, record)
        if not args.no_gifs:
            render_growth_gif(out / "gifs" / f"{name}_growth.gif", name, P, y, st,
                              st["bridge_base_mask"].astype(bool),
                              st["gomory_hu_pruned_mask"].astype(bool), base, final,
                              args.max_frames, args.fps, args.dpi)
            render_anomaly_gif(out / "gifs" / f"{name}_anomaly.gif", name, P, y, snapshots,
                               base, int(diag["n_claims"]), args.fps, args.dpi)
    except Exception as error:  # keep the batch going; record why
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        record["traceback"] = traceback.format_exc()
    record["seconds"] = time.perf_counter() - started
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if record["status"] == "ok":
        result_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"[{record['status']}] {name} {record['seconds']:.0f}s "
          + (f"ARI={record['deltric']['ari_all']:.3f} F1={record['deltric']['f1']:.3f}"
             if record["status"] == "ok" else record["error"][:300]), flush=True)
    return record


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------

METHODS = ("deltric", "deltric_pre_reassign", "production", "hdbscan")
METRIC_KEYS = ("ari_all", "ari_real", "precision", "recall", "f1", "n_pred_anomaly", "n_clusters")


def flatten(r: dict) -> dict:
    row = {k: r.get(k) for k in ("dataset", "family", "n", "d", "k_true", "n_true_anomaly",
                                 "growth_rounds", "large_seed_components", "accepted_growth_edges",
                                 "growth_added_points", "gomory_hu_cut_edges", "cost_gate_claims",
                                 "anomaly_replay_verified", "pipeline_seconds", "seconds")}
    for method in METHODS:
        for key in METRIC_KEYS:
            row[f"{method}_{key}"] = (r.get(method) or {}).get(key)
    return row


def aggregate(out: Path, records: list[dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ok = [r for r in records if r["status"] == "ok"]
    failed = [r for r in records if r["status"] != "ok"]
    rows = [flatten(r) for r in ok]
    if rows:
        with (out / "metrics.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    with (out / "failures.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "error"])
        writer.writerows([[r["dataset"], r.get("error", "")] for r in failed])

    # Family means.
    fams = sorted({r["family"] for r in ok})
    fam_rows = []
    for fam in fams:
        rs = [r for r in ok if r["family"] == fam]
        row = dict(family=fam, datasets=len(rs))
        for method in ("deltric", "production", "hdbscan"):
            for key in ("ari_all", "ari_real", "f1"):
                vals = [r[method][key] for r in rs if method in r and np.isfinite(r[method][key])]
                row[f"{method}_{key}"] = float(np.mean(vals)) if vals else None
        fam_rows.append(row)
    if fam_rows:
        with (out / "family_means.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(fam_rows[0]))
            writer.writeheader(); writer.writerows(fam_rows)

    if ok:
        fig, axes = plt.subplots(1, 3, figsize=(19, 6.4))
        for ax, other, key, label in [
            (axes[0], "hdbscan", "ari_all", "ARI (all points)"),
            (axes[1], "production", "ari_all", "ARI (all points)"),
            (axes[2], "hdbscan", "f1", "anomaly F1"),
        ]:
            pts = [(r[other][key], r["deltric"][key], r["n_true_anomaly"] > 0) for r in ok
                   if other in r and np.isfinite(r[other][key]) and np.isfinite(r["deltric"][key])]
            if pts:
                a = np.array(pts, dtype=float)
                ax.scatter(a[a[:, 2] == 0, 0], a[a[:, 2] == 0, 1], s=14, c="#64748b", label="no true anomalies")
                ax.scatter(a[a[:, 2] == 1, 0], a[a[:, 2] == 1, 1], s=14, c="#2563eb", label="with anomalies")
                ax.plot([-.1, 1], [-.1, 1], c="black", lw=.7)
                wins = int((a[:, 1] > a[:, 0] + 1e-9).sum())
                ax.set_title(f"{label}: DelTriC vs {other}  (DelTriC better on {wins}/{len(a)})")
            ax.set_xlabel(other); ax.set_ylabel("DelTriC (current pipeline)")
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "overview.png", dpi=110)
        plt.close(fig)
        write_index(out, ok, failed, fam_rows)


def write_index(out: Path, ok: list[dict], failed: list[dict], fam_rows: list[dict]):
    def fmt(v):
        return "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else (
            f"{v:.3f}" if isinstance(v, float) else html.escape(str(v)))

    head = ("dataset", "n", "d", "k", "anom", "DelTriC ARI", "ARI normals", "P", "R", "F1",
            "pre-reassign F1", "production ARI", "production F1", "HDBSCAN ARI", "HDBSCAN F1",
            "rounds", "GH cuts", "files")
    lines = []
    for r in sorted(ok, key=lambda r: (r["family"], r["dataset"])):
        dl, pre, hd, pr = r["deltric"], r["deltric_pre_reassign"], r["hdbscan"], r.get("production", {})
        name = r["dataset"]
        links = (f'<a href="summary/{name}.png">summary</a> <a href="stages/{name}.png">stages</a> '
                 f'<a href="gifs/{name}_growth.gif">growth</a> <a href="gifs/{name}_anomaly.gif">anomaly</a>')
        cells = [name, r["n"], r["d"], r["k_true"], r["n_true_anomaly"], dl["ari_all"], dl["ari_real"],
                 dl["precision"], dl["recall"], dl["f1"], pre["f1"], pr.get("ari_all"), pr.get("f1"),
                 hd["ari_all"], hd["f1"], r["growth_rounds"], r["gomory_hu_cut_edges"]]
        lines.append("<tr>" + "".join(f"<td>{fmt(c)}</td>" for c in cells) + f"<td>{links}</td></tr>")
    fam_html = "".join(
        "<tr>" + "".join(f"<td>{fmt(row.get(k))}</td>" for k in
                         ("family", "datasets", "deltric_ari_all", "production_ari_all",
                          "hdbscan_ari_all", "deltric_f1", "production_f1", "hdbscan_f1")) + "</tr>"
        for row in fam_rows)
    fail_html = "".join(f"<li><b>{html.escape(r['dataset'])}</b>: {html.escape(r.get('error', ''))}</li>"
                        for r in failed)
    (out / "index.html").write_text(f"""<!doctype html><meta charset="utf-8"><title>DelTriC full analysis</title>
<style>body{{font:13px system-ui;margin:16px}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:2px 6px;text-align:right}}
td:first-child,th:first-child{{text-align:left}}th{{background:#eee;position:sticky;top:0;cursor:pointer}}</style>
<h1>DelTriC full analysis ({len(ok)} datasets ok, {len(failed)} failed)</h1>
<p>Current pipeline = plot_seed_component_recomputed_growth.py with run_plot_stages.sh defaults. Anomaly = y&lt;0.
ARI all includes noise as a label; ARI normals is over true normals only. <a href="metrics.csv">metrics.csv</a>
· <a href="family_means.csv">family_means.csv</a></p>
<img src="overview.png" style="max-width:100%">
<h2>Family means</h2><table><tr><th>family</th><th>#</th><th>DelTriC ARI</th><th>production ARI</th><th>HDBSCAN ARI</th>
<th>DelTriC F1</th><th>production F1</th><th>HDBSCAN F1</th></tr>{fam_html}</table>
<h2>Datasets</h2><table id="t"><tr>{''.join(f'<th>{h}</th>' for h in head)}</tr>{''.join(lines)}</table>
<h2>Failures</h2><ul>{fail_html or '<li>none</li>'}</ul>
<script>document.querySelectorAll('#t th').forEach((th,i)=>th.onclick=()=>{{const t=th.closest('table'),
rows=[...t.rows].slice(1),dir=th.dataset.d=-(th.dataset.d||-1);rows.sort((a,b)=>{{const x=a.cells[i].innerText,
y=b.cells[i].innerText,p=parseFloat(x),q=parseFloat(y);return dir*(isNaN(p)||isNaN(q)?x.localeCompare(y):p-q)}});
rows.forEach(r=>t.appendChild(r))}})</script>
""")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "results/full_analysis")
    parser.add_argument("--jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    parser.add_argument("--timeout", type=float, default=12 * 3600, help="per-dataset pipeline timeout (s)")
    parser.add_argument("--only", nargs="*", default=None, help="substring filter on dataset names")
    parser.add_argument("--limit", type=int, default=None, help="run only the N smallest (smoke test)")
    parser.add_argument("--resume", action="store_true", help="skip datasets already finished")
    parser.add_argument("--no-gifs", action="store_true")
    parser.add_argument("--no-production", dest="production", action="store_false",
                        help="skip the production cluster_tri comparison run")
    parser.add_argument("--umap-n-epochs", default="100",
                        help="UMAP epochs for the pipeline and production runs; "
                             "'auto' = legacy size-dependent rule")
    parser.add_argument("--max-frames", type=int, default=70)
    parser.add_argument("--fps", type=float, default=6)
    parser.add_argument("--dpi", type=int, default=80)
    parser.add_argument("--list", action="store_true", help="print the dataset list and exit")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)

    items = discover(args.data.resolve(), args.out / "_derived")
    if args.only:
        items = [(n, p) for n, p in items if any(s in n for s in args.only)]
    sizes = {}
    for name, path in items:
        with np.load(path, allow_pickle=False) as z:
            sizes[name] = z["X"].shape
    items.sort(key=lambda it: -sizes[it[0]][0] * max(1.0, math.log(sizes[it[0]][1] + 1)))
    if args.limit:
        items = items[-args.limit:]
    if args.list:
        for name, path in items:
            print(f"{name:70s} n={sizes[name][0]:>6} d={sizes[name][1]:>5}  {path}")
        print(f"{len(items)} datasets")
        return
    (args.out / "datasets.txt").write_text("".join(f"{n}\t{p}\n" for n, p in items))

    if args.aggregate_only:
        records = [json.loads(p.read_text()) for p in sorted((args.out / "per_dataset").glob("*.json"))]
    else:
        print(f"{len(items)} datasets, {args.jobs} workers -> {args.out}", flush=True)
        jobs = [(name, path, args) for name, path in items]
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.jobs, maxtasksperchild=1) as pool:
            records = list(pool.imap_unordered(run_one, jobs, chunksize=1))
    aggregate(args.out, records)
    ok = sum(r["status"] == "ok" for r in records)
    print(f"Done: {ok}/{len(records)} ok. Open {args.out / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
