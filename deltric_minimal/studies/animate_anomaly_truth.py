#!/usr/bin/env python3
"""Animate DelTriC's current anomaly reclamation against ground truth.

Example:
    python deltric_minimal/studies/animate_anomaly_truth.py \
        --data deltric_minimal/data/odds_thyroid.npz \
        --out-dir deltric_minimal/studies/results/anomaly_truth

Default ground truth: y < 0 is anomalous (repository convention).
For binary 0=normal, 1=anomaly labels use --truth-format binary.
Snapshots observe the production cost gate; no detector is reimplemented.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
import numpy as np
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from anomaly_f1_dijkstra_vs_baseline import BASE_CONFIG
from deltric_minimal.utils_component_growth import (
    _assign_anomalies_cost_gate, _knn_core_distance, cluster_tri,
)


def confusion(truth: np.ndarray, labels: np.ndarray) -> dict:
    predicted = labels < 0
    tp = int(np.count_nonzero(truth & predicted))
    fp = int(np.count_nonzero(~truth & predicted))
    fn = int(np.count_nonzero(truth & ~predicted))
    tn = int(np.count_nonzero(~truth & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, precision=precision, recall=recall, f1=f1)


def observe_cost_gate(labels, edges, lengths, core, args):
    """Read accepted claims from the actual production function's frame.

    A targeted trace avoids copying the algorithm into a visualization. It
    watches n_claims after each increment and restores any prior trace on exit.
    Snapshots are capped while retaining the exact initial and final states.
    """
    initial = np.asarray(labels).copy()
    snapshots = [(0, initial)]
    stride = max(1, math.ceil(np.count_nonzero(initial < 0) / (args.max_frames - 1)))
    observed = 0
    code = _assign_anomalies_cost_gate.__code__

    def trace_claim(frame, event, arg):
        nonlocal observed
        if event == "line":
            local = frame.f_locals
            diag = local.get("diagnostics")
            count = diag.get("n_claims", 0) if diag is not None else 0
            if count > observed:
                observed = count
                if count % stride == 0:
                    snapshots.append((count, local["owner"].copy()))
        return trace_claim

    def select_frame(frame, event, arg):
        return trace_claim if frame.f_code is code else None

    previous = sys.gettrace()
    try:
        sys.settrace(select_frame)
        final, diag = _assign_anomalies_cost_gate(
            initial, edges, lengths, len(initial), args.penalty_power, args.reach,
            core_distance=core, density_power=args.density_power,
            density_clip=args.density_clip,
        )
    finally:
        sys.settrace(previous)
    if diag["n_claims"] != observed:
        raise RuntimeError("Production claim tracing changed; cannot render a faithful animation")
    if snapshots[-1][0] != observed:
        snapshots.append((observed, final.copy()))
    if not np.array_equal(snapshots[-1][1], final):
        raise RuntimeError("Final animation snapshot does not match production labels")
    return snapshots, final, diag


def run_one(path: Path, args) -> Path:
    with np.load(path, allow_pickle=False) as data:
        X = np.asarray(data["X"], dtype=float)
        raw_truth = np.asarray(data[args.label_key]).reshape(-1)
    if X.ndim != 2 or len(X) < 3 or len(raw_truth) != len(X):
        raise ValueError(f"{path}: expected X[n,d] and one truth label per point")
    if not np.isfinite(X).all() or not np.isfinite(raw_truth).all():
        raise ValueError(f"{path}: features and labels must be finite")
    if args.truth_format == "binary":
        if not np.isin(raw_truth, [0, 1]).all():
            raise ValueError("Binary truth requires 0=normal and 1=anomaly")
        truth = raw_truth == 1
    else:
        truth = raw_truth < 0
    X = StandardScaler().fit_transform(X)
    print(f"Building graph: {path.stem} ({len(X)} points)", flush=True)
    initial = cluster_tri(X, **BASE_CONFIG)
    state = cluster_tri.last_component_growth
    core = _knn_core_distance(X, args.density_k) if args.density_power else None
    snapshots, final, diag = observe_cost_gate(
        initial, state["edge_keys"], state["orig_edge_sizes"], core, args,
    )
    display = np.asarray(state["X_proj"])[:, :2]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.8))
    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        span = np.maximum(np.ptp(display, axis=0), 1.0)
        ax.set_xlim(display[:, 0].min() - .04 * span[0], display[:, 0].max() + .04 * span[0])
        ax.set_ylim(display[:, 1].min() - .04 * span[1], display[:, 1].max() + .04 * span[1])
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].set_title("Ground truth (fixed)")
    axes[0].scatter(*display[~truth].T, s=5, c="#c5c9ce", linewidths=0)
    axes[0].scatter(*display[truth].T, s=45, c="#2563eb", marker="*", zorder=3)
    axes[0].legend(handles=[
        Line2D([], [], marker="o", color="#c5c9ce", linestyle="", label="Normal"),
        Line2D([], [], marker="*", color="#2563eb", linestyle="", label="Anomaly"),
    ], loc="lower left", fontsize=9)
    cluster_ids = np.unique(initial[initial >= 0])
    palette = plt.get_cmap("tab20" if len(cluster_ids) <= 20 else "hsv")
    cluster_colors = {int(cluster): palette(i if len(cluster_ids) <= 20 else i / len(cluster_ids))
                      for i, cluster in enumerate(cluster_ids)}
    cluster_points = axes[1].scatter([], [], s=8, linewidths=0)
    noise_points = axes[1].scatter([], [], s=22, c="#737373", marker="x", linewidths=.8)
    reclaimed_points = axes[1].scatter([], [], s=30, facecolors="none", edgecolors="black",
                                      linewidths=.65, zorder=3)
    axes[1].legend(handles=[
        Line2D([], [], marker="o", color="#7c3aed", linestyle="", label="Color = cluster membership"),
        Line2D([], [], marker="x", color="#737373", linestyle="", label="Unassigned / predicted anomaly"),
        Line2D([], [], marker="o", markerfacecolor="none", color="black", linestyle="", label="Reclaimed point (ring)"),
    ], loc="lower left", fontsize=8)
    if args.cluster_ids:
        for cluster in cluster_ids:
            center = np.median(display[initial == cluster], axis=0)
            axes[1].text(*center, str(cluster), fontsize=7, ha="center", va="center",
                         bbox=dict(facecolor="white", edgecolor="none", alpha=.7, pad=.8), zorder=4)
    # Normal points stay quiet; both error classes have prominent markers.
    categories = [
        ("TN", "#c5c9ce", "o", 5, "TN: normal, correctly accepted"),
        ("TP", "#2563eb", "*", 45, "TP: anomaly, correctly detected"),
        ("FP", "#dc2626", "x", 42, "FP: normal, incorrectly flagged"),
        ("FN", "#f59e0b", "^", 40, "FN: anomaly, missed"),
    ]
    artists = {key: axes[2].scatter([], [], s=size, c=color, marker=marker,
                                  linewidths=.9 if marker == "x" else 0, zorder=i+1)
               for i, (key, color, marker, size, _) in enumerate(categories)}
    fig.legend(handles=[Line2D([], [], marker=marker, color=color, linestyle="",
                               label=label, markersize=8)
                        for _, color, marker, _, label in categories],
               loc="lower center", ncol=2, fontsize=9, frameon=False)
    fig.suptitle(f"{path.stem} — anomaly decisions against truth\n"
                 f"Current cumulative-cost reclamation: p={args.penalty_power:g}, "
                 f"reach={args.reach:g}, density power={args.density_power:g}", fontsize=12)
    fig.text(.5, .105, "Positive = anomaly. Coordinates: 2D UMAP; costs: standardized original features.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .14, 1, .90))
    sequence = [0] * args.hold_start + list(range(1, len(snapshots)))
    sequence += [len(snapshots)-1] * args.hold_end

    def update(index):
        count, labels = snapshots[sequence[index]]
        predicted = labels < 0
        assigned = ~predicted
        cluster_points.set_offsets(display[assigned])
        cluster_points.set_color(np.asarray([cluster_colors[int(c)] for c in labels[assigned]]).reshape(-1, 4))
        noise_points.set_offsets(display[predicted])
        reclaimed_points.set_offsets(display[(initial < 0) & assigned])
        axes[1].set_title(f"DelTriC clusters ({len(cluster_ids)} groups)\n"
                          f"Reclaimed {count}/{diag['n_claims']} points; {int(predicted.sum())} unassigned",
                          fontsize=10)
        masks = dict(TN=~truth & ~predicted, TP=truth & predicted,
                     FP=~truth & predicted, FN=truth & ~predicted)
        for key, artist in artists.items():
            artist.set_offsets(display[masks[key]])
        score = confusion(truth, labels)
        axes[2].set_title(
            "Predictions against truth\n"
            f"TP {score['TP']}   FP {score['FP']}   FN {score['FN']}   TN {score['TN']}\n"
            f"Precision {score['precision']:.3f}   Recall {score['recall']:.3f}   F1 {score['f1']:.3f}",
            fontsize=10,
        )
        return (*artists.values(), cluster_points, noise_points, reclaimed_points)

    output = args.out_dir / f"{path.stem}_truth.gif"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    try:
        animation = FuncAnimation(fig, update, frames=len(sequence), blit=False,
                                  interval=1000/args.fps)
        animation.save(output, writer=PillowWriter(fps=args.fps), dpi=args.dpi)
    finally:
        plt.close(fig)
    metadata = dict(dataset=str(path.resolve()), truth_format=args.truth_format,
                    label_key=args.label_key, mode=diag["mode"],
                    initial=confusion(truth, initial), final=confusion(truth, final),
                    n_claims=diag["n_claims"], snapshot_claim_counts=[n for n, _ in snapshots],
                    n_clusters=len(cluster_ids), cluster_ids_shown=args.cluster_ids,
                    frames=len(sequence), base_config=BASE_CONFIG,
                    parameters={key: getattr(args, key) for key in
                                ["penalty_power", "reach", "density_power", "density_k", "density_clip"]})
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(f"Wrote {output}: {metadata['final']}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, default=ROOT/"results/anomaly_truth")
    parser.add_argument("--label-key", default="y")
    parser.add_argument("--truth-format", choices=["negative", "binary"], default="negative")
    parser.add_argument("--penalty-power", type=float, default=3)
    parser.add_argument("--reach", type=float, default=25)
    parser.add_argument("--density-power", type=float, default=4)
    parser.add_argument("--density-k", type=int, default=15)
    parser.add_argument("--density-clip", action="store_true")
    parser.add_argument("--cluster-ids", action="store_true", help="Label clusters at their initial median positions")
    parser.add_argument("--max-frames", type=int, default=60, help="Maximum distinct snapshots, including endpoints")
    parser.add_argument("--fps", type=float, default=8)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--hold-start", type=int, default=8)
    parser.add_argument("--hold-end", type=int, default=16)
    args = parser.parse_args()
    if args.max_frames < 2 or args.fps <= 0 or args.dpi < 1 or min(args.hold_start,args.hold_end) < 1:
        parser.error("max-frames >= 2, fps > 0, dpi >= 1, and hold durations >= 1 are required")
    if args.density_k < 1 or args.reach < 0 or args.penalty_power < 0:
        parser.error("density-k >= 1 and nonnegative reach and penalty-power are required")
    if len({p.stem for p in args.data}) != len(args.data):
        parser.error("Input dataset stems must be unique to avoid overwriting outputs")
    for path in args.data:
        run_one(path, args)


if __name__ == "__main__":
    main()
