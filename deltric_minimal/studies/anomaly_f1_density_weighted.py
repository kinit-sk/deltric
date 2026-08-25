#!/usr/bin/env python3
"""Does an original-space density term improve anomaly-detection F1?

Motivation. The Dijkstra reclamation stage scores candidate edges purely on
*projected* (UMAP) length. UMAP preserves topology, not metric: it pulls an
isolated point toward whichever manifold it is nearest to, so a genuine noise
point can end up with a short projected edge and get reclaimed by a cluster --
a false negative that projected length alone can never detect.

Hypothesis. Modulating the edge length by an *original-space* density ratio
restores the missing signal. For candidate ``v`` claimed by cluster ``c``:

    rho    = core_distance[v] / median(core_distance over members of c)
    l_eff  = projected_length * rho ** density_power

where ``core_distance`` is the mean distance to a point's k nearest neighbours
in the original space (an inverse-density proxy that, unlike a radius/kernel
density, stays meaningful at d=50 or d=100). ``rho > 1`` means "v sits in a
void relative to the inside of c". ``l_eff`` feeds both the priority-queue
weight and the ``stop_ratio`` gate -- the gate is the half that can actually
move recall, since the weight only reorders which cluster wins a contest.

``density_power=0`` reproduces the current behaviour exactly, so this is a
clean ablation. ``density_clip`` makes the correction one-sided (rho < 1 is
clamped to 1) so unusually dense candidates are never made *cheaper* to claim.

Design. The expensive part (UMAP + component growth + merge) is run once per
dataset with the reassignment disabled; every variant then replays only
``_assign_anomalies_dijkstra`` against that identical frozen graph. So any F1
difference is attributable to the density term and not to UMAP seed variance.

Usage:
    python studies/anomaly_f1_density_weighted.py --out studies/results/anomaly_density
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score, f1_score, precision_score, recall_score,
)
from sklearn.preprocessing import StandardScaler

_DELTRIC_MINIMAL_DIR = Path(__file__).resolve().parents[1]
_DEFAULT_DATA_DIR = _DELTRIC_MINIMAL_DIR / "data"
sys.path.insert(0, str(_DELTRIC_MINIMAL_DIR.parent))

from deltric_minimal.utils_component_growth import (  # noqa: E402
    _assign_anomalies_dijkstra,
    _knn_core_distance,
    cluster_tri,
)
from deltric_minimal.studies.anomaly_f1_dijkstra_vs_baseline import (  # noqa: E402
    BASE_CONFIG,
    DATASET_STEMS,
    KIND_NAMES,
    scores,
)

# (density_power, density_clip) variants. (0.0, False) is the current default
# and doubles as the control.
VARIANTS: list[tuple[float, bool]] = [
    (0.0, False),
    (0.5, False),
    (1.0, False),
    (2.0, False),
    (0.5, True),
    (1.0, True),
    (2.0, True),
]


def variant_name(power: float, clip: bool) -> str:
    if power == 0.0:
        return "q0"
    return f"q{power:g}" + ("_clip" if clip else "")


def run_one(path: Path, penalty_power: float, stop_ratio: float,
            density_k: int, variants: list[tuple[float, bool]]) -> dict:
    npz = np.load(path, allow_pickle=False)
    X = StandardScaler().fit_transform(np.asarray(npz["X"], dtype=np.float64))
    y = np.asarray(npz["y"]).astype(np.int64)
    y_true_anomaly = y < 0
    kind = np.asarray(npz["anomaly_kind"]) if "anomaly_kind" in npz.files else None

    started = time.perf_counter()
    # Frozen graph: everything up to (but excluding) the reassignment stage.
    cluster_tri(X, **BASE_CONFIG, component_growth_anomaly_reassign=False)
    state = cluster_tri.last_component_growth
    merged_labels = state["merged_labels"]
    core_distance = _knn_core_distance(X, density_k)
    graph_seconds = time.perf_counter() - started

    row = {
        "dataset": path.stem,
        "n": len(X),
        "d": int(X.shape[1]),
        "n_true_anomaly": int(y_true_anomaly.sum()),
        "graph_seconds": round(graph_seconds, 2),
    }
    base = scores(y_true_anomaly, merged_labels)
    row["noreassign_precision"] = base["precision"]
    row["noreassign_recall"] = base["recall"]
    row["noreassign_f1"] = base["f1"]
    _real = ~y_true_anomaly
    row["noreassign_ari_real"] = float(
        adjusted_rand_score(y[_real], merged_labels[_real])
    )
    row["noreassign_ari_all"] = float(adjusted_rand_score(y, merged_labels))
    _asg = merged_labels >= 0
    row["noreassign_ari_assigned"] = (
        float(adjusted_rand_score(y[_asg], merged_labels[_asg])) if _asg.any() else float("nan")
    )
    row["noreassign_frac_assigned"] = float(_asg.mean())

    best = None
    for power, clip in variants:
        labels, _ = _assign_anomalies_dijkstra(
            merged_labels, state["edge_keys"], state["projected_edge_sizes"],
            len(X), penalty_power=penalty_power, stop_ratio=stop_ratio,
            core_distance=core_distance, density_power=power, density_clip=clip,
        )
        tag = variant_name(power, clip)
        got = scores(y_true_anomaly, labels)
        # The reassignment stage exists to improve *clustering*, so track ARI
        # over the genuinely-clustered points too: a density term that fixes
        # anomaly F1 by refusing to claim anything would show up here as an
        # ARI regression.
        real = ~y_true_anomaly
        row[f"{tag}_ari_real"] = float(adjusted_rand_score(y[real], labels[real]))
        # ari_all keeps the noise points in, with -1 as its own label: here
        # reclaiming a *true* anomaly into a cluster is a penalty, whereas
        # ari_real cannot see that mistake at all.
        row[f"{tag}_ari_all"] = float(adjusted_rand_score(y, labels))
        asg = labels >= 0
        row[f"{tag}_ari_assigned"] = (
            float(adjusted_rand_score(y[asg], labels[asg])) if asg.any() else float("nan")
        )
        row[f"{tag}_frac_assigned"] = float(asg.mean())
        row[f"{tag}_precision"] = got["precision"]
        row[f"{tag}_recall"] = got["recall"]
        row[f"{tag}_f1"] = got["f1"]
        row[f"{tag}_n_pred_anomaly"] = got["n_pred_anomaly"]
        if kind is not None:
            for code, name in KIND_NAMES.items():
                mask = kind == code
                if mask.any():
                    row[f"{tag}_recall_{name}"] = float((labels[mask] < 0).mean())
        if best is None or got["f1"] > best[1]:
            best = (tag, got["f1"])

    for power, clip in variants:
        tag = variant_name(power, clip)
        if tag != "q0":
            row[f"delta_f1_{tag}"] = row[f"{tag}_f1"] - row["q0_f1"]
    row["best_variant"] = best[0]

    parts = "  ".join(
        f"{variant_name(p, c)}={row[variant_name(p, c) + '_f1']:.4f}"
        for p, c in variants
    )
    print(f"  {path.stem:<30} d={row['d']:<4} true={row['n_true_anomaly']:<4} "
          f"{parts}   best={best[0]}  {graph_seconds:.1f}s", flush=True)
    return row


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=_DEFAULT_DATA_DIR, type=Path)
    parser.add_argument("--glob", nargs="*", default=None)
    parser.add_argument("--out", default=Path("studies/results/anomaly_density"), type=Path)
    parser.add_argument("--penalty-power", type=float, default=2.0)
    parser.add_argument("--stop-ratio", type=float, default=3.0)
    parser.add_argument("--powers", nargs="*", type=float, default=None,
                        help="density_power values to sweep (overrides VARIANTS)")
    parser.add_argument("--clip", action="store_true",
                        help="also sweep the one-sided (clipped) variants")
    parser.add_argument("--density-k", type=int, default=15,
                        help="k for the original-space kNN core distance")
    args = parser.parse_args()

    if args.glob:
        files = sorted({f for pat in args.glob for f in args.data.glob(pat)})
        if not files:
            raise SystemExit(f"no datasets matched {args.glob} under {args.data}")
    else:
        files = [args.data / f"{stem}.npz" for stem in DATASET_STEMS]
        missing = [f for f in files if not f.exists()]
        if missing:
            raise SystemExit(f"missing datasets: {missing}")

    global VARIANTS
    if args.powers is not None:
        VARIANTS = [(0.0, False)] + [(p, False) for p in args.powers if p != 0.0]
        if args.clip:
            VARIANTS += [(p, True) for p in args.powers if p != 0.0]

    print(f"{len(files)} datasets  penalty_power={args.penalty_power}  "
          f"stop_ratio={args.stop_ratio}  density_k={args.density_k}", flush=True)
    rows = [run_one(f, args.penalty_power, args.stop_ratio, args.density_k, VARIANTS)
            for f in files]

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out / "anomaly_f1_density_weighted.csv")

    print("\n=== mean over datasets ===")
    print(f"{'variant':<12} {'precision':>10} {'recall':>8} {'F1':>8} {'dF1 vs q0':>10} {'wins':>6} {'losses':>7} {'ARI_real':>9}")
    print(f"{'no-reassign':<12} {np.mean([r['noreassign_precision'] for r in rows]):>10.4f} "
          f"{np.mean([r['noreassign_recall'] for r in rows]):>8.4f} "
          f"{np.mean([r['noreassign_f1'] for r in rows]):>8.4f} {'':>10} {'':>6} {'':>7} "
          f"{np.mean([r['noreassign_ari_real'] for r in rows]):>9.4f}")
    for power, clip in VARIANTS:
        tag = variant_name(power, clip)
        deltas = [r[f"{tag}_f1"] - r["q0_f1"] for r in rows]
        print(f"{tag:<12} {np.mean([r[f'{tag}_precision'] for r in rows]):>10.4f} "
              f"{np.mean([r[f'{tag}_recall'] for r in rows]):>8.4f} "
              f"{np.mean([r[f'{tag}_f1'] for r in rows]):>8.4f} "
              f"{np.mean(deltas):>+10.4f} "
              f"{sum(d > 1e-9 for d in deltas):>6} {sum(d < -1e-9 for d in deltas):>7} "
              f"{np.mean([r[f'{tag}_ari_real'] for r in rows]):>9.4f}")

    kinds = sorted({k for r in rows for k in r if "_recall_" in k and k.startswith("q0")})
    if kinds:
        print("\n=== recall by anomaly kind (mean) ===")
        names = [k.split("_recall_")[1] for k in kinds]
        print(f"{'variant':<12} " + " ".join(f"{n:>10}" for n in names))
        for power, clip in VARIANTS:
            tag = variant_name(power, clip)
            vals = [np.mean([r[f"{tag}_recall_{n}"] for r in rows if f"{tag}_recall_{n}" in r])
                    for n in names]
            print(f"{tag:<12} " + " ".join(f"{v:>10.4f}" for v in vals))

    print(f"\nWrote {args.out}/anomaly_f1_density_weighted.csv")


if __name__ == "__main__":
    main()
