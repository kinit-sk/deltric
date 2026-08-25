#!/usr/bin/env python3
"""Plot the timing comparison from ``dijkstra_timing_study.py``.

Reads ``dijkstra_timing.csv`` (one row per dataset: ``base_seconds_median``
is the pipeline without the Dijkstra anomaly-reassignment stage, i.e. the
"old" behavior; ``dijkstra_seconds_median`` is that stage's own added cost)
and produces a 2x2 figure:

  1. grouped bar, base vs dijkstra seconds per dataset, sorted by n
  2. dijkstra_seconds vs n on log-log axes, to see the empirical scaling
  3. overhead_pct (dijkstra / base) vs n, log-x
  4. dijkstra_seconds vs n_anomalies_out (the stage's actual workload)

Usage:
    python studies/plot_dijkstra_timing.py \\
        --csv studies/results/dijkstra_timing/dijkstra_timing.csv \\
        --out studies/results/dijkstra_timing/dijkstra_timing.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Fixed categorical order: "old" pipeline stays a neutral blue in every
# panel, the added Dijkstra stage stays a warm orange. Never swapped, never
# reused for anything else in this figure.
COLOR_BASE = "#4C72B0"
COLOR_DIJKSTRA = "#DD8452"
INK_MUTED = "#6b6b6b"


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    for row in rows:
        for key, val in row.items():
            if key == "dataset":
                continue
            try:
                row[key] = float(val)
            except (TypeError, ValueError):
                pass
    return rows


def plot(rows: list[dict], out_path: Path) -> None:
    rows = sorted(rows, key=lambda r: r["n"])
    names = [r["dataset"] for r in rows]
    n = np.array([r["n"] for r in rows])
    base = np.array([r["base_seconds_median"] for r in rows])
    dij = np.array([r["dijkstra_seconds_median"] for r in rows])
    overhead = np.array([r["overhead_pct"] for r in rows])
    n_rounds = np.array([r["n_rounds"] for r in rows])

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        "Dijkstra anomaly-reassignment stage: added wall-clock cost",
        fontsize=13, fontweight="bold",
    )

    # --- panel 1: base vs dijkstra across all datasets, ranked by n ---
    # A grouped bar per dataset only reads cleanly up to a few dozen
    # datasets; with ~148 the labels collide into an unreadable smear, so
    # this is a ranked step plot instead (full per-dataset numbers live in
    # the CSV). Datasets are already sorted by n (see `rows` sort in main).
    ax = axes[0, 0]
    rank = np.arange(len(names))
    ax.fill_between(rank, base, step="mid", color=COLOR_BASE, alpha=0.85, label="old (no Dijkstra)")
    ax.fill_between(rank, dij, step="mid", color=COLOR_DIJKSTRA, alpha=0.85, label="Dijkstra stage (added)")
    ax.set_yscale("log")
    ax.set_xlim(0, len(names) - 1)
    ax.set_xlabel(f"datasets ranked by n ({len(names)} total, low -> high)")
    ax.set_ylabel("seconds (log scale, median of repeats)")
    ax.set_title("Wall-clock time across all datasets", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)

    # --- panel 2: dijkstra_seconds and base_seconds vs n, log-log ---
    ax = axes[0, 1]
    ax.scatter(n, base, s=28, color=COLOR_BASE, label="old (no Dijkstra)")
    ax.scatter(n, dij, s=28, color=COLOR_DIJKSTRA, label="Dijkstra stage (added)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n points (log)")
    ax.set_ylabel("seconds (log)")
    ax.set_title("Scaling with dataset size", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.5)

    # --- panel 3: overhead % vs n ---
    ax = axes[1, 0]
    ax.scatter(n, overhead, s=32, color=COLOR_DIJKSTRA)
    ax.axhline(np.median(overhead), color=INK_MUTED, linewidth=1, linestyle="--",
               label=f"median {np.median(overhead):.1f}%")
    ax.set_xscale("log")
    ax.set_xlabel("n points (log)")
    ax.set_ylabel("overhead (%) = dijkstra / old x 100")
    ax.set_title("Relative overhead vs dataset size", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.5)

    # --- panel 4: dijkstra_seconds vs number of competitive-Dijkstra rounds ---
    # n_rounds (not n_anomalies_out, which is *remaining* noise and can be
    # large simply because nothing was reclaimable) is the actual iteration
    # count the stage pays for, so it is the honest workload proxy here.
    ax = axes[1, 1]
    ax.scatter(n_rounds, dij, s=32, color=COLOR_DIJKSTRA)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel("competitive-Dijkstra rounds (n_rounds, symlog)")
    ax.set_ylabel("dijkstra seconds")
    ax.set_title("Stage cost vs actual workload", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.5)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("results/dijkstra_timing/dijkstra_timing.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/dijkstra_timing/dijkstra_timing.png"))
    args = parser.parse_args()

    rows = load_rows(args.csv)
    if not rows:
        raise SystemExit(f"no rows in {args.csv}")
    plot(rows, args.out)


if __name__ == "__main__":
    main()
