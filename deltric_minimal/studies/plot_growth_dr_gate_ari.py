#!/usr/bin/env python3
"""Plot ``growth_dr_gate``'s effect on clustering ARI.

Reads ``results/growth_dr_gate_ari/growth_dr_gate_ari.csv`` (written by
``eval_growth_dr_gate_ari.py``) and draws a per-dataset grouped bar chart of
the four variants (baseline_umap, risky_lle, safe_pca, gated_lle), plus a
panel marking which datasets the gate actually fired on.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

VARIANTS = ("baseline_umap", "risky_lle", "safe_pca", "gated_lle")
COLORS = {
    "baseline_umap": "#4c78a8",
    "risky_lle": "#e45756",
    "safe_pca": "#72b7b2",
    "gated_lle": "#eda100",
}
DEFAULT_CSV = Path(__file__).parent / "results" / "growth_dr_gate_ari" / "growth_dr_gate_ari.csv"


def load(csv_path: Path) -> dict[str, dict[str, dict]]:
    by_dataset: dict[str, dict[str, dict]] = {}
    with csv_path.open() as handle:
        for row in csv.DictReader(handle):
            by_dataset.setdefault(row["dataset"], {})[row["variant"]] = row
    return by_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).parent / "results" / "growth_dr_gate_ari" / "growth_dr_gate_ari.png",
    )
    args = parser.parse_args()

    by_dataset = load(args.csv)
    datasets = sorted(by_dataset)
    n = len(datasets)
    width = 0.2
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    for i, variant in enumerate(VARIANTS):
        aris = [float(by_dataset[d][variant]["ari"]) for d in datasets]
        offset = (i - (len(VARIANTS) - 1) / 2) * width
        bars = ax.bar(
            x + offset, aris, width=width * 0.9,
            color=COLORS[variant], label=variant, zorder=3,
        )
        if variant == "gated_lle":
            for bar, d in zip(bars, datasets):
                if by_dataset[d][variant].get("gate_fired") == "True":
                    ax.text(
                        bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                        "gate fired", rotation=90, ha="center", va="bottom",
                        fontsize=7, color="#111111",
                    )

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("ARI")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.set_title(
        "growth_dr_gate: does gating a risky LLE projection (fall back to PCA "
        "when the depth-3 tree flags it) recover ARI lost to degenerate LLE graphs?\n"
        "'gate fired' marks datasets where growth_dr_gate rejected the LLE projection and fell back to PCA"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
