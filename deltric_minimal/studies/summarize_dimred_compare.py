#!/usr/bin/env python3
"""Summarize slurm/run_dimred_compare.sh: DelTriC+UMAP vs DelTriC+t-SNE vs HDBSCAN.

Reads results/dimred_compare/{umap,tsne}/<subdir>/<stem>.{json,npz} written by
run_plot_stages.sh. ARI is over all points (anomalies/noise included). Anomaly
F1 treats y < 0 as anomaly and predicted -1 as flagged; HDBSCAN is refitted
with the run_plot_stages.sh settings (mcs=10, standardized original space)
because its labels are not saved. Writes summary.csv next to the results.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score, f1_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hdbscan_comparison import fit_hdbscan  # noqa: E402

RES = ROOT / "results/dimred_compare"


def load(method: str) -> dict[str, Path]:
    base = RES / method
    return {str(p.relative_to(base).with_suffix("")): p.with_suffix("")
            for p in base.rglob("*.json")}


def scores(y: np.ndarray, pred: np.ndarray) -> tuple[float, float | None]:
    ari = float(adjusted_rand_score(y, pred))
    if not np.any(y < 0):
        return ari, None
    return ari, float(f1_score(y < 0, pred < 0, zero_division=0))


def main():
    umap, tsne = load("umap"), load("tsne")
    rows = []
    for name in sorted(set(umap) & set(tsne)):
        data = ROOT / "data" / f"{name}.npz"
        d = np.load(data)
        y = np.asarray(d["y"], dtype=np.int64)
        row = dict(dataset=name, n=len(y), d=d["X"].shape[1], has_anomaly=bool(np.any(y < 0)))
        for method, base in (("umap", umap[name]), ("tsne", tsne[name])):
            pred = np.load(base.with_suffix(".npz"))["final_prediction"]
            row[f"{method}_ari"], row[f"{method}_f1"] = scores(y, pred)
            row[f"{method}_seconds"] = json.loads(base.with_suffix(".json").read_text())[
                "deltric_pipeline_seconds"]
        hdb = np.asarray(fit_hdbscan(StandardScaler().fit_transform(d["X"]),
                                     min_cluster_size=10, min_samples=None,
                                     cluster_selection_method="eom",
                                     cluster_selection_epsilon=0.0))
        row["hdbscan_ari"], row["hdbscan_f1"] = scores(y, hdb)
        rows.append(row)

    with open(RES / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"{len(rows)} datasets finished by both (umap {len(umap)}, tsne {len(tsne)})")
    for key in ("ari", "f1"):
        cols = [f"umap_{key}", f"tsne_{key}", f"hdbscan_{key}"]
        sub = [r for r in rows if r[cols[0]] is not None]
        a = np.array([[r[c] for c in cols] for r in sub], dtype=float)
        print(f"\n{key.upper()} mean over {len(sub)} datasets: " + "  ".join(
            f"{c.split('_')[0]}={m:.4f}" for c, m in zip(cols, a.mean(0))))
        for i, j in ((1, 0), (0, 2), (1, 2)):
            diff = a[:, i] - a[:, j]
            print(f"  {cols[i].split('_')[0]} - {cols[j].split('_')[0]}: mean {diff.mean():+.4f}, "
                  f"wins {np.sum(diff > 1e-9)} / losses {np.sum(diff < -1e-9)} / "
                  f"ties {np.sum(abs(diff) <= 1e-9)}")
    t = np.array([[r["umap_seconds"], r["tsne_seconds"]] for r in rows])
    print(f"\nDelTriC runtime median: umap {np.median(t[:, 0]):.1f}s, tsne {np.median(t[:, 1]):.1f}s")


if __name__ == "__main__":
    main()
