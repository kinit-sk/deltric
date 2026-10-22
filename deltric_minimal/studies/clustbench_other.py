#!/usr/bin/env python3
"""Prepare a frozen dataset-level split and run only its development datasets.

Uses the exact stage program and defaults from run_plot_stages.sh (2026-09-17).
Never fits or plots held-out datasets. Source noise label 0 maps to local -1.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from urllib.request import urlopen

import numpy as np
from sklearn.metrics import adjusted_rand_score, confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
VERSION = "1.1.0"
COMMIT = "ee6d7a28eb0ff922d29b618b595795a45f1b5cb8"
URL = f"https://raw.githubusercontent.com/gagolews/clustering-data-v1/{COMMIT}/other"
CHAMELEON = ["chameleon_t4_8k", "chameleon_t5_8k", "chameleon_t7_10k", "chameleon_t8_8k"]
CONTROLS = ["iris", "iris5", "square"]
SEED = 20260917
STAGE_ARGS = [
    "--dim-reduction", "umap", "--project-dim", "2",
    "--umap-n-epochs", "100", "--umap-n-neighbors", "15",
    "--seed-hard-limit", "0.0", "--hard-growth-limit", "1.05",
    "--projected-hard-growth-limit", "1.35", "--initial-relation", "union",
    "--component-growth-knn", "50", "--component-growth-min-edges", "10",
    "--hard-gate-mode", "strict", "--no-redundancy-pruning",
    "--gomory-hu-cut-size", "6", "--gomory-hu-min-component-points", "10",
    "--gomory-hu-hull-ratio-skip-threshold", "0.0",
    "--gomory-hu-growth-edges-only", "--restore-intra-component-edges",
    "--projected-hard-limit", "--no-multiplicative-hard-limit",
    "--outlier-growth", "--outlier-reassignment", "cost_gate",
    "--outlier-cost-gate-penalty-power", "0.5", "--outlier-cost-gate-reach", "10.0",
    "--outlier-cost-gate-density-power", "0.5", "--outlier-cost-gate-density-k", "15",
    "--no-outlier-cost-gate-density-clip", "--outlier-growth-hard-limit", "3.0",
    "--projected-outlier-growth-hard-limit", "3.0", "--quiet",
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(data: Path, holdout_data: Path) -> dict:
    manifest_path = data / "split.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for entry in manifest["datasets"]:
            for relative, digest in entry["sha256"].items():
                if sha(data / relative) != digest:
                    raise ValueError(f"Dataset checksum mismatch: {relative}")
        return manifest
    shuffled = CHAMELEON.copy()
    random.Random(SEED).shuffle(shuffled)
    holdout = set(shuffled[:2])
    manifest = dict(suite_version=VERSION, source_commit=COMMIT, split_seed=SEED,
                    split_unit="whole dataset", reference_partition="labels0",
                    source_noise_label=0, local_noise_label=-1,
                    policy="Two seeded Chameleon datasets held out; remaining two and hdbscan development; three no-noise controls development. No parameter selection using holdout.",
                    datasets=[])
    for name in CHAMELEON + ["hdbscan"] + CONTROLS:
        split = "holdout" if name in holdout else "development"
        destination = holdout_data if split == "holdout" else data
        raw = destination / "source"
        raw.mkdir(parents=True, exist_ok=True)
        files = []
        for suffix in ("data.gz", "labels0.gz", "txt"):
            target = raw / f"{name}.{suffix}"
            with urlopen(f"{URL}/{name}.{suffix}", timeout=60) as response:
                target.write_bytes(response.read())
            files.append(target)
        X = np.loadtxt(files[0], ndmin=2)
        source_y = np.loadtxt(files[1], dtype=np.int64)
        if len(X) != len(source_y) or not np.isfinite(X).all() or source_y.min() < 0:
            raise ValueError(f"Invalid source data: {name}")
        y = source_y - 1
        target = destination / f"{name}.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, X=X, y=y, source_y=source_y)
        files.append(target)
        entry = dict(dataset=f"other/{name}", name=name, split=split,
                     role="no-noise control" if name in CONTROLS else "noise benchmark",
                     n=len(X), d=X.shape[1], n_noise=int((y < 0).sum()),
                     path=os.path.relpath(target, data),
                     sha256={os.path.relpath(p, data): sha(p) for p in files})
        manifest["datasets"].append(entry)
        print(f"Prepared {name}: {split}, n={len(X)}, noise={entry['n_noise']}", flush=True)
    (data / "source" / "README.txt").write_bytes(urlopen(f"{URL}/README.txt", timeout=60).read())
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "other")
    parser.add_argument("--holdout-data", type=Path, default=ROOT / "data" / "holdout" / "other")
    parser.add_argument("--out", type=Path, default=ROOT / "studies" / "results" / "clustbench_other")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.data = args.data.resolve()
    args.out = args.out.resolve()
    manifest = prepare(args.data, args.holdout_data.resolve())
    if args.prepare_only:
        return
    args.out.mkdir(parents=True, exist_ok=True)
    source_files = [ROOT / name for name in ("run_plot_stages.sh", "plot_seed_component_recomputed_growth.py", "utils_component_growth.py", "utils_pruning.py", "plot_seed_component_metrics.py", "hdbscan_comparison.py")]
    provenance = dict(stage_arguments=STAGE_ARGS, split_sha256=sha(args.data / "split.json"),
                      python=sys.version, packages={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "umap-learn", "numba", "networkx", "matplotlib")},
                      code_sha256={p.name: sha(p) for p in source_files},
                      git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                      git_status=subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True))
    provenance["code_sha256"][Path(__file__).name] = sha(Path(__file__))
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    env = dict(os.environ, MPLBACKEND="Agg", MPLCONFIGDIR="/tmp/mplconfig", NUMBA_CACHE_DIR="/tmp/numba_cache", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
    rows = []
    for entry in manifest["datasets"]:
        if entry["split"] != "development":
            continue
        name = entry["name"]
        out = args.out / f"{name}.png"
        command = [sys.executable, str(ROOT / "plot_seed_component_recomputed_growth.py"), "--data", str(args.data / entry["path"]), "--out", str(out), *STAGE_ARGS]
        print(f"Running development dataset {name}", flush=True)
        started = time.perf_counter()
        with (args.out / f"{name}.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        with np.load(out.with_suffix(".npz"), allow_pickle=False) as result:
            prediction = result["final_prediction"]
        with np.load(args.data / entry["path"], allow_pickle=False) as loaded:
            y = loaded["y"]
        true, pred = y < 0, prediction < 0
        tn, fp, fn, tp = confusion_matrix(true, pred, labels=[False, True]).ravel()
        precision, recall, f1, _ = precision_recall_fscore_support(true, pred, average="binary", zero_division=0)
        row = dict(dataset=entry["dataset"], role=entry["role"], n=len(y), n_true_noise=int(true.sum()), n_pred_noise=int(pred.sum()),
                   precision=float(precision) if true.any() else "", recall=float(recall) if true.any() else "", f1=float(f1) if true.any() else "",
                   false_positive_rate=float(fp / (fp + tn)), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
                   ari_all=float(adjusted_rand_score(y, prediction)), ari_non_noise=float(adjusted_rand_score(y[~true], prediction[~true])),
                   seconds=round(time.perf_counter() - started, 2))
        rows.append(row)
        with (args.out / "development_metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps(row), flush=True)
    noisy = [row for row in rows if row["n_true_noise"] > 0]
    summary = dict(evaluated_split="development only", n_noise_datasets=len(noisy), n_controls=len(rows)-len(noisy),
                   macro={key: float(np.mean([row[key] for row in noisy])) for key in ("precision", "recall", "f1", "ari_all", "ari_non_noise")},
                   holdout_evaluated=False)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
