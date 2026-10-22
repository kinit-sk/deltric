#!/usr/bin/env python3
"""Frozen-configuration benchmark: DelTriC vs. standard clusterers (Perun job).

Methods (all on the same rows; every baseline sees standardized features,
DelTriC standardizes internally):

  deltric        ``deltric.DelTriC()`` with its packaged defaults
  hdbscan        sklearn HDBSCAN, min_cluster_size=10, eom (repo standard)
  umap_hdbscan   the same HDBSCAN on a 2-D UMAP (n_neighbors=15, 100 epochs,
                 i.e. DelTriC's own projection settings)
  kmeans         k-means, k = oracle number of truth groups, n_init=10
  genie          genieclust Genie, k = oracle, gini_threshold=0.3 (default)
  optics         sklearn OPTICS, library defaults
  birch          sklearn Birch, n_clusters = oracle, threshold default

"oracle number of truth groups" counts the anomaly label -1 as one group when
present, i.e. the number of distinct values in ``y``. KMeans, Genie and Birch
assign every point, so they predict no anomalies (anomaly F1 = 0, recall = 0).

Datasets with more than ``--n-max`` rows are subsampled once, stratified by
``y`` (proportional, at least one row per label, fixed seed); every method
then runs on that identical subsample. The indices are kept in the prepared
cache and recorded in the per-task JSON.

Every (dataset, method) pair runs in its own single-threaded subprocess with a
wall-clock timeout, so timings are comparable CPU-for-CPU and one crash or
blow-up cannot take down the run. Only ``fit_predict`` (plus feature
standardization) is timed. Peak RSS of the worker is recorded.

Outputs under ``--out``:
  tasks/<dataset>__<method>.json   one record per task (resumable)
  results.csv                      one row per task
  summary.csv                      per-method means over (a) all datasets
                                   each method finished and (b) the common
                                   subset that every method finished
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import resource
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "studies"))

METHODS = ("deltric", "hdbscan", "umap_hdbscan", "kmeans", "genie", "optics", "birch")
SEED = 0
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def sample_indices(labels: np.ndarray, limit: int, seed: int) -> np.ndarray:
    """Proportional, deterministic sampling that does not lose rare labels."""
    if limit == 0 or len(labels) <= limit:
        return np.arange(len(labels))
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    if len(counts) > limit:
        raise ValueError("--n-max must be at least the number of truth labels")
    capacity = counts - 1
    exact = capacity * ((limit - len(counts)) / capacity.sum())
    allocation = np.floor(exact).astype(int) + 1
    remaining = limit - int(allocation.sum())
    order = np.argsort(-(exact - np.floor(exact)), kind="stable")
    allocation[order[:remaining]] += 1
    rng = np.random.default_rng(seed)
    chosen = [rng.choice(np.flatnonzero(inverse == i), size=int(n), replace=False)
              for i, n in enumerate(allocation)]
    return np.sort(np.concatenate(chosen))


def discover(data_dir: Path, exclude: list[str], only: list[str]) -> list[tuple[str, Path]]:
    items = []
    for path in sorted(data_dir.glob("*.npz")):
        name = path.stem
        tag = name.split("_")[0]
        if tag in exclude or name in exclude:
            continue
        if only and tag not in only and name not in only:
            continue
        items.append((name, path))
    return items


def prepare(name: str, path: Path, prepared_dir: Path, n_max: int) -> dict:
    """Load once, subsample, store the exact rows every method will see."""
    target = prepared_dir / f"{name}.npz"
    if target.exists():
        with np.load(target, allow_pickle=False) as z:
            return dict(name=name, prepared=str(target), n_orig=int(z["n_orig"]),
                        n=int(len(z["y"])), d=int(z["X"].shape[1]))
    with np.load(path, allow_pickle=False) as z:
        y_full = np.asarray(z["y"], dtype=np.int64)
        idx = sample_indices(y_full, n_max, SEED)
        X = np.asarray(z["X"], dtype=np.float64)
        X = X[idx] if len(idx) < len(X) else X
    y = y_full[idx]
    if not np.all(np.isfinite(X)):
        raise ValueError(f"{name}: non-finite values in X")
    tmp = prepared_dir / f".{name}.tmp.npz"
    np.savez(tmp, X=X, y=y, idx=idx, n_orig=len(y_full))
    tmp.rename(target)
    return dict(name=name, prepared=str(target), n_orig=len(y_full), n=len(y), d=X.shape[1])


# ---------------------------------------------------------------------------
# metrics (same definitions as perun_full_analysis.scores)
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
# methods
# ---------------------------------------------------------------------------

def _standardize(X):
    from sklearn.preprocessing import StandardScaler
    return StandardScaler().fit_transform(X)


def _hdbscan(Z):
    from sklearn.cluster import HDBSCAN
    return HDBSCAN(min_cluster_size=10, cluster_selection_method="eom", n_jobs=1).fit_predict(Z)


def run_method(method: str, X: np.ndarray, n_groups: int) -> np.ndarray:
    if method == "deltric":
        from deltric import DelTriC
        return DelTriC().fit_predict(X)
    Z = _standardize(X)
    if method == "hdbscan":
        return _hdbscan(Z)
    if method == "umap_hdbscan":
        from umap import UMAP
        P = UMAP(n_components=2, n_neighbors=min(15, len(Z) - 1), n_epochs=100,
                 random_state=42, transform_seed=42).fit_transform(Z)
        return _hdbscan(P)
    if method == "kmeans":
        from sklearn.cluster import KMeans
        return KMeans(n_clusters=n_groups, n_init=10, random_state=SEED).fit_predict(Z)
    if method == "genie":
        import genieclust
        if n_groups == 1:
            return np.zeros(len(Z), dtype=np.int64)
        return genieclust.Genie(n_clusters=n_groups, gini_threshold=0.3).fit_predict(Z)
    if method == "optics":
        from sklearn.cluster import OPTICS
        return OPTICS(n_jobs=1).fit_predict(Z)
    if method == "birch":
        from sklearn.cluster import Birch
        return Birch(n_clusters=n_groups).fit_predict(Z)
    raise ValueError(method)


def worker(prepared: str, method: str, out: str) -> None:
    with np.load(prepared, allow_pickle=False) as z:
        X, y, n_orig = z["X"], z["y"].astype(np.int64), int(z["n_orig"])
    n_groups = int(len(np.unique(y)))
    record = dict(dataset=Path(prepared).stem, method=method, n=len(y), d=X.shape[1],
                  n_orig=n_orig, subsampled=len(y) < n_orig, n_groups_oracle=n_groups,
                  k_true=int(len(np.unique(y[y >= 0]))), n_true_anomaly=int((y < 0).sum()),
                  status="ok")
    t0, c0 = time.perf_counter(), time.process_time()
    pred = np.asarray(run_method(method, X, n_groups), dtype=np.int64)
    record["seconds"] = time.perf_counter() - t0
    record["cpu_seconds"] = time.process_time() - c0
    record.update(scores(y, pred))
    record["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    tmp = Path(out).with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=1))
    tmp.rename(out)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def cost_estimate(info: dict, method: str) -> float:
    """Rough relative cost, only used to start the longest tasks first."""
    n, d = info["n"], info["d"]
    quadratic = method in ("optics", "hdbscan", "genie", "birch")
    base = n * n * max(d, 8) if quadratic else n * math.log(n + 2) * max(d, 8) * 50
    return base * (5.0 if method == "deltric" else 1.0)


def launch(info: dict, method: str, task_dir: Path, timeout: float) -> dict:
    out = task_dir / f"{info['name']}__{method}.json"
    if out.exists():
        return json.loads(out.read_text())
    log = task_dir / f"{info['name']}__{method}.log"
    env = dict(os.environ, **{k: "1" for k in THREAD_ENV})
    cmd = [sys.executable, "-u", __file__, "worker", info["prepared"], method, str(out)]
    t0 = time.perf_counter()
    with open(log, "w") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
        try:
            code = proc.wait(timeout=timeout)
            status = "ok" if code == 0 and out.exists() else f"failed(exit {code})"
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            status = "timeout"
    if status == "ok":
        log.unlink(missing_ok=True)
        return json.loads(out.read_text())
    record = dict(dataset=info["name"], method=method, n=info["n"], d=info["d"],
                  n_orig=info["n_orig"], status=status,
                  seconds=time.perf_counter() - t0)
    # Failures and timeouts are persisted too, so a resubmit does not retry
    # them; delete the JSON to force a retry.
    out.write_text(json.dumps(record, indent=1))
    return record


FIELDS = ["dataset", "method", "status", "n", "d", "n_orig", "subsampled", "k_true",
          "n_groups_oracle", "n_true_anomaly", "seconds", "cpu_seconds", "peak_rss_gb",
          "ari_all", "ari_real", "precision", "recall", "f1", "TP", "FP", "FN", "TN",
          "n_pred_anomaly", "n_clusters"]


def summarize(records: list[dict], out: Path, methods: list[str]) -> None:
    with open(out / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in sorted(records, key=lambda r: (r["dataset"], r["method"])):
            w.writerow(r)
    by = {}
    for r in records:
        by.setdefault(r["dataset"], {})[r["method"]] = r
    common = [ds for ds, m in by.items()
              if all(m.get(k, {}).get("status") == "ok" for k in methods)]

    def mean(vals):
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
        return float(np.mean(vals)) if vals else float("nan")

    rows = []
    for scope, datasets in (("all_finished", list(by)), ("common_subset", common)):
        for m in methods:
            rs = [by[ds][m] for ds in datasets if by[ds].get(m, {}).get("status") == "ok"]
            # Anomaly metrics only where the truth has anomalies.
            ra = [r for r in rs if r.get("n_true_anomaly", 0) > 0]
            all_m = [by[ds].get(m) for ds in by if m in by[ds]]
            rows.append(dict(
                scope=scope, method=m, n_datasets=len(rs),
                n_timeout=sum(r["status"] == "timeout" for r in all_m) if scope == "all_finished" else "",
                n_failed=sum(r["status"].startswith("failed") for r in all_m) if scope == "all_finished" else "",
                mean_ari_all=mean([r["ari_all"] for r in rs]),
                mean_ari_real=mean([r["ari_real"] for r in rs]),
                mean_anomaly_f1=mean([r["f1"] for r in ra]),
                mean_anomaly_precision=mean([r["precision"] for r in ra]),
                mean_anomaly_recall=mean([r["recall"] for r in ra]),
                mean_seconds=mean([r["seconds"] for r in rs]),
                median_seconds=float(np.median([r["seconds"] for r in rs])) if rs else float("nan"),
                total_hours=sum(r["seconds"] for r in rs) / 3600,
                max_peak_rss_gb=max((r.get("peak_rss_gb", 0) for r in rs), default=float("nan")),
            ))
    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\ncommon subset: {len(common)}/{len(by)} datasets")
    for r in rows:
        print(f"{r['scope']:<14} {r['method']:<13} n={r['n_datasets']:<3} "
              f"ARI={r['mean_ari_all']:.3f} ARI_real={r['mean_ari_real']:.3f} "
              f"F1={r['mean_anomaly_f1']:.3f} mean_s={r['mean_seconds']:.1f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-max", type=int, default=50_000)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--prep-jobs", type=int, default=8)
    ap.add_argument("--timeout-hours", type=float, default=12.0)
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    ap.add_argument("--exclude", nargs="*", default=[], help="dataset tags (e.g. D73) or stems")
    ap.add_argument("--only", nargs="*", default=[], help="restrict to these tags/stems")
    args = ap.parse_args()

    task_dir, prepared_dir = args.out / "tasks", args.out / "prepared"
    task_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps(dict(
        vars(args), data_dir=str(args.data_dir), out=str(args.out), seed=SEED,
        python=sys.version, started=time.strftime("%Y-%m-%d %H:%M:%S")), indent=1))

    items = discover(args.data_dir, args.exclude, args.only)
    print(f"{len(items)} datasets, {len(args.methods)} methods, n_max={args.n_max}, "
          f"jobs={args.jobs}", flush=True)
    infos = []
    with ThreadPoolExecutor(args.prep_jobs) as pool:
        futs = {pool.submit(subprocess.run, [sys.executable, __file__, "prepare", str(p),
                                             str(prepared_dir), str(args.n_max)],
                            capture_output=True, text=True): name for name, p in items}
        for fut in as_completed(futs):
            res = fut.result()
            if res.returncode:
                print(f"PREP FAILED {futs[fut]}\n{res.stderr}", flush=True)
                continue
            infos.append(json.loads(res.stdout.strip().splitlines()[-1]))
            print(f"prepared {infos[-1]}", flush=True)

    tasks = sorted(((i, m) for i in infos for m in args.methods),
                   key=lambda t: -cost_estimate(*t))
    records, t_start = [], time.perf_counter()
    with ThreadPoolExecutor(args.jobs) as pool:
        futs = {pool.submit(launch, i, m, task_dir, args.timeout_hours * 3600): (i, m)
                for i, m in tasks}
        for k, fut in enumerate(as_completed(futs), 1):
            i, m = futs[fut]
            try:
                r = fut.result()
            except Exception:
                traceback.print_exc()
                continue
            records.append(r)
            print(f"[{k}/{len(tasks)} {(time.perf_counter() - t_start) / 3600:.2f}h] "
                  f"{r['dataset']:<34} {m:<13} {r['status']:<10} "
                  f"{r.get('seconds', float('nan')):9.1f}s "
                  f"ARI={r.get('ari_all', float('nan')):.3f} F1={r.get('f1', float('nan')):.3f} "
                  f"RSS={r.get('peak_rss_gb', float('nan')):.1f}G", flush=True)
            if k % 20 == 0:
                summarize(records, args.out, args.methods)
    summarize(records, args.out, args.methods)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        worker(*sys.argv[2:5])
    elif len(sys.argv) > 1 and sys.argv[1] == "prepare":
        path, prepared_dir, n_max = Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
        print(json.dumps(prepare(path.stem, path, prepared_dir, n_max)))
    else:
        main()
