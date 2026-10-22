#!/usr/bin/env python3
"""Optuna search for a *robust* cost_gate config (p, reach, q, clip).

Companion to anomaly_cost_gate_study.py. That script's own caveats say the
manually grid-searched "best" config (q=4, reach=25, p=3) was selected on the
same 100 datasets it's reported on -- flagged there as benchmark overfitting
-- and that at that config F1 is flat across reach 25->150 ("measures well,
looks fragile"). This script replaces the manual grid with a TPE search whose
objective is built to reject exactly those failure modes rather than just
maximize mean F1:

  * Per-dataset scores are converted to DELTAS against the noreassign
    baseline, so the ~56/100 datasets every variant leaves untouched
    contribute 0 instead of anchoring the mean at a constant.
  * A deadband of 0.02 F1 zeroes out deltas the parent study says are not
    distinguishable from single-seed UMAP noise, so TPE cannot chase them.
  * Deltas are bucketed by dimensionality (the axis the parent study shows
    the largest swings on -- d=20 alone goes noreassign 0.233 -> best-config
    0.629) before averaging, so no bucket votes twice by being over-sampled.
  * The aggregate is mean bucket delta AND mean of the worst quartile of
    buckets (CVaR-style), so a config that wins on average but collapses on
    one dimensionality bucket is penalised, not rewarded.
  * ari_all enters only as a one-sided floor (penalised if it regresses,
    not rewarded if it improves) because the parent study found F1 and
    ari_all can move in opposite directions for the same variant.
  * The existing alternating-dataset holdout split (A/B) is folded into the
    objective itself as a disagreement penalty -- S(all) - 0.5*|S(A)-S(B)| --
    instead of tune-on-A-then-gate-on-B, because thousands of TPE trials
    would overfit split A outright, and gating throws away half of the ~44
    "live" datasets.
  * The reported winner is NOT the single best trial. After the search,
    trials are re-scored by the mean objective of every other trial within a
    small radius in normalized (p, reach, q) space (same clip). The winner
    is the argmax of that smoothed score -- a config in a broad plateau, not
    a lucky spike -- which is what "flat across reach 25->150" was already
    telling us to look for.

(Design consulted with Opus; see the module docstring above for what it
argued against: plain mean-minus-std, which is minimized by doing nothing
once 56/100 deltas are zero.)

--------------------------------------------------------------------------
PARALLELISM
--------------------------------------------------------------------------
Each trial replays cost_gate over all cached graphs -- pure numpy/heapq, no
UMAP, no training -- so trial cost is small (measured ~8-10s / trial for the
full 100 datasets on one core) and the search is trial-count bound, not
per-trial compute bound.

v1 of this script shared one Optuna JournalStorage file across all worker
processes so the TPE sampler could see every worker's trials. On perun with
64 workers that backfired: the shared file's lock (symlink-based, chosen
because perun home dirs are NFS) spent more wall-clock time being waited on
than trials spent computing (13k "waiting >10s for the lock" warnings vs 8k
completed trials in a 90-minute run). v2 instead runs one INDEPENDENT
in-memory TPE study per worker process -- no shared storage, no lock, no
contention, and it scales to any core count. The cost is that workers don't
see each other's trials (each does its own smaller, still-effective TPE
search), which is a good trade once you're running thousands of trials
total: cross-worker sample sharing was a nice-to-have, lock contention was a
hard bottleneck.

Each worker appends one JSON line per completed trial to its own
worker_<seed>.jsonl in --out (flushed immediately) -- exclusive to that
process, so no contention, and if the job dies mid-run every trial that
finished before the kill is still on disk. Run with --collect-only at any
time (job running or finished, or after a crash) to build the report from
whatever's on disk so far, same as the recovery done manually after the v1
run's SLURM time-limit kill.

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
    # laptop, quick check
    python studies/anomaly_cost_gate_optuna.py --trials 500 --workers 8

    # perun (see slurm/run_anomaly_cost_gate_optuna.sh)
    sbatch slurm/run_anomaly_cost_gate_optuna.sh

    # peek at partial results any time without touching a running job
    python studies/anomaly_cost_gate_optuna.py --collect-only \
        --out studies/results/anomaly_cost_gate_optuna
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import optuna
from sklearn.metrics import adjusted_rand_score, f1_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anomaly_cost_gate_study import (  # noqa: E402
    build_variants, cost_gate, evaluate, load_cache, no_reassign, report,
    report_by_dim, write_csv,
)

_MINIMAL_DIR = Path(__file__).resolve().parents[1]
_DEFAULT_CACHE = _MINIMAL_DIR / "studies" / "results" / "anomaly_cost_gate" / "cache"
_DEFAULT_OUT = _MINIMAL_DIR / "studies" / "results" / "anomaly_cost_gate_optuna"

DEADBAND = 0.02
WORST_FRACTION = 0.25
ARI_PENALTY_WEIGHT = 1.0
SPLIT_DISAGREEMENT_WEIGHT = 0.5
PLATEAU_RADIUS = 0.10

SEARCH_SPACE = dict(p=(0.0, 4.0), reach=(1.0, 150.0), q=(0.0, 6.0))


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def _prepare_rows(entries):
    """Per-dataset ground truth + noreassign baseline, computed once."""
    rows = []
    for entry in entries:
        y = entry["y"]
        true_anomaly = y < 0
        baseline_assigned, _ = no_reassign(entry)
        rows.append(dict(
            entry=entry, y=y, true_anomaly=true_anomaly, d=entry["d"],
            baseline_f1=f1_score(true_anomaly, baseline_assigned < 0, zero_division=0),
            baseline_ari=float(adjusted_rand_score(y, baseline_assigned)),
        ))
    return rows


def _score_config(rows, p, reach, q, clip):
    """Per-dataset delta-F1 and delta-ARI (vs noreassign) for one config."""
    deltas = np.empty(len(rows))
    ari_deltas = np.empty(len(rows))
    for i, row in enumerate(rows):
        assigned, _ = cost_gate(row["entry"], p=p, reach=reach, q=q, clip=clip)
        f1 = f1_score(row["true_anomaly"], assigned < 0, zero_division=0)
        ari = float(adjusted_rand_score(row["y"], assigned))
        deltas[i] = f1 - row["baseline_f1"]
        ari_deltas[i] = ari - row["baseline_ari"]
    return deltas, ari_deltas


def _robust_score(deltas, ari_deltas, ds) -> float:
    """mean + worst-quartile of dimensionality-bucketed, deadbanded F1 delta,
    minus a one-sided ari_all regression penalty."""
    if len(deltas) == 0:
        return float("nan")
    tilde = np.sign(deltas) * np.maximum(0.0, np.abs(deltas) - DEADBAND)
    bucket_means = np.array(sorted(
        float(tilde[ds == d].mean()) for d in sorted(set(ds.tolist()))
    ))
    n_worst = max(1, math.ceil(WORST_FRACTION * len(bucket_means)))
    worst = bucket_means[:n_worst]
    ari_delta_mean = float(ari_deltas.mean())
    return (0.5 * bucket_means.mean() + 0.5 * worst.mean()
            - ARI_PENALTY_WEIGHT * max(0.0, -ari_delta_mean))


def make_objective(rows):
    # Same alternating-by-sorted-name split as anomaly_cost_gate_study.py's
    # --holdout, so results are directly comparable to that script's output.
    split_a = np.array([i % 2 == 0 for i in range(len(rows))])
    ds_arr = np.array([r["d"] for r in rows], dtype=float)

    def objective(trial: optuna.Trial) -> float:
        p = trial.suggest_float("p", *SEARCH_SPACE["p"])
        reach = trial.suggest_float("reach", *SEARCH_SPACE["reach"], log=True)
        q = trial.suggest_float("q", *SEARCH_SPACE["q"])
        clip = trial.suggest_categorical("clip", [False, True])

        deltas, ari_deltas = _score_config(rows, p, reach, q, clip)
        s_all = _robust_score(deltas, ari_deltas, ds_arr)
        s_a = _robust_score(deltas[split_a], ari_deltas[split_a], ds_arr[split_a])
        s_b = _robust_score(deltas[~split_a], ari_deltas[~split_a], ds_arr[~split_a])

        trial.set_user_attr("s_all", s_all)
        trial.set_user_attr("s_a", s_a)
        trial.set_user_attr("s_b", s_b)
        trial.set_user_attr("mean_f1_delta", float(deltas.mean()))
        trial.set_user_attr("mean_ari_delta", float(ari_deltas.mean()))

        return s_all - SPLIT_DISAGREEMENT_WEIGHT * abs(s_a - s_b)

    return objective


# ---------------------------------------------------------------------------
# distributed search -- independent per-worker studies, no shared storage
# ---------------------------------------------------------------------------

def _worker(n_trials: int, cache_dir: Path, seed: int, out_dir: Path) -> None:
    if n_trials <= 0:
        return
    rows = _prepare_rows(load_cache(cache_dir))
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True),
    )
    jsonl_path = out_dir / f"worker_{seed}.jsonl"
    # Exclusive to this process -- no other worker ever touches this file,
    # so no locking is needed, unlike the shared-journal design this replaced.
    with jsonl_path.open("a") as handle:
        def checkpoint(_study, trial):
            if trial.state != optuna.trial.TrialState.COMPLETE:
                return
            handle.write(json.dumps({
                "seed": seed, "number": trial.number, "value": trial.value,
                "params": trial.params, "user_attrs": trial.user_attrs,
            }) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

        study.optimize(make_objective(rows), n_trials=n_trials,
                        callbacks=[checkpoint], show_progress_bar=False)


def run_search(cache_dir: Path, out_dir: Path, n_trials: int, n_workers: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # A worker_<seed>.jsonl left over from a previous, possibly-differently-
    # configured run would silently mix stale trials into this one's report.
    # Every fresh invocation of run_search starts clean; use --collect-only
    # to read an in-progress or finished run's files without launching a
    # new search (and without touching them).
    for stale in out_dir.glob("worker_*.jsonl"):
        stale.unlink()

    n_workers = max(1, n_workers)
    per_worker = [n_trials // n_workers] * n_workers
    for i in range(n_trials % n_workers):
        per_worker[i] += 1

    ctx = mp.get_context("spawn")
    started = time.perf_counter()
    procs = []
    for i, n in enumerate(per_worker):
        proc = ctx.Process(target=_worker, args=(n, cache_dir, 1000 + i, out_dir))
        proc.start()
        procs.append(proc)
    for proc in procs:
        proc.join()
    elapsed = time.perf_counter() - started
    print(f"{n_trials} trials across {len(procs)} workers in {elapsed:.1f}s "
          f"({n_trials / max(elapsed, 1e-9):.1f} trials/s)", flush=True)


def collect_trials(out_dir: Path) -> list[dict]:
    """Merge every worker's checkpointed trials. Safe to call on a
    still-running job -- skips a trailing line that's mid-write."""
    trials = []
    for path in sorted(out_dir.glob("worker_*.jsonl")):
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    trials.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # last line of a live file, still being written
    return trials


# ---------------------------------------------------------------------------
# plateau selection + reporting
# ---------------------------------------------------------------------------

def _normalize(params: dict) -> np.ndarray:
    p_lo, p_hi = SEARCH_SPACE["p"]
    r_lo, r_hi = math.log(SEARCH_SPACE["reach"][0]), math.log(SEARCH_SPACE["reach"][1])
    q_lo, q_hi = SEARCH_SPACE["q"]
    return np.array([
        (params["p"] - p_lo) / (p_hi - p_lo),
        (math.log(params["reach"]) - r_lo) / (r_hi - r_lo),
        (params["q"] - q_lo) / (q_hi - q_lo),
    ])


def plateau_best(trials: list[dict], radius: float = PLATEAU_RADIUS):
    """Argmax of the LOCALLY-AVERAGED objective, not the single best trial.

    A config that only wins because it landed exactly on a spike (the
    reach=25 vs reach=25->150-is-flat situation the parent study flags) will
    have neighbours that score worse; a config in a genuine plateau won't.
    Ties the recommendation to the shape of the trial cloud, which is free
    once the search has run.
    """
    if not trials:
        return None
    coords = np.array([_normalize(t["params"]) for t in trials])
    clips = np.array([t["params"]["clip"] for t in trials])
    values = np.array([t["value"] for t in trials])
    smoothed = np.empty(len(trials))
    support = np.empty(len(trials), dtype=int)
    for i in range(len(trials)):
        neighborhood = (clips == clips[i]) & (np.linalg.norm(coords - coords[i], axis=1) <= radius)
        smoothed[i] = values[neighborhood].mean()
        support[i] = int(neighborhood.sum())
    best_i = int(np.argmax(smoothed))
    return trials[best_i], float(smoothed[best_i]), int(support[best_i])


def _fmt_params(params: dict) -> str:
    return (f"p={params['p']:.3f} reach={params['reach']:.2f} "
            f"q={params['q']:.3f} clip={params['clip']}")


def final_report(trials: list[dict], entries, out_dir: Path) -> None:
    print(f"\n{len(trials)} completed trials")
    if not trials:
        print("nothing checkpointed yet")
        return

    import csv
    with (out_dir / "optuna_trials.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "number", "value", "p", "reach", "q", "clip",
                          "s_all", "s_a", "s_b", "mean_f1_delta", "mean_ari_delta"])
        for t in trials:
            writer.writerow([t["seed"], t["number"], t["value"],
                              t["params"]["p"], t["params"]["reach"],
                              t["params"]["q"], t["params"]["clip"],
                              t["user_attrs"]["s_all"], t["user_attrs"]["s_a"],
                              t["user_attrs"]["s_b"], t["user_attrs"]["mean_f1_delta"],
                              t["user_attrs"]["mean_ari_delta"]])

    raw_best = max(trials, key=lambda t: t["value"])
    print(f"\nraw best trial   worker={raw_best['seed']}#{raw_best['number']}  "
          f"objective={raw_best['value']:+.4f}  {_fmt_params(raw_best['params'])}")
    print(f"  s_all={raw_best['user_attrs']['s_all']:+.4f} "
          f"s_a={raw_best['user_attrs']['s_a']:+.4f} s_b={raw_best['user_attrs']['s_b']:+.4f}")

    plateau = plateau_best(trials)
    trial, smoothed, support = plateau
    print(f"\nplateau-smoothed winner  worker={trial['seed']}#{trial['number']}  "
          f"smoothed_objective={smoothed:+.4f}  raw_objective={trial['value']:+.4f}  "
          f"support={support} neighbours (radius={PLATEAU_RADIUS})")
    print(f"  {_fmt_params(trial['params'])}")
    attrs = trial["user_attrs"]
    print(f"  s_all={attrs['s_all']:+.4f} s_a={attrs['s_a']:+.4f} s_b={attrs['s_b']:+.4f} "
          f"mean_f1_delta={attrs['mean_f1_delta']:+.4f} mean_ari_delta={attrs['mean_ari_delta']:+.4f}")

    # Compare the recommended config against the shipped stage, the earlier
    # (overfit) manual-grid pick, and doing nothing, on the same evaluate()
    # path anomaly_cost_gate_study.py uses -- so the numbers are directly
    # comparable to that script's --sweep headline output.
    variants = build_variants("headline", argparse.Namespace())
    variants = [v for v in variants if v[0] in ("noreassign", "edge_p2_s3", "cd_q4_r25")]
    variants.append(("optuna_recommended", cost_gate, dict(trial["params"])))
    rows = evaluate(entries, variants)
    report(rows, variants, "recommended config vs shipped / prior manual pick")
    report_by_dim(rows, variants)
    write_csv(rows, out_dir / "optuna_recommended_comparison.csv")

    (out_dir / "optuna_recommended_config.json").write_text(json.dumps({
        "params": trial["params"],
        "worker_seed": trial["seed"],
        "trial_number": trial["number"],
        "raw_objective": trial["value"],
        "smoothed_objective": smoothed,
        "plateau_support": support,
        **attrs,
    }, indent=2))
    print(f"\nWrote {out_dir}/optuna_recommended_config.json, "
          f"optuna_trials.csv, optuna_recommended_comparison.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, default=_DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--trials", type=int, default=4000,
                         help="total trials across all workers")
    parser.add_argument("--workers", type=int,
                         default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)),
                         help="worker processes (default: $SLURM_CPUS_PER_TASK or all cores)")
    parser.add_argument("--collect-only", action="store_true",
                         help="don't search -- just report on whatever's checkpointed in --out "
                              "(safe to run against an in-progress or finished job)")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    if not args.collect_only:
        print(f"{args.trials} trials, {args.workers} workers, cache={args.cache}", flush=True)
        run_search(args.cache, args.out, args.trials, args.workers)

    trials = collect_trials(args.out)
    entries = load_cache(args.cache)
    final_report(trials, entries, args.out)


if __name__ == "__main__":
    main()
