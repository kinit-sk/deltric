#!/usr/bin/env python3
"""Compare isolated source snapshots on identical inputs, with exact stage hashes.

Example (pin OMP/BLAS/NUMBA threads to one in the caller):
  python verify_quadratic_fix.py --baseline /path/to/before --candidate /path/to/after \
      --data dataset.npz --out results/equivalence/dataset

The baseline directory must be an untouched copy of all estimator modules.
Both variants run in fresh processes; saved labels are compared with array_equal.
Stage hashes include dtype, shape and exact bytes, including floating-point bits.
Use --limit 50000 for development data; sampling uses holdout_benchmark's seed-0
stratification. Prepared D72 inputs should be passed without --limit.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np


def fingerprint(value):
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "shape": list(value.shape),
                "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
    if isinstance(value, dict):
        return {str(k): fingerprint(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [fingerprint(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value).hex()
    if isinstance(value, np.generic):
        return value.item()
    return value


def worker(code: Path, data: Path, out: Path):
    sys.path.insert(0, str(code.resolve()))
    import plot_seed_component_recomputed_growth as pipeline
    from deltric import DelTriC
    stages = []
    for name in ("_sequential_growth", "_bounded_intra_component_restore_mask",
                 "_component_topology_metrics", "_add_component_area_perimeter_metrics",
                 "_candidate_vertex_cut_mask", "_assign_anomalies_cost_gate"):
        original = getattr(pipeline, name)

        def wrap(fn, stage):
            @functools.wraps(fn)
            def run(*args, **kwargs):
                start = time.perf_counter()
                result = fn(*args, **kwargs)
                seconds = time.perf_counter() - start
                payload = args[0] if stage == "_add_component_area_perimeter_metrics" else result
                stages.append({"stage": stage, "seconds": seconds,
                               "output": fingerprint(payload)})
                return result
            return run
        setattr(pipeline, name, wrap(original, name))
    with np.load(data, allow_pickle=False) as z:
        X = z["X"]
    start = time.perf_counter()
    labels = DelTriC().fit_predict(X)
    elapsed = time.perf_counter() - start
    np.save(out.with_suffix(".npy"), labels)
    record = {"n": len(X), "seconds": elapsed,
              "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "labels": fingerprint(labels), "stages": stages,
              "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(code.glob("*.py"))}}
    out.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ("n", "seconds", "peak_rss_kib")}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--candidate", required=True, type=Path)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--worker", action="store_true")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True) if not a.worker else a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.worker:
        worker(a.candidate, a.data, a.out)
        return
    data = a.data.resolve()
    if a.limit:
        from holdout_benchmark import sample_indices
        with np.load(data, allow_pickle=False) as z:
            indices = sample_indices(z["y"], a.limit, 0)
            np.savez(a.out / "input.npz", X=z["X"][indices], indices=indices)
        data = (a.out / "input.npz").resolve()
    variants = [("before", a.baseline)] if a.baseline else []
    variants.append(("after", a.candidate))
    for name, code in variants:
        with (a.out / f"{name}.log").open("w") as log:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker",
                            "--candidate", str(code.resolve()), "--data", str(data),
                            "--out", str((a.out / name).resolve())],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    after = json.loads((a.out / "after.json").read_text())
    result = {"n": after["n"], "after_seconds": after["seconds"]}
    if a.baseline:
        before = json.loads((a.out / "before.json").read_text())
        result.update(before_seconds=before["seconds"],
                      speedup=before["seconds"] / after["seconds"],
                      labels_equal=bool(np.array_equal(np.load(a.out / "before.npy"),
                                                       np.load(a.out / "after.npy"))),
                      stages_equal=[(s["stage"], s["output"]) for s in before["stages"]]
                      == [(s["stage"], s["output"]) for s in after["stages"]])
    (a.out / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if a.baseline and not (result["labels_equal"] and result["stages_equal"]):
        raise SystemExit("FAIL: baseline and candidate differ")


if __name__ == "__main__":
    main()
