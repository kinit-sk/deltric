#!/bin/bash
# Submit with: sbatch slurm/run_dijkstra_timing.sh
# Run from ~/deltric_minimal on perun (see slurm/run_edge_rule_ari.sh for the
# venv setup -- same numpy/scipy/scikit-learn/umap/joblib stack, no torch
# needed since this only touches utils_component_growth.py).
#
# Measures how much wall-clock time the Dijkstra anomaly-reassignment stage
# (utils_component_growth._assign_anomalies_dijkstra) adds on top of the
# pre-existing component-growth pipeline, across all 148 curated datasets
# under data/.
#
#SBATCH --job-name=dijkstra_timing
#SBATCH --chdir=/mnt/home/maseke423/deltric_minimal
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=2:00:00

set -euo pipefail

# --chdir above pins the run to the repo root regardless of which directory
# `sbatch` was invoked from (a plain `cd "$SLURM_SUBMIT_DIR"` broke when the
# job was submitted from inside slurm/ itself: it landed in the wrong
# directory, `.venv/bin/python` and `studies/...` didn't resolve, and the
# job died in under a second with exit 127 before any log file existed at
# the expected path).
mkdir -p slurm/logs studies/results

echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $(hostname)"
echo "PWD:      $(pwd)"
echo "CPUs:     $SLURM_CPUS_PER_TASK"

export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplconfig"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_cache"
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"

# Each worker runs its own single-threaded UMAP + growth pass; parallelism
# goes across datasets via joblib, so pin BLAS/numba threads to 1 or workers
# oversubscribe the node.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

# --jobs is kept below the core count (rather than =64) because this is a
# timing measurement: heavier contention adds noise to the per-run wall
# clock. --repeats takes the median per dataset to further damp that noise.
.venv/bin/python -u studies/dijkstra_timing_study.py \
    --data data \
    --out studies/results/dijkstra_timing \
    --repeats 5 \
    --jobs 32

.venv/bin/python -u studies/plot_dijkstra_timing.py \
    --csv studies/results/dijkstra_timing/dijkstra_timing.csv \
    --out studies/results/dijkstra_timing/dijkstra_timing.png

echo "Done."
