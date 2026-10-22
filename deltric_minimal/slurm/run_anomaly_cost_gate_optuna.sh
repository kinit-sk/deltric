#!/bin/bash
# Submit with: sbatch slurm/run_anomaly_cost_gate_optuna.sh
# Run from ~/deltric_minimal on perun (see slurm/run_edge_rule_ari.sh for the
# base venv setup -- same numpy/scipy/scikit-learn stack, no torch/umap
# needed at search time). Additionally needs optuna, not in that list yet:
#   uv pip install --python .venv/bin/python optuna
#
# Needs studies/results/anomaly_cost_gate/cache/ populated first -- it's the
# frozen per-dataset graphs anomaly_cost_gate_study.py builds with
# --build-cache. That step needs umap-learn and the full component-growth
# pipeline; the search itself doesn't. Cheapest path is to build it on the
# laptop (`python studies/anomaly_cost_gate_study.py --build-cache`, ~6 min)
# and push the ~5MB result rather than rebuilding here:
#   rsync -avz studies/results/anomaly_cost_gate/cache/ \
#       perun:~/deltric_minimal/studies/results/anomaly_cost_gate/cache/
#
# See studies/anomaly_cost_gate_optuna.py's module docstring for what's being
# optimized and why (robustness objective, not mean F1).
#
#SBATCH --job-name=cost_gate_optuna
#SBATCH --chdir=/mnt/home/maseke423/deltric_minimal
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=martin.senkyr@intern.kinit.sk
# No --time: falls back to cpu_short's own cap (2 days, `sinfo -p cpu_short`).
# That's the real ceiling regardless of what's set here -- there's no way to
# ask Slurm for "no limit" -- but it's far more than this needs. The run
# checkpoints every completed trial to disk per-worker (see the script's
# module docstring), so there's no reason to want a *tighter* limit either:
# unlike the v1 shared-storage design, killing this run mid-flight costs you
# nothing but the trials still in flight when it dies.

set -euo pipefail

mkdir -p slurm/logs studies/results/anomaly_cost_gate_optuna

echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $(hostname)"
echo "PWD:      $(pwd)"
echo "CPUs:     $SLURM_CPUS_PER_TASK"

export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplconfig"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_cache"
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"

# One worker process per core, each replaying cost_gate (pure numpy/heapq,
# no BLAS-heavy work) over the frozen cache -- pin threads to 1 or the 64
# worker processes oversubscribe the node against each other.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

# Measured ~8-10s/trial single-threaded for all 100 datasets (a laptop core,
# the q>0 density branch; not yet re-measured on perun hardware under the v2
# no-contention design, so treat the estimate below as approximate). v1 of
# this job shared one Optuna storage file across all 64 workers and hit lock
# contention so bad it never got past ~8300/20000 trials in 90 minutes -- see
# the script's module docstring for why. v2 runs independent per-worker
# studies with no shared state, so there's no contention ceiling: expect
# ~7 trials/s aggregate at 64 workers, i.e. --trials 30000 in roughly
# 70-90 min wall clock (~75-95 core-hours at 64 cpus). Deliberately not set
# much higher than that: the recovered v1 run already had its winning basin
# supported by 1144 of its 8361 trials, i.e. the search plateaus well before
# 30000 -- more trials past that buys diminishing precision, not a
# meaningfully different config. Check progress any time without touching
# the running job (see --collect-only below); --mail-user above pings when
# the job itself ends (completed OR failed).
.venv/bin/python -u studies/anomaly_cost_gate_optuna.py \
    --cache studies/results/anomaly_cost_gate/cache \
    --out studies/results/anomaly_cost_gate_optuna \
    --trials 30000 \
    --workers "$SLURM_CPUS_PER_TASK"

echo "Done."

# Check progress without touching the running job, from the login node:
#   .venv/bin/python studies/anomaly_cost_gate_optuna.py --collect-only \
#       --out studies/results/anomaly_cost_gate_optuna
