#!/bin/bash
# Submit with: sbatch slurm/run_edge_rule_ari.sh
# Run from ~/deltric_minimal on perun.
#
# Before first use, on the LOGIN node (compute nodes have no uv):
#   uv venv --python 3.10 .venv
#   uv pip install --python .venv/bin/python numpy==2.2.6 scipy==1.15.3 \
#       scikit-learn==1.7.2 umap-learn==0.5.9.post2 numba matplotlib==3.10.7 \
#       networkx==3.3 pandas==2.3.3 statsmodels==0.14.5 joblib
#   uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
#
# torch is needed only because utils_pruning.py defines an nn.Module subclass at
# module scope outside its own `try: import torch` guard (utils_pruning.py:28 vs
# :1021), so the module cannot be imported without it.
#
#SBATCH --job-name=edge_rule_ari
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=2:00:00

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs results

echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $(hostname)"
echo "CPUs:     $SLURM_CPUS_PER_TASK"

export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplconfig"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_cache"
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"

# There are only 48 runs (12 datasets x 4 variants) and each is largely a
# single-threaded UMAP + growth pass, so parallelism goes across runs.  Pin the
# BLAS/numba threads to 1 or the workers oversubscribe the node.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

.venv/bin/python -u studies/eval_edge_rule_ari.py \
    --data data \
    --out studies/results/edge_rule_ari \
    --jobs 24

echo "Done."
