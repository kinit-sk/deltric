#!/bin/bash
# Submit with: sbatch slurm/run_full_analysis.sh
# Run from ~/deltric_minimal on perun (venv as in slurm/run_edge_rule_ari.sh;
# needs umap-learn + matplotlib + pillow, all already in .venv there).
#
# Runs the current DelTriC pipeline (run_plot_stages.sh defaults) over every
# dataset under data/ except data/holdout/**, plus the BBC NLP embeddings,
# and writes metrics, stage images, growth GIFs and anomaly GIFs -- see
# studies/perun_full_analysis.py's module docstring. Resumable: resubmitting
# skips datasets that already have results/full_analysis/per_dataset/*.json.
#
# Fetch results to the laptop afterwards:
#   rsync -avz perun:~/deltric_minimal/results/full_analysis/ results/full_analysis/
#
#SBATCH --job-name=deltric_full_analysis
#SBATCH --chdir=/mnt/home/maseke423/deltric_minimal
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=martin.senkyr@intern.kinit.sk
# No --time: cpu_short's own cap (2 days) applies.

set -euo pipefail

OUT=results/full_analysis
mkdir -p slurm/logs "$OUT"

echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $(hostname)"
echo "PWD:      $(pwd)"
echo "CPUs:     $SLURM_CPUS_PER_TASK"

export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplconfig"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_cache"
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"

# One dataset per worker process; pin BLAS/numba threads so the 64 workers
# don't oversubscribe the node.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

.venv/bin/python -u studies/perun_full_analysis.py \
    --out "$OUT" \
    --jobs "$SLURM_CPUS_PER_TASK" \
    --resume

echo "Done. Open $OUT/index.html"
