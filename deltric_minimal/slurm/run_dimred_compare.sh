#!/bin/bash
# Submit with: sbatch slurm/run_dimred_compare.sh
# DelTriC with UMAP vs DelTriC with t-SNE projection (run_plot_stages.sh
# defaults otherwise) vs HDBSCAN, over every non-holdout multi-D dataset.
# 2-D data bypasses the projection, so it is skipped. Resumable (SKIP_EXISTING).
#
# Results: results/dimred_compare/{umap,tsne}/<data subdir>/<stem>.json
# Summarize: .venv/bin/python studies/summarize_dimred_compare.py
#
#SBATCH --job-name=deltric_dimred_compare
#SBATCH --chdir=/mnt/home/maseke423/deltric_minimal
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=12:00:00

set -euo pipefail
mkdir -p slurm/logs results/dimred_compare

export MPLCONFIGDIR="${TMPDIR:-/tmp}/mplconfig"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_cache"
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1

# Multi-D, non-holdout datasets.
.venv/bin/python - > results/dimred_compare/datasets.txt <<'PY'
import glob
import numpy as np
paths = sorted(glob.glob("data/*.npz") + glob.glob("data/other/*.npz")
               + glob.glob("data/testing/*/*.npz"))
for p in paths:
    if "holdout" in p.split("/"):
        continue
    if np.load(p)["X"].shape[1] > 2:
        print(p)
PY
echo "Datasets: $(wc -l < results/dimred_compare/datasets.txt)"

run_one() {
  method=$1 f=$2
  out="results/dimred_compare/$method/$(dirname "${f#data/}")"
  mkdir -p "$out"
  if ! DIM_REDUCTION="$method" OUT="$out" DATA_GLOB="$f" SKIP_EXISTING=true \
       EXPECTED_DATASET_COUNT=0 EXPECTED_ARI_COUNT=0 \
       timeout 3h bash run_plot_stages.sh > /dev/null 2>"$out/$(basename "$f" .npz).err"; then
    echo "FAIL $method $f"
  else
    rm -f "$out/$(basename "$f" .npz).err"
  fi
}
export -f run_one

for m in umap tsne; do sed "s|^|$m |" results/dimred_compare/datasets.txt; done \
  | xargs -P "$SLURM_CPUS_PER_TASK" -L 1 bash -c 'run_one "$0" "$1"'

.venv/bin/python studies/summarize_dimred_compare.py
