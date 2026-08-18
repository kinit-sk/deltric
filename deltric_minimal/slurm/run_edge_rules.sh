#!/bin/bash
# Submit with: sbatch slurm/run_edge_rules.sh
# Run from ~/deltric_minimal on perun (see slurm/run_edge_rule_ari.sh for the
# venv setup — same numpy/scipy/scikit-learn/joblib stack, no torch needed).
#
# edge_rule_extraction.py (in studies/) imports edge_pruning_study.py from the
# sibling kinit repo. Check out kinit next to deltric_minimal on perun too, or
# set KINIT_REPO to its path.
#
# This job needs kinit's results/edge_dump/ to be present. It is produced by
# kinit's edge_pruning_study.py --dump-edges and is ~99 MB, so push it from
# the laptop:
#   rsync -avz /path/to/kinit/results/edge_dump/ perun:~/kinit/results/edge_dump/
#
# Unlike the laptop runs, this one uses --max-edges-per-run 0, i.e. every edge
# of every labelled run rather than a subsample. That is the whole point of
# moving here: the subsample existed only to make the fits finish locally.
#
#SBATCH --job-name=edge_rules
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=256
#SBATCH --mem=0
#SBATCH --time=2:00:00

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs

echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $(hostname)"
echo "CPUs:     $SLURM_CPUS_PER_TASK"
echo "Mem:      ${SLURM_MEM_PER_NODE:-unset} MB (0 = whole node)"

KINIT_REPO="${KINIT_REPO:-../kinit}"
DUMP_DIR="$KINIT_REPO/results/edge_dump"
if [ ! -d "$DUMP_DIR" ]; then
    echo "ERROR: $DUMP_DIR is missing. See the rsync line in this script's header, or set KINIT_REPO." >&2
    exit 1
fi
echo "Dump:     $(ls "$DUMP_DIR"/*.npz | wc -l) runs, $(du -sh "$DUMP_DIR" | cut -f1)"

# Each worker fits its own model, so the BLAS threads inside a fit would
# oversubscribe the node 256-fold. Pin them to 1 and let joblib own the cores.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export KINIT_REPO

.venv/bin/python -u studies/edge_rule_extraction.py \
    --dump "$DUMP_DIR" \
    --out studies/results/edge_rules \
    --max-edges-per-run 0 \
    --jobs "$SLURM_CPUS_PER_TASK"

echo "Done."
