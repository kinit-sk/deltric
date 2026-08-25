#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# An active Conda environment takes precedence.  Otherwise use a colocated
# virtual environment when present, then fall back to Python on PATH.
if [[ -z "${CONDA_PREFIX:-}" && -f "$SCRIPT_DIR/.venv/bin/activate" ]]; then
  source "$SCRIPT_DIR/.venv/bin/activate"
fi
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"

# Effective component-growth setup shared with the current development
# run_plot_stages.sh and run_plot_component_edge_redundancy.sh.
OUT="${OUT:-results/prune_stages_component_growth}"
PROJECT_DIM="${PROJECT_DIM:-2}"
DIM_REDUCTION="${DIM_REDUCTION:-umap}"
UMAP_N_EPOCHS="${UMAP_N_EPOCHS:-100}"
UMAP_N_NEIGHBORS="${UMAP_N_NEIGHBORS:-15}"
HARD_SEED_LIMIT="${HARD_SEED_LIMIT:-0.0}"
SEED_HARD_LIMIT="${SEED_HARD_LIMIT:-$HARD_SEED_LIMIT}"
# Joint ARI sweep with all eligible Gomory--Hu cuts (size <= 6) enabled.
HARD_GROWTH_LIMIT="${HARD_GROWTH_LIMIT:-2.0}"
PROJECTED_HARD_GROWTH_LIMIT="${PROJECTED_HARD_GROWTH_LIMIT:-1.25}"
GROWTH_INITIAL_RELATION="${GROWTH_INITIAL_RELATION:-union}"
GROWTH_KNN="${GROWTH_KNN:-50}"
GROWTH_MIN_EDGES="${GROWTH_MIN_EDGES:-10}"
PROJECTED_HARD_LIMIT="${PROJECTED_HARD_LIMIT:-true}"
MULTIPLICATIVE_HARD_LIMIT="${MULTIPLICATIVE_HARD_LIMIT:-false}"
RESTORE_INTRA_COMPONENT_EDGES="${RESTORE_INTRA_COMPONENT_EDGES:-true}"
GOMORY_HU_CUT_SIZE="${GOMORY_HU_CUT_SIZE:-6}"
GOMORY_HU_MIN_COMPONENT_POINTS="${GOMORY_HU_MIN_COMPONENT_POINTS:-10}"
# A positive value protects a non-compact component from GH cuts when its
# weighted outer-hull/non-hull ratio reaches this value.  0 disables the gate.
GOMORY_HU_HULL_RATIO_SKIP_THRESHOLD="${GOMORY_HU_HULL_RATIO_SKIP_THRESHOLD:-0.0}"
SKIP_EXISTING="${SKIP_EXISTING:-false}"
mkdir -p "$OUT"

PROJECTED_ARGS=()
if [[ "$PROJECTED_HARD_LIMIT" == "true" ]]; then
  PROJECTED_ARGS+=(--projected-hard-limit)
else
  PROJECTED_ARGS+=(--no-projected-hard-limit)
fi
if [[ "$MULTIPLICATIVE_HARD_LIMIT" == "true" ]]; then
  PROJECTED_ARGS+=(--multiplicative-hard-limit)
else
  PROJECTED_ARGS+=(--no-multiplicative-hard-limit)
fi
RESTORE_ARGS=()
if [[ "$RESTORE_INTRA_COMPONENT_EDGES" == "true" ]]; then
  RESTORE_ARGS+=(--restore-intra-component-edges)
else
  RESTORE_ARGS+=(--no-restore-intra-component-edges)
fi

files=(data/*.npz)
if [[ ${#files[@]} -ne 12 ]]; then
  echo "Expected exactly 12 curated datasets in data, found ${#files[@]}."
  exit 1
fi

for file in "${files[@]}"; do
  stem=$(basename "$file" .npz)
  if [[ "$SKIP_EXISTING" == "true" && -f "$OUT/${stem}.png" ]]; then
    continue
  fi
  python plot_seed_component_recomputed_growth.py \
    --data "$file" --out "$OUT/${stem}.png" \
    --dim-reduction "$DIM_REDUCTION" --project-dim "$PROJECT_DIM" \
    --umap-n-epochs "$UMAP_N_EPOCHS" --umap-n-neighbors "$UMAP_N_NEIGHBORS" \
    --seed-hard-limit "$SEED_HARD_LIMIT" \
    --hard-growth-limit "$HARD_GROWTH_LIMIT" \
    --projected-hard-growth-limit "$PROJECTED_HARD_GROWTH_LIMIT" \
    --initial-relation "$GROWTH_INITIAL_RELATION" \
    --component-growth-knn "$GROWTH_KNN" \
    --component-growth-min-edges "$GROWTH_MIN_EDGES" \
    --hard-gate-mode strict \
    --no-redundancy-pruning \
    --gomory-hu-cut-size "$GOMORY_HU_CUT_SIZE" \
    --gomory-hu-min-component-points "$GOMORY_HU_MIN_COMPONENT_POINTS" \
    --gomory-hu-hull-ratio-skip-threshold "$GOMORY_HU_HULL_RATIO_SKIP_THRESHOLD" \
    --quiet "${RESTORE_ARGS[@]}" "${PROJECTED_ARGS[@]}"
done
