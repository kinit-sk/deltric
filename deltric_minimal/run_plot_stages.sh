#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# An activated Conda environment takes precedence.  Otherwise, prefer a
# colocated virtual environment, then the development project's environment.
# If neither exists, use Python from PATH.
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  for activate in \
    "$SCRIPT_DIR/.venv/bin/activate" \
    "$SCRIPT_DIR/../deltric/execution/bench-venv/bin/activate"
  do
    if [[ -f "$activate" ]]; then
      source "$activate"
      break
    fi
  done
fi
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
mkdir -p "$MPLCONFIGDIR"

# Select the implementation used both for clustering and for the diagnostic
# plots.  The default remains the legacy pipeline; component_growth is an
# explicit opt-in so existing runs are unchanged.
DELTRIC_MODE="${DELTRIC_MODE:-growth}"
if [[ "$DELTRIC_MODE" == "component_growth" || "$DELTRIC_MODE" == "growth" ]]; then
  OUT="${OUT:-results/prune_stages_component_growth}"
  PROJECT_DIM="${PROJECT_DIM:-2}"
  DIM_REDUCTION="${DIM_REDUCTION:-umap}"
  UMAP_N_EPOCHS="${UMAP_N_EPOCHS:-100}"
  UMAP_N_NEIGHBORS="${UMAP_N_NEIGHBORS:-15}"
  SEED_HARD_LIMIT="${SEED_HARD_LIMIT:-0.25}"
  HARD_LIMIT="${HARD_LIMIT:-0.9}"
  GROWTH_HARD_LIMIT="${GROWTH_HARD_LIMIT:-$HARD_LIMIT}"
  MILD_LIMIT="${MILD_LIMIT:-0.5}"
  GROWTH_INITIAL_RELATION="${GROWTH_INITIAL_RELATION:-union}"
  GROWTH_RELATION="${GROWTH_RELATION:-union}"
  GROWTH_KNN="${GROWTH_KNN:-50}"
  GROWTH_MIN_EDGES="${GROWTH_MIN_EDGES:-10}"
  HARD_GATE_MODE="${HARD_GATE_MODE:-knn_relaxed}"
  SEED_RELAXED_HARD_LIMIT="${SEED_RELAXED_HARD_LIMIT:-0.75}"
  GROWTH_RELAXED_HARD_LIMIT="${GROWTH_RELAXED_HARD_LIMIT:-1.7}"
LOCAL_KNN_SELECTIVITY_THRESHOLD="${LOCAL_KNN_SELECTIVITY_THRESHOLD:-0.1}"
LOCAL_KNN_SELECTIVITY_SCOPE="${LOCAL_KNN_SELECTIVITY_SCOPE:-edge}"
  BRIDGE_PRUNING="${BRIDGE_PRUNING:-false}"
  BRIDGE_MAX_HOPS="${BRIDGE_MAX_HOPS:-10}"
  OUTLIER_LIMIT="${OUTLIER_LIMIT:-1.0}"
  OPEN_SPACE_RELAXATION="${OPEN_SPACE_RELAXATION:-false}"
  OUTER_LONG_RATIO="${OUTER_LONG_RATIO:-3.0}"
  OUTER_RELAXATION="${OUTER_RELAXATION:-2.0}"
  OUTER_TRANSITION_WIDTH="${OUTER_TRANSITION_WIDTH:-0.4054651081}"
  PROJECTED_HARD_LIMIT="${PROJECTED_HARD_LIMIT:-true}"
  MULTIPLICATIVE_HARD_LIMIT="${MULTIPLICATIVE_HARD_LIMIT:-false}"
  SKIP_EXISTING="${SKIP_EXISTING:-false}"
  mkdir -p "$OUT"

  files=(data/*.npz)
  echo "Running over ${#files[@]} dataset(s) in data/."

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
  BRIDGE_ARGS=(--bridge-max-hops "$BRIDGE_MAX_HOPS")
  if [[ "$BRIDGE_PRUNING" == "true" ]]; then
    BRIDGE_ARGS+=(--bridge-pruning)
  fi
  OUTLIER_ARGS=()
  if [[ -n "$OUTLIER_LIMIT" && "$OUTLIER_LIMIT" != "none" ]]; then
    OUTLIER_ARGS+=(--component-growth-outlier-limit "$OUTLIER_LIMIT")
  fi
  OPEN_SPACE_ARGS=(
    --outer-long-ratio "$OUTER_LONG_RATIO"
    --outer-relaxation "$OUTER_RELAXATION"
    --outer-transition-width "$OUTER_TRANSITION_WIDTH"
  )
  if [[ "$OPEN_SPACE_RELAXATION" == "true" ]]; then
    OPEN_SPACE_ARGS+=(--open-space-relaxation)
  fi

  for f in "${files[@]}"; do
    stem=$(basename "$f" .npz)
    if [[ "$SKIP_EXISTING" == "true" && -f "$OUT/${stem}.png" ]]; then
      continue
    fi
    python plot_prunning_stages_2x2_growth.py \
      --data "$f" \
      --out "$OUT/${stem}.png" \
      --dim-reduction "$DIM_REDUCTION" \
      --project-dim "$PROJECT_DIM" \
      --umap-n-epochs "$UMAP_N_EPOCHS" \
      --umap-n-neighbors "$UMAP_N_NEIGHBORS" \
      --hard-limit "$HARD_LIMIT" \
      --seed-hard-limit "$SEED_HARD_LIMIT" \
      --growth-hard-limit "$GROWTH_HARD_LIMIT" \
      --mild-limit "$MILD_LIMIT" \
      --initial-relation "$GROWTH_INITIAL_RELATION" \
      --growth-relation "$GROWTH_RELATION" \
      --component-growth-knn "$GROWTH_KNN" \
      --component-growth-min-edges "$GROWTH_MIN_EDGES" \
      --hard-gate-mode "$HARD_GATE_MODE" \
      --seed-relaxed-hard-limit "$SEED_RELAXED_HARD_LIMIT" \
      --growth-relaxed-hard-limit "$GROWTH_RELAXED_HARD_LIMIT" \
      --local-knn-selectivity-threshold "$LOCAL_KNN_SELECTIVITY_THRESHOLD" \
      --local-knn-selectivity-scope "$LOCAL_KNN_SELECTIVITY_SCOPE" \
      "${BRIDGE_ARGS[@]}" \
      "${OUTLIER_ARGS[@]}" \
      "${OPEN_SPACE_ARGS[@]}" \
      "${PROJECTED_ARGS[@]}"
  done
  exit 0
fi

BASELINE="${BASELINE:-false}"
BASELINE_SCORE="${BASELINE_SCORE:-prune_score}"
BASELINE_THRESHOLD="${BASELINE_THRESHOLD:-0.70}"
MILD_POWER="${MILD_POWER:-2.0}"
if [[ "$BASELINE" == "true" ]]; then
  OUT="results/prune_stages_baseline_${BASELINE_SCORE}_thr${BASELINE_THRESHOLD}"
  BASELINE_ARGS=(--baseline --baseline-score "$BASELINE_SCORE" --baseline-threshold "$BASELINE_THRESHOLD")
else
  OUT=results/prune_stages_all_hlim1p5_mlim--0p5_nuclei
  BASELINE_ARGS=()
fi
mkdir -p "$OUT"
MLP_FEATURE_WORKERS="${MLP_FEATURE_WORKERS:-1}"
POINT_EDGE_COMPARE="${POINT_EDGE_COMPARE:-nuclei_rule}"
NUCLEI_RULE_A="${NUCLEI_RULE_A:-0.8}"
NUCLEI_QUANTILE="${NUCLEI_QUANTILE:-0.8}"
NUCLEI_TYPE="${NUCLEI_TYPE:-blue}"
DENSITY_NEIGHBORS="${DENSITY_NEIGHBORS:-10}"
NUCLEUS_MIN_EDGES="${NUCLEUS_MIN_EDGES:-20}"
NUCLEUS_PEAK_PERCENTILE="${NUCLEUS_PEAK_PERCENTILE:-70}"
NUCLEUS_NMS_RADIUS="${NUCLEUS_NMS_RADIUS:-0.9}"
NUCLEUS_PEAK_DENSITY_FLOOR="${NUCLEUS_PEAK_DENSITY_FLOOR:-30}"
NUCLEUS_PEAK_SPACE="${NUCLEUS_PEAK_SPACE:-original}"

files=(data/*.npz)
echo "Running over ${#files[@]} dataset(s) in data/."

for f in "${files[@]}"; do
  stem=$(basename "$f" .npz)
  python plot_pruning_stages_2x2.py \
    --data "$f" \
    --out "$OUT/${stem}.png" \
    --prune-mode pointwise_dual_space \
    --prune-param 1.0 \
    --hard-limit 1.5 \
    --mild-limit -0.5 \
    --mild-power "$MILD_POWER" \
    --projected-hard-limit \
    --point-edge-ratio -0.0 \
    --point-edge-compare "$POINT_EDGE_COMPARE" \
    --nuclei-rule-a "$NUCLEI_RULE_A" \
    --nuclei-quantile "$NUCLEI_QUANTILE" \
    --nuclei-type "$NUCLEI_TYPE" \
    --point-edge-subset all \
    --neighbor-stat median \
    --prune-regime none \
    --anomaly-sensitivity 0.1 \
    --multiplicative-hard-limit \
    --auto-threshold \
    --dim-reduction umap \
    --project-dim 2 \
    --umap-n-epochs 100 \
    --umap-n-neighbors 15 \
    --mlp-feature-workers "$MLP_FEATURE_WORKERS" \
    --density-neighbors "$DENSITY_NEIGHBORS" \
    --nucleus-min-edges "$NUCLEUS_MIN_EDGES" \
    --nucleus-peak-percentile "$NUCLEUS_PEAK_PERCENTILE" \
    --nucleus-nms-radius "$NUCLEUS_NMS_RADIUS" \
    --nucleus-peak-density-floor "$NUCLEUS_PEAK_DENSITY_FLOOR" \
    --nucleus-peak-space "$NUCLEUS_PEAK_SPACE" \
    "${BASELINE_ARGS[@]}"
done
