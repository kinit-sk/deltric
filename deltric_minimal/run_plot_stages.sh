#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Prefer an already-active Conda environment; otherwise use a colocated venv.
if [[ -z "${CONDA_PREFIX:-}" && -f "$SCRIPT_DIR/.venv/bin/activate" ]]; then
  source "$SCRIPT_DIR/.venv/bin/activate"
fi
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"
PYTHON_CACHE_TAG="$(python -c 'import sys; print(f"py{sys.version_info.major}{sys.version_info.minor}")')"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/deltric_numba_cache_${PYTHON_CACHE_TAG}}"
mkdir -p "$MPLCONFIGDIR"

# The default diagnostic uses the same seed construction and round-based,
# component-relative growth as run_plot_component_edge_redundancy.sh.  The
# legacy pointwise pipeline remains available through another DELTRIC_MODE.
DELTRIC_MODE="${DELTRIC_MODE:-growth}"
if [[ "$DELTRIC_MODE" == "component_growth" || "$DELTRIC_MODE" == "growth" ]]; then
  OUT="${OUT:-results/prune_stages_component_growth}"
  PROJECT_DIM="${PROJECT_DIM:-2}"
  DIM_REDUCTION="${DIM_REDUCTION:-umap}"
  UMAP_N_EPOCHS="${UMAP_N_EPOCHS:-100}"
  UMAP_N_NEIGHBORS="${UMAP_N_NEIGHBORS:-15}"
  # Epsilon is selected by run_sweep_hdbscan_epsilon.sh with the other
  # HDBSCAN settings left at their conventional defaults.
  HDBSCAN_MIN_CLUSTER_SIZE="${HDBSCAN_MIN_CLUSTER_SIZE:-10}"
  HDBSCAN_MIN_SAMPLES="${HDBSCAN_MIN_SAMPLES:-default}"
  HDBSCAN_SELECTION_METHOD="${HDBSCAN_SELECTION_METHOD:-eom}"
  HDBSCAN_CLUSTER_SELECTION_EPSILON="${HDBSCAN_CLUSTER_SELECTION_EPSILON:-0.0}"
  # Tuned separately on the held-out multi-D and native-2-D validation suites.
  HARD_SEED_LIMIT="${HARD_SEED_LIMIT:-0.75}"
  SEED_HARD_LIMIT="${SEED_HARD_LIMIT:-$HARD_SEED_LIMIT}"
  SEED_HARD_LIMIT_2D="${SEED_HARD_LIMIT_2D:-0.1}"
  # Prefer the first significant edge-length mode; use its robust median
  # fallback when no meaningful valley separates a denser seed core.
  SEED_BASELINE="${SEED_BASELINE:-first_mode}"
  # Native 2-D data bypasses UMAP entirely and needs a stricter original-space
  # gate. Multi-D data uses UMAP only for topology and has its own tuned pair.
  # Legacy HARD_GROWTH_LIMIT / PROJECTED_HARD_GROWTH_LIMIT remain multi-D
  # aliases for one-off overrides.
  HARD_GROWTH_LIMIT_MULTID="${HARD_GROWTH_LIMIT_MULTID:-${HARD_GROWTH_LIMIT:-1.6}}"
  PROJECTED_HARD_GROWTH_LIMIT_MULTID="${PROJECTED_HARD_GROWTH_LIMIT_MULTID:-${PROJECTED_HARD_GROWTH_LIMIT:-2.8}}"
  OUTLIER_COST_GATE_REACH_MULTID="${OUTLIER_COST_GATE_REACH_MULTID:-${OUTLIER_COST_GATE_REACH:-15.0}}"
  HARD_GROWTH_LIMIT_2D="${HARD_GROWTH_LIMIT_2D:-1.5}"
  PROJECTED_HARD_GROWTH_LIMIT_2D="${PROJECTED_HARD_GROWTH_LIMIT_2D:-1.5}"
  OUTLIER_COST_GATE_REACH_2D="${OUTLIER_COST_GATE_REACH_2D:-15.0}"
  GROWTH_INITIAL_RELATION="${GROWTH_INITIAL_RELATION:-union}"
  GROWTH_KNN="${GROWTH_KNN:-50}"
  GROWTH_MIN_EDGES="${GROWTH_MIN_EDGES:-10}"
  # Growth eligibility is intentionally a little less strict than final
  # component eligibility; final noise labeling still uses GROWTH_MIN_EDGES.
  GROWTH_SEED_MIN_EDGES="${GROWTH_SEED_MIN_EDGES:-10}"
  # knn_relaxed with the relaxed and strict seed limits equal is identical to
  # strict mode, so use the equivalent simpler mode explicitly.
  HARD_GATE_MODE="${HARD_GATE_MODE:-strict}"
  PROJECTED_HARD_LIMIT="${PROJECTED_HARD_LIMIT:-true}"
  MULTIPLICATIVE_HARD_LIMIT="${MULTIPLICATIVE_HARD_LIMIT:-false}"
  GOMORY_HU_CUT_SIZE="${GOMORY_HU_CUT_SIZE:-7}"
  GOMORY_HU_CUT_SIZE_2D="${GOMORY_HU_CUT_SIZE_2D:-8}"
  GOMORY_PRUNING_MODE="${GOMORY_PRUNING_MODE:-points}"
  GOMORY_POINT_CUT_SIZE="${GOMORY_POINT_CUT_SIZE:-4}"
  GOMORY_HU_MIN_COMPONENT_POINTS="${GOMORY_HU_MIN_COMPONENT_POINTS:-10}"
  GOMORY_HU_MIN_PARTITION_FRACTION="${GOMORY_HU_MIN_PARTITION_FRACTION:-0.2}"
  GOMORY_HU_HULL_RATIO_SKIP_THRESHOLD="${GOMORY_HU_HULL_RATIO_SKIP_THRESHOLD:-0.0}"
  # Apply contracted GH only to components with weighted short-neighbour M2
  # below this value. Set to "none" to disable the M2 gate.
  GOMORY_HU_M2_MAX="${GOMORY_HU_M2_MAX:-1.0}"
  # A component must also have this restored-reference original-space shape
  # score (100*sqrt(triangle area)/one-support boundary length) before GH runs.
  # Set to "none" to disable the gate.
  GOMORY_HU_SHAPE_SCORE_MIN="${GOMORY_HU_SHAPE_SCORE_MIN:-12}"
  # Skip GH for graph-like components with too little filled-triangle support.
  GOMORY_HU_MIN_TRIANGLE_EDGE_RATIO="${GOMORY_HU_MIN_TRIANGLE_EDGE_RATIO:-0.3}"
  # GH must use the completed real graph; contracting to growth-only edges
  # can report a low cut that does not split that real component.
  GOMORY_HU_GROWTH_EDGES_ONLY="${GOMORY_HU_GROWTH_EDGES_ONLY:-false}"
  # Boundary classification is intentionally post-growth only.  It never
  # changes seed restoration or growth; it limits the final GH candidates.
  POST_GROWTH_BOUNDARY_CLASSIFICATION="${POST_GROWTH_BOUNDARY_CLASSIFICATION:-true}"
  GOMORY_HU_BOUNDARY_EDGES_ONLY="${GOMORY_HU_BOUNDARY_EDGES_ONLY:-true}"
  # Primitive boundary + this many incident-edge layers. Two gives GH access
  # to the thin bridge interior immediately behind the visual boundary.
  BOUNDARY_EDGE_EXPANSION_HOPS="${BOUNDARY_EDGE_EXPANSION_HOPS:-2}"
  RESTORE_INTRA_COMPONENT_EDGES="${RESTORE_INTRA_COMPONENT_EDGES:-true}"
  RESTORE_HARD_LIMIT="${RESTORE_HARD_LIMIT:-2.0}"
  INTERIOR_REAL_RESTORATION="${INTERIOR_REAL_RESTORATION:-true}"
  # Experimental: restore only within each original seed before growth. This
  # prevents a temporary growth merge from creating protected restored bridges.
  SEED_ONLY_RESTORATION="${SEED_ONLY_RESTORATION:-false}"
  GROWTH_RESTORATION_MODE="${GROWTH_RESTORATION_MODE:-real}"
  # Experimental final phase; false preserves the original DelTriC graph and
  # labels exactly. When enabled, only unassigned points can be added.
  OUTLIER_GROWTH="${OUTLIER_GROWTH:-true}"
  # Default: prototype cost-gate reassignment. Frozen growth remains an
  # explicit fallback via OUTLIER_REASSIGNMENT=frozen_growth.
  OUTLIER_REASSIGNMENT="${OUTLIER_REASSIGNMENT:-cost_gate}"
  OUTLIER_COST_GATE_PENALTY_POWER="${OUTLIER_COST_GATE_PENALTY_POWER:-0.5}"
  # The previous implementation used the density ratio linearly.  Preserve
  # that behaviour now that DENSITY_POWER is a real exponent.
  OUTLIER_COST_GATE_DENSITY_POWER="${OUTLIER_COST_GATE_DENSITY_POWER:-0.5}"
  OUTLIER_COST_GATE_DENSITY_K="${OUTLIER_COST_GATE_DENSITY_K:-15}"
  OUTLIER_COST_GATE_DENSITY_CLIP="${OUTLIER_COST_GATE_DENSITY_CLIP:-false}"
  OUTLIER_GROWTH_HARD_LIMIT="${OUTLIER_GROWTH_HARD_LIMIT:-3.0}"
  PROJECTED_OUTLIER_GROWTH_HARD_LIMIT="${PROJECTED_OUTLIER_GROWTH_HARD_LIMIT:-3.0}"
  COMPONENT_GROWTH="${COMPONENT_GROWTH:-true}"
  SKIP_EXISTING="${SKIP_EXISTING:-false}"
  mkdir -p "$OUT"

  # Override DATA_GLOB for a focused diagnostic subset without changing the
  # curated default.  Set EXPECTED_DATASET_COUNT=0 to skip the count check.
  DATA_GLOB="${DATA_GLOB:-data/*.npz}"
  EXPECTED_DATASET_COUNT="${EXPECTED_DATASET_COUNT:-13}"
  EXPECTED_ARI_COUNT="${EXPECTED_ARI_COUNT:-12}"
  files=($DATA_GLOB)
  if [[ "$EXPECTED_DATASET_COUNT" != "0" && ${#files[@]} -ne "$EXPECTED_DATASET_COUNT" ]]; then
    echo "Expected ${EXPECTED_DATASET_COUNT} datasets matching ${DATA_GLOB}, found ${#files[@]}."
    exit 1
  fi

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
  if [[ "$RESTORE_HARD_LIMIT" != "none" && "$RESTORE_HARD_LIMIT" != "default" ]]; then
    RESTORE_ARGS+=(--restore-hard-limit "$RESTORE_HARD_LIMIT")
  fi
  if [[ "$SEED_ONLY_RESTORATION" == "true" ]]; then
    RESTORE_ARGS+=(--seed-only-restoration)
  else
    RESTORE_ARGS+=(--no-seed-only-restoration --growth-restoration-mode "$GROWTH_RESTORATION_MODE")
  fi
  GOMORY_GRAPH_ARGS=()
  if [[ "$GOMORY_HU_GROWTH_EDGES_ONLY" == "true" ]]; then
    GOMORY_GRAPH_ARGS+=(--gomory-hu-growth-edges-only)
  else
    GOMORY_GRAPH_ARGS+=(--no-gomory-hu-growth-edges-only)
  fi
  GOMORY_BOUNDARY_ARGS=()
  if [[ "$GOMORY_HU_BOUNDARY_EDGES_ONLY" == "true" ]]; then
    GOMORY_BOUNDARY_ARGS+=(--gomory-hu-boundary-edges-only)
  else
    GOMORY_BOUNDARY_ARGS+=(--no-gomory-hu-boundary-edges-only)
  fi
  BOUNDARY_DIAGNOSTIC_ARGS=()
  if [[ "$POST_GROWTH_BOUNDARY_CLASSIFICATION" == "true" ]]; then
    BOUNDARY_DIAGNOSTIC_ARGS+=(--post-growth-boundary-classification --restoration-boundary-plot)
  fi
  INTERIOR_RESTORATION_ARGS=()
  if [[ "$INTERIOR_REAL_RESTORATION" == "true" ]]; then
    INTERIOR_RESTORATION_ARGS+=(--interior-real-restoration)
  else
    INTERIOR_RESTORATION_ARGS+=(--no-interior-real-restoration)
  fi
  GOMORY_M2_ARGS=()
  if [[ "$GOMORY_HU_M2_MAX" != "none" && "$GOMORY_HU_M2_MAX" != "default" ]]; then
    GOMORY_M2_ARGS+=(--gomory-hu-m2-max "$GOMORY_HU_M2_MAX")
  fi
  GOMORY_SHAPE_ARGS=()
  if [[ "$GOMORY_HU_SHAPE_SCORE_MIN" != "none" && "$GOMORY_HU_SHAPE_SCORE_MIN" != "default" ]]; then
    GOMORY_SHAPE_ARGS+=(--gomory-hu-shape-score-min "$GOMORY_HU_SHAPE_SCORE_MIN")
  fi
  GOMORY_TRIANGLE_DENSITY_ARGS=()
  if [[ "$GOMORY_HU_MIN_TRIANGLE_EDGE_RATIO" != "none" && "$GOMORY_HU_MIN_TRIANGLE_EDGE_RATIO" != "default" ]]; then
    GOMORY_TRIANGLE_DENSITY_ARGS+=(--gomory-hu-min-triangle-edge-ratio "$GOMORY_HU_MIN_TRIANGLE_EDGE_RATIO")
  fi
  OUTLIER_GROWTH_ARGS=()
  if [[ "$OUTLIER_GROWTH" == "true" ]]; then
    OUTLIER_GROWTH_ARGS+=(--outlier-growth)
  else
    OUTLIER_GROWTH_ARGS+=(--no-outlier-growth)
  fi
  OUTLIER_COST_GATE_ARGS=()
  if [[ "$OUTLIER_COST_GATE_DENSITY_CLIP" == "true" ]]; then
    OUTLIER_COST_GATE_ARGS+=(--outlier-cost-gate-density-clip)
  else
    OUTLIER_COST_GATE_ARGS+=(--no-outlier-cost-gate-density-clip)
  fi
  COMPONENT_GROWTH_ARGS=()
  if [[ "$COMPONENT_GROWTH" != "true" ]]; then
    COMPONENT_GROWTH_ARGS+=(--no-component-growth)
  fi
  ARI_SUMMARIES=()
  for f in "${files[@]}"; do
    stem=$(basename "$f" .npz)
    if [[ "$SKIP_EXISTING" == "true" && -f "$OUT/${stem}.png" ]]; then
      if [[ "$stem" != *1000d* && -f "$OUT/${stem}.json" ]]; then
        ARI_SUMMARIES+=("$OUT/${stem}.json")
      fi
      continue
    fi
    python plot_seed_component_recomputed_growth.py \
      --data "$f" \
      --out "$OUT/${stem}.png" \
      --dim-reduction "$DIM_REDUCTION" \
      --project-dim "$PROJECT_DIM" \
      --umap-n-epochs "$UMAP_N_EPOCHS" \
      --umap-n-neighbors "$UMAP_N_NEIGHBORS" \
      --hdbscan-min-cluster-size "$HDBSCAN_MIN_CLUSTER_SIZE" \
      --hdbscan-min-samples "$HDBSCAN_MIN_SAMPLES" \
      --hdbscan-selection-method "$HDBSCAN_SELECTION_METHOD" \
      --hdbscan-cluster-selection-epsilon "$HDBSCAN_CLUSTER_SELECTION_EPSILON" \
      --seed-hard-limit "$SEED_HARD_LIMIT" \
      --seed-hard-limit-2d "$SEED_HARD_LIMIT_2D" \
      --seed-baseline "$SEED_BASELINE" \
      --hard-growth-limit "$HARD_GROWTH_LIMIT_MULTID" \
      --projected-hard-growth-limit "$PROJECTED_HARD_GROWTH_LIMIT_MULTID" \
      --hard-growth-limit-2d "$HARD_GROWTH_LIMIT_2D" \
      --projected-hard-growth-limit-2d "$PROJECTED_HARD_GROWTH_LIMIT_2D" \
      --initial-relation "$GROWTH_INITIAL_RELATION" \
      --component-growth-knn "$GROWTH_KNN" \
      --component-growth-min-edges "$GROWTH_MIN_EDGES" \
      --growth-seed-min-edges "$GROWTH_SEED_MIN_EDGES" \
      --hard-gate-mode "$HARD_GATE_MODE" \
      --no-redundancy-pruning \
      --gomory-hu-cut-size "$GOMORY_HU_CUT_SIZE" \
      --gomory-hu-cut-size-2d "$GOMORY_HU_CUT_SIZE_2D" \
      --gomory-pruning-mode "$GOMORY_PRUNING_MODE" \
      --gomory-point-cut-size "$GOMORY_POINT_CUT_SIZE" \
      --boundary-edge-expansion-hops "$BOUNDARY_EDGE_EXPANSION_HOPS" \
      --gomory-hu-min-component-points "$GOMORY_HU_MIN_COMPONENT_POINTS" \
      --gomory-hu-min-partition-fraction "$GOMORY_HU_MIN_PARTITION_FRACTION" \
      --gomory-hu-hull-ratio-skip-threshold "$GOMORY_HU_HULL_RATIO_SKIP_THRESHOLD" \
      "${GOMORY_M2_ARGS[@]}" \
      "${GOMORY_SHAPE_ARGS[@]}" \
      "${GOMORY_TRIANGLE_DENSITY_ARGS[@]}" \
      --outlier-reassignment "$OUTLIER_REASSIGNMENT" \
      --outlier-cost-gate-penalty-power "$OUTLIER_COST_GATE_PENALTY_POWER" \
      --outlier-cost-gate-reach "$OUTLIER_COST_GATE_REACH_MULTID" \
      --outlier-cost-gate-reach-2d "$OUTLIER_COST_GATE_REACH_2D" \
      --outlier-cost-gate-density-power "$OUTLIER_COST_GATE_DENSITY_POWER" \
      --outlier-cost-gate-density-k "$OUTLIER_COST_GATE_DENSITY_K" \
      --outlier-growth-hard-limit "$OUTLIER_GROWTH_HARD_LIMIT" \
      --projected-outlier-growth-hard-limit "$PROJECTED_OUTLIER_GROWTH_HARD_LIMIT" \
      "${GOMORY_GRAPH_ARGS[@]}" \
      "${GOMORY_BOUNDARY_ARGS[@]}" \
      "${OUTLIER_GROWTH_ARGS[@]}" \
      "${OUTLIER_COST_GATE_ARGS[@]}" \
      "${COMPONENT_GROWTH_ARGS[@]}" \
      --quiet \
      "${RESTORE_ARGS[@]}" \
      "${INTERIOR_RESTORATION_ARGS[@]}" \
      "${BOUNDARY_DIAGNOSTIC_ARGS[@]}" \
      "${PROJECTED_ARGS[@]}"
    if [[ "$stem" != *1000d* ]]; then
      ARI_SUMMARIES+=("$OUT/${stem}.json")
    fi
  done
  python - "$EXPECTED_ARI_COUNT" "${ARI_SUMMARIES[@]}" <<'PY'
import json
import sys

expected = int(sys.argv[1])
paths = sys.argv[2:]
aris = []
for path in paths:
    with open(path) as handle:
        value = json.load(handle).get("ari")
    if value is not None:
        aris.append(float(value))
if expected and len(aris) != expected:
    raise SystemExit(
        f"Expected ARI from {expected} non-1000D datasets, found {len(aris)}."
    )
if aris:
    print(f"Average ARI over {len(aris)} non-1000D datasets: {sum(aris) / len(aris):.4f}")
PY
fi
