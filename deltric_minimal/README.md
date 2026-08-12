# Minimal DelTriC component-growth plots

This directory is a self-contained copy of the component-growth plotting
workflow and the 12 curated input datasets.  The graph is built in UMAP space;
edge-size decisions are computed in the original space.

## Setup

Install the required Conda environment:

```bash
conda env create -f environment.yml
conda activate deltric-minimal
```

## Run

Run every dataset with the current defaults:

```bash
./run_plot_stages.sh
```

Plots are written under `results/prune_stages_component_growth/` by default.
All options are environment variables; for example:

```bash
UMAP_N_NEIGHBORS=30 GROWTH_KNN=50 HARD_LIMIT=1.2 ./run_plot_stages.sh
```

The runnable files are:

- `run_plot_stages.sh` — batch launcher and exposed parameters.
- `plot_prunning_stages_2x2_growth.py` — visualization wrapper; it calls the
  same component-growth backend used by the algorithm.
- `utils_pruning.py` — canonical UMAP, triangulation, and original-space
  edge-length helpers (renamed from `utils.py`).
- `utils_component_growth.py` — seed construction and component-growth logic.
- `data/` — the 12 input `.npz` datasets and their summaries.

No project-relative import is required.  When a Conda environment is active,
the launcher uses it directly.
