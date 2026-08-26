# Minimal DelTriC component-growth diagnostics

This self-contained directory reproduces the effective default workflow in
the development `run_plot_stages.sh`, without the legacy pruning pipeline or
inactive growth variants.

It uses UMAP only to construct the Delaunay topology.  All primary edge-size
decisions are computed in the original, standardized feature space.

## Setup

```bash
conda env create -f environment.yml
conda activate deltric-minimal
```

## Run

```bash
cd deltric_minimal
./run_plot_stages.sh
```

The default setup is:

```text
seed hard limit              0.0
component growth limit       1.5
projected growth guard       1.0
initial relation             union original-space 50-NN
minimum seed size            10 edges
restore intra-component edges true
Gomory--Hu cut size          6
minimum GH side size         10 points
```

The final stage restores original Delaunay edges whose endpoints belong to
the same completed growth component, then removes eligible Gomory--Hu cuts.
Set `GOMORY_HU_HULL_RATIO_SKIP_THRESHOLD` above zero to protect non-compact
components: GH pruning is skipped when their weighted outer-hull/non-hull
ratio reaches that value.  The default `0.0` disables this protection gate.

All exposed controls are environment variables.  For example:

```bash
GOMORY_HU_HULL_RATIO_SKIP_THRESHOLD=0.075 ./run_plot_stages.sh
```

The bundled data intentionally excludes the 1000-dimensional dataset; the
launcher therefore expects 12 curated `.npz` inputs.
