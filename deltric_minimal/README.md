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
multi-D seed hard limit      0.8
native-2D seed hard limit    0.1
seed baseline                first significant edge-length mode, median fallback
multi-D growth guards        original 1.5, projected 2.8
native-2D growth guards      original/projected 1.5
initial relation             union original-space 50-NN
minimum seed size            10 edges
restoration                  real internal edges up to limit 2.0
Gomory pruning               point cuts; 7 edges (multi-D), 8 (native 2-D)
Gomory shape gate            original-space score >= 12
minimum GH side size         10 points
```

The final stage restores qualifying original Delaunay edges within completed
components, then removes eligible point-based Gomory cuts.  The shape gate is
computed from the original-space triangle area and boundary length, so long,
thin components are protected from that final pruning phase.

All exposed controls are environment variables.  For example:

```bash
HARD_GROWTH_LIMIT_MULTID=1.6 ./run_plot_stages.sh
```

The bundled data intentionally excludes the 1000-dimensional dataset; the
launcher therefore expects 12 curated `.npz` inputs.
