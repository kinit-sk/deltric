# Clustbench `other`: development and paper holdout

This experiment uses suite **v1.1.0**, pinned to source commit
`ee6d7a28eb0ff922d29b618b595795a45f1b5cb8`.
[Suite documentation](https://clustering-benchmarks.gagolewski.com/weave/suite-v1.html#sec-battery-other).

The split was fixed before fitting or scoring. Whole datasets are reserved,
because splitting points from one clustering problem would expose its geometry
during development. No points are removed or synthetic anomalies added.

| Dataset | Role | Points | Labeled noise |
| --- | --- | ---: | ---: |
| chameleon_t5_8k | Development | 8000 | 1187 |
| chameleon_t7_10k | Development | 10000 | 926 |
| hdbscan | Development | 2309 | 510 |
| chameleon_t4_8k | Paper holdout, unevaluated | 8000 | 761 |
| chameleon_t8_8k | Paper holdout, unevaluated | 8000 | 346 |
| iris | Development false-alarm control | 150 | 0 |
| iris5 | Development false-alarm control | 105 | 0 |
| square | Development false-alarm control | 1000 | 0 |

Two of the four Chameleon datasets were randomly selected for holdout using
Python `random.Random(20260917).shuffle` on the explicitly ordered list in
`clustbench_other.py`. HDBSCAN's demonstration dataset supplies an additional
development source. Chameleon datasets are related: this is a dataset-level
holdout within one family, not an independent source-family generalization test.
The two held-out datasets are too few to establish broad anomaly-detection
performance alone.

## Data and label conventions

Data are separated physically:

- `deltric_minimal/data/other/`: six working NPZ files used by the runner.
- `deltric_minimal/data/holdout/other/`: two NPZ files reserved for later paper benchmarking.
- Each folder has `source/` with untouched compressed files and citation descriptions.
- `data/other/split.json`: membership, dimensions, noise counts and SHA-256 checksums; paths are relative to this folder.

Each NPZ stores raw `X`, original `source_y`, and local `y = source_y - 1`.
The source uses `0` for noise; local DelTriC uses `-1`. Positive source cluster
IDs become local IDs starting at zero. Labels are never passed into the graph,
growth, bridge pruning or cost-gate assignment stages.
[Source format](https://clustering-benchmarks.gagolewski.com/weave/file-format.html).

Only source metadata and integrity checks are examined for held-out datasets.
They are not standardized, fitted, plotted, scored or used to choose parameters.
The runner has no option to evaluate the holdout accidentally. After freezing
the algorithm and its parameters for the paper, create a separate final evaluation
that explicitly consumes the frozen holdout.

## Run

From the repository root:

```bash
deltric_minimal/.venv/bin/python deltric_minimal/studies/clustbench_other.py
```

`--prepare-only` downloads/converts and records the split without running the
algorithm. Existing manifests are reused and every recorded checksum is checked;
a rerun never redraws the split. Custom directories use `--data`, `--holdout-data` and `--out`.

The runner invokes `plot_seed_component_recomputed_growth.py` with the explicit
current defaults from `run_plot_stages.sh`: standardize each dataset independently,
seed limit 0, original growth limit 1.05, projected guard 1.35, union 50-NN,
minimum 10 seed edges, restored intra-component edges, contracted growth-edge
Gomory–Hu cuts of size at most 6 with minimum sides of 10 points, followed by
density cost-gate reassignment (`p=0.5`, `reach=10`, `q=0.5`, `k=15`, no clipping).
This is the current launcher pipeline; the older study's `cluster_tri` configuration
is a different experiment. No parameter search is performed here.

The five noisy datasets and square are two-dimensional, so the existing pipeline
uses their standardized coordinates directly for Delaunay topology. Iris and
iris5 use UMAP for topology with the existing fixed seed 42. Package versions and
code hashes are recorded in `provenance.json`; these may differ from the older
`environment.yml` pins.

## Outputs and interpretation

Outputs live under `deltric_minimal/studies/results/clustbench_other/`:

- `development_metrics.csv`: noise precision/recall/F1, confusion counts,
  false-positive rate, ARI including all points, and ARI restricted to true
  non-noise points. Noise is the positive class; predictions below zero mean noise.
- `summary.json`: equal-weight mean across the three noisy development datasets.
- `provenance.json`: exact arguments, versions, source-code hashes and split hash.
- Per-dataset PNG, prediction/graph NPZ, stage JSON and execution log.

F1/precision/recall are blank for no-noise controls, whose false-positive rate is
the relevant anomaly metric. They are excluded from the macro noise scores.
The existing diagnostic plots also run HDBSCAN with their current defaults;
the CSV scores DelTriC's `final_prediction` only. Runtime includes diagnostic
plotting and that HDBSCAN fit, so it is not an algorithm-only timing benchmark.

These are exploratory development scores. The suite's noise labels include
outliers and irrelevant points between clusters; they are a useful operational
anomaly target, not necessarily the same target as semantic or domain anomalies.
[Noise-label methodology](https://clustering-benchmarks.gagolewski.com/weave/noise-points.html).
For the paper, cite the suite and version as requested in its documentation and
the original sources described in each preserved `.txt` file. In particular,
some Chameleon partitions were created by the suite's curators, and the source
notes that t4/t5 are not mentioned in the commonly attributed CHAMELEON paper.

## Initial development run, 2026-09-17

| Noisy development dataset | Precision | Recall | F1 | ARI, non-noise |
| --- | ---: | ---: | ---: | ---: |
| chameleon_t5_8k | 0.9983 | 0.5013 | 0.6674 | 0.0688 |
| chameleon_t7_10k | 1.0000 | 0.5994 | 0.7495 | 0.9305 |
| hdbscan | 0.9825 | 0.6608 | 0.7902 | 0.2203 |
| Equal-weight mean | 0.9936 | 0.5871 | 0.7357 | 0.4066 |

Noise-free controls: iris 0/150 false alarms, iris5 5/105 (4.76%), square
0/1000. The current algorithm is conservative about declaring anomalies:
precision is high but 34–50% of reference noise is absorbed into clusters.
The t5 diagnostic plot shows several reference clusters merged, consistent
with its low clustering ARI. These findings should guide development without
opening the held-out t4/t8 results.

The completed run produced outputs for exactly the six development datasets.
Source checksum reuse, label conversion, dimensions, noise counts, prediction
shapes and confusion-count totals were checked; anomaly F1 was independently
recomputed from the saved prediction arrays. No holdout outputs were produced.

The folders were reorganized after the initial run. Its original manifest is
preserved as `original_split.json` alongside the original provenance; historical
stage JSON files retain the paths used at execution time. Dataset bytes and
split membership are unchanged. New runs use the updated manifest paths.
