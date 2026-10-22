# Anomaly GIFs against ground truth

`animate_anomaly_truth.py` animates the current production cumulative-cost anomaly reclamation in three synchronized panels: fixed ground truth, colored DelTriC clusters, and predictions against truth. It uses the existing anomaly benchmark preset and standardized original features. All points are displayed at their 2D UMAP coordinates.

```bash
python deltric_minimal/studies/animate_anomaly_truth.py --data deltric_minimal/data/odds_thyroid.npz deltric_minimal/data/odds_cardio.npz deltric_minimal/data/odds_pendigits.npz --out-dir deltric_minimal/studies/results/anomaly_truth
```

The default label key is `y`, with negative labels meaning anomalies and nonnegative labels meaning normal. For 0=normal, 1=anomaly datasets, pass `--truth-format binary`. To read the preserved ODDS labels, pass `--label-key y_odds --truth-format binary`. This choice is explicit because 0/1 labels can also be cluster IDs.

| Marker | Meaning |
|---|---|
| Red cross (FP) | Normal point incorrectly flagged as anomalous |
| Orange triangle (FN) | Ground-truth anomaly accepted as normal |
| Blue star (TP) | Ground-truth anomaly correctly flagged |
| Gray dot (TN) | Normal point correctly accepted |

Each frame shows confusion counts, precision, recall, and F1. A false positive can become a true negative when reclaimed; a true positive can become a false negative. Points already assigned before reclamation remain fixed, so existing false negatives are visible from the first frame.

The middle panel uses a stable color per predicted cluster. Gray crosses are unassigned points, and black rings highlight points reclaimed during the animation. Colors reflect predicted cluster membership, not ground-truth classes. Use `--cluster-ids` to add numeric cluster IDs at their initial median positions.

The utility observes accepted claims in the actual production function using a targeted Python trace; it does not copy or change the detection algorithm. Frames represent accepted-claim progress, not the older animation's competitive rounds. Initial and final states are always included. `--max-frames` bounds distinct snapshots; `--fps`, `--hold-start`, `--hold-end`, and `--dpi` control presentation. A zero-claim run holds the single unchanged state.

Defaults match the prior benchmark: penalty power 3, reach 25, density power 4, density k 15, no density clipping. Override with `--penalty-power`, `--reach`, `--density-power`, `--density-k`, or `--density-clip`. Ground-truth labels are used only for display and scoring, never for graph construction or reclamation.

Each GIF has a JSON sidecar containing initial/final scores, configuration, and snapshot claim counts. No holdout dataset is automatically loaded; only explicitly supplied inputs are processed.

Use the project's Python environment with its numeric, plotting, UMAP, and PyTorch dependencies. The existing production module currently needs PyTorch at import time even for this geometric workflow.
