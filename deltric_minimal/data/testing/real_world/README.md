# Real-world clustering + anomaly benchmarks: testing split

Fixed on 2026-09-24 before running DelTriC. Whole datasets were assigned to the two splits, balanced by modality and anomaly origin. The split is purposive, not random.

Testing datasets: Dry Bean, HCV, Anuran Frogs, SMD machine-1-1, Gaia Vela-Puppis patch.

Holdout datasets: Avila, Yeast, HTRU2, SDSS colours, Mice Protein, Gaia Orion patch. They are stored under `data/holdout/real_world/` and stay unfit, unplotted, unscored and unavailable to parameter selection. The big and high-dimensional sets that run only on Perun are listed in `data/PERUN_DATASETS.md`.

Each NPZ contains:
- published `X`, unscaled (the evaluator standardizes each dataset);
- `y`: -1 = anomaly, 0..k-1 = cluster, ordered by size;
- `source_y`: the original class label as a string.

Unlike `additional_anomaly/`, these sets carry **cluster ground truth as well as anomaly ground truth**, so ARI on the inliers is meaningful. The exception is HCV and SMD, which have k=1 (anomaly-only).

`anomaly_origin` in each JSON:
- `natural`: the source labels anomalies (disease, server incidents, field stars).
- `natural_rare_class`: a rare class kept at its natural frequency.
- `constructed`: a class downsampled by us, ADBench style. Report these separately.

| dataset | n | d | k | anomalies | origin |
|---|---|---|---|---|---|
| dry_bean | 13611 | 16 | 6 | BOMBAY, 3.8% | natural_rare_class |
| hcv | 589 | 12 | 1 | hepatitis/fibrosis/cirrhosis, 10.7% | natural |
| anuran_frogs | 7195 | 22 | 7 | 3 rare genera, 4.6% | natural_rare_class |
| smd_machine_1_1 | 14240 | 38 | 1 | labelled incidents, 9.5% | natural |
| gaia_vela_puppis | 20000 | 5 | 5 | Gaia field stars, 95.2% | natural (high contamination) |

Notes:
- SMD is a time series treated as i.i.d. rows (every second row of the test set). Anomalies come in contiguous incidents.
- The Gaia patch uses 5D astrometry only and keeps the field at its natural distribution. It replaces the older single-cluster `gaia_*` sets, whose subsampled field formed a compact blob denser than the cluster. Its contamination is high by nature, so read its metrics as "clusters in a real background": anomaly F1 is near-trivial at 95% field, so report member recovery (precision/recall of the non-field class) and ARI on members instead.

Everything is rebuilt by `uv run --with xlrd studies/make_real_world_datasets.py` (or `--only <name>`). Raw downloads are cached in `source/`. Each JSON records the source URL, construction and citation, plus the SHA-256 of both the NPZ and the source.
