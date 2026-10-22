# Additional anomaly benchmarks: testing split

Fixed on 2026-09-17 before running DelTriC. Whole datasets are assigned by documented suitability, not by measured performance. This is a purposive split, not a random sample of all anomaly datasets.

Testing datasets: Musk, Glass, Waveform, Arrhythmia, Ionosphere.

Holdout datasets: Stamps, PageBlocks, Vowels, Letter, ALOI. They are stored under `data/holdout/additional_anomaly/` and remain unfit, unplotted, unscored and unavailable to parameter selection. The stronger suitability group is reserved for later evaluation; its selection does not establish independent or unbiased generalization across all domains.

Each local NPZ contains published `X`, local `y` (-1 anomaly, 0 inlier), and preserved `source_y` (1 anomaly, 0 inlier). Published feature values, duplicates and constant columns are retained. The untouched source downloads are in `source/`. Dataset JSON files record source URLs, documentation, exact counts and SHA-256 hashes; `split.json` records both groups. ADBench downloads are pinned to commit `3dac8221081e190f157d78e93bfa8867f90d0965`. Glass uses the ODDS nine-feature version.

Evaluation uses the existing `plot_seed_component_recomputed_growth.py` and the current `run_plot_stages.sh` defaults, copied from the existing Clustbench runner. It standardizes each evaluated dataset independently and is transductive: all points in a testing dataset are available without their labels. UMAP uses fixed random state 42. There is no parameter search. This is the current launcher pipeline, which differs from the older `cluster_tri` production experiments.

Results: `studies/results/additional_anomaly_2026-09-17/`. Exact commands, package versions and existing code hashes are in `provenance.json`; saved predictions permit independent metric checks. Positive class for anomaly precision/recall/F1 is local `y < 0`; predicted anomalies are `final_prediction < 0`. Ionosphere has 35.9% anomalies and is reported separately as a high-contamination boundary case. Binary source labels do not establish correct inlier clustering, so clustering ARI is not included in the testing metric table. Runtime includes diagnostic plotting and the stage program's HDBSCAN comparison, rather than algorithm-only runtime.

Cite the benchmark collection and original sources linked in each dataset JSON. Benchmark variants from the same source family should not be counted as independent datasets.
