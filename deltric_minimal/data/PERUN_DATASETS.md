# Perun-only benchmarks (big or high-dimensional)

These are too large for the laptop (16 GB RAM, ~30 GB free disk) under the rule
**n ≤ 20k and d ≤ 100**. Run them on Perun. The dev/holdout assignment was fixed on
2026-09-24 together with the laptop sets in `data/{testing,holdout}/real_world/`, and
before any DelTriC run on them. As with the laptop sets, the holdout rows must not be
fit, plotted or scored during tuning.

Store them in the repo format: `X` (n, d) float64; `y` with -1 = anomaly and
0..k-1 = cluster, ordered by size; `source_y` holding the original label. Each gets a
JSON sidecar with source URL, construction and SHA-256 (see
`studies/make_real_world_datasets.py` for `encode` / `downsample_class` / `write`,
which can be reused as they are).

`anomaly_origin`: **natural**: the source labels anomalies or faults.
**natural_rare_class**: a genuinely rare class kept at its natural frequency.
**constructed**: a class downsampled by us (ADBench style); report these separately.

| # | Dataset | Split | n | d | k | Anomaly | Origin | Download size | Why Perun |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Covertype | dev | 581,012 | 54 | 6 | Cottonwood/Willow (0.5%) | natural_rare_class | 11 MB zip | n |
| 2 | NSL-KDD | dev | 148,517 | 41 (~122 one-hot) | 4 | U2R + R2L attacks | natural | ~20 MB | n, d after one-hot |
| 3 | MIT-BIH heartbeat | dev | 109,446 | 187 | 4 | Fusion (F) beats (0.7%) | natural_rare_class | ~100 MB | n, d |
| 4 | Fashion-MNIST | dev | 70,000 | 784 | 9 | one class at 2% | constructed | ~30 MB | n, d |
| 5 | scRNA real doublets (hm-12k, pbmc-ch, …) | dev | 6k–40k | ~20k genes → 50 PCs | cell types / species | doublets | natural | **752 MB** zip | d, size, R needed |
| 6 | CIC-IDS2017 | holdout | 2.8 M | 78 | benign + families | rare attack families | natural | **~1.1 GB** CSV | n, size |
| 7 | Sensorless Drive Diagnosis | holdout | 58,509 | 48 | 10 | one fault class at 2% | constructed | 11 MB | n |
| 8 | UCI HAR | holdout | 10,299 | 561 | 5 | one activity at 3% | constructed | 60 MB | d |
| 9 | NLP-ADBench N24News (+ AG News) | holdout | ~19k test | 768 (BERT) / 3072 (OpenAI) | news topics | ADBench anomaly topic | natural_rare_class (ADBench definition) | 59 MB BERT; GPT train files **1.6–6 GB** | d, size |
| 10 | Zheng68k PBMC | holdout | 68,579 | 32k genes → 50 PCs | 9 cell types | CD34+, dendritic (<2%) | natural_rare_class | ~130 MB | n, d |

## Per-dataset notes

### 1. Covertype (dev)
- Source: https://archive.ics.uci.edu/static/public/31/covertype.zip (also `sklearn.datasets.fetch_covtype`).
- Clusters: the 6 major cover types. Anomaly: Cottonwood/Willow (class 4, 2,747 rows, 0.47%) at its natural rate.
- Features: 10 continuous and 44 binary (wilderness area and soil type) columns. Try both the 10 continuous only and all 54; the binary columns dominate kNN distances.
- Subsample to 50k–100k stratified for the first runs.

### 2. NSL-KDD (dev)
- Kaggle (verify the slug): `hassan06/nslkdd`. Mirrors of the original UNB files, KDDTrain+.txt and KDDTest+.txt, also exist on GitHub, e.g. `defcom17/NSL_KDD`.
- Map the 39 attack names to the 5 standard categories. Clusters: normal, DoS, Probe. Anomalies: R2L and U2R (natural, rare in train).
- One-hot `protocol_type`, `service` and `flag` (≈122-d), or drop them (38 numeric features). Log-transform the byte counts.
- Contains many exact duplicates: record `duplicate_feature_rows`.

### 3. MIT-BIH heartbeat (dev)
- Kaggle: `shayanfazeli/heartbeat` (`mitbih_train.csv` + `mitbih_test.csv`, 187 samples per beat, already segmented and normalized).
- Clusters: N, S, V, Q. Anomaly: F (fusion beats, 803 rows, 0.7%).

### 4. Fashion-MNIST (dev)
- OpenML id 40996: `sklearn.datasets.fetch_openml(data_id=40996)`.
- Clusters: 9 classes. Anomaly: one class (e.g. "Bag") downsampled to 2% (constructed).
- Run on raw 784-d pixels, and optionally on a 50-PC PCA fitted on the clusters only. Record which.

### 5. scRNA-seq real doublets (dev)
- Zenodo: https://zenodo.org/records/4062232 → `real_datasets.zip` (752 MB, 16 datasets, from Xi & Li 2021, Cell Systems).
- Anomaly: experimentally determined doublets (natural). Best picks:
  - `hm-12k`: human/mouse mixture, so clusters = species and species-mixed doublets are ground truth.
  - `pbmc-ch`: cell-hashing doublets.
- Files are `.rds`. Convert on Perun with R (`readRDS`), or `rpy2`, to counts + labels.
- Preprocessing: log1p(CP10k) → top 2000 HVG → 50 PCs. Document this in the JSON.
- Cluster labels: species for hm-*. For PBMC, use the published cell-type annotation if present; otherwise use k=1 (anomaly-only).

### 6. CIC-IDS2017 (holdout)
- UNB: https://www.unb.ca/cic/datasets/ids-2017.html (the form is only a name/email wall) or Kaggle (verify the slug): `chethuhn/network-intrusion-dataset`, which contains the `MachineLearningCVE` CSVs.
- Clusters: benign plus the large attack families (DoS Hulk, PortScan, DDoS). Anomalies: rare families (Heartbleed, Infiltration, SQL injection, XSS, Bot).
- Clean: drop the `Inf`/`NaN` rows and the constant columns, and strip whitespace from column names. Subsample benign to ~200k.

### 7. Sensorless Drive Diagnosis (holdout)
- https://archive.ics.uci.edu/static/public/325/dataset+for+sensorless+drive+diagnosis.zip
- 11 balanced classes (5,319 each). Clusters: 10 classes. Anomaly: 1 class at 2% (constructed).

### 8. UCI HAR (holdout)
- https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip
- Merge train and test (10,299 × 561). Clusters: 5 activities. Anomaly: one activity (e.g. WALKING_UPSTAIRS) at 3% (constructed).
- n fits the laptop, but d=561 does not.

### 9. NLP-ADBench (holdout)
- Hugging Face: https://huggingface.co/datasets/kendx/NLP-ADBench (no login).
- Use the **precomputed** embeddings only: BERT 768-d, and OpenAI text-embedding-3-large 3072-d.
- The `*.jsonl` files carry `original_label`, so the topics give clusters and the ADBench anomaly topic gives the anomalies.
- Start with N24News test BERT (59 MB) and AG News. Don't download the GPT *train* files (1.6 GB AG News, 6 GB emotion) to the laptop.
- BBC is already local in `data/nlp_adbench/`; don't count it as independent.

### 10. Zheng68k PBMC (holdout)
- Counts: 10x Genomics "Fresh 68k PBMCs (Donor A)" filtered matrices, https://www.10xgenomics.com/datasets.
- Labels: `68k_pbmc_barcodes_annotation.tsv` in https://github.com/10XGenomics/single-cell-3prime-paper. These are correlation-to-FACS-purified-reference labels, not direct sorting.
- Clusters: the major types (CD8, CD4, NK, B, CD14 monocytes). Anomalies: CD34+ and dendritic cells (natural rare).
- Use the same preprocessing as #5. The labels are known to be noisy between the T-cell subtypes; consider merging those to "T cell".

## Kaggle API token (needed for #2, #3, and optionally #6)

1. Log in at https://www.kaggle.com → click your avatar → **Settings** → **API** → **Create New Token**. This downloads `kaggle.json`.
2. Install it on the laptop and on Perun:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```
   On Perun, copy it with `scp ~/.kaggle/kaggle.json perun:~/.kaggle/` and run `chmod 600` there too. Never commit this file.
3. Download (no global install is needed):
   ```bash
   uv run --with kaggle kaggle datasets download -d shayanfazeli/heartbeat -p data/perun/mitbih --unzip
   uv run --with kaggle kaggle datasets download -d hassan06/nslkdd -p data/perun/nslkdd --unzip
   ```
   Some datasets require clicking "I understand and accept" on their Kaggle page once before the API will serve them (error 403).
4. Alternatively, set `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables instead of the file (handy in SLURM scripts).

## Perun checklist
- Put the raw downloads under `$SCRATCH` (or `$WORK`), not `$HOME`. Only the final `.npz` and `.json` files go to `data/perun/{testing,holdout}/`.
- For kNN-graph runs with n > 100k, subsample stratified first (fixed seed, recorded in the JSON), then scale up.
- Put the high-dim sets (HAR, Fashion-MNIST, heartbeat, NLP embeddings) through DelTriC on the raw features. Only use PCA where noted, and record it.
