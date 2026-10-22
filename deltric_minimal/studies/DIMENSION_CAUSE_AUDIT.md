# Why DelTriC struggles: a causal audit

2026-09-24. This audit evaluates the **current `run_plot_stages.sh` pipeline**, not the older `cluster_tri` experiment. Production clustering code was not modified. There are 47 saved dataset/intervention runs, exact growth traces, graph-cut measurements, and a synthetic counterexample. Only development and existing curated inputs were evaluated; paper holdouts were not used.

**Anomaly-metric correction:** the original conclusions below concern clustering. Anomaly detection is a coequal objective, and its F1 does not necessarily track ARI. The added analysis scores all 47 saved configurations at all four stages (188 rows). Disabling restoration improves HDBSCAN-demo ARI but lowers anomaly F1; final cost-gate reassignment lowers anomaly F1 on all three noisy development datasets. See the anomaly analysis below before interpreting the proposed changes as overall improvements.

**The strongest demonstrated cause is an interaction between permissive component growth and edge restoration before bridge pruning. Growth joins clusters through noise. Restoration then adds supporting connections to those mistaken joins, raising their cut capacity above the pruning threshold.** Seed contraction locks in additional errors. The 2D path has fewer effective checks, but the evidence does **not** establish that 2D is inherently or uniformly worse than higher dimensions.

## What was actually compared

The pipeline standardizes features, obtains Delaunay edges, selects seeds using union 50-NN and the median edge-length thresholds, grows with limits 1.05 original / 1.35 projected, restores within-component edges, performs contracted candidate Gomory–Hu cuts of size at most 6 with sides of at least 10 points, then reassigns outliers with cost-gate parameters `p=.5, reach=10, q=.5, k=15`.

The runner calls the repository's seed builder, sequential growth, contracted GH, prediction, and cost-gate functions. It substitutes only the specific intervention being tested. Recall10 is passed as ones because it is recorded for display but does not participate in the growth acceptance condition. For all three noisy development datasets, baseline predictions reproduce the existing saved partitions exactly (prediction-vs-prediction ARI = 1.0). A separate invocation of the actual plotting CLI verifies HDBSCAN-demo baseline scoring. Runtime/package versions and input/code SHA-256 values are recorded in `results/dimension_cause_audit/provenance_*.json`.

Scores below are **ARI on reference non-noise points**. Predictions of `-1` remain in the metric; points are excluded only using reference noise labels. `summary.csv` also contains all-point ARI. Oracle interventions deliberately use labels and are diagnostics, not deployable algorithms.

The claimed dimension ordering is not reproduced on the 12 curated inputs:

| Current pipeline | Datasets | Mean ARI |
| --- | ---: | ---: |
| 2D curated inputs | 5 | 0.7045 |
| Higher-dimensional curated inputs | 7 | 0.4494 |

These are different dataset families, not matched experiments, so neither mean estimates a causal effect of dimension. The 2D harder-shapes and moons scores are 0.9817 and 0.9833; the 2D 20-blob score is 0.2290. The higher-dimensional scores range from 0.1970 on close S-curves to 0.8050 on digits. Earlier files under `results/prune_stages_component_growth/` use different growth parameters/backend behavior and cannot be pooled with current-launcher results. The README's growth defaults also differ from the shell launcher; this audit follows the launcher.

## Proof 1: growth merges distinct clusters through noise

| Dataset | Seed ARI | After growth | After GH | Final |
| --- | ---: | ---: | ---: | ---: |
| HDBSCAN demonstration | 0.6742 | 0.2185 | 0.2185 | 0.2203 |
| Chameleon t5 | 0.1523 | 0.0409 | 0.0689 | 0.0688 |
| Chameleon t7 | 0.1244 | 0.9246 | 0.9246 | 0.9305 |

HDBSCAN-demo starts with **zero mixed-reference seed components**. Both it and t5 have **zero direct edges between different non-noise reference classes** in the accepted seed and growth edge sets. Nevertheless their grown components contain multiple reference classes. The connecting paths necessarily pass through reference noise. This is not an inference from a picture: the saved graph explicitly contains those paths.

An exact replay of `_sequential_growth` records the following HDBSCAN-demo merge (point IDs and rounds are zero-based):

- Round 5, edge 1765, endpoints **1389 and 1521**, both reference noise.
- It joins a **319-point component containing reference class 2** to a **313-point component containing reference class 3**.
- The original and projected normalized ratios are both **0.7192811562**. The edge passes 1.05 and 1.35, so the code accepts it.
- Subsequent merges attach class 0 at round 14 and class 1 at round 20.

One explicit path in the pre-restoration graph is `47 → 155 → 21 → 239 → 1922 → 2134`, with reference labels `0 → -1 → -1 → -1 → -1 → 3`. All five edges were added during growth. Their standardized lengths are 0.07422, 0.08765, 0.09687, 0.09467, and 0.06385.

The mechanism is in `plot_seed_component_recomputed_growth.py:1183–1231`: compute median and upper-quartile spread over **all incident Delaunay edges**, accept any boundary edge satisfying

`r = (length - median) / max(Q75 - median, 1e-12) < 1.05`

and the projected analogue, then merge its endpoint components. These incident statistics include edges that were not accepted. Either component can qualify the connection; agreement from both sides is not required. There is no accumulated path-density-valley test before a merge. The later density-aware cost gate operates after component identities have already been fixed.

Evidence: `hdbscan_growth_trace.json`, `chameleon_t5_8k_growth_trace.json`, and `graph_diagnostics.json`. The traces assert exact equality with the original accepted-growth mask; acceptance ratios are captured at the actual accepted round, not approximated using the plot's first-seen diagnostic.

## Proof 2: restoration prevents the intended repair

Keep data, seed graph, growth, GH settings and reassignment fixed; change only whether intra-component edges are restored before GH:

| Dataset | Current final ARI | No restoration | Difference |
| --- | ---: | ---: | ---: |
| HDBSCAN demonstration | 0.2203 | **0.8224** | +0.6020 |
| Chameleon t5 | 0.0688 | **0.2476** | +0.1788 |
| Varied-density 5 clusters | 0.5411 | 0.5323 | -0.0088 |

This is a causal ablation: seeds and the pre-restoration grown graph are unchanged. The production restoration step at `plot_seed_component_recomputed_growth.py:1488` adds **every original Delaunay edge inside a grown component**, including edges that failed earlier selection. Some of those added edges then count as extra capacity in GH. Capacity is edge count, not density or geometric confidence.

Measure minimum cuts between the largest seed representatives of reference classes in HDBSCAN-demo's **actual contracted candidate graph**:

| Reference classes | Before restoration | After restoration |
| --- | ---: | ---: |
| 0 and 1 | 1 | 14 |
| 0 and 2 | 1 | 11 |
| 0 and 3 | 1 | 14 |
| 1 and 3 | 1 | 17 |
| 2 and 3 | 2 | 11 |

The cutoff is **6**. After restoration these pairs have no cut of eligible capacity. In the largest component, candidate edges increase from **484 to 1,112**, while eligible GH tree cuts drop from **57 to 0**. That directly connects the code to the failed repair, rather than merely showing a score improvement.

There is a second, distinct consistency defect: some restored cross-seed edges are **not in the removable candidate graph**. A successful candidate-graph cut therefore need not disconnect the graph ultimately retained. `restoration_counterexample.json` demonstrates this with two 12-point planar seed strips: the default routine removes the one growth bridge but leaves a restored bridge, retaining one component. Including both bridges in the cut graph produces two components. However, making all cross-seed edges removable did **not** improve the tested real datasets (HDBSCAN-demo 0.2161, t5 0.0676). It does not solve the increased-capacity problem, and is not the main demonstrated explanation for these failures.

Disabling restoration alone is also not a finished fix: it produces **23 clusters instead of 6** on HDBSCAN-demo and **90 instead of 6** on t5. It exchanges much of the merging error for fragmentation. See `restoration_ablation.png` and `cut_inflation.json`.

## Proof 3: the dimension branch changes the algorithm, but UMAP is not a universal remedy

`utils_pruning.py:92` applies UMAP only when `X.shape[1] > project_dim`; line 114 returns the input coordinates for 2D. Consequently, in native 2D, original and projected lengths are identical. Their medians and spreads are identical too. Algebraically, growth's two tests become

`r < 1.05 AND r < 1.35`, which is exactly `r < 1.05`.

The projected guard contributes **zero additional vetoes on every native-2D baseline**. Changing its limit above 1.05 cannot affect these decisions. The seed's two median gates similarly collapse to one. Moreover, union 50-NN retains **96.79%–99.69% of all Delaunay edges** across the eight 2D baselines; it offers little additional filtering on those graphs. Some higher-dimensional datasets also have weak kNN filtering, so this is not exclusive to 2D.

For a matched intervention, append an all-zero coordinate to standardized 2D data. Every Euclidean distance is preserved exactly, with no extra information. Keeping native topology produces **identical edges, lengths, seeds, grown graphs, retained graphs and final labels** on all seven tested inputs. Allowing the extra coordinate to trigger UMAP gives:

| Same input information | Native 2D ARI | Zero coordinate + UMAP, seed 42 |
| --- | ---: | ---: |
| Easy compound shapes | 0.7872 | 0.4364 |
| Harder compound shapes | 0.9817 | 0.3567 |
| Moons | 0.9833 | 0.2526 |
| Varied-density 5 clusters | 0.5411 | 0.6360 |
| 20 blobs | 0.2290 | 0.3760 |
| Chameleon t5 | 0.0688 | 0.2399 |
| HDBSCAN demonstration | 0.2203 | 0.5060 |

The HDBSCAN-demo improvement persists with UMAP seeds 43 and 44 (0.6413 and 0.6225). Moons degrades with both (0.2013 and 0.2210); UMAP splits its two true clusters into 13–18 final clusters. This intervention changes topology and projected gates together, so it does not isolate which UMAP effect contributes how much. It **does** prove that the apparent dimension effect can be a pipeline-branch effect without any additional feature information. The explicit seed-42 UMAP intervention was also verified array-for-array against the unmodified production dimension branch.

Evidence: `padding_invariance.json`, `production_padding_check.json`, and `dimension_branch.png`.

## Proof 4: seed construction is global and seed errors cannot be repaired

With seed limit zero, the original-space seed threshold is one global median over Delaunay edge lengths. On `varied_2d_5c`, the median within-class edge lengths range from **0.01678 to 0.09991**. The densest class retains **95.5%** of its within-class edges in seeds; the sparsest retains **14.4%**. This measures a strong imbalance from a shared absolute threshold. It does not by itself prove every subsequent error is caused by that imbalance.

The contracted GH routine treats each initial connected seed as indivisible. Growth only unions components, and the final cost gate changes only points currently labeled `-1`. Thus two differently labeled true points connected inside one seed cannot later be separated by the default pipeline. Count those already-forced wrong same-cluster pairs:

| Dataset | Pairs already joined inside mixed seeds |
| --- | ---: |
| Easy compound shapes | 15,969 |
| Varied-density 5 clusters | 57,181 |
| 20 blobs | 446,867 |
| Chameleon t5 | 15,510 |
| HDBSCAN demonstration | 0 |

For seed class counts `n_c`, the count is `((sum n_c)^2 - sum(n_c^2))/2`, summing over seeds and excluding reference noise. These are structural constraints, not fitted explanations.

Yet seed cleanup alone is insufficient: an oracle removing reference-crossing and noise-incident seed edges yields final ARI 0.0687 on t5 and 0.2203 on HDBSCAN-demo. Later growth recreates the problem. Conversely, an oracle removing reference-crossing and noise-incident edges **after GH**, followed by the normal cost gate, reaches 0.8406 and 0.9983. The retained geometry can support substantially better partitions; the present edge decisions do not recover them. Oracle scores are not estimates of attainable unsupervised performance.

## What should change next

1. **Separate partition decisions from edge restoration.** Perform bridge decisions on the evidence graph before restoration, then restore edges only inside the resulting fixed partitions. The measured cut inflation makes this a well-supported clustering target. The observed fragmentation and anomaly-F1 regression mean removing restoration without revisiting cut selection and anomaly decisions is incomplete.
2. **Evaluate component-to-component joins explicitly.** A short edge, or a chain of short noise edges, is insufficient evidence of a cluster merge. Test a local density-valley or component-boundary criterion before joining active components; these are proposed experiments, not validated fixes.
3. **Allow evidence-based splitting of impure seeds, or make seeding locally scaled.** Current contraction forbids recovery, and the measured density imbalance explains why one global median is an unreliable starting point.
4. **Make topology choice explicit and evaluate matched inputs.** Do not use “dimension > 2” as an implicit switch in an alleged dimension comparison. Do not simply force UMAP everywhere: the controlled moons result disproves that remedy under these settings.

The final cost gate cannot undo component merges. Tuning its reach/density parameters alone therefore cannot repair the principal demonstrated failure. Growth limits, GH thresholds, topology choice, and seed criteria should be assessed with both merge and split errors and with a frozen, consistent pipeline. No held-out performance claim or universal remedy is established by this development audit.

## Anomaly detection: F1 changes the diagnosis and recommendation

For these saved datasets, **reference anomaly = `y < 0`; predicted anomaly = prediction `< 0`**. Anomalies are the positive class. Precision is `TP/(TP+FP)`, recall is `TP/(TP+FN)`, and F1 is `2TP/(2TP+FP+FN)`. Counts are independently cross-checked against scikit-learn. These label conventions must not be transferred blindly to external datasets with binary `1 = anomaly` labels.

All three noisy baselines reproduce the existing development noise scores:

| Dataset | Precision | Recall | Final anomaly F1 | Clustering ARI, non-noise |
| --- | ---: | ---: | ---: | ---: |
| Chameleon t5 | 0.9983 | 0.5013 | 0.6674 | 0.0688 |
| Chameleon t7 | 1.0000 | 0.5994 | 0.7495 | 0.9305 |
| HDBSCAN demonstration | 0.9825 | 0.6608 | 0.7902 | 0.2203 |

The anomaly detector has high precision but misses 34–50% of reference anomalies. Good anomaly F1 and good clustering ARI are distinct outcomes: merging two true clusters can severely hurt ARI without changing either point's anomaly status.

The interventions make the distinction measurable:

| Dataset | Intervention | Clustering ARI | Anomaly F1 |
| --- | --- | ---: | ---: |
| HDBSCAN demonstration | Current pipeline | 0.2203 | 0.7902 |
| HDBSCAN demonstration | No restoration | **0.8224** | **0.7566** |
| HDBSCAN demonstration | Zero coordinate + UMAP, seed 42 | 0.5060 | 0.6981 |
| Chameleon t5 | Current pipeline | 0.0688 | 0.6674 |
| Chameleon t5 | No restoration | 0.2476 | 0.6825 |
| Chameleon t5 | Zero coordinate + UMAP, seed 42 | 0.2399 | 0.4283 |

On HDBSCAN-demo, no restoration keeps final false positives at 6 but increases missed anomalies from 173 to 196. It therefore **does not dominate the original pipeline on both objectives**. Forcing UMAP improves clustering on both shown datasets but reduces anomaly F1 on both.

The final reassignment step is a separately demonstrated source of anomaly errors:

| Dataset | F1 before reassignment | F1 after reassignment | True anomalies absorbed | True normal points recovered |
| --- | ---: | ---: | ---: | ---: |
| Chameleon t5 | 0.7688 | 0.6674 | 175 | 45 |
| Chameleon t7 | 0.9078 | 0.7495 | 247 | 39 |
| HDBSCAN demonstration | 0.8543 | 0.7902 | 56 | 11 |

This is a fixed-run stage ablation, not a refit or parameter search: the pre-reassignment labels are saved as `before`, and the final labels as `final`. Existing non-noise assignments are verified unchanged. On t7, for example, reassignment reduces false positives from 39 to 0 but converts **247 correctly detected anomalies into false negatives**; recall falls from 0.8661 to 0.5994. Across the three datasets, equal-weight mean F1 falls from approximately **0.844 to 0.736**. This directly supports revisiting cost-gate assignment when anomaly F1 matters, even though cost-gate tuning cannot repair merged clusters.

The gate is a distance-budget rule: it assigns currently unassigned points when their accumulated density-weighted path cost is within `reach=10`, using edge-length and density references from completed components. It has no calibrated probability that a point is an inlier. The observed changes prove that this setting admits too many reference anomalies for F1 on these development datasets. They do not isolate the individual causal contribution of reach, density power, or contaminated component reference statistics; those require further controlled parameter ablations.

There is also a scope limitation in the dimensionality comparison: **the 12 curated baselines contain no reference anomalies**, so they cannot establish an anomaly-recall/F1 advantage for either dimension group. Their anomaly F1 is recorded as unavailable, not as zero or one. False-positive rate remains meaningful: the current 20D variable-density baseline incorrectly rejects 27.82% of normal points; the 2D varied-density baseline rejects 13.44%. A dimension/F1 claim needs matched anomaly-containing inputs and the same anomaly-label definition.

The revised development priority is to report **anomaly precision/recall/F1 alongside clustering ARI**, keep the pre-reassignment detector as an explicit comparator, and choose among changes using both outcomes. Neither “disable restoration” nor “force UMAP” is justified as a general improvement. Cost-gate reassignment deserves its own diagnosis because it materially lowers anomaly F1 even on t7, where clustering ARI is already high. No parameter selection or holdout evaluation was performed for this addition.

Run `deltric_minimal/.venv/bin/python deltric_minimal/studies/dimension_anomaly_metrics.py` to reproduce this addition directly from saved arrays. Outputs: `anomaly_metrics.csv`, `anomaly_metrics.json`, and `anomaly_reassignment.csv` under the audit results directory. Future runs of `dimension_cause_audit.py` also include anomaly metrics in each stage's score dictionary. Existing historical run JSON files are preserved; the added CSV/JSON contains the retrospective scores.

## Reproduction and artifacts

From the repository root, using its existing `.venv` (including Torch, which this repository currently imports while defining optional neural-network classes):

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MPLBACKEND=Agg
PY=deltric_minimal/.venv/bin/python
$PY deltric_minimal/studies/dimension_cause_audit.py
$PY deltric_minimal/studies/dimension_cause_audit.py --datasets chameleon_t5_8k chameleon_t7_10k hdbscan
$PY deltric_minimal/studies/dimension_cause_audit.py --datasets 01_shapes_2d_compound_easy_3600_k6 02_shapes_2d_compound_harder_4800_k6 11_v2_moons_2d_2500 varied_2d_5c blobs_large_2d_20c chameleon_t5_8k hdbscan --variants zero_pad_none zero_pad_umap
$PY deltric_minimal/studies/dimension_cause_audit.py --datasets chameleon_t5_8k hdbscan varied_2d_5c --variants oracle_seed oracle_final_cut no_restore all_cross_seed_removable
$PY deltric_minimal/studies/dimension_cause_audit.py --datasets hdbscan 11_v2_moons_2d_2500 chameleon_t5_8k --variants zero_pad_umap --umap-seeds 43 44
$PY deltric_minimal/studies/dimension_cause_evidence.py
$PY deltric_minimal/studies/plot_dimension_cause_audit.py
```

All results are under `deltric_minimal/studies/results/dimension_cause_audit/`. Each run has a JSON summary and NPZ arrays, including point labels, graph edges, masks, lengths and predictions. `summary.csv` consolidates all 47 configurations. `graph_diagnostics.json` contains per-class seed statistics and explicit noise-bridge witnesses. The two growth-trace files record actual acceptance decisions. The graphs and numbers are reproducible without changing clustering source files. Generated results follow the repository's existing git-ignore policy; the three audit scripts and this report are the durable source artifacts.
