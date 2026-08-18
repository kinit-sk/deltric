# DelTriC edge pruning: rule extraction from the local-feature corpus

Date: 2026-08-18
Repositories: `deltric` (`prototype`), `kinit` (`master`)
Compute: perun `cpu_short` partition
Companion to: `DELTRIC_DIMRED_EDGE_PRUNING_REPORT.md`

## Executive conclusion

**A four-question decision tree beats the 19-feature logistic model it was
meant to approximate — and deploying it makes clustering worse.**

Read §7 before acting on anything below it. The edge-level result in this
section is real and reproducible, but the ARI measurement in §7 shows it does
not transfer to cluster quality: mean ARI over the 12 curated datasets falls
from 0.457 to 0.414, with no dataset materially improved. The rest of this
section describes what was learned about *ranking bridges*, not a deployable
change.

On shape-grouped holdout over 24 shapes and 1,335,537 Delaunay edges, pooled
across PCA / UMAP / PCA→UMAP:

| Model | AUC | Intra-cluster retention @80% bridge recall |
| --- | ---: | ---: |
| `orig_length` alone | 0.687 | 0.276 |
| `orig_knn_scale_ratio` alone | 0.769 | 0.376 |
| 19-feature logistic (the prior baseline) | 0.771 | 0.611 |
| L1 logistic, 5 non-zero terms | 0.759 | 0.604 |
| **Depth-4 tree** | **0.822** | **0.697** |

This is the unusual case where interpretability costs nothing. The tree is not
a lossy summary of the logistic model; it is a better model, because the real
decision surface has interactions (a long edge is fine *if* the local kNN scale
is also large) that an additive score cannot express.

Three caveats bound that claim, and they matter:

1. The tree wins on ranking and on the safety metric, but its **average
   precision is lower** (0.339 vs 0.429). A depth-4 tree emits only 16 distinct
   scores, so it cannot rank finely within a leaf. It is a good gate, not a good
   probability estimate.
2. **Depth 3 is not enough.** Under a safe labelling rule, no depth-3 leaf is
   majority-bridge, so the depth-3 tree degenerates to "always KEEP". Depth 4 is
   the minimum that isolates bridge-dominated regions.
3. Dimension coverage is lopsided (§6). The synthetic families dominate; the
   curated DelTriC datasets contribute 3–9 runs each.

## 1. What was done

`edge_pruning_study.py` already computed 22 per-edge features and reported a
grouped logistic AUC of ~0.80, but it **discarded the per-edge matrix** —
`_run_one` balanced-sampled 400 positives + 400 negatives per run, fitted CV,
and kept only aggregates. Nothing per-edge was ever written to disk, so no rule
could be fitted against the real class balance.

Two changes:

- `edge_pruning_study.py` gained `--dump-edges DIR` / `--dump-max-edges-per-run`.
  It writes one compressed `.npz` per run plus a `dump_index.csv`. The
  aggregate outputs are byte-identical with and without the flag (verified;
  only the `seconds` timing column differs).
- New `edge_rule_extraction.py` fits both interpretable families — depth-limited
  trees and L1/L2 linear scores — under the same `GroupKFold` protocol as the
  existing baseline, and exports literal rules.

The dump is the **full** edge population, not the balanced sample. This matters:
a rule fitted at a 50% positive rate places its thresholds in the wrong place
for a graph that is 10.8% bridges.

### Cost

The expensive artifact — the 1,524 stored supercomputer embeddings — was read,
never recomputed. All 438 runs of feature extraction took **3.7 minutes** total
(slowest single run 2.3s), because every Delaunay triangulation happens in the
2-D projection regardless of ambient dimension.

Model fitting is the expensive half and went to perun: 180 (subset, model, fold)
tasks per feature variant across 256 cores, ~12 minutes wall clock, against
~3 hours for the serial local attempt.

## 2. The recommended rule

Depth-4 tree, raw features, fitted on all 1.34M edges
(`results/edge_rules/rules_tree_depth4_raw.py`):

```python
if orig_knn_scale_ratio <= 1.285209:
    KEEP                                    # support 1,051,124
elif orig_length <= 0.665956:
    KEEP                                    # support 18,490
elif ambient_projected_density_log_ratio <= 1.715939:
    if orig_knn_scale_ratio <= 2.604881:
        KEEP                                # bridge rate 0.108, support 93,005
    else:
        PRUNE                               # bridge rate 0.614, support 31,058
else:
    if endpoint_density_asymmetry <= 0.274422:
        KEEP                                # bridge rate 0.292, support 79,586
    else:
        PRUNE                               # bridge rate 0.679, support 62,274
```

Feature definitions (all from `edge_pruning_study.py:193` `_edge_feature_arrays`):

- `orig_knn_scale_ratio` = `L_o / sqrt(s_o(u) * s_o(v))`, where `s_o` is the
  ambient kNN radius at the endpoint (k=15).
- `orig_length` = original edge length divided by the run's **median** original
  edge length.
- `ambient_projected_density_log_ratio` = `log(ambient_edge_scale / projected_edge_scale)`.
- `endpoint_density_asymmetry` = `|log(s_o(u) / s_o(v))|`.

At its own leaf labels this rule:

| Quantity | Value |
| --- | ---: |
| Edges pruned | 93,332 (7.0% of the graph) |
| Bridge precision | 0.658 |
| Bridge recall | 0.425 |
| **Intra-cluster edge retention** | **0.973** |

That is the profile DelTriC wants from a gray-band gate: it removes a small,
high-purity slice and leaves 97.3% of the true intra-cluster graph untouched.
It is explicitly *not* a complete bridge detector — it finds 43% of bridges.

### The linear alternative

If a smooth score is preferred over a gate, the 5-term L1 model
(`results/edge_rules/rules_linear_raw.json`, `logistic_l1_cn20`):

```text
S(e) = -0.8044
       + 1.9614 * endpoint_density_asymmetry
       + 0.4631 * orig_incident_mean_ratio
       + 0.0301 * endpoint_context_disagreement
       + 0.0282 * projected_knn_scale_ratio
       + 0.0159 * orig_knn_scale_ratio
PRUNE if S(e) >= -0.110      # 80% bridge recall operating point
```

AUC 0.759 against the 19-feature model's 0.771 — a 0.012 loss for dropping 14
of 19 terms. Note how lopsided the weights are: `endpoint_density_asymmetry`
carries the model almost alone, and the last three terms are near-noise. That
is itself evidence the additive form is a poor fit to this problem.

## 3. Negative result: within-run normalisation made things worse

Section 7 of the companion report proposed converting ratio features to
within-run median/IQR-normalised values so thresholds transfer across datasets.
That was tested directly and **it does not work**:

| Pooled model | Raw AUC | Normalised AUC |
| --- | ---: | ---: |
| `orig_knn_scale_ratio` alone | 0.769 | 0.699 |
| 19-feature logistic | 0.771 | 0.754 |
| Depth-3 tree | 0.796 | 0.701 |
| Depth-4 tree | 0.822 | 0.745 |

Normalisation lost on every model. The likely mechanism: the features are
*already* scale-free by construction — `orig_knn_scale_ratio` divides by the
local ambient kNN radius, `orig_length` divides by the run median. Dividing
again by the within-run IQR removes the between-run differences in spread that
genuinely carry signal. A run whose edge lengths are tightly clustered is a
different regime from one with a long tail, and the raw feature preserves that.

The `--leaf-prune-rate` and transform machinery are kept in the script, but the
raw variant is the one to use.

## 4. Two bugs found and fixed during this work

Both were silent — they produced plausible-looking numbers.

### 4.1 The L1 models were not sparse

scikit-learn 1.8 deprecated `penalty="l1"` in favour of `l1_ratio`, and passing
the old argument **silently fits L2** rather than raising. Separately,
liblinear minimises `C * sum(loss) + ||w||_1`, so a `C` that yields 2 non-zero
coefficients on 2,000 rows is numerically irrelevant on 1.36M. The first run
reported "sparse" models with 19/19 non-zero coefficients at every setting.

Fixed by expressing the grid as `C*n` and running the sparsity assertion at the
actual row count. `assert_l1_is_sparse` now fails the job rather than producing
a dense "sparse" model.

### 4.2 The tree's PRUNE labels would have damaged the graph

`class_weight="balanced"` is correct for *fitting* — at a 10.8% base rate,
unweighted splits ignore bridges — but it is wrong as a *decision rule*. It
flips a leaf to PRUNE at barely above the base rate. The first run's depth-3
tree labelled `PRUNE` a leaf with an **observed bridge rate of 0.110 and
support 221,684**: a region that is 89% genuine intra-cluster edges.

Leaves are now labelled by observed bridge rate against an explicit
`--leaf-prune-rate` (default 0.5, i.e. majority-bridge). Split structure and the
probability ranking behind every reported AUC are unchanged; only the labels are.

This is what collapsed the depth-3 tree to "always KEEP" — an honest result. At
depth 3, no leaf is majority-bridge.

## 5. Porting checklist

The depth-4 rule needs four features. Against
`deltric_minimal/utils_component_growth.py`:

| Feature | Backend status |
| --- | --- |
| `orig_length` | **present** — `orig_edge_sizes` (`:785`); needs dividing by the run median |
| `orig_knn_scale_ratio` | absent — needs ambient kNN radius per point |
| `ambient_projected_density_log_ratio` | absent — needs ambient *and* projected kNN radius |
| `endpoint_density_asymmetry` | absent — needs ambient kNN radius per point |

The three missing features share one dependency: **the per-point kNN radius in
ambient and projected space**. The backend already performs the ambient search
in `_original_knn_directed_relations` (`:70`) and explicitly throws the radii
away:

```python
indices = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean").fit(X).kneighbors(
    X, return_distance=False          # utils_component_growth.py:84
)[:, 1:]
```

Flipping that to `return_distance=True` yields `s_o = distances[:, -1]` at no
extra search cost. Only the projected-space query is genuinely new work.

Two mismatches to resolve when porting:

- **k differs.** The study used `k=15`; the backend defaults to `knn_k=30` and
  `run_plot_stages.sh` sets 50. `orig_knn_scale_ratio` is defined against the
  k-th neighbour radius, so the thresholds above are calibrated for k=15 and
  will shift under a different k. Either fit at the deployment k or compute the
  radius at k=15 separately.
- **`orig_length` is median-normalised** per run in the study
  (`orig_length / median(orig_length)`), whereas the backend's
  `orig_edge_sizes` is a raw length. The division must be added.

Full attribution for every rule is in `results/edge_rules/porting_checklist.csv`.

## 6. Limitations

- **Corpus is lopsided by dimension.** d=2/5/50/100 have 75–117 runs each; d=8/10/20/64
  have only 3–9, and those are exactly the curated `ext_*` DelTriC datasets. The
  grouped holdout is well-powered on synthetic families and thin where it matters
  most.
- **The target is a proxy.** An edge is positive when its endpoints carry
  different labels. A manifold may legitimately connect two labelled regions.
  This inherits the caveat from §2 of the companion report.
- **No backend validation.** Scope was research-only: the rule has not been run
  through `component_growth_graph`, so its effect on final ARI is unmeasured.
  AUC and edge-level retention are not the same thing as cluster quality.
- **Single seed, one tree.** No seed variation, no bagging, no cost-sensitive
  threshold sweep beyond the two recall targets.
- Trees are unstable to resampling by nature. The depth-4 thresholds should be
  treated as one draw, not as calibrated constants, until a seeded rerun confirms
  the split structure is stable.

## 7. ARI: the rule does not improve clustering, and usually harms it

This was step 1 of the old next-steps list — the only test that actually
matters. It has now been run (perun `cpu_short` partition).

The depth-4 rule was wired into `component_growth_graph` as an **additional
veto** on top of the existing gates: an edge the rule prunes is removed, an edge
it keeps still has to pass every existing threshold. It can therefore only
remove edges, never add them. All 12 curated datasets were run under
`run_plot_stages.sh` growth defaults, with the rule applied to the seed phase,
the growth phase, or both.

### Result

| Variant | Mean ARI | Δ vs baseline | Improved | Worsened | Worst Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | **0.4572** | — | — | — | — |
| rule on seed + growth | 0.4135 | **−0.0438** | 0 | 5 | −0.270 |
| rule on seed only | 0.4479 | −0.0093 | 0 | 4 | −0.066 |
| rule on growth only | 0.4507 | −0.0066 | 2 | 3 | −0.077 |

**No variant improves the mean, and the best-performing configuration is the one
that does the least.** Applying the rule everywhere is the worst option. Of 36
dataset/variant comparisons, exactly 2 showed any improvement, both smaller than
+0.002 — noise.

Per dataset, with the rule applied to both phases:

| Dataset | d | Base ARI | Rule ARI | Δ | Pruned | Bridge prec. | Intra ret. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 04_v3_vardensity_blobs_20d | 20 | 0.308 | 0.038 | **−0.270** | 19.2% | 0.828 | 0.959 |
| 02_v3_overlap_blobs_10d | 10 | 0.389 | 0.227 | −0.162 | 13.2% | 0.231 | 0.867 |
| 59_h2mg_64_50 | 64 | 0.107 | 0.039 | −0.068 | 54.5% | 0.231 | 0.490 |
| 61_h2mg_64_70 | 64 | 0.056 | 0.034 | −0.022 | 54.0% | 0.321 | 0.514 |
| 14_v2_blobs_8d | 8 | 0.207 | 0.204 | −0.002 | 4.7% | 0.356 | 0.957 |
| the other 7 | | | | ±0.000 | 0.1–6.7% | 0.29–1.00 | 0.95–1.00 |

### Was this the real algorithm, and is it reproducible?

Yes. The harness calls `cluster_tri`, and its output was verified
**bit-identical** to the path `plot_prunning_stages_2x2_growth.py` uses
(`component_growth_graph` → `_merge_component_outliers`) called with
`run_plot_stages.sh` values spelled out verbatim. `outer_relaxation` is the one
parameter the harness leaves at its default rather than the launcher's 2.0; it
is read only inside `if open_space_relaxation:` branches, which are off, so it
is inert.

**Absolute ARI is not stable across machines.** Re-running the same config
locally gives baseline 0.335 on dataset 04 where perun gave 0.308, and 0.346 vs
0.389 on `02_v3_overlap_blobs_10d`. UMAP is seeded (`random_state=42`) but
float-level differences change the projection enough to move the Delaunay graph.
Consequences:

- The **±0.01 deltas** in the variant summary are within this noise and should
  not be read as real. That includes the two "improvements".
- The **large negative effects are robust**: dataset 04 reproduces at −0.2685
  locally against −0.270 on perun. `02_v3` reproduces in direction but not
  magnitude (−0.060 local vs −0.162 remote).

The conclusion rests on the large effects and on the fact that no dataset
improved, not on the mean difference of −0.044.

### Why it fails — and this is the interesting part

**Edge-level accuracy does not transfer to component quality, because
connectivity is a chain property.**

Dataset 04 is the cleanest demonstration. There the rule performed *well* by
every edge-level metric this study was optimising: 82.8% bridge precision, and
it retained **95.9% of genuine intra-cluster edges** — better than the 97.3%
pooled figure only slightly, and far above any threshold that would look alarming
on a PR-curve. ARI still collapsed from 0.308 to 0.038.

The mechanism is visible in the component counts:

| 04_v3_vardensity_blobs_20d | ARI | k found | Noise | ARI on assigned points |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.308 | 21 | 45.0% | 0.741 |
| rule, both phases | 0.038 | 38 | 66.1% | 0.244 |

True k is 6. The rule did not mislabel points — it **shattered** the graph:
components rose 21 → 38 and noise 45% → 66%. The 4% of intra-cluster edges it
cut were not randomly distributed; they were disproportionately the long,
low-density, high-scale-ratio edges that hold a cluster together across a sparse
region. Those are exactly the edges the tree was trained to distrust, because in
the labelled corpus that geometry usually *is* a bridge. Within a single cluster,
that same geometry is a cut vertex.

A retention metric averaged over edges cannot see this. Cutting 4% of intra
edges uniformly at random would be nearly harmless; cutting the 4% that are
articulation points is fatal. **The safety metric this study used — intra-cluster
retention — is the wrong safety metric.** The right one is something like the
increase in connected components within each ground-truth cluster.

Second failure mode: the two `h2mg` datasets drew **54% prune rates**, against
7.0% in the corpus the tree was fitted on. That is severe distribution shift, and
it is unsurprising the rule is destructive there. `orig_knn_scale_ratio > 1.285`
is not a scale-free condition in the way the feature's construction implies.

### What this means for the earlier conclusion

The §0 result stands as stated — the depth-4 tree really does rank bridges better
than the 19-feature logistic model on held-out shapes. But the framing that the
rule is "a safe gate" was **wrong**, and the AUC comparison in the executive
summary should not be read as evidence that any of these models is ready to
deploy. The prior report's decision to keep this research-only was correct.

Interpretability was never the bottleneck. The bottleneck is that per-edge
bridge classification is the wrong objective for a component-growth algorithm.

### Next steps this suggests

1. **Change the target.** Score edges by their effect on component structure —
   e.g. penalise cutting an edge whose removal disconnects a ground-truth
   cluster — rather than by whether their endpoints differ in label.
2. **Add a connectivity guard.** If any edge-level rule is deployed, make it
   incapable of disconnecting an already-connected component: apply it only to
   edges that are not bridges in the graph-theoretic sense. This is cheap and
   would have prevented most of the damage above.
3. **Investigate the 54% prune rates** on `h2mg`. Either the feature is not
   scale-free across these regimes or those datasets sit outside the corpus.

## 8. Older next steps (edge-level, now lower priority)

1. **Seeded stability check** — refit the depth-4 tree over 5 resamples and
   report threshold variance. If the splits move materially, quote the rule as a
   family rather than as fixed constants.
3. **Sweep `--leaf-prune-rate`** from 0.5 down toward the base rate and plot
   intra-retention against bridge recall. 0.5 is a defensible default, not an
   optimised choice.
4. **Retest depth 5–6.** Depth 4 beat depth 3 by 0.026 AUC and the trend had not
   flattened; the depth cap was set a priori, not tuned.

## 8. Reproducibility

Scripts (in `kinit`):

- `edge_pruning_study.py` — now with `--dump-edges`

Scripts (in `deltric_minimal/studies/`, reorganized after this report was
written — `edge_rule_extraction.py` was originally in `kinit` alongside
`edge_pruning_study.py`, and imports it from there via `_kinit_path.py`):

- `edge_rule_extraction.py`
- (slurm launcher: `deltric_minimal/slurm/run_edge_rules.sh`, 256-core job)

Scripts (in `deltric_minimal`), added for §7:

- `edge_rule.py` — new; the depth-4 rule and its four features. Verified
  bit-identical (max |diff| = 0.0) against `_edge_feature_arrays` on a shared
  input, so any ARI difference is the rule, not a porting slip.
- `eval_edge_rule_ari.py` — new; the 12-dataset × 4-variant ARI harness.
- `slurm/run_edge_rule_ari.sh` — new.
- `utils_component_growth.py` — gained an `edge_rule` / `edge_rule_scope` hook,
  defaulting to `None`. Verified that `edge_rule=None` produces label arrays
  bit-identical to the untouched call path.

Two things found while porting, both pre-existing:

- The study's own `rules_tree_depth4_raw.py` export emits bare feature names
  (`orig_knn_scale_ratio <= ...`) instead of dict lookups, so it raises
  `NameError` if called. `_tree_to_python` in `edge_rule_extraction.py` needs a
  fix; the thresholds in `edge_rule.py` were transcribed from the JSON instead.
- `utils_pruning.py` cannot be imported without `torch`: the `try: import torch`
  guard at `:28` does not cover the `nn.Module` subclass defined at module scope
  at `:1021`. `environment.yml` does not list torch.

```bash
# ARI measurement (perun, ~1 min on 64 cores)
sbatch slurm/run_edge_rule_ari.sh
```

```bash
# 1. Dump the per-edge feature table (in kinit, local, ~4 min)
uv run python edge_pruning_study.py --out results/edge_study \
  --dump-edges results/edge_dump --max-n 5000

# 2. Push to perun (the dump is ~100 MB; ~/kinit must exist there too)
rsync -avz results/edge_dump/ perun:~/kinit/results/edge_dump/

# 3. Fit and export (in deltric_minimal, perun, ~12 min on 256 cores)
sbatch slurm/run_edge_rules.sh

# 4. Fetch
rsync -avz perun:~/deltric_minimal/studies/results/edge_rules/ studies/results/edge_rules/
```

Locally, without a cluster, add `--max-edges-per-run 800 --jobs -1` and run
`studies/edge_rule_extraction.py` from `deltric_minimal` (needs a sibling
`kinit` checkout — see `studies/_kinit_path.py`).

Outputs in `studies/results/edge_rules/`: `rule_model_comparison.csv` (72 rows: 2
variants × 4 subsets × 9 models), `rules_tree_depth{2,3,4}_{raw,normalized}.{txt,json,py}`,
`rules_linear_{raw,normalized}.{txt,json}`, `porting_checklist.csv`,
`manifest.json`.

### Sanity gates passed

- The unchanged 19-feature grouped logistic reproduces the companion report
  exactly: 0.8119 / 0.8153 / 0.8065 / 0.7969 (pca / pca_umap / umap / pooled)
  against the reported 0.812 / 0.815 / 0.806 / 0.797.
- 304 labelled runs, 24 shape groups, 10.8% pooled positive rate, consistent
  with the reported ~9.2% (the difference is the full population versus the
  balanced sample).
- Both perun jobs completed with empty stderr.
