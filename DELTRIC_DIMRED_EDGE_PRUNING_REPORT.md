# DelTriC dimensional reduction and local edge-pruning investigation

Date: 2026-08-14  
Revised: 2026-08-14 — added section 4.1 (t-SNE through the actual
component-growth backend), corrected the section 6 variant table against the
final CSVs, and revised the t-SNE recommendation in section 6 and the final
decision accordingly.  
Repositories: `deltric` (`prototype`) and `kinit` (`master`)  
Benchmark snapshot: external supercomputer run `13.8:14:20` (not part of either repo)

## Executive conclusion

The answer is **partly yes, but not in the strong hyperparameter-free sense**.

An edge can be scored locally from its original-space length, the lengths of
the other Delaunay edges incident to its endpoints, ambient-space kNN scale,
projected-neighbour precision, and a one- or two-hop context. These signals are
substantially better than chance and remain useful after the obvious longest
edges have been removed. However:

1. Original edge length and original local scale are the dominant signals.
   A purely topology-based rule is not enough.
2. Endpoint context is useful but moderate. Across 438 stored embedding runs,
   endpoint context disagreement had macro AUC about `0.66` on semantic
   cross-label bridge edges and about `0.76` on the ambient-kNN non-local proxy
   under UMAP.
3. Two-hop context is not a universal improvement. It is helpful on some
   overlap/digits cases, neutral on variable density, and damaging on the
   close-S-curves case.
4. A grouped local/context classifier transfers across held-out shapes, but
   imperfectly: AUC is about `0.80` on the complete edge-study sample and
   about `0.90` in the smaller, deliberately balanced conditional ablation.
5. UMAP is a good practical projection for DelTriC's 2-D Delaunay front-end.
   It improves local neighbourhood recall and manifold recovery dramatically,
   but it distorts global distances, density, and centroid arrangement. PCA
   remains the useful global-geometry baseline.
6. t-SNE, when finally run through the actual component-growth backend, beat
   UMAP on mean large-component ARI (`0.450` versus `0.419` over the 12 curated
   datasets; `0.301` versus `0.247` over the 7 where the projection actually
   changes the graph). The win is not uniform — see section 4.1. The mechanism
   is that t-SNE's tighter clumps let more Delaunay edges clear the *projected*
   hard gate, so t-SNE is systematically more permissive through this pipeline.
   This is better read as evidence that **the projected hard gate is
   mis-calibrated against projection scale** than as evidence that t-SNE
   clusters better.

The production recommendation is therefore:

> Keep DelTriC's robust original-space global gate and ambient kNN seed logic.
> Add an adaptive local score in a middle, ambiguous band. Use endpoint-local
> incident-length/kNN-scale features first; use endpoint context and two-hop
> context as secondary evidence. Protect trusted mutual-kNN and alternative-path
> edges, and allow the classifier to abstain. Keep `local_knn_selectivity_scope`
> at `edge` by default; use `edge_2hop` only after dataset-specific validation.

This is a calibrated pruning policy, not a claim that a local neighbourhood
contains enough information to recover semantic cluster membership in every
dataset. Any final binary decision still has an operating threshold, even if
all features are normalized adaptively.

## 1. What was investigated

DelTriC's relevant pipeline is:

```text
original points
    -> 2-D projection
    -> Delaunay triangulation in projected space
    -> candidate edges
    -> original/projected edge statistics and kNN relations
    -> hard gate, component growth, optional bridge pruning
    -> connected components / clusters
```

The important design choice is that the topology is projected-space topology,
while the main edge-size decisions are computed from the original points. The
current component-growth backend also exposes three local selectivity scopes:

- `edge`: the candidate edge's two endpoint contexts;
- `edge_2hop`: the projected-Delaunay context out to two graph hops;
- `component`: the current growing component's aggregate selectivity.

The investigation covered four evidence layers:

| Evidence | Scope | Purpose |
| --- | --- | --- |
| Existing supercomputer benchmark | 31 shapes, 1,524 stored embedding runs | Compare PCA, UMAP, and PCA→UMAP on point-neighbourhood and global-geometry metrics |
| New edge-feature study | 438/438 successful runs, 146 full dataset IDs, 22 edge properties | Test local pruning hypotheses directly on Delaunay edges |
| New DR variant study | 32 synthetic datasets at `n=500`, 13 methods; 415 successes and one recorded failure | Compare UMAP settings and other reduction algorithms under the same 2-D Delaunay edge test |
| Direct DelTriC audit | All 12 curated datasets, UMAP/PCA/t-SNE; plus 12 UMAP local-scope runs | Check the actual component-growth backend rather than only a reimplementation |

The edge study uses the stored benchmark embeddings, so it does not silently
rerun or alter the large supercomputer UMAP jobs. The direct audit calls
`deltric_minimal.utils_component_growth.component_growth_graph` with the
settings exposed by `run_plot_stages.sh`.

## 2. Targets and limitations of the edge experiment

There is no universal, unlabeled ground-truth definition of a “bad” Delaunay
edge. Two targets were therefore kept separate throughout:

### Semantic bridge target

For labelled datasets, an edge is positive when its endpoints have different
non-negative class labels. This is a useful bridge proxy, but it is not a
perfect definition of a bad edge: a manifold may legitimately connect two
labelled regions, and a class boundary may contain useful graph edges.

There are 304 label-target rows in the main edge study. The average positive
rate is about 9.2%, so accuracy is not reported as the primary measure.

### Ambient non-local proxy

For datasets without a usable multi-class semantic target, an edge is positive
when neither direction of the edge occurs in the original-space union kNN
relation. This target is explicitly a proxy. The direct ambient kNN disagreement
feature is therefore oracle-like for this target and is not allowed into the
cross-shape model features.

There are 134 proxy-target rows, with an average positive rate of about 29.2%.

All feature scores below are **oriented ROC AUC**, meaning that a feature is
allowed to work in either direction. A score of `0.5` is chance. The reported
means are macro averages over runs; medians are often much higher because many
easy separated datasets are close to perfectly ordered.

## 3. Existing PCA/UMAP benchmark

The supercomputer snapshot contains 508 rows per method for PCA, UMAP, and
PCA→UMAP. The overall results are:

| Method | Recall@5 | Recall@10 | Recall@15 | N-preserve | N-surprise | Distance Spearman | Density corr. | Centroid corr. | Manifold corr. | Fit seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PCA | 0.2385 | 0.2779 | 0.3050 | 0.9360 | 0.1112 | 0.8473 | 0.3909 | 0.9131 | 0.3277 | 0.0429 |
| PCA→UMAP | 0.3312 | 0.3770 | 0.4010 | 0.9315 | 0.0797 | 0.5374 | 0.1442 | 0.3319 | 0.7101 | 46.2471 |
| UMAP | **0.3413** | **0.3852** | **0.4072** | **0.9410** | **0.0819** | 0.5353 | 0.1551 | 0.3695 | **0.7095** | 46.5019 |

UMAP beats PCA on recall@10 for 23 of 31 shapes, with mean per-shape gain
`+0.067`. The largest gains are on `clusters_on_swiss_roll`, `s_curve`,
`swiss_roll`, `swiss_roll_hole`, `severed_sphere`, and `torus`. PCA wins on
several already-separated 2-D external shapes and on some global blob/shape
arrangements.

The trade-off is systematic rather than accidental:

- UMAP improves local recall and manifold-coordinate recovery.
- PCA preserves pairwise distance ordering, density, and centroid arrangement.
- UMAP's density correlation is lower than PCA on 29 of 31 shapes.
- UMAP's distance Spearman correlation is lower than PCA on all 31 shapes.
- PCA→UMAP does not recover the global geometry; it behaves much like UMAP.

This matches the intended UMAP parameter trade-off: `n_neighbors` controls the
local/global balance, and lower `min_dist` creates tighter clumps. The official
[UMAP parameter guide](https://umap-learn.readthedocs.io/en/latest/parameters.html)
describes these effects directly. The broader algorithm comparison follows the
[scikit-learn manifold-learning documentation](https://scikit-learn.org/stable/modules/manifold.html).

## 4. Direct audit of the DelTriC component-growth backend

The direct audit used the settings from the current growth launcher:

```text
UMAP, n_neighbors=15, n_epochs=100, projected hard gate enabled
seed hard limit=0.25, growth/hard limit=0.9
initial relation=union, growth relation=union, kNN=50
hard_gate_mode=knn_relaxed
seed relaxed limit=0.75, growth relaxed limit=1.7
local selectivity threshold=0.1, scope=edge
```

Input features were standardized exactly as the DelTriC plotting wrapper does.
The direct run succeeded on all 24 dataset/method combinations.

| Curated dataset | PCA: ARI / final edges | UMAP: ARI / final edges | Observation |
| --- | ---: | ---: | --- |
| compound easy, 3,600×2 | 0.730 / 9,175 | 0.730 / 9,175 | Projection is skipped for native 2-D input |
| compound harder, 4,800×2 | 0.919 / 11,698 | 0.919 / 11,698 | Same native 2-D graph |
| overlap blobs, 4,000×10 | 0.000 / 824 | 0.196 / 3,992 | UMAP avoids PCA graph collapse |
| variable-density blobs, 4,500×20 | 0.012 / 1,592 | 0.170 / 3,242 | UMAP retains more usable structure |
| close S-curves, 5,000×50 | 0.041 / 8,025 | 0.266 / 11,411 | UMAP is materially better, but more permissive |
| moons, 2,500×2 | 0.966 / 6,371 | 0.966 / 6,371 | Native 2-D graph |
| blobs 8-D, 3,500 points | 0.007 / 2,077 | 0.221 / 6,325 | UMAP avoids over-pruning |
| H2MG 64-D/50 | 0.011 / 590 | 0.075 / 1,218 | Both are difficult; UMAP retains more edges |
| H2MG 64-D/70 | 0.014 / 605 | 0.065 / 1,138 | Same pattern |
| digits, 1,797×64 | 0.078 / 1,489 | 0.741 / 3,865 | Largest practical UMAP win |

Across all 12 curated datasets, mean large-component ARI is `0.419` for UMAP
versus `0.288` for PCA, and mean final edge count is `6,407` versus `5,075`.
The UMAP mean is not a claim that UMAP is always correct: it is a better
front-end for this graph-growth rule on this curated suite, while still
requiring conservative edge decisions downstream.

Note that 5 of the 12 curated datasets are native 2-D, where the projection is
skipped entirely and every method produces a byte-identical graph. Those 5 ties
dilute both means. The paired comparison over the **7 datasets where the
projection actually changes the Delaunay graph** is far starker: UMAP `0.247`
versus PCA `0.024`. That is the honest number for "does the projection choice
matter", and it is the one to quote.

### 4.1 t-SNE through the actual backend

The report originally rejected t-SNE on the strength of the bounded `n=500`
variant suite (section 6) without ever running it through
`component_growth_graph`. That gap has now been closed. `utils_pruning.py`
already supports `method='tsne'` natively (perplexity default 30, `init='pca'`,
`random_state=42` — the same configuration as `tsne30` in the variant suite),
so no DelTriC code was changed. All 12 curated datasets, identical growth
settings to the table above:

| Curated dataset | PCA | UMAP | t-SNE | Final edges P / U / T |
| --- | ---: | ---: | ---: | --- |
| compound easy, 3,600×2 | 0.730 | 0.730 | 0.730 | tie — native 2-D |
| compound harder, 4,800×2 | 0.919 | 0.919 | 0.919 | tie — native 2-D |
| overlap blobs, 4,000×10 | 0.000 | **0.196** | 0.000 | 824 / 3,992 / 8,403 |
| variable-density blobs, 4,500×20 | 0.012 | **0.170** | 0.162 | 1,592 / 3,242 / 8,770 |
| close S-curves, 5,000×50 | 0.041 | **0.266** | 0.200 | 8,025 / 11,411 / 11,612 |
| moons, 2,500×2 | 0.966 | 0.966 | 0.966 | tie — native 2-D |
| blobs 8-D, 3,500 points | 0.007 | **0.221** | 0.206 | 2,077 / 6,325 / 8,371 |
| H2MG 64-D/50 | 0.011 | 0.075 | **0.418** | 590 / 1,218 / 3,281 |
| H2MG 64-D/70 | 0.014 | 0.065 | **0.322** | 605 / 1,138 / 3,316 |
| blobs large 2-D, 5,000×2 | 0.129 | 0.129 | 0.129 | tie — native 2-D |
| digits, 1,797×64 | 0.078 | 0.741 | **0.798** | 1,489 / 3,865 / 4,367 |
| varied 2-D, 2,500×2 | 0.549 | 0.549 | 0.549 | tie — native 2-D |
| **Mean, all 12** | 0.288 | 0.419 | **0.450** | |
| **Mean, 7 non-tied** | 0.024 | 0.247 | **0.301** | |

**The mean flips to t-SNE, but win/loss is 3–4 in UMAP's favour.** The mean
moves because the magnitudes are strongly asymmetric. t-SNE's wins are large
and concentrated on the hardest high-dimensional sets — H2MG 64-D at `0.418`
versus `0.075` and `0.322` versus `0.065`, a 5–6× gap on the two datasets this
report previously described only as "both are difficult". Three of UMAP's four
wins are near-ties (`0.170`/`0.162`, `0.221`/`0.206`, and close S-curves
`0.266`/`0.200`). UMAP has exactly one decisive win: overlap blobs.

**One mechanism explains the whole table: t-SNE is more permissive here.** The
final edge counts show it directly — t-SNE retains far more edges on every
projected dataset (8,403 vs 3,992; 8,770 vs 3,242; 3,281 vs 1,218). t-SNE's
tight clumps shrink projected edge lengths, so more Delaunay edges clear the
projected hard gate at the same threshold. That rescues the cases where UMAP
over-prunes into nothing — on H2MG/50 UMAP kept only 1,218 of 6,123 candidate
edges and scored `0.075` — and it is fatal where clusters genuinely overlap: on
overlap blobs t-SNE collapsed to **3 large components and ARI `0.000`**, an
over-merging failure, the exact mirror of PCA's `0.000` on the same dataset,
which was an over-pruning failure down to 824 edges and zero large components.

**The cost argument against t-SNE does not survive measurement at this scale.**
t-SNE was *faster* than UMAP on most curated datasets: 15.3s vs 47.1s on
overlap blobs, 16.9s vs 25.8s on variable density, 11.5s vs 19.5s on blobs 8-D,
5.0s vs 5.6s on digits. It was slower only on close S-curves (17.8s vs 3.4s)
and marginally on H2MG. Section 6's "it is expensive" claim was never measured
and is wrong for 2–5k points. It may still hold at the 20k-point target —
scikit-learn's Barnes-Hut t-SNE is O(n log n), so it should not blow up, but
nobody has measured it there.

**Caveats.** Single seed, one t-SNE configuration, 12 datasets. Every ARI in
this table is still low in absolute terms on the hard sets — `0.20`–`0.30` is a
relative win over a failure, not a working clustering. The near-ties above are
not distinguishable from seed noise; the five-seed pass in section 9 applies to
this table as much as to the scope sweep.

### Local selectivity scope sweep

The following four cases were rerun with identical UMAP and growth settings,
changing only the local selectivity scope:

| Dataset | `edge` ARI / edges | `edge_2hop` ARI / edges | `component` ARI / edges |
| --- | ---: | ---: | ---: |
| overlap blobs | 0.196 / 3,992 | 0.205 / 3,995 | 0.205 / 3,994 |
| variable density | 0.170 / 3,242 | 0.170 / 3,242 | 0.170 / 3,242 |
| close S-curves | **0.266 / 11,411** | 0.190 / 11,230 | 0.098 / 8,213 |
| digits | 0.741 / 3,865 | **0.750 / 3,942** | 0.749 / 3,893 |
| Macro mean | **0.343** | 0.329 | 0.306 |

The close-S-curves result is the key negative result. A broader context can
mistake a legitimate manifold chain for a bridge when the graph is crowded or
the projection folds nearby strands together. `edge_2hop` should therefore not
become the unconditional default merely because it has more context.

## 5. Edge properties: what worked and what did not

The main edge study evaluates 22 properties. The most useful UMAP results are
shown below; PCA and PCA→UMAP are included in the interpretation where they
change the conclusion.

| Property | Label bridge AUC | Ambient non-local AUC | Finding |
| --- | ---: | ---: | --- |
| Original edge length | 0.889 | 0.989 | Strong baseline; necessary but mostly a global/scale rule |
| Original edge / endpoint kNN scale | 0.903 | 0.998 | Strongest non-oracle local normalization |
| Original edge / minimum incident length | 0.902 | 0.946 | Best simple incident-neighbour contrast |
| Projected edge length | 0.892 | 0.878 | Useful with UMAP; weak with PCA on the non-local proxy |
| Endpoint context disagreement | 0.664 | 0.757 | Real local signal after excluding the candidate edge |
| Endpoint minimum precision | 0.659 | 0.742 | Similar signal, easy to interpret operationally |
| Two-hop context disagreement | 0.688 | 0.587 | Moderate on semantic bridges, inconsistent on the proxy |
| Ambient/projected density log ratio | 0.715 | 0.547 | Useful at projection/density transitions, not universal |
| Edge-centre density | 0.611 | 0.655 | Weak-to-moderate and projection-dependent |
| Endpoint density asymmetry | 0.648 | 0.626 | Useful as a transition indicator, not a decision alone |
| Common-neighbour count | 0.531 | 0.529 | Essentially chance |
| Triangle count / boundary flag | 0.522 | 0.512 | Essentially chance in this Delaunay graph |
| Direct ambient union-kNN disagreement | 0.744 | 1.000 | Strong, but oracle-like for the proxy target |

The length numbers are not a reason to discard local features. They say that
the first question should be “is the edge unusually large relative to the
local scale?” rather than “can a topology statistic identify it?” The latter
statistics fail because nearly every interior 2-D Delaunay edge participates in
similar triangle structures; boundary status and triangle count do not encode
semantic separation reliably.

### Conditioning on the obvious length signal

To test whether local context adds anything beyond a global length threshold,
the 111 balanced `n=500` label runs were evaluated in length bands. UMAP results
are macro AUC over the runs in each band:

| Length band | Incident-min ratio | kNN-scale ratio | Endpoint context | Two-hop context |
| --- | ---: | ---: | ---: | ---: |
| All edges | 0.943 | 0.942 | 0.673 | 0.644 |
| Original length p25–p75 | 0.742 | 0.745 | 0.614 | 0.650 |
| Original length p50–p90 | **0.801** | 0.794 | 0.661 | 0.642 |
| Below original p90 | 0.797 | **0.804** | 0.666 | 0.635 |

The gray-band result is the practical answer to the colleague's hypothesis:
local information remains predictive after obvious long edges are removed, but
endpoint context is a secondary signal rather than a replacement for local
length/scale normalization. The p25–p75 band is noisier because it contains
fewer positive bridges and many genuinely ambiguous edges.

### Combined local model

The cross-shape grouped logistic model uses 19 features and holds out complete
shape groups. It excludes the direct ambient kNN disagreement features.

| Training sample | PCA | UMAP | PCA→UMAP | Pooled |
| --- | ---: | ---: | ---: | ---: |
| Complete 438-run edge study | 0.812 ± 0.041 | 0.806 ± 0.061 | 0.815 ± 0.069 | 0.797 ± 0.044 |
| Balanced conditional `n=500` ablation, length only | 0.858 | 0.865 | 0.867 | 0.863 |
| Same ablation, context only | 0.821 | 0.753 | 0.758 | 0.791 |
| Same ablation, length + context | **0.910** | 0.852 | 0.863 | **0.899** |

The two rows are intentionally not conflated. They use different subsets and
sampling schemes. Together they show that the feature family transfers
moderately across shapes, and that context can improve the length-only model
in a controlled balanced study, but the improvement is not guaranteed for every
projection or sample. A future MLP should be evaluated with the same grouped
holdout protocol and retrained after the latest dataset refresh; prior synthetic
MLP results should not be treated as this experiment's validation result.

## 6. Dimensional-reduction comparison at the edge level

The companion variant suite uses 32 synthetic datasets (`n=500`, 16 shape
families at two ambient dimensions where available), the same 2-D Delaunay
edge extraction, the same 22 features, and the same targets. The standard DR
metrics below are macro averages over all 32 successful runs per normal method;
LLE has 31 successful runs.

| Method | Recall@10 | N-preserve | N-surprise | Distance Spearman | Density corr. | Manifold corr. | Label-edge endpoint-context AUC | Degenerate runs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PCA | 0.502 | 0.958 | 0.088 | 0.879 | 0.566 | 0.448 | 0.722 | 0/32 |
| UMAP `n_neighbors=5` | 0.581 | 0.947 | 0.044 | 0.543 | 0.173 | 0.550 | 0.689 | 0/32 |
| UMAP `n_neighbors=15` | 0.596 | 0.959 | 0.044 | 0.554 | 0.197 | **0.603** | 0.694 | 0/32 |
| UMAP `n_neighbors=50` | 0.563 | 0.957 | 0.049 | 0.640 | 0.229 | 0.440 | 0.701 | 0/32 |
| UMAP `n_neighbors=100` | 0.545 | 0.955 | 0.052 | 0.645 | 0.215 | 0.407 | 0.688 | 0/32 |
| UMAP `min_dist=0` | 0.583 | 0.956 | 0.044 | 0.536 | 0.169 | 0.582 | 0.703 | 0/32 |
| PCA→UMAP `n=50` | 0.563 | 0.957 | 0.049 | 0.640 | 0.229 | 0.440 | 0.701 | 0/32 |
| t-SNE, perplexity 30 | **0.685** | **0.963** | **0.033** | 0.720 | 0.579 | 0.410 | **0.825** | 0/32 |
| MDS | 0.521 | 0.954 | 0.082 | **0.888** | **0.593** | 0.387 | 0.708 | 0/32 |
| Isomap, 15 neighbours | 0.432 | 0.950 | 0.095 | 0.783 | 0.400 | 0.486 | 0.690 | 0/32 |
| Random projection | 0.377 | 0.933 | 0.125 | 0.688 | 0.470 | 0.457 | 0.706 | 0/32 |
| LLE, 15 neighbours | 0.337 | 0.908 | 0.134 | 0.551 | 0.289 | 0.431 | 0.576 | **11/31** |
| Spectral embedding, 15 neighbours | 0.309 | 0.916 | 0.118 | 0.544 | 0.243 | 0.557 | 0.575 | **12/32** |

These figures are per-run means recomputed directly from
`dimred_variants/variant_dr_metrics.csv` (32 rows per method, 31 for LLE). An
earlier revision of this table carried values from a superseded aggregation
that were shifted by roughly `0.01`–`0.02`; the method ordering was unchanged
and no conclusion in this report depended on the difference. MDS is still not a
practical answer at the 20k-point DelTriC scale despite its strong global
geometry on this small suite.

The manifold-correlation column has been added because it is the one metric
where UMAP `n_neighbors=15` is the outright best method (`0.603`, against
t-SNE's `0.410`). It was omitted from the earlier revision, which made the
UMAP-versus-t-SNE comparison look more one-sided than it is. Manifold recovery
matters directly for the swiss-roll, s-curve, and clusters-on-swiss-roll cases
in the curated suite, and it is the strongest single argument for keeping UMAP
as the default.

### Algorithm decisions

**UMAP — keep as the production default.** The default `n_neighbors=15` is a
good local choice. `n_neighbors=50` is a credible alternative when the
projection should preserve more broad geometry; it improves distance Spearman
and edge projected-length behavior but slightly lowers local recall. `min_dist=0`
creates useful clumps but worsens density preservation and should not be used
as a universal setting.

**PCA — keep as a diagnostic and global baseline.** It is fast, stable, and
preserves global pairwise structure, but it misses curved manifolds and can
over-prune the DelTriC graph after the projected hard gate.

**t-SNE — promoted from "research control" to a live candidate.** It had the
best local recall and strongest contextual edge AUC in the bounded synthetic
test, and section 4.1 now shows it also wins on mean backend ARI. Two of the
three original objections did not survive testing: it was *faster* than UMAP on
most curated datasets, and openTSNE has provided a `transform()` for new points
for years, so "not naturally a stable transform" is outdated. The objection
that does survive is that it breaks the local/global tradeoff in a way that
makes it more permissive through the hard gate — good on sparse
high-dimensional sets, catastrophic on genuinely overlapping clusters. It is
not yet validated at 20k points. Treat it as a candidate requiring a seeded,
at-scale run, not as a drop-in replacement.

Note also that t-SNE contradicts this report's own framing of the local/global
tradeoff as "systematic rather than accidental" (section 3). t-SNE achieves
PCA-level density correlation (`0.579` vs PCA's `0.566`) *and* the best local
recall in the suite. If the tradeoff were structural, that row could not exist.
The tradeoff described in section 3 is a property of UMAP's objective, not a
law about 2-D embeddings.

**MDS — global control only.** It has the strongest distance preservation in
the small variant suite, but its pairwise cost makes it unsuitable for the
large DelTriC sweep.

**Isomap — possible niche alternative.** It preserves geodesic structure, but
the shortest-path/eigensolver costs grow quickly and its local edge performance
was below PCA/UMAP. The scikit-learn documentation gives the relevant
nearest-neighbour, shortest-path, and eigendecomposition costs.

**LLE and spectral embedding — reject as routine DelTriC projections.** LLE
failed on `blobs_informative_subspace__d50__n500` because the ARPACK factor was
singular, and 11 of its 31 successful runs were graph-degenerate. Spectral
embedding produced disconnected/degenerate projected graphs in 12 of 32
runs. These are algorithm/topology failures, not merely low scores.

**Random projection — useful speed baseline, not a quality choice.** It is
cheap and non-degenerate but has the lowest local recall in the normal-method
group.

The one real-data variant check on 1,797-point digits was consistent with the
large benchmark: PCA recall@10 was `0.118`, UMAP `n_neighbors=15` was `0.470`,
and t-SNE was `0.582`; the latter does not justify its production cost.

## 7. Proposed DelTriC local scoring rule

The following is a concrete, implementable rule design. It intentionally does
not hard-code final weights; those must be learned/calibrated on held-out
shapes.

For a projected-Delaunay edge `e=(u,v)`, compute:

```text
L_o       = original-space length of e
L_p       = projected-space length of e
s_o(u)    = ambient kNN radius at u
s_o(v)    = ambient kNN radius at v
r_knn     = log(L_o / sqrt(s_o(u) * s_o(v)))
r_inc     = log(L_o / median(other incident original edge lengths))
r_min     = log(L_o / minimum(other incident original edge length))
r_proj    = log(L_p / local projected edge scale)

p_1       = 1 - endpoint-context ambient-kNN precision
p_2       = 1 - two-hop ambient-kNN precision
d_density = abs(log(s_o(u) / s_o(v)))
rho       = local density of nearby Delaunay edge centres
```

All ratios should be clipped and converted to within-run robust ranks or
median/IQR-normalized values. This prevents a dimension or global scale change
from changing the meaning of a threshold.

### Three-band decision policy

1. **Protect / keep band**
   - edge is mutual ambient kNN, or has strong union-kNN support;
   - endpoint incident ratios are ordinary;
   - a short alternative path or dense local support exists;
   - do not prune solely because the projection makes it look long.

2. **Hard-prune band**
   - existing original-space hard gate rejects it, and the edge is not a
     protected trusted relation;
   - or `r_knn`/`r_inc` is extreme and both endpoint context disagreement and
     density-transition evidence agree.

3. **Gray band**
   - use a calibrated logistic/MLP score from the feature vector;
   - let the model abstain when the score is near 0.5 or outside the training
     support;
   - use `p_1`, `p_2`, and `d_density` as tie-breakers, not as independent
     hard gates.

A simple initial score for experiments is:

```text
S(e) = w1*r_knn + w2*r_inc + w3*r_min + w4*r_proj
       + w5*p_1 + w6*p_2 + w7*d_density - w8*rho
```

The signs shown are intuitive, not fitted coefficients. The weights and final
operating point must be selected with shape-grouped validation against both
bridge precision/recall and final clustering ARI. A threshold is unavoidable:
adaptive normalization removes arbitrary units, not the cost of false pruning
versus false keeping.

### Why this fits the existing component-growth backend

The current backend already has the right architectural separation:

- projected space determines the candidate Delaunay graph;
- original space supplies edge-size and kNN scale information;
- strict seeds establish components;
- later growth can be more permissive;
- local selectivity is a cached relation rather than a dense all-pairs matrix.

The safest next implementation is to add the gray-band score after the existing
hard gate and before/inside growth, preserving all strict seeds. Do not replace
the current seed graph with an unconstrained MLP prediction. A false negative
at seeding can disconnect an entire cluster, whereas an abstained growth edge
can be reconsidered later.

## 8. What failed or remains unproven

### No universal local-only theorem

Two edges can have the same local lengths and two-hop topology while belonging
to different semantic structures in different datasets. The experiments show
predictive regularity, not identifiability. A local rule can be useful and
still fail on folds, sparse manifolds, density transitions, or semantically
overlapping clusters.

### Topology-only pruning failed

Common neighbours, triangle count, and Delaunay boundary status were near
chance. They should be retained only as model inputs if they improve a held-out
model; they should not be hand-written pruning rules.

### Broader context is not monotonically better

The close-S-curves scope sweep is a direct counterexample. More context can
make a valid manifold edge look globally inconsistent. This is why `edge` is
the default scope and why a two-hop feature should be allowed to abstain.

### Projection can change the graph, not only the edge score

UMAP and PCA may produce different Delaunay edges. Comparing only projected
edge lengths would miss this. The direct audit shows UMAP often retains more
edges and gives better ARI, but it also makes the graph more permissive and
more expensive.

### The projected hard gate is mis-calibrated against projection scale

This is the most important thing section 4.1 exposed, and it was not visible
before t-SNE was run through the backend.

Ordering the three projections by how many edges they retain gives
PCA < UMAP < t-SNE on every projected dataset, and the two catastrophic
failures in the curated suite sit at the two ends of that ordering — PCA
over-pruning overlap blobs to 824 edges and zero large components, t-SNE
over-merging the same dataset to 3 large components. Both are ARI `0.000`. They
are the same defect seen from opposite sides: a fixed projected-length
threshold is being applied to projected lengths whose scale is set by whichever
projection was chosen.

On that reading, t-SNE did not cluster better than UMAP on H2MG. It moved where
the threshold happened to land, and on those two sparse 64-D datasets UMAP's
landing spot was bad (1,218 of 6,123 edges retained, ARI `0.075`). The
projection ranking in section 4.1 is therefore substantially an artifact of
gate calibration, and should not be read as a ranking of embedding quality.

This points back at the best-supported finding in section 5: gate on
local-scale-normalized length rather than on raw projected length. If that fix
works, the choice of projection should matter considerably less than section
4.1 makes it appear — and the right way to validate the fix is to check whether
it *narrows* the spread between PCA, UMAP, and t-SNE on this suite.

### Large-scale alternatives were intentionally bounded

MDS, Isomap, LLE, and spectral embedding were run on `n=500` synthetic data and
on one 1,797-point digits set. They were not run across the 20k-point
supercomputer grid, and none has been run through the component-growth backend.
The stored large benchmark remains the authoritative comparison for PCA and
default UMAP at scale.

t-SNE is the exception: it has now been run through the actual backend on all
12 curated datasets at 1,797–5,000 points (section 4.1). It has still not been
run at 20k points, and the section 4.1 result is single-seed.

### Reproducibility caveat in the DelTriC environment

`environment.yml` declares networkx, UMAP, pandas, and statsmodels but not
PyTorch. `utils_pruning.py` contains optional GNN imports, yet some GNN class
definitions are evaluated even when the optional import is unavailable. The
direct backend audit therefore supplied the host CPU PyTorch installation in
`PYTHONPATH`; no DelTriC pruning code was changed. This should be fixed by
guarding all torch-dependent class definitions under `_gnn_available`, or by
declaring torch as a dependency if it is genuinely required.

## 9. Recommended next experiments

1. **Retrain one gray-band classifier** on the refreshed edge corpus, using
   shape/dataset-grouped splits and no direct ambient-kNN disagreement feature.
2. **Tune the operating point for graph outcomes**, not AUC alone: retained
   intra-cluster edge recall, bridge precision, number of large components,
   noise rate, and final ARI/NMI.
3. **Run five fixed seeds** on the 12 curated DelTriC datasets, for UMAP *and*
   t-SNE, and report the variance of Delaunay edge membership and final
   components. Several gaps in the section 4.1 table are small enough that they
   may not survive this.
4. **Compare UMAP `n_neighbors` 15 versus 50** in the actual component-growth
   backend. The variant study suggests 50 is a useful global/local compromise,
   but it needs a final-graph validation.
4b. **Recalibrate the projected hard gate to a scale-free statistic** — a
   within-run quantile or local-scale-normalized ratio rather than an absolute
   projected length — and rerun section 4.1. Success criterion is that the
   PCA/UMAP/t-SNE spread *narrows*, and specifically that PCA's overlap-blobs
   collapse and t-SNE's overlap-blobs over-merge both disappear. This is the
   highest-value experiment on this list.
4c. **Run t-SNE at 20k points** on the supercomputer grid before treating it as
   anything more than a candidate. Barnes-Hut is O(n log n) so it should be
   tractable, but it is unmeasured. The cost objection in section 6 was wrong
   at 2–5k points; that says nothing about 20k.
5. **Add explicit manifold-chain protection**: an edge with a short alternative
   path, high endpoint precision, and ordinary original local scale should not
   be bridge-pruned merely because a two-hop aggregate is inconsistent.
6. **Validate on unseen real embeddings** with no semantic labels. Use stability
   under resampling/seeds, graph connectivity, and downstream clustering
   consistency; do not call the ambient-kNN proxy ground truth.
7. **Fix the optional PyTorch import guard** before handing the audit to a clean
   DelTriC environment.

## 10. Reproducibility artifacts

Scripts and results below were reorganized after this report was written:
DR-algorithm-comparison scripts stayed in `kinit`, and scripts whose whole
purpose is DelTriC edge pruning moved into `deltric_minimal/studies/` in this
repo. The moved scripts import `edge_pruning_study.py` from a sibling `kinit`
checkout (see `deltric_minimal/studies/_kinit_path.py`).

### kinit scripts (dim-reduction comparison)

- `edge_pruning_study.py` — computes the 22 edge features, targets, AUCs, and
  grouped model. Kept in kinit because `dimred_variant_study.py` depends on it.
- `dimred_variant_study.py` — compares UMAP settings, PCA, t-SNE, MDS, Isomap,
  LLE, spectral embedding, and random projection.
- `evaluate_variant_metrics.py` — scores variant embeddings with the standard
  kinit DR metrics.

### deltric_minimal/studies scripts (DelTriC-specific)

- `edge_conditional_study.py` — evaluates gray length bands and feature
  ablations.
- `deltric_growth_audit.py` — calls the actual DelTriC component-growth
  backend without plotting.

### Main output directories

- kinit: `results/edge_study`, `results/dimred_variants`,
  `results/dimred_external_digits`
- deltric_minimal/studies: `results/edge_conditional`,
  `results/deltric_growth_audit.csv`, `results/deltric_growth_audit_tsne.csv`
  (section 4.1), `results/deltric_scope_audit.csv`
- Stored supercomputer summaries: `summary_overall.csv` from the external
  benchmark snapshot named above (not part of either repo)

### Commands

Main edge study (from kinit):

```bash
uv run python edge_pruning_study.py --out results/edge_study --max-n 5000
```

Variant suite (from kinit):

```bash
uv run python dimred_variant_study.py \
  --out results/dimred_variants --heavy-max-n 500 \
  --shapes blobs_separated,blobs_two,blobs_varied_density,blobs_hierarchical,\
blobs_informative_subspace,blobs_hearts,blobs_meeples,nested_spheres,\
uniform_hypercube,moons,s_curve,swiss_roll,swiss_roll_hole,severed_sphere,\
torus,clusters_on_swiss_roll
```

Direct DelTriC audit (from `deltric_minimal`; needs the optional dependency
set — torch, umap-learn, networkx, pandas, statsmodels — documented in the
repo README):

```bash
python studies/deltric_growth_audit.py --methods umap,pca \
  --out studies/results/deltric_growth_audit.csv
```

t-SNE backend audit (section 4.1; `method='tsne'` is already supported by
`utils_pruning.py`, so no DelTriC code was changed):

```bash
python studies/deltric_growth_audit.py --methods tsne \
  --out studies/results/deltric_growth_audit_tsne.csv
```

Local-scope audit:

```bash
python studies/deltric_growth_audit.py --methods umap \
  --scopes edge,edge_2hop,component \
  --datasets 02_v3_overlap_blobs_10d_4000_k6,04_v3_vardensity_blobs_20d_4500_k6,\
10_v3_close_scurves_50d_5000_k6,digits \
  --out studies/results/deltric_scope_audit.csv
```

## Final decision

Use **UMAP 2-D plus original-space adaptive edge scoring** as the main DelTriC
research direction. Keep PCA for global-geometry diagnostics and as a useful
failure comparison. Do not switch to MDS, LLE, spectral embedding, or random
projection as the routine production projection.

**Revised on t-SNE.** An earlier version of this report also ruled out t-SNE.
That rejection was made without ever running t-SNE through the
component-growth backend, and two of its three stated grounds — cost and
out-of-sample transform — turned out to be wrong or outdated. Section 4.1 shows
t-SNE ahead of UMAP on mean backend ARI. UMAP stays the default on three
grounds that do hold: it is the best method in the suite on manifold recovery
(`0.603` vs `0.410`), it is the only one of the two validated at 20k points,
and t-SNE's single decisive loss is an over-merging collapse to ARI `0.000`,
which is a worse failure mode for DelTriC than under-merging. t-SNE is now a
live candidate pending the seeded and at-scale runs in section 9.

The more useful conclusion is that the projection ranking is partly an artifact
of hard-gate calibration rather than a ranking of embedding quality, and that
fixing the gate (experiment 4b) should take priority over choosing between
these projections at all.

For pruning, the strongest defensible claim is:

> Local edge neighbourhoods contain enough information to improve a global
> length gate, especially when local original-space scale is included; they do
> not contain enough information to replace global calibration or guarantee a
> correct semantic edge decision.

That is a positive result for DelTriC: the existing component-growth design can
be strengthened with local residual evidence, while the experiments also give
clear guardrails against over-pruning manifold chains and over-interpreting
projection-space topology.
