# DelTriC growth asymmetry audit — 8 October 2026

Unequal growth is real in controlled tests and in the unequal-density development dataset. This does not establish that the unspecified animation the user remembers has the same cause. There is no demonstrated fixed left/right preference in the main round-based growth function. Density, initial seeds, graph geometry, cost normalization, competition, and animation conventions can all produce apparent or actual imbalance.

## What was tested

`growth_asymmetry_audit.py` calls the existing clustering functions without modifying them. No holdout inputs are read.

- Fresh launcher-default runs: `varied_2d_5c` and `11_v2_moons_2d_2500`.
- Historical saved full-analysis graphs: those two datasets plus `04_v3_vardensity_blobs_20d_4500_k6` and `14_v2_blobs_8d_3500_k7`.
- Controlled weighted graphs: equal branches, longer edges on one branch, a changed neighbor-spacing input on one branch, a two-source bottleneck, and exact cost ties. These isolate function behavior; they are not end-to-end Delaunay/UMAP benchmark datasets.
- A fixed-graph permutation of point and seed-component IDs; comparison of accepted edges, acceptance rounds, and final component partitions.
- Comparison of the actual competitive assignment rule against independent per-cluster shortest paths through initially unassigned points. This comparator changes ownership semantics and is not an automatically justified replacement.

The two fresh runs reproduce exactly when replaying growth with their saved settings, including bounded real restoration. Cost-gate replay reproduces all six saved final label arrays exactly. Fixed-graph renumbering changes no growth edge or round, with partition ARI 1.0 in all six cases. Reversing cluster IDs changes no cost-gate assignments in these six datasets.

Historical growth artifacts do **not** reproduce with today's growth function and their recorded limits. Accepted-mask mismatches are 571, 554, 2,759, and 1,899 respectively. Their reconstructed animation states are descriptive evidence about those saved artifacts, not a replay of today's algorithm. The recorded historical settings were seed 0, original growth 1.05, projected growth 1.35, and assignment reach 10. Today's native-2D launcher uses seed 0.1, growth 1.5, no additional projected growth cap, bounded restoration 2.0, and assignment reach 15. Both fresh cases use the median fallback of the first-mode seeding procedure.

## Measured imbalance in current defaults

Each of the five reference classes in `varied_2d_5c` has 500 points. Coverage means membership in an active growing component; it does **not** measure correct cluster assignment.

| Reference class | Median within-class edge length | Active at seeds | Active after main growth | Assigned after final phase |
|---|---:|---:|---:|---:|
| 0, densest by edge spacing | 0.0168 | 99.8% | 99.8% | 99.8% |
| 1 | 0.0326 | 92.4% | 96.0% | 99.2% |
| 2 | 0.0634 | 53.4% | 79.6% | 94.8% |
| 3 | 0.0763 | 48.6% | 70.2% | 84.0% |
| 4, sparsest by edge spacing | 0.0999 | 14.0% | 15.8% | 53.8% |

The two moons have much closer median edge lengths, 0.04273 and 0.04238. Their active coverage starts at 83.0% versus 86.3% and ends at 94.2% versus 92.7%. Their final assigned coverage is 99.44% versus 99.68%. Thus unequal sampling/seeding can create modest differences without a persistent one-sided advantage.

## Mathematics of the main growth stage

For a component C, let m_C be the median length of its already accepted/reference internal edges and s_C = max(Q75_C − m_C, 10^-12). A boundary edge of original-space length l passes when

    (l − m_C) / s_C < alpha,
    equivalently l < m_C + alpha s_C.

Multi-dimensional inputs also need the corresponding projected-space test. Native 2D currently disables that extra cap. The reference includes bounded restorations; merely incident rejected boundary edges are excluded.

With alpha 1.5 and seed lengths ranging uniformly from 0.7 to 1.3, the first threshold is 1.225. Two 8-hop branches of length 1.0 both grow completely, on precisely the same rounds. Changing only the right branch's lengths to 1.4 produces 8 additions on the left and zero on the right. This is threshold-induced asymmetry, not a traversal-order effect.

Components with **more than** 10 initial seed edges may grow. A global seed cutoff can leave sparse areas without an eligible seed even when a dense area is almost entirely seeded. A boundary edge may qualify from either incident active component; it need not satisfy both components' scales. All evaluations finish before merges/statistics updates. Updated reference distributions can subsequently loosen or tighten the gate; there is no theorem that growth always accelerates.

A round is a boundary-expansion event, not equal physical distance. It can absorb a whole small seed component or merge two active components. Equal rounds therefore need not imply equal numbers of points or equal screen distance.

## Mathematics of the final cost gate

The current defaults use p = q = 0.5. For a candidate directed hop u → v from component C,

    w_C(u,v) = [l_uv / avg_C × (r(v)/r_C)^q]^(1+p)
             = (l_uv / avg_C)^1.5 × (r(v)/r_C)^0.75.

Here avg_C is the mean original-space length of **all Delaunay edges** with both endpoints initially assigned to C, including edges absent from the retained graph. r(v) is mean distance to the 15 nearest neighbors; r_C is its median over C. Component reference values stay fixed during assignment. An assignment requires accumulated path cost at most the reach budget.

Consequences, with other quantities fixed:

- Doubling edge length multiplies hop cost by 2^1.5 = 2.828.
- Multiplying target neighbor spacing by four multiplies hop cost by 4^0.75 = 2.828.
- Doubling the component's mean edge length divides its hop costs by 2.828. Different component scales therefore affect competition.
- A geometric segment split into m equal hops, with equal density factors, has total cost proportional to m^-0.5. Many short hops can be cheaper than a few long hops over the same distance.

Using the actual cost-gate function with equal unit-length 8-hop branches and reach 10 assigns all 8 points on each side. Changing only the right-side supplied neighbor-spacing values from 1 to 4 assigns 8 on the left and 3 on the right. This isolates the density term; the supplied neighbor values were controlled inputs, not recomputed from a synthetic point cloud.

## Competition and exact ties

The implementation permanently assigns each noise point to its first successful claimant. Other clusters cannot propagate through that point. With cluster-dependent costs, this is competitive region growth, not the same optimization as minimizing independently computed per-cluster path costs at every destination.

A controlled junction demonstrates the distinction. A reaches the junction at cost 1, B at cost 2. Continuing to the tail costs A another 100 and B another 1. At reach 5, the implementation leaves the tail unassigned because A owns the junction; B's independent feasible path costs 3.

In actual frozen saved graphs, the independent-path comparator changes:

| Run | Changed point assignments | Additional assigned points |
|---|---:|---:|
| Current unequal-density 2D | 5 | 3 |
| Current moons | 0 | 0 |
| Historical unequal-density 2D | 0 | 0 |
| Historical moons | 0 | 0 |
| Historical variable-density 20D | 25 | 10 |
| Historical 8D blobs | 2 | 0 |

These differences isolate ownership/path semantics, not improved accuracy. Exact cost ties are resolved using the heap tuple `(cost, node_id, cluster_id, edge_id)`. A symmetric controlled junction changes owner when cluster IDs are swapped. No assignment changed when IDs were reversed in the six dataset cases above, so tie bias is demonstrated as possible, not identified as their main cause.

## What the animation can exaggerate

`perun_full_analysis.py` colors a merged growing component using its largest initial seed. A merge can recolor an already covered region, making a color appear to grow rapidly without corresponding new point coverage. With more than 20 seeds, its palette also reuses colors. The final graph stages choose colors afresh.

The growth GIF samples rounds; the assignment GIF samples cumulative accepted claims. Neither uses physical time or equal radial advance. Projected screen distance is also not original feature-space distance. In native 2D UMAP is bypassed, so UMAP cannot explain the two fresh cases here. For multi-dimensional cases it affects both the display and Delaunay topology, and the projected growth gate can affect decisions.

## Recent research

- Liu, Ma and Zhong, **Assessing and improving reliability of neighbor embedding methods: a map-continuity perspective**, Nature Communications, 30 May 2025. Identifies embedding discontinuities that exaggerate separation or create spurious structures: https://arxiv.org/abs/2410.16608 ; published DOI https://doi.org/10.1038/s41467-025-60434-9 . This supports caution about reading growth speed from an embedding, not a diagnosis of a particular DelTriC run.
- Poorna and Cicerone, **ScaleMAP: Preserving Local Density and Neighborhood Structure in Low-Dimensional Embeddings**, 28 May 2026, preprint: https://arxiv.org/abs/2605.30597 . Describes loss of neighborhood scale in UMAP/PaCMAP and suppression of sparse bridges. Its proposed remedy was not tested in this audit.
- UMAP's own clustering documentation states that density is not fully preserved and that false tears can occur: https://umap-learn.readthedocs.io/en/latest/clustering.html .

## Reproduction and artifacts

From the repository root, use the existing environment. To produce the two current inputs, run `deltric_minimal/run_plot_stages.sh` with `DATA_GLOB=data/varied_2d_5c.npz` and `OUT=studies/results/growth_asymmetry_audit/current_varied`, then with `DATA_GLOB=data/11_v2_moons_2d_2500.npz` and `OUT=studies/results/growth_asymmetry_audit/current_moons`. Set `EXPECTED_DATASET_COUNT=0` and `EXPECTED_ARI_COUNT=0` for these focused runs.

Run `deltric_minimal/.venv/bin/python deltric_minimal/studies/growth_asymmetry_audit.py`. Current artifact inspection is included when those two directories exist. Setting `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mplconfig NUMBA_CACHE_DIR=/tmp/deltric_asymmetry_numba` avoids excessive native threading and uses writable caches.

Outputs under `studies/results/growth_asymmetry_audit/`: `summary.json`, `class_coverage.csv`, per-run NPZ arrays, `growth_asymmetry_current.png`, and `growth_asymmetry_historical.png`. Assertions check the controls, current growth replay, all cost replays, reconstructed coverage, and fixed-graph permutation invariance. The clustering implementation was not edited.

## Follow-up: the user specifically means anomaly reassignment

The seeding and main-growth measurements above should not be treated as measurements of reassignment speed. The follow-up script `anomaly_reassignment_asymmetry.py` observes every accepted claim in the actual `_assign_anomalies_cost_gate` execution, recording its node, component, accumulated cost, and queue position. It verifies the saved final labels in two current cases and one historical variable-density case. The new outputs are in `results/growth_asymmetry_audit/reassignment_only/`.

### Queue order versus final reach

All components share one min-heap. It chooses the smallest accumulated path cost, so it offers no equal-per-component or equal-per-direction turns. The reach limit applies to each path; it is not a total budget depleted by all claims. Component edge-length and density references remain frozen throughout this phase.

For two equal-length eight-hop branches, with p=q=0.5:

| Right-side supplied neighbor spacing | Reach | Left/right final claims | Right claims when left finishes |
|---|---:|---:|---:|
| Equal to left | 30 | 8 / 8 | 7 (one-claim serialization difference) |
| Four times left | 30 | 8 / 8 | 2 |
| Four times left | 15 | 8 / 5 | 2 |

The middle case looks markedly one-sided during playback even though both branches are completely assigned at the end. The last case has both an ordering difference and genuinely unequal final reach. These controls change the supplied density inputs while preserving hop lengths to isolate the formula.

In a separate controlled graph, 16 frontier points all have cost exactly 1. Numbering the eight left points before the eight right points makes the first eight claims all appear on the left. Both sides are in the same cost layer and finish 8/8. Thus equal-cost serialization can produce a pure playback artifact. This demonstrates a possibility on a controlled weighted graph, not a claim that real continuous datasets contain many exact ties.

### Some older study GIFs have much stronger exponents

`density_fixed_formula_viz.py` sets p=3, reach=25 and q=4 for its headline fixed-density animation; it also renders explicit q=0 and q=2 variants. The general hop cost is

    (length / avg_C)^(1+p) × (neighbor_spacing / reference_C)^[q(1+p)].

The headline study therefore uses length exponent 4 and neighbor-spacing exponent **16**, compared with 1.5 and 0.75 in the current launcher. Merely doubling relative neighbor spacing multiplies a headline-study hop cost by 2^16 = **65,536**. The production function reproduces 8 left claims and zero right claims on the controlled graph with these settings and right-side neighbor spacing doubled. This parameter sensitivity is directly established by the formula and experiment; which variant the user's remembered GIF used is still unknown.

For a locally uniform route, if edge length and neighbor spacing both scale with h and component references stay fixed, the cost of crossing a fixed distance D approximately scales as

    D × h^[p + q(1+p)].

The exponent is 1.25 at current settings, or 19 in the older headline study. This approximation ignores boundary geometry and alternative graph routes, but explains why many short hops through a dense direction can cost less than fewer long hops through a sparse direction. It does not imply that the globally densest component always wins: normalization by each component's own reference can cancel a uniform density change.

### Observed reassignment-only measurements

In the current two-moons run, the stage receives 72 unassigned points from one reference class and 91 from the other. It reclaims 65/72 = **90.3%** and 87/91 = **95.6%**. Counting only newly colored points would show 65 versus 87, partially reflecting different available populations. Median claim costs are 4.60 and 4.36.

The historical 20D variable-density run shows much more separated queue timing. Within the first quarter of all 727 claims, the densest reference class has already received **96% of its eventual reassignments**; the sparsest class has received **1.2%**. This is measured ordering of the production function on that frozen graph, not inferred from screen area. The classes are reference-data regions, not directional halves of an identified user example.

The earlier independently computed paths comparison still applies to this stage: permanent ownership can block another component's cheaper downstream path. It changed five assignments in the current unequal-density run and 25 in the historical 20D run, but it has not been shown to improve clustering accuracy.

For reproduction, run `deltric_minimal/.venv/bin/python deltric_minimal/studies/anomaly_reassignment_asymmetry.py` after generating the two current inputs above. It produces `controlled_claim_order.png`, `dataset_claim_order.png`, claim-by-claim JSON, and NPZ arrays. No reassignment implementation was changed.
