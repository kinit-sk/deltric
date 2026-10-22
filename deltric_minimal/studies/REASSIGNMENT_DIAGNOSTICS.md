# Reassignment diagnostics on the laptop

Launch from the repository root:

```bash
./deltric_minimal/run_reassignment_diagnostics.sh
```

The window reads saved cases immediately. It uses the existing Python environment,
Tk, Matplotlib, and Pillow. Launching it does not recompute clustering.

## Investigating a failure

1. Rank by anomaly F1, ARI over all points, or ARI over true normals. The filter
   isolates F1 regressions, ARI regressions, or Gaia datasets. Negative deltas are
   regressions. A case can improve ARI while damaging anomaly detection.
2. Play or scrub **Full pipeline**: seeds, individual growth rounds, restoration,
   pruning, then actual reassignment claim order. **Reassignment** focuses on the
   frozen-graph comparison. The quality chart updates with the visible stage.
3. Use the graph projection, PCA, or original feature pairs. Gaia adds sky position,
   proper motion, and parallax/proper-motion pairs. These change the display only.
4. Click a point. Inspect its true label, initial and final cluster, claim number,
   original incoming-edge length, core distance, cumulative cost, and actual path.
   Black lines show the actual winning path. Existing assigned points never move.
5. Select a candidate cluster to ask why it did not win. The inspector replays the
   actual production queue to identify proposals rejected after the point was
   claimed. It also computes an optimistic route that ignores competing ownership.
   Orange dashed lines show that hypothetical route, not an actual accepted path.
6. Change reach, penalty, or density, then run a reassignment experiment. Sources,
   projection, graph, and original-space distances stay fixed. **Compare default**
   places the recorded result beside the experiment. Restore returns to the saved run.
7. Toggle **cost / density diagnostics** for incoming-hop scale and density ratios,
   winning-cost distributions, anomaly confusion counts, and anomaly-kind recall
   when kind labels are available. Points in the ratio scatter are also clickable.
8. Save the current chart, settings, and point explanation with the save button.

The inspector has a scrollbar. Use the Matplotlib toolbar to zoom or pan, then
turn off that toolbar mode to select a point again.

## What the evidence means

The baseline is the labeling immediately before reassignment in the current
`DelTriC()` estimator. It is not an old `cluster_tri` preset or a separate run.
The runner observes the actual function and verifies that replay produces exactly
the estimator's final labels. Ground truth is used only for scoring and diagnostics.

ARI is permutation invariant. Truth colors and predicted-cluster colors belong to
separate label sets. Recovering a normal point is not proof it reached the correct
cluster; seed-majority mismatches are a heuristic. F1 and recall are unavailable
for datasets without true anomalies, rather than reported as zero.

Alternative-route costs deliberately use different traversal semantics: original
assigned points remain barriers, while initially unassigned vertices can be crossed
even if a competing cluster later owns them. A cheap hypothetical path does not
prove the production queue should have chosen it. The inspector distinguishes
this from an actual queued proposal and its recorded rejection.

Growth frames reconstruct connected-component labels from the accepted edges of
each round. Internal restoration edges do not change those component memberships.
Animations batch claim prefixes for short playback; the window can scrub each claim.
This is a focused development-data suite, not an exhaustive benchmark or a search
for optimal settings. Holdouts are excluded. The clustering implementation is unchanged.

## Reproduce or extend

```bash
# Five-minute default budget, 45 seconds per case; validated caches are reused.
deltric_minimal/.venv/bin/python deltric_minimal/studies/reassignment_diagnostics.py evaluate

# A targeted run, with a strict total and per-case limit.
deltric_minimal/.venv/bin/python deltric_minimal/studies/reassignment_diagnostics.py evaluate \
  --budget 900 --timeout 150 --datasets gaia_ngc_2632 gaia_ngc_2516 \
  testing/real_world/gaia_vela_puppis blobs_informative_subspace__d50__n500__anom10

# PNG dashboards, cost anatomy, reassignment GIFs, and full-pipeline GIFs.
deltric_minimal/.venv/bin/python deltric_minimal/studies/reassignment_diagnostics.py export --top 6
```

Results live in `studies/results/reassignment_diagnostics/`: `ranking.csv`, run
records and logs, `cases/*.npz` and JSON metadata, and `exports/`. Caches include
dataset and pipeline-code fingerprints. Experiments affect only diagnostics and
never overwrite the recorded default case.
