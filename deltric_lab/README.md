# DelTriC Reassignment Lab

A local investigation workspace for anomaly reassignment. Uses the existing
DelTriC Python environment and a browser; there is no frontend build, CDN,
database service, or extra server dependency.

From the repository root:

```bash
./deltric_lab/run.sh
```

The launcher opens `http://127.0.0.1:8765`. Use `--no-browser` to keep the
launcher from opening a tab, or `--port 8766` if the port is occupied. Stop
the server with Ctrl+C. Saved work stays on disk.

## Investigating Gaia

1. Open **Gaia NGC 2632**, then **First harm**. Point 836 is a useful starting
   example in the previously recorded case: source cluster 16 absorbs it at
   claim 4, and all 12 fixed sources in that cluster are true anomalies.
2. Use **Single** for a large canvas, **Split** for a comparison, or **Focus
   mode** to give the graph the workspace. Scroll to zoom, drag to pan, and
   click to inspect. Zoom survives scrubbing, selection, and reloads. Linked
   views share a camera; unlinked views have independent cameras.
3. Compare **Current frame**, **Before reassignment**, **Ground truth**,
   **Recorded default**, and **Active run final**. Choose the graph projection,
   PCA, or the stored Gaia feature pairs. Source-cluster focus isolates a
   cluster's fixed sources and eventual claims; region selection uses Shift-drag.
4. Replay individual claims or batches. **Full pipeline** includes reconstructed
   seed/growth/restoration/pruning memberships before reassignment. **First
   harm** jumps to the first newly absorbed true anomaly. Keyboard shortcuts:
   Space plays/pauses, arrows step, F fits all, S splits, Escape clears selection.
5. The inspector shows the winning path, every hop's edge length, cluster mean,
   density ratio, hop cost, and cumulative cost. Competing attempts distinguish
   acceptance, existing ownership, and over-budget attempts that were never
   queued. Future evidence is faded. Click a path point or graph neighbor to
   follow the investigation.
6. **Cost anatomy** compares successful normal/anomaly claim costs, hop ratios,
   and source contamination. Click a ratio point or source-cluster table row
   to investigate it in the main canvas.
7. **Experiments** saves reach/penalty/density variants and reach sweeps. The
   density-floor switch runs the production gate's `density_clip=True` variant.
   Every experiment holds the graph, source labels, and core distances fixed.
   Opening one automatically compares it with its recorded default. Experiments
   retain a frozen baseline snapshot even after a fresh dataset run. An example
   zero-reach run is saved for NGC 2632.
8. **Notebook** saves editable observations with their run, selected point,
   and replay frame. **Return to evidence** restores that context. Unsubmitted
   draft text is also included in the saved workspace.
9. **PNG** downloads the main canvas. **Export evidence** downloads metadata,
   claims, queue attempts, notes, selected-point evidence, and workspace state.
   A run-evidence copy is also written to `data/exports/`.

## Normal examples

Reassignment also recovers normal points. Comparing those claims with the
harmful ones shows what a healthy claim looks like. Choose **Normal examples**
in the dataset filter, or sort by **Best Δ F1**. **Normal claim** in the replay
bar (or **Inspect a normal claim** in the empty inspector) steps to the next
claim of a true normal whose destination's fixed normal sources mostly share
its truth label. Repeated clicks cycle through these claims. The point filter
separates **Normal claims · majority match** from **mismatch**, and the
inspector states the destination's majority label.

| Dataset (search term) | What to look at |
| --- | --- |
| `blobs_varied_density__d2__n2000__anom10` | Reassignment helps: F1 0.51 → 0.72, ARI 0.73 → 0.91. All 199 normal claims match; only 4 anomalies are absorbed. Point 1136 (claim 3) is a one-hop claim from source 828 into cluster 66: 368 fixed sources, no anomalies, cost 2.37. |
| `11_v2_moons_2d_2500__anom10` | F1 +0.13. All 149 normal claims match, but 29 anomalies are absorbed. This mixed case shows good and bad claims side by side. Normal claims start at claim 9 (point 1057). |
| `other/square` | A clean run: 146 normal claims, all matching, and no anomalies in the data. |
| `gaia_ic_2391` | The Gaia counterpart: nine normal claims, no absorbed anomalies, F1 unchanged. Compare its hop costs with NGC 2632. |
| `varied_2d_5c` | 351 normal claims; 47 land in a cluster with a different majority label. Use the **mismatch** filter to see where clean claims still go to the wrong cluster. |

A majority match is evidence, not proof. Gaia cases contain only one normal
class, so every normal claim matches there.

## Dataset runs

**All development data** lists repository `.npz` datasets. Open an unrecorded
dataset and select a time limit (10–900 seconds, default 180). The runner launches
the current estimator in a separate process and writes a case and log. Existing
cases offer **Record fresh case**. Matching fingerprints reuse the evaluation
cache. Holdout, raw, and source directories are excluded, as in the original
diagnostics runner. Inputs need `X` and `y`; `y < 0` denotes a true anomaly.
Place additional compatible datasets under `deltric_minimal/data/`, and reload
the app. Without true anomalies, F1 is unavailable rather than zero.

Existing diagnostic cases are read from
`deltric_minimal/studies/results/reassignment_diagnostics/cases/`. New cases
are stored here and take precedence over that original cache. A changed dataset
or pipeline fingerprint is marked in the UI. Reassignment is always replayed
against the current production gate and checked against the saved final labels.
A case that no longer replays must be regenerated.

## Files and persistence

```text
deltric_lab/
  run.sh               Local launcher
  server.py            HTTP endpoints, saved records, local background jobs
  engine.py            Production queue observation and evidence calculations
  test_engine.py       Trace and cost correctness tests
  static/
    index.html         Workspace interface
    style.css          Light theme and responsive layout
    app.js             Canvas navigation, replay, inspectors, saved state
  data/                Generated local work; ignored by Git
    cases/             Recorded pipeline cases (.npz + .json)
    experiments/       Immutable results and settings (.npz + .json)
    snapshots/         Frozen baseline graph/data used by experiments
    notes/             Editable observations and evidence links (.json)
    workspaces/        Camera, selection, frame, layers, draft, active run
    logs/              Dataset-run output
    exports/           Run evidence bundles
```

The Python engine observes the production gate with `sys.settrace`, including
the exact ownership and budget branches. It does not replace the gate with a
second implementation. Only one local compute job runs at a time; the browser
remains interactive. The server binds to loopback and accepts same-origin local
requests. The interface makes no external requests.

## Interpretation and current limits

- ARI and anomaly scores in the cards are computed at the exact visible frame.
  Quality-chart lines use the recorded sampled timeline. Source-cluster and
  cost-anatomy summaries describe the complete active run.
- “Normal claimed” does not mean correct cluster assignment. Ground truth is
  used only for scoring/explanation; independent truth and cluster IDs can have
  different colors.
- Inspector truth badges are neutral. The current-frame outcome is green for
  a true anomaly left unassigned, red for an assigned true anomaly, and amber
  for an unassigned normal. An assigned normal still needs its cluster checked.
- Claim paths follow each panel's view: current claims for replay, complete
  active/recorded traces for final comparisons, and none before reassignment or
  in ground truth. Arrows point from the winning parent to the claimed point.
- Graph edges represent topology. Cost calculations use standardized original
  features, not distances on the displayed projection. The frontier layer shows
  unassigned neighbors reachable in one further hop within the remaining budget;
  it is not a sorted live heap display.
- Hypothetical independent routes ignore competing ownership. They do not prove
  that a candidate should win. The old route comparator is only exposed with
  density floor off; actual queue evidence works with both floor settings.
- Upstream stages show reconstructed memberships, not every internal pruning or
  growth decision. The detailed queue evidence concerns reassignment.
- Experiments offer the existing formula and its density-floor variant. Entirely
  new reassignment algorithms require adding engine support. Perun execution is
  intentionally deferred; the case/experiment format can later support imported
  remote runs.

Run evidence tests:

```bash
deltric_minimal/.venv/bin/python deltric_lab/test_engine.py
```

Checks cover production parity, competing ownership, over-budget attempts,
temporal visibility, cumulative hop arithmetic, zero reach, density floor,
and fixed sources.

Browser regressions (with the local server running) also verify actual canvas
edge coordinates, multi-hop selections, temporal outcomes, filters, projections,
and separate recorded/active traces:

```bash
playwright-cli open http://127.0.0.1:8765
playwright-cli run-code "$(cat deltric_lab/test_browser.js)"
```

The browser check intercepts workspace writes to preserve saved investigations.
