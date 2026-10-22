#!/usr/bin/env python3
"""Differential stage checks against an untouched source directory.

Run with the candidate environment and --baseline /path/to/before. Exercises
random retained graphs, missing labels, singleton components, empty selections,
restoration modes, floating-point reductions, and cut choices independently of
projection. Full-dataset verification lives in verify_quadratic_fix.py.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import Delaunay

from verify_quadratic_fix import fingerprint


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def equal(a, b, context):
    assert fingerprint(a) == fingerprint(b), context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import plot_seed_component_recomputed_growth as new
    import utils_component_growth as new_utils
    old = load("baseline_pipeline", args.baseline / "plot_seed_component_recomputed_growth.py")
    old_utils = load("baseline_growth", args.baseline / "utils_component_growth.py")
    rng = np.random.default_rng(91273)
    checks = 0
    # Equal-cost claims must retain the original heap tie-breaking by label.
    tie_edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    tie_labels = np.array([0, 0, -1, 7, 7])
    params = (tie_labels, tie_edges, np.ones(4), 5, 0.5, 15.0)
    before = old_utils._assign_anomalies_cost_gate(*params)
    after = new_utils._assign_anomalies_cost_gate(*params)
    equal(before, after, "equal-cost claim tie")
    assert after[0][2] == 0
    checks += 1
    # Either bridge endpoint separates the same side sizes. Keep the smaller
    # vertex ID, even when edges arrive in a non-sorted insertion order.
    tie_edges = np.array([[3, 5], [0, 2], [2, 3], [4, 5], [1, 2], [3, 4], [0, 1]])
    kwargs = dict(max_cut_points=4, min_component_points=2, min_partition_fraction=0.2)
    before = old._candidate_vertex_cut_mask(tie_edges, np.ones(7, dtype=bool), **kwargs)
    after = new._candidate_vertex_cut_mask(tie_edges, np.ones(7, dtype=bool), **kwargs)
    equal(before, after, "balanced-cut tie")
    assert after[1]["largest_components"][0]["accepted_cut_points"] == [2]
    checks += 1
    for labels in (np.array([], dtype=np.int64), np.array([-1, 5, 0, 5, -1, 21, 0])):
        for indices in (None, np.arange(len(labels), dtype=np.int64)[::2]):
            selected = np.arange(len(labels)) if indices is None else indices
            groups = new_utils._group_indices_by_label(labels, indices)
            equal(groups, {int(c): selected[labels[selected] == c] for c in np.unique(labels[selected])}, "groups")
            checks += 1
    for trial in range(24):
        n = 30 + trial * 4
        X = rng.normal(size=(n, 2))
        triangles = Delaunay(X).simplices
        edges = np.unique(np.sort(np.concatenate((triangles[:, [0, 1]], triangles[:, [0, 2]],
                                                 triangles[:, [1, 2]])), axis=1), axis=0)
        lengths = np.linalg.norm(X[edges[:, 0]] - X[edges[:, 1]], axis=1)
        projected = lengths * rng.uniform(0.5, 2, len(edges))
        retained = rng.random(len(edges)) < (0.0 if trial == 0 else (1.0 if trial == 1 else 0.4))
        labels, bearing = old._final_graph_components(edges, retained, n)
        counts = np.bincount(labels[edges[retained, 0]], minlength=int(labels.max()) + 1)
        for minimum in (0, 3):
            kwargs = dict(minimum_edges=minimum, edge_lengths=lengths, X_proj=X)
            before = old._component_topology_metrics(edges, retained, n, **kwargs)
            after = new._component_topology_metrics(edges, retained, n, **kwargs)
            equal(before, after, f"topology {trial}")
            for original in (None, np.column_stack((X, X[:, 0] ** 2))):
                a, b = copy.deepcopy(before), copy.deepcopy(after)
                old._add_component_area_perimeter_metrics(a, edges, retained, triangles, X, original)
                new._add_component_area_perimeter_metrics(b, edges, retained, triangles, X, original)
                equal(a, b, f"area {trial}")
                checks += 1
            checks += 1
        for bound in (None, 0.0, 2.0):
            params = (edges, lengths, retained, labels, bearing, bound)
            equal(old._bounded_intra_component_restore_mask(*params),
                  new._bounded_intra_component_restore_mask(*params), f"restore {trial}")
            checks += 1
        for mode in ("seed_only", "virtual", "real"):
            params = (edges, lengths, projected, retained, labels, counts, 2,
                      np.ones(n), 1.6, 2.8)
            kwargs = dict(restore_hard_limit=None if trial % 3 == 0 else 2.0, restoration_mode=mode)
            equal(old._sequential_growth(*params, **kwargs), new._sequential_growth(*params, **kwargs),
                  f"growth {trial} {mode}")
            checks += 1
        for eligible in (np.ones(n, dtype=bool), rng.random(n) > 0.1):
            kwargs = dict(max_cut_points=4, min_component_points=2,
                          min_partition_fraction=0.2, component_eligible_points=eligible)
            equal(old._candidate_vertex_cut_mask(edges, retained, **kwargs),
                  new._candidate_vertex_cut_mask(edges, retained, **kwargs), f"cut {trial}")
            checks += 1
        for density in (0.0, 0.5):
            source_labels = labels * 3  # Non-contiguous component IDs.
            source_labels[rng.random(n) < 0.4] = -1
            if trial == 0:
                source_labels[:] = -1
            if trial == 1:
                source_labels[:] = 0
            params = (source_labels, edges, lengths, n, 0.5, 15.0)
            kwargs = dict(core_distance=rng.uniform(0.2, 2.0, n), density_power=density,
                          density_clip=bool(trial % 2))
            equal(old_utils._assign_anomalies_cost_gate(*params, **kwargs),
                  new_utils._assign_anomalies_cost_gate(*params, **kwargs), f"cost {trial}")
            checks += 1
    print(f"PASS: {checks} exact differential checks")


if __name__ == "__main__":
    main()
