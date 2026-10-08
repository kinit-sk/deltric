"""Installable DelTriC estimator using the tested current-stage pipeline."""

from __future__ import annotations

from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile

import numpy as np


class DelTriC:
    """Current-stage DelTriC; ``-1`` labels denote remaining outliers.

    The default values reproduce the validated current-stage configuration.
    Native 2-D data uses only original-space geometry; its legacy projected
    growth argument is retained for compatibility but deliberately ignored.
    """

    def __init__(
        self, *,
        seed_hard_limit=0.75,
        seed_hard_limit_2d=0.1,
        seed_baseline="first_mode",
        first_mode_fallback=True,
        hard_growth_limit=1.6,
        hard_growth_limit_2d=1.5,
        projected_hard_growth_limit=2.8,
        projected_hard_growth_limit_2d=1.5,
        min_component_edges=10,
        growth_seed_min_edges=10,
        restore_hard_limit=2.0,
        gomory_point_cut_size=4,
        gomory_shape_score_min=12.0,
        gomory_min_partition_fraction=0.2,
        boundary_edge_expansion_hops=2,
        outlier_cost_gate_reach=15.0,
        outlier_cost_gate_reach_2d=15.0,
        outlier_cost_gate_penalty_power=0.5,
        outlier_cost_gate_density_power=0.5,
        outlier_cost_gate_density_k=15,
        umap_n_neighbors=15,
        umap_n_epochs=100,
    ):
        self.seed_hard_limit = seed_hard_limit
        self.seed_hard_limit_2d = seed_hard_limit_2d
        if seed_baseline not in {"first_mode", "median"}:
            raise ValueError("seed_baseline must be 'first_mode' or 'median'")
        self.seed_baseline = seed_baseline
        self.first_mode_fallback = first_mode_fallback
        self.hard_growth_limit = hard_growth_limit
        self.hard_growth_limit_2d = hard_growth_limit_2d
        self.projected_hard_growth_limit = projected_hard_growth_limit
        self.projected_hard_growth_limit_2d = projected_hard_growth_limit_2d
        self.min_component_edges = min_component_edges
        self.growth_seed_min_edges = growth_seed_min_edges
        self.restore_hard_limit = restore_hard_limit
        self.gomory_point_cut_size = gomory_point_cut_size
        self.gomory_shape_score_min = gomory_shape_score_min
        self.gomory_min_partition_fraction = gomory_min_partition_fraction
        self.boundary_edge_expansion_hops = boundary_edge_expansion_hops
        self.outlier_cost_gate_reach = outlier_cost_gate_reach
        self.outlier_cost_gate_reach_2d = outlier_cost_gate_reach_2d
        self.outlier_cost_gate_penalty_power = outlier_cost_gate_penalty_power
        self.outlier_cost_gate_density_power = outlier_cost_gate_density_power
        self.outlier_cost_gate_density_k = outlier_cost_gate_density_k
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_n_epochs = umap_n_epochs

    def fit(self, X, y=None):
        self.labels_ = self.fit_predict(X)
        return self

    def fit_predict(self, X, y=None):
        from plot_seed_component_recomputed_growth import main

        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or len(X) < 3:
            raise ValueError("X must have shape (n_samples, n_features), n_samples >= 3")
        with tempfile.TemporaryDirectory(prefix="deltric_") as temp_dir:
            temp = Path(temp_dir)
            data_path, out_path = temp / "input.npz", temp / "result.png"
            np.savez_compressed(data_path, X=X)
            argv = [
                "deltric", "--data", str(data_path), "--out", str(out_path),
                "--metrics-only", "--quiet", "--dim-reduction", "umap", "--project-dim", "2",
                "--umap-n-epochs", str(self.umap_n_epochs), "--umap-n-neighbors", str(self.umap_n_neighbors),
                "--seed-hard-limit", str(self.seed_hard_limit), "--seed-hard-limit-2d", str(self.seed_hard_limit_2d),
                "--seed-baseline", self.seed_baseline, "--hard-growth-limit", str(self.hard_growth_limit),
                "--hard-growth-limit-2d", str(self.hard_growth_limit_2d),
                "--projected-hard-growth-limit", str(self.projected_hard_growth_limit),
                "--projected-hard-growth-limit-2d", str(self.projected_hard_growth_limit_2d),
                "--initial-relation", "union", "--component-growth-knn", "50",
                "--component-growth-min-edges", str(self.min_component_edges),
                "--growth-seed-min-edges", str(self.growth_seed_min_edges),
                "--hard-gate-mode", "strict", "--projected-hard-limit", "--no-multiplicative-hard-limit",
                # The current-stage runner deliberately leaves the legacy
                # redundancy edge-pruning experiment disabled.
                "--no-redundancy-pruning",
                "--gomory-pruning-mode", "points", "--gomory-point-cut-size", str(self.gomory_point_cut_size),
                "--gomory-hu-m2-max", "1.0",
                "--gomory-hu-min-triangle-edge-ratio", "0.3", "--post-growth-boundary-classification",
                "--gomory-hu-boundary-edges-only", "--boundary-edge-expansion-hops", str(self.boundary_edge_expansion_hops),
                "--restore-intra-component-edges",
                "--interior-real-restoration", "--growth-restoration-mode", "real",
                "--outlier-growth", "--outlier-reassignment", "cost_gate",
                "--outlier-cost-gate-penalty-power", str(self.outlier_cost_gate_penalty_power),
                "--outlier-cost-gate-density-power", str(self.outlier_cost_gate_density_power),
                "--outlier-cost-gate-density-k", str(self.outlier_cost_gate_density_k), "--no-outlier-cost-gate-density-clip",
                "--outlier-cost-gate-reach", str(self.outlier_cost_gate_reach),
                "--outlier-cost-gate-reach-2d", str(self.outlier_cost_gate_reach_2d),
            ]
            if not self.first_mode_fallback:
                argv.append("--first-mode-no-fallback")
            if self.restore_hard_limit is not None:
                argv.extend(["--restore-hard-limit", str(self.restore_hard_limit)])
            if self.gomory_shape_score_min is not None:
                argv.extend(["--gomory-hu-shape-score-min", str(self.gomory_shape_score_min)])
            previous_argv = sys.argv
            try:
                sys.argv = argv
                with redirect_stdout(sys.stderr):
                    main()
            finally:
                sys.argv = previous_argv
            with np.load(out_path.with_suffix(".npz"), allow_pickle=False) as result:
                return np.asarray(result["final_prediction"], dtype=np.int64)


def fit_predict(X, **kwargs):
    """Convenience equivalent to ``DelTriC(**kwargs).fit_predict(X)``."""
    return DelTriC(**kwargs).fit_predict(X)
