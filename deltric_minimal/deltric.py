"""Installable DelTriC estimator using the tested current-stage pipeline."""

from __future__ import annotations

from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile

import numpy as np


class DelTriC:
    """Current-stage DelTriC; ``-1`` labels denote remaining outliers."""

    def __init__(
        self, *, seed_hard_limit=0.75, seed_hard_limit_2d=0.1,
        hard_growth_limit=1.6, hard_growth_limit_2d=1.5,
        projected_hard_growth_limit=2.8, projected_hard_growth_limit_2d=1.5,
        outlier_cost_gate_reach=15.0, outlier_cost_gate_reach_2d=15.0,
        umap_n_neighbors=15, umap_n_epochs=100,
    ):
        self.seed_hard_limit = seed_hard_limit
        self.seed_hard_limit_2d = seed_hard_limit_2d
        self.hard_growth_limit = hard_growth_limit
        self.hard_growth_limit_2d = hard_growth_limit_2d
        self.projected_hard_growth_limit = projected_hard_growth_limit
        self.projected_hard_growth_limit_2d = projected_hard_growth_limit_2d
        self.outlier_cost_gate_reach = outlier_cost_gate_reach
        self.outlier_cost_gate_reach_2d = outlier_cost_gate_reach_2d
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
                "--seed-baseline", "first_mode", "--hard-growth-limit", str(self.hard_growth_limit),
                "--hard-growth-limit-2d", str(self.hard_growth_limit_2d),
                "--projected-hard-growth-limit", str(self.projected_hard_growth_limit),
                "--projected-hard-growth-limit-2d", str(self.projected_hard_growth_limit_2d),
                "--initial-relation", "union", "--component-growth-knn", "50",
                "--component-growth-min-edges", "10", "--growth-seed-min-edges", "10",
                "--hard-gate-mode", "strict", "--projected-hard-limit", "--no-multiplicative-hard-limit",
                "--gomory-pruning-mode", "points", "--gomory-point-cut-size", "4",
                "--gomory-hu-m2-max", "1.0", "--gomory-hu-shape-score-min", "12",
                "--gomory-hu-min-triangle-edge-ratio", "0.3", "--post-growth-boundary-classification",
                "--gomory-hu-boundary-edges-only", "--boundary-edge-expansion-hops", "2",
                "--restore-intra-component-edges", "--restore-hard-limit", "2.0",
                "--interior-real-restoration", "--growth-restoration-mode", "real",
                "--outlier-growth", "--outlier-reassignment", "cost_gate",
                "--outlier-cost-gate-penalty-power", "0.5", "--outlier-cost-gate-density-power", "0.5",
                "--outlier-cost-gate-density-k", "15", "--no-outlier-cost-gate-density-clip",
                "--outlier-cost-gate-reach", str(self.outlier_cost_gate_reach),
                "--outlier-cost-gate-reach-2d", str(self.outlier_cost_gate_reach_2d),
            ]
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
