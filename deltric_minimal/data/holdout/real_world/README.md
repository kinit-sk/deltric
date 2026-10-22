# Real-world clustering + anomaly benchmarks: holdout split

Fixed on 2026-09-24 before running DelTriC. **Do not fit, plot or score these during tuning.** They are reserved for the final paper evaluation. See `data/testing/real_world/README.md` for the format, label encoding and anomaly-origin definitions.

| dataset | n | d | k | anomalies | origin |
|---|---|---|---|---|---|
| avila | 10430 | 10 | 9 | copyists B, W, C, 1.5% | natural_rare_class |
| yeast | 1484 | 8 | 7 | VAC, POX, ERL, 3.7% | natural_rare_class |
| htru2 | 17898 | 8 | 1 | pulsars, 9.2% | natural |
| sdss_colors | 12632 | 5 | 2 | QSO at 5% | constructed |
| mice_protein | 995 | 77 | 7 | t-SC-s at 5% | constructed |
| gaia_orion | 20000 | 5 | 13 | Gaia field stars, 93.2% | natural (high contamination) |

Notes:
- Mice Protein missing values are imputed with the class mean, which uses the labels; disclose this in the paper. There are ~15 replicate rows per mouse.
- SDSS excludes redshift on purpose.
- The Gaia Orion patch is a young, crowded star-forming region with overlapping groups. It is harder than the dev patch.

Rebuild: `uv run --with xlrd studies/make_real_world_datasets.py`.
