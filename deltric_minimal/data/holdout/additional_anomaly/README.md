# Reserved additional anomaly benchmarks

Holdout: Stamps, PageBlocks, Vowels, Letter, ALOI.

This split was fixed before running the five other new datasets. Only download integrity, feature/label validity and published construction metadata were inspected. No DelTriC or HDBSCAN fitting, projection, diagnostic plotting, scoring or parameter selection is performed on these datasets.

Freeze the algorithm, preprocessing and parameters before a separate final evaluation. Keep this folder out of development glob patterns. `split.json` records both groups with paths relative to `deltric_minimal/`; per-dataset JSON files and untouched `source/` downloads preserve provenance.

See `data/testing/additional_anomaly/README.md` for evaluation conventions and limitations.
