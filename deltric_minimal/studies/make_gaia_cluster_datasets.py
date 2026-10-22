#!/usr/bin/env python3
"""Build real open-cluster anomaly datasets from Gaia DR3 + Hunt & Reffert (2023).

Analogous to the ODDS/ADBench sets in ``data/``, but the "inlier cluster" is
a real Milky Way open cluster and the "noise" is the actual Gaia field
population around it -- not a synthetic construction. Ground truth comes
from an independent source (Hunt & Reffert's own HDBSCAN-based membership
pipeline), so scoring a detector against it is a fair comparison rather than
circular.

Membership ground truth: Hunt & Reffert, 2023, A&A 673, A114
  "Improving the open cluster census. II. An all-sky cluster catalogue with
  Gaia DR3" (arXiv:2303.13424), VizieR catalogue J/A+A/673/A114.
  https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/673/A114

For each requested cluster:
  1. Look up its sky position and half-light radius (``r50``) in the
     catalogue's ``clusters`` table.
  2. Pull that cluster's member stars from the ``members`` table (Gaia DR3
     source_id, membership probability, astrometry, photometry). Members
     with ``Prob >= prob_threshold`` are the inlier class.
  3. Cone-search the live Gaia DR3 archive (``gaiadr3.gaia_source`` via TAP)
     around the cluster centre out to ``radius_mult * r50``, with basic
     astrometric-quality cuts (``parallax_over_error > 3``, ``ruwe < 1.4``)
     and a magnitude cut, to get the surrounding field population.
  4. Union the two star sets by ``source_id``; anything that isn't a
     high-probability member is field/noise. The field pool is randomly
     subsampled (seeded) down to hit a target noise rate (``noise_rate``,
     default 10%, matching the ODDS/ADBench convention of anomalies as a
     minority class) instead of keeping the raw cone, which outnumbers the
     cluster by 50-200x even at these small radii.

Output ``gaia_<slug>.npz`` holds (following this repo's ``data/`` convention):
  X          (n, 7) float64   ra, dec, parallax, pmra, pmdec, phot_g_mean_mag, bp_rp
  y          (n,) int64       0 = cluster member, -1 = field/noise
  y_gaia     (n,) int64       identical to y (naming parity with y_odds/y_adbench)
  source_id  (n,) int64       Gaia DR3 source_id, for traceability
plus a sidecar ``gaia_<slug>.json`` with provenance and generation metadata.

Usage:
    uv run studies/make_gaia_cluster_datasets.py
    uv run studies/make_gaia_cluster_datasets.py --clusters Melotte_22 NGC_2632
    uv run studies/make_gaia_cluster_datasets.py --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

_DELTRIC_MINIMAL_DIR = Path(__file__).resolve().parents[1]
_DEFAULT_DATA_DIR = _DELTRIC_MINIMAL_DIR / "data"

CATALOG = "J/A+A/673/A114"
CATALOG_DOC = "https://arxiv.org/abs/2303.13424"
CATALOG_SOURCE = "https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/673/A114"

# A handful of well-known, richly-populated open clusters spanning a range of
# distance, angular size, and member count -- not a quality-flag filter, since
# ``isGMMMemb`` in the catalogue flags binary/grouped systems, not reliability.
DEFAULT_CLUSTERS = [
    "Melotte_22",  # Pleiades: nearby, large on sky, N~1700
    "NGC_2632",    # Praesepe / Beehive, N~1300
    "NGC_2516",    # denser and farther, N~3800
    "NGC_6705",    # M11 Wild Duck: distant, compact (r50 ~ 0.06 deg), N~1900
    "IC_2391",     # small and nearby, N~380
]

FEATURE_COLUMNS = ["ra", "dec", "parallax", "pmra", "pmdec", "phot_g_mean_mag", "bp_rp"]


def slugify(name: str) -> str:
    return name.lower().replace("-", "_")


def fetch_cluster_row(name: str):
    from astroquery.vizier import Vizier

    v = Vizier(row_limit=-1, columns=["**"])
    table = v.query_constraints(catalog=f"{CATALOG}/clusters", Name=name)
    if not table or len(table[0]) == 0:
        raise SystemExit(f"cluster {name!r} not found in {CATALOG}/clusters")
    return table[0][0]


def fetch_members(name: str, prob_threshold: float):
    from astroquery.vizier import Vizier

    v = Vizier(row_limit=-1, columns=["**"])
    table = v.query_constraints(catalog=f"{CATALOG}/members", Name=name)
    if not table:
        raise SystemExit(f"no members found for cluster {name!r}")
    members = table[0]
    prob = np.asarray(members["Prob"], dtype=np.float64)
    return members[prob >= prob_threshold]


def fetch_field_stars(ra: float, dec: float, radius_deg: float, mag_limit: float):
    from astroquery.gaia import Gaia

    query = f"""
    SELECT source_id, ra, dec, parallax, pmra, pmdec, phot_g_mean_mag, bp_rp
    FROM gaiadr3.gaia_source
    WHERE 1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {radius_deg}))
    AND phot_g_mean_mag < {mag_limit}
    AND parallax_over_error > 3 AND ruwe < 1.4
    AND parallax IS NOT NULL AND pmra IS NOT NULL AND pmdec IS NOT NULL AND bp_rp IS NOT NULL
    """
    job = Gaia.launch_job_async(query)
    return job.get_results()


def build_one(
    name: str,
    data_dir: Path,
    radius_mult: float,
    mag_limit: float,
    prob_threshold: float,
    noise_rate: float,
    seed: int,
    dry_run: bool,
) -> None:
    slug = slugify(name)
    cluster_row = fetch_cluster_row(name)
    ra, dec, r50 = (float(cluster_row[c]) for c in ("RA_ICRS", "DE_ICRS", "r50"))
    radius_deg = radius_mult * r50

    members = fetch_members(name, prob_threshold)
    n_members = len(members)
    if n_members == 0:
        raise SystemExit(f"{name}: no members with Prob >= {prob_threshold}")

    print(f"[{name}] centre=({ra:.3f}, {dec:.3f}) r50={r50:.3f}deg "
          f"radius={radius_deg:.3f}deg members(Prob>={prob_threshold})={n_members}")

    if dry_run:
        return

    field = fetch_field_stars(ra, dec, radius_deg, mag_limit)
    print(f"[{name}] field cone pulled: {len(field)} stars (pre-subsample)")

    member_ids = set(int(s) for s in members["GaiaDR3"])
    field_ids = np.asarray(field["source_id"], dtype=np.int64)
    is_field_noise = np.array([sid not in member_ids for sid in field_ids])
    noise_rows = field[is_field_noise]

    target_noise = int(round(n_members * noise_rate / (1.0 - noise_rate)))
    if len(noise_rows) > target_noise:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(noise_rows), size=target_noise, replace=False)
        keep.sort()
        noise_rows = noise_rows[keep]
    elif len(noise_rows) < target_noise:
        print(f"[{name}] warning: only {len(noise_rows)} field stars available, "
              f"below target {target_noise} for noise_rate={noise_rate}")
    n_noise = len(noise_rows)

    def feats_from_members(m):
        bp_rp = np.asarray(m["BPmag"], dtype=np.float64) - np.asarray(m["RPmag"], dtype=np.float64)
        return np.column_stack([
            np.asarray(m["RA_ICRS"], dtype=np.float64),
            np.asarray(m["DE_ICRS"], dtype=np.float64),
            np.asarray(m["Plx"], dtype=np.float64),
            np.asarray(m["pmRA"], dtype=np.float64),
            np.asarray(m["pmDE"], dtype=np.float64),
            np.asarray(m["Gmag"], dtype=np.float64),
            bp_rp,
        ])

    def feats_from_field(f):
        return np.column_stack([np.asarray(f[c], dtype=np.float64) for c in FEATURE_COLUMNS])

    X_member = feats_from_members(members)
    X_noise = feats_from_field(noise_rows)
    X = np.vstack([X_member, X_noise])
    y = np.concatenate([np.zeros(n_members, dtype=np.int64), -np.ones(n_noise, dtype=np.int64)])
    source_id = np.concatenate([
        np.asarray(members["GaiaDR3"], dtype=np.int64),
        np.asarray(noise_rows["source_id"], dtype=np.int64),
    ])

    finite = np.isfinite(X).all(axis=1)
    if not finite.all():
        print(f"[{name}] dropping {(~finite).sum()} rows with non-finite features")
        X, y, source_id = X[finite], y[finite], source_id[finite]

    out_npz = data_dir / f"gaia_{slug}.npz"
    np.savez_compressed(out_npz, X=X, y=y, y_gaia=y, source_id=source_id)

    sha256 = hashlib.sha256(out_npz.read_bytes()).hexdigest()
    meta = {
        "dataset": f"gaia_{slug}",
        "cluster_name": name,
        "source": CATALOG_SOURCE,
        "documentation": CATALOG_DOC,
        "field_source": "Gaia DR3 gaiadr3.gaia_source (TAP cone search)",
        "npz_sha256": sha256,
        "n": int(len(X)),
        "d": int(X.shape[1]),
        "n_anomaly": int(n_noise),
        "features": FEATURE_COLUMNS,
        "cluster_centre_icrs": [ra, dec],
        "r50_deg": r50,
        "cone_radius_deg": radius_deg,
        "mag_limit": mag_limit,
        "prob_threshold": prob_threshold,
        "target_noise_rate": noise_rate,
        "label_encoding": {
            "y": "-1 field/noise, 0 cluster member",
            "y_gaia": "-1 field/noise, 0 cluster member",
        },
        "preprocessing": (
            "No scaling or deduplication in stored data; evaluator standardizes "
            "features. Field pool cone-searched live from Gaia DR3, quality-cut "
            "(parallax_over_error > 3, ruwe < 1.4), then randomly subsampled "
            f"(seed={seed}) to a target noise rate of {noise_rate:.0%}."
        ),
    }
    (data_dir / f"gaia_{slug}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[{name}] wrote {out_npz.name}  n={len(X)} (members={n_members}, noise={n_noise})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clusters", nargs="+", default=DEFAULT_CLUSTERS)
    parser.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA_DIR)
    parser.add_argument("--radius-mult", type=float, default=1.5,
                         help="cone search radius as a multiple of the cluster's r50")
    parser.add_argument("--mag-limit", type=float, default=19.0,
                         help="Gaia phot_g_mean_mag cut for the field cone")
    parser.add_argument("--prob-threshold", type=float, default=0.5,
                         help="minimum Hunt & Reffert membership probability to count as inlier")
    parser.add_argument("--noise-rate", type=float, default=0.10,
                         help="target fraction of the final dataset that is field/noise")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true",
                         help="only print cluster geometry and member counts, no downloads/writes")
    args = parser.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    for name in args.clusters:
        build_one(
            name, args.data_dir, args.radius_mult, args.mag_limit,
            args.prob_threshold, args.noise_rate, args.seed, args.dry_run,
        )


if __name__ == "__main__":
    main()
