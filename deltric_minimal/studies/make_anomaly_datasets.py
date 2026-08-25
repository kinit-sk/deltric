#!/usr/bin/env python3
"""Generate anomaly-labelled datasets from the existing clean ones.

Only 16 of the datasets in ``data/`` carry ground-truth noise labels
(``blobs_with_noise__*`` and ``blobs_meeples__*``), which makes anomaly
detection impossible to score anywhere else. Rather than relabelling old
datasets after the fact -- which would only encode some detector's opinion
about which points are noise -- this script keeps every original point and
its label untouched and *appends* anomalies whose ground truth is known by
construction.

Each generated dataset mixes several anomaly kinds, recorded per point in
``anomaly_kind`` so a single run can be scored per kind:

  uniform   scattered background noise over the inflated bounding box
  far       global outliers pushed well outside the data hull
  bridge    interstitial points in the gaps between two clusters (hard)
  micro     small tight clumps of anomalies in empty regions (hard: a
            detector may call them a cluster instead of noise)
  subspace  an inlier displaced far along a random subset of dimensions,
            in-distribution everywhere else (d >= 5 only)

Every candidate anomaly is rejection-sampled: it is kept only if its
distance to the nearest inlier exceeds ``margin``, the ``margin_pct``
percentile of the inliers' own k-th nearest-neighbour distance. That is
what makes the ``-1`` label defensible -- no "anomaly" is placed inside
the region a real cluster occupies.

Output ``<stem>__anom<pct>.npz`` holds:
  X             (n_inlier + n_anom, d) float32, anomalies appended last
  y             int64, original labels for inliers, -1 for anomalies
  t             carried over from the base dataset if present, -1 for anomalies
  anomaly_kind  int8, 0 = inlier, 1..5 = the kinds above
plus a sidecar ``.json`` with the generation metadata and validation stats.

Usage:
    python studies/make_anomaly_datasets.py                 # default suite
    python studies/make_anomaly_datasets.py --rate 0.05 0.15
    python studies/make_anomaly_datasets.py --stems blobs_separated__d2__n500
    python studies/make_anomaly_datasets.py --kinds uniform bridge --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

_DELTRIC_MINIMAL_DIR = Path(__file__).resolve().parents[1]
_DEFAULT_DATA_DIR = _DELTRIC_MINIMAL_DIR / "data"

KIND_CODES = {"inlier": 0, "uniform": 1, "far": 2, "bridge": 3, "micro": 4, "subspace": 5}
ALL_KINDS = ["uniform", "far", "bridge", "micro", "subspace"]

# Clean, cluster-structured families. Excluded on purpose:
#   blobs_with_noise / blobs_meeples  already carry -1 labels
#   uniform_hypercube                 no structure, so "anomaly" is undefined
#   digits / *_h2mg_*                 real data, injected noise would be trivial
DEFAULT_FAMILIES = [
    "blobs_separated", "blobs_anisotropic", "blobs_varied_density",
    "blobs_hierarchical", "blobs_two", "blobs_hearts",
    "blobs_informative_subspace", "moons", "nested_spheres",
    "clusters_on_swiss_roll",
]
DEFAULT_SINGLETONS = [
    "01_shapes_2d_compound_easy_3600_k6", "02_shapes_2d_compound_harder_4800_k6",
    "04_v3_vardensity_blobs_20d_4500_k6", "10_v3_close_scurves_50d_5000_k6",
    "11_v2_moons_2d_2500", "14_v2_blobs_8d_3500_k7",
    "varied_2d_5c", "blobs_large_2d_20c",
]
# k == 1 manifolds: off-manifold anomalies are meaningful but there are no
# clusters, so "bridge" degenerates into a chord. Opt in with --manifolds.
MANIFOLD_FAMILIES = [
    "swiss_roll", "swiss_roll_hole", "s_curve", "helix", "torus", "severed_sphere",
]


def select_stems(data_dir: Path, families: list[str], singletons: list[str]) -> list[str]:
    stems = []
    for path in sorted(data_dir.glob("*.npz")):
        stem = path.stem
        if stem.endswith(tuple(f"anom{p}" for p in range(1, 100))) or "__anom" in stem:
            continue
        if stem in singletons or stem.split("__")[0] in families:
            stems.append(stem)
    return stems


def _accept(cand: np.ndarray, nn: NearestNeighbors, margin: float) -> np.ndarray:
    """Mask of candidates far enough from every inlier to be honestly labelled -1."""
    if len(cand) == 0:
        return np.zeros(0, dtype=bool)
    dist, _ = nn.kneighbors(cand, n_neighbors=1)
    return dist[:, 0] > margin


def _sample_validated(propose, n_want: int, nn: NearestNeighbors, margin: float,
                      rng: np.random.Generator, kind: str) -> tuple[np.ndarray, float]:
    """Rejection-sample until ``n_want`` candidates clear ``margin``.

    If a kind cannot fill its quota (tightly packed clusters leave little
    valid room for bridges, say), the margin is relaxed in steps rather than
    silently emitting fewer points -- the relaxation is reported and stored.
    """
    kept: list[np.ndarray] = []
    n_kept = 0
    used_margin = margin
    for round_idx in range(12):
        if round_idx and round_idx % 4 == 0:  # relax only after honest attempts
            used_margin *= 0.8
        for _ in range(4):
            batch = np.asarray(propose(max(64, 4 * (n_want - n_kept))), dtype=np.float64)
            ok = batch[_accept(batch, nn, used_margin)]
            if len(ok):
                kept.append(ok)
                n_kept += len(ok)
            if n_kept >= n_want:
                out = np.vstack(kept)[:n_want]
                return out, used_margin
    # Some kinds have no valid room in some geometries -- e.g. on nested
    # spheres the midpoint between the inner and outer shell lands on the
    # middle shell. Return what was found (possibly nothing) and let the
    # caller top up rather than emitting points that are not really anomalies.
    out = np.vstack(kept) if kept else np.zeros((0, 0))
    print(f"      ! {kind}: only {len(out)}/{n_want} valid points"
          + (f" (margin relaxed to {used_margin / margin:.2f}x)" if used_margin < margin else ""))
    return out, used_margin


def make_anomalies(X: np.ndarray, y: np.ndarray, n_anom: int, kinds: list[str],
                   rng: np.random.Generator, knn_k: int, margin_pct: float,
                   ) -> tuple[np.ndarray, np.ndarray, dict]:
    n, d = X.shape
    center = X.mean(axis=0)
    lo, hi = X.min(axis=0), X.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    dim_std = np.maximum(X.std(axis=0), 1e-9)
    scale = float(np.mean(dim_std))
    radii = np.linalg.norm(X - center, axis=1)
    r_max = float(radii.max())

    nn = NearestNeighbors(n_neighbors=min(knn_k + 1, n)).fit(X)
    knn_dist, _ = nn.kneighbors(X)
    knn_d = knn_dist[:, -1]                      # k-th NN distance of each inlier
    margin = float(np.percentile(knn_d, margin_pct))

    labels = np.unique(y)
    by_label = {int(lab): np.flatnonzero(y == lab) for lab in labels}
    multi_cluster = len(labels) >= 2

    kinds = [k for k in kinds if not (k == "subspace" and d < 5)]

    def prop_uniform(m):
        blo, bhi = center - 1.15 * (center - lo), center + 1.15 * (hi - center)
        return rng.uniform(blo, bhi, size=(m, d))

    def prop_far(m):
        u = rng.normal(size=(m, d))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        return center + u * (r_max * rng.uniform(1.2, 2.0, size=(m, 1)))

    def prop_bridge(m):
        # Interpolate between points of two *different* clusters, so the
        # sample lands in the gap the clusters leave between them.
        if multi_cluster:
            pair = [rng.choice(labels, size=2, replace=False) for _ in range(m)]
            ia = np.array([rng.choice(by_label[int(p[0])]) for p in pair])
            ib = np.array([rng.choice(by_label[int(p[1])]) for p in pair])
        else:
            ia = rng.integers(0, n, size=m)
            ib = rng.integers(0, n, size=m)
        w = rng.uniform(0.35, 0.65, size=(m, 1))
        return (1 - w) * X[ia] + w * X[ib] + rng.normal(scale=0.02 * scale, size=(m, d))

    def prop_micro_center(m):
        return prop_uniform(m)

    def prop_subspace(m):
        base = X[rng.integers(0, n, size=m)].copy()
        n_shift = max(1, int(np.ceil(0.2 * d)))
        for row in base:
            dims = rng.choice(d, size=n_shift, replace=False)
            sign = rng.choice([-1.0, 1.0], size=n_shift)
            row[dims] += sign * rng.uniform(3.0, 6.0, size=n_shift) * dim_std[dims]
        return base

    quota = {k: n_anom // len(kinds) for k in kinds}
    for k in kinds[: n_anom - sum(quota.values())]:
        quota[k] += 1

    chunks: list[np.ndarray] = []
    codes: list[np.ndarray] = []
    relaxed: dict[str, float] = {}
    dropped: list[str] = []
    for kind in kinds:
        want = quota[kind]
        if want <= 0:
            continue
        if kind == "micro":
            # A handful of tight clumps rather than scattered singletons.
            n_clumps = max(2, want // 25)
            centers, used = _sample_validated(prop_micro_center, n_clumps, nn, margin, rng, kind)
            sizes = np.full(len(centers), want // len(centers))
            sizes[: want - int(sizes.sum())] += 1
            pts = []
            for c, size in zip(centers, sizes):
                if size <= 0:
                    continue
                clump = c + rng.normal(scale=0.05 * scale, size=(int(size), d))
                pts.append(clump[_accept(clump, nn, 0.5 * used)])
            got = np.vstack(pts) if pts else np.zeros((0, d))
        else:
            propose = {"uniform": prop_uniform, "far": prop_far,
                       "bridge": prop_bridge, "subspace": prop_subspace}[kind]
            got, used = _sample_validated(propose, want, nn, margin, rng, kind)
        if used < margin and len(got):
            relaxed[kind] = round(used / margin, 3)
        if len(got) == 0:
            dropped.append(kind)
            continue
        chunks.append(got.reshape(-1, d))
        codes.append(np.full(len(got), KIND_CODES[kind], dtype=np.int8))

    # Top up any shortfall with uniform background noise, which is the kind
    # that can always be placed, so the contamination rate stays as requested.
    short = n_anom - sum(len(c) for c in chunks)
    if short > 0:
        extra, used = _sample_validated(prop_uniform, short, nn, margin, rng, "uniform")
        chunks.append(extra.reshape(-1, d))
        codes.append(np.full(len(extra), KIND_CODES["uniform"], dtype=np.int8))

    A = np.vstack(chunks)
    kind_codes = np.concatenate(codes)
    order = rng.permutation(len(A))          # so kinds are not contiguous in the file
    A, kind_codes = A[order], kind_codes[order]

    a_dist, _ = nn.kneighbors(A, n_neighbors=1)
    stats = {
        "margin": round(margin, 6),
        "inlier_knn_median": round(float(np.median(knn_d)), 6),
        "anom_nn_dist_min": round(float(a_dist.min()), 6),
        "anom_nn_dist_median": round(float(np.median(a_dist)), 6),
        "margin_relaxed": relaxed,
        "kinds_dropped": dropped,
        "counts": {k: int((kind_codes == KIND_CODES[k]).sum()) for k in kinds},
    }
    return A, kind_codes, stats


def build(path: Path, rate: float, kinds: list[str], seed: int, knn_k: int,
          margin_pct: float, out_dir: Path, dry_run: bool) -> dict | None:
    npz = np.load(path, allow_pickle=False)
    X = np.asarray(npz["X"], dtype=np.float64)
    y = np.asarray(npz["y"]).astype(np.int64)
    if (y < 0).any():
        print(f"  skip {path.stem}: already has {int((y < 0).sum())} noise labels")
        return None

    n = len(X)
    n_anom = int(round(rate * n / (1.0 - rate)))   # final contamination == rate
    rng = np.random.default_rng(abs(hash(path.stem)) % (2**31) + seed)
    A, kind_codes, stats = make_anomalies(X, y, n_anom, kinds, rng, knn_k, margin_pct)

    X_out = np.vstack([X, A]).astype(np.float32)
    y_out = np.concatenate([y, np.full(len(A), -1, dtype=np.int64)])
    kind_out = np.concatenate([np.zeros(n, dtype=np.int8), kind_codes])
    arrays = {"X": X_out, "y": y_out, "anomaly_kind": kind_out}
    if "t" in npz.files:
        t = np.asarray(npz["t"], dtype=np.float32)
        arrays["t"] = np.concatenate([t, np.full(len(A), -1.0, dtype=np.float32)])

    stem = f"{path.stem}__anom{int(round(rate * 100)):02d}"
    meta = {
        "base": path.stem, "source_n": n, "d": int(X.shape[1]),
        "n_anomaly": int(len(A)), "contamination": round(len(A) / len(X_out), 4),
        "kinds": stats["counts"], "kind_codes": KIND_CODES,
        "seed": seed, "knn_k": knn_k, "margin_pct": margin_pct,
        "validation": {k: stats[k] for k in
                       ("margin", "inlier_knn_median", "anom_nn_dist_min",
                        "anom_nn_dist_median", "margin_relaxed", "kinds_dropped")},
    }
    print(f"  {stem:<48} n={len(X_out):<5} anom={len(A):<4} "
          f"({meta['contamination']:.0%})  min_nn_dist={stats['anom_nn_dist_min']:.3f} "
          f"> margin={stats['margin']:.3f}"
          + ("  [relaxed]" if stats["margin_relaxed"] else "")
          + (f"  [no {'/'.join(stats['kinds_dropped'])}]" if stats["kinds_dropped"] else ""))
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / f"{stem}.npz", **arrays)
        (out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=_DEFAULT_DATA_DIR, type=Path)
    parser.add_argument("--out", default=None, type=Path, help="default: same as --data")
    parser.add_argument("--stems", nargs="*", default=None, help="explicit base stems")
    parser.add_argument("--rate", nargs="*", type=float, default=[0.10],
                        help="target contamination fraction(s)")
    parser.add_argument("--kinds", nargs="*", default=ALL_KINDS, choices=ALL_KINDS)
    parser.add_argument("--manifolds", action="store_true",
                        help="also generate for the k=1 manifold families")
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--margin-pct", type=float, default=95.0,
                        help="percentile of inlier k-NN distance used as the "
                             "minimum anomaly-to-inlier distance")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_dir = args.out or args.data
    families = DEFAULT_FAMILIES + (MANIFOLD_FAMILIES if args.manifolds else [])
    stems = args.stems or select_stems(args.data, families, DEFAULT_SINGLETONS)
    if not stems:
        raise SystemExit("no base datasets selected")

    print(f"{len(stems)} base datasets x {len(args.rate)} rate(s), kinds={args.kinds}"
          + ("  [DRY RUN]" if args.dry_run else ""))
    metas = []
    for rate in args.rate:
        print(f"\n--- contamination {rate:.0%} ---")
        for stem in stems:
            path = args.data / f"{stem}.npz"
            if not path.exists():
                raise SystemExit(f"missing base dataset: {path}")
            meta = build(path, rate, args.kinds, args.seed, args.knn_k,
                         args.margin_pct, out_dir, args.dry_run)
            if meta:
                metas.append(meta)

    print(f"\n{len(metas)} datasets, {sum(m['n_anomaly'] for m in metas)} anomalies total")
    n_relaxed = sum(1 for m in metas if m["validation"]["margin_relaxed"])
    if n_relaxed:
        print(f"{n_relaxed} needed a relaxed margin (tightly packed base clusters)")
    if not args.dry_run:
        print(f"Wrote to {out_dir}")


if __name__ == "__main__":
    main()
