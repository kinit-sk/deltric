#!/usr/bin/env python3
"""Build laptop-sized real-world clustering + anomaly benchmarks.

Every dataset has cluster ground truth *and* anomaly ground truth where the
source allows it:

  y          (n,) int64   -1 anomaly, 0..k-1 cluster id
  source_y   (n,) object  original class label (string) from the source
  X          (n, d) float64, published features (no scaling; evaluator standardizes)

``anomaly_origin`` in each JSON says how anomalies were obtained:
  natural              -- the source itself labels them as anomalies/faults
  natural_rare_class   -- a genuinely rare class of the source, kept at its
                          natural frequency (no downsampling)
  constructed          -- a semantically distinct class downsampled by us
                          (ADBench-style); report separately in the paper

The dev/holdout split is fixed here before any DelTriC run (see SPLIT).
Holdout datasets must not be fit, plotted or scored during tuning.

Raw downloads are kept under ``<split>/real_world/source/`` with SHA-256.

Usage:
    uv run studies/make_real_world_datasets.py
    uv run studies/make_real_world_datasets.py --only dry_bean yeast
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import urllib.parse
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

import numpy as np

_DELTRIC_MINIMAL_DIR = Path(__file__).resolve().parents[1]
_DATA_DIR = _DELTRIC_MINIMAL_DIR / "data"

UCI = "https://archive.ics.uci.edu/static/public"
SMD = "https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/ServerMachineDataset"
SDSS_SQL = "https://skyserver.sdss.org/dr18/SkyServerWS/SearchTools/SqlSearch"

SPLIT = {
    "dry_bean": "testing",
    "hcv": "testing",
    "anuran_frogs": "testing",
    "smd_machine_1_1": "testing",
    "gaia_vela_puppis": "testing",
    "avila": "holdout",
    "yeast": "holdout",
    "htru2": "holdout",
    "sdss_colors": "holdout",
    "mice_protein": "holdout",
    "gaia_orion": "holdout",
}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fetch(url: str, dest: Path) -> bytes:
    """Download once into ``dest``; reuse the cached copy afterwards."""
    if dest.exists():
        return dest.read_bytes()
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "deltric-bench/1.0"})
    with urllib.request.urlopen(req, timeout=900) as r:
        data = r.read()
    dest.write_bytes(data)
    return data


def zip_member(blob: bytes, name: str) -> bytes:
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        return z.read(name)


def encode(labels: np.ndarray, anomaly_classes: set[str]) -> tuple[np.ndarray, list[str]]:
    """Map string labels to -1 (anomaly) / 0..k-1 (clusters, by descending size)."""
    labels = labels.astype(str)
    normal = [c for c in labels if c not in anomaly_classes]
    uniq, counts = np.unique(normal, return_counts=True)
    order = [str(u) for u in uniq[np.argsort(-counts, kind="stable")]]
    lut = {c: i for i, c in enumerate(order)}
    y = np.array([-1 if c in anomaly_classes else lut[c] for c in labels], dtype=np.int64)
    return y, order


def downsample_class(labels: np.ndarray, cls: set[str], rate: float, seed: int) -> np.ndarray:
    """Keep all non-``cls`` rows and enough ``cls`` rows for ``rate`` of the final n."""
    is_a = np.isin(labels.astype(str), list(cls))
    n_norm = int((~is_a).sum())
    target = int(round(n_norm * rate / (1.0 - rate)))
    idx_a = np.flatnonzero(is_a)
    if len(idx_a) > target:
        idx_a = np.sort(np.random.default_rng(seed).choice(idx_a, size=target, replace=False))
    return np.sort(np.concatenate([np.flatnonzero(~is_a), idx_a]))


# ----------------------------------------------------------------- builders
# Each returns (X, source_y, anomaly_classes, meta) with meta holding
# source/documentation/features/construction/anomaly_origin.


def build_dry_bean(src: Path):
    import pandas as pd
    from scipy.io import arff

    url = f"{UCI}/602/dry+bean+dataset.zip"
    blob = fetch(url, src / "dry_bean.zip")
    raw, _ = arff.loadarff(io.StringIO(zip_member(blob, "DryBeanDataset/Dry_Bean_Dataset.arff").decode()))
    df = pd.DataFrame(raw)
    lab = df.pop("Class").str.decode("utf-8").to_numpy()
    return df.to_numpy(np.float64), lab, {"BOMBAY"}, {
        "source": url,
        "documentation": "https://doi.org/10.24432/C50S4B",
        "citation": "Koklu & Ozkan (2020), Computers and Electronics in Agriculture 174:105507",
        "features": list(df.columns),
        "anomaly_origin": "natural_rare_class",
        "construction": "Six bean varieties are clusters; BOMBAY (522 grains, 3.8%, much larger seeds) is the anomaly class at natural frequency.",
    }, blob


def build_hcv(src: Path):
    import pandas as pd

    url = f"{UCI}/571/hcv+data.zip"
    blob = fetch(url, src / "hcv.zip")
    df = pd.read_csv(io.BytesIO(zip_member(blob, "hcvdat0.csv")), index_col=0)
    n_raw = len(df)
    df = df.dropna().reset_index(drop=True)
    lab = df.pop("Category").to_numpy()
    df["Sex"] = (df["Sex"] == "f").astype(np.float64)
    anomalies = {c for c in set(lab) if not c.startswith("0=")}
    return df.to_numpy(np.float64), lab, anomalies, {
        "source": url,
        "documentation": "https://doi.org/10.24432/C5D612",
        "citation": "Hoffmann et al. (2018), J. Lab. Precis. Med. 3:58",
        "features": list(df.columns),
        "anomaly_origin": "natural",
        "construction": (f"Blood donors are the single cluster; hepatitis, fibrosis, cirrhosis and 'suspect donor' "
                         f"patients are anomalies. Dropped {n_raw - len(df)} rows with missing lab values; Sex encoded f=1, m=0."),
    }, blob


def build_anuran_frogs(src: Path):
    import pandas as pd

    url = f"{UCI}/406/anuran+calls+mfccs.zip"
    blob = fetch(url, src / "anuran_frogs.zip")
    df = pd.read_csv(io.BytesIO(zip_member(blob, "Frogs_MFCCs.csv")))
    lab = df["Species"].to_numpy()
    feats = [c for c in df.columns if c.startswith("MFCCs_")]
    rare = {"ScinaxRuber", "OsteocephalusOophagus", "Rhinellagranulosa"}
    return df[feats].to_numpy(np.float64), lab, rare, {
        "source": url,
        "documentation": "https://doi.org/10.24432/C5CC9H",
        "citation": "Colonna et al. (2016), Anuran Calls (MFCCs), UCI",
        "features": feats,
        "anomaly_origin": "natural_rare_class",
        "construction": "Seven species with >=270 syllables are clusters; the three rarest species (330 syllables, 4.6%, three different genera) are anomalies at natural frequency. Several syllables per recording (RecordID not used).",
    }, blob


def build_smd_machine_1_1(src: Path):
    parts = {}
    blobs = b""
    for part in ("test", "test_label"):
        url = f"{SMD}/{part}/machine-1-1.txt"
        b = fetch(url, src / f"smd_machine-1-1_{part}.txt")
        blobs += b
        parts[part] = np.loadtxt(io.BytesIO(b), delimiter=",")
    X, lab = parts["test"][::2], parts["test_label"][::2].astype(int)
    lab = np.where(lab == 1, "anomaly", "normal")
    return X, lab, {"anomaly"}, {
        "source": f"{SMD}/test/machine-1-1.txt",
        "documentation": "https://github.com/NetManAIOps/OmniAnomaly",
        "citation": "Su et al. (2019), KDD, OmniAnomaly",
        "features": [f"metric_{i}" for i in range(X.shape[1])],
        "anomaly_origin": "natural",
        "construction": "Test half of server machine-1-1 (28479 x 38, labelled incidents), every 2nd timestep kept (n=14240) to stay under the laptop limit. Rows are treated as i.i.d. operating states; there is one 'normal' label, so cluster ARI is not meaningful.",
    }, blobs


def build_avila(src: Path):
    import pandas as pd

    url = f"{UCI}/459/avila.zip"
    blob = fetch(url, src / "avila.zip")
    df = pd.read_csv(io.BytesIO(zip_member(blob, "avila/avila-tr.txt")), header=None)
    lab = df.pop(10).to_numpy()
    rare = {"B", "W", "C"}
    return df.to_numpy(np.float64), lab, rare, {
        "source": url,
        "documentation": "https://doi.org/10.24432/C5K02X",
        "citation": "De Stefano et al. (2018), Eng. Appl. Artif. Intell. 72:99-110",
        "features": ["intercolumnar_distance", "upper_margin", "lower_margin", "exploitation", "row_number",
                     "modular_ratio", "interlinear_spacing", "weight", "peak_number", "modular_ratio_over_interlinear"],
        "anomaly_origin": "natural_rare_class",
        "construction": "Training file only (10430 rows). Nine copyists with >=266 patterns are clusters; the three rarest copyists B, W, C (152 rows, 1.5%) are anomalies at natural frequency. Hard: copyists overlap heavily.",
    }, blob


def build_yeast(src: Path):
    import pandas as pd

    url = f"{UCI}/110/yeast.zip"
    blob = fetch(url, src / "yeast.zip")
    df = pd.read_csv(io.BytesIO(zip_member(blob, "yeast.data")), sep=r"\s+", header=None)
    lab = df[9].to_numpy()
    feats = ["mcg", "gvh", "alm", "mit", "erl", "pox", "vac", "nuc"]
    rare = {"VAC", "POX", "ERL"}
    return df.iloc[:, 1:9].to_numpy(np.float64), lab, rare, {
        "source": url,
        "documentation": "https://doi.org/10.24432/C5KG68",
        "citation": "Nakai & Kanehisa (1992), Genomics 14(4):897-911",
        "features": feats,
        "anomaly_origin": "natural_rare_class",
        "construction": "Seven localisation sites with >=35 proteins are clusters; VAC, POX, ERL (55 proteins, 3.7%) are anomalies at natural frequency. Two features (erl, pox) are near-binary.",
    }, blob


def build_htru2(src: Path):
    url = f"{UCI}/372/htru2.zip"
    blob = fetch(url, src / "htru2.zip")
    arr = np.loadtxt(zip_member(blob, "HTRU_2.csv").decode().splitlines(), delimiter=",")
    lab = np.where(arr[:, 8] == 1, "pulsar", "rfi_noise")
    return arr[:, :8], lab, {"pulsar"}, {
        "source": url,
        "documentation": "https://doi.org/10.24432/C5DK6R",
        "citation": "Lyon et al. (2016), MNRAS 459(1):1104-1123",
        "features": ["ip_mean", "ip_std", "ip_kurtosis", "ip_skew", "dm_mean", "dm_std", "dm_kurtosis", "dm_skew"],
        "anomaly_origin": "natural",
        "construction": "Real pulsars (1639, 9.2%) are anomalies among RFI/noise candidates. Binary labels only: one 'normal' label, so cluster ARI is not meaningful.",
    }, blob


SDSS_COLS = "s.specobjid, s.class, p.psfMag_u, p.psfMag_g, p.psfMag_r, p.psfMag_i, p.psfMag_z, p.modelMag_r"
SDSS_WHERE = ("s.zWarning = 0 AND p.clean = 1 AND p.psfMag_r BETWEEN 15 AND 19.5")


def build_sdss_colors(src: Path, seed: int = 0):
    import pandas as pd

    frames, blobs = [], b""
    for cls, top in (("STAR", 6000), ("GALAXY", 6000), ("QSO", 2000)):
        sql = (f"SELECT TOP {top} {SDSS_COLS} FROM SpecObj s JOIN PhotoObj p ON s.bestobjid = p.objid "
               f"WHERE s.class = '{cls}' AND {SDSS_WHERE} ORDER BY s.specobjid")
        url = SDSS_SQL + "?" + urllib.parse.urlencode({"cmd": sql, "format": "csv"})
        b = fetch(url, src / f"sdss_dr18_{cls.lower()}.csv")
        blobs += b
        frames.append(pd.read_csv(io.BytesIO(b), skiprows=1))
    df = pd.concat(frames, ignore_index=True)
    lab = df["class"].to_numpy()
    keep = downsample_class(lab, {"QSO"}, 0.05, seed)
    df, lab = df.iloc[keep].reset_index(drop=True), lab[keep]
    X = np.column_stack([
        df.psfMag_u - df.psfMag_g, df.psfMag_g - df.psfMag_r,
        df.psfMag_r - df.psfMag_i, df.psfMag_i - df.psfMag_z,
        df.psfMag_r - df.modelMag_r,
    ]).astype(np.float64)
    return X, lab, {"QSO"}, {
        "source": SDSS_SQL,
        "sql_where": SDSS_WHERE,
        "documentation": "https://www.sdss4.org/dr17/ ; SkyServer DR18 SqlSearch",
        "citation": "SDSS DR18, Almeida et al. (2023), ApJS 267:44",
        "features": ["u-g", "g-r", "r-i", "i-z", "psf_minus_model_r"],
        "anomaly_origin": "constructed",
        "construction": (f"Spectroscopic classes STAR and GALAXY (first 6000 each by specobjid, i.e. earliest SDSS-I/II plates; clean photometry, "
                         f"15<r_psf<19.5) are clusters; QSOs downsampled to 5% (seed={seed}) are anomalies. "
                         "Colours + concentration only; redshift deliberately excluded (it would make QSOs trivial)."),
    }, blobs


def build_mice_protein(src: Path, seed: int = 0):
    import pandas as pd

    url = f"{UCI}/342/mice+protein+expression.zip"
    blob = fetch(url, src / "mice_protein.zip")
    df = pd.read_excel(io.BytesIO(zip_member(blob, "Data_Cortex_Nuclear.xls")))
    lab = df["class"].to_numpy()
    feats = [c for c in df.columns if c.endswith("_N")]
    X = df[feats]
    X = X.fillna(X.groupby(lab).transform("mean")).to_numpy(np.float64)
    keep = downsample_class(lab, {"t-SC-s"}, 0.05, seed)
    return X[keep], lab[keep], {"t-SC-s"}, {
        "source": url,
        "documentation": "https://doi.org/10.24432/C50S3Z",
        "citation": "Higuera et al. (2015), PLoS ONE 10(6):e0129126",
        "features": feats,
        "anomaly_origin": "constructed",
        "construction": (f"Seven genotype/treatment/behaviour classes are clusters; trisomic, saline-injected, shock-context "
                         f"mice (t-SC-s) downsampled to 5% (seed={seed}) are anomalies. Missing protein values imputed with "
                         "the class mean (uses labels; note this in the paper). ~15 replicate measurements per mouse."),
    }, blob


HR23 = "J/A+A/673/A114"
GAIA_QUALITY = "parallax_over_error > 5 AND ruwe < 1.4 AND phot_g_mean_mag < 18"


def _gaia_patch(src: Path, tag: str, ra: tuple, dec: tuple, plx: tuple,
                max_n: int = 20000, min_members: int = 50, prob: float = 0.5, seed: int = 0):
    """All Gaia DR3 stars in an (ra, dec, parallax) box; H&R23 members are clusters, the rest is field.

    Unlike make_gaia_cluster_datasets.py (one cluster + subsampled cone), the field
    here keeps its natural 5D distribution, several clusters share the patch, and
    only astrometry is used (no magnitude/colour, which splits a cluster along its
    main sequence).
    """
    import pandas as pd
    from astropy.table import Table

    gaia_csv = src / f"gaia_{tag}_field.csv"
    if not gaia_csv.exists():
        from astroquery.gaia import Gaia

        # The padded POLYGON only lets the archive use its spatial index; the
        # BETWEEN cuts define the selection.
        poly = f"{ra[0]-1},{dec[0]-1},{ra[1]+1},{dec[0]-1},{ra[1]+1},{dec[1]+1},{ra[0]-1},{dec[1]+1}"
        q = (f"SELECT source_id, ra, dec, parallax, pmra, pmdec FROM gaiadr3.gaia_source "
             f"WHERE 1=CONTAINS(POINT('ICRS', ra, dec), POLYGON('ICRS', {poly})) AND ra BETWEEN {ra[0]} AND {ra[1]} AND dec BETWEEN {dec[0]} AND {dec[1]} "
             f"AND parallax BETWEEN {plx[0]} AND {plx[1]} AND {GAIA_QUALITY} "
             f"AND pmra IS NOT NULL AND pmdec IS NOT NULL")
        src.mkdir(parents=True, exist_ok=True)
        Gaia.launch_job_async(q).get_results().to_pandas().to_csv(gaia_csv, index=False)
    mem_csv = src / f"gaia_{tag}_hr23_members.csv"
    if not mem_csv.exists():
        from astroquery.vizier import Vizier

        v = Vizier(row_limit=-1, columns=["Name", "GaiaDR3", "Prob"])
        t = v.query_constraints(catalog=f"{HR23}/members", RA_ICRS=f"{ra[0]} .. {ra[1]}",
                                DE_ICRS=f"{dec[0]} .. {dec[1]}", Plx=f"{plx[0]} .. {plx[1]}")[0]
        Table(t).to_pandas().to_csv(mem_csv, index=False)
    raw = gaia_csv.read_bytes() + mem_csv.read_bytes()

    df = pd.read_csv(gaia_csv)
    mem = pd.read_csv(mem_csv).rename(columns={"GaiaDR3": "source_id"})
    mem = mem.sort_values("Prob", ascending=False).drop_duplicates("source_id")
    df = df.merge(mem, on="source_id", how="left")
    in_df = df.Name.notna()
    sizes = df.loc[in_df & (df.Prob >= prob), "Name"].value_counts()
    big = set(sizes[sizes >= min_members].index)
    # Stars of small clusters or with low membership probability are neither
    # clean cluster members nor clean field: drop them.
    member = in_df & df.Name.isin(big) & (df.Prob >= prob)
    df = df[member | ~in_df].reset_index(drop=True)
    member = df.Name.isin(big) & (df.Prob >= prob)
    n_dropped = int(in_df.sum() - member.sum())
    field_idx = np.flatnonzero(~member)
    n_field_total = len(field_idx)
    if len(df) > max_n:
        rng = np.random.default_rng(seed)
        field_idx = rng.choice(field_idx, max_n - int(member.sum()), replace=False)
    keep = np.sort(np.concatenate([np.flatnonzero(member), field_idx]))
    df, member = df.iloc[keep].reset_index(drop=True), member.to_numpy()[keep]
    lab = np.where(member, df.Name.astype(str), "field")
    feats = ["ra", "dec", "parallax", "pmra", "pmdec"]
    X = df[feats].to_numpy(np.float64)
    frac = len(field_idx) / n_field_total
    return X, lab, {"field"}, {
        "source": f"Gaia DR3 archive TAP (gaiadr3.gaia_source) + VizieR {HR23}/members",
        "gaia_box": {"ra": ra, "dec": dec, "parallax_mas": plx, "quality": GAIA_QUALITY},
        "documentation": "https://doi.org/10.1051/0004-6361/202346285 ; https://gea.esac.esa.int/archive/",
        "citation": "Gaia Collaboration (2023), A&A 674:A1; Hunt & Reffert (2023), A&A 673:A114",
        "features": feats,
        "anomaly_origin": "natural",
        "construction": (f"Every Gaia DR3 star in the box passing the quality cuts. Stars that are Hunt & Reffert (2023) "
                         f"members with Prob>={prob} of a cluster with >={min_members} such members in the box are clusters; "
                         f"all other stars are field (= anomaly / background, label 'field'). {n_dropped} stars of smaller "
                         f"clusters or with lower membership probability were dropped. Field subsampled uniformly to "
                         f"{frac:.1%} of {n_field_total} (seed={seed}) to fit n<={max_n}; the field's shape is unchanged. "
                         "Contamination is high by nature: this is 'clusters in a real background', not rare-point detection. "
                         "5D astrometry only; photometry excluded."),
    }, raw


def build_gaia_vela_puppis(src: Path):
    return _gaia_patch(src, "vela_puppis", ra=(103, 118), dec=(-40, -33), plx=(2, 7))


def build_gaia_orion(src: Path):
    return _gaia_patch(src, "orion", ra=(78, 88), dec=(-5, 4), plx=(2, 3.2))


BUILDERS ={name: globals()[f"build_{name}"] for name in SPLIT}


def write(name: str, data_dir: Path) -> dict:
    split = SPLIT[name]
    out_dir = data_dir / split / "real_world"
    src = out_dir / "source"
    X, lab, anomalies, meta, raw = BUILDERS[name](src)
    y, clusters = encode(lab, anomalies)
    assert np.isfinite(X).all(), name
    out = out_dir / f"{name}.npz"
    np.savez_compressed(out, X=X, y=y, source_y=lab.astype(str))
    info = {
        "name": name,
        "split": split,
        "path": str(out.relative_to(_DELTRIC_MINIMAL_DIR)),
        "n": int(len(X)),
        "d": int(X.shape[1]),
        "k_clusters": len(clusters),
        "clusters": clusters,
        "anomaly_classes": sorted(anomalies),
        "n_anomaly": int((y < 0).sum()),
        "contamination": float((y < 0).mean()),
        "duplicate_feature_rows": int(len(X) - len(np.unique(X, axis=0))),
        "label_encoding": {"y": "-1 anomaly, 0..k-1 cluster (ordered by size)", "source_y": "original class label"},
        "preprocessing": "Published features, no scaling; evaluator standardizes per dataset.",
        **meta,
        "sha256": sha256_bytes(out.read_bytes()),
        "source_sha256": sha256_bytes(raw),
    }
    (out_dir / f"{name}.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"[{split:7s}] {name:16s} n={info['n']:6d} d={info['d']:3d} k={info['k_clusters']:2d} "
          f"anom={info['n_anomaly']} ({info['contamination']:.1%}) {info['anomaly_origin']}")
    return info


def write_split(infos: list[dict], data_dir: Path) -> None:
    for split in ("testing", "holdout"):
        path = data_dir / split / "real_world" / "split.json"
        old = json.loads(path.read_text())["datasets"] if path.exists() else []
        merged = {d["name"]: d for d in old}
        merged.update({d["name"]: {k: d[k] for k in ("name", "split", "path", "n", "d", "k_clusters",
                                                     "n_anomaly", "anomaly_origin", "sha256")}
                       for d in infos if d["split"] == split})
        path.write_text(json.dumps({
            "created": str(date.today()),
            "split_unit": "whole dataset",
            "policy": "Split fixed before any DelTriC run, balanced by modality and anomaly origin. Holdout is not fit, plotted or scored during tuning.",
            "holdout_evaluated": False,
            "datasets": sorted(merged.values(), key=lambda d: d["name"]),
        }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=list(SPLIT), default=list(SPLIT))
    parser.add_argument("--data-dir", type=Path, default=_DATA_DIR)
    args = parser.parse_args()
    write_split([write(name, args.data_dir) for name in args.only], args.data_dir)


if __name__ == "__main__":
    main()
