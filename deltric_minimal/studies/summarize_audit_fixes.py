"""Aggregate verify_audit_fixes.py results into per-group paired deltas vs baseline."""
from pathlib import Path
import json
import sys

import numpy as np

OUT = Path(__file__).resolve().parent / 'results/verify_audit_fixes'
REAL = {'anuran_frogs', 'odds_cardio', 'gaia_ngc_2516', 'hcv', 'adbench_waveform',
        'odds_thyroid', 'odds_pendigits'}


def group(row):
    if row['dataset'] in REAL:
        return 'real'
    return '2D' if row['d'] == 2 else 'multiD'


rows = [json.loads(p.read_text()) for p in OUT.glob('*.json')]
by = {(r['dataset'], r['variant']): r for r in rows}
variants = sorted({r['variant'] for r in rows}, key=lambda v: (v != 'baseline', v))
stage = sys.argv[1] if len(sys.argv) > 1 else 'final'

print(f"stage={stage}; mean over datasets where both baseline and variant exist; "
      f"W/L = datasets improved/worsened by > 0.01")
for metric in ('ari_clean', 'f1'):
    print(f"\n=== {metric} ===")
    print(f"{'variant':26s}" + ''.join(f"{g:>30s}" for g in ('2D', 'multiD', 'real', 'ALL')))
    for v in variants:
        cells = []
        for g in ('2D', 'multiD', 'real', 'ALL'):
            base, var = [], []
            for (d, vv), r in by.items():
                if vv != v or (g != 'ALL' and group(r) != g) or (d, 'baseline') not in by:
                    continue
                b = by[(d, 'baseline')]['final'][metric]
                x = r[stage][metric]
                if b is None or x is None:
                    continue
                base.append(b)
                var.append(x)
            if not base:
                cells.append(f"{'-':>30s}")
                continue
            base, var = np.array(base), np.array(var)
            diff = var - base
            cells.append(f"{var.mean():.3f} ({diff.mean():+.3f}) {int((diff > .01).sum()):2d}/{int((diff < -.01).sum()):2d} n={len(base):2d}")
        print(f"{v:26s}" + ''.join(f"{c:>30s}" for c in cells))
