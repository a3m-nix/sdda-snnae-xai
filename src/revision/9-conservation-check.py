#!/usr/bin/env python3
"""Encoder-divergence mass conservation check on E1 arrays (read-only, numpy only, no TF).
For each protocol x normalised variant: sum_j D_j vs sum_i s_i, theory sum_i s_i*C_i/(C_i+eps)."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/home/jupyteruser/nix-ext-backup/0-eksperimen-final/0-revisi-sdda-v1')
JOBS = BASE / 'eq9-ablation-v1' / 'jobs'
OUT = BASE / 'conservation-check'; OUT.mkdir(exist_ok=True)
EPS, SD_EPS = 1e-12, 1e-8
KEYS = ['mag2.0', 'mag4.0', 'mag6.0', 'donor', 'cf']
rows = []
dirs = sorted(d for d in JOBS.glob('*_seed*_fold*') if (d / 'done.json').is_file() and not d.name.endswith('.partial'))
print(f'jobs found: {len(dirs)}')
for d in dirs:
    dn = json.loads((d / 'done.json').read_text())
    A = np.load(d / 'arrays.npz', allow_pickle=False)
    W = A['W'].astype(np.float64); mu = A['mu_sdp'].astype(np.float64); sd = A['sd_sdp'].astype(np.float64)
    for k in KEYS:
        xb = A[f'{k}__xb'].astype(np.float64); S = A[f'{k}__S'].astype(np.float64)
        c = np.abs(xb[:, :, None] * W[None, :, :])          # (n, features, neurons)
        C = c.sum(1)                                         # (n, neurons)
        ch = c / (C[:, None, :] + EPS)
        sig = {'z_norm': np.abs(S - mu) / (sd + SD_EPS), 'absdev_norm': np.abs(S - mu), 'rho_norm': S}
        for v, s in sig.items():
            D = np.einsum('nji,ni->nj', ch, s)
            tot, tgt = D.sum(1), s.sum(1)
            theory = (s * C / (C + EPS)).sum(1)
            stored = A[f'{k}__attr_{v}'].astype(np.float64)
            rel = np.abs(tot - tgt) / np.maximum(np.abs(tgt), 1e-300)
            rows.append({'dataset': dn['dataset'], 'seed': dn['seed_idx'], 'fold': dn['fold'], 'protocol': k,
                         'variant': v, 'n_samples': len(S), 'neuron_entries': C.size,
                         'n_C_zero': int((C == 0).sum()), 'n_C_below_1e-6': int((C < 1e-6).sum()),
                         'min_C': float(C.min()),
                         'n_target_zero': int((tgt == 0).sum()),
                         'max_abs_resid': float(np.abs(tot - tgt).max()),
                         'max_rel_resid': float(rel[tgt > 0].max()) if (tgt > 0).any() else 0.0,
                         'median_rel_resid': float(np.median(rel[tgt > 0])) if (tgt > 0).any() else 0.0,
                         'max_gap_vs_theory': float(np.abs(tot - theory).max()),
                         'max_abs_diff_vs_stored_float32': float(np.abs(D - stored).max()),
                         'max_rel_resid_stored_float32': float((np.abs(stored.sum(1) - tgt) /
                                                                np.maximum(np.abs(tgt), 1e-300))[tgt > 0].max())
                         if (tgt > 0).any() else 0.0})
R = pd.DataFrame(rows)
R.to_csv(OUT / 'conservation_per_job.csv', index=False)
g = R.groupby(['dataset', 'variant', 'protocol']).agg(
    jobs=('seed', 'size'), samples=('n_samples', 'sum'), neuron_entries=('neuron_entries', 'sum'),
    C_zero=('n_C_zero', 'sum'), C_below_1e6=('n_C_below_1e-6', 'sum'), min_C=('min_C', 'min'),
    target_zero=('n_target_zero', 'sum'), max_abs_resid=('max_abs_resid', 'max'),
    max_rel_resid=('max_rel_resid', 'max'), median_rel_resid=('median_rel_resid', 'median'),
    max_gap_theory=('max_gap_vs_theory', 'max'), max_diff_stored=('max_abs_diff_vs_stored_float32', 'max'),
    max_rel_resid_stored=('max_rel_resid_stored_float32', 'max')).reset_index()
g.to_csv(OUT / 'conservation_summary.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 100)
fmt = lambda x: f'{x:.2e}' if isinstance(x, float) else x
print(g.to_string(index=False, formatters={c: fmt for c in g.columns if g[c].dtype == float}))
print('\nOVERALL:', {'jobs': R[['dataset', 'seed', 'fold']].drop_duplicates().shape[0],
                     'C_zero_total': int(R.n_C_zero.sum()), 'min_C': f"{R.min_C.min():.3e}",
                     'max_rel_resid_float64': f"{R.max_rel_resid.max():.2e}",
                     'max_gap_vs_theory': f"{R.max_gap_vs_theory.max():.2e}",
                     'max_rel_resid_stored_float32': f"{R.max_rel_resid_stored_float32.max():.2e}"})
print('Files:', OUT / 'conservation_summary.csv', '|', OUT / 'conservation_per_job.csv')
