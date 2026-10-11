#!/usr/bin/env python3
"""G1 (R1-2): how SDDA relates to the reconstruction-based anomaly score. DESCRIPTIVE only (no hypothesis tests).
 (a) sample level : Spearman(sum_j D_j, reconstruction score) per unit; AUROC of sum_j D_j for detected vs not.
 (b) feature level: per-sample Spearman(D_j, e_j) between attribution vector and feature reconstruction error.
 Variants: z_norm (SDDA_z), rho_raw (TSA_adapted), rho_norm, absdev_norm."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.metrics import roc_auc_score
BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/home/jupyteruser/nix-ext-backup/0-eksperimen-final/0-revisi-sdda-v1')
JOBS = BASE / 'eq9-ablation-v1' / 'jobs'; OUT = BASE / 'score-link-check'; OUT.mkdir(exist_ok=True)
KEYS = ['mag2.0', 'mag4.0', 'mag6.0', 'donor', 'cf']; VARS = ['z_norm', 'rho_raw', 'rho_norm', 'absdev_norm']
def rowspear(A, B):
    ra = np.apply_along_axis(rankdata, 1, A); rb = np.apply_along_axis(rankdata, 1, B)
    ra -= ra.mean(1, keepdims=True); rb -= rb.mean(1, keepdims=True)
    den = np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1)); return np.where(den > 0, (ra * rb).sum(1) / np.where(den > 0, den, 1), np.nan)
U = []
dirs = sorted(d for d in JOBS.glob('*_seed*_fold*') if (d / 'done.json').is_file())
print(f'jobs found: {len(dirs)}')
for d in dirs:
    dn = json.loads((d / 'done.json').read_text()); A = np.load(d / 'arrays.npz', allow_pickle=False); th = float(A['th'])
    for k in KEYS:
        pe = A[f'{k}__pe'].astype(float); e = A[f'{k}__e'].astype(float); det = pe > th
        for v in VARS:
            D = A[f'{k}__attr_{v}'].astype(float); mass = D.sum(1)
            sp = spearmanr(mass, pe).statistic if np.ptp(mass) > 0 and np.ptp(pe) > 0 else np.nan
            auc = roc_auc_score(det, mass) if 0 < det.mean() < 1 else np.nan
            fs = rowspear(D, e)
            U.append({'dataset': dn['dataset'], 'seed_idx': dn['seed_idx'], 'fold': dn['fold'], 'protocol': k, 'variant': v,
                      'n': len(pe), 'detected_rate': det.mean(), 'spearman_mass_vs_score': sp, 'auroc_mass_detected': auc,
                      'feature_spearman_D_vs_e_mean': np.nanmean(fs), 'feature_spearman_D_vs_e_median': np.nanmedian(fs)})
U = pd.DataFrame(U); U.to_csv(OUT / 'g1_per_unit.csv', index=False)
S = U.groupby(['dataset', 'protocol', 'variant']).agg(units=('n', 'size'), detected_rate=('detected_rate', 'mean'),
    spearman_mass_score=('spearman_mass_vs_score', 'mean'), spearman_mass_score_min=('spearman_mass_vs_score', 'min'),
    spearman_mass_score_max=('spearman_mass_vs_score', 'max'), auroc_mass_detected=('auroc_mass_detected', 'mean'),
    auroc_units=('auroc_mass_detected', 'count'), feat_spearman_D_e=('feature_spearman_D_vs_e_mean', 'mean')).reset_index()
S.to_csv(OUT / 'g1_summary.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
print('\n(a) sample level: mean over 30 units of Spearman(sum attribution, reconstruction score); AUROC for detected (NaN if one class)')
print(S.pivot_table(index=['dataset', 'protocol'], columns='variant', values='spearman_mass_score').round(3).to_string())
print('\n    AUROC(sum attribution -> detected):')
print(S.pivot_table(index=['dataset', 'protocol'], columns='variant', values='auroc_mass_detected').round(3).to_string())
print('\n(b) feature level: mean per-sample Spearman(D_j, e_j) between attribution and feature reconstruction error')
print(S.pivot_table(index=['dataset', 'protocol'], columns='variant', values='feat_spearman_D_e').round(3).to_string())
print('\nFiles:', OUT)
