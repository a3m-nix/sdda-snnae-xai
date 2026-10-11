#!/usr/bin/env python3
"""Suppression-mass analysis on E1 outputs (read-only, numpy/pandas, no TF). SECONDARY, pre-specified:
 share_supp = sum_{i: rho_i<mu_i} delta_i / sum_i delta_i  (deviation mass invisible to activity-based attribution)
 Strata (fixed threshold, no tuning): suppression-dominant share_supp>0.5 vs activation-dominant <=0.5.
 Contrast d = P@3(z_norm) - P@3(rho_norm) [same projection]; reference: z_norm - rho_raw (TSA_adapted).
 Family (Holm, 4 tests): interaction d_supp - d_act per unit, datasets x {ctrl mag6, donor}.
 Decision: SUPPORTED only if corrected-t-Holm and Wilcoxon-Holm both p<0.05 and mean & Hodges-Lehmann same sign."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import t as tdist, wilcoxon, spearmanr
BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/home/jupyteruser/nix-ext-backup/0-eksperimen-final/0-revisi-sdda-v1')
SPLIT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('/home/jupyteruser/nix-ext-backup/0-eksperimen-final/paper3-faithfulness-v4-3x10/raw/split_audit.csv')
JOBS = BASE / 'eq9-ablation-v1' / 'jobs'; OUT = BASE / 'suppression-check'; OUT.mkdir(exist_ok=True)
PROTO = {'mag2.0': ('scaled_std', 2.0), 'mag4.0': ('scaled_std', 4.0), 'mag6.0': ('scaled_std', 6.0),
         'donor': ('empirical_attack_donor', 1.0)}
sa = pd.read_csv(SPLIT); R_RATIO = (sa.n_test / sa.n_train_normal).groupby(sa.dataset).mean().to_dict()

def holm(p):
    o, prev = {}, 0.0
    for i, (k, v) in enumerate(sorted(p.items(), key=lambda x: 1.0 if not np.isfinite(x[1]) else x[1])):
        v = 1.0 if not np.isfinite(v) else v; prev = max(prev, (len(p) - i) * v); o[k] = min(1.0, prev)
    return o

def stats(d, r):
    d = np.asarray(d, float); J = len(d)
    out = {'units': J, 'mean': d.mean() if J else np.nan, 'wins_pos': int((d > 0).sum()), 'wins_neg': int((d < 0).sum())}
    if J < 10:
        return {**out, 'hl': np.nan, 'ci95_unadj_low': np.nan, 'ci95_unadj_high': np.nan, 'p_corr': np.nan, 'p_wil': np.nan}
    i, j = np.triu_indices(J); hl = float(np.median((d[i] + d[j]) / 2))
    se = np.sqrt((1 / J + r) * d.var(ddof=1)); q = tdist.ppf(.975, J - 1)
    pc = (1.0 if d.mean() == 0 else 0.0) if se == 0 else float(2 * tdist.sf(abs(d.mean() / se), J - 1))
    try: pw = float(wilcoxon(d).pvalue)
    except Exception: pw = 1.0
    return {**out, 'hl': hl, 'ci95_unadj_low': d.mean() - q * se, 'ci95_unadj_high': d.mean() + q * se, 'p_corr': pc, 'p_wil': pw}

rows, cfrows = [], []
dirs = sorted(d for d in JOBS.glob('*_seed*_fold*') if (d / 'done.json').is_file())
print(f'jobs found: {len(dirs)}')
for d in dirs:
    dn = json.loads((d / 'done.json').read_text()); A = np.load(d / 'arrays.npz', allow_pickle=False)
    mu, sd = A['mu_sdp'].astype(float), A['sd_sdp'].astype(float)
    F = pd.read_csv(d / 'faith_samples.csv'); C = pd.read_csv(d / 'cf_samples.csv')
    for k in list(PROTO) + ['cf']:
        S = A[f'{k}__S'].astype(float); dl = np.abs(S - mu) / (sd + 1e-8); sup = S < mu
        tot = dl.sum(1); share = np.where(tot > 0, (dl * sup).sum(1) / np.where(tot > 0, tot, 1), np.nan)
        base = {'dataset': dn['dataset'], 'seed_idx': dn['seed_idx'], 'fold': dn['fold'], 'protocol': k}
        if k == 'cf':
            P = C.pivot_table(index='sample', columns='method', values='log_audc')
            for i in range(len(S)):
                cfrows.append({**base, 'sample': i, 'share_supp': share[i], 'silent': bool(S[i].sum() == 0),
                               'n_supp_neurons': int(sup[i].sum()), 'dAUDC_z_minus_rho_raw': P.loc[i, 'z_norm'] - P.loc[i, 'rho_raw'],
                               'dAUDC_z_minus_rho_norm': P.loc[i, 'z_norm'] - P.loc[i, 'rho_norm']})
            continue
        mode, mag = PROTO[k]
        P = F[(F.perturbation_mode == mode) & (F.magnitude == mag)].pivot_table(index='sample', columns='method', values='p3')
        det = F[(F.perturbation_mode == mode) & (F.magnitude == mag) & (F.method == 'z_norm')].set_index('sample').detected
        for i in range(len(S)):
            rows.append({**base, 'sample': i, 'share_supp': share[i], 'silent': bool(S[i].sum() == 0),
                         'n_supp_neurons': int(sup[i].sum()), 'detected': bool(det.loc[i]),
                         'p3_z_norm': P.loc[i, 'z_norm'], 'p3_rho_norm': P.loc[i, 'rho_norm'], 'p3_rho_raw': P.loc[i, 'rho_raw']})
S_ = pd.DataFrame(rows); CF = pd.DataFrame(cfrows)
S_['d_norm'] = S_.p3_z_norm - S_.p3_rho_norm; S_['d_tsa'] = S_.p3_z_norm - S_.p3_rho_raw
S_['stratum'] = np.where(S_.share_supp > 0.5, 'supp_dominant', 'act_dominant')
S_.to_csv(OUT / 'suppression_samples.csv', index=False); CF.to_csv(OUT / 'suppression_cf_samples.csv', index=False)

# 1) descriptive
desc = S_.groupby(['dataset', 'protocol']).agg(n=('sample', 'size'), share_mean=('share_supp', 'mean'),
        share_median=('share_supp', 'median'), frac_supp_dominant=('stratum', lambda s: (s == 'supp_dominant').mean()),
        silent=('silent', 'sum')).reset_index()
cfd = CF.groupby(['dataset']).agg(n=('sample', 'size'), share_mean=('share_supp', 'mean'), share_median=('share_supp', 'median'),
        frac_supp_dominant=('share_supp', lambda s: (s > 0.5).mean()), silent=('silent', 'sum')).reset_index().assign(protocol='cf (detected attacks)')
desc = pd.concat([desc, cfd], ignore_index=True); desc.to_csv(OUT / 'descriptive.csv', index=False)

# 2) stratum means (descriptive) + per-stratum contrasts
U = S_.groupby(['dataset', 'protocol', 'stratum', 'seed_idx', 'fold']).agg(d_norm=('d_norm', 'mean'), d_tsa=('d_tsa', 'mean'),
        p3_z=('p3_z_norm', 'mean'), p3_rho_norm=('p3_rho_norm', 'mean'), p3_rho_raw=('p3_rho_raw', 'mean'), n=('sample', 'size')).reset_index()
strat = U.groupby(['dataset', 'protocol', 'stratum']).agg(units=('n', 'size'), samples=('n', 'sum'), p3_z=('p3_z', 'mean'),
        p3_rho_norm=('p3_rho_norm', 'mean'), p3_rho_raw=('p3_rho_raw', 'mean'), d_norm=('d_norm', 'mean'), d_tsa=('d_tsa', 'mean')).reset_index()
strat.to_csv(OUT / 'stratum_means.csv', index=False)

# 3) pre-specified family: interaction (d_supp - d_act) per unit, z_norm vs rho_norm
fam, ref = [], []
for ds in sorted(S_.dataset.unique()):
    for k in ['mag6.0', 'donor', 'mag2.0', 'mag4.0']:
        W = U[(U.dataset == ds) & (U.protocol == k)].pivot_table(index=['seed_idx', 'fold'], columns='stratum', values=['d_norm', 'd_tsa'])
        for col, lst in [('d_norm', fam), ('d_tsa', ref)]:
            if ('supp_dominant' not in W[col]) or ('act_dominant' not in W[col]):
                inter = np.array([])
            else:
                inter = (W[col]['supp_dominant'] - W[col]['act_dominant']).dropna().values
            lst.append({'dataset': ds, 'protocol': k, 'primary': k in ('mag6.0', 'donor'),
                        'contrast': f'interaction {col}: (z - rho) supp_dominant - act_dominant', **stats(inter, R_RATIO[ds])})
def decide(rows_):
    prim = [r for r in rows_ if r['primary']]
    hc = holm({i: r['p_corr'] for i, r in enumerate(prim)}); hw = holm({i: r['p_wil'] for i, r in enumerate(prim)})
    for i, r in enumerate(prim):
        r['p_corr_holm'], r['p_wil_holm'] = hc[i], hw[i]
        sc, sw = hc[i] < .05, hw[i] < .05
        r['decision'] = (('SUPPORTED_POS' if r['mean'] > 0 else 'SUPPORTED_NEG') if sc and sw and np.sign(r['mean']) == np.sign(r['hl'])
                         else 'INCONSISTENT_DIRECTION' if sc and sw else 'INCONSISTENT' if (sc or sw) else 'NOT_SUPPORTED')
    for r in rows_:
        r.setdefault('decision', 'descriptive (not in family)')
    return pd.DataFrame(rows_)
FAM, REF = decide(fam), decide(ref)
FAM.to_csv(OUT / 'interaction_tests_z_vs_rho_norm.csv', index=False); REF.to_csv(OUT / 'interaction_tests_z_vs_tsa.csv', index=False)

# 4) continuous association (descriptive): per-unit Spearman(share_supp, d_norm), mean over units
sp = S_.groupby(['dataset', 'protocol', 'seed_idx', 'fold']).apply(
    lambda g: spearmanr(g.share_supp, g.d_norm).statistic if g.share_supp.nunique() > 1 and g.d_norm.nunique() > 1 else np.nan)
spm = sp.groupby(['dataset', 'protocol']).agg(['mean', 'count']).reset_index().rename(columns={'mean': 'mean_spearman_share_vs_d', 'count': 'units'})

pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
print('\n1) DESCRIPTIVE suppression share (fraction of delta mass from neurons with rho<mu):')
print(desc.round(3).to_string(index=False))
print('\n2) STRATUM MEANS (P@3; d_norm = z_norm - rho_norm, d_tsa = z_norm - rho_raw):')
print(strat.round(4).to_string(index=False))
print('\n3) PRE-SPECIFIED FAMILY (Holm over 4 primary: datasets x {mag6, donor}); z_norm vs rho_norm interaction:')
cols = ['dataset', 'protocol', 'units', 'mean', 'hl', 'ci95_unadj_low', 'ci95_unadj_high', 'wins_pos', 'wins_neg', 'p_corr', 'p_wil', 'p_corr_holm', 'p_wil_holm', 'decision']
print(FAM.reindex(columns=cols).round(4).to_string(index=False))
print('\n   reference (z_norm vs rho_raw = TSA_adapted):')
print(REF.reindex(columns=cols).round(4).to_string(index=False))
print('\n4) mean per-unit Spearman(share_supp, d_norm):'); print(spm.round(3).to_string(index=False))
sil = CF[CF.silent]
print(f'\n5) SILENT-ENCODER detected attacks (sum rho = 0): n={len(sil)}; mean dAUDC z-rho_raw = {sil.dAUDC_z_minus_rho_raw.mean() if len(sil) else float("nan"):.4f} (lower AUDC = better)')
print('Files:', OUT)
