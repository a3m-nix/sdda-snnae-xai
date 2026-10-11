#!/usr/bin/env python3
"""SDDA revision audit (JKSUCIS major revision) — read-only, CSV/NPZ only, no TF.

RUN:
  python 7-revision-csv-audit.py
  python 7-revision-csv-audit.py --base /path/to/paper3-faithfulness-v4-3x10

A1  Corrected repeated k-fold t-test (Nadeau & Bengio 2003; Bouckaert & Frank 2004)
    on the SAME 11 run-level metrics and SAME (seed,fold) units as the manuscript.
    Parity check: mean differences must match tables/table_statistics_all_pairs.csv.
A2  Faithfulness stratified by `detected` (controlled all magnitudes + empirical donor),
    plus corrected test SDDA_z vs others inside each stratum.
A3  Baseline degeneracy audit: neurons with sd_sdp ~ 0 (dead / always-on) per unit.

Does NOT read feature_behavior.csv or suppression.csv (large).
Outputs go to a NEW folder next to BASE: revision-audit-<UTC timestamp>/
"""
import argparse, json, sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as tdist, wilcoxon

DEFAULT = '/home/jupyteruser/nix-ext-backup/0-eksperimen-final/paper3-faithfulness-v4-3x10'
METHODS = ['SDDA_z', 'SDDA_KL', 'SDDA_RW', 'TSA_adapted', 'RawError', 'KernelSHAP', 'Random']
FOCUS = ['TSA_adapted', 'KernelSHAP', 'RawError', 'SDDA_RW']
KEY = ['dataset', 'seed_idx', 'fold', 'method']
EPS_STAB = 0.1  # max(CFG["stability_eps"]) in v4-3x10


def holm(p):
    out, prev = {}, 0.0
    a = sorted(p.items(), key=lambda x: x[1])
    for i, (k, v) in enumerate(a):
        prev = max(prev, (len(a) - i) * v)
        out[k] = min(1.0, prev)
    return out


def corrected_t(d, r):
    """Corrected resampled t-test. d: per-unit paired differences, r = n_test/n_train."""
    d = np.asarray(d, float)
    J = len(d)
    m, s2 = d.mean(), d.var(ddof=1) if J > 1 else 0.0
    if J < 3:
        return np.nan, np.nan
    if s2 == 0:
        return (0.0, 1.0) if m == 0 else (np.inf, 0.0)
    t = m / np.sqrt((1.0 / J + r) * s2)
    return float(t), float(2 * tdist.sf(abs(t), J - 1))


def safe_wilcoxon(d):
    try:
        return float(wilcoxon(d).pvalue)
    except Exception:
        return 1.0


def run_level(base):
    raw = base / 'raw'
    F = pd.read_csv(raw / 'faithfulness.csv')
    C = pd.read_csv(raw / 'counterfactual.csv')
    S = pd.read_csv(raw / 'stability.csv')
    R = pd.read_csv(raw / 'runtime.csv')
    Fmax = F[F.magnitude == F.magnitude.max()]
    fctrl = (Fmax[Fmax.perturbation_mode == 'scaled_std'].groupby(KEY)
             .agg(p3=('p3', 'mean'), ndcg5=('ndcg5', 'mean'), feature_auc=('feature_auc', 'mean'))
             .reset_index())
    fdon = (F[F.perturbation_mode == 'empirical_attack_donor'].groupby(KEY)
            .agg(p3=('p3', 'mean')).reset_index())
    Cr = C.groupby(KEY).agg(log_audc=('log_audc', 'mean'), log_auic=('log_auic', 'mean'),
                            restricted_k=('restricted_deletion_k', 'mean'),
                            restricted_insertion_k=('restricted_insertion_k', 'mean')).reset_index()
    eps = S.eps.max()
    if not np.isclose(eps, EPS_STAB):
        print(f'[warn] max stability eps = {eps}, expected {EPS_STAB}')
    Sr = (S[(S.regime == 'input_noise') & (S.eps == eps)].groupby(KEY)
          .agg(spearman=('spearman', 'mean'), jaccard5=('jaccard5', 'mean')).reset_index())
    Rr = R.groupby(KEY).agg(sec=('seconds_per_sample', 'mean')).reset_index()
    spec = [(fctrl, 'p3', 'faith_p3_controlled'), (fdon, 'p3', 'faith_p3_donor'),
            (fctrl, 'ndcg5', 'faith_ndcg5'), (fctrl, 'feature_auc', 'faith_feature_auc'),
            (Cr, 'log_audc', 'log_audc'), (Cr, 'log_auic', 'log_auic'),
            (Cr, 'restricted_k', 'restricted_k'),
            (Cr, 'restricted_insertion_k', 'restricted_insertion_k'),
            (Sr, 'spearman', 'stability_spearman'), (Sr, 'jaccard5', 'stability_jaccard'),
            (Rr, 'sec', 'latency_s')]
    out = []
    for df, col, metric in spec:
        t = df[KEY + [col]].rename(columns={col: 'value'})
        t['metric'] = metric
        out.append(t)
    return F, pd.concat(out, ignore_index=True)


def ratio_by_dataset(base):
    sa = pd.read_csv(base / 'raw' / 'split_audit.csv')
    r = (sa.n_test / sa.n_train_normal).groupby(sa.dataset).mean()
    return r.to_dict(), sa


def a1(RUN, r, base, out):
    rows = []
    for (ds, metric), g in RUN.groupby(['dataset', 'metric']):
        P = g.pivot_table(index=['seed_idx', 'fold'], columns='method', values='value')
        ms = [m for m in METHODS if m in P and P[m].notna().all()]
        pw, pc, info = {}, {}, {}
        for a, b in combinations(ms, 2):
            d = (P[a] - P[b]).values
            k = f'{a} vs {b}'
            pw[k] = safe_wilcoxon(d)
            tc, pc[k] = corrected_t(d, r[ds])
            info[k] = dict(mean_difference=float(d.mean()), sd_difference=float(d.std(ddof=1)),
                           units=len(d), wins_a=int((d > 0).sum()), wins_b=int((d < 0).sum()),
                           t_corrected=tc)
        hw, hc = holm(pw), holm(pc)
        for k in pw:
            rows.append(dict(dataset=ds, metric=metric, comparison=k, r_test_train=r[ds],
                             **info[k], p_wilcoxon=pw[k], p_wilcoxon_holm=hw[k],
                             p_corrected=pc[k], p_corrected_holm=hc[k]))
    T = pd.DataFrame(rows)
    T.to_csv(out / 'A1_corrected_ttest_all_pairs.csv', index=False)
    # Parity with manuscript statistics table
    ref_path = base / 'tables' / 'table_statistics_all_pairs.csv'
    if ref_path.is_file():
        ref = pd.read_csv(ref_path)
        mm = T.merge(ref, on=['dataset', 'metric', 'comparison'], suffixes=('', '_ref'))
        dmax = float(np.abs(mm.mean_difference - mm.mean_difference_ref).max())
        pmax = float(np.abs(mm.p_wilcoxon_holm - mm.p_holm_all_pairs).max())
        print(f'[A1 parity] matched {len(mm)}/{len(ref)} rows | max |Δmean| = {dmax:.2e} '
              f'| max |Δp_holm| = {pmax:.2e}')
        if len(mm) != len(ref) or dmax > 1e-9:
            print('[A1 parity] WARNING: run-level reconstruction differs from manuscript table.')
    return T


def a2(F, r, out):
    g = (F.groupby(['dataset', 'perturbation_mode', 'magnitude', 'detected', 'method'])
         .agg(n=('sample', 'size'), p3=('p3', 'mean'), ndcg5=('ndcg5', 'mean'),
              feature_auc=('feature_auc', 'mean')).reset_index())
    kk = ['dataset', 'perturbation_mode', 'magnitude', 'method']
    tot = F.groupby(kk).size().rename('n_total').reset_index()
    g = g.merge(tot, on=kk)
    g['share'] = g.n / g.n_total
    g.to_csv(out / 'A2_faithfulness_by_detected.csv', index=False)
    rows = []
    for (ds, mode, mag, det), h in F.groupby(['dataset', 'perturbation_mode', 'magnitude', 'detected']):
        U = h.groupby(['seed_idx', 'fold', 'method']).p3.mean().unstack('method')
        for b in FOCUS:
            if 'SDDA_z' not in U or b not in U:
                continue
            d = (U['SDDA_z'] - U[b]).dropna().values
            tc, pc = corrected_t(d, r[ds])
            rows.append(dict(dataset=ds, perturbation_mode=mode, magnitude=mag, detected=det,
                             comparison=f'SDDA_z vs {b}', units=len(d),
                             mean_difference_p3=float(d.mean()) if len(d) else np.nan,
                             p_corrected=pc))
    S = pd.DataFrame(rows)
    S.to_csv(out / 'A2_detected_stratum_tests.csv', index=False)
    return g, S


def a3(base, out):
    rows = []
    for p in sorted((base / 'baselines').glob('*/seed*_fold*/base/baseline.npz')):
        ds, unit = p.parts[-4], p.parts[-3]
        b = np.load(p)
        mu, sd = b['mu_sdp'].astype(float), b['sd_sdp'].astype(float)
        zero = sd < 1e-6
        rows.append(dict(dataset=ds, unit=unit, n_neurons=len(sd), n_val=int(b['n_val']),
                         n_sd_zero=int(zero.sum()), n_dead=int((zero & (mu < 1e-6)).sum()),
                         n_always_on=int((zero & (mu > 1 - 1e-6)).sum()),
                         n_sd_below_0_05=int((sd < 0.05).sum()),
                         min_sd=float(sd.min()), median_sd=float(np.median(sd)),
                         max_z_for_one_step=float((1 / 3) / (sd.min() + 1e-8)),
                         zero_idx=';'.join(map(str, np.flatnonzero(zero)))))
    A = pd.DataFrame(rows)
    A.to_csv(out / 'A3_baseline_degeneracy.csv', index=False)
    return A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default=DEFAULT)
    args = ap.parse_args()
    base = Path(args.base).resolve()
    for f in ['raw/faithfulness.csv', 'raw/counterfactual.csv', 'raw/stability.csv',
              'raw/runtime.csv', 'raw/split_audit.csv']:
        if not (base / f).is_file():
            sys.exit(f'Missing: {base / f}')
    out = base.parent / ('revision-audit-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))
    out.mkdir(parents=True, exist_ok=False)
    pd.set_option('display.width', 220); pd.set_option('display.max_rows', 300)

    r, sa = ratio_by_dataset(base)
    print('r = n_test/n_train_normal:', {k: round(v, 4) for k, v in r.items()})
    F, RUN = run_level(base)
    units = RUN.groupby(['dataset', 'metric', 'method']).size()
    print('units per dataset/metric/method: min', units.min(), 'max', units.max())

    T = a1(RUN, r, base, out)
    foc = T[T.comparison.str.startswith('SDDA_z vs ') & T.comparison.str.split(' vs ').str[1].isin(FOCUS)]
    print('\n=== A1: SDDA_z vs focus methods (positive mean = SDDA_z higher) ===')
    print(foc[['dataset', 'metric', 'comparison', 'mean_difference', 'wins_a', 'wins_b',
               'p_wilcoxon_holm', 'p_corrected', 'p_corrected_holm']]
          .round(4).to_string(index=False))
    sig = T.assign(w=T.p_wilcoxon_holm < .05, c=T.p_corrected_holm < .05)
    print('\n[A1] significant pairs (of all 21x11x2): Wilcoxon-Holm', int(sig.w.sum()),
          '| corrected-Holm', int(sig.c.sum()), '| lost', int((sig.w & ~sig.c).sum()))

    g, S = a2(F, r, out)
    print('\n=== A2: p3 by detected (SDDA_z, TSA_adapted, KernelSHAP, RawError) ===')
    keep = g[g.method.isin(['SDDA_z'] + FOCUS[:3])]
    print(keep.pivot_table(index=['dataset', 'perturbation_mode', 'magnitude', 'detected'],
                           columns='method', values='p3').round(3).to_string())
    print('\nshare of samples by detected (SDDA_z rows):')
    print(g[g.method == 'SDDA_z'][['dataset', 'perturbation_mode', 'magnitude', 'detected', 'n', 'share']]
          .round(3).to_string(index=False))
    print('\n=== A2: corrected test SDDA_z vs others within stratum (p3) ===')
    print(S.round(4).to_string(index=False))

    A = a3(base, out)
    print('\n=== A3: baseline degeneracy (sd_sdp < 1e-6) ===')
    print(A.groupby('dataset').agg(units=('unit', 'size'), units_with_zero_sd=('n_sd_zero', lambda v: int((v > 0).sum())),
                                   mean_zero_sd=('n_sd_zero', 'mean'), max_zero_sd=('n_sd_zero', 'max'),
                                   mean_dead=('n_dead', 'mean'), mean_always_on=('n_always_on', 'mean'),
                                   mean_sd_lt_005=('n_sd_below_0_05', 'mean'),
                                   max_z_one_step=('max_z_for_one_step', 'max')).round(3).to_string())

    json.dump({'base': str(base), 'r_test_train': r, 'created_utc': datetime.now(timezone.utc).isoformat()},
              open(out / 'manifest.json', 'w'), indent=2)
    print('\nOUTPUT:', out)


if __name__ == '__main__':
    main()
