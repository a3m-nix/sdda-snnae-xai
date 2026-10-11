#!/usr/bin/env python3
"""Two-test statistical decisions reported in the revised manuscript (Supplementary Tables S7 and S13).

S7  : all 462 paired comparisons (2 datasets x 11 metrics x 21 method pairs) over 30 seed-fold units.
      Each comparison is tested with the Wilcoxon signed-rank test and with the corrected resampled t-test
      (Nadeau and Bengio, 2003; r = n_test / n_train_normal). Holm correction is applied separately to each test
      within every dataset-metric family of 21 comparisons. Hodges-Lehmann (HL) = median of Walsh averages.
      Decision: SUPPORTED only if both Holm-adjusted p < 0.05 AND sign(mean difference) == sign(HL);
      significant under both but signs differ -> INCONSISTENT_DIRECTION; significant under one test only ->
      INCONSISTENT; otherwise NOT_SUPPORTED. Lower is better for log-AUDC, restricted deletion/insertion k, latency.
S13 : re-test of the suppression-stratified deletion interaction of Table 10. Per seed-fold unit,
      delta = mean log-AUDC(SDDA-z) - mean log-AUDC(TSA-adapted) within each suppression stratum;
      interaction = delta(high) - delta(low). Same two tests, Holm over the two datasets, corrected-t 95% CI.

Inputs (read only):
  --pipeline  results of the main evaluation (raw/faithfulness.csv, counterfactual.csv, stability.csv, runtime.csv,
              split_audit.csv; tables/table_suppression_interaction.csv for the parity check)
  --ablation  replay of the main evaluation (jobs/*/cf_samples.csv with the suppression stratum per sample)
Outputs (new folder, never overwritten): S7_all_pairwise_decisions.csv, S7_decision_counts.csv, S13_table10_retest.csv,
  summary.txt. The script stops with CHECK FAILED if a reported number is not reproduced.
"""
import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as tdist, wilcoxon

SERVER = Path('/home/jupyteruser/nix-ext-backup/0-eksperimen-final')
ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
ap.add_argument('--pipeline', type=Path, default=SERVER / 'paper3-faithfulness-v4-3x10')
ap.add_argument('--ablation', type=Path, default=SERVER / '0-revisi-sdda-v1' / 'eq9-ablation-v1')
ap.add_argument('--out', type=Path, default=SERVER / '0-revisi-sdda-v1' / 'two-test-decisions')
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=False)

METHODS = ['SDDA_z', 'SDDA_KL', 'SDDA_RW', 'TSA_adapted', 'RawError', 'KernelSHAP', 'Random']
KEY = ['dataset', 'seed_idx', 'fold', 'method']
LOWER_IS_BETTER = {'log_audc', 'restricted_k', 'restricted_insertion_k', 'latency_s'}
ALPHA = 0.05
LOG = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def require(cond, msg):
    if not cond:
        say(f'CHECK FAILED: {msg}')
        (args.out / 'summary.txt').write_text('\n'.join(LOG) + '\n')
        sys.exit(1)


# ============================================================ START TESTS
def holm(p):
    """Holm step-down adjustment; p: dict name -> p-value."""
    out, prev = {}, 0.0
    items = sorted(p.items(), key=lambda kv: kv[1])
    for i, (k, v) in enumerate(items):
        prev = max(prev, (len(items) - i) * v)
        out[k] = min(1.0, prev)
    return out


def corrected_t(d, r):
    """Corrected resampled t-test (Nadeau and Bengio, 2003). Returns t, p, 95% CI half-width."""
    d = np.asarray(d, float)
    J = len(d)
    m, s2 = d.mean(), d.var(ddof=1)
    if s2 == 0:
        return (0.0, 1.0, 0.0) if m == 0 else (np.inf, 0.0, 0.0)
    se = np.sqrt((1.0 / J + r) * s2)
    t = m / se
    return float(t), float(2 * tdist.sf(abs(t), J - 1)), float(tdist.ppf(0.975, J - 1) * se)


def wilcoxon_p(d):
    try:
        return float(wilcoxon(d).pvalue)
    except ValueError:                                   # all differences zero
        return 1.0


def hodges_lehmann(d):
    d = np.asarray(d, float)
    i, j = np.triu_indices(len(d))
    return float(np.median((d[i] + d[j]) / 2))


def decide(p_corr_holm, p_wil_holm, mean, hl):
    if p_corr_holm < ALPHA and p_wil_holm < ALPHA:
        return 'SUPPORTED' if (mean != 0 and np.sign(mean) == np.sign(hl)) else 'INCONSISTENT_DIRECTION'
    return 'INCONSISTENT' if (p_corr_holm < ALPHA or p_wil_holm < ALPHA) else 'NOT_SUPPORTED'
# ============================================================ END TESTS


# ============================================================ START S7
raw = args.pipeline / 'raw'
sa = pd.read_csv(raw / 'split_audit.csv')
R_TEST_TRAIN = (sa.n_test / sa.n_train_normal).groupby(sa.dataset).mean().to_dict()
say('r = n_test / n_train_normal:', {k: round(v, 4) for k, v in R_TEST_TRAIN.items()})

F = pd.read_csv(raw / 'faithfulness.csv')
C = pd.read_csv(raw / 'counterfactual.csv')
S = pd.read_csv(raw / 'stability.csv')
T = pd.read_csv(raw / 'runtime.csv')
Fm = F[F.magnitude == F.magnitude.max()]
ctrl = Fm[Fm.perturbation_mode == 'scaled_std'].groupby(KEY)
donor = F[F.perturbation_mode == 'empirical_attack_donor'].groupby(KEY)
cf = C.groupby(KEY)
st = S[(S.regime == 'input_noise') & (S.eps == S.eps.max())].groupby(KEY)
say('controlled magnitude:', F.magnitude.max(), '| stability eps:', S.eps.max())
RUN = pd.concat({
    'faith_p3_controlled': ctrl.p3.mean(), 'faith_p3_donor': donor.p3.mean(), 'faith_ndcg5': ctrl.ndcg5.mean(),
    'faith_feature_auc': ctrl.feature_auc.mean(), 'log_audc': cf.log_audc.mean(), 'log_auic': cf.log_auic.mean(),
    'restricted_k': cf.restricted_deletion_k.mean(), 'restricted_insertion_k': cf.restricted_insertion_k.mean(),
    'stability_spearman': st.spearman.mean(), 'stability_jaccard': st.jaccard5.mean(),
    'latency_s': T.groupby(KEY).seconds_per_sample.mean()}, axis=1).reset_index()

rows = []
for ds in sorted(RUN.dataset.unique()):
    for metric in [c for c in RUN.columns if c not in KEY]:
        P = RUN[RUN.dataset == ds].pivot_table(index=['seed_idx', 'fold'], columns='method', values=metric)
        require(list(P.columns.sort_values()) == sorted(METHODS) and len(P) == 30 and P.notna().all().all(),
                f'{ds} {metric}: expected 7 methods x 30 complete units')
        res = {}
        for a, b in combinations(METHODS, 2):
            d = (P[a] - P[b]).values
            t, pc, _ = corrected_t(d, R_TEST_TRAIN[ds])
            res[f'{a} vs {b}'] = dict(dataset=ds, metric=metric, comparison=f'{a} vs {b}', units=len(d),
                                      mean_difference=float(d.mean()), hodges_lehmann=hodges_lehmann(d),
                                      wins_a=int((d > 0).sum()), wins_b=int((d < 0).sum()), t_corrected=t,
                                      p_wilcoxon=wilcoxon_p(d), p_corrected=pc)
        hw = holm({k: v['p_wilcoxon'] for k, v in res.items()})
        hc = holm({k: v['p_corrected'] for k, v in res.items()})
        for k, v in res.items():
            v.update(p_wilcoxon_holm=hw[k], p_corrected_holm=hc[k])
            v['decision'] = decide(hc[k], hw[k], v['mean_difference'], v['hodges_lehmann'])
            a, b = k.split(' vs ')
            sign = -1 if metric in LOWER_IS_BETTER else 1
            v['better'] = (a if v['mean_difference'] * sign > 0 else b) if v['decision'] == 'SUPPORTED' else ''
            rows.append(v)
S7 = pd.DataFrame(rows)
S7.to_csv(args.out / 'S7_all_pairwise_decisions.csv', index=False)
counts = S7.groupby(['dataset', 'metric']).decision.value_counts().unstack(fill_value=0).reset_index()
counts.to_csv(args.out / 'S7_decision_counts.csv', index=False)

tot = S7.decision.value_counts()
wil_only = int((S7.p_wilcoxon_holm < ALPHA).sum())
z_tsa = S7[S7.comparison == 'SDDA_z vs TSA_adapted']
say(f'\nS7: {len(S7)} comparisons | supported {tot.get("SUPPORTED", 0)} | inconsistent {tot.get("INCONSISTENT", 0)} | '
    f'inconsistent direction {tot.get("INCONSISTENT_DIRECTION", 0)} | not supported {tot.get("NOT_SUPPORTED", 0)}')
say(f'S7: significant under Wilcoxon-Holm alone: {wil_only}')
say(f'S7: SDDA-z vs TSA-adapted supported: {int((z_tsa.decision == "SUPPORTED").sum())} of {len(z_tsa)}')
require(len(S7) == 462, 'expected 462 comparisons')
require((tot.get('SUPPORTED', 0), tot.get('INCONSISTENT', 0), tot.get('INCONSISTENT_DIRECTION', 0),
         tot.get('NOT_SUPPORTED', 0)) == (257, 108, 0, 97), 'decision totals differ from 257/108/0/97')
require(wil_only == 365, 'Wilcoxon-Holm count differs from 365')
require(len(z_tsa) == 22 and (z_tsa.decision == 'SUPPORTED').sum() == 0, 'SDDA-z vs TSA-adapted is not 0 of 22')

ref = args.pipeline / 'tables' / 'table_statistics_all_pairs.csv'      # Wilcoxon-Holm table of the original submission
if ref.is_file():
    m = S7.merge(pd.read_csv(ref), on=['dataset', 'metric', 'comparison'], suffixes=('', '_ref'))
    dmax = float(np.abs(m.mean_difference - m.mean_difference_ref).max())
    pmax = float(np.abs(m.p_wilcoxon_holm - m.p_holm_all_pairs).max())
    say(f'S7 parity with the original Wilcoxon-Holm table: {len(m)} matched rows, max |d mean| {dmax:.1e}, '
        f'max |d p| {pmax:.1e}')
    require(len(m) > 0 and dmax < 1e-9 and pmax < 1e-9, 'run-level metrics differ from the original evaluation')
# ============================================================ END S7


# ============================================================ START S13
jobs = sorted(d for d in (args.ablation / 'jobs').glob('*_seed*_fold*') if (d / 'done.json').is_file())
require(len(jobs) == 60, f'expected 60 finished ablation jobs, found {len(jobs)}')
CF = pd.concat([pd.read_csv(j / 'cf_samples.csv', usecols=['dataset', 'seed_idx', 'fold', 'sample', 'stratum',
                                                           'method', 'log_audc']) for j in jobs], ignore_index=True)
P = (CF[CF.method.isin(['z_norm', 'rho_raw'])]                       # z_norm = SDDA-z, rho_raw = TSA-adapted
     .pivot_table(index=['dataset', 'seed_idx', 'fold', 'sample', 'stratum'], columns='method', values='log_audc')
     .dropna())
P['delta'] = P.z_norm - P.rho_raw
U = P.groupby(['dataset', 'seed_idx', 'fold', 'stratum']).delta.mean().unstack()
t10 = pd.read_csv(args.pipeline / 'tables' / 'table_suppression_interaction.csv').set_index('dataset')
rows = []
for ds, g in U.groupby(level=0):
    d = (g['high'] - g['low']).dropna().values
    t, pc, half = corrected_t(d, R_TEST_TRAIN[ds])
    rows.append(dict(dataset=ds, units=len(d), delta_low=g['low'].mean(), delta_high=g['high'].mean(),
                     interaction=float(d.mean()), hodges_lehmann=hodges_lehmann(d), units_negative=int((d < 0).sum()),
                     units_positive=int((d > 0).sum()), ci95_low=float(d.mean() - half), ci95_high=float(d.mean() + half),
                     t_corrected=t, p_corrected=pc, p_wilcoxon=wilcoxon_p(d),
                     table10_interaction=float(t10.loc[ds, 'high_minus_low_interaction'])))
S13 = pd.DataFrame(rows)
hc = holm(dict(zip(S13.dataset, S13.p_corrected)))
hw = holm(dict(zip(S13.dataset, S13.p_wilcoxon)))
S13['p_corrected_holm'] = S13.dataset.map(hc)
S13['p_wilcoxon_holm'] = S13.dataset.map(hw)
S13['decision'] = [decide(r.p_corrected_holm, r.p_wilcoxon_holm, r.interaction, r.hodges_lehmann) for r in S13.itertuples()]
S13['note'] = 'negative interaction = SDDA-z relatively better (lower log-AUDC) under high suppression'
S13.to_csv(args.out / 'S13_table10_retest.csv', index=False)
say('\nS13:')
say(S13[['dataset', 'units', 'interaction', 'hodges_lehmann', 'units_negative', 'ci95_low', 'ci95_high',
         'p_wilcoxon_holm', 'p_corrected_holm', 'decision']].round(4).to_string(index=False))
for r in S13.itertuples():
    require(r.units == 30, f'{r.dataset}: expected 30 units')
    require(abs(r.interaction - r.table10_interaction) < 1e-6, f'{r.dataset}: interaction does not reproduce Table 10')
require(dict(zip(S13.dataset, S13.decision)) == {'ciciot2023': 'INCONSISTENT', 'edge_iiotset': 'NOT_SUPPORTED'},
        'S13 decisions differ from the manuscript')
# ============================================================ END S13

say('\nAll checks passed. Outputs in', args.out)
(args.out / 'summary.txt').write_text('\n'.join(LOG) + '\n')
