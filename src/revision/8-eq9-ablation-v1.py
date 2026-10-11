#!/usr/bin/env python3
"""Eq.(9) ablation 3x2 for SDDA revision (JKSUCIS) — v1 (final reviewed).

QUESTION (pre-specified analysis plan): does baseline-relative spike deviation add measurable
attribution value over raw spike activity when the weight-projection mechanism is identical?

VARIANTS (signal x projection), attribution_j = sum_i P_ji * signal_i, then rank-normalised:
  signal : rho = spike density | absdev = |rho - mu| | z = |rho - mu| / (sd + 1e-8)
  proj   : raw = |xbn_j * W_ji|  | norm = raw / sum_j raw   (per neuron i)
  rho_raw == TSA_adapted (anchor), z_norm == SDDA_z (anchor).
External comparators: RawError (recomputed), KernelSHAP + Random (stored arrays, identical samples).

PROTOCOL = paper3-faithfulness-v4-3x10 (notebook 3-4 v2). All perturbations/samples are REPLAYED
from the original per-job RNG and must equal the stored npz exactly. Anchors must reproduce
(100% of samples): stored attributions + top-3, perturbed errors, deletion/insertion curves, and
per-sample metrics in raw/faithfulness.csv and raw/counterfactual.csv. Any mismatch -> job FAILS.
No refitting: scaler, threshold, baseline mu/sd are loaded from final artefacts.

CHECKPOINT: a job counts as done only if done.json carries the CURRENT job-code hash
(hash of all computation functions + PROTOCOL). Stale jobs are archived to jobs/_stale/ and re-run.
Analysis code changes do not invalidate finished jobs.

USAGE (tmux, from 0-revisi-sdda-v1/):
  python3 8-eq9-ablation-v1.py --pilot                       # edge s0f0 + cic s0f0, parity/timing/RAM
  python3 8-eq9-ablation-v1.py --run --workers 6 --threads 4  # 60 jobs, resumable, final analysis at end
  python3 8-eq9-ablation-v1.py --status
  python3 8-eq9-ablation-v1.py --analyze                     # final: requires 60/60 valid jobs
  python3 8-eq9-ablation-v1.py --analyze --provisional       # partial -> tables_provisional/
Exit codes: 0 ok | 1 failed jobs | 2 incomplete.
"""
import os, sys


def _arg(name, default):
    if name in sys.argv:
        i = sys.argv.index(name)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


_T = str(int(_arg('--threads', '4')))
# Thread limits MUST be set before numpy / TF import (prevents oversubscription -> server hang).
os.environ.update({
    'OMP_NUM_THREADS': _T, 'OPENBLAS_NUM_THREADS': _T, 'MKL_NUM_THREADS': _T,
    'NUMEXPR_NUM_THREADS': _T, 'TF_NUM_INTRAOP_THREADS': _T, 'TF_NUM_INTEROP_THREADS': _T,
    'TF_CPP_MIN_LOG_LEVEL': '2', 'TF_ENABLE_ONEDNN_OPTS': '0', 'TF_DETERMINISTIC_OPS': '1',
    'TF_CUDNN_DETERMINISTIC': '1', 'CUDA_VISIBLE_DEVICES': '-1', 'PYTHONHASHSEED': '42',
})

import argparse, gc, hashlib, inspect, json, platform, resource, shutil, subprocess, time, traceback
from datetime import datetime
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd

VERSION = 'eq9-ablation-v1'
ROOT = Path('/home/jupyteruser/nix-ext-backup/0-eksperimen-final')
SRC = ROOT / 'paper3-faithfulness-v4-3x10'
V83 = ROOT / '0-snn-ae-unified-v8-3-fair-benchmark'
OUT_DEFAULT = ROOT / '0-revisi-sdda-v1' / 'eq9-ablation-v1'

PROTOCOL = {
    'source': str(SRC), 'v83': str(V83),
    'datasets': ['edge_iiotset', 'ciciot2023'], 'seeds': [0, 1, 2], 'folds': list(range(10)),
    'seed': 42, 'dim': 33, 'pb': 4096, 'faith_n': 100, 'm_features': 3,
    'magnitudes': [2.0, 4.0, 6.0], 'shap_bg': 50, 'cf_per_subtype': 10,
    'EPS': 1e-12, 'sd_eps': 1e-8, 'primary_magnitude': 6.0,
    # Parity: ALL samples must match (no fraction tolerance). Numeric tolerance only on values.
    'parity': {'atol_rank': 1e-6, 'rtol_err': 1e-4, 'atol_err': 1e-8, 'rtol_curve': 1e-4,
               'atol_curve': 1e-7, 'rtol_metric': 1e-6, 'atol_metric': 1e-9},
}
VARIANTS = {  # name: (signal, projection)
    'rho_raw': ('rho', 'raw'), 'rho_norm': ('rho', 'norm'),
    'absdev_raw': ('absdev', 'raw'), 'absdev_norm': ('absdev', 'norm'),
    'z_raw': ('z', 'raw'), 'z_norm': ('z', 'norm'),
}
ANCHORS = {'z_norm': 'SDDA_z', 'rho_raw': 'TSA_adapted', 'RawError': 'RawError'}
STORED_ONLY = ['KernelSHAP', 'Random']
ALL_METHODS = list(VARIANTS) + ['RawError'] + STORED_ONLY
MS_NAME = {**ANCHORS, 'KernelSHAP': 'KernelSHAP', 'Random': 'Random'}  # methods checked vs manuscript CSV

# +1 = higher is better, -1 = lower is better
DIRECTION = {'p3': 1, 'ndcg5': 1, 'feature_auc': 1, 'log_audc': -1, 'log_auic': 1,
             'restricted_deletion_k': -1, 'restricted_insertion_k': -1, 'deletion_flip': 1, 'insertion_flip': 1}

ANALYSIS_PLAN = {
    'status': 'pre-specified analysis plan (not an independent registration). Specified after the A1-A3 '
              'audit of existing v4-3x10 results and before any result of this ablation was observed.',
    'question': 'Does baseline-relative spike deviation add measurable attribution value over raw spike '
                'activity when the projection mechanism is identical?',
    'primary_family': [f'{ds}|{proto}|z_{pj} vs rho_{pj}' for ds in PROTOCOL['datasets']
                       for proto in ['p3_ctrl_m6', 'p3_donor'] for pj in ['norm', 'raw']],
    'primary_metric': 'P@3 (higher is better); controlled scaled_std magnitude 6.0 and empirical attack donor',
    'unit': 'paired (seed, fold), J = 30 per dataset; per-unit mean over samples',
    'tests': ['corrected resampled t-test (Nadeau & Bengio 2003), r = n_test / n_train_normal; a sensitivity '
              'analysis for resampling dependence, NOT an elimination of dependence between folds/seeds',
              'Wilcoxon signed-rank (two-sided) with Hodges-Lehmann estimate for direction'],
    'multiplicity': 'Holm over the 8 primary contrasts, applied separately to each test',
    'decision_rule': 'SUPPORTED only if both Holm-adjusted p < 0.05 AND mean difference and Hodges-Lehmann '
                     'estimate have the same sign; both significant but opposite signs -> INCONSISTENT_DIRECTION; '
                     'exactly one significant -> INCONSISTENT; neither -> NOT_SUPPORTED. '
                     'Winner label uses the metric direction (higher/lower is better).',
    'effect_size': 'mean paired difference; corrected-t 95% CI NOT adjusted for multiplicity (Holm is applied '
                   'to p-values only); Hodges-Lehmann estimate; units won of 30',
    'claim_levels': [
        'consistent benefit across the full design requires SUPPORTED delta-better in all 8 primary contrasts '
        '(2 datasets x 2 protocols x 2 projections)',
        'partial support -> claim limited to the dataset/protocol/projection conditions that pass',
        'secondary: delta and absdev both better than rho but delta vs absdev not significant -> baseline deviation '
        'shows benefit; the additional contribution of sigma scaling is not established (not: shown absent)',
        'no supported advantage -> formulation difference only; no demonstrated faithfulness benefit'],
    'secondary': ['mechanism contrasts among the 6 variants (+RawError, KernelSHAP, Random)',
                  'controlled magnitudes 2 and 4', 'detected-stratified P@3',
                  'deletion/insertion: log_audc(-), log_auic(+), restricted_deletion_k(-), '
                  'restricted_insertion_k(-), deletion_flip(+), insertion_flip(+)'],
    'secondary_not_conditions': 'AUDC/AUIC results are reported in full but are not conditions for primary claims',
    'interpretation': 'faithfulness with respect to the reconstruction score; not causal evidence about attacks; '
                      'score-based deletion/insertion structurally favours RawError',
    'final_evidence_requirement': 'all 60 jobs valid under one job-code hash',
}


# ----------------------------------------------------------------------------- utils
def now():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def sha_file(p, chunk=8 << 20):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


_SHA_CACHE = {}


def sha_cached(p):
    """sha256 cached per process by (path, size, mtime) — large dataset NPZ is hashed once per worker."""
    p = Path(p).resolve()
    st = p.stat()
    k = (str(p), st.st_size, st.st_mtime_ns)
    if k not in _SHA_CACHE:
        _SHA_CACHE[k] = sha_file(p)
    return _SHA_CACHE[k]


def source_files(ds, s, f):
    """All provenance inputs of one job (model artefacts, stored npz, dataset, split, reference tables)."""
    md = SRC / 'models' / ds / f'seed{s}_fold{f}'
    jd = SRC / 'jobs' / job_id(ds, s, f)
    files = {'model': md / 'model.keras', 'scaler': md / 'scaler.pkl', 'indices': md / 'indices.npz',
             'threshold': md / 'threshold.json',
             'baseline': SRC / 'baselines' / ds / f'seed{s}_fold{f}' / 'base' / 'baseline.npz',
             'counterfactual_curves': jd / 'counterfactual_curves.npz',
             'real_anomaly_attributions': jd / 'real_anomaly_attributions.npz',
             'faith_empirical_donor': jd / 'faith_empirical_donor.npz',
             'dataset_npz': SRC / 'data' / f'{ds}_balanced.npz',
             'split_npz': V83 / 'splits' / f'{ds}_holdout_eval_split.npz',
             'ref_faithfulness_csv': SRC / 'raw' / 'faithfulness.csv',
             'ref_counterfactual_csv': SRC / 'raw' / 'counterfactual.csv',
             'ref_split_audit_csv': SRC / 'raw' / 'split_audit.csv'}
    for mag in PROTOCOL['magnitudes']:
        files[f'faith_mag{mag}'] = jd / f'faith_mag{mag}.npz'
    return files


def sha_obj(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def savejson(p, obj):
    p = Path(p)
    q = p.with_suffix(p.suffix + '.tmp')
    q.write_text(json.dumps(obj, indent=2, default=_json_default, allow_nan=True))
    os.replace(q, p)


def mem_available_gb():
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) / 1024 ** 2
    except Exception:
        pass
    return float('nan')


def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2  # Linux: KB


def wait_for_memory(min_gb, tag=''):
    waited = 0
    while True:
        a = mem_available_gb()
        if not (a < min_gb):
            return
        if waited % 60 == 0:
            print(f'[{now()}]{tag} MemAvailable {a:.1f} GB < {min_gb} GB, waiting...', flush=True)
        time.sleep(10)
        waited += 10


class ParityError(RuntimeError):
    pass


def require(ok, msg, exc=RuntimeError):
    if not ok:
        raise exc(msg)


def job_id(ds, s, f):
    return f'{ds}_seed{s}_fold{f}'


def all_jobs(datasets=None):
    ds_list = datasets or PROTOCOL['datasets']
    return [(ds, s, f) for ds in ds_list for s in PROTOCOL['seeds'] for f in PROTOCOL['folds']]


# ----------------------------------------------------------------------------- TF setup (worker only)
def setup_tf(threads):
    import tensorflow as tf
    from tensorflow.keras.layers import Layer
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(threads)
    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()

    @tf.keras.utils.register_keras_serializable()
    class SpikingLayer(Layer):  # identical to notebook 3-4 v2 (needed for .keras loading)
        def __init__(self, units, threshold=0.5, decay=0.7, surrogate_scale=10.0, time_steps=3,
                     return_sequences=False, **kw):
            super().__init__(**kw)
            self.units = int(units); self.threshold = float(threshold); self.decay = float(decay)
            self.surrogate_scale = float(surrogate_scale); self.time_steps = int(time_steps)
            self.return_sequences = bool(return_sequences)

        def build(self, input_shape):
            self.input_dim = int(input_shape[-1])
            self.w = self.add_weight(shape=(self.input_dim, self.units),
                                     initializer=tf.keras.initializers.HeNormal(), trainable=True, name='weights')
            self.b = self.add_weight(shape=(self.units,), initializer='zeros', trainable=True, name='bias')
            super().build(input_shape)

        @tf.custom_gradient
        def spike_function(self, membrane, threshold):
            spikes = tf.cast(tf.greater(membrane, threshold), tf.float32)
            def grad(dy):
                sg = self.surrogate_scale / tf.square(self.surrogate_scale * tf.abs(membrane - threshold) + 1.0)
                return dy * sg, None
            return spikes, grad

        def call(self, inputs, training=None):
            bs = tf.shape(inputs)[0]
            membrane = tf.zeros((bs, self.units), dtype=inputs.dtype)
            spikes = []
            is_seq = len(inputs.shape) == 3
            for t in range(self.time_steps):
                cur_in = inputs[:, t, :] if is_seq else inputs
                cur = tf.matmul(cur_in, self.w) + self.b
                cur = cur / tf.sqrt(tf.cast(tf.shape(cur_in)[-1], tf.float32))
                membrane = self.decay * membrane + cur
                spk = self.spike_function(membrane, self.threshold)
                membrane = membrane - spk * self.threshold
                spikes.append(spk)
            allspk = tf.stack(spikes, axis=1)
            return allspk if self.return_sequences else tf.reduce_mean(allspk, axis=1)

        def get_config(self):
            c = super().get_config()
            c.update({'units': self.units, 'threshold': self.threshold, 'decay': self.decay,
                      'surrogate_scale': self.surrogate_scale, 'time_steps': self.time_steps,
                      'return_sequences': self.return_sequences})
            return c

    return tf, {'SpikingLayer': SpikingLayer, 'Custom>SpikingLayer': SpikingLayer}


# ----------------------------------------------------------------------------- protocol functions (copied from 3-4 v2)
def rn(a):
    from scipy.stats import rankdata
    return np.vstack([rankdata(x) / len(x) for x in a]).astype(np.float32)


def predict(m, x):
    return m.predict(x, batch_size=PROTOCOL['pb'], verbose=0)


def err(m, x):
    h = predict(m, x)
    fe = (x - h) ** 2
    return np.mean(fe, axis=1), fe


def ranking_metrics(a, truth):
    from sklearn.metrics import average_precision_score, roc_auc_score
    order = np.argsort(-a)
    pos = np.empty(len(a), int)
    pos[order] = np.arange(1, len(a) + 1)
    y = np.zeros(len(a), int)
    y[list(truth)] = 1
    k = min(5, len(a))
    dcg = sum(y[order[i]] / np.log2(i + 2) for i in range(k))
    ideal = sum(1 / np.log2(i + 2) for i in range(min(k, len(truth))))
    return {'p1': len(set(order[:1]) & set(truth)), 'p3': len(set(order[:3]) & set(truth)) / 3,
            'r5': len(set(order[:5]) & set(truth)) / len(truth),
            'mean_rank': np.mean([pos[j] for j in truth]), 'mrr': 1.0 / min(pos[j] for j in truth),
            'ndcg5': dcg / ideal, 'average_precision': average_precision_score(y, a),
            'feature_auc': roc_auc_score(y, a)}


def pick_sub(y, lab, det, n, rng):
    pick = []
    for st in np.unique(lab[y == 1]):
        cand = np.where(det & (lab == st))[0]
        take = min(n, len(cand))
        sel = rng.choice(cand, take, replace=False) if take else []
        pick.extend(sel)
    return np.asarray(pick, int)


def curves(m, x, order, ref):
    """Deletion/insertion score curves, identical to notebook; one predict per method per step."""
    n, d = x.shape
    de, ins = x.copy(), ref.copy()
    A = np.zeros((n, d + 1), np.float32)
    B = A.copy()
    A[:, 0] = err(m, de)[0]
    B[:, 0] = err(m, ins)[0]
    rr = np.arange(n)
    for k in range(d):
        cc = order[:, k]
        de[rr, cc] = ref[rr, cc]
        ins[rr, cc] = x[rr, cc]
        A[:, k + 1] = err(m, de)[0]
        B[:, k + 1] = err(m, ins)[0]
    return A, B


def curve_metrics(A, B, th):
    n, d1 = A.shape
    d = d1 - 1
    kd = np.full(n, np.nan)
    ki = np.full(n, np.nan)
    ki[B[:, 0] > th] = 0
    for k in range(d):
        z = np.isnan(kd) & (A[:, k + 1] <= th); kd[z] = k + 1
        z = np.isnan(ki) & (B[:, k + 1] > th); ki[z] = k + 1
    trap = getattr(np, 'trapezoid', None) or np.trapz
    ad = trap(np.log1p(A / th), axis=1) / d
    ai = trap(np.log1p(B / th), axis=1) / d
    return kd, ki, ad, ai


def variant_scores(m, bn, en, W, mu, sdv, x):
    S = predict(en, x).mean(1)
    xb = predict(bn, x)
    pe, e = err(m, x)
    c = np.abs(xb[:, :, None] * W[None, :, :])
    ch = c / (c.sum(1, keepdims=True) + PROTOCOL['EPS'])
    sig = {'rho': S, 'absdev': np.abs(S - mu), 'z': np.abs(S - mu) / (sdv + PROTOCOL['sd_eps'])}
    proj = {'raw': c, 'norm': ch}
    raw_attr, ranked = {}, {}
    for name, (sg, pj) in VARIANTS.items():
        a = np.einsum('nji,ni->nj', proj[pj], sig[sg])
        raw_attr[name] = a.astype(np.float32)
        ranked[name] = rn(a)
    ranked['RawError'] = rn(e)
    return ranked, raw_attr, {'S': S.astype(np.float32), 'xb': xb.astype(np.float32),
                              'e': e.astype(np.float32), 'pe': pe}


# ----------------------------------------------------------------------------- parity (strict: all samples)
def compare_rows(a, b, atol, rtol=0.0, top3=False):
    a, b = np.asarray(a), np.asarray(b)
    require(a.shape == b.shape, f'shape mismatch {a.shape} vs {b.shape}', ParityError)
    close = np.isclose(a, b, atol=atol, rtol=rtol)
    rows = close.all(axis=1) if a.ndim == 2 else close
    out = {'n_rows': int(len(rows)), 'rows_mismatch': int((~rows).sum()),
           'max_abs_diff': float(np.nanmax(np.abs(a.astype(float) - b.astype(float)))) if a.size else 0.0}
    if top3:
        ta = np.sort(np.argsort(-a, axis=1)[:, :3], axis=1)
        tb = np.sort(np.argsort(-b, axis=1)[:, :3], axis=1)
        out['top3_mismatch'] = int((~(ta == tb).all(axis=1)).sum())
    out['pass'] = out['rows_mismatch'] == 0 and out.get('top3_mismatch', 0) == 0
    return out


def check(par, key, stats):
    par[key] = stats
    require(stats['pass'], f'Parity failed: {key} {stats}', ParityError)


def metric_parity(par, key, new, ref, keys, exact, close):
    """Every new row must find its manuscript row; exact columns identical, close columns within tol."""
    PR = PROTOCOL['parity']
    kn = set(map(tuple, new[keys].itertuples(index=False, name=None)))
    kr = set(map(tuple, ref[keys].itertuples(index=False, name=None)))
    res = {'n_new': int(len(new)), 'n_ref': int(len(ref)), 'dup_new': int(new.duplicated(keys).sum()),
           'dup_ref': int(ref.duplicated(keys).sum()), 'keys_only_new': len(kn - kr), 'keys_only_ref': len(kr - kn)}
    if res['dup_new'] or res['dup_ref'] or res['keys_only_new'] or res['keys_only_ref'] or len(new) != len(ref):
        res['pass'] = False
        check(par, key, res)
    mm = new.merge(ref, on=keys, how='inner', suffixes=('', '_ms'), validate='one_to_one')
    res['matched'] = int(len(mm))
    for c in exact:
        a, b = mm[c], mm[c + '_ms']
        same = (a == b) | (a.isna() & b.isna())
        res[f'{c}_mismatch'] = int((~same).sum())
    for c in close:
        a, b = mm[c].astype(float).values, mm[c + '_ms'].astype(float).values
        same = np.isclose(a, b, rtol=PR['rtol_metric'], atol=PR['atol_metric'], equal_nan=True)
        res[f'{c}_mismatch'] = int((~same).sum())
    res['pass'] = res['matched'] == len(new) == len(ref) and all(v == 0 for k, v in res.items()
                                                                  if k.endswith('_mismatch'))
    check(par, key, res)


# ----------------------------------------------------------------------------- data
def load_pool(ds):
    z = np.load(SRC / 'data' / f'{ds}_balanced.npz', allow_pickle=True)
    cols = [str(c) for c in z['feature_cols']]
    require(len(cols) == PROTOCOL['dim'] and 'frame.time_epoch' not in cols, f'feature schema {ds}')
    ei = np.load(V83 / 'splits' / f'{ds}_holdout_eval_split.npz')['eval_idx']
    Xall = z['X']; X = Xall[ei]; del Xall
    yall = z['y']; y = yall[ei]; del yall
    lall = z['label_type']; lab = lall[ei]; del lall
    z.close(); gc.collect()
    return X, y, lab, cols


def load_refs(ds):
    """Manuscript per-sample metrics (v4-3x10 raw) for strict metric parity."""
    F = pd.read_csv(SRC / 'raw' / 'faithfulness.csv')
    F = F[(F.dataset == ds) & F.method.isin(set(MS_NAME.values()))].reset_index(drop=True)
    C = pd.read_csv(SRC / 'raw' / 'counterfactual.csv')
    C = C[(C.dataset == ds) & C.method.isin(set(MS_NAME.values()))].reset_index(drop=True)
    return F, C


# ----------------------------------------------------------------------------- job-code hash + checkpoint state
JOB_CODE_FUNCS = ['setup_tf', 'rn', 'predict', 'err', 'ranking_metrics', 'pick_sub', 'curves', 'curve_metrics',
                  'variant_scores', 'compare_rows', 'check', 'metric_parity', 'load_pool', 'load_refs', 'run_job']


_JOB_CODE_HASH = None


def job_code_hash():
    global _JOB_CODE_HASH
    if _JOB_CODE_HASH is not None:
        return _JOB_CODE_HASH
    g = globals()
    src = ''.join(inspect.getsource(g[n]) for n in JOB_CODE_FUNCS)
    _JOB_CODE_HASH = sha_obj({'src': src, 'protocol': PROTOCOL, 'variants': VARIANTS, 'anchors': ANCHORS,
                              'stored_only': STORED_ONLY, 'ms_name': MS_NAME})
    return _JOB_CODE_HASH


def job_state(out, ds, s, f):
    jid = job_id(ds, s, f)
    jd = out / 'jobs' / jid
    dj = jd / 'done.json'
    if dj.is_file():
        try:
            ok = json.loads(dj.read_text()).get('job_code_hash') == job_code_hash()
        except Exception:
            ok = False
        return 'done' if ok else 'stale'
    if (out / 'jobs' / (jid + '.failed.json')).exists():
        return 'failed'
    if (out / 'jobs' / (jid + '.partial')).exists() or jd.exists():
        return 'partial'
    return 'pending'


def is_done(out, ds, s, f):
    return job_state(out, ds, s, f) == 'done'


def archive_stale(out, jid):
    for name in [jid, jid + '.partial']:
        p = out / 'jobs' / name
        if p.exists():
            dst = out / 'jobs' / '_stale' / f'{name}__{datetime.now().strftime("%Y%m%dT%H%M%S")}'
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(dst))


def verify_metadata(out):
    """Locked plan + protocol must match this script (used by final analysis)."""
    problems = []
    ap, mp = out / 'analysis_plan.json', out / 'manifest.json'
    if not ap.is_file() or json.loads(ap.read_text()).get('plan_sha256') != sha_obj(ANALYSIS_PLAN):
        problems.append('analysis_plan.json missing or differs from this script')
    if not mp.is_file():
        problems.append('manifest.json missing')
    else:
        man = json.loads(mp.read_text())
        if man.get('version') != VERSION or man.get('protocol_hash') != sha_obj({'protocol': PROTOCOL,
                                                                                 'variants': VARIANTS}):
            problems.append('manifest version/protocol hash differs from this script')
    return problems


def verify_provenance(valid_dirs):
    """Re-hash every source file and compare with source_sha256 stored in each done.json."""
    problems = []
    for jd in valid_dirs:
        dn = json.loads((jd / 'done.json').read_text())
        sf = source_files(dn['dataset'], dn['seed_idx'], dn['fold'])
        stored = dn.get('source_sha256', {})
        for k, p in sf.items():
            if k not in stored:
                problems.append(f'{jd.name}: {k} not recorded')
            elif not p.is_file():
                problems.append(f'{jd.name}: {k} missing on disk')
            elif sha_cached(p) != stored[k]:
                problems.append(f'{jd.name}: {k} changed since job ran')
    return problems


def init_output(out):
    out.mkdir(parents=True, exist_ok=True)
    for d in ['jobs', 'logs', 'combined', 'tables']:
        (out / d).mkdir(exist_ok=True)
    ap = out / 'analysis_plan.json'
    if ap.exists():
        old = json.loads(ap.read_text())
        require(old.get('plan_sha256') == sha_obj(ANALYSIS_PLAN),
                'analysis_plan.json differs from this script. The plan is locked; refusing to run.')
    else:
        savejson(ap, {'plan': ANALYSIS_PLAN, 'plan_sha256': sha_obj(ANALYSIS_PLAN), 'created': now()})
    mp = out / 'manifest.json'
    ss = sha_file(Path(__file__).resolve())
    ph = sha_obj({'protocol': PROTOCOL, 'variants': VARIANTS})
    if mp.exists():
        man = json.loads(mp.read_text())
        require(man.get('version') == VERSION and 'protocol_hash' in man,
                f'manifest.json in {out} was not written by {VERSION}; use a new --out folder.')
        require(man['protocol_hash'] == ph, 'Output folder holds a different PROTOCOL/VARIANTS; use a new --out.')
        if ss not in man['script_sha256_seen']:
            man['script_sha256_seen'].append(ss)
            man.setdefault('script_log', []).append({'script_sha256': ss, 'job_code_hash': job_code_hash(),
                                                     'time': now()})
            savejson(mp, man)
    else:
        import sklearn, scipy
        man = {'version': VERSION, 'protocol_hash': ph, 'protocol': PROTOCOL, 'variants': VARIANTS,
               'script_sha256_seen': [ss], 'script_log': [{'script_sha256': ss, 'job_code_hash': job_code_hash(),
                                                           'time': now()}],
               'created': now(), 'host': platform.node(), 'python': platform.python_version(),
               'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__,
               'scipy': scipy.__version__,
               'source_tables_sha256': {p: sha_file(SRC / p) for p in
                                        ['raw/split_audit.csv', 'raw/faithfulness.csv', 'raw/counterfactual.csv']
                                        if (SRC / p).is_file()}}
        savejson(mp, man)
    return man


# ----------------------------------------------------------------------------- one job
def run_job(tf, custom, ds, s, f, data, out):
    import joblib
    X, y, lab, cols, Fref, Cref, loaded_sha = data
    P = PROTOCOL
    PR = P['parity']
    jid = job_id(ds, s, f)
    final = out / 'jobs' / jid
    if is_done(out, ds, s, f):
        return 'skip'
    archive_stale(out, jid)  # stale (other job-code hash) or leftover partial: archived, never deleted
    part = out / 'jobs' / (jid + '.partial')
    part.mkdir(parents=True)
    fail = out / 'jobs' / (jid + '.failed.json')
    if fail.exists():
        fail.unlink()
    T = {'start': time.time()}
    md = SRC / 'models' / ds / f'seed{s}_fold{f}'
    bd = SRC / 'baselines' / ds / f'seed{s}_fold{f}' / 'base' / 'baseline.npz'
    src_files = source_files(ds, s, f)
    for k, p in src_files.items():
        require(p.is_file(), f'missing source {k}: {p}')
    hashes = {k: sha_cached(p) for k, p in src_files.items()}
    for k, v in loaded_sha.items():  # in-memory data must come from the same file versions
        require(hashes[k] == v, f'{k} changed on disk after it was loaded by this worker')

    m = tf.keras.models.load_model(md / 'model.keras', custom_objects=custom, compile=False)
    require(m.input_shape[-1] == P['dim'], 'model dim')
    sc = joblib.load(md / 'scaler.pkl')
    th = float(json.loads((md / 'threshold.json').read_text())['threshold'])
    ix = np.load(md / 'indices.npz')
    tv, te, vl, tr = ix['tv'], ix['te'], ix['vl_local'], ix['tr_local']
    bz = np.load(bd)
    b = {k: bz[k] for k in bz.files}
    mu, sdv = b['mu_sdp'], b['sd_sdp']
    bn = tf.keras.Model(m.input, m.get_layer('batch_norm').output)
    en = tf.keras.Model(m.input, m.get_layer('encoder').output)
    W = m.get_layer('encoder').w.numpy()

    # ---- leakage / split audit (no fitting happens in this script)
    leak = {'trainval_test_overlap': int(np.intersect1d(tv, te).size),
            'train_val_overlap': int(np.intersect1d(tr, vl).size),
            'train_attacks': int((y[tv[tr]] != 0).sum()), 'val_attacks': int((y[tv[vl]] != 0).sum()),
            'baseline_n_val_equals_len_val': bool(int(b['n_val']) == len(vl)),
            'scaler_fit_scope': 'training_normal_only (loaded)', 'threshold_scope': 'validation_normal (loaded)',
            'baseline_scope': 'validation_normal (loaded)', 'donor_source': 'attack rows of train-val pool'}
    require(leak['trainval_test_overlap'] == 0 and leak['train_val_overlap'] == 0, f'split overlap {leak}')
    require(leak['train_attacks'] == 0 and leak['val_attacks'] == 0, f'attacks in train/val {leak}')
    require(leak['baseline_n_val_equals_len_val'], 'baseline n_val != len(validation)')

    xt = sc.transform(X[te]).astype(np.float32)
    yt, lt = y[te], lab[te]
    ae = err(m, xt)[0]
    T['detection'] = time.time()

    # ---- RNG replay (exact order of notebook run(): bg, fpick, truth, signs, donor, donor fs, cpick)
    rng = np.random.RandomState(P['seed'] + 1000 * s + f)
    bg_idx = rng.choice(len(vl), min(P['shap_bg'], len(vl)), replace=False)
    src_idx = np.where((yt == 0) & (ae <= th))[0]
    n = min(P['faith_n'], len(src_idx))
    fpick = rng.choice(src_idx, n, replace=False)
    fsd = np.asarray(b['feat_std'])
    valid = np.where(fsd > 1e-8)[0]
    truth = np.asarray([rng.choice(valid, P['m_features'], replace=False) for _ in range(n)])
    signs = rng.choice([-1.0, 1.0], (n, P['m_features']))
    donor_pool = np.where(y[tv] == 1)[0]
    donor_idx = rng.choice(donor_pool, n, replace=len(donor_pool) < n)
    leak['donor_test_overlap'] = int(np.intersect1d(tv[donor_idx], te).size)
    require(leak['donor_test_overlap'] == 0, 'donor rows overlap test rows')
    donors = sc.transform(X[tv[donor_idx]]).astype(np.float32)
    xd = xt[fpick].copy()
    donor_truth = []
    for i in range(n):
        cand = valid[np.abs(donors[i, valid] - xd[i, valid]) > 1e-8]
        require(len(cand) >= P['m_features'], 'insufficient donor features')
        fs = rng.choice(cand, P['m_features'], replace=False)
        donor_truth.append(fs)
        xd[i, fs] = donors[i, fs]
    donor_truth = np.asarray(donor_truth)
    det = (yt == 1) & (ae > th)
    cpick = pick_sub(yt, lt, det, P['cf_per_subtype'], rng)

    par = {}
    stored = {}
    for mag in P['magnitudes']:
        z = np.load(src_files[f'faith_mag{mag}'], allow_pickle=True)
        stored[f'mag{mag}'] = {k: z[k] for k in z.files}
        require(np.array_equal(z['test_local_idx'], fpick), f'fpick mismatch mag{mag}', ParityError)
        require(np.array_equal(z['truth'], truth), f'truth mismatch mag{mag}', ParityError)
    z = np.load(src_files['faith_empirical_donor'], allow_pickle=True)
    stored['donor'] = {k: z[k] for k in z.files}
    require(np.array_equal(z['test_local_idx'], fpick), 'donor fpick mismatch', ParityError)
    require(np.array_equal(z['donor_trainval_local_idx'], donor_idx), 'donor_idx mismatch', ParityError)
    require(np.array_equal(z['truth'], donor_truth), 'donor truth mismatch', ParityError)
    zc = np.load(src_files['counterfactual_curves'], allow_pickle=True)
    stored['cf_curves'] = {k: zc[k] for k in zc.files}
    za = np.load(src_files['real_anomaly_attributions'], allow_pickle=True)
    stored['cf'] = {k: za[k] for k in za.files}
    require(np.array_equal(zc['test_local_idx'], cpick), 'cpick mismatch (curves npz)', ParityError)
    require(np.array_equal(za['test_local_idx'], cpick), 'cpick mismatch (attributions npz)', ParityError)
    par['exact_index_replay'] = {'fpick': True, 'truth': True, 'donor_idx': True, 'donor_truth': True,
                                 'cpick': True, 'n_faith': int(n), 'n_cf': int(len(cpick)), 'pass': True}
    T['replay'] = time.time()

    arrays = {'th': np.float64(th), 'W': W.astype(np.float32), 'mu_sdp': mu, 'sd_sdp': sdv, 'feat_std': fsd,
              'fpick': fpick, 'eval_local_fpick': te[fpick], 'truth': truth, 'signs': signs, 'bg_idx': bg_idx,
              'donor_idx': donor_idx, 'donor_truth': donor_truth, 'cpick': cpick, 'eval_local_cpick': te[cpick]}
    frows = []

    def faith_block(key, mode, mag, xx, tru, st):
        ranked, raw_attr, aux = variant_scores(m, bn, en, W, mu, sdv, xx)
        for mine, theirs in ANCHORS.items():
            check(par, f'{key}:rank:{mine}=={theirs}',
                  compare_rows(ranked[mine], st[theirs], PR['atol_rank'], top3=True))
        check(par, f'{key}:perturbed_error',
              compare_rows(aux['pe'], st['perturbed_error'], PR['atol_err'], PR['rtol_err']))
        allm = dict(ranked)
        for k in STORED_ONLY:
            allm[k] = st[k]
        detected = aux['pe'] > th
        for meth in ALL_METHODS:
            A = allm[meth]
            src = 'stored_v4_3x10' if meth in STORED_ONLY else 'recomputed'
            for i in range(len(xx)):
                frows.append({'dataset': ds, 'seed_idx': s, 'fold': f, 'method': meth, 'source': src,
                              'perturbation_mode': mode, 'magnitude': mag, 'sample': i,
                              'test_local_idx': int(fpick[i]), 'eval_local_idx': int(te[fpick[i]]),
                              'detected': bool(detected[i]), 'perturbed_error': float(aux['pe'][i]),
                              'truth': ';'.join(map(str, tru[i])),
                              **ranking_metrics(A[i], set(map(int, tru[i])))})
        for k, v in aux.items():
            arrays[f'{key}__{k}'] = v
        for k, v in raw_attr.items():
            arrays[f'{key}__attr_{k}'] = v
        for k, v in allm.items():
            arrays[f'{key}__rank_{k}'] = np.asarray(v, np.float32)
        arrays[f'{key}__x'] = xx.astype(np.float32)

    for mag in P['magnitudes']:
        xp = xt[fpick].copy()
        for i, fs in enumerate(truth):
            xp[i, fs] += signs[i] * mag * fsd[fs]
        require(int(sum(np.sum(xp[i, fs] != xt[fpick[i], fs]) for i, fs in enumerate(truth))) == n * P['m_features'],
                'perturbation did not change all slots')
        faith_block(f'mag{mag}', 'scaled_std', mag, xp, truth, stored[f'mag{mag}'])
    faith_block('donor', 'empirical_attack_donor', 1.0, xd, donor_truth, stored['donor'])
    Fn = pd.DataFrame(frows)
    # per-sample metric parity vs manuscript raw/faithfulness.csv (anchors + stored comparators)
    Fa = Fn[Fn.method.isin(list(MS_NAME))].copy()
    Fa['method'] = Fa.method.map(MS_NAME)
    metric_parity(par, 'faith:metrics_vs_manuscript', Fa,
                  Fref[(Fref.seed_idx == s) & (Fref.fold == f)],
                  ['dataset', 'seed_idx', 'fold', 'method', 'perturbation_mode', 'magnitude', 'sample'],
                  exact=['test_local_idx', 'detected', 'p1', 'p3', 'r5'],
                  close=['mean_rank', 'mrr', 'ndcg5', 'average_precision', 'feature_auc'])
    T['faithfulness'] = time.time()

    # ---- counterfactual deletion / insertion (secondary): samples + references locked from stored npz
    ref = stored['cf_curves']['reference'].astype(np.float32)
    xa = xt[cpick]
    la = lt[cpick]
    require(np.all(err(m, ref)[0] <= th), 'reference not below threshold')
    ranked, raw_attr, aux = variant_scores(m, bn, en, W, mu, sdv, xa)
    for mine, theirs in ANCHORS.items():
        check(par, f'cf:rank:{mine}=={theirs}', compare_rows(ranked[mine], stored['cf'][theirs],
                                                             PR['atol_rank'], top3=True))
    allm = dict(ranked)
    for k in STORED_ONLY:
        allm[k] = stored['cf'][k]
    crows = []
    d = xa.shape[1]
    for meth in ALL_METHODS:
        if meth in STORED_ONLY:
            A = stored['cf_curves'][f'{meth}_deletion']; B = stored['cf_curves'][f'{meth}_insertion']
            src = 'stored_v4_3x10'
        else:
            A, B = curves(m, xa, np.argsort(-allm[meth], axis=1), ref)
            src = 'recomputed'
            if meth in ANCHORS:
                theirs = ANCHORS[meth]
                check(par, f'cf:curve_del:{meth}=={theirs}',
                      compare_rows(A, stored['cf_curves'][f'{theirs}_deletion'], PR['atol_curve'], PR['rtol_curve']))
                check(par, f'cf:curve_ins:{meth}=={theirs}',
                      compare_rows(B, stored['cf_curves'][f'{theirs}_insertion'], PR['atol_curve'], PR['rtol_curve']))
        kd, ki, ad, ai = curve_metrics(np.asarray(A, np.float32), np.asarray(B, np.float32), th)  # dtype as notebook
        arrays[f'cf__deletion_{meth}'] = np.asarray(A, np.float32)
        arrays[f'cf__insertion_{meth}'] = np.asarray(B, np.float32)
        for i in range(len(xa)):
            crows.append({'dataset': ds, 'seed_idx': s, 'fold': f, 'method': meth, 'source': src, 'sample': i,
                          'test_local_idx': int(cpick[i]), 'eval_local_idx': int(te[cpick[i]]),
                          'subtype': str(la[i]), 'stratum': str(stored['cf_curves']['stratum'][i]),
                          'suppression_ratio': float(stored['cf_curves']['suppression_ratio'][i]),
                          'deletion_flip': bool(np.isfinite(kd[i])), 'deletion_k': kd[i],
                          'restricted_deletion_k': kd[i] if np.isfinite(kd[i]) else d,
                          'insertion_flip': bool(np.isfinite(ki[i])), 'insertion_k': ki[i],
                          'restricted_insertion_k': ki[i] if np.isfinite(ki[i]) else d,
                          'log_audc': float(ad[i]), 'log_auic': float(ai[i]), 'max_k': d})
    Cn = pd.DataFrame(crows)
    Ca = Cn[Cn.method.isin(list(MS_NAME))].copy()
    Ca['method'] = Ca.method.map(MS_NAME)
    metric_parity(par, 'cf:metrics_vs_manuscript', Ca, Cref[(Cref.seed_idx == s) & (Cref.fold == f)],
                  ['dataset', 'seed_idx', 'fold', 'method', 'sample'],
                  exact=['test_local_idx', 'deletion_flip', 'insertion_flip', 'deletion_k', 'insertion_k',
                         'restricted_deletion_k', 'restricted_insertion_k'],
                  close=['log_audc', 'log_auic'])
    for k, v in aux.items():
        arrays[f'cf__{k}'] = v
    for k, v in raw_attr.items():
        arrays[f'cf__attr_{k}'] = v
    for k, v in allm.items():
        arrays[f'cf__rank_{k}'] = np.asarray(v, np.float32)
    arrays['cf__x'] = xa.astype(np.float32)
    arrays['cf__reference'] = ref
    arrays['cf__subtype'] = np.asarray(la).astype(str)
    T['counterfactual'] = time.time()

    # ---- write (atomic: .partial -> final; done.json last)
    Fn.to_csv(part / 'faith_samples.csv', index=False)
    Cn.to_csv(part / 'cf_samples.csv', index=False)
    np.savez_compressed(part / 'arrays.npz', **arrays)
    savejson(part / 'parity.json', par)
    savejson(part / 'leakage_audit.json', leak)
    timing = {k: round(v - T['start'], 2) for k, v in T.items() if k != 'start'}
    timing['total_s'] = round(time.time() - T['start'], 2)
    savejson(part / 'done.json', {'dataset': ds, 'seed_idx': s, 'fold': f, 'version': VERSION,
                                  'job_code_hash': job_code_hash(),
                                  'script_sha256': sha_file(Path(__file__).resolve()),
                                  'source_sha256': hashes, 'timing_s': timing, 'n_faith': int(n),
                                  'n_cf': int(len(cpick)), 'n_parity_checks': len(par),
                                  'all_parity_pass': all(v.get('pass', False) for v in par.values()),
                                  'peak_rss_gb': round(peak_rss_gb(), 2), 'completed': now(),
                                  'host': platform.node()})
    os.replace(part, final)
    del m, bn, en, xt, xa, xd, arrays, frows, crows, stored, Fn, Cn
    tf.keras.backend.clear_session()
    gc.collect()
    return timing


def run_jobs(out, ds, jobs, threads, min_free_gb, tag=''):
    tf, custom = setup_tf(threads)
    wait_for_memory(min_free_gb, tag)
    print(f'[{now()}]{tag} hashing + loading {ds} ...', flush=True)
    sf = source_files(ds, *jobs[0])
    loaded_sha = {k: sha_cached(sf[k]) for k in ['dataset_npz', 'split_npz', 'ref_faithfulness_csv',
                                                 'ref_counterfactual_csv', 'ref_split_audit_csv']}
    X, y, lab, cols = load_pool(ds)
    Fref, Cref = load_refs(ds)
    data = (X, y, lab, cols, Fref, Cref, loaded_sha)
    print(f'[{now()}]{tag} loaded {ds}: X{X.shape} | peak RSS {peak_rss_gb():.2f} GB | '
          f'MemAvailable {mem_available_gb():.1f} GB', flush=True)
    n_fail = 0
    for (s, f) in jobs:
        if is_done(out, ds, s, f):
            print(f'[{now()}]{tag} skip {job_id(ds, s, f)} (done, same job-code hash)', flush=True)
            continue
        wait_for_memory(min_free_gb / 2, tag)
        try:
            t = run_job(tf, custom, ds, s, f, data, out)
            print(f'[{now()}]{tag} done {job_id(ds, s, f)} timing={t} peak RSS {peak_rss_gb():.2f} GB',
                  flush=True)
        except Exception as e:
            n_fail += 1
            savejson(out / 'jobs' / (job_id(ds, s, f) + '.failed.json'),
                     {'error': repr(e), 'parity_error': isinstance(e, ParityError),
                      'traceback': traceback.format_exc(), 'time': now()})
            print(f'[{now()}]{tag} FAILED {job_id(ds, s, f)}: {e!r}', flush=True)
            tf.keras.backend.clear_session(); gc.collect()
    return n_fail


# ----------------------------------------------------------------------------- launcher
def status(out, quiet=False):
    rows = [{'dataset': ds, 'state': job_state(out, ds, s, f)} for ds, s, f in all_jobs()]
    t = pd.DataFrame(rows).groupby(['dataset', 'state']).size().unstack(fill_value=0)
    if not quiet:
        print(t.to_string())
    return t


def worker_cmd(out, ds, js, threads, min_free_gb, tag):
    return [sys.executable, str(Path(__file__).resolve()), '--worker', '--dataset', ds,
            '--jobs', ','.join(f'{s}:{f}' for s, f in js), '--threads', str(threads),
            '--min-free-gb', str(min_free_gb), '--out', str(out), '--tag', tag]


def pilot(args, out):
    targets = [('edge_iiotset', args.pilot_seed, args.pilot_fold), ('ciciot2023', args.pilot_seed, args.pilot_fold)]
    rc = 0
    for ds, s, f in targets:  # sequential, one dataset per process (isolated memory)
        if is_done(out, ds, s, f):
            print(f'[{now()}] pilot {job_id(ds, s, f)} already done (same job-code hash)', flush=True)
            continue
        p = subprocess.run(worker_cmd(out, ds, [(s, f)], args.threads, args.min_free_gb, f'pilot-{ds}'))
        rc = rc or p.returncode
    for ds, s, f in targets:
        jd = out / 'jobs' / job_id(ds, s, f)
        if not is_done(out, ds, s, f):
            print(f'\nPILOT FAILED {job_id(ds, s, f)} -> {out / "jobs" / (job_id(ds, s, f) + ".failed.json")}')
            rc = 1
            continue
        dn = json.loads((jd / 'done.json').read_text())
        par = json.loads((jd / 'parity.json').read_text())
        print(f'\n=== PILOT {job_id(ds, s, f)} ===')
        print(f"parity checks: {dn['n_parity_checks']} | all pass: {dn['all_parity_pass']} | "
              f"n_faith {dn['n_faith']} | n_cf {dn['n_cf']} | peak RSS {dn['peak_rss_gb']} GB")
        print('timing (cumulative s):', dn['timing_s'])
        worst = {k: v.get('max_abs_diff') for k, v in par.items() if 'max_abs_diff' in v}
        print('max |diff| per check:', json.dumps(worst, default=str))
        Fp = pd.read_csv(jd / 'faith_samples.csv')
        print(Fp.groupby(['perturbation_mode', 'magnitude', 'method']).p3.mean().unstack('method')
              .round(3).to_string())
    print(f'\nMemAvailable now {mem_available_gb():.1f} GB | CPUs {os.cpu_count()}')
    return rc


def launch(args, out):
    for ds, s, f in all_jobs(args.datasets):
        if job_state(out, ds, s, f) == 'stale':
            print(f'[{now()}] stale (other job-code hash): {job_id(ds, s, f)} -> will be archived and re-run')
    pending = [j for j in all_jobs(args.datasets) if not is_done(out, *j)]
    print(f'[{now()}] pending jobs: {len(pending)} | workers={args.workers} threads={args.threads} '
          f'| MemAvailable {mem_available_gb():.1f} GB | CPUs {os.cpu_count()}', flush=True)
    if args.workers * args.threads > (os.cpu_count() or 1):
        print(f'[warn] workers*threads = {args.workers * args.threads} > CPUs {os.cpu_count()}', flush=True)
    by_ds = {}
    for ds, s, f in pending:
        by_ds.setdefault(ds, []).append((s, f))
    require(args.workers >= 1, '--workers must be >= 1')
    if by_ds:
        # split each dataset into chunks (one dataset per process); total chunks == max(workers, n_datasets)
        dsl = list(by_ds)
        chunks = []
        for i, ds in enumerate(dsl):
            k = max(1, args.workers // len(dsl) + (1 if i < args.workers % len(dsl) else 0))
            k = min(k, len(by_ds[ds]))
            chunks += [(ds, by_ds[ds][w::k]) for w in range(k)]
        queue = list(enumerate(chunks))
        running, last = [], 0.0
        while queue or running:  # concurrency never exceeds --workers
            for r in running[:]:
                if r[2].poll() is not None:
                    r[3].close(); running.remove(r)
                    print(f'[{now()}] worker w{r[0]:02d} {r[1]} exit code {r[2].returncode}', flush=True)
            if queue and len(running) < args.workers and not (mem_available_gb() < args.min_free_gb):
                w, (ds, js) = queue.pop(0)
                log = open(out / 'logs' / f'worker{w:02d}_{ds}.log', 'a')
                p = subprocess.Popen(worker_cmd(out, ds, js, args.threads, args.min_free_gb, f'w{w:02d}'),
                                     stdout=log, stderr=subprocess.STDOUT)
                running.append((w, ds, p, log))
                print(f'[{now()}] started worker w{w:02d} {ds} jobs={len(js)} (running {len(running)}/'
                      f'{args.workers}) -> logs/worker{w:02d}_{ds}.log', flush=True)
                time.sleep(args.stagger)
                continue
            if time.time() - last >= 60:
                st = status(out, quiet=True)
                done = int(st['done'].sum()) if 'done' in st.columns else 0
                fails = int(st['failed'].sum()) if 'failed' in st.columns else 0
                print(f'[{now()}] progress {done}/{len(all_jobs())} done | failed {fails} | running '
                      f'{len(running)} | queued {len(queue)} | MemAvailable {mem_available_gb():.1f} GB', flush=True)
                last = time.time()
            time.sleep(5)
    st = status(out)
    if 'failed' in st.columns and st['failed'].sum() > 0:
        print('Some jobs FAILED. Inspect jobs/*.failed.json. Final analysis NOT run.')
        return 1
    if not all(is_done(out, *j) for j in all_jobs()):
        print('Not all 60 jobs are valid; final analysis NOT run. Re-run --run to resume.')
        return 2
    return analyze(out, provisional=False)


# ----------------------------------------------------------------------------- statistics
def holm(p):
    out, prev = {}, 0.0
    a = sorted(p.items(), key=lambda x: (1.0 if not np.isfinite(x[1]) else x[1]))
    for i, (k, v) in enumerate(a):
        v = 1.0 if not np.isfinite(v) else v
        prev = max(prev, (len(a) - i) * v)
        out[k] = min(1.0, prev)
    return out


def hodges_lehmann(d):
    d = np.asarray(d, float)
    if len(d) == 0:
        return np.nan
    i, j = np.triu_indices(len(d))
    return float(np.median((d[i] + d[j]) / 2.0))


def paired_stats(d, r):
    from scipy.stats import t as tdist, wilcoxon
    d = np.asarray(d, float)
    J = len(d)
    res = {'units': J, 'mean_diff': float(d.mean()) if J else np.nan, 'hodges_lehmann': hodges_lehmann(d),
           'wins_a': int((d > 0).sum()), 'wins_b': int((d < 0).sum()), 'ties': int((d == 0).sum())}
    if J < 3:
        res.update(p_corrected=np.nan, ci95_unadj_low=np.nan, ci95_unadj_high=np.nan, p_wilcoxon=np.nan)
        return res
    s2 = d.var(ddof=1)
    se = np.sqrt((1.0 / J + r) * s2)
    q = tdist.ppf(0.975, J - 1)
    if se == 0:
        p = 1.0 if d.mean() == 0 else 0.0
    else:
        p = float(2 * tdist.sf(abs(d.mean() / se), J - 1))
    try:
        pw = float(wilcoxon(d).pvalue)
    except Exception:
        pw = 1.0
    res.update(p_corrected=p, ci95_unadj_low=float(d.mean() - q * se), ci95_unadj_high=float(d.mean() + q * se), p_wilcoxon=pw)
    return res


def metric_direction(metric):
    base = metric.split('_ctrl_')[0].replace('_donor', '')
    require(base in DIRECTION, f'no direction defined for metric {metric}')
    return DIRECTION[base]


def decision(pc, pw, md, hl, direction):
    sc, sw = pc < 0.05, pw < 0.05
    if sc and sw:
        if md != 0 and np.sign(md) == np.sign(hl):
            return 'SUPPORTED_A_BETTER' if md * direction > 0 else 'SUPPORTED_B_BETTER'
        return 'INCONSISTENT_DIRECTION'
    if sc or sw:
        return 'INCONSISTENT'
    return 'NOT_SUPPORTED'


def run_level(F, C):
    K = ['dataset', 'seed_idx', 'fold', 'method']
    out = []
    ctrl = F[F.perturbation_mode == 'scaled_std']
    for mag, g in ctrl.groupby('magnitude'):
        a = g.groupby(K).agg(p3=('p3', 'mean'), ndcg5=('ndcg5', 'mean'), feature_auc=('feature_auc', 'mean')).reset_index()
        for col in ['p3', 'ndcg5', 'feature_auc']:
            out.append(a[K + [col]].rename(columns={col: 'value'}).assign(metric=f'{col}_ctrl_m{int(mag)}'))
    don = F[F.perturbation_mode == 'empirical_attack_donor']
    a = don.groupby(K).agg(p3=('p3', 'mean'), ndcg5=('ndcg5', 'mean'), feature_auc=('feature_auc', 'mean')).reset_index()
    for col in ['p3', 'ndcg5', 'feature_auc']:
        out.append(a[K + [col]].rename(columns={col: 'value'}).assign(metric=f'{col}_donor'))
    cagg = C.groupby(K).agg(log_audc=('log_audc', 'mean'), log_auic=('log_auic', 'mean'),
                            restricted_deletion_k=('restricted_deletion_k', 'mean'),
                            restricted_insertion_k=('restricted_insertion_k', 'mean'),
                            deletion_flip=('deletion_flip', 'mean'), insertion_flip=('insertion_flip', 'mean')).reset_index()
    for col in ['log_audc', 'log_auic', 'restricted_deletion_k', 'restricted_insertion_k', 'deletion_flip',
                'insertion_flip']:
        out.append(cagg[K + [col]].rename(columns={col: 'value'}).assign(metric=col))
    return pd.concat(out, ignore_index=True)


def contrast(RUN, ds, metric, a, b, r):
    g = RUN[(RUN.dataset == ds) & (RUN.metric == metric)]
    Pv = g.pivot_table(index=['seed_idx', 'fold'], columns='method', values='value')
    return paired_stats((Pv[a] - Pv[b]).dropna().values, r)


def add_decisions(rows, direction):
    hc = holm({i: x['p_corrected'] for i, x in enumerate(rows)})
    hw = holm({i: x['p_wilcoxon'] for i, x in enumerate(rows)})
    for i, x in enumerate(rows):
        x['p_corrected_holm'], x['p_wilcoxon_holm'] = hc[i], hw[i]
        dirn = direction(x) if callable(direction) else direction
        x['direction'] = 'higher_better' if dirn > 0 else 'lower_better'
        x['decision'] = decision(hc[i], hw[i], x['mean_diff'], x['hodges_lehmann'], dirn)
        a, bm = x['comparison'].split(' vs ')
        x['better_method'] = a if x['decision'] == 'SUPPORTED_A_BETTER' else (
            bm if x['decision'] == 'SUPPORTED_B_BETTER' else '')
    return rows


def analyze(out, provisional=False):
    print(f'[{now()}] analysis (provisional={provisional}) ...', flush=True)
    strict = not provisional
    jobs = all_jobs()
    states = {job_id(*j): job_state(out, *j) for j in jobs}
    valid = [out / 'jobs' / job_id(*j) for j in jobs if states[job_id(*j)] == 'done']
    problems = verify_metadata(out)
    if len(valid) != len(jobs):
        problems.append(f'{len(valid)}/{len(jobs)} valid jobs; states {pd.Series(states).value_counts().to_dict()}')
    if not valid:
        print('No valid jobs.', problems); return 2
    dns = {j.name: json.loads((j / 'done.json').read_text()) for j in valid}
    hashes = {d['job_code_hash'] for d in dns.values()}
    if len(hashes) != 1:
        problems.append(f'valid jobs carry different job-code hashes: {hashes}')
    print(f'[{now()}] verifying provenance of {len(valid)} jobs (hashing sources) ...', flush=True)
    problems += verify_provenance(valid)
    F = pd.concat([pd.read_csv(j / 'faith_samples.csv') for j in valid], ignore_index=True)
    C = pd.concat([pd.read_csv(j / 'cf_samples.csv') for j in valid], ignore_index=True)
    prow = []
    for j in valid:
        dn, par = dns[j.name], json.loads((j / 'parity.json').read_text())
        nf = int(((F.dataset == dn['dataset']) & (F.seed_idx == dn['seed_idx']) & (F.fold == dn['fold'])).sum())
        nc = int(((C.dataset == dn['dataset']) & (C.seed_idx == dn['seed_idx']) & (C.fold == dn['fold'])).sum())
        exp_f = len(ALL_METHODS) * (len(PROTOCOL['magnitudes']) + 1) * PROTOCOL['faith_n']
        exp_c = len(ALL_METHODS) * dn['n_cf']
        row = {'job': j.name, 'n_checks': len(par), 'checks_passed': sum(bool(v.get('pass')) for v in par.values()),
               'all_pass': all(bool(v.get('pass')) for v in par.values()) and bool(dn.get('all_parity_pass')),
               'max_abs_diff_any': max([v.get('max_abs_diff', 0) or 0 for v in par.values()] + [0]),
               'n_faith': dn['n_faith'], 'faith_rows': nf, 'faith_rows_expected': exp_f,
               'cf_rows': nc, 'cf_rows_expected': exp_c, 'peak_rss_gb': dn.get('peak_rss_gb'),
               'total_s': dn['timing_s']['total_s'], 'job_code_hash': dn['job_code_hash'][:16]}
        prow.append(row)
        if not row['all_pass']:
            problems.append(f'{j.name}: parity not all pass')
        if dn['n_faith'] != PROTOCOL['faith_n'] or nf != exp_f or nc != exp_c or dn['n_cf'] < 1:
            problems.append(f"{j.name}: row counts faith {nf}/{exp_f}, cf {nc}/{exp_c}, n_faith {dn['n_faith']}")
    PJ = pd.DataFrame(prow)
    if strict and problems:
        print('FINAL ANALYSIS REFUSED (no final tables written):')
        print('\n'.join(' - ' + x for x in problems[:50]))
        return 2
    tdir = out / ('tables' if strict else 'tables_provisional')
    cdir = out / ('combined' if strict else 'combined_provisional')
    tdir.mkdir(exist_ok=True); cdir.mkdir(exist_ok=True)
    if problems:
        (tdir / 'PROVISIONAL_PROBLEMS.txt').write_text('\n'.join(problems))
    PJ.to_csv(tdir / 'parity_per_job.csv', index=False)
    F.to_csv(cdir / 'faith_samples_all.csv.gz', index=False)
    C.to_csv(cdir / 'cf_samples_all.csv.gz', index=False)
    sa = pd.read_csv(SRC / 'raw' / 'split_audit.csv')
    r = (sa.n_test / sa.n_train_normal).groupby(sa.dataset).mean().to_dict()
    RUN = run_level(F, C)
    RUN.to_csv(tdir / 'run_level_values.csv', index=False)

    # PRIMARY: 8 pre-specified contrasts (P@3, higher is better)
    prim = []
    for ds in PROTOCOL['datasets']:
        for proto in ['p3_ctrl_m6', 'p3_donor']:
            for pj in ['norm', 'raw']:
                prim.append({'dataset': ds, 'metric': proto, 'projection': pj, 'comparison': f'z_{pj} vs rho_{pj}',
                             'r': r[ds], **contrast(RUN, ds, proto, f'z_{pj}', f'rho_{pj}', r[ds])})
    Pm = pd.DataFrame(add_decisions(prim, +1))
    bad = Pm[Pm.units != len(PROTOCOL['seeds']) * len(PROTOCOL['folds'])]
    if len(bad):
        msg = f'primary contrasts without 30 paired units: {bad[["dataset", "metric", "comparison", "units"]].to_dict("records")}'
        if strict:
            shutil.rmtree(tdir); shutil.rmtree(cdir)
            print('FINAL ANALYSIS REFUSED:', msg); return 2
        problems.append(msg); (tdir / 'PROVISIONAL_PROBLEMS.txt').write_text('\n'.join(problems))
    Pm.to_csv(tdir / 'primary_tests.csv', index=False)

    # SECONDARY: all pairs per dataset x metric, Holm within family, metric-aware direction
    sec = []
    for (ds, metric), g in RUN.groupby(['dataset', 'metric']):
        meths = [mm for mm in ALL_METHODS if mm in set(g.method)]
        rows = [{'dataset': ds, 'metric': metric, 'comparison': f'{a} vs {bm}',
                 **contrast(RUN, ds, metric, a, bm, r[ds])} for a, bm in combinations(meths, 2)]
        sec.extend(add_decisions(rows, metric_direction(metric)))
    S2 = pd.DataFrame(sec)
    S2.to_csv(tdir / 'secondary_all_pairs.csv', index=False)

    # SECONDARY: detected-stratified P@3 (descriptive means + mechanism contrasts, Holm within stratum)
    strat, means = [], []
    for (ds, mode, mag, detv), g in F.groupby(['dataset', 'perturbation_mode', 'magnitude', 'detected']):
        U = g.groupby(['seed_idx', 'fold', 'method']).p3.mean().unstack('method')
        means.append({'dataset': ds, 'mode': mode, 'magnitude': mag, 'detected': detv,
                      'n_samples': int((g.method == 'z_norm').sum()), 'units': len(U), **U.mean().to_dict()})
        rows = []
        for pj in ['norm', 'raw']:
            for a, bm in [(f'z_{pj}', f'rho_{pj}'), (f'absdev_{pj}', f'rho_{pj}'), (f'z_{pj}', f'absdev_{pj}')]:
                rows.append({'dataset': ds, 'mode': mode, 'magnitude': mag, 'detected': detv,
                             'comparison': f'{a} vs {bm}', **paired_stats((U[a] - U[bm]).dropna().values, r[ds])})
        strat.extend(add_decisions(rows, +1))
    pd.DataFrame(means).to_csv(tdir / 'detected_stratified_means.csv', index=False)
    pd.DataFrame(strat).to_csv(tdir / 'detected_stratified_tests.csv', index=False)
    VM = RUN.groupby(['dataset', 'metric', 'method']).value.agg(['mean', 'std', 'count']).reset_index()
    VM.to_csv(tdir / 'method_means.csv', index=False)

    pd.set_option('display.width', 230); pd.set_option('display.max_rows', 500)
    L = [f'eq9-ablation analysis {now()} | {"FINAL" if tdir.name == "tables" else "PROVISIONAL"} | '
         f'valid jobs {len(valid)}/{len(jobs)} | job-code hash {next(iter(hashes))[:16]} | r={r}',
         f'parity: {int(PJ.all_pass.sum())}/{len(PJ)} jobs with all checks passed '
         f'(checks per job: {PJ.n_checks.min()}-{PJ.n_checks.max()})',
         '\nPRIMARY (pre-specified; Holm over 8 on p-values; SUPPORTED needs both tests significant + same direction; '
         'CI = corrected-t 95% NOT adjusted for multiplicity):',
         Pm[['dataset', 'metric', 'comparison', 'mean_diff', 'ci95_unadj_low', 'ci95_unadj_high', 'hodges_lehmann', 'wins_a',
             'wins_b', 'p_corrected_holm', 'p_wilcoxon_holm', 'decision']].round(4).to_string(index=False),
         '\nMETHOD MEANS (run level):',
         VM[VM.metric.isin(['p3_ctrl_m6', 'p3_donor', 'log_audc', 'log_auic'])]
         .pivot_table(index=['dataset', 'metric'], columns='method', values='mean').round(4).to_string()]
    sup = Pm[Pm.decision == 'SUPPORTED_A_BETTER']
    L += [f'\nCLAIM SCOPE: delta-better SUPPORTED in {len(sup)}/8 primary contrasts '
          f'(full-design claim requires 8/8). Passing conditions: '
          + (', '.join(f"{x.dataset}|{x.metric}|{x.projection}" for x in sup.itertuples()) or 'none')]
    mech = S2[S2.comparison.isin([f'{a} vs {b}' for a, b in combinations(ALL_METHODS, 2)
                                  if a in VARIANTS and b in VARIANTS])]
    L += ['\nSECONDARY mechanism decisions among variants (count):',
          mech.groupby(['dataset', 'decision']).size().unstack(fill_value=0).to_string()]
    txt = '\n'.join(L)
    (tdir / 'summary.txt').write_text(txt)
    print(txt, flush=True)
    print(f'\nTables: {tdir}', flush=True)
    return 0 if tdir.name == 'tables' else 2


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=str(OUT_DEFAULT))
    ap.add_argument('--pilot', action='store_true')
    ap.add_argument('--pilot-seed', type=int, default=0)
    ap.add_argument('--pilot-fold', type=int, default=0)
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--status', action='store_true')
    ap.add_argument('--analyze', action='store_true')
    ap.add_argument('--provisional', action='store_true')
    ap.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--dataset', help=argparse.SUPPRESS)
    ap.add_argument('--jobs', help=argparse.SUPPRESS)
    ap.add_argument('--tag', default='', help=argparse.SUPPRESS)
    ap.add_argument('--datasets', nargs='+', choices=PROTOCOL['datasets'], default=None)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--min-free-gb', type=float, default=6.0)
    ap.add_argument('--stagger', type=int, default=45, help='seconds between worker starts (dataset loading)')
    args = ap.parse_args()
    out = Path(args.out).resolve()
    require(not str(out).startswith(str(SRC.resolve())), 'Output must not be inside the source folder.')

    if args.status:
        status(out); return 0
    if args.analyze:
        return analyze(out, provisional=args.provisional)
    init_output(out)
    if args.worker:
        jobs = [tuple(map(int, x.split(':'))) for x in args.jobs.split(',') if x]
        return 1 if run_jobs(out, args.dataset, jobs, args.threads, args.min_free_gb, f' [{args.tag}]') else 0
    if args.pilot:
        return pilot(args, out)
    if args.run:
        return launch(args, out)
    ap.print_help()
    return 0


if __name__ == '__main__':
    sys.exit(main())
