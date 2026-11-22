"""PubChem-tier recall channel: every tier structure within +-ppm of the query's neutral mass, scored by query-centric
analog propagation (train library), formula plausibility from MS2 and PubChem documentation priors.

Shared by offline evaluation/training (eval_pc_channel.py, train_channel.py) and the Kaggle merger runner, so features
are computed by one code path. Tier arrays are found anywhere under `pop_root` (memory-mapped).
"""
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors

from analog import ap_scores
from formula_expl import explained, parse

RDLogger.DisableLog('rdApp.*')
PRIORS = ['lsid', 'lpmid', 'lpatent', 'lmesh', 'lsynonyms', 'n_vendors', 'n_depositors', 'np', 'src_mask', 'lotus_refs',
          'formula_isomers']
HAND = ['s_pop', 's_ap', 's_ap_pop', 's_ap_pop_np', 's_ap_pop_np_fx']
# conservative learned feature set: no depositor-source bits and no documentation counts beyond lsid (pseudo-test truths
# all have public spectra, which inflates those fields)
FEATS = ['ap', 'ap_sum', 'ap_top_tan', 'ap_best_tan', 'ap_top_sim', 'abs_ppm', 'fexpl', 'fexpl_gap', 'n_formulas', 'n_heavy',
         'pr_lsid', 'np_coconut', 'np_lotus', 'np_npatlas', 'in_train', 'ap_z', 'ap_gap', 'pop_z', 'pop_gap', 'ap_rank',
         'pop_rank', 'n_window_log', 'fdelta', 'fdelta_gap']
G = {}


def _find(root, name):
    hits = []
    for r, _, files in os.walk(root, followlinks=True):
        if name in files:
            hits.append(os.path.join(r, name))
    if not hits:
        raise FileNotFoundError(name)
    return sorted(hits, key=len)[0]


def init(pop_root, analog_index, train_keys, formula_of=None):
    G['fpg'] = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    for n in ['mass', 'off', 'smiles', 'ik14'] + PRIORS:
        G[n] = np.load(_find(pop_root, f'pc_{n}.npy'), mmap_mode='r')
    G['idx'] = analog_index
    G['train_keys'] = train_keys
    G['formula_of'] = formula_of or {}


def _cx(fa, fb):
    a, b = parse(fa), parse(fb)
    if a is None or b is None:
        return 99.0
    return sum(abs(a.get(k, 0) - b.get(k, 0)) * (0.5 if k == 'H' else 1.0) for k in set(a) | set(b))


def fdelta(formulas, an, top=10):
    """Per candidate formula: max over the top analogs of sim^4 * exp(-delta_complexity/2), where delta_complexity counts
    atoms changed between candidate and analog formula (H half weight): simple modifications (+O, +CH2) score high."""
    if an is None or not len(an) or not G.get('formula_of'):
        return {f: 0.0 for f in formulas}
    a = an.head(top)
    af = [G['formula_of'].get(k, '') for k in a.key]
    w = a.sim.values ** 4
    return {f: float(max(wi * np.exp(-_cx(f, x) / 2.0) for x, wi in zip(af, w))) for f in formulas}


def _z(x):
    x = np.asarray(x, float); s = x.std()
    return (x - x.mean()) / s if s > 1e-9 else np.zeros_like(x)


def window(args):
    """args: (molecule_id, neutral mass, spectra [(mz,it,mode)], fspec [(mz,it,prec,sign)], ppm, keep[, force_ik14]).
    force_ik14 (offline evaluation only) keeps that structure's row whatever its scores. Returns (info, DataFrame|None)."""
    mid, target, spectra, fspec, ppm, keep = args[:6]
    force = args[6] if len(args) > 6 else None
    ex_src = args[7] if len(args) > 7 else ()
    t0 = time.time()
    lo = int(np.searchsorted(G['mass'], target * (1 - ppm * 1e-6)))
    hi = int(np.searchsorted(G['mass'], target * (1 + ppm * 1e-6)))
    info = dict(molecule_id=mid, n_window=hi - lo)
    if hi <= lo:
        return info, None
    rows = np.arange(lo, hi)
    off, buf = G['off'], G['smiles']
    smis = [bytes(buf[off[r]:off[r + 1]]).decode() for r in rows]
    fps, ok, fml, nh = [], np.zeros(len(rows), bool), [''] * len(rows), np.zeros(len(rows), int)
    for i, s in enumerate(smis):
        m = Chem.MolFromSmiles(s)
        if m is None:
            fps.append(np.zeros(2048, bool)); continue
        fps.append(G['fpg'].GetFingerprintAsNumPy(m).astype(bool)); ok[i] = True
        fml[i] = rdMolDescriptors.CalcMolFormula(m); nh[i] = m.GetNumHeavyAtoms()
    fp = np.stack(fps)
    an = G['idx'].search(spectra, target, exclude_ext_keys=(force,) if force else (),
                                  exclude_ext_src=ex_src) if spectra else None
    f = ap_scores(fp, an, G['idx'].fp_of_key)
    D = pd.DataFrame(dict(molecule_id=mid, row=rows, ik14=np.asarray(G['ik14'][lo:hi]).astype(str), smiles=smis,
                          formula=fml, n_heavy=nh, ppm=(np.asarray(G['mass'][lo:hi]) - target) / target * 1e6))
    for k, v in f.items():
        D[k] = v
    for p in PRIORS:
        D['pr_' + p] = np.asarray(G[p][lo:hi]).astype(float)
    D = D[ok].copy()
    fx = {fm: explained(fm, fspec) for fm in D.formula.unique()}
    D['fexpl'] = D.formula.map(fx).astype(float)
    D['fexpl_gap'] = D.fexpl - np.nanmax(D.fexpl.values) if D.fexpl.notna().any() else np.nan
    D['n_formulas'] = len(fx)
    D['fdelta'] = D.formula.map(fdelta(list(fx), an)).astype(float)
    pop = D.pr_lsid.values
    D['s_pop'] = pop
    D['s_ap'] = D.ap.values
    D['s_ap_pop'] = _z(D.ap.values) + 0.4 * _z(pop)
    D['s_ap_pop_np'] = D.s_ap_pop + 0.5 * ((D.pr_np.values.astype(int) & 3) > 0)
    D['s_ap_pop_np_fx'] = D.s_ap_pop_np + 5.0 * np.nan_to_num(D.fexpl_gap.values, nan=0.0)
    D = D.sort_values('pr_lsid', ascending=False).drop_duplicates('ik14')
    keep_idx = set()
    for s in HAND:
        keep_idx |= set(D.nlargest(keep, s).index)
    if force is not None:
        keep_idx |= set(D.index[D.ik14 == force])
        info['truth_in_window'] = bool((D.ik14 == force).any())
        for s in HAND:
            r = D[s].rank(ascending=False, method='first')[D.ik14 == force]
            info['rank_' + s] = float(r.min()) if len(r) else np.nan
    info.update(n_unique=len(D), n_analogs=0 if an is None else len(an),
                top_sim=float(an.sim.iloc[0]) if an is not None and len(an) else 0.0, sec=round(time.time() - t0, 1))
    return info, D.loc[sorted(keep_idx)].assign(n_window=hi - lo)


def add_rel(D):
    """Derived per-molecule features (identical offline and on Kaggle)."""
    D = D.copy()
    D['abs_ppm'] = D.ppm.abs()
    npb = D.pr_np.values.astype(np.int64)
    D['np_coconut'] = npb & 1; D['np_lotus'] = (npb >> 1) & 1; D['np_npatlas'] = (npb >> 2) & 1
    D['in_train'] = D.ik14.isin(G.get('train_keys', set())).astype(int)
    g = D.groupby('molecule_id')
    D['ap_z'] = (D.ap - g.ap.transform('mean')) / (g.ap.transform('std') + 1e-9)
    D['ap_gap'] = D.ap - g.ap.transform('max')
    D['pop_z'] = (D.pr_lsid - g.pr_lsid.transform('mean')) / (g.pr_lsid.transform('std') + 1e-9)
    D['pop_gap'] = D.pr_lsid - g.pr_lsid.transform('max')
    D['ap_rank'] = np.log1p(g.ap.rank(ascending=False))
    D['pop_rank'] = np.log1p(g.pr_lsid.rank(ascending=False))
    D['n_window_log'] = np.log1p(D.n_window)
    if 'fdelta' in D:
        D['fdelta_gap'] = D.fdelta - D.groupby('molecule_id').fdelta.transform('max')
    return D


def jobs_of(test, targets, ppm=10.0, keep=300):
    jobs = []
    for m, t in test.groupby('molecule_id'):
        if m not in targets or not np.isfinite(targets[m]):
            continue
        spectra = [(np.asarray(r.ms2_mzs, float), np.asarray(r.ms2_normalized_intensities, float),
                    1 if r.ionization_mode == 'positive' else -1) for r in t.itertuples()]
        spectra = [s for s in spectra if len(s[0]) >= 3]
        fspec = [(np.asarray(r.ms2_mzs, float), np.asarray(r.ms2_normalized_intensities, float), float(r.precursor_mz),
                  1 if r.ionization_mode == 'positive' else -1) for r in t.itertuples()]
        jobs.append((m, float(targets[m]), spectra, fspec, float(ppm), int(keep)))
    return jobs


def run_windows(jobs, procs=4, log=print):
    infos, parts, t0 = [], [], time.time()
    with Pool(procs) as pool:
        for i, (info, D) in enumerate(pool.imap_unordered(window, jobs, chunksize=2)):
            infos.append(info)
            if D is not None:
                parts.append(D)
            if i % 100 == 0:
                log(f'channel {i}/{len(jobs)} molecules {time.time() - t0:.0f}s')
    return pd.DataFrame(infos), (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame())


def score(D, model_json):
    import lightgbm as lgb
    cfg = json.loads(model_json) if isinstance(model_json, str) else model_json
    D = add_rel(D)
    D['ch_score'] = lgb.Booster(model_str=cfg['model']).predict(D[cfg['feats']].astype(float))
    D['ch_rank'] = D.groupby('molecule_id').ch_score.rank(ascending=False, method='first')
    return D
