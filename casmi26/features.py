"""Candidate features for the learned final merger.

Works on the output directory of a run of the 0.500 stack (scoring rerun or pseudo-test run) and is used unchanged
offline and inside the submission notebook.

Feature groups (per candidate c of molecule m):
  pipeline  final rank, frozen-0.465 rank, channel ranks (main / engine2 / pubchem / prefusion), ICEBERG and
            GLACIER similarity, neural scorer outputs, molecule-level library maximum
  analog    query-centric analog propagation over the train library (analog.py)
  prior     PubChem documentation arrays (substances, PubMed, patents, MeSH, synonyms, depositors, vendors,
            natural-product bits, depositor-group bits, LOTUS references, formula isomer count) by InChIKey-14
  struct    in train, formula equals leader formula, mass error, heavy atoms
All within-molecule comparisons are added as z-scores and gaps to the molecule maximum.
"""
from __future__ import annotations

import json
import math
import os
import pickle
import re
from functools import lru_cache

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog('rdApp.*')
_TE = rdMolStandardize.TautomerEnumerator()
_FPG = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
ADD = {'[M+H]+': 1.007276, '[M+NH4]+': 18.033823, '[M-H2O+H]+': 1.007276 - 18.010565, '[M-2H2O+H]+': 1.007276 - 36.02113,
       '[M+Na]+': 22.989218, '[M+K]+': 38.963158, '[M-H]-': -1.007276, '[M-H2O-H]-': -1.007276 - 18.010565,
       '[M+CH2O2-H]-': 44.998201, '[M+Cl]-': 34.969402}
PRIOR_F16 = ['lsid', 'lpmid', 'lpatent', 'lmesh', 'lsynonyms']
PRIOR_INT = ['n_depositors', 'n_vendors', 'np', 'src_mask', 'lotus_refs', 'formula_isomers']


@lru_cache(maxsize=400000)
def mol_info(s):
    """(metric key, standard ik14, formula, exact mass, n_heavy, fp bool array) or None."""
    m = Chem.MolFromSmiles(str(s))
    if m is None:
        return None
    try:
        mk = Chem.MolToInchiKey(_TE.Canonicalize(m))[:14]
        sk = Chem.MolToInchiKey(m)[:14]
    except Exception:
        return None
    return (mk, sk, rdMolDescriptors.CalcMolFormula(m), ExactMolWt(m), m.GetNumHeavyAtoms(),
            _FPG.GetFingerprintAsNumPy(m).astype(bool))


class PubChemPrior:
    """Memory-mapped PubChem tier arrays; lookup of documentation features by mass window and InChIKey-14."""

    def __init__(self, pop_dir):
        f = lambda n: np.load(os.path.join(pop_dir, n), mmap_mode='r')
        mass_path = os.path.join(pop_dir, 'pc_mass.npy')
        if not os.path.exists(mass_path):
            mass_path = os.path.join(pop_dir, 'casmi26-pubchem-tier', 'pc_mass.npy')
        self.mass = np.load(mass_path, mmap_mode='r')
        self.ik14 = f('pc_ik14.npy')
        self.arr = {n: f(f'pc_{n}.npy') for n in PRIOR_F16 + PRIOR_INT}

    def lookup(self, target, keys, ppm=40.0):
        """keys: iterable of standard InChIKey-14 strings. Returns {key: dict(features)} for keys found."""
        tol = target * ppm * 1e-6 + 0.01
        a, b = np.searchsorted(self.mass, target - tol), np.searchsorted(self.mass, target + tol)
        if b <= a:
            return {}
        win = np.asarray(self.ik14[a:b])
        want = {k.encode(): k for k in keys if k}
        out = {}
        for j in np.flatnonzero(np.isin(win, list(want))):
            k = want[win[j]]
            row = a + j
            d = {n: float(self.arr[n][row]) for n in PRIOR_F16 + PRIOR_INT}
            if k not in out or d['lsid'] > out[k]['lsid']:
                out[k] = d
        return out


def molecule_targets(test):
    t = test.copy()
    t['nm'] = t.precursor_mz.astype(float) - t.adduct.map(ADD)
    return t.groupby('molecule_id').nm.median().to_dict()


def _z(x):
    x = np.asarray(x, float)
    s = np.nanstd(x)
    return (x - np.nanmean(x)) / s if s > 1e-9 else np.zeros_like(x)


def build_features(run_dir, test, train_keys, prior: PubChemPrior | None, analog_index=None, extra_lists=None,
                   max_extra=0):
    """run_dir: stack output dir. test: test.parquet DataFrame. train_keys: set of train standard ik14.
    Returns a candidate DataFrame (one row per molecule x candidate metric key)."""
    sub = pd.read_csv(os.path.join(run_dir, 'submission.csv'), dtype=str, keep_default_na=False)
    final = {m: [s for s in t.split(';') if s] for m, t in zip(sub.molecule_id, sub.smiles)}
    ev = pd.read_parquet(os.path.join(run_dir, 'frozen_candidate_evidence.parquet')) \
        if os.path.exists(os.path.join(run_dir, 'frozen_candidate_evidence.parquet')) else pd.DataFrame()
    nf_path = os.path.join(run_dir, 'neural_candidate_features.parquet')
    nf = pd.read_parquet(nf_path) if os.path.exists(nf_path) else pd.DataFrame()
    pc = json.load(open(os.path.join(run_dir, 'pc_lists.json'))) if os.path.exists(os.path.join(run_dir, 'pc_lists.json')) else {}
    eng = json.load(open(os.path.join(run_dir, 'eng_lists.json'))) if os.path.exists(os.path.join(run_dir, 'eng_lists.json')) else {}
    union = pickle.load(open(os.path.join(run_dir, 'union_channels.pkl'), 'rb')) \
        if os.path.exists(os.path.join(run_dir, 'union_channels.pkl')) else {}
    base = union.get('BASE') or {}
    ice_all, gl_all = union.get('ICE') or {}, union.get('GL') or {}
    targets = molecule_targets(test)
    nspec = test.groupby('molecule_id').size().to_dict()
    npos = test.assign(p=test.ionization_mode == 'positive').groupby('molecule_id').p.sum().to_dict()
    evg = {m: g for m, g in ev.groupby('molecule_id')} if len(ev) else {}
    nfg = {m: g for m, g in nf.groupby('molecule_id')} if len(nf) else {}
    rows = []
    for m, flist in final.items():
        target = targets.get(m, np.nan)
        cands = []                        # (smiles, source tag)
        for s in flist:
            cands.append((s, 'final'))
        extra = []
        if max_extra:
            for s in (base.get(m, ([],))[0] if base.get(m) else [])[:60]:
                extra.append(s)
            extra += (eng.get(m) or {}).get('smiles', [])[:40]
            extra += (pc.get(m) or {}).get('pc', [])[:25]
        for s in extra:
            cands.append((s, 'extra'))
        # dedupe by metric key, keep first
        seen, uniq = set(), []
        for s, tag in cands:
            info = mol_info(s)
            if info is None or info[0] in seen:
                continue
            seen.add(info[0]); uniq.append((s, tag, info))
        n_final = sum(1 for _, t, _ in uniq if t == 'final')
        if max_extra:
            ext = [u for u in uniq if u[1] == 'extra'][:max_extra]
            uniq = [u for u in uniq if u[1] == 'final'] + ext
        if not uniq:
            continue
        # channel ranks by metric key
        def rmap(lst):
            out = {}
            for r, s in enumerate(lst, 1):
                info = mol_info(s)
                if info is not None:
                    out.setdefault(info[0], r)
            return out
        r_final = rmap(flist)
        r_main = rmap(base.get(m, ([],))[0]) if base.get(m) else {}
        r_eng = rmap((eng.get(m) or {}).get('smiles', []))
        r_pc = rmap((pc.get(m) or {}).get('pc', []))
        pc_fz = {}
        for s, v in zip((pc.get(m) or {}).get('pc', []), (pc.get(m) or {}).get('pc_fz', [])):
            info = mol_info(s)
            if info is not None and v is not None:
                pc_fz.setdefault(info[0], float(v))
        main_sc = {}
        if base.get(m):
            for s, v in zip(base[m][0], base[m][3]):
                info = mol_info(s)
                if info is not None:
                    main_sc.setdefault(info[0], float(v))
        e = evg.get(m)
        e_by = {}
        if e is not None:
            for r in e.itertuples(index=False):
                info = mol_info(r.smiles)
                if info is not None:
                    e_by[info[0]] = r
        n_by = {}
        g = nfg.get(m)
        if g is not None:
            for r in g.itertuples(index=False):
                info = mol_info(r.smiles)
                if info is not None:
                    n_by[info[0]] = r
        lib_max = float(e['library_max'].iloc[0]) if e is not None and len(e) else 0.0
        leader_formula = uniq[0][2][2]
        prior_map = prior.lookup(target, [u[2][1] for u in uniq]) if (prior is not None and np.isfinite(target)) else {}
        ice_m = ice_all.get(m, {}) or {}
        gl_m = gl_all.get(m, {}) or {}
        for s, tag, info in uniq:
            mk, sk, formula, mass, nheavy, fp = info
            er = e_by.get(mk); nr = n_by.get(mk)
            ice = getattr(er, 'ice', np.nan) if er is not None else ice_m.get(s, np.nan)
            gl = getattr(er, 'gl', np.nan) if er is not None else gl_m.get(s, np.nan)
            row = dict(molecule_id=m, smiles=s, key=mk, sk=sk, tag=tag, formula=formula,
                       final_rank=r_final.get(mk, 0), frozen_rank=int(getattr(er, 'baseline_rank', 0)) if er is not None else 0,
                       main_rank=r_main.get(mk, int(getattr(er, 'main_rank', 0)) if er is not None else 0),
                       eng_rank=r_eng.get(mk, 0), pc_rank=r_pc.get(mk, 0),
                       prefusion_rank=int(getattr(er, 'prefusion_rank', 0)) if er is not None else 0,
                       main_score=main_sc.get(mk, np.nan), pc_fz=pc_fz.get(mk, np.nan),
                       ice=float(ice) if ice is not None else np.nan, gl=float(gl) if gl is not None else np.nan,
                       neural_score=float(getattr(nr, 'neural_score', np.nan)) if nr is not None else np.nan,
                       neural_cos=float(getattr(nr, 'neural_cos', np.nan)) if nr is not None else np.nan,
                       neural_fp=float(getattr(nr, 'neural_fp', np.nan)) if nr is not None else np.nan,
                       lib_max=lib_max, in_train=int(sk in train_keys), same_formula=int(formula == leader_formula),
                       ppm=(mass - target) / target * 1e6 if np.isfinite(target) else np.nan, n_heavy=nheavy,
                       n_spec=nspec.get(m, 0), n_pos=npos.get(m, 0), n_final=n_final, target=target)
            p = prior_map.get(sk)
            row['in_pubchem'] = int(p is not None)
            for n in PRIOR_F16 + PRIOR_INT:
                row['pr_' + n] = p[n] if p is not None else 0.0
            npb = int(row['pr_np']); srcb = int(row['pr_src_mask'])
            row['np_coconut'], row['np_lotus'], row['np_npatlas'] = npb & 1, (npb >> 1) & 1, (npb >> 2) & 1
            for b in range(9):
                row[f'src_b{b}'] = (srcb >> b) & 1
            row['_fp'] = fp
            rows.append(row)
    D = pd.DataFrame(rows)
    if analog_index is not None and len(D):
        D = add_analog_features(D, test, analog_index)
    return add_relative_features(D)


def add_analog_features(D, test, idx, fp_of_key=None):
    from analog import ap_scores
    targets = molecule_targets(test)
    parts = []
    for m, g in D.groupby('molecule_id', sort=False):
        t = test[test.molecule_id == m]
        spectra = [(np.asarray(r.ms2_mzs, float), np.asarray(r.ms2_normalized_intensities, float),
                    1 if r.ionization_mode == 'positive' else -1) for r in t.itertuples()]
        spectra = [s for s in spectra if len(s[0]) >= 3]
        an = idx.search(spectra, targets[m]) if spectra else None
        f = ap_scores(np.stack(g['_fp'].values), an, idx.fp_of_key) if an is not None else {}
        g = g.copy()
        for k, v in f.items():
            g[k] = v
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def add_relative_features(D):
    if not len(D):
        return D
    D = D.copy()
    D['log_final_rank'] = np.log(np.where(D.final_rank > 0, D.final_rank, 40))
    for c in ['frozen_rank', 'main_rank', 'eng_rank', 'pc_rank', 'prefusion_rank']:
        D['log_' + c] = np.log(np.where(D[c] > 0, D[c], 80))
    zcols = ['ice', 'gl', 'neural_score', 'neural_cos', 'neural_fp', 'main_score', 'pc_fz', 'pr_lsid', 'pr_lpmid',
             'pr_lpatent', 'pr_lmesh', 'pr_lsynonyms', 'ap', 'ap_sum', 'ap_best_tan']
    for c in zcols:
        if c not in D:
            continue
        D[c + '_z'] = D.groupby('molecule_id')[c].transform(lambda x: pd.Series(_z(x.values), index=x.index))
        D[c + '_gap'] = D[c] - D.groupby('molecule_id')[c].transform('max')
    D['pr_pop'] = D.pr_lsid + D.pr_lpmid
    D['pr_vendors'] = np.log1p(D.pr_n_vendors)
    D['pr_deps'] = np.log1p(D.pr_n_depositors)
    D['pr_lotus'] = np.log1p(D.pr_lotus_refs)
    D['pr_iso'] = np.log1p(D.pr_formula_isomers)
    D['pr_pop_gap'] = D.pr_pop - D.groupby('molecule_id').pr_pop.transform('max')
    D['frac_formula'] = D.groupby(['molecule_id', 'formula']).key.transform('size') / D.groupby('molecule_id').key.transform('size')
    return D.drop(columns=['_fp'])
