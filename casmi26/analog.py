"""Query-centric analog propagation over the train spectral library (CPU, numba).

For a query molecule (its spectra) and a neutral target mass M:
  1. representatives: one spectrum per (structure, polarity, library) with the most peaks;
  2. shifted similarity between the merged query spectrum of each polarity and every representative of the same
     polarity within +-WIN Da: peaks match directly OR after shifting by delta = M - mass(rep) (modified cosine,
     greedy one-to-one, sqrt intensities);
  3. ap(c) = max_a Tanimoto(c, a) * sim(a)^POW for a candidate c (also returned: top analogs).

Exclusion hooks let a simulation hide the query structure's own spectra (class-2 situation).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit, prange

TOL = 0.01


@njit(cache=True)
def _shift_cos(qm, qi, rm, ri, shift, tol):
    """Greedy modified cosine; qi, ri already sqrt-scaled and L2-normalised."""
    nq, nr = len(qm), len(rm)
    if nq == 0 or nr == 0:
        return 0.0, 0
    # candidate pairs
    cap = 4 * (nq + nr) + 16
    ps = np.empty(cap); pi = np.empty(cap, np.int64); pj = np.empty(cap, np.int64)
    n = 0
    for s_ix in range(2):
        sh = 0.0 if s_ix == 0 else shift
        if s_ix == 1 and abs(shift) < tol:
            break
        j0 = 0
        for i in range(nq):
            target = qm[i] - sh
            while j0 < nr and rm[j0] < target - tol:
                j0 += 1
            j = j0
            while j < nr and rm[j] <= target + tol:
                if n < cap:
                    ps[n] = qi[i] * ri[j]; pi[n] = i; pj[n] = j; n += 1
                j += 1
    if n == 0:
        return 0.0, 0
    order = np.argsort(-ps[:n])
    usedq = np.zeros(nq, np.bool_); usedr = np.zeros(nr, np.bool_)
    tot = 0.0; k = 0
    for o in order:
        a = pi[o]; b = pj[o]
        if usedq[a] or usedr[b]:
            continue
        usedq[a] = True; usedr[b] = True; tot += ps[o]; k += 1
    return tot, k


@njit(parallel=True, cache=True)
def _batch(qm, qi, offs, mz, it, idxs, shifts, tol, min_match):
    out = np.zeros(len(idxs))
    for t in prange(len(idxs)):
        r = idxs[t]
        a, b = offs[r], offs[r + 1]
        s, k = _shift_cos(qm, qi, mz[a:b], it[a:b], shifts[t], tol)
        out[t] = s if k >= min_match else 0.0
    return out


def _norm(it):
    v = np.sqrt(np.maximum(it, 0))
    return v / (np.linalg.norm(v) + 1e-12)


def merge_spectra(spectra, tol=0.01, top=150):
    """Union of peaks across spectra (bin by tol), max-normalised intensities averaged."""
    mz = np.concatenate([s[0] for s in spectra]); it = np.concatenate([s[1] / max(s[1].max(), 1e-12) for s in spectra])
    o = np.argsort(mz); mz, it = mz[o], it[o]
    out_m, out_i = [], []
    i = 0
    while i < len(mz):
        j = i
        while j + 1 < len(mz) and mz[j + 1] - mz[i] <= tol:
            j += 1
        w = it[i:j + 1]
        out_m.append(float((mz[i:j + 1] * w).sum() / w.sum())); out_i.append(float(w.sum()))
        i = j + 1
    out_m = np.asarray(out_m); out_i = np.asarray(out_i)
    if len(out_m) > top:
        s = np.argsort(-out_i)[:top]; s.sort(); out_m, out_i = out_m[s], out_i[s]
    return out_m, out_i / max(out_i.max(), 1e-12)


class AnalogIndex:
    def __init__(self, peaks_npz, meta_parquet, struct_mass: dict, win=200.0, smiles_of: dict | None = None):
        self.smiles_of = smiles_of or {}
        self._fpc = {}
        d = np.load(peaks_npz)
        self.mz, self.it_raw, self.off = d['mz'].astype(np.float64), d['it'], d['off']
        import pyarrow.parquet as _pq
        cols = ['ingest_lib', 'inchikey14', 'ionization_mode', 'adduct', 'precursor_mz']
        has_src = 'src' in _pq.read_schema(meta_parquet).names
        meta = pd.read_parquet(meta_parquet, columns=cols + (['src'] if has_src else []))
        if not has_src:
            meta['src'] = meta.ingest_lib.astype(str)
        meta['n'] = np.diff(self.off)
        meta['mode'] = np.where(meta.ionization_mode.astype(str) == 'positive', 1, -1)
        meta['mass'] = meta.inchikey14.map(struct_mass)
        meta = meta[meta.mass.notna() & (meta.n >= 3)]
        # representative: most peaks per (structure, mode, lib)
        meta = meta.sort_values('n', ascending=False).drop_duplicates(['inchikey14', 'mode', 'src'])
        meta = meta.sort_values('mass')
        self.rep = meta.index.values.astype(np.int64)
        self.rep_key = meta.inchikey14.values.astype(str)
        self.rep_mass = meta.mass.values.astype(np.float64)
        self.rep_mode = meta['mode'].values.astype(np.int8)
        self.rep_lib = meta.ingest_lib.astype(str).values
        self.rep_src = meta.src.astype(str).values
        # sqrt-normalised intensities per spectrum (in place, float64)
        it = np.empty(len(self.it_raw), np.float64)
        for r in self.rep:
            a, b = self.off[r], self.off[r + 1]
            it[a:b] = _norm(self.it_raw[a:b].astype(np.float64))
        self.it = it
        self.win = win

    def fp_of_key(self, key):
        if key not in self._fpc:
            from rdkit import Chem
            from rdkit.Chem import rdFingerprintGenerator
            gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
            m = Chem.MolFromSmiles(self.smiles_of.get(key, ''))
            self._fpc[key] = gen.GetFingerprintAsNumPy(m).astype(bool) if m is not None else np.zeros(2048, bool)
        return self._fpc[key]

    def search(self, spectra, target, exclude_keys=(), exclude_libs=(), min_match=3, top=100, exclude_ext_keys=(), exclude_ext_src=()):
        """spectra: list of (mz, it, mode). Returns DataFrame of best analog per structure: key, sim, shift."""
        best = {}
        lo = np.searchsorted(self.rep_mass, target - self.win); hi = np.searchsorted(self.rep_mass, target + self.win)
        sel = np.arange(lo, hi)
        if exclude_keys:
            sel = sel[~np.isin(self.rep_key[sel], list(exclude_keys))]
        if exclude_libs:
            sel = sel[~np.isin(self.rep_lib[sel], list(exclude_libs))]
        if exclude_ext_keys:          # offline pseudo-test: hide the query structure's external spectra only
            sel = sel[~(np.isin(self.rep_key[sel], list(exclude_ext_keys)) & np.isin(self.rep_lib[sel], ['ext', 'mbh']))]
        if exclude_ext_src:           # leave-source-out: hide every external spectrum from the query's institution/library
            sel = sel[~(np.isin(self.rep_src[sel], list(exclude_ext_src)) & np.isin(self.rep_lib[sel], ['ext', 'mbh']))]
        for md in (1, -1):
            grp = [(m, i) for m, i, mm in spectra if mm == md]
            if not grp:
                continue
            qm, qi = merge_spectra(grp)
            qi = _norm(qi)
            s = sel[self.rep_mode[sel] == md]
            if len(s) == 0:
                continue
            shifts = target - self.rep_mass[s]
            sc = _batch(qm, qi, self.off, self.mz, self.it, self.rep[s], shifts, TOL, min_match)
            for j in np.argsort(-sc)[:4 * top]:
                if sc[j] <= 0:
                    break
                k = self.rep_key[s[j]]
                if sc[j] > best.get(k, (0, 0))[0]:
                    best[k] = (float(sc[j]), float(shifts[j]))
        items = sorted(best.items(), key=lambda x: -x[1][0])[:top]
        return pd.DataFrame([(k, v[0], v[1]) for k, v in items], columns=['key', 'sim', 'shift'])


def tanimoto_matrix(A, B):
    """A (n, nbits) bool, B (m, nbits) bool -> (n, m) Tanimoto."""
    A = A.astype(np.float32); B = B.astype(np.float32)
    inter = A @ B.T
    return inter / (A.sum(1)[:, None] + B.sum(1)[None, :] - inter + 1e-9)


def ap_scores(cand_fp, analogs: pd.DataFrame, fp_of_key, power=4.0):
    """cand_fp (n, nbits) bool; returns dict of feature arrays."""
    n = len(cand_fp)
    if analogs is None or len(analogs) == 0:
        z = np.zeros(n)
        return dict(ap=z, ap_sum=z, ap_top_tan=z, ap_best_tan=z, ap_top_sim=np.zeros(n))
    afp = np.stack([fp_of_key(k) for k in analogs.key])
    tan = tanimoto_matrix(cand_fp, afp)
    w = analogs.sim.values ** power
    return dict(ap=(tan * w[None]).max(1), ap_sum=np.log1p((tan * w[None]).sum(1)), ap_top_tan=tan[:, 0],
                ap_best_tan=tan.max(1), ap_top_sim=np.full(n, analogs.sim.values[0]))
