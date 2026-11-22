"""Formula plausibility from MS2: fraction of fragment intensity explained by subformulae of a candidate formula.

Ions are neutral subformulae with any H count 0..H+2, shifted by the electron mass for the ion polarity; tolerance
max(ppm * m/z, abs_tol). Peaks within 0.5 Da of the precursor are ignored (the precursor itself explains nothing).
"""
import re

import numpy as np

M = dict(C=12.0, H=1.00782503207, N=14.0030740048, O=15.9949146196, P=30.97376163, S=31.97207100, F=18.99840322,
         Cl=34.96885268, Br=78.9183371, I=126.904473, Si=27.9769265325, B=11.0093054, Se=79.9165213)
E = 0.00054857990


def parse(f):
    d = {}
    for el, n in re.findall(r'([A-Z][a-z]?)(\d*)', str(f)):
        if el not in M:
            return None
        d[el] = d.get(el, 0) + (int(n) if n else 1)
    return d or None


def submasses(d, maxm):
    tot = np.zeros(1)
    for e in d:
        if e == 'H':
            continue
        g = np.arange(d[e] + 1) * M[e]
        tot = (tot[:, None] + g[None, :]).ravel()
        tot = tot[tot <= maxm + 1]
    hs = np.arange(d.get('H', 0) + 3) * M['H']
    tot = (tot[:, None] + hs[None, :]).ravel()
    return np.unique(tot[tot <= maxm + 3])


def explained(formula, spectra, ppm=5.0, abs_tol=0.002):
    """spectra: list of (mz, it, prec, sign). Mean explained-intensity fraction over spectra (nan if none usable)."""
    d = parse(formula)
    if d is None:
        return np.nan
    vals = []
    for mz, it, prec, sign in spectra:
        ok = mz < prec - 0.5
        if ok.sum() < 2:
            continue
        m, i = mz[ok], it[ok]
        ions = np.sort(submasses(d, prec) - sign * E)
        j = np.searchsorted(ions, m)
        tol = np.maximum(ppm * 1e-6 * m, abs_tol)
        d1 = np.abs(ions[np.clip(j, 0, len(ions) - 1)] - m)
        d0 = np.abs(ions[np.clip(j - 1, 0, len(ions) - 1)] - m)
        vals.append(i[np.minimum(d0, d1) <= tol].sum() / i.sum())
    return float(np.mean(vals)) if vals else np.nan


def spectra_of(test_rows):
    """test_rows: DataFrame rows of one molecule -> list for explained()."""
    out = []
    for r in test_rows.itertuples():
        out.append((np.asarray(r.ms2_mzs, float), np.asarray(r.ms2_normalized_intensities, float), float(r.precursor_mz),
                    1 if r.ionization_mode == 'positive' else -1))
    return out
