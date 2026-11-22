"""Diagnostic: fraction of MS2 intensity explained by subformulae of the true formula, per library."""
import re, sys, numpy as np, pandas as pd
M = dict(C=12.0, H=1.00782503207, N=14.0030740048, O=15.9949146196, P=30.97376163, S=31.97207100,
         F=18.99840322, Cl=34.96885268, Br=78.9183371, I=126.904473)
E = 0.00054857990
def parse(f):
    d = {}
    for el, n in re.findall(r'([A-Z][a-z]?)(\d*)', f):
        if el not in M: return None
        d[el] = d.get(el, 0) + (int(n) if n else 1)
    return d
def submasses(d, maxm):
    els = [e for e in d if e != 'H']
    grids = [np.arange(d[e] + 1) * M[e] for e in els]
    tot = np.zeros(1)
    for g in grids:
        tot = (tot[:, None] + g[None, :]).ravel(); tot = tot[tot <= maxm + 1]
    hs = np.arange(d.get('H', 0) + 3) * M['H']
    tot = (tot[:, None] + hs[None, :]).ravel()
    return np.unique(np.round(tot[tot <= maxm + 3], 5))
data = np.load('/home/user/data/trainlib_peaks.npz'); MZ, IT, OFF = data['mz'], data['it'], data['off']
meta = pd.read_parquet('/home/user/data/trainlib_meta.parquet', columns=['ingest_lib','molecular_formula','adduct','precursor_mz','instrument_type'])
rng = np.random.default_rng(0)
rows = []
for lib, g in meta[meta.adduct.isin(['[M+H]+','[M-H]-'])].groupby('ingest_lib', observed=True):
    idx = rng.choice(g.index.values, size=min(400, len(g)), replace=False)
    for i in idx:
        f = parse(str(meta.molecular_formula[i]));
        if not f: continue
        p = float(meta.precursor_mz[i]); mz = MZ[OFF[i]:OFF[i+1]].astype(np.float64); it = IT[OFF[i]:OFF[i+1]].astype(np.float64)
        if len(mz) < 3: continue
        sm = submasses(f, p)
        sign = 1 if meta.adduct[i] == '[M+H]+' else -1
        # ion = neutral subformula (with any H count) -/+ electron (H count free covers +H/-H)
        ions = np.sort(sm - sign * E)
        j = np.searchsorted(ions, mz); tol = np.maximum(10e-6 * mz, 0.003)
        d1 = np.abs(ions[np.clip(j, 0, len(ions)-1)] - mz); d0 = np.abs(ions[np.clip(j-1, 0, len(ions)-1)] - mz)
        ok = np.minimum(d0, d1) <= tol
        rows.append(dict(lib=lib, instr=str(meta.instrument_type[i]), npk=len(mz), frac_int=it[ok].sum()/it.sum(), frac_pk=ok.mean()))
r = pd.DataFrame(rows)
print(r.groupby('lib')[['npk','frac_int','frac_pk']].median().round(3))
print(r.groupby('lib')[['frac_int']].quantile(0.1).round(3))
