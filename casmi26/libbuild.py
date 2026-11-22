"""Compact CSR spectral library from train.parquet (importable; same arithmetic as build_lib.py)."""
import re
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

K, FLOOR = 100, 0.002
_M = dict(C=12.0, H=1.00782503207, N=14.0030740048, O=15.9949146196, P=30.97376163, S=31.97207100, F=18.99840322,
          Cl=34.96885268, Br=78.9183371, I=126.904473, Si=27.9769265325, B=11.0093054, Se=79.9165213, Na=22.98977,
          K=38.96371)


def formula_mass(f):
    tot = 0.0
    for el, n in re.findall(r'([A-Z][a-z]?)(\d*)', str(f)):
        if el not in _M:
            return np.nan
        tot += _M[el] * (int(n) if n else 1)
    return tot


def build(train_path, out_prefix, log=print):
    pf = pq.ParquetFile(train_path)
    cols = ['ingest_lib', 'inchikey14', 'normalized_smiles', 'molecular_formula', 'ionization_mode', 'adduct', 'precursor_mz']
    mzs, its, lens, metas = [], [], [], []
    t0 = time.time()
    for rg in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(rg, columns=cols + ['ms2_mzs', 'ms2_normalized_intensities'])
        mzl = t.column('ms2_mzs').to_pylist(); itl = t.column('ms2_normalized_intensities').to_pylist()
        prec = t.column('precursor_mz').to_numpy()
        for m, i, p in zip(mzl, itl, prec):
            m = np.asarray(m, np.float64); i = np.asarray(i, np.float64)
            ok = (m > 0) & (m <= p + 2.0) & np.isfinite(i) & (i > 0)
            m, i = m[ok], i[ok]
            if len(i):
                i = i / i.max(); ok = i >= FLOOR; m, i = m[ok], i[ok]
                if len(i) > K:
                    s = np.argpartition(-i, K)[:K]; m, i = m[s], i[s]
                o = np.argsort(m); m, i = m[o], i[o]
            mzs.append(m.astype(np.float32)); its.append(i.astype(np.float32)); lens.append(len(m))
        metas.append(t.select(cols).to_pandas())
        del t, mzl, itl
    off = np.zeros(len(lens) + 1, np.int64); off[1:] = np.cumsum(lens)
    np.savez(out_prefix + '_peaks.npz', mz=np.concatenate(mzs), it=np.concatenate(its), off=off)
    meta = pd.concat(metas, ignore_index=True)
    for c in ['ingest_lib', 'ionization_mode', 'adduct']:
        meta[c] = meta[c].astype('category')
    meta.to_parquet(out_prefix + '_meta.parquet')
    log(f'library {len(meta):,} spectra {off[-1]:,} peaks {time.time() - t0:.0f}s')
    return meta
