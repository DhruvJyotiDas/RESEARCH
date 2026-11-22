"""Compact CSR spectral library from train.parquet (importable; same arithmetic as build_lib.py)."""
import os
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


def src_token(s):
    """Source institution / library token: last ':'-field, lower case; MULTIPLEX partitions collapse to one."""
    s = pd.Series(s).astype(str).str.split(':').str[-1].str.lower()
    return s.where(~s.str.startswith('multiplex'), 'multiplex').values


def build_extref(train_prefix, out_prefix, extlib_dir=None, mbh_path=None, log=print):
    """Train library + external reference spectra (extlib GNPS-full, MassBank-harmonized LC-ESI MS2) in one CSR library
    with the same meta columns, for analog search. External rows get ingest_lib 'ext' / 'mbh' (one representative per
    structure, mode and source in AnalogIndex)."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolDescriptors
    RDLogger.DisableLog('rdApp.*')
    t0 = time.time()
    d = np.load(train_prefix + '_peaks.npz')
    mzs, its, lens = [d['mz']], [d['it']], [np.diff(d['off'])]
    metas = [pd.read_parquet(train_prefix + '_meta.parquet')]
    for c in ['ingest_lib', 'ionization_mode', 'adduct']:
        metas[0][c] = metas[0][c].astype(str)
    metas[0]['src'] = metas[0].ingest_lib

    def add(meta, mz_list, it_list):
        ms, is_, ls = [], [], []
        for m, i, p in zip(mz_list, it_list, meta.precursor_mz.values):
            m = np.asarray(m, np.float64); i = np.asarray(i, np.float64)
            ok = (m > 0) & (m <= p + 2.0) & np.isfinite(i) & (i > 0)
            m, i = m[ok], i[ok]
            if len(i):
                i = i / i.max(); ok = i >= FLOOR; m, i = m[ok], i[ok]
                if len(i) > K:
                    s = np.argpartition(-i, K)[:K]; m, i = m[s], i[s]
                o = np.argsort(m); m, i = m[o], i[o]
            ms.append(m.astype(np.float32)); is_.append(i.astype(np.float32)); ls.append(len(m))
        mzs.append(np.concatenate(ms) if ms else np.zeros(0, np.float32))
        its.append(np.concatenate(is_) if is_ else np.zeros(0, np.float32)); lens.append(np.array(ls, np.int64))
        metas.append(meta)

    fcache = {}

    def formula(s):
        if s not in fcache:
            m = Chem.MolFromSmiles(str(s))
            fcache[s] = rdMolDescriptors.CalcMolFormula(m) if m is not None else ''
        return fcache[s]

    if extlib_dir:
        em = pd.read_parquet(os.path.join(extlib_dir, 'ext_meta.parquet'))
        EMZ = np.load(os.path.join(extlib_dir, 'ext_mz.npy'), mmap_mode='r')
        EIN = np.load(os.path.join(extlib_dir, 'ext_in.npy'), mmap_mode='r')
        meta = pd.DataFrame(dict(ingest_lib='ext', inchikey14=em.inchikey14.astype(str), normalized_smiles=em.normalized_smiles,
                                 molecular_formula=em.normalized_smiles.map(formula), ionization_mode=em.ionization_mode.astype(str),
                                 adduct=em.adduct.astype(str), precursor_mz=em.precursor_mz.astype(float),
                                 src=src_token(em.source_library)))
        add(meta, (np.asarray(EMZ[i]) for i in range(len(em))), (np.asarray(EIN[i]) for i in range(len(em))))
        log(f'extref: extlib {len(meta):,} spectra {time.time() - t0:.0f}s')
    if mbh_path:
        b = pd.read_parquet(mbh_path)
        b = b[b.instrument_type.astype(str).str.contains('LC-ESI', na=False) & (b.ms_type == 'MS2') & b.smiles.notna()
              & b.inchikey.notna()].reset_index(drop=True)
        meta = pd.DataFrame(dict(ingest_lib='mbh', inchikey14=b.inchikey.str[:14], normalized_smiles=b.smiles,
                                 molecular_formula=b.smiles.map(formula),
                                 ionization_mode=np.where(b.ion_mode.astype(str).str.lower().str.startswith('p'), 'positive', 'negative'),
                                 adduct=b.adduct.astype(str), precursor_mz=pd.to_numeric(b.precursor_mz, errors='coerce'),
                                 src=src_token(b.source_institution)))
        ok = meta.precursor_mz.notna().values
        add(meta[ok], b.mzs[ok], b.intensities[ok])
        log(f'extref: mbh {int(ok.sum()):,} spectra {time.time() - t0:.0f}s')
    lens = np.concatenate(lens)
    off = np.zeros(len(lens) + 1, np.int64); off[1:] = np.cumsum(lens)
    np.savez(out_prefix + '_peaks.npz', mz=np.concatenate(mzs), it=np.concatenate(its), off=off)
    meta = pd.concat(metas, ignore_index=True)
    for c in ['ingest_lib', 'ionization_mode', 'adduct', 'src']:
        meta[c] = meta[c].astype('category')
    meta.to_parquet(out_prefix + '_meta.parquet')
    log(f'extref {len(meta):,} spectra {off[-1]:,} peaks {time.time() - t0:.0f}s')
    return meta
