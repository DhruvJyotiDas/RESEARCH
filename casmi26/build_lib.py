"""Build a compact CSR spectral library from train.parquet.

Per spectrum: drop peaks > precursor+2 Da, drop peaks < floor*base, keep top-K by intensity,
sort by m/z, store float32 m/z and float32 normalised intensity. Output: npz + meta parquet.
"""
import sys, time
import numpy as np, pandas as pd, pyarrow.parquet as pq

SRC = sys.argv[1]; OUT = sys.argv[2]
K = 100; FLOOR = 0.002
pf = pq.ParquetFile(SRC)
meta_cols = ['ingest_lib', 'inchikey14', 'normalized_smiles', 'molecular_formula', 'ionization_mode',
             'instrument_type', 'adduct', 'precursor_mz', 'precursor_error_ppm', 'num_peaks',
             'base_peak_intensity', 'collision_energy_ev']
mz_chunks, in_chunks, lens, metas = [], [], [], []
t0 = time.time()
for rg in range(pf.metadata.num_row_groups):
    t = pf.read_row_group(rg, columns=meta_cols + ['ms2_mzs', 'ms2_normalized_intensities'])
    mzl = t.column('ms2_mzs').to_pylist(); itl = t.column('ms2_normalized_intensities').to_pylist()
    prec = t.column('precursor_mz').to_numpy()
    for m, i, p in zip(mzl, itl, prec):
        m = np.asarray(m, np.float64); i = np.asarray(i, np.float64)
        ok = (m > 0) & (m <= p + 2.0) & np.isfinite(i) & (i > 0)
        m, i = m[ok], i[ok]
        if len(i):
            i = i / i.max()
            ok = i >= FLOOR; m, i = m[ok], i[ok]
            if len(i) > K:
                s = np.argpartition(-i, K)[:K]; m, i = m[s], i[s]
            o = np.argsort(m); m, i = m[o], i[o]
        mz_chunks.append(m.astype(np.float32)); in_chunks.append(i.astype(np.float32)); lens.append(len(m))
    md = t.select(meta_cols).to_pandas()
    md['ce_max'] = md.collision_energy_ev.map(lambda x: float(np.max(x)) if x is not None and len(x) else np.nan)
    md['ce_n'] = md.collision_energy_ev.map(lambda x: len(x) if x is not None else 0)
    metas.append(md.drop(columns=['collision_energy_ev']))
    print(rg, len(lens), f'{time.time()-t0:.0f}s', flush=True)
lens = np.asarray(lens, np.int64)
off = np.zeros(len(lens) + 1, np.int64); off[1:] = np.cumsum(lens)
np.savez(OUT + '_peaks.npz', mz=np.concatenate(mz_chunks), it=np.concatenate(in_chunks), off=off)
meta = pd.concat(metas, ignore_index=True)
for c in ['ingest_lib', 'ionization_mode', 'instrument_type', 'adduct']:
    meta[c] = meta[c].astype('category')
meta.to_parquet(OUT + '_meta.parquet')
print('done', len(meta), off[-1], f'{time.time()-t0:.0f}s')
