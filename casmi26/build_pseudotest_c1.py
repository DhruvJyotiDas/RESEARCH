"""Pseudo-test v3: class-1 queries (structures in train, external spectra that are not near-duplicates of any train
spectrum of the same structure) mixed with not-yet-used non-train structures.

usage: python build_pseudotest_c1.py v3 n_class1 prev_labels.csv [...]
"""
import hashlib
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Descriptors import ExactMolWt

RDLogger.DisableLog('rdApp.*')
VER, N1 = sys.argv[1], int(sys.argv[2])
prev = set(pd.concat([pd.read_csv(p).ik14 for p in sys.argv[3:]])) if len(sys.argv) > 3 else set()
TA = ['[M+H]+', '[M+NH4]+', '[M-H2O+H]+', '[M-2H2O+H]+', '[M+Na]+', '[M+K]+', '[M-H]-', '[M-H2O-H]-', '[M+CH2O2-H]-', '[M+Cl]-']
AMAP = {'[M+H-H2O]+': '[M-H2O+H]+', '[M+H-2H2O]+': '[M-2H2O+H]+', '[M-H-H2O]-': '[M-H2O-H]-', '[M+HCOO]-': '[M+CH2O2-H]-'}
ADD = {'[M+H]+': 1.007276, '[M+NH4]+': 18.033823, '[M-H2O+H]+': 1.007276 - 18.010565, '[M-2H2O+H]+': 1.007276 - 36.02113,
       '[M+Na]+': 22.989218, '[M+K]+': 38.963158, '[M-H]-': -1.007276, '[M-H2O-H]-': -1.007276 - 18.010565,
       '[M+CH2O2-H]-': 44.998201, '[M+Cl]-': 34.969402}
meta = pd.read_parquet('/home/user/data/trainlib_meta.parquet', columns=['inchikey14', 'precursor_mz', 'adduct'])
d = np.load('/home/user/data/trainlib_peaks.npz'); OFF, MZ, IT = d['off'], d['mz'], d['it']
by_key = meta.groupby('inchikey14').indices
train_keys = set(by_key)


def cos(m1, i1, m2, i2, tol=0.01):
    a = np.sqrt(i1); a /= np.linalg.norm(a) + 1e-12
    b = np.sqrt(i2); b /= np.linalg.norm(b) + 1e-12
    j = np.searchsorted(m2, m1)
    s = 0.0
    used = set()
    for k in range(len(m1)):
        for jj in (j[k] - 1, j[k]):
            if 0 <= jj < len(m2) and jj not in used and abs(m2[jj] - m1[k]) <= tol:
                s += a[k] * b[jj]; used.add(jj); break
    return s


def max_train_cos(key, mz, it, prec):
    best = 0.0
    for r in by_key.get(key, [])[:200]:
        if abs(meta.precursor_mz.values[r] - prec) > 0.02:
            continue
        m2 = MZ[OFF[r]:OFF[r + 1]].astype(float); i2 = IT[OFF[r]:OFF[r + 1]].astype(float)
        if len(m2):
            best = max(best, cos(mz, it, m2, i2))
    return best


rows = []
b = pd.read_parquet('/home/user/data/ext/mbh/spectra.parquet')
b = b[b.instrument_type.str.contains('LC-ESI', na=False) & (b.ms_type == 'MS2') & b.smiles.notna()]
b['ik14'] = b.inchikey.str[:14]; b['adduct'] = b.adduct.replace(AMAP)
b = b[b.adduct.isin(TA)]
for r in b.itertuples(index=False):
    rows.append(dict(src='mbh:' + str(r.source_institution), ik14=r.ik14, smiles=r.smiles, adduct=r.adduct,
                     mode='positive' if str(r.ion_mode).lower().startswith('p') else 'negative',
                     prec=float(r.precursor_mz), mz=np.asarray(r.mzs, float), it=np.asarray(r.intensities, float), ce=[40.0]))
m = pd.read_parquet('/home/user/data/ext/extlib/ext_meta.parquet')
EMZ = np.load('/home/user/data/ext/extlib/ext_mz.npy', mmap_mode='r'); EIN = np.load('/home/user/data/ext/extlib/ext_in.npy', mmap_mode='r')
m['adduct'] = m.adduct.replace(AMAP)
sel = m[~m.source_library.str.startswith('MULTIPLEX') & m.adduct.isin(TA)]
for i, r in sel.iterrows():
    mz = np.asarray(EMZ[i], float); it = np.asarray(EIN[i], float); ok = (mz > 0) & (it > 0)
    rows.append(dict(src='ext:' + r.source_library, ik14=r.inchikey14, smiles=r.normalized_smiles, adduct=r.adduct,
                     mode=r.ionization_mode, prec=float(r.precursor_mz), mz=mz[ok], it=it[ok], ce=[40.0]))
df = pd.DataFrame(rows)
df = df[~df.ik14.isin(prev)]
mass = {s: ExactMolWt(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) is not None else np.nan for s in df.smiles.unique()}
df['ppm'] = (df.prec - (df.smiles.map(mass) + df.adduct.map(ADD))) / df.prec * 1e6
df = df[df.ppm.abs() <= 20]
def clean(r):
    ok = (r.mz > 0) & (r.mz <= r.prec + 2) & (r.it > 0)
    mz, it = r.mz[ok], r.it[ok]
    if len(mz) < 4: return None
    o = np.argsort(mz); return mz[o], it[o] / it[o].max()
cl = df.apply(clean, axis=1); df = df[cl.notna()].copy(); cl = cl[cl.notna()]
df['mz'] = [c[0] for c in cl]; df['it'] = [c[1] for c in cl]
c1 = df[df.ik14.isin(train_keys)].copy()
rng = np.random.default_rng(5)
keys = rng.permutation(c1.ik14.unique())
keep, tested = [], 0
for k in keys:
    g = c1[c1.ik14 == k]
    good = [i for i, r in g.iterrows() if max_train_cos(k, r.mz, r.it, r.prec) < 0.98]
    tested += 1
    if good:
        keep.append(g.loc[good])
    if len(keep) >= N1:
        break
c1 = pd.concat(keep)
print('class-1 structures', c1.ik14.nunique(), 'tested', tested)
c2 = df[~df.ik14.isin(train_keys)].copy()
# NP-like stratification: the hidden test is Enveda natural products, so most class-2 queries should be structures in the
# COCONUT/LOTUS candidate pool; no source library may supply more than 20% of them (v3 draft was 290 alkylamines)
sys.path.insert(0, '/home/user/RESEARCH/casmi26')
from analyze_pseudo import key as metric_key
pool_keys = set(pd.read_parquet('/home/user/data/pool/pool_meta.parquet', columns=['key']).key)
st = c2.drop_duplicates('ik14')[['ik14', 'smiles', 'src']].copy()
st['lib'] = st.src.str.split(':').str[-1]
st['in_pool'] = st.smiles.map(metric_key).isin(pool_keys)
n2 = 400 - c1.ik14.nunique()
n_pool = min(int(round(0.7 * n2)), int(st.in_pool.sum()))
cap = max(5, int(0.2 * n2))
st = st.sample(frac=1, random_state=9)
st = st[st.groupby('lib').cumcount() < cap]
pick = pd.concat([st[st.in_pool].head(n_pool), st[~st.in_pool].head(n2 - n_pool)])
print('class-2 in pool', int(pick.in_pool.sum()), 'of', len(pick)); print(pick.lib.value_counts().head(12))
c2 = c2[c2.ik14.isin(set(pick.ik14))]
df = pd.concat([c1.assign(cls=1), c2.assign(cls=2)])
df = df.sample(frac=1, random_state=3)
df['b'] = df.ik14 + '|' + df.adduct
df = df.drop_duplicates('b'); df = df[df.groupby('ik14').cumcount() < 5]
BPI = float(pd.read_parquet('/home/user/data/comp/test.parquet').base_peak_intensity.median())
mol_id = {k: 'm_' + hashlib.md5(('pt' + VER + k).encode()).hexdigest()[:6] for k in df.ik14.unique()}
out = pd.DataFrame(dict(
    molecule_id=df.ik14.map(mol_id).values,
    spectrum_id=['s_' + hashlib.md5(f'{VER}{k}{i}'.encode()).hexdigest()[:8] for i, k in enumerate(df.ik14)],
    ms2_mzs=list(df.mz), ms2_normalized_intensities=list(df.it), base_peak_intensity=BPI, adduct=df.adduct.values,
    ionization_mode=df['mode'].values, instrument_type='timsTOF', precursor_mz=df.prec.values,
    collision_energy_orig='40', collision_energy_ev=[[40.0]] * len(df), collision_energy_orig_units='eV')).sample(frac=1, random_state=4)
import os
os.makedirs(f'/home/user/data/pseudo_{VER}', exist_ok=True)
out.to_parquet(f'/home/user/data/pseudo_{VER}/test.parquet', index=False)
pd.DataFrame(dict(molecule_id=sorted(out.molecule_id.unique()), smiles='CCO')).to_csv(f'/home/user/data/pseudo_{VER}/sample_submission.csv', index=False)
lab = df.drop_duplicates('ik14')[['ik14', 'smiles', 'src', 'cls']].assign(molecule_id=lambda x: x.ik14.map(mol_id))
lab.to_csv(f'/home/user/data/pseudo_labels_{VER}.csv', index=False)
# exclusion: only the non-train structures must be hidden from the external overlay; class-1 structures must ALSO be
# hidden there (their external spectra are the queries), they keep their train references
open(f'/home/user/data/pseudo_{VER}/exclude_ik14.txt', 'w').write('\n'.join(sorted(lab.ik14)))
print('molecules', out.molecule_id.nunique(), 'spectra', len(out)); print(lab.cls.value_counts()); print(out.adduct.value_counts())
