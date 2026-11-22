"""Build a leak-free pseudo-test (structures absent from train.parquet) in test.parquet format.

Sources: MassBank-harmonized (samartalwar) LC-ESI MS2 and taeyangg4 extlib (non-MULTIPLEX rows).
Labels are written to a separate file that is never uploaded with the pseudo-test.
"""
import re, hashlib, json, numpy as np, pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
rng = np.random.default_rng(20261122)
TA = ['[M+H]+','[M+NH4]+','[M-H2O+H]+','[M-2H2O+H]+','[M+Na]+','[M+K]+','[M-H]-','[M-H2O-H]-','[M+CH2O2-H]-','[M+Cl]-']
AMAP = {'[M+H-H2O]+': '[M-H2O+H]+', '[M+H-2H2O]+': '[M-2H2O+H]+', '[M-H-H2O]-': '[M-H2O-H]-', '[M+HCOO]-': '[M+CH2O2-H]-',
        '[M+FA-H]-': '[M+CH2O2-H]-', '[M+CHO2]-': '[M+CH2O2-H]-'}
train_keys = set(pd.read_parquet('/home/user/data/trainlib_meta.parquet', columns=['inchikey14']).inchikey14.unique())
te_vis = pd.read_parquet('/home/user/data/comp/test.parquet')
BPI = float(te_vis.base_peak_intensity.median())

def ce_ev(s, prec):
    s = str(s) if s is not None else ''
    nums = [float(x) for x in re.findall(r'\d+(?:\.\d+)?', s)]
    if not nums: return [40.0]
    if '%' in s or 'nce' in s.lower() or 'nominal' in s.lower():
        return [round(n * prec / 500.0, 1) for n in nums[:3]]
    if s.strip().upper().endswith('V') and 'EV' not in s.upper():   # raw voltage, unreliable
        return [40.0]
    return nums[:3]

rows = []
# --- MassBank harmonized
b = pd.read_parquet('/home/user/data/ext/mbh/spectra.parquet')
b = b[b.instrument_type.str.contains('LC-ESI', na=False) & (b.ms_type == 'MS2') & b.smiles.notna()]
b['ik14'] = b.inchikey.str[:14]
b['adduct'] = b.adduct.replace(AMAP)
b = b[b.adduct.isin(TA) & ~b.ik14.isin(train_keys)]
for r in b.itertuples(index=False):
    rows.append(dict(src='mbh:' + str(r.source_institution), ik14=r.ik14, smiles=r.smiles, adduct=r.adduct,
                     mode='positive' if str(r.ion_mode).lower().startswith('p') else 'negative',
                     prec=float(r.precursor_mz), mz=np.asarray(r.mzs, float), it=np.asarray(r.intensities, float),
                     ce=ce_ev(r.collision_energy, float(r.precursor_mz)), ce_orig=str(r.collision_energy)))
# --- extlib (non multiplex)
m = pd.read_parquet('/home/user/data/ext/extlib/ext_meta.parquet')
MZ = np.load('/home/user/data/ext/extlib/ext_mz.npy', mmap_mode='r'); IN = np.load('/home/user/data/ext/extlib/ext_in.npy', mmap_mode='r')
m['adduct'] = m.adduct.replace(AMAP)
sel = m[~m.source_library.str.startswith('MULTIPLEX') & m.adduct.isin(TA) & ~m.inchikey14.isin(train_keys)]
for i, r in sel.iterrows():
    mz = np.asarray(MZ[i], float); it = np.asarray(IN[i], float); ok = (mz > 0) & (it > 0)
    rows.append(dict(src='ext:' + r.source_library, ik14=r.inchikey14, smiles=r.normalized_smiles, adduct=r.adduct,
                     mode=r.ionization_mode, prec=float(r.precursor_mz), mz=mz[ok], it=it[ok], ce=[40.0], ce_orig='unknown'))
df = pd.DataFrame(rows)
# clean peaks: drop > prec+2, normalise, need >= 3 peaks
def clean(r):
    mz, it = r.mz, r.it
    ok = (mz > 0) & (mz <= r.prec + 2) & (it > 0)
    mz, it = mz[ok], it[ok]
    if len(mz) < 3: return None
    o = np.argsort(mz); mz, it = mz[o], it[o] / it.max()
    return mz, it
cl = df.apply(clean, axis=1); df = df[cl.notna()].copy(); cl = cl[cl.notna()]
df['mz'] = [c[0] for c in cl]; df['it'] = [c[1] for c in cl]
# check precursor consistency with structure (drop >20 ppm)
ADD = {'[M+H]+': 1.007276, '[M+NH4]+': 18.033823, '[M-H2O+H]+': 1.007276 - 18.010565, '[M-2H2O+H]+': 1.007276 - 36.02113,
       '[M+Na]+': 22.989218, '[M+K]+': 38.963158, '[M-H]-': -1.007276, '[M-H2O-H]-': -1.007276 - 18.010565,
       '[M+CH2O2-H]-': 44.998201, '[M+Cl]-': 34.969402}
from rdkit.Chem.Descriptors import ExactMolWt
mass = {}
for s in df.smiles.unique():
    mol = Chem.MolFromSmiles(s); mass[s] = ExactMolWt(mol) if mol is not None else np.nan
df['M'] = df.smiles.map(mass)
df['ppm'] = (df.prec - (df.M + df.adduct.map(ADD))) / df.prec * 1e6
df = df[df.ppm.abs() <= 20]
df['family'] = df.src.str.replace(r'-PART\d|:.*ALKYLAMINES.*', ':ALKYLAMINES', regex=True)
print(df.groupby('src').ik14.nunique().sort_values(ascending=False).head(25))
# choose structures: cap alkylamines, keep everything else; then cap spectra per structure at 6 (diverse adduct/CE)
st = df.groupby('ik14').agg(src=('src', 'first'), n=('src', 'size')).reset_index()
alk = st[st.src.str.contains('ALKYLAMINES')]; other = st[~st.src.str.contains('ALKYLAMINES')]
alk = alk.sample(n=min(70, len(alk)), random_state=1)
keep = pd.concat([other, alk])
if len(keep) > 420: keep = keep.sample(n=420, random_state=2)
df = df[df.ik14.isin(set(keep.ik14))]
df = df.sample(frac=1, random_state=3)
df['b'] = df.ik14 + '|' + df.adduct + '|' + df.ce.map(lambda c: str(round(c[0] / 10)))
df = df.drop_duplicates('b')
df = df[df.groupby('ik14').cumcount() < 6]
mol_id = {k: 'm_' + hashlib.md5(('pt' + k).encode()).hexdigest()[:6] for k in df.ik14.unique()}
out = pd.DataFrame(dict(
    molecule_id=df.ik14.map(mol_id).values,
    spectrum_id=['s_' + hashlib.md5(f'{k}{i}'.encode()).hexdigest()[:8] for i, k in enumerate(df.ik14)],
    ms2_mzs=list(df.mz), ms2_normalized_intensities=list(df.it), base_peak_intensity=BPI,
    adduct=df.adduct.values, ionization_mode=df['mode'].values, instrument_type='timsTOF', precursor_mz=df.prec.values,
    collision_energy_orig=[','.join(str(int(round(x))) for x in c) for c in df.ce], collision_energy_ev=list(df.ce),
    collision_energy_orig_units='eV'))
out = out.sample(frac=1, random_state=4).reset_index(drop=True)
out.to_parquet('/home/user/data/pseudo/test.parquet', index=False)
pd.DataFrame(dict(molecule_id=sorted(out.molecule_id.unique()), smiles='CCO')).to_csv('/home/user/data/pseudo/sample_submission.csv', index=False)
lab = df.drop_duplicates('ik14')[['ik14', 'smiles', 'src']].assign(molecule_id=lambda d: d.ik14.map(mol_id))
lab.to_csv('/home/user/data/pseudo_labels_v1.csv', index=False)
open('/home/user/data/pseudo/exclude_ik14.txt', 'w').write('\n'.join(sorted(lab.ik14)))
print('molecules', out.molecule_id.nunique(), 'spectra', len(out)); print(out.adduct.value_counts()); print(lab.src.str.split(':').str[0].value_counts())
print(out.groupby('molecule_id').size().describe())
