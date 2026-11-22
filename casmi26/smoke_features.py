"""Smoke test of features.build_features on a synthetic run dir built from the visible test (labels known)."""
import os, sys, time, re
import numpy as np, pandas as pd
sys.path.insert(0, '/home/user/RESEARCH/casmi26')
from features import build_features, PubChemPrior, mol_info
from analog import AnalogIndex

t0 = time.time()
RUN = '/home/user/data/smoke_run'; os.makedirs(RUN, exist_ok=True)
te = pd.read_parquet('/home/user/data/comp/test.parquet')
meta = pd.read_parquet('/home/user/data/trainlib_meta.parquet', columns=['inchikey14', 'normalized_smiles', 'molecular_formula', 'precursor_mz', 'adduct'])
# truth per visible molecule via precursor match (visible test = enveda-180 rows)
k = lambda d: d.precursor_mz.round(4).astype(str) + '|' + d.adduct.astype(str)
mp = meta.assign(k=k(meta)).drop_duplicates('k').set_index('k')
te['k'] = k(te)
truth = te.groupby('molecule_id').k.first().map(mp.normalized_smiles)
mols = list(truth.index[:40])
te = te[te.molecule_id.isin(mols)]
pool = pd.read_parquet('/home/user/data/pool/pool_meta.parquet', columns=['smiles', 'formula'])
byf = pool.groupby('formula').smiles.apply(list).to_dict()
rows = []
for m in mols:
    s = truth[m]; f = mol_info(s)[2]
    iso = [x for x in byf.get(f, []) if x != s][:9]
    lst = iso[:3] + [s] + iso[3:]
    rows.append((m, ';'.join(lst)))
pd.DataFrame(rows, columns=['molecule_id', 'smiles']).to_csv(f'{RUN}/submission.csv', index=False)
pd.DataFrame(dict(molecule_id=mols, smiles=[truth[m] for m in mols])).to_csv(f'{RUN}/labels.csv', index=False)
train_keys = set(meta.inchikey14.unique())
prior = PubChemPrior('/home/user/data/pop')
st = meta.drop_duplicates('inchikey14')
M = dict(C=12.0, H=1.00782503207, N=14.0030740048, O=15.9949146196, P=30.97376163, S=31.97207100, F=18.99840322, Cl=34.96885268, Br=78.9183371, I=126.904473)
def fmass(f):
    tot = 0.0
    for el, n in re.findall(r'([A-Z][a-z]?)(\d*)', str(f)):
        if el not in M: return np.nan
        tot += M[el] * (int(n) if n else 1)
    return tot
idx = AnalogIndex('/home/user/data/trainlib_peaks.npz', '/home/user/data/trainlib_meta.parquet',
                  dict(zip(st.inchikey14, st.molecular_formula.map(fmass))), smiles_of=dict(zip(st.inchikey14, st.normalized_smiles)))
print('setup', f'{time.time()-t0:.0f}s', flush=True)
t1 = time.time()
D = build_features(RUN, te, train_keys, prior, analog_index=idx)
print('features', D.shape, f'{time.time()-t1:.0f}s for {len(mols)} molecules')
print(D[['molecule_id', 'final_rank', 'in_train', 'in_pubchem', 'pr_pop', 'pr_vendors', 'ap', 'ap_z', 'same_formula']].head(12))
print('in_pubchem rate', D.in_pubchem.mean(), 'nan ap', D.ap.isna().mean())
