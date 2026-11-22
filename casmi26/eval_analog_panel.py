"""Local check of query-centric analog propagation on the 250 enveda-np-examples molecules (class-2 simulation:
every spectrum of the true structure hidden), candidates = same-formula pool structures (train + COCONUT)."""
import re, sys, time
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog('rdApp.*')
sys.path.insert(0, '/home/user/RESEARCH/casmi26')
from analog import AnalogIndex, ap_scores

M = dict(C=12.0, H=1.00782503207, N=14.0030740048, O=15.9949146196, P=30.97376163, S=31.97207100,
         F=18.99840322, Cl=34.96885268, Br=78.9183371, I=126.904473, Si=27.9769265325, B=11.0093054, Se=79.9165213, Na=22.98977, K=38.96371)
def fmass(f):
    tot = 0.0
    for el, n in re.findall(r'([A-Z][a-z]?)(\d*)', str(f)):
        if el not in M: return np.nan
        tot += M[el] * (int(n) if n else 1)
    return tot

t0 = time.time()
meta = pd.read_parquet('/home/user/data/trainlib_meta.parquet', columns=['ingest_lib', 'inchikey14', 'normalized_smiles', 'molecular_formula', 'ionization_mode', 'adduct', 'precursor_mz'])
st = meta.drop_duplicates('inchikey14')[['inchikey14', 'normalized_smiles', 'molecular_formula']]
st['mass'] = st.molecular_formula.map(fmass)
smass = dict(zip(st.inchikey14, st.mass))
idx = AnalogIndex('/home/user/data/trainlib_peaks.npz', '/home/user/data/trainlib_meta.parquet', smass, win=200.0)
print('index', len(idx.rep), f'{time.time()-t0:.0f}s', flush=True)
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
smi_of = dict(zip(st.inchikey14, st.normalized_smiles))
_fpc = {}
def fp_smiles(s):
    if s not in _fpc:
        m = Chem.MolFromSmiles(s)
        _fpc[s] = gen.GetFingerprintAsNumPy(m).astype(bool) if m is not None else np.zeros(2048, bool)
    return _fpc[s]
fp_of_key = lambda k: fp_smiles(smi_of[k])

pool = pd.read_parquet('/home/user/data/pool/pool_meta.parquet', columns=['key', 'smiles', 'formula', 'mass', 'src', 'train_sid', 'np_likeness'])
d = np.load('/home/user/data/trainlib_peaks.npz')
OFF, MZ, IT = d['off'], d['mz'], d['it']
q = meta[meta.ingest_lib == 'enveda-np-examples']
rows = []
for k, g in q.groupby('inchikey14'):
    form = g.molecular_formula.iloc[0]
    target = smass[k]
    spectra = [(MZ[OFF[i]:OFF[i+1]].astype(float), IT[OFF[i]:OFF[i+1]].astype(float), 1 if g.ionization_mode[i] == 'positive' else -1) for i in g.index]
    spectra = [s for s in spectra if len(s[0]) >= 3]
    if not spectra: continue
    an = idx.search(spectra, target, exclude_keys={k})
    cands = pool[pool.formula == form].drop_duplicates('key')
    truth_smi = g.normalized_smiles.iloc[0]
    tm = Chem.MolFromSmiles(truth_smi)
    from rdkit.Chem.MolStandardize import rdMolStandardize
    tkey = Chem.MolToInchiKey(rdMolStandardize.TautomerEnumerator().Canonicalize(tm))[:14]
    if tkey not in set(cands.key):
        continue
    cfp = np.stack([fp_smiles(s) for s in cands.smiles])
    f = ap_scores(cfp, an, fp_of_key)
    for j, (ck, cs) in enumerate(zip(cands.key, cands.smiles)):
        rows.append(dict(q=k, c=ck, y=int(ck == tkey), n=len(cands), np_like=cands.np_likeness.values[j], **{kk: v[j] for kk, v in f.items()}))
    if len(rows) and len(set(r['q'] for r in rows)) % 25 == 0:
        print(len(set(r['q'] for r in rows)), f'{time.time()-t0:.0f}s', flush=True)
D = pd.DataFrame(rows)
D.to_parquet('/home/user/data/eval_analog_panel.parquet')
def mrr(score):
    out = []
    for qq, gq in D.assign(s=score).groupby('q'):
        s = gq.s.values; t = np.flatnonzero(gq.y.values)[0]
        rk = 1 + (s > s[t]).sum() + 0.5 * ((s == s[t]).sum() - 1); out.append(1 / rk if rk <= 25 else 0)
    return np.mean(out)
print('queries', D.q.nunique(), 'median cands', D.groupby('q').size().median())
for name in ['ap', 'ap_sum', 'ap_best_tan', 'ap_top_tan', 'np_like']:
    print(f'{name:12s} MRR {mrr(D[name].fillna(-9)):.3f}')
print('random', D.groupby('q').n.first().map(lambda n: np.mean(1 / np.arange(1, min(n, 25) + 1)) * min(n, 25) / n).mean())
