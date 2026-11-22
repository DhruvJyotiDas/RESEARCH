"""Per-stage truth ranks for a pseudo-test run of the stack.

usage: python analyze_pseudo.py <kernel_output_dir> <labels.csv>
Writes <out>/stage_ranks.parquet (molecule x stage -> rank of truth, 0 = absent) and prints MRR / top-1 / Hit@25.
"""
import glob
import json
import os
import sys
from functools import lru_cache

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog('rdApp.*')
_TE = rdMolStandardize.TautomerEnumerator()


@lru_cache(maxsize=500000)
def key(s):
    """Metric identity: RDKit tautomer canonicalisation, InChIKey first block."""
    m = Chem.MolFromSmiles(str(s))
    if m is None:
        return None
    try:
        return Chem.MolToInchiKey(_TE.Canonicalize(m))[:14]
    except Exception:
        return None


def truth_rank(smiles_list, tkey, limit=None):
    seen = set()
    r = 0
    for s in smiles_list:
        k = key(s)
        if k is None or k in seen:
            continue
        seen.add(k)
        r += 1
        if limit and r > limit:
            break
        if k == tkey:
            return r
    return 0


def summarize(ranks):
    r = np.asarray(ranks, float)
    rr = np.where((r > 0) & (r <= 25), 1 / np.maximum(r, 1), 0)
    return dict(mrr=rr.mean(), top1=(r == 1).mean(), hit5=((r > 0) & (r <= 5)).mean(), hit25=((r > 0) & (r <= 25)).mean(),
                anywhere=(r > 0).mean())


def main(out_dir, labels_csv):
    lab = pd.read_csv(labels_csv)
    lab['tkey'] = lab.smiles.map(key)
    T = dict(zip(lab.molecule_id, lab.tkey))
    stages = {}
    for f in sorted(glob.glob(os.path.join(out_dir, '**/submission*.csv'), recursive=True)):
        name = os.path.relpath(f, out_dir).replace('.csv', '')
        s = pd.read_csv(f, dtype=str, keep_default_na=False)
        stages[name] = {m: t.split(';') for m, t in zip(s.molecule_id, s.smiles)}
    for fname, field in [('pc_lists.json', 'pc'), ('eng_lists.json', 'smiles')]:
        p = os.path.join(out_dir, fname)
        if os.path.exists(p):
            d = json.load(open(p))
            stages['ch_' + fname.split('_')[0]] = {m: v.get(field, []) for m, v in d.items()}
    p = os.path.join(out_dir, 'union_channels.pkl')
    if os.path.exists(p):
        import pickle
        u = pickle.load(open(p, 'rb'))
        if u.get('BASE'):
            stages['ch_main60'] = {m: v[0] for m, v in u['BASE'].items()}
    rows = []
    for m, tk in T.items():
        row = dict(molecule_id=m)
        for name, lists in stages.items():
            row[name] = truth_rank(lists.get(m, []), tk)
        rows.append(row)
    R = pd.DataFrame(rows).merge(lab, on='molecule_id')
    R.to_parquet(os.path.join(out_dir, 'stage_ranks.parquet'))
    summ = pd.DataFrame({name: summarize(R[name]) for name in stages}).T.sort_values('mrr')
    pd.set_option('display.width', 200)
    print(summ.round(3))
    if 'in_pool' in R:
        final = 'submission' if 'submission' in R else summ.index[-1]
        print('\nby pool membership (final):')
        print(R.groupby('in_pool')[final].apply(lambda r: pd.Series(summarize(r))).unstack().round(3))
    return R


if __name__ == '__main__':
    main(*sys.argv[1:3])
