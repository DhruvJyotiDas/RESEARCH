"""Offline evaluation of the PubChem x analog channel on a pseudo-test (see channel.py).

usage: python eval_pc_channel.py <pseudo_dir> <labels.csv> <out.parquet> [ppm=10] [keep=300]
Writes all kept candidates (with truth flag y) and <out>_info.csv (per-molecule window stats and hand-score truth ranks).
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import channel
import libbuild
from analog import AnalogIndex
from features import molecule_targets


def main(pdir, labels, out, ppm=10.0, keep=300, lib=os.environ.get('CH_LIB', '/home/user/data/trainlib')):
    t0 = time.time()
    te = pd.read_parquet(os.path.join(pdir, 'test.parquet'))
    lab = pd.read_csv(labels)
    truth = dict(zip(lab.molecule_id, lab.ik14))
    st = pd.read_parquet(lib + '_meta.parquet',
                         columns=['inchikey14', 'normalized_smiles', 'molecular_formula']).drop_duplicates('inchikey14')
    st['molecular_formula'] = st.molecular_formula.astype(str)
    train_keys = set(pd.read_parquet('/home/user/data/trainlib_meta.parquet', columns=['inchikey14']).inchikey14)
    idx = AnalogIndex(lib + '_peaks.npz', lib + '_meta.parquet',
                      dict(zip(st.inchikey14, st.molecular_formula.map(libbuild.formula_mass))),
                      smiles_of=dict(zip(st.inchikey14, st.normalized_smiles)))
    channel.init('/home/user/data/pop', idx, train_keys, dict(zip(st.inchikey14, st.molecular_formula)))
    src = dict(zip(lab.molecule_id, libbuild.src_token(lab.src))) if os.environ.get('CH_LOSO') else {}
    jobs = [j + (truth[j[0]], (src[j[0]],) if j[0] in src else ()) for j in channel.jobs_of(te, molecule_targets(te), ppm, keep)]
    I, D = channel.run_windows(jobs)
    D['y'] = (D.ik14 == D.molecule_id.map(truth)).astype(int)
    I.to_csv(out.replace('.parquet', '_info.csv'), index=False)
    D.to_parquet(out)
    n = len(I)
    print('molecules', n, 'truth in window', round(I.truth_in_window.mean(), 3), 'median window', I.n_window.median())
    for s in channel.HAND:
        r = I['rank_' + s].fillna(1e9)
        print(f'{s:15s} MRR@25 {np.where(r <= 25, 1 / r, 0).sum() / n:.4f} top1 {(r == 1).mean():.3f} '
              f'hit25 {(r <= 25).mean():.3f} hit100 {(r <= 100).mean():.3f} hit300 {(r <= 300).mean():.3f}')
    print('total', round(time.time() - t0), 's')


if __name__ == '__main__':
    a = sys.argv[1:]
    main(a[0], a[1], a[2], *(float(x) for x in a[3:4]), *(int(x) for x in a[4:5]))
