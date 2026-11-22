"""Merger training rows for one pseudo-test run: stack outputs + out-of-fold PubChem-channel candidates.

usage: python prep_merger_data.py <run_dir> <pseudo_dir> <pcch.parquet> <pcch_info.csv> <channel_oof.parquet> <out.parquet> [ch_top=10]

Channel scores come from train_channel.py's out-of-fold predictions (never in-sample). Truth rows that the offline
evaluation forced into the channel's kept set (no hand score ranked them within the keep limit) are dropped, so the
channel offers exactly what it would offer on Kaggle.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import libbuild
from analog import AnalogIndex
from features import PubChemPrior, build_features


def main(run_dir, pdir, pcch, info, oof, out, ch_top=10, keep=300):
    te = pd.read_parquet(os.path.join(pdir, 'test.parquet'))
    I = pd.read_csv(info)
    O = pd.read_parquet(oof)
    O = O[O.gid.str.startswith(pcch + '|')].copy()
    hand = [c for c in I.columns if c.startswith('rank_')]
    forced_only = set(I.molecule_id[(I[hand].fillna(1e9) > keep).all(axis=1) & I.truth_in_window.astype(bool)])
    drop = O.molecule_id.isin(forced_only) & (O.y == 1)
    print('channel rows', len(O), 'forced-only truth rows dropped', int(drop.sum()))
    C = O[~drop].rename(columns={'oof': 'ch_score'})[['molecule_id', 'ik14', 'smiles', 'ch_score']].copy()
    C['ch_rank'] = C.groupby('molecule_id').ch_score.rank(ascending=False, method='first')
    st = pd.read_parquet('/home/user/data/trainlib_meta.parquet',
                         columns=['inchikey14', 'normalized_smiles', 'molecular_formula']).drop_duplicates('inchikey14')
    idx = AnalogIndex('/home/user/data/trainlib_peaks.npz', '/home/user/data/trainlib_meta.parquet',
                      dict(zip(st.inchikey14, st.molecular_formula.map(libbuild.formula_mass))),
                      smiles_of=dict(zip(st.inchikey14, st.normalized_smiles)))
    D = build_features(run_dir, te, set(st.inchikey14), PubChemPrior('/home/user/data/pop'), analog_index=idx,
                       channel=C, ch_top=ch_top)
    D.to_parquet(out)
    print(out, D.shape, D.tag.value_counts().to_dict())


if __name__ == '__main__':
    a = sys.argv[1:]
    main(*a[:6], *(int(x) for x in a[6:7]))
