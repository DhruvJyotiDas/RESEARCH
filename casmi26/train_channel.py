"""Learned ranker for the PubChem x analog channel (eval_pc_channel.py outputs), grouped CV over molecules.

usage: python train_channel.py out_model.json pcch_v1.parquet:pcch_v1_info.csv [...]
MRR is computed over ALL molecules of each set (truth outside the kept candidates counts as 0).
"""
import json
import sys

import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import channel
FEATS = channel.FEATS


def prep(path, info_path, train_keys):
    channel.G['train_keys'] = train_keys
    D = channel.add_rel(pd.read_parquet(path))
    D['set'] = path
    D['gid'] = path + '|' + D.molecule_id
    return D, pd.read_csv(info_path)


def mrr(D, score, n_total):
    rr = 0.0; top1 = 0
    for _, g in D.assign(s=score).groupby('gid', sort=False):
        o = np.argsort(-g.s.values, kind='stable'); y = g.y.values[o]
        h = np.flatnonzero(y[:25])
        if len(h):
            rr += 1 / (h[0] + 1); top1 += h[0] == 0
    return rr / n_total, top1 / n_total


def main(out, *specs):
    train_keys = set(pd.read_parquet('/home/user/data/trainlib_meta.parquet', columns=['inchikey14']).inchikey14.unique())
    parts, ns = [], {}
    for s in specs:
        p, i = s.split(':')
        D, I = prep(p, i, train_keys); parts.append(D); ns[p] = len(I)
    D = pd.concat(parts, ignore_index=True)
    params = dict(objective='lambdarank', learning_rate=0.03, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.7,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, label_gain=[0, 1],
                  lambdarank_truncation_level=30)
    oof = np.zeros(len(D)); iters = []
    for tr, va in GroupKFold(5).split(D, groups=D.gid):
        a, b = D.iloc[tr], D.iloc[va]
        m = lgb.train(params, lgb.Dataset(a[FEATS], a.y, group=a.groupby('gid', sort=False).size().values), 800,
                      valid_sets=[lgb.Dataset(b[FEATS], b.y, group=b.groupby('gid', sort=False).size().values)],
                      callbacks=[lgb.early_stopping(80, verbose=False)])
        iters.append(m.best_iteration); oof[va] = m.predict(b[FEATS], num_iteration=m.best_iteration)
    D['oof'] = oof
    for p, g in D.groupby('set'):
        n = ns[p]
        print(p.split('/')[-1], 'hand s_ap_pop_np_fx MRR %.4f top1 %.3f' % mrr(g, g.s_ap_pop_np_fx.values, n),
              '| learned MRR %.4f top1 %.3f' % mrr(g, g.oof.values, n))
    full = lgb.train(params, lgb.Dataset(D[FEATS], D.y, group=D.groupby('gid', sort=False).size().values), int(np.median(iters)))
    imp = pd.Series(full.feature_importance('gain'), index=FEATS).sort_values(ascending=False)
    print(imp.head(15).round(0).to_dict())
    json.dump(dict(model=full.model_to_string(), feats=FEATS), open(out, 'w'))
    D[['gid', 'molecule_id', 'ik14', 'smiles', 'y', 'oof']].to_parquet(out.replace('.json', '_oof.parquet'))


if __name__ == '__main__':
    main(sys.argv[1], *sys.argv[2:])
