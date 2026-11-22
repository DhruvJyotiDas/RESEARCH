"""Train and cross-validate the learned final merger on pseudo-test runs.

usage: python train_merger.py out_model.json run1_features.parquet:labels1.csv [run2...]
Each features parquet comes from features.build_features on a pseudo run; labels give the truth SMILES.
Prints OOF MRR@25 / top-1 of (a) the stack's final order, (b) the model, (c) blends final + alpha * model, and
(d) leader-safe variants; writes the full-data model with the chosen blend as JSON (LightGBM model string).
"""
import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, '/home/user/RESEARCH/casmi26')
from analyze_pseudo import key  # metric identity

BASE_FEATS = ['log_final_rank', 'log_frozen_rank', 'log_main_rank', 'log_eng_rank', 'log_pc_rank', 'log_prefusion_rank',
              'ice_z', 'ice_gap', 'gl_z', 'gl_gap', 'neural_score_z', 'neural_score_gap', 'neural_cos_z', 'neural_fp_z',
              'main_score_z', 'pc_fz_z', 'pc_fz_gap',
              'pr_pop', 'pr_pop_gap', 'pr_lsid_z', 'pr_lpmid_z', 'pr_lpatent', 'pr_lmesh', 'pr_lsynonyms', 'pr_vendors',
              'pr_deps', 'pr_lotus', 'pr_iso', 'np_coconut', 'np_lotus', 'np_npatlas'] + [f'src_b{b}' for b in range(9)] + [
              'in_pubchem', 'in_train', 'same_formula', 'frac_formula', 'abs_ppm', 'n_heavy', 'lib_max', 'n_spec', 'n_pos',
              'ap', 'ap_z', 'ap_gap', 'ap_best_tan', 'ap_top_sim', 'ap_sum_z']


def load(spec):
    fpath, lpath = spec.split(':')
    D = pd.read_parquet(fpath)
    lab = pd.read_csv(lpath)
    T = dict(zip(lab.molecule_id, lab.smiles.map(key)))
    D['y'] = (D.key == D.molecule_id.map(T)).astype(int)
    D['abs_ppm'] = D.ppm.abs()
    D['run'] = fpath
    D['gid'] = D.run + '|' + D.molecule_id
    n_mol = lab.molecule_id.nunique()
    return D, n_mol


def mrr_of(D, score, n_total):
    rr, top1 = [], []
    for _, g in D.assign(s=score).groupby('gid', sort=False):
        o = np.argsort(-g.s.values, kind='stable')
        y = g.y.values[o]
        hit = np.flatnonzero(y[:25])
        rr.append(1 / (hit[0] + 1) if len(hit) else 0.0)
        top1.append(float(len(hit) and hit[0] == 0))
    pad = n_total - len(rr)
    return (np.sum(rr) / n_total, np.sum(top1) / n_total) if n_total else (np.nan, np.nan)


def leader_safe(D, score, margin):
    """Keep the stack's rank 1 unless the model prefers another candidate by more than `margin` (z units)."""
    out = score.copy()
    for _, g in D.assign(s=score).groupby('gid', sort=False):
        lead = g.index[g.final_rank.values == 1]
        if len(lead) == 0:
            continue
        li = lead[0]
        best = g.s.idxmax()
        if best != li and g.s[best] - g.s[li] < margin:
            out[li] = g.s.max() + 1e-6
    return out


def main(out_path, *specs):
    parts, n_total = [], 0
    for s in specs:
        D, n = load(s); parts.append(D); n_total += n
    D = pd.concat(parts, ignore_index=True)
    feats = [f for f in BASE_FEATS if f in D.columns]
    D = D.sort_values(['gid', 'final_rank']).reset_index(drop=True)
    base = -D.log_final_rank.values
    print('molecules', n_total, 'with truth among candidates', D.groupby('gid').y.max().sum())
    print('stack final order  MRR %.4f top1 %.4f' % mrr_of(D, base, n_total))
    params = dict(objective='lambdarank', metric='ndcg', ndcg_eval_at=[1, 5], learning_rate=0.03, num_leaves=15,
                  min_data_in_leaf=40, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
                  verbose=-1, label_gain=[0, 1])
    oof = np.zeros(len(D))
    gkf = GroupKFold(n_splits=5)
    iters = []
    for tr, va in gkf.split(D, groups=D.gid):
        Dt, Dv = D.iloc[tr], D.iloc[va]
        gt = Dt.groupby('gid', sort=False).size().values; gv = Dv.groupby('gid', sort=False).size().values
        dt = lgb.Dataset(Dt[feats], Dt.y, group=gt); dv = lgb.Dataset(Dv[feats], Dv.y, group=gv)
        m = lgb.train(params, dt, 600, valid_sets=[dv], callbacks=[lgb.early_stopping(60, verbose=False)])
        iters.append(m.best_iteration)
        oof[va] = m.predict(Dv[feats], num_iteration=m.best_iteration)
    z = lambda x: pd.Series(x).groupby(D.gid).transform(lambda v: (v - v.mean()) / (v.std() + 1e-9)).values
    print('model alone        MRR %.4f top1 %.4f' % mrr_of(D, oof, n_total))
    best = (mrr_of(D, base, n_total)[0], 0.0)
    for a in [0.25, 0.5, 1.0, 2.0, 4.0]:
        sc = z(base) + a * z(oof)
        r = mrr_of(D, sc, n_total)
        ls = mrr_of(D, leader_safe(D, pd.Series(sc), 1.0).values, n_total)
        print(f'blend a={a:<4} MRR {r[0]:.4f} top1 {r[1]:.4f} | leader-safe(1.0) MRR {ls[0]:.4f} top1 {ls[1]:.4f}')
        if r[0] > best[0]:
            best = (r[0], a)
    # per-run breakdown at the chosen alpha
    sc = z(base) + best[1] * z(oof)
    for run, g in D.groupby('run'):
        n = g.gid.nunique()
        print(run.split('/')[-1], 'stack %.4f' % mrr_of(g, -g.log_final_rank.values, n)[0],
              'merged %.4f' % mrr_of(g, sc[g.index], n)[0])
    full = lgb.train(params, lgb.Dataset(D[feats], D.y, group=D.groupby('gid', sort=False).size().values),
                     int(np.median(iters)) or 100)
    imp = pd.Series(full.feature_importance('gain'), index=feats).sort_values(ascending=False)
    print(imp.head(20).round(1))
    json.dump(dict(model=full.model_to_string(), feats=feats, alpha=best[1]), open(out_path, 'w'))
    print('saved', out_path, 'alpha', best[1])


if __name__ == '__main__':
    main(sys.argv[1], *sys.argv[2:])
