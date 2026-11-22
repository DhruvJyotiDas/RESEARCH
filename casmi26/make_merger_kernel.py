"""Append the learned-merger overlay cell to a stack notebook.

usage: python make_merger_kernel.py <base.ipynb> <model.json> <out_dir> <kernel_id> [--margin M] [--alpha A] [--keep1]
The cell embeds analog.py, features.py, libbuild.py, the merger runner and the model, runs the runner in its own
process (fresh memory, hard timeout), validates its output and only then replaces /kaggle/working/submission.csv.
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

RUNNER = r'''
import json, os, sys, time
import numpy as np, pandas as pd
code_dir, comp, root, pop_dir, model_path, out_path = sys.argv[1:7]
sys.path.insert(0, code_dir)
import lightgbm as lgb
import libbuild
from analog import AnalogIndex
from features import build_features, PubChemPrior
t0 = time.time()
cfg = json.load(open(model_path))
te = pd.read_parquet(os.path.join(comp, 'test.parquet'))
meta = libbuild.build(os.path.join(comp, 'train.parquet'), os.path.join(root, 'merger_trainlib'))
st = meta.drop_duplicates('inchikey14')
idx = AnalogIndex(os.path.join(root, 'merger_trainlib_peaks.npz'), os.path.join(root, 'merger_trainlib_meta.parquet'),
                  dict(zip(st.inchikey14, st.molecular_formula.map(libbuild.formula_mass))),
                  smiles_of=dict(zip(st.inchikey14, st.normalized_smiles)))
train_keys = set(st.inchikey14)
del meta, st
prior = PubChemPrior(pop_dir)
print('MERGER setup', round(time.time() - t0), 's', flush=True)
C = None
if cfg.get('channel_model'):
    try:
        import channel as CH
        from features import molecule_targets
        CH.init(pop_dir, idx, train_keys)
        I, C = CH.run_windows(CH.jobs_of(te, molecule_targets(te), 10.0, 300), procs=max(1, min(4, os.cpu_count() or 1)),
                              log=lambda x: print('MERGER', x, flush=True))
        C = CH.score(C, cfg['channel_model'])
        print('MERGER channel', C.shape, round(time.time() - t0), 's', flush=True)
    except Exception as e:
        import traceback; traceback.print_exc(); C = None
        if cfg.get('require_channel'):
            raise
D = build_features(root, te, train_keys, prior, analog_index=idx, channel=C, ch_top=int(cfg.get('ch_top', 10)))
D['abs_ppm'] = D.ppm.abs()
print('MERGER features', D.shape, round(time.time() - t0), 's', flush=True)
model = lgb.Booster(model_str=cfg['model'])
for f in cfg['feats']:
    if f not in D:
        D[f] = np.nan
pred = model.predict(D[cfg['feats']].astype(float))
D['pred'] = pred
z = lambda v: ((v - v.mean()) / (v.std(ddof=0) + 1e-9)).fillna(0.0)
D['base'] = -D.log_final_rank
D['score'] = D.groupby('molecule_id').base.transform(z) + cfg['alpha'] * D.groupby('molecule_id').pred.transform(z)
sub = pd.read_csv(os.path.join(root, 'submission.csv'), dtype=str, keep_default_na=False)
out, changed, changed1, added = [], 0, 0, 0
for mid, text in zip(sub.molecule_id, sub.smiles):
    old = [s for s in text.split(';') if s]
    g = D[D.molecule_id == mid]
    if len(g) < 2 or not old:
        out.append(text); continue
    g = g.assign(fr=np.where(g.final_rank > 0, g.final_rank, 99)).sort_values(['score', 'fr'], ascending=[False, True])
    new = list(dict.fromkeys(g.smiles))
    lead_keep = cfg.get('keep1', False) or (cfg.get('lib_keep', 9) <= float(g.lib_max.iloc[0]))
    sc = dict(zip(g.smiles, g.score))
    if cfg.get('margin') is not None and old[0] in sc and new[0] != old[0] and sc[new[0]] - sc[old[0]] < cfg['margin']:
        lead_keep = True
    if lead_keep and old[0] in sc:
        new = [old[0]] + [x for x in new if x != old[0]]
    new += [x for x in old if x not in set(new)]          # stack candidates the feature builder could not parse
    new = new[:25]
    changed += new != old[:25]; changed1 += new[0] != old[0]; added += len(set(new) - set(old))
    out.append(';'.join(new))
res = pd.DataFrame(dict(molecule_id=sub.molecule_id, smiles=out))
res.to_csv(out_path, index=False)
json.dump(dict(changed=int(changed), changed_top1=int(changed1), added=int(added), seconds=round(time.time() - t0)),
          open(out_path + '.json', 'w'))
print('MERGER done: lists changed', changed, 'rank-1 changed', changed1, round(time.time() - t0), 's', flush=True)
'''

CELL = r'''# ===== Learned final merger (dhruvjyoti): pipeline evidence + analog propagation + PubChem priors =====
import json as _mg_json, os as _mg_os, subprocess as _mg_sp, sys as _mg_sys, time as _mg_time, glob as _mg_glob, shutil as _mg_sh
from pathlib import Path as _MgPath
_MG_SOURCES = %(sources)s
_MG_MODEL = %(model)s
def _mg_run():
    root = _MgPath('/kaggle/working'); code = root / 'merger_code'; code.mkdir(exist_ok=True)
    for n, t in _MG_SOURCES.items():
        (code / n).write_text(t)
    (code / 'model.json').write_text(_MG_MODEL)
    sub = root / 'submission.csv'
    if not sub.is_file():
        print('[merger] no submission.csv; skipped'); return
    if isinstance(globals().get('NB_T0'), (int, float)) and _mg_time.time() - NB_T0 > 8.0 * 3600:
        print('[merger] session guard: skipped'); return
    pops = sorted(_mg_glob.glob('/kaggle/input/**/pc_ik14.npy', recursive=True), key=len)
    if not pops:
        print('[merger] popularity arrays not attached; skipped'); return
    comp = str(COMP)
    out = root / 'submission_merger.csv'
    left = 8.6 * 3600 - (_mg_time.time() - NB_T0) if isinstance(globals().get('NB_T0'), (int, float)) else 3000
    p = _mg_sp.run([_mg_sys.executable, str(code / 'merger_runner.py'), str(code), comp, str(root), _mg_os.path.dirname(pops[0]),
                    str(code / 'model.json'), str(out)], capture_output=True, text=True, timeout=max(300, min(3000, int(left))))
    print(p.stdout[-3000:], p.stderr[-2000:])
    if p.returncode != 0 or not out.is_file():
        print('[merger] runner failed; submission kept'); return
    import pandas as _mg_pd
    a = _mg_pd.read_csv(sub, dtype=str, keep_default_na=False); b = _mg_pd.read_csv(out, dtype=str, keep_default_na=False)
    assert list(a.molecule_id) == list(b.molecule_id)
    from rdkit import Chem as _mg_Chem
    for x, y in zip(a.smiles, b.smiles):
        ys = y.split(';')
        assert 1 <= len(ys) <= 25 and len(set(ys)) == len(ys) and all(ys), 'bad merger list'
        assert all(_mg_Chem.MolFromSmiles(t) is not None for t in ys if t not in x.split(';')), 'unparseable new SMILES'
    _mg_sh.copy(sub, root / 'submission_pre_merger.csv')
    _mg_sh.copy(out, sub)
    print('[merger] applied', open(str(out) + '.json').read())
try:
    _mg_run()
except Exception as _mg_e:
    import traceback; traceback.print_exc(); print('[merger] failed; submission kept')
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('base'); ap.add_argument('model'); ap.add_argument('out_dir'); ap.add_argument('kernel_id')
    ap.add_argument('--margin', type=float, default=None); ap.add_argument('--alpha', type=float, default=None)
    ap.add_argument('--keep1', action='store_true'); ap.add_argument('--lib_keep', type=float, default=9.0)
    ap.add_argument('--channel_model', default=None); ap.add_argument('--ch_top', type=int, default=10)
    a = ap.parse_args()
    cfg = json.load(open(a.model))
    if a.alpha is not None:
        cfg['alpha'] = a.alpha
    cfg.update(margin=a.margin, keep1=a.keep1, lib_keep=a.lib_keep, ch_top=a.ch_top)
    if a.channel_model:
        cfg['channel_model'] = open(a.channel_model).read()
    sources = {n: open(os.path.join(HERE, n)).read() for n in ['analog.py', 'features.py', 'libbuild.py', 'channel.py', 'formula_expl.py']}
    sources['merger_runner.py'] = RUNNER
    cell = CELL % dict(sources=repr(sources), model=repr(json.dumps(cfg)))
    compile(cell, 'merger_cell', 'exec'); compile(RUNNER, 'runner', 'exec')
    nb = json.load(open(a.base))
    DROP = ('# ── Figure: what this notebook adds', '# The field as the Code tab showed it', '# What each layer did in this run')
    nb['cells'] = [c for c in nb['cells'] if c['cell_type'] == 'code' and not ''.join(c['source']).lstrip().startswith(DROP)]
    for c in nb['cells']:
        c['outputs'] = []; c['execution_count'] = None
    nb['cells'].append(dict(cell_type='code', execution_count=None, metadata={}, outputs=[], source=cell))
    os.makedirs(a.out_dir, exist_ok=True)
    name = a.kernel_id.split('/')[1]
    json.dump(nb, open(os.path.join(a.out_dir, name + '.ipynb'), 'w'))
    meta = json.load(open(os.path.join(os.path.dirname(a.base), 'kernel-metadata.json')))
    meta.update(id=a.kernel_id, title=name.replace('-', ' '), code_file=name + '.ipynb', is_private=True)
    json.dump(meta, open(os.path.join(a.out_dir, 'kernel-metadata.json'), 'w'), indent=1)
    size = os.path.getsize(os.path.join(a.out_dir, name + '.ipynb'))
    print('written', a.out_dir, 'notebook bytes', size)
    assert size < 1_000_000, 'notebook exceeds the 1 MB source limit'


if __name__ == '__main__':
    main()
