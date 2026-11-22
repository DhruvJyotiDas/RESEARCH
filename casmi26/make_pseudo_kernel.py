"""Patch the public 0.500 stack notebook so that one run scores a custom pseudo-test instead of test.parquet.

Purpose: leak-free offline evaluation of the whole stack (pseudo-test structures are absent from train.parquet,
so no community model has seen them). The patched notebook is for evaluation runs only, never for submission.

Patches
  1. session cell: redirect COMP / glob('**/test.parquet') / Path.rglob('test.parquet') to /kaggle/working/pcomp
  2. frozen 0.465 pipeline (base64 inside the pinned-sources cell): honour $PSEUDO_COMP in its three data-location
     sites; the pinned SHA-256 is recomputed and replaced
  3. external-library overlay: drop reference spectra whose InChIKey-14 is in the pseudo-test exclusion list
Usage: python make_pseudo_kernel.py <base.ipynb> <out_dir> <kernel_id> <pseudo_dataset_slug>
"""
import ast
import base64
import copy
import hashlib
import json
import os
import sys

base_path, out_dir, kernel_id, pseudo_slug = sys.argv[1:5]
nb = json.load(open(base_path))
src = lambda i: ''.join(nb['cells'][i]['source'])


def set_src(i, text):
    nb['cells'][i]['source'] = text


def replace_once(text, old, new, what):
    assert text.count(old) == 1, f'{what}: anchor found {text.count(old)} times'
    return text.replace(old, new)


# ---------------------------------------------------------------- 1. session cell redirect
SESSION, ANCHOR = 10, "assert COMP is not None, 'Attach the competition data (enveda-CASMI26-molecule-id-mass-spectra)'\n"
REDIRECT = ANCHOR + r'''# ===== PSEUDO-TEST REDIRECT (offline leak-free evaluation run; never submitted) =====
import glob as _pt_glob, shutil as _pt_shutil, pathlib as _pt_pl
_PT = sorted(_pt_glob.glob('/kaggle/input/**/%s/test.parquet', recursive=True), key=len)
assert _PT, 'pseudo-test dataset not attached'
_pt_src = Path(_PT[0]).parent
_pc = Path('/kaggle/working/pcomp'); _pc.mkdir(parents=True, exist_ok=True)
_pt_shutil.copy(_pt_src / 'test.parquet', _pc / 'test.parquet')
_pt_shutil.copy(_pt_src / 'sample_submission.csv', _pc / 'sample_submission.csv')
if not (_pc / 'train.parquet').exists():
    os.symlink(str(COMP / 'train.parquet'), str(_pc / 'train.parquet'))
COMP = _pc
os.environ['PSEUDO_COMP'] = str(_pc)
os.environ['PSEUDO_EXCLUDE'] = str(_pt_src / 'exclude_ik14.txt')
_pt_orig_glob = _pt_glob.glob
def _pt_pseudo_glob(pattern, *a, **k):
    r = _pt_orig_glob(pattern, *a, **k)
    if str(pattern).endswith('test.parquet'):
        r = [str(_pc / 'test.parquet')] + [x for x in r if not str(x).endswith('test.parquet')]
    return r
_pt_glob.glob = _pt_pseudo_glob
_pt_orig_rglob = _pt_pl.Path.rglob
def _pt_pseudo_rglob(self, pattern, *a, **k):
    if str(pattern) == 'test.parquet':
        return iter([_pc / 'test.parquet'])
    return _pt_orig_rglob(self, pattern, *a, **k)
_pt_pl.Path.rglob = _pt_pseudo_rglob
print('PSEUDO-TEST redirect active:', _pt_src, flush=True)
''' % pseudo_slug
set_src(SESSION, replace_once(src(SESSION), ANCHOR, REDIRECT, 'session'))

# ---------------------------------------------------------------- 2. frozen pipeline
PINNED = 16
text = src(PINNED)
tree = ast.parse(text)
b64 = next(n.value.value for n in ast.walk(tree)
           if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'FROZEN_B64')
frozen = json.loads(base64.b64decode(b64))
fcell = lambda i: ''.join(frozen['cells'][i]['source'])
# cell 4: locate_data_dir
c4 = replace_once(fcell(4), 'def locate_data_dir():\n',
                  "def locate_data_dir():\n    if os.environ.get('PSEUDO_COMP'):\n        return Path(os.environ['PSEUDO_COMP'])\n",
                  'frozen cell 4')
frozen['cells'][4]['source'] = c4
c26 = replace_once(fcell(26), "COMP = os.path.dirname(find('test.parquet'))",
                   "COMP = os.environ.get('PSEUDO_COMP') or os.path.dirname(find('test.parquet'))", 'frozen cell 26')
frozen['cells'][26]['source'] = c26
c28 = replace_once(fcell(28), "assert comp in {f'/kaggle/input/{slug}', f'/kaggle/input/competitions/{slug}'}",
                   "assert comp in {f'/kaggle/input/{slug}', f'/kaggle/input/competitions/{slug}'} or os.environ.get('PSEUDO_COMP')",
                   'frozen cell 28')
frozen['cells'][28]['source'] = c28
raw = json.dumps(frozen, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
new_b64 = base64.b64encode(raw).decode('ascii')
new_sha = hashlib.sha256(raw).hexdigest()
text = replace_once(text, b64, new_b64, 'FROZEN_B64 literal')
old_sha = 'd1e757c2b71b46e6e8a5777f14302a4737d0fdbcd56d8365093c039b93c5fd88'
text = replace_once(text, f"FROZEN_SHA256 = '{old_sha}'", f"FROZEN_SHA256 = '{new_sha}'", 'FROZEN_SHA256')
set_src(PINNED, text)

# ---------------------------------------------------------------- 3. extlib overlay exclusion
OV = 30
t = src(OV)
t = replace_once(t, "            meta_df = pd.read_parquet(meta_hits[0])\n",
                 "            meta_df = pd.read_parquet(meta_hits[0])\n"
                 "            _pt_ex = set(open(os.environ['PSEUDO_EXCLUDE']).read().split()) if os.environ.get('PSEUDO_EXCLUDE') else set()\n"
                 "            meta_df = meta_df[~meta_df.inchikey14.astype(str).isin(_pt_ex)]\n"
                 "            print('[apex-extlib] pseudo-test exclusion, Source A rows kept', len(meta_df))\n",
                 'overlay source A')
t = replace_once(t, "columns=['smiles', 'exact_mass', 'ion_mode', 'mzs', 'intensities', 'is_casmi_subset'])\n",
                 "columns=['smiles', 'exact_mass', 'ion_mode', 'mzs', 'intensities', 'is_casmi_subset', 'inchikey'])\n"
                 "            _pt_ex = set(open(os.environ['PSEUDO_EXCLUDE']).read().split()) if os.environ.get('PSEUDO_EXCLUDE') else set()\n"
                 "            mb_df = mb_df[~mb_df.inchikey.astype(str).str[:14].isin(_pt_ex)]\n",
                 'overlay source B')
set_src(OV, t)

# ---------------------------------------------------------------- syntax check of every code cell (compile only)
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        compile(''.join(c['source']), f'cell{i}', 'exec')
for i, c in enumerate(frozen['cells']):
    if c['cell_type'] == 'code':
        s = ''.join(c['source'])
        if s.lstrip().startswith(('!', '%')):
            continue
        try:
            compile(s, f'frozen{i}', 'exec')
        except SyntaxError as e:
            print('frozen cell', i, 'syntax (pre-existing?)', e)

# drop documentation-only markdown cells and code outputs (Kaggle 1 MB source limit)
nb['cells'] = [c for c in nb['cells'] if c['cell_type'] == 'code']
for c in nb['cells']:
    c['outputs'] = []; c['execution_count'] = None
os.makedirs(out_dir, exist_ok=True)
name = kernel_id.split('/')[1]
json.dump(nb, open(os.path.join(out_dir, name + '.ipynb'), 'w'))
meta = json.load(open(os.path.join(os.path.dirname(base_path), 'kernel-metadata.json')))
meta.update(id=kernel_id, title=name.replace('-', ' '), code_file=name + '.ipynb', is_private=True)
meta['dataset_sources'] = list(meta['dataset_sources']) + [f'dhruvjyoti/{pseudo_slug}']
json.dump(meta, open(os.path.join(out_dir, 'kernel-metadata.json'), 'w'), indent=1)
print('patched notebook written; frozen sha', new_sha)
