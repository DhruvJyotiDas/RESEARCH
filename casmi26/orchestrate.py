"""Push queued Kaggle kernels as GPU slots free up and submit finished probe kernels.

State file keeps what has been pushed / submitted so the loop is restartable.
usage: KAGGLE_API_TOKEN=... python orchestrate.py state.json
"""
import json
import os
import subprocess
import sys
import time

COMP = 'enveda-CASMI26-molecule-id-mass-spectra'
KDIR = '/home/user/RESEARCH/casmi26/kernels'
# (folder, kernel id, submit after commit?, submission message)
PLAN = [
    ('probe_top1', 'dhruvjyoti/casmi26-probe-top1', True, 'probe: keep rank 1 only (top-1 accuracy)'),
    # submission reruns hold GPU slots for hours: pseudo-test runs (merger training data) go before the extra probes
    ('pseudo_v1', 'dhruvjyoti/casmi26-pseudotest-run-v1', False, ''),
    ('pseudo_v2', 'dhruvjyoti/casmi26-pseudotest-run-v2', False, ''),
    ('pseudo_v3', 'dhruvjyoti/casmi26-pseudotest-run-v3', False, ''),
    ('probe_trainonly', 'dhruvjyoti/casmi26-probe-trainonly', True, 'probe: keep only train-structure candidates (class-1 share)'),
    ('probe_poolonly', 'dhruvjyoti/casmi26-probe-poolonly', True, 'probe: keep only pool (train+COCONUT) candidates'),
]
ALREADY_PUSHED = {'dhruvjyoti/casmi26-probe-top1', 'dhruvjyoti/casmi26-pseudotest-run-v1'}
OUT = '/home/user/kaggle_out'


def sh(*args):
    r = subprocess.run(list(args), capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()


def status(kid):
    out = sh('kaggle', 'kernels', 'status', kid)
    for s in ('COMPLETE', 'ERROR', 'CANCEL', 'RUNNING', 'QUEUED'):
        if s in out:
            return s
    return out[-80:]


def allowed_now():
    out = sh('python3', '-c', f"from kaggle.api.kaggle_api_extended import KaggleApi as A;a=A();a.authenticate();"
                              f"print(a.competition_get_submission_limits('{COMP}').num_allowed_now)")
    try:
        return int(out.split()[-1])
    except Exception:
        return 0


def main(state_path):
    st = json.load(open(state_path)) if os.path.exists(state_path) else {}
    st.setdefault('pushed', []); st.setdefault('submitted', []); st.setdefault('downloaded', [])
    st['pushed'] = sorted(set(st['pushed']) | ALREADY_PUSHED)
    while True:
        todo = False
        for folder, kid, submit, msg in PLAN:
            if kid not in st['pushed']:
                todo = True
                out = sh('kaggle', 'kernels', 'push', '-p', os.path.join(KDIR, folder))
                print(time.strftime('%H:%M'), 'push', kid, '->', out[-120:], flush=True)
                if 'successfully pushed' in out:
                    st['pushed'].append(kid)
                else:
                    break                      # keep priority order: wait for a slot
            elif submit and kid not in st['submitted']:
                todo = True
                s = status(kid)
                if s == 'COMPLETE' and allowed_now() > 0:
                    out = sh('kaggle', 'competitions', 'submit', COMP, '-k', kid, '-v', '1', '-f', 'submission.csv', '-m', msg)
                    print(time.strftime('%H:%M'), 'submit', kid, '->', out[-160:], flush=True)
                    if 'Submission ref' in out or 'Successfully' in out:
                        st['submitted'].append(kid)
                elif s in ('ERROR', 'CANCEL'):
                    print(time.strftime('%H:%M'), kid, 'failed with', s, flush=True)
                    st['submitted'].append(kid)
            elif not submit and kid not in st['downloaded']:
                todo = True
                s = status(kid)
                if s in ('COMPLETE', 'ERROR', 'CANCEL'):
                    d = os.path.join(OUT, folder); os.makedirs(d, exist_ok=True)
                    out = sh('kaggle', 'kernels', 'output', kid, '-p', d)
                    print(time.strftime('%H:%M'), kid, s, 'downloaded ->', d, out[-100:], flush=True)
                    st['downloaded'].append(kid)
        json.dump(st, open(state_path, 'w'), indent=1)
        if not todo:
            print('all done', flush=True)
            return
        time.sleep(120)


if __name__ == '__main__':
    main(sys.argv[1])
