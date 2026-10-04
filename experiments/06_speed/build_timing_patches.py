# -*- coding: utf-8 -*-
"""Speed-benchmark instrumentation for the released DyGLib-derived trees (DyGLib, TPNet, DSRD).

For each tree, three files get env-gated hooks (no effect when the variables are unset):
  train_link_prediction.py    SPEED_TRAIN_BATCHES=N: after a warm-up (20 batches, fewer on short splits) time N training batches
                              (CUDA-synchronised), print 'SPEED-TRAIN {...}', save the model as the run's
                              checkpoint and exit.
  evaluate_models_utils.py    SPEED_EVAL_BATCHES=N: once armed, the evaluation loop times N batches after
                              the same warm-up, prints 'SPEED-EVAL {...}' and exits.
  evaluate_link_prediction.py arms the hook right before the (transductive) test-split evaluation.
Peak memory = torch.cuda.max_memory_allocated (reset at the start of the timed window) and
torch.cuda.max_memory_reserved.  Output: patched/<tree>/ next to this script, a full copy of each tree (without saved
models, results, logs and data) in which the three files above are replaced by their instrumented versions.
"""
import ast
import io
import os
import re
import shutil

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common import BASELINES  # noqa: E402

# source trees: DyGLib (with the fixed-length-padding DyGFormer), TPNet as released, DSRD with the row correction
TREES = {
    'DyGLib': os.path.join(BASELINES, 'DyGLib'),
    'TPNet': os.path.join(BASELINES, 'TPNet_released'),
    'DSRD': os.path.join(BASELINES, 'DSRD_honest'),
}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'patched')

HOOK_STATE = '''
# SPEED BENCHMARK: env-gated timing of the evaluation loop; inert unless armed by the caller.
import os as _sp_os, time as _sp_time, json as _sp_json
_SP = {'on': False, 'count': 0, 'warm': 20, 'n': 500, 't0': None, 'info': {}}


def speed_arm(**info):
    n = int(_sp_os.environ.get('SPEED_EVAL_BATCHES', '0'))
    if n <= 0:
        return
    total = int(info.get('total_batches', 0))
    _SP.update({'on': True, 'count': 0, 'n': n, 'warm': min(20, max(0, total // 5)), 't0': None, 'info': info})


def _speed_tick():
    """Called at the top of every evaluation batch.  Returns True when the caller must stop."""
    if not _SP['on']:
        return False
    import torch as _t
    _SP['count'] += 1
    if _SP['count'] == _SP['warm'] + 1:
        if _t.cuda.is_available():
            _t.cuda.synchronize(); _t.cuda.reset_peak_memory_stats()
        _SP['t0'] = _sp_time.time()
        return False
    total = int(_SP['info'].get('total_batches', 0))
    done = _SP['count'] - 1 - _SP['warm']          # timed batches completed before this one
    target = min(_SP['n'], max(1, total - _SP['warm'] - 1))   # stop inside this loop (the last batch stays untimed)
    if _SP['t0'] is not None and done >= target:
        if _t.cuda.is_available():
            _t.cuda.synchronize()
        sec = _sp_time.time() - _SP['t0']
        rec = dict(_SP['info'])
        rec.update({'phase': 'eval', 'timed_batches': int(done), 'seconds': sec,
                    'ms_per_batch': 1000.0 * sec / max(done, 1),
                    'peak_mem_alloc_mb': (_t.cuda.max_memory_allocated() / 2 ** 20) if _t.cuda.is_available() else 0.0,
                    'peak_mem_reserved_mb': (_t.cuda.max_memory_reserved() / 2 ** 20) if _t.cuda.is_available() else 0.0})
        print('SPEED-EVAL ' + _sp_json.dumps(rec), flush=True)
        _SP['on'] = False
        _sp_os._exit(0)
    return False

'''

LOOP_ANCHOR = 'for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):'
TRAIN_HOOK = '''
{ind}# SPEED BENCHMARK: time SPEED_TRAIN_BATCHES training batches after a warm-up, save, exit
{ind}if int(os.environ.get('SPEED_TRAIN_BATCHES', '0')) > 0:
{ind}    import time as _sp_time, json as _sp_json
{ind}    _n = int(os.environ['SPEED_TRAIN_BATCHES'])
{ind}    _total = len(train_idx_data_loader)
{ind}    _warm = min(20, max(0, _total // 5))
{ind}    _g = globals().setdefault('_SPEED_TRAIN', {{'count': 0, 't0': None}})
{ind}    _g['count'] += 1
{ind}    if _g['count'] == _warm + 1:
{ind}        if torch.cuda.is_available():
{ind}            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
{ind}        _g['t0'] = _sp_time.time()
{ind}    _done = _g['count'] - _warm
{ind}    if _g['t0'] is not None and (_done >= _n or _g['count'] >= _total):
{ind}        if torch.cuda.is_available():
{ind}            torch.cuda.synchronize()
{ind}        _sec = _sp_time.time() - _g['t0']
{ind}        print('SPEED-TRAIN ' + _sp_json.dumps({{'phase': 'train', 'model': args.model_name, 'dataset': args.dataset_name,
{ind}              'batch_size': int(args.batch_size), 'timed_batches': int(_done), 'seconds': _sec,
{ind}              'ms_per_batch': 1000.0 * _sec / max(_done, 1), 'total_batches': int(_total),
{ind}              'peak_mem_alloc_mb': (torch.cuda.max_memory_allocated() / 2 ** 20) if torch.cuda.is_available() else 0.0,
{ind}              'peak_mem_reserved_mb': (torch.cuda.max_memory_reserved() / 2 ** 20) if torch.cuda.is_available() else 0.0}}), flush=True)
{ind}        {save}
{ind}        os._exit(0)
'''


def rd(path):
    s = io.open(path, encoding='utf-8', newline='').read()
    return s.replace('\r\n', '\n'), ('\r\n' in s)


def wr(path, s, crlf):
    ast.parse(s)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    io.open(path, 'w', encoding='utf-8', newline='\r\n' if crlf else '\n').write(s)


def patch_tree(name, src):
    out = os.path.join(OUT, name)
    if not os.path.isdir(out):                 # a full copy of the tree; the three instrumented files then replace their originals
        shutil.copytree(src, out, ignore=shutil.ignore_patterns('__pycache__', 'saved_models', 'saved_results', 'logs', 'processed_data'))
        for d in ('logs', 'saved_models', 'saved_results'):   # the released scripts expect these folders to exist
            os.makedirs(os.path.join(out, d), exist_ok=True)
    # ---- evaluate_models_utils.py: state + hook at the top of every evaluation loop
    u, crlf = rd(os.path.join(src, 'evaluate_models_utils.py'))
    assert 'from utils.DataLoader import Data' in u, name
    u = u.replace('from utils.DataLoader import Data\n', 'from utils.DataLoader import Data\n' + HOOK_STATE, 1)
    anchors = (LOOP_ANCHOR, 'for batch_idx, test_data_indices in enumerate(test_idx_data_loader_tqdm):')   # the EdgeBank loop
    n_loops = sum(u.count(x) for x in anchors)
    assert n_loops >= 1, name
    lines = u.split('\n')
    out_lines = []
    for ln in lines:
        out_lines.append(ln)
        if ln.strip() in anchors:
            ind = ln[:len(ln) - len(ln.lstrip())] + '    '
            out_lines.append(ind + 'if _speed_tick():')
            out_lines.append(ind + '    break')
    u = '\n'.join(out_lines)
    wr(os.path.join(out, 'evaluate_models_utils.py'), u, crlf)
    # ---- evaluate_link_prediction.py: arm before the test-split evaluation(s)
    e, crlf_e = rd(os.path.join(src, 'evaluate_link_prediction.py'))
    n_arm = 0
    for anchor in ('test_losses, test_metrics = evaluate_model_link_prediction(',
                   'test_losses, test_metrics = evaluate_edge_bank_link_prediction(',
                   'evaluate_edge_bank_link_prediction(args=args'):
        idx = e.find(anchor)
        if idx < 0:
            continue
        line_start = e.rfind('\n', 0, idx) + 1
        ind = e[line_start:idx]
        arm = (ind + "import evaluate_models_utils as _emu   # SPEED BENCHMARK\n"
               + ind + "_emu.speed_arm(model=args.model_name, dataset=args.dataset_name, batch_size=int(args.batch_size), "
               "strategy=args.negative_sample_strategy, total_batches=len(test_idx_data_loader))\n")
        e = e[:line_start] + arm + e[line_start:]
        n_arm += 1
    assert n_arm >= 1, name
    assert 'import os' in e, name
    wr(os.path.join(out, 'evaluate_link_prediction.py'), e, crlf_e)
    # ---- train_link_prediction.py: hook after the optimizer step of the training batch loop
    t, crlf_t = rd(os.path.join(src, 'train_link_prediction.py'))
    es, _ = rd(os.path.join(src, 'utils', 'EarlyStopping.py'))
    m = re.search(r'def save_checkpoint\(self, model: nn\.Module(, args[^)]*)?\)', es)
    assert m, name
    save = 'early_stopping.save_checkpoint(model, args)' if m.group(1) else 'early_stopping.save_checkpoint(model)'
    idx = t.find('optimizer.step()')
    assert idx > 0, name
    line_end = t.find('\n', idx) + 1
    line_start = t.rfind('\n', 0, idx) + 1
    ind = t[line_start:idx]
    assert 'early_stopping = EarlyStopping(' in t[:idx], name        # created before the batch loop
    assert 'train_idx_data_loader' in t[:idx], name
    hook = TRAIN_HOOK.format(ind=ind, save=save)
    t = t[:line_end] + hook.lstrip('\n') + t[line_end:]
    assert 'import os' in t and 'import torch' in t, name
    wr(os.path.join(out, 'train_link_prediction.py'), t, crlf_t)
    print(name, 'patched: eval loops %d, arm points %d, save call: %s' % (n_loops, n_arm, save))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='write the instrumented copies of the three baseline trees to patched/')
    ap.add_argument('--out', default=OUT, help='output folder (default: patched/ next to this script)')
    OUT = ap.parse_args().out
    for name, src in TREES.items():
        patch_tree(name, src)
