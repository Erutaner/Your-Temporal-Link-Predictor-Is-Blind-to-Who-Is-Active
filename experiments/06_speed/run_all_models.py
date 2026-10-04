# -*- coding: utf-8 -*-
"""Time the twelve models on one exclusive GPU (training time per epoch, scoring time of the test split, memory).

    python run_all_models.py --data_root <root> --out_dir <dir> [--models m1,m2] [--datasets d1,d2] [--repeats 3]
                             [--patched_dir experiments/06_speed/patched] [--dry_run]

Run build_timing_patches.py first: it writes copies of the released training / evaluation scripts of DyGLib, TPNet
and DSRD with a timing hook that is inert unless the environment variables below are set.  Nothing else may use the
GPU while this runs.  For every (model, stream) and repeat:
  training   the model's own training script with SPEED_TRAIN_BATCHES=300: after a warm-up, 300 training batches
             of 200 events are timed and the run stops (the barely-trained model is saved as the run's checkpoint);
  scoring    the model's own evaluation script (random negatives, batches of 200) with SPEED_EVAL_BATCHES=500: after
             a warm-up, up to 500 test batches are timed.
Each phase prints one line (SPEED-TRAIN / SPEED-EVAL) with the seconds, ms per batch, the number of batches in the
whole split / epoch and the peak GPU memory; the driver appends it to <out_dir>/results.jsonl together with the
model, stream and repeat.  Our model is timed on one full training epoch (SFS_SPEED=2: two epochs, the second is
reported) and on the full scoring pass of the test split (SFS_SPEED=1).  make_table_all_models.py and the figure
scripts turn results.jsonl into the per-split / per-epoch seconds: for the released models ms per batch times the
number of batches, for ours the measured pass.  Copy results.jsonl to results/speed/ to redraw the paper figures.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common import MODEL, MAIN_FLAGS, flag_string, link_data  # noqa: E402

DATASETS = ['uci', 'wikipedia', 'enron', 'reddit', 'lastfm', 'Flights']
DYGLIB = ['JODIE', 'DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer']
MODELS = DYGLIB + ['EdgeBank', 'TPNet', 'DSRD', 'SFS']            # SFS = our model
TRAIN_BATCHES = 300
EVAL_BATCHES = 500


def run(cmd, log, env, timeout):
    """Run a shell command with the extra env, capture its output to log, return (rc, text)."""
    e = dict(os.environ)
    e.update(env)
    with open(log, 'w') as fh:
        try:
            p = subprocess.run(cmd, shell=True, stdout=fh, stderr=subprocess.STDOUT, env=e, timeout=timeout)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            rc = -9
    return rc, open(log, errors='ignore').read()


def grab(text, tag):
    """The last '<tag> {json}' record in the captured output (tqdm leaves it on a carriage-return line)."""
    m = re.findall(tag + r' (\{[^{}]*\})', text)
    return json.loads(m[-1]) if m else None


def commands(model, ds, py, patched, data_root, out_dir):
    """(train_cmd or None, eval_cmd, cwd) for one model on one stream."""
    common = '--dataset_name %s --load_best_configs --num_runs 1 --seed_start 0 --gpu 0' % ds
    if model in DYGLIB:
        cwd = os.path.join(patched, 'DyGLib')
        return ('%s -u train_link_prediction.py --model_name %s %s' % (py, model, common),
                '%s -u evaluate_link_prediction.py --model_name %s %s --negative_sample_strategy random' % (py, model, common), cwd)
    if model == 'EdgeBank':
        cwd = os.path.join(patched, 'DyGLib')
        return (None, '%s -u evaluate_link_prediction.py --model_name EdgeBank %s --negative_sample_strategy random' % (py, common), cwd)
    if model == 'TPNet':
        cwd = os.path.join(patched, 'TPNet')
        tp = common.replace(' --seed_start 0', '')        # the released TPNet has no --seed_start (run 0 = seed 0)
        return ('%s -u train_link_prediction.py --prefix speed --model_name TPNet --use_random_projection %s' % (py, tp),
                '%s -u evaluate_link_prediction.py --prefix speed --model_name TPNet --use_random_projection %s --negative_sample_strategy random' % (py, tp), cwd)
    if model == 'DSRD':
        cwd = os.path.join(patched, 'DSRD')
        return ('%s -u train_link_prediction.py --model_name DSRD %s' % (py, common),
                '%s -u evaluate_link_prediction.py --model_name DSRD %s --negative_sample_strategy random' % (py, common), cwd)
    if model == 'SFS':
        out = os.path.join(out_dir, 'sfs', ds)
        cfg = os.path.join(MODEL, 'configs', ds + '.json')
        return ('mkdir -p %s && %s -u -W ignore scripts/train_sfs.py --config %s --data_root %s --out_dir %s --run 0 --tag _speed '
                '--epochs 2 --patience 2 --print_every 1 %s' % (out, py, cfg, data_root, out, flag_string(MAIN_FLAGS[ds])),
                '%s -u -W ignore scripts/evaluate_sfs.py --ckpt %s/%s_speed_run0.pt --config %s --data_root %s '
                '--settings transductive --periods test --strategies random --betas 0 --out %s/%s_speed_eval.json'
                % (py, out, ds, cfg, data_root, out, ds), MODEL)
    raise ValueError(model)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--patched_dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'patched'))
    p.add_argument('--models', default=','.join(MODELS))
    p.add_argument('--datasets', default=','.join(DATASETS))
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--timeout', type=int, default=4 * 3600, help='per phase, seconds')
    p.add_argument('--dry_run', action='store_true')
    a = p.parse_args()
    py = sys.executable
    logs = os.path.join(a.out_dir, 'logs')
    results = os.path.join(a.out_dir, 'results.jsonl')
    if not a.dry_run:
        os.makedirs(logs, exist_ok=True)
        for tree in ('DyGLib', 'TPNet', 'DSRD'):
            if any(commands(m, DATASETS[0], py, a.patched_dir, a.data_root, a.out_dir)[2].endswith(tree) for m in a.models.split(',')):
                link_data(os.path.join(a.patched_dir, tree), a.data_root)

    def emit(rec):
        with open(results, 'a') as fh:
            fh.write(json.dumps(rec) + '\n')
        print(json.dumps(rec), flush=True)

    for model in a.models.split(','):
        for ds in a.datasets.split(','):
            tr, ev, cwd = commands(model, ds, py, a.patched_dir, a.data_root, a.out_dir)
            if a.dry_run:
                print('\n# %s %s (in %s)\n%s\n%s' % (model, ds, cwd, tr, ev))
                continue
            for r in range(a.repeats):
                tag = '%s_%s_r%d' % (model, ds, r)
                if tr is not None:
                    env = {'SPEED_TRAIN_BATCHES': str(TRAIN_BATCHES), 'SFS_SPEED': '2', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}
                    t0 = time.time()
                    rc, text = run('cd %s && %s' % (cwd, tr), os.path.join(logs, tag + '.train.log'), env, a.timeout)
                    rec = grab(text, 'SPEED-EPOCH' if model == 'SFS' else 'SPEED-TRAIN')
                    emit({'model': model, 'dataset': ds, 'repeat': r, 'phase': 'train', 'rc': rc, 'wall_s': time.time() - t0,
                          'measure': rec, 'error': None if rec else text[-600:]})
                env = {'SPEED_EVAL_BATCHES': str(EVAL_BATCHES), 'SFS_SPEED': '1', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}
                t0 = time.time()
                rc, text = run('cd %s && %s' % (cwd, ev), os.path.join(logs, tag + '.eval.log'), env, a.timeout)
                rec = grab(text, 'SPEED-SCORE' if model == 'SFS' else 'SPEED-EVAL')
                emit({'model': model, 'dataset': ds, 'repeat': r, 'phase': 'eval', 'rc': rc, 'wall_s': time.time() - t0,
                      'measure': rec, 'error': None if rec else text[-600:]})
    print('done', flush=True)


if __name__ == '__main__':
    main()
