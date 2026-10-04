# -*- coding: utf-8 -*-
"""Cost of our model against its own settings (number of scales, chunk count, within-chunk records) and against the
stream length; one exclusive GPU, our model only.

    python run_cost_curves.py --data_root <root> --out_dir <dir> [--exps S3,S2] [--repeats 3] [--dry_run]

Same conventions as run_all_models.py: two-epoch training with SFS_SPEED=2 (the second epoch's SPEED-EPOCH record:
seconds, steps, events, peak memory) and a full scoring pass of the test split with SFS_SPEED=1 (SPEED-SCORE:
seconds, positives, peak memory); three repeats; every record goes to <out_dir>/results.jsonl.
S3  five streams, each with the settings of the main runs plus one change:
      K1 = a single rate; pd1 / pd2 / pd4 = one / two (the main setting) / four rates per decade;
      fr{0,1}_t{48,96,192} = within-chunk records off / on x number of training chunks
    (the main configuration therefore appears twice, as pd2 and as fr1_t<T>).
S2  LastFM and Flights cut to their first 10 / 30 / 100 % of events; the prefix streams <ds>_p10 and <ds>_p30 are
    built here under <data_root>/processed_data/ from the first rows of the csv and the matching edge-feature rows,
    with the prefix's own 70 / 15 / 15 split, and their configs are written to <out_dir>/configs/.
Copy results.jsonl to results/speed/results_s23.jsonl to redraw the cost figure and the stream-length table.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common import MODEL, MAIN_FLAGS, flag_string  # noqa: E402

S3_DATASETS = ['uci', 'wikipedia', 'enron', 'reddit', 'lastfm']
S3_VARIANTS = [('K1', '--kmin 1 --per_decade 0'), ('pd1', '--per_decade 1'), ('pd2', ''), ('pd4', '--per_decade 4 --kmax 32')]
for _fr in (1, 0):
    for _t in (48, 96, 192):
        S3_VARIANTS.append(('fr%d_t%d' % (_fr, _t), '--fresh_records %d --t_train %d' % (_fr, _t)))
S2_DATASETS = ['lastfm', 'Flights']
S2_PREFIXES = [('p10', 0.10), ('p30', 0.30), ('p100', 1.0)]


def run(cmd, log, env, timeout):
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
    m = re.findall(tag + r' (\{[^{}]*\})', text)
    return json.loads(m[-1]) if m else None


def build_prefix(ds, name, frac, data_root, cfg_dir):
    """<data_root>/processed_data/<ds>_<name>/ml_<ds>_<name>.{csv,npy,_node.npy} from the first frac of the stream,
    plus <cfg_dir>/<ds>_<name>.json (the parent's config with the dataset renamed).  Idempotent."""
    import numpy as np
    import pandas as pd
    new = '%s_%s' % (ds, name)
    d = os.path.join(data_root, 'processed_data', new)
    cfg = os.path.join(cfg_dir, new + '.json')
    if os.path.exists(os.path.join(d, 'ml_%s_node.npy' % new)) and os.path.exists(cfg):
        return cfg
    src = os.path.join(data_root, 'processed_data', ds)
    df = pd.read_csv(os.path.join(src, 'ml_%s.csv' % ds), index_col=0)
    ts = df.ts.values
    assert (np.diff(ts) >= 0).all(), 'stream not time-sorted'
    assert (df.idx.values == np.arange(1, len(df) + 1)).all(), 'idx not 1..n'
    m = int(round(frac * len(df)))
    ef = np.load(os.path.join(src, 'ml_%s.npy' % ds))
    assert ef.shape[0] == len(df) + 1
    os.makedirs(d, exist_ok=True)
    df.iloc[:m].to_csv(os.path.join(d, 'ml_%s.csv' % new))
    np.save(os.path.join(d, 'ml_%s.npy' % new), ef[:m + 1])
    np.save(os.path.join(d, 'ml_%s_node.npy' % new), np.load(os.path.join(src, 'ml_%s_node.npy' % ds)))
    c = json.load(open(os.path.join(MODEL, 'configs', ds + '.json')))
    c['dataset'] = new
    os.makedirs(cfg_dir, exist_ok=True)
    json.dump(c, open(cfg, 'w'), indent=1)
    print('built prefix %s: %d of %d events' % (new, m, len(df)), flush=True)
    return cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--exps', default='S3,S2')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--timeout', type=int, default=3600, help='per phase, seconds')
    p.add_argument('--dry_run', action='store_true')
    a = p.parse_args()
    py = sys.executable
    logs = os.path.join(a.out_dir, 'logs')
    results = os.path.join(a.out_dir, 'results.jsonl')
    if not a.dry_run:
        os.makedirs(logs, exist_ok=True)

    def emit(rec):
        with open(results, 'a') as fh:
            fh.write(json.dumps(rec) + '\n')
        print(json.dumps(rec), flush=True)

    def measure(exp, ds, variant, cfg, flags):
        out = os.path.join(a.out_dir, exp, ds, variant)
        name = os.path.splitext(os.path.basename(cfg))[0]
        tr = ('mkdir -p %s && %s -u -W ignore scripts/train_sfs.py --config %s --data_root %s --out_dir %s --run 0 --tag _speed '
              '--epochs 2 --patience 2 --print_every 1 %s' % (out, py, cfg, a.data_root, out, flags))
        ev = ('%s -u -W ignore scripts/evaluate_sfs.py --ckpt %s/%s_speed_run0.pt --config %s --data_root %s '
              '--settings transductive --periods test --strategies random --betas 0 --out %s/%s_speed_eval.json'
              % (py, out, name, cfg, a.data_root, out, name))
        if a.dry_run:
            print('\n# %s %s %s\n%s\n%s' % (exp, ds, variant, tr, ev))
            return
        for r in range(a.repeats):
            tag = '%s_%s_%s_r%d' % (exp, ds, variant, r)
            env = {'SFS_SPEED': '2', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}
            t0 = time.time()
            rc, text = run('cd %s && %s' % (MODEL, tr), os.path.join(logs, tag + '.train.log'), env, a.timeout)
            rec = grab(text, 'SPEED-EPOCH')
            k = re.findall(r'decades=([0-9.]+) K=([0-9]+)', text)
            emit({'exp': exp, 'dataset': ds, 'variant': variant, 'flags': flags, 'repeat': r, 'phase': 'train', 'rc': rc,
                  'wall_s': time.time() - t0, 'K': int(k[-1][1]) if k else None, 'measure': rec,
                  'error': None if rec else text[-600:]})
            env = {'SFS_SPEED': '1', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}
            t0 = time.time()
            rc, text = run('cd %s && %s' % (MODEL, ev), os.path.join(logs, tag + '.eval.log'), env, a.timeout)
            rec = grab(text, 'SPEED-SCORE')
            emit({'exp': exp, 'dataset': ds, 'variant': variant, 'flags': flags, 'repeat': r, 'phase': 'eval', 'rc': rc,
                  'wall_s': time.time() - t0, 'measure': rec, 'error': None if rec else text[-600:]})

    for exp in a.exps.split(','):
        if exp == 'S3':
            for ds in S3_DATASETS:
                for variant, extra in S3_VARIANTS:
                    measure('S3', ds, variant, os.path.join(MODEL, 'configs', ds + '.json'), (flag_string(MAIN_FLAGS[ds]) + ' ' + extra).strip())
        elif exp == 'S2':
            for ds in S2_DATASETS:
                for name, frac in S2_PREFIXES:
                    cfg = os.path.join(MODEL, 'configs', ds + '.json') if frac >= 1.0 else (
                        None if a.dry_run else build_prefix(ds, name, frac, a.data_root, os.path.join(a.out_dir, 'configs')))
                    measure('S2', ds, name, cfg or os.path.join(a.out_dir, 'configs', '%s_%s.json' % (ds, name)), flag_string(MAIN_FLAGS[ds]))
    print('done', flush=True)


if __name__ == '__main__':
    main()
