# -*- coding: utf-8 -*-
"""Shared locations and settings for the experiment scripts.

Everything is relative to the package root (the folder that holds model/, baselines/, results/ and
experiments/), so the package can be unpacked anywhere.  Import with

    import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from common import ROOT, RESULTS, OUTPUTS, MODEL, DATASETS, MAIN_FLAGS
"""
import os
import shutil
import subprocess
import sys

try:                                   # the table scripts echo markdown with Greek letters
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, 'model')            # our model: tgmin/, scripts/, tools/, configs/
BASELINES = os.path.join(ROOT, 'baselines')    # the baseline trees (corrected released code)
RESULTS = os.environ.get('REPRO_RESULTS', os.path.join(ROOT, 'results'))   # evaluation files and timing records; override to point the table scripts at your own runs
OUTPUTS = os.environ.get('REPRO_OUTPUTS', os.path.join(ROOT, 'outputs'))   # tables (.tex/.md) and figures (.pdf/.png) are written here

# the thirteen streams in the order of the paper's tables: (folder / config name, display name)
DATASETS = [('wikipedia', 'Wikipedia'), ('reddit', 'Reddit'), ('mooc', 'MOOC'), ('lastfm', 'LastFM'),
            ('enron', 'Enron'), ('SocialEvo', 'Social Evo.'), ('uci', 'UCI'), ('Flights', 'Flights'),
            ('CanParl', 'Can. Parl.'), ('USLegis', 'US Legis.'), ('UNtrade', 'UN Trade'),
            ('UNvote', 'UN Vote'), ('Contacts', 'Contact')]
NAMES = dict(DATASETS)

# Training settings of the main results, one line per stream.  They were chosen on the validation
# split (random negatives) and are passed to model/scripts/train_sfs.py explicitly; model/configs/<stream>.json
# holds the same values (a run without these flags differs only in fresh_state), and anything not listed here
# (block mode, epochs, number of timescales, ...) comes from the config or the script's defaults.  Meaning of the keys:
#   t_train      number of training blocks C (streams with one block per time step take it from the config)
#   e_train      1 = train the node table, 0 = fixed random node codes
#   dropout, wd  dropout and weight decay
#   neg_pool     training negatives drawn from all training destinations ('train') or from the
#                destinations seen so far ('seen')
#   fresh_state  1 = within-block events also update the partner-sum state (Enron and Contact only)
#   set_M        number of recent events read by the recent interaction encoding
MAIN_FLAGS = {
    'wikipedia': dict(t_train=96,  e_train=0, dropout=0.0, wd=0.0,  neg_pool='seen',  fresh_state=0, set_M=20, lr=0.001),
    'reddit':    dict(t_train=48,  e_train=0, dropout=0.0, wd=0.01, neg_pool='seen',  fresh_state=0, set_M=20, lr=0.001),
    'mooc':      dict(t_train=192, e_train=0, dropout=0.0, wd=0.0,  neg_pool='seen',  fresh_state=0, set_M=20, lr=0.001),
    'lastfm':    dict(t_train=192, e_train=0, dropout=0.0, wd=0.0,  neg_pool='seen',  fresh_state=0, set_M=20, lr=0.001),
    'enron':     dict(t_train=192, e_train=0, dropout=0.0, wd=0.0,  neg_pool='seen',  fresh_state=1, set_M=20, lr=0.001),
    'SocialEvo': dict(t_train=96,  e_train=0, dropout=0.0, wd=0.01, neg_pool='seen',  fresh_state=0, set_M=20, lr=0.001),
    'uci':       dict(t_train=192, e_train=1, dropout=0.2, wd=0.0,  neg_pool='train', fresh_state=0, set_M=20, lr=0.001),
    'Flights':   dict(              e_train=1, dropout=0.0, wd=0.01, neg_pool='train', fresh_state=0, set_M=20, lr=0.001),
    'CanParl':   dict(              e_train=0, dropout=0.0, wd=0.0,  neg_pool='seen',  fresh_state=0, set_M=20, lr=0.001),
    'USLegis':   dict(              e_train=1, dropout=0.2, wd=0.0,  neg_pool='train', fresh_state=0, set_M=20, lr=0.001),
    'UNtrade':   dict(              e_train=0, dropout=0.0, wd=0.0,  neg_pool='seen',  fresh_state=0, set_M=20, lr=0.001),
    'UNvote':    dict(              e_train=0, dropout=0.0, wd=0.01, neg_pool='seen',  fresh_state=0, set_M=20, lr=0.001),
    'Contacts':  dict(t_train=192, e_train=0, dropout=0.0, wd=0.0,  neg_pool='seen',  fresh_state=1, set_M=20, lr=0.001),
}
MAIN_SEEDS = [0, 1, 2, 3, 4]
# patience of the early stopping used for the main runs and the ablations
PATIENCE = {ds: (50 if ds in ('lastfm', 'Flights') else 100) for ds, _ in DATASETS}

# the five streams of the ablations and of the transfer study (three seeds each)
ABLATION_DATASETS = ['wikipedia', 'reddit', 'uci', 'enron', 'lastfm']
ABLATION_SEEDS = [0, 1, 2]


def flag_string(flags):
    """dict -> '--k v --k v' for train_sfs.py."""
    parts = []
    for k, v in flags.items():
        parts.append('--%s %s' % (k, ('%g' % v) if isinstance(v, float) else str(v)))
    return ' '.join(parts)


def main_run_files(out_dir, ds, seed):
    """Checkpoint and evaluation file of one main run, as train_sfs.py / evaluate_sfs.py name them."""
    base = os.path.join(out_dir, ds, '%s_final_run%d' % (ds, seed))
    return base + '.pt', base + '_eval.json'


def speed_medians():
    """Per-model, per-stream medians of the all-model timing records (results/speed/results.jsonl):
    returns (scoring, training) dicts, model -> {stream: seconds}, with the released models extrapolated from
    their timed batches to the whole split / epoch and our model ('Ours') measured in full."""
    import json
    import collections
    import numpy as np
    recs = [json.loads(l) for l in open(os.path.join(RESULTS, 'speed', 'results.jsonl'), encoding='utf-8') if l.strip()]
    by = collections.defaultdict(list)
    for r in recs:
        if r['measure'] is not None:
            by[(r['model'], r['dataset'], r['phase'])].append(r['measure'])
    scoring, training = collections.defaultdict(dict), collections.defaultdict(dict)
    for (model, ds, phase), ms in by.items():
        name = 'Ours' if model == 'SFS' else model
        if model == 'SFS':
            v = float(np.median([m['train_seconds'] if phase == 'train' else m['seconds'] for m in ms]))
        else:
            v = float(np.median([m['ms_per_batch'] * m['total_batches'] / 1000.0 for m in ms]))
        # rounded as the timing table prints them: whole seconds for an epoch, tenths for the scoring pass
        (training if phase == 'train' else scoring)[name][ds] = float('%.0f' % v) if phase == 'train' else float('%.1f' % v)
    return dict(scoring), dict(training)


def sh(cmd, cwd=None, env=None, dry_run=False):
    """Print and run one shell command; stop the script if it fails."""
    print(cmd, flush=True)
    if dry_run:
        return
    e = dict(os.environ)
    e.update(env or {})
    rc = subprocess.call(cmd, shell=True, cwd=cwd, env=e)
    if rc != 0:
        raise SystemExit('command failed (exit %d): %s' % (rc, cmd))


def link_data(tree, data_root):
    """The released baseline code reads ./processed_data/<stream>/ml_<stream>.* relative to its own folder:
    make <tree>/processed_data point at <data_root>/processed_data."""
    dst = os.path.join(tree, 'processed_data')
    src = os.path.join(data_root, 'processed_data')
    if os.path.exists(dst) or os.path.islink(dst):
        return
    if not os.path.isdir(src):
        raise SystemExit('no processed_data folder under %s (see data/README.md)' % data_root)
    try:
        os.symlink(src, dst, target_is_directory=True)
    except (OSError, NotImplementedError):
        raise SystemExit('could not create the link %s -> %s; create it (or copy the folder) by hand' % (dst, src))
