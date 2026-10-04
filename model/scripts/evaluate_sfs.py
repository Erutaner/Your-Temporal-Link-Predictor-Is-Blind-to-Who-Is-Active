# -*- coding: utf-8 -*-
"""Evaluate one trained run under the reference protocol, all settings at
once (the counterpart of DyGLib's evaluate_link_prediction.py):

  settings   transductive (every val / test edge) and inductive (the val /
             test edges touching a node unseen in training);
  strategies random, historical, inductive negative sampling;
  metrics    AP and AUC-ROC, per batch of 200 positives and their
             negatives, averaged.

Samplers and seeds follow the reference exactly: transductive val / test
seeds 0 / 2 on the full graph's node sets; inductive val / test seeds 1 / 3
on the new-node subset's own nodes; historical / inductive strategies take
last_observed_time = end of train (val) or end of val (test).

Readout: score = f + beta * log lambda_u.  beta is chosen per setting and
strategy on the VALIDATION split from --betas (the largest validation AP;
ties -> the smaller beta) and applied to the test split.  The whole beta
grid is saved so any readout can be read off later.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from tgmin import sfs                                  # noqa: E402
from tgmin.checkpoint import load_run                  # noqa: E402
from tgmin.vendor.utils import NegativeEdgeSampler     # noqa: E402

STRATEGIES = ('random', 'historical', 'inductive')


def fit_activity_mlp(run, hidden=32, epochs=5, batch=4096, seed=0):
    """The small-network activity term: a small classifier of 'the source of a real event
    at t' against 'a random training source at t', on log(1 + counts) of
    the source strictly before t, fitted on the TRAINING period only.  Its
    logit replaces log lambda_u as the readout term b."""
    model, feeder, meta, device = run['model'], run['feeder'], run['meta'], run['device']
    T_train = meta['T'][0]
    train = meta['train_data']
    src = np.asarray(train.src_node_ids, dtype=np.int64)
    tq = np.asarray(train.node_interact_times, dtype=np.float64)
    c_start = np.asarray(feeder.c_start[:T_train], dtype=np.float64)
    local = np.searchsorted(c_start, tq, side='right') - 1
    assert (local >= 0).all()
    pool = np.unique(src)
    rng = np.random.RandomState(seed)
    neg = pool[rng.randint(0, len(pool), len(src))]
    xs, ys = [], []
    with torch.no_grad():
        for c in np.unique(local):
            m_ = local == c
            ent = feeder.entering(int(c))
            states = model.entry_states(ent, feeder, int(c))
            cp = model.read(feeder, int(c), ent, states, src[m_], tq[m_])[2]
            cn = model.read(feeder, int(c), ent, states, neg[m_], tq[m_])[2]
            xs += [torch.log1p(cp).float().cpu(), torch.log1p(cn).float().cpu()]
            ys += [torch.ones(int(m_.sum())), torch.zeros(int(m_.sum()))]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    X = torch.cat(xs).to(device)
    Y = torch.cat(ys).to(device)
    torch.manual_seed(seed)
    mlp = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.GELU(),
                              torch.nn.Linear(hidden, 1)).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-2)
    lossf = torch.nn.BCEWithLogitsLoss()
    n = X.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for lo in range(0, n, batch):
            idx = perm[lo:lo + batch]
            opt.zero_grad()
            loss = lossf(mlp(X[idx]).squeeze(-1), Y[idx])
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        print('[activity mlp] epoch %d loss %.4f (n=%d)' % (ep + 1, tot / n, n), flush=True)
    mlp.eval()
    return mlp


def make_sampler(meta, setting, period, strategy):
    """The reference sampler of one (setting, period, strategy)."""
    if setting == 'transductive':
        d = meta['full_data']
        seed = 0 if period == 'val' else 2
    else:
        d = meta['new_node_val_data' if period == 'val'
                 else 'new_node_test_data']
        seed = 1 if period == 'val' else 3
    if strategy == 'random':
        return NegativeEdgeSampler(src_node_ids=d.src_node_ids,
                                   dst_node_ids=d.dst_node_ids, seed=seed)
    last = (meta['train_data'].node_interact_times[-1] if period == 'val'
            else meta['val_data'].node_interact_times[-1])
    return NegativeEdgeSampler(src_node_ids=d.src_node_ids,
                               dst_node_ids=d.dst_node_ids,
                               interact_times=d.node_interact_times,
                               last_observed_time=last,
                               negative_sample_strategy=strategy, seed=seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--data_root', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch_size', type=int, default=200)
    p.add_argument('--betas', default='0,0.5,1,1.5,2',
                   help='readout coefficients tried on the validation split')
    p.add_argument('--settings', default='transductive,inductive')
    p.add_argument('--strategies', default='random,historical,inductive')
    p.add_argument('--periods', default='val,test',
                   help='val only = model selection during a search')
    p.add_argument('--out', default=None,
                   help='JSON output (default: <ckpt without .pt>_eval.json)')
    p.add_argument('--readout_term', default='ground',
                   choices=['ground', 'count_slow', 'count_fast', 'count_mid', 'recency', 'mlp'],
                   help='the term b of f + beta * b '
                        '(default: the ground process log lambda_u)')
    p.add_argument('--b_only', type=int, default=0, choices=[0, 1],
                   help='score with b alone (f zeroed, beta = 1)')
    a = p.parse_args()
    betas = [float(x) for x in a.betas.split(',')]
    settings = a.settings.split(',')
    strategies = a.strategies.split(',')
    periods = a.periods.split(',')
    run = load_run(a.ckpt, a.config, a.data_root, a.device)
    model, feeder, meta, ground, ts = (run['model'], run['feeder'],
                                       run['meta'], run['ground'], run['ts'])
    T_train, T_val, _ = meta['T']
    if ground is None and a.readout_term == 'ground':
        betas = [0.0]
    if a.b_only:
        betas = [1.0]
    mlp = fit_activity_mlp(run) if a.readout_term == 'mlp' else None
    speed = bool(os.environ.get('SFS_SPEED'))       # speed benchmark: time the scoring passes
    grid = {}
    t0 = time.time()
    for setting in settings:
        grid[setting] = {}
        for period in periods:
            grid[setting][period] = {}
            period_idx = 1 if period == 'val' else 2
            first = T_train if period == 'val' else T_train + T_val
            subset = None if setting == 'transductive' else meta[
                'new_node_val_data' if period == 'val'
                else 'new_node_test_data']
            for strategy in strategies:
                sampler = make_sampler(meta, setting, period, strategy)
                if speed and torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                ts0 = time.time()
                s = sfs.score_split(model, feeder, meta, period_idx, sampler,
                                    a.batch_size, first, ts, subset=subset,
                                    ground=ground, term=a.readout_term, mlp=mlp)
                if speed:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    print('SPEED-SCORE ' + json.dumps({
                        'setting': setting, 'period': period, 'strategy': strategy,
                        'positives': int(s['n']), 'seconds': time.time() - ts0,
                        'peak_mem_mb': (torch.cuda.max_memory_allocated() / 2 ** 20)
                        if torch.cuda.is_available() else 0.0}), flush=True)
                if a.b_only:
                    s['pos_f'][:] = 0.0
                    s['neg_f'][:] = 0.0
                res = {}
                for b in betas:
                    ap, auc = sfs.protocol_metrics(s, b, a.batch_size)
                    res['%g' % b] = {'ap': ap, 'auc': auc}
                grid[setting][period][strategy] = {'n': int(s['n']),
                                                   'by_beta': res}
                print('[%s %s %s] n=%d  ' % (setting, period, strategy,
                                             s['n'])
                      + '  '.join('b=%g ap=%.4f auc=%.4f' % (
                          b, res['%g' % b]['ap'], res['%g' % b]['auc'])
                          for b in betas)
                      + '  (%.0fs)' % (time.time() - t0), flush=True)
    # beta chosen on the validation split of the same setting and strategy
    selected = {}
    for setting in settings:
        selected[setting] = {}
        for strategy in strategies:
            if 'val' not in grid[setting]:          # no validation split scored (e.g. a timing run): first beta
                best = betas[0]
                sel = {'beta': best}
            else:
                v = grid[setting]['val'][strategy]['by_beta']
                best = max(betas, key=lambda b: (v['%g' % b]['ap'], -b))
                sel = {'beta': best, 'val_ap': v['%g' % best]['ap'],
                       'val_auc': v['%g' % best]['auc']}
            if 'test' in grid[setting]:
                t = grid[setting]['test'][strategy]['by_beta']['%g' % best]
                sel.update({'test_ap': t['ap'], 'test_auc': t['auc']})
            selected[setting][strategy] = sel
    out = {'dataset': run['cfg']['dataset'], 'ckpt': os.path.abspath(a.ckpt),
           'run': run['c'].get('run'), 'betas': betas,
           'batch_size': a.batch_size, 'grid': grid, 'selected': selected,
           'readout_term': a.readout_term, 'b_only': int(a.b_only),
           'seconds': time.time() - t0}
    path = a.out or (os.path.splitext(a.ckpt)[0] + '_eval.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print('EVAL-DONE ' + json.dumps({'dataset': out['dataset'],
                                     'run': out['run'],
                                     'selected': selected}), flush=True)


if __name__ == '__main__':
    main()
