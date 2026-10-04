# -*- coding: utf-8 -*-
"""Device / batch-invariance check of a trained checkpoint on one stream:

  1. the whole TEST period scored on the GPU and on the CPU with identical
     negatives (the random protocol): every score compared, AP of both
     reported;
  2. the first test chunk scored on the GPU in batches of 1, 2, 5, 50, 300
     and as a whole, each compared with the CPU whole-chunk scores (a read
     must not depend on what else is in its batch).

Run it after any change to a read kernel.
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


def test_sampler(meta):
    full = meta['full_data']
    return NegativeEdgeSampler(src_node_ids=full.src_node_ids,
                               dst_node_ids=full.dst_node_ids, seed=2)


@torch.no_grad()
def first_chunk_batched(run, batch, n_max=300):
    """(pos, neg) scores of the first n_max test edges of the first test
    chunk, scored in batches of `batch` queries; negatives = the random
    protocol's first draws (same seeded sampler, batches of 200)."""
    model, feeder, meta = run['model'], run['feeder'], run['meta']
    T_train, T_val, _ = meta['T']
    data = meta['test_data']
    n = min(n_max, int(meta['test_chunk_sizes'][0]))
    sampler = test_sampler(meta)
    sampler.reset_random_state()
    neg = np.empty(n, dtype=np.int64)
    for lo in range(0, n, 200):
        hi = min(n, lo + 200)
        _, b = sampler.sample(size=hi - lo)
        neg[lo:hi] = b
    src, dst = data.src_node_ids, data.dst_node_ids
    tq = np.asarray(data.node_interact_times, dtype=np.float64)
    c = T_train + T_val
    ent = feeder.entering(c)
    states = model.entry_states(ent, feeder, c)
    pos_s, neg_s = [], []
    for lo in range(0, n, batch):
        hi = min(n, lo + batch)
        xu = model.read(feeder, c, ent, states, src[lo:hi], tq[lo:hi])
        ps = model.score(feeder, c, ent, states, src[lo:hi], dst[lo:hi],
                         tq[lo:hi], xu=xu)
        ns = model.score(feeder, c, ent, states, src[lo:hi], neg[lo:hi],
                         tq[lo:hi], xu=xu)
        pos_s.append(ps.max(1).values.cpu().numpy())
        neg_s.append(ns.max(1).values.cpu().numpy())
    return np.concatenate(pos_s), np.concatenate(neg_s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--data_root', required=True)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    res = {}
    batches = {}
    for dev in (a.device, 'cpu'):
        run = load_run(a.ckpt, a.config, a.data_root, dev)
        meta = run['meta']
        T_train, T_val, _ = meta['T']
        t0 = time.time()
        s = sfs.score_split(run['model'], run['feeder'], meta, 2,
                            test_sampler(meta), 200, T_train + T_val, run['ts'])
        ap, _auc = sfs.protocol_metrics(s, 0.0, 200)
        res[dev] = {'pos': s['pos_f'], 'neg': s['neg_f'], 'ap': ap,
                    's': time.time() - t0}
        print('%s: test AP %.6f (%.1fs, nan=%d)' % (
            dev, ap, res[dev]['s'],
            int(np.isnan(s['pos_f']).sum() + np.isnan(s['neg_f']).sum())),
            flush=True)
        if dev == a.device:
            for b in (1, 2, 5, 50, 300):
                batches[b] = first_chunk_batched(run, b)
    g, cp = res[a.device], res['cpu']
    n = len(g['pos'])
    out = {'dataset': run['cfg']['dataset'], 'period': {
        'scored_edges': 2 * n, 'ap_device': g['ap'], 'ap_cpu': cp['ap'],
        'max_abs_diff': float(max(np.abs(g['pos'] - cp['pos']).max(),
                                  np.abs(g['neg'] - cp['neg']).max())),
        'n_diff_gt_1e-3': int((np.abs(g['pos'] - cp['pos']) > 1e-3).sum()
                              + (np.abs(g['neg'] - cp['neg']) > 1e-3).sum())}}
    n0 = len(batches[1][0])
    out['batch_invariance'] = {
        'B%d' % b: float(max(np.abs(pb - cp['pos'][:n0]).max(),
                             np.abs(nb - cp['neg'][:n0]).max()))
        for b, (pb, nb) in batches.items()}
    ok = (abs(out['period']['ap_device'] - out['period']['ap_cpu']) < 1e-3
          and out['period']['n_diff_gt_1e-3'] <= 5
          and max(out['batch_invariance'].values()) < 1e-3)
    out['ok'] = bool(ok)
    print(json.dumps(out))
    print('SFS-DEVICE-CHECK-' + ('OK' if ok else 'FAILED'))


if __name__ == '__main__':
    main()
