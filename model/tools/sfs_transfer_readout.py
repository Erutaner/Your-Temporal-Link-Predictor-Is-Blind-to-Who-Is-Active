# -*- coding: utf-8 -*-
"""Transfer of the ground-process readout to another backbone:
score = logit_backbone + beta * log lambda_u(t), with lambda_u our fitted ground process
(K + 1 parameters, fitted on the training stream only) and beta chosen on the validation
split of the same setting and strategy.  The backbone is NOT retrained: it only has to
provide its per-edge logits.

Input: the backbone's per-edge dumps, one .npz per (strategy, split) with arrays
  src, neg_src, t, pos, neg   (source ids of the positives / negatives, the shared timestamps,
                              the positive / negative logits, in evaluation order)
named <prefix>_<strategy>_<split>.npz, split in {val, nn_val, test, nn_test}.

    python tools/sfs_transfer_readout.py --dataset mooc --config configs/mooc.json --data_root <root>
        --ground_json ground_params.json --dumps_dir <dir> --prefix strict_mooc_seed0 --out <json>

Output JSON: the same grid / selected structure as scripts/evaluate_sfs.py, plus the
backbone-only numbers (beta = 0) for the record.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from tgmin import sfs                                   # noqa: E402
from tgmin.data import build_lp_bundle                  # noqa: E402

STRATEGIES = ('random', 'historical', 'inductive')
SPLITS = {'transductive': ('val', 'test'), 'inductive': ('nn_val', 'nn_test')}


def decayed_counts(feeder, chunk_starts, first_chunk, nodes, tq, lam):
    """[n, K] float64: exact decayed incident counts of `nodes` strictly before their query
    times (rank rule), computed like SFSModel.read: entering counts of the query's chunk
    decayed to t, plus the within-chunk records."""
    n = int(nodes.shape[0])
    dev = feeder.device
    out = torch.zeros((n, lam.shape[0]), dtype=torch.float64, device=dev)
    lam_t = torch.tensor(lam, device=dev)
    local = np.searchsorted(chunk_starts, tq, side='right') - 1
    assert (local >= 0).all(), 'query before the first chunk of its period'
    for lc in np.unique(local):
        c = int(first_chunk + lc)
        m = np.where(local == lc)[0]
        ent = feeder.entering(c)
        nodes_t = torch.tensor(nodes[m], device=dev)
        tq_t = torch.tensor(tq[m], device=dev)
        dec = torch.exp(-lam_t.unsqueeze(0) * (tq_t - ent['c_start']).clamp(min=0).unsqueeze(1))
        cnt = ent['ncount'][nodes_t].double() * dec
        q_idx, _partner, dt, _k = feeder.fresh(c, nodes[m], tq[m])
        if q_idx.shape[0]:
            qi = torch.tensor(q_idx, device=dev)
            for lo in range(0, int(q_idx.shape[0]), sfs.FRESH_SLICE):
                hi = min(int(q_idx.shape[0]), lo + sfs.FRESH_SLICE)
                w = torch.exp(-torch.tensor(dt[lo:hi], device=dev).unsqueeze(1) * lam_t.unsqueeze(0))
                cnt.index_add_(0, qi[lo:hi], w)
        out[torch.tensor(m, device=dev)] = cnt
    return out


def oracle_counts(feeder, node, t_query, lam):
    """Brute-force decayed count of `node` strictly before t_query (rank rule), summed over the
    node's ENDPOINT RECORDS as the feeder does: a self-loop event (u == v) is two records of the
    same node and counts twice (the convention of the trained model and of the fitted ground
    process; sfs.oracle_state counts such an event once)."""
    r_q = np.searchsorted(feeder.tsu, t_query, side='left')
    n = np.zeros(lam.shape[0])
    for e in range(feeder.E_n):
        if feeder.rank[e] >= r_q:
            continue
        hits = int(feeder.u[e] == node) + int(feeder.v[e] == node)
        if hits:
            n += hits * np.exp(-lam * (t_query - feeder.t[e]))
    return n


def log_intensity(counts, lam, theta, theta_mu):
    """log lambda_u = log(mu + sum_k w_k n_k), w_k = lam_k e^{theta_k}, mu = lam_min e^{theta_mu}
    (GroundProcess.rates / log_intensity)."""
    lam_t = torch.tensor(lam, device=counts.device)
    w = lam_t * torch.exp(torch.tensor(theta, device=counts.device, dtype=torch.float64))
    mu = float(np.min(lam)) * math.exp(theta_mu)
    return torch.log(mu + (counts * w.unsqueeze(0)).sum(-1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--data_root', required=True)
    p.add_argument('--ground_json', required=True)
    p.add_argument('--dumps_dir', required=True)
    p.add_argument('--prefix', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch_size', type=int, default=200)
    p.add_argument('--betas', default='0,0.5,1,1.5,2')
    p.add_argument('--selfcheck', type=int, default=8, help='queries compared with the brute-force oracle')
    a = p.parse_args()
    t0 = time.time()
    device = a.device if torch.cuda.is_available() else 'cpu'
    cfg = json.load(open(a.config))
    g = json.load(open(a.ground_json))[a.dataset]
    lam = np.asarray(g['lam'], dtype=np.float64)
    betas = [float(x) for x in a.betas.split(',')]

    bundle, _train_bundle, meta = build_lp_bundle(a.dataset, a.data_root, device, t_train=int(cfg['t_train']),
                                                  chunk_mode=cfg['chunk_mode'], want_feats=False)
    T_train, T_val, T_test = meta['T']
    ts = [np.asarray(t, dtype=np.float64) for t in meta['ts_np']]
    lam_grid, _ = sfs.scale_grid(np.concatenate([t for t in ts[:T_train] if t.size]))
    assert lam_grid.shape == lam.shape and np.allclose(lam_grid, lam), 'scale grid differs from the fitted ground process'
    edges = [e.cpu().numpy() for e in bundle['edges']]
    feeder = sfs.SFSFeeder(int(bundle['num_nodes']), edges, ts, lam, device)
    starts = {'val': (T_train, feeder.c_start[T_train:T_train + T_val]),
              'test': (T_train + T_val, feeder.c_start[T_train + T_val:])}
    print('[transfer] %s: chunks %s, K=%d, ground epoch %s' % (a.dataset, meta['T'], lam.shape[0], g.get('epoch')), flush=True)

    # brute-force check of the counts on a few queries (rank rule included)
    if a.selfcheck > 0:
        rs = np.random.RandomState(0)
        d_test = meta['test_data']
        idx = rs.choice(len(d_test.src_node_ids), size=min(a.selfcheck, len(d_test.src_node_ids)), replace=False)
        nodes = d_test.src_node_ids[idx].astype(np.int64)
        tq = np.asarray(d_test.node_interact_times[idx], dtype=np.float64)
        got = decayed_counts(feeder, starts['test'][1], starts['test'][0], nodes, tq, lam).cpu().numpy()
        worst = 0.0
        for i in range(len(idx)):
            n_ref = oracle_counts(feeder, int(nodes[i]), float(tq[i]), lam)
            worst = max(worst, float(np.max(np.abs(got[i] - n_ref) / (1.0 + np.abs(n_ref)))))
        print('[transfer] selfcheck: %d queries, worst relative count error %.2e' % (len(idx), worst), flush=True)
        assert worst < 1e-4, 'count mismatch against the brute-force oracle'

    grid, selected, backbone = {}, {}, {}
    cache = {}

    def logs_for(period, nodes, tq):
        key = (period, nodes.tobytes(), tq.tobytes())
        if key not in cache:
            first, cs = starts[period]
            cnt = decayed_counts(feeder, cs, first, nodes, tq, lam)
            cache[key] = log_intensity(cnt, lam, g['theta'], g['theta_mu']).float().cpu().numpy()
        return cache[key]

    for setting, (sv, st_) in SPLITS.items():
        grid[setting] = {'val': {}, 'test': {}}
        selected[setting] = {}
        backbone[setting] = {}
        for strategy in STRATEGIES:
            scores = {}
            for split, period in ((sv, 'val'), (st_, 'test')):
                f = os.path.join(a.dumps_dir, '%s_%s_%s.npz' % (a.prefix, strategy, split))
                d = np.load(f)
                src = d['src'].astype(np.int64)
                neg_src = d['neg_src'].astype(np.int64)
                # timestamps come from OUR loader (the backbone's loader may rescale discrete
                # timestamps); the dumped source sequence must match ours edge for edge
                ref = meta[{'val': 'val_data', 'test': 'test_data', 'nn_val': 'new_node_val_data',
                            'nn_test': 'new_node_test_data'}[split]]
                assert src.shape[0] == len(ref.src_node_ids) and np.array_equal(src, ref.src_node_ids.astype(np.int64)),                     'dumped %s split does not match our %s edges' % (split, split)
                tq = np.asarray(ref.node_interact_times, dtype=np.float64)
                s = {'pos_f': d['pos'].astype(np.float32), 'neg_f': d['neg'].astype(np.float32), 'n': int(src.shape[0])}
                s['pos_b'] = logs_for(period, src, tq)
                s['neg_b'] = s['pos_b'] if np.array_equal(neg_src, src) else logs_for(period, neg_src, tq)
                scores[period] = s
                res = {}
                for b in betas:
                    ap, auc = sfs.protocol_metrics(s, b, a.batch_size)
                    res['%g' % b] = {'ap': ap, 'auc': auc}
                grid[setting][period][strategy] = {'n': s['n'], 'by_beta': res}
                print('[%s %s %s] n=%d  ' % (setting, period, strategy, s['n'])
                      + '  '.join('b=%g ap=%.4f' % (b, res['%g' % b]['ap']) for b in betas)
                      + '  (%.0fs)' % (time.time() - t0), flush=True)
            v = grid[setting]['val'][strategy]['by_beta']
            best = max(betas, key=lambda b: (v['%g' % b]['ap'], -b))
            t = grid[setting]['test'][strategy]['by_beta']
            selected[setting][strategy] = {'beta': best, 'val_ap': v['%g' % best]['ap'], 'val_auc': v['%g' % best]['auc'],
                                           'test_ap': t['%g' % best]['ap'], 'test_auc': t['%g' % best]['auc']}
            backbone[setting][strategy] = {'test_ap': t['0']['ap'], 'test_auc': t['0']['auc']}
    out = {'dataset': a.dataset, 'backbone_prefix': a.prefix, 'ground': g, 'betas': betas, 'batch_size': a.batch_size,
           'grid': grid, 'selected': selected, 'backbone_only': backbone, 'seconds': time.time() - t0}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(out, f, indent=1)
    print('TRANSFER-DONE ' + json.dumps({'dataset': a.dataset, 'selected': selected, 'backbone_only': backbone}), flush=True)


if __name__ == '__main__':
    main()
