# -*- coding: utf-8 -*-
"""Train / evaluate the scale-free continuous-time state trunk with its
closed-form ground process on one stream.

Per epoch: one pass over the train chunks (tgmin.sfs.train_epoch), then the
random-negative protocol on val and test; at every epoch that improves the
random-negative validation AP and AUC (the model-selection rule) also the
historical and inductive protocols, bare and with + log lambda_u.  The
checkpoint and the reported numbers are those of the selected epoch.

The per-stream config (configs/<dataset>.json) carries the chunking
(t_train, chunk_mode) and the training regime (e_train, dropout, wd,
neg_pool); a command-line flag given explicitly overrides it.  Every other
flag defaults to the main configuration.
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

from tgmin import sfs                                          # noqa: E402
from tgmin.data import build_lp_bundle, train_neg_like         # noqa: E402
from tgmin.vendor.utils import NegativeEdgeSampler, set_random_seed  # noqa: E402

REGIME_KEYS = ('e_train', 'dropout', 'wd', 'neg_pool')


def train_neg_seen(train_bundle, rng, device):
    """Per-epoch training negatives with a CAUSAL destination pool: for
    the pairs scored at index t (chunk t+1's edges) the negative dst is
    drawn uniformly from the dsts observed in chunks <= t (falls back to
    chunk t+1's own dsts when nothing has been seen yet)."""
    out = []
    seen = np.zeros(0, dtype=np.int64)
    for t in range(train_bundle['T']):
        e = train_bundle['edges'][t]
        if e.numel():
            seen = np.union1d(seen, e[1].cpu().numpy())
        pos = train_bundle['train_pos'][t]
        n = int(pos.shape[1])
        if n == 0:
            out.append(torch.zeros(2, 0, dtype=torch.long, device=device))
            continue
        pool = seen if seen.size else np.unique(pos[1].cpu().numpy())
        neg_dst = pool[rng.randint(0, len(pool), n)]
        out.append(torch.stack([
            pos[0],
            torch.tensor(neg_dst, dtype=torch.long, device=device)]))
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--data_root', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--run', type=int, default=0, help='seed')
    p.add_argument('--tag', default='_final')
    p.add_argument('--epochs', type=int, default=None, help='default: config')
    p.add_argument('--patience', type=int, default=100)
    p.add_argument('--lr', type=float, default=None, help='default: config')
    p.add_argument('--hlr_mult', type=float, default=10.0,
                   help='learning-rate multiplier of the zero-init channel '
                        'heads')
    p.add_argument('--batch_size', type=int, default=200,
                   help='metric batch of the evaluation protocols')
    p.add_argument('--print_every', type=int, default=1)
    p.add_argument('--t_train', type=int, default=None, help='default: config')
    p.add_argument('--chunk_mode', default=None, choices=['equal', 'native'],
                   help='default: config')
    # training regime (default: the config's values)
    p.add_argument('--e_train', type=int, default=None, choices=[0, 1],
                   help='1: train the node table E; 0: fixed random codes')
    p.add_argument('--dropout', type=float, default=None)
    p.add_argument('--wd', type=float, default=None,
                   help='decoupled weight decay on E')
    p.add_argument('--neg_pool', default=None, choices=['train', 'seen'],
                   help="training negatives' destination pool: all train "
                        'destinations, or those seen up to the entering '
                        'chunk')
    # the model (defaults = the main configuration)
    p.add_argument('--fresh_state', type=int, default=0, choices=[0, 1],
                   help='0: within-chunk records enter only the scalar '
                        'channels (default); 1: also the state')
    p.add_argument('--ground', type=int, default=1, choices=[0, 1],
                   help='learn the ground process lambda_u by its '
                        'likelihood and read f + log lambda_u under the '
                        'historical / inductive protocols')
    p.add_argument('--count_channels', type=int, default=1, choices=[0, 1],
                   help='0: the pair / endpoint count channels of the '
                        'scoring head are zeroed (trunk ablation)')
    p.add_argument('--fresh_records', type=int, default=1, choices=[0, 1],
                   help='1: the within-chunk records strictly before t enter '
                        'the counts, sketches and feature states (default); '
                        '0: entering state decayed to t only (chunking ablation)')
    p.add_argument('--per_decade', type=float, default=2.0,
                   help='scales per decade of the log-spaced grid')
    p.add_argument('--kmax', type=int, default=16,
                   help='cap on the number of scales')
    p.add_argument('--kmin', type=int, default=2,
                   help='floor on the number of scales (1: single-scale '
                        'ablation, the geometric middle of the range)')
    p.add_argument('--d', type=int, default=128)
    p.add_argument('--rank', type=int, default=16,
                   help='rank of the per-scale bilinear channels')
    p.add_argument('--second', type=int, default=1, choices=[0, 1],
                   help='second-order state (ablation: 0)')
    p.add_argument('--static', type=int, default=1, choices=[0, 1],
                   help='static E[u].E[v] channel (ablation: 0)')
    p.add_argument('--squash', type=int, default=1, choices=[0, 1],
                   help='signed log link on the bilinear channels')
    p.add_argument('--sketch', type=int, default=256,
                   help='dimension of the count-sketch stream (0: off)')
    p.add_argument('--set_M', type=int, default=20,
                   help='last-M events of the candidate-conditioned set '
                        'encoder (0: off; needs --sketch)')
    p.add_argument('--set_hidden', type=int, default=64,
                   help='width of the set encoder')
    p.add_argument('--feat', type=int, default=1, choices=[0, 1],
                   help='edge-feature state (when the stream has an '
                        'informative feature table)')
    p.add_argument('--eval_hist', type=int, default=0, choices=[0, 1],
                   help='also evaluate the historical and inductive '
                        'protocols inside the training loop (slow; the '
                        'full evaluation is scripts/evaluate_sfs.py)')
    return p.parse_args()


def main():
    a = parse_args()
    cfg = json.load(open(a.config))
    dataset = cfg['dataset']
    a.t_train = a.t_train or int(cfg['t_train'])
    a.chunk_mode = a.chunk_mode or cfg['chunk_mode']
    a.epochs = a.epochs or int(cfg.get('epochs', 500))
    a.lr = a.lr or float(cfg.get('lr', 1e-3))
    for k in REGIME_KEYS:
        if getattr(a, k) is None:
            setattr(a, k, cfg[k])
    device = a.device if torch.cuda.is_available() else 'cpu'
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()

    bundle, train_bundle, meta = build_lp_bundle(
        dataset, a.data_root, device, t_train=a.t_train,
        chunk_mode=a.chunk_mode, want_feats=bool(a.feat))
    T_train, T_val, T_test = meta['T']
    N = int(bundle['num_nodes'])
    # the stream holds EVERY event of the three periods
    n_edges = sum(int(e.shape[1]) for e in bundle['edges'])
    n_periods = sum(len(meta[k].src_node_ids) for k in
                    ('train_data', 'val_data', 'test_data'))
    assert n_edges == n_periods, (n_edges, n_periods)
    print('[data] %s nodes=%d chunks=%s events=%d' % (
        dataset, N, meta['T'], n_edges), flush=True)

    edges_np_all = [e.detach().cpu().numpy() for e in bundle['edges']]
    ts_np_full = [np.asarray(t, dtype=np.float64) for t in meta['ts_np']]
    ts_train = np.concatenate([t for t in ts_np_full[:T_train] if t.size])
    lam, ginfo = sfs.scale_grid(ts_train, per_decade=a.per_decade,
                                kmax=a.kmax, kmin=a.kmin)
    print('[sfs] scale grid from TRAIN timestamps: dt_min=%g span=%g '
          'decades=%.2f K=%d lam=[%s]' % (
              ginfo['dt_min'], ginfo['span'], ginfo['decades'], ginfo['K'],
              ', '.join('%.3g' % x for x in lam)), flush=True)

    set_random_seed(a.run)
    feeder = sfs.SFSFeeder(N, edges_np_all, ts_np_full, lam, device)
    print('[sfs] feeder: events=%d directed pairs=%d unique ts=%d' % (
        feeder.E_n, feeder.P, feeder.M - 1), flush=True)
    feat_dim = 0
    if a.feat and meta['efeat'] is not None and meta['feat_dim'] > 0:
        eids_all = np.concatenate([np.asarray(e, dtype=np.int64)
                                   for e in meta['eids_np'] if len(e)])
        fe = meta['efeat'][torch.tensor(eids_all, device=device)]
        assert fe.shape[0] == feeder.E_n
        feeder.set_features(fe)
        feat_dim = int(meta['feat_dim'])
        print('[sfs] edge-feature state on: F=%d (events %d)' % (
            feat_dim, fe.shape[0]), flush=True)
    elif a.feat:
        print('[sfs] --feat 1 but no informative feature stream: off',
              flush=True)
    model = sfs.SFSModel(N, lam, d=a.d, rank=a.rank,
                         second_order=bool(a.second), e_train=bool(a.e_train),
                         static=bool(a.static), squash=bool(a.squash),
                         dropout=a.dropout, feat_dim=feat_dim,
                         sketch_dim=a.sketch, set_M=a.set_M,
                         set_hidden=a.set_hidden,
                         fresh_state=bool(a.fresh_state),
                         fresh_records=bool(a.fresh_records),
                         count_channels=bool(a.count_channels)).to(device)
    ground = wcache = None
    if a.ground:
        spool_g = np.unique(np.concatenate(
            [train_bundle['edges'][t][0].cpu().numpy()
             for t in range(train_bundle['T'])
             if train_bundle['edges'][t].numel()]))
        ground = sfs.GroundProcess(lam).to(device)
        wcache = sfs.WindowCache(feeder, spool_g, device)
        print('[sfs] ground process by likelihood: sources=%d' % len(spool_g),
              flush=True)
    hot = model.hot_params()
    hot_ids = set(id(p) for p in hot)
    e_ids = set(id(p) for p in model.E.parameters())
    base = [p for p in model.parameters()
            if id(p) not in hot_ids and id(p) not in e_ids and p.requires_grad]
    groups = [{'params': base, 'lr': a.lr, 'weight_decay': 0.0},
              {'params': hot, 'lr': a.lr * a.hlr_mult, 'weight_decay': 0.0}]
    if a.e_train:
        groups.append({'params': list(model.E.parameters()), 'lr': a.lr,
                       'weight_decay': a.wd})
    if ground is not None:
        # the ground process has K+1 log-scale parameters: its own rate
        groups.append({'params': list(ground.parameters()), 'lr': 0.05,
                       'weight_decay': 0.0})
    optimizer = torch.optim.AdamW(groups)
    n_params = sum(int(p.numel()) for p in model.parameters()
                   if p.requires_grad)
    print('[sfs] params=%d (hot %d) lr=%g hlr=%g regime=%s' % (
        n_params, sum(int(p.numel()) for p in hot), a.lr, a.lr * a.hlr_mult,
        {k: getattr(a, k) for k in REGIME_KEYS}), flush=True)

    full = meta['full_data']
    val_sampler = NegativeEdgeSampler(src_node_ids=full.src_node_ids,
                                      dst_node_ids=full.dst_node_ids, seed=0)
    test_sampler = NegativeEdgeSampler(src_node_ids=full.src_node_ids,
                                       dst_node_ids=full.dst_node_ids, seed=2)
    hist_val = hist_test = ind_val = ind_test = None
    if a.eval_hist:
        # the reference historical / inductive samplers (both endpoints
        # replaced), seeded as the reference pipeline does (val 0, test 2)
        ind_val = NegativeEdgeSampler(
            src_node_ids=full.src_node_ids, dst_node_ids=full.dst_node_ids,
            interact_times=full.node_interact_times,
            last_observed_time=meta['train_data'].node_interact_times[-1],
            negative_sample_strategy='inductive', seed=0)
        ind_test = NegativeEdgeSampler(
            src_node_ids=full.src_node_ids, dst_node_ids=full.dst_node_ids,
            interact_times=full.node_interact_times,
            last_observed_time=meta['val_data'].node_interact_times[-1],
            negative_sample_strategy='inductive', seed=2)
        hist_val = NegativeEdgeSampler(
            src_node_ids=full.src_node_ids, dst_node_ids=full.dst_node_ids,
            interact_times=full.node_interact_times,
            last_observed_time=meta['train_data'].node_interact_times[-1],
            negative_sample_strategy='historical', seed=0)
        hist_test = NegativeEdgeSampler(
            src_node_ids=full.src_node_ids, dst_node_ids=full.dst_node_ids,
            interact_times=full.node_interact_times,
            last_observed_time=meta['val_data'].node_interact_times[-1],
            negative_sample_strategy='historical', seed=2)
    rng = np.random.RandomState(a.run)

    best = {'val_ap': -1.0, 'val_auc': -1.0, 'test_ap': 0.0,
            'test_auc': 0.0, 'epoch': -1, 'loss': 0.0}
    ck_path = os.path.join(a.out_dir, '%s%s_run%d.pt' % (dataset, a.tag, a.run))
    bad = 0
    n_ran = 0
    for epoch in range(1, a.epochs + 1):
        te = time.time()
        model.train()
        nan2 = (float('nan'), float('nan'))
        negs = (train_neg_seen(train_bundle, rng, device)
                if a.neg_pool == 'seen'
                else train_neg_like(train_bundle, rng, device))
        speed = int(os.environ.get('SFS_SPEED', '0'))   # speed benchmark: time N epochs, then stop
        if speed and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            te = time.time()
        loss, skipped, steps = sfs.train_epoch(
            model, feeder, train_bundle, ts_np_full, negs, optimizer,
            ground=ground, wcache=wcache)
        if speed:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            print('SPEED-EPOCH ' + json.dumps({
                'epoch': epoch, 'train_seconds': time.time() - te,
                'steps': int(steps), 'train_edges': int(len(meta['train_data'].src_node_ids)),
                'peak_mem_mb': (torch.cuda.max_memory_allocated() / 2 ** 20)
                if torch.cuda.is_available() else 0.0}), flush=True)
            if epoch >= speed:
                raise SystemExit(0)
        model.eval()
        val_ap, val_auc, _ = sfs.eval_period(
            model, feeder, meta, 1, val_sampler, a.batch_size, T_train,
            ts_np_full)
        test_ap, test_auc, _ = sfs.eval_period(
            model, feeder, meta, 2, test_sampler, a.batch_size,
            T_train + T_val, ts_np_full)
        hv = ht = iv = it_ = nan2
        hvs = hts = ivs = its = nan2
        n_ran = epoch
        improved = (val_ap >= best['val_ap'] and val_auc >= best['val_auc'])
        if a.eval_hist and (improved or epoch % 5 == 0 or epoch == 1):
            r1 = sfs.eval_period_hist(model, feeder, meta, 1, hist_val,
                                      a.batch_size, T_train, ts_np_full,
                                      ground=ground)
            r2 = sfs.eval_period_hist(model, feeder, meta, 2, hist_test,
                                      a.batch_size, T_train + T_val,
                                      ts_np_full, ground=ground)
            hv, ht = r1[:2], r2[:2]
            if ground is not None:
                hvs, hts = r1[3:5], r2[3:5]
            r3 = sfs.eval_period_hist(model, feeder, meta, 1, ind_val,
                                      a.batch_size, T_train, ts_np_full,
                                      ground=ground)
            r4 = sfs.eval_period_hist(model, feeder, meta, 2, ind_test,
                                      a.batch_size, T_train + T_val,
                                      ts_np_full, ground=ground)
            iv, it_ = r3[:2], r4[:2]
            if ground is not None:
                ivs, its = r3[3:5], r4[3:5]
        if improved:
            best = {'val_ap': val_ap, 'val_auc': val_auc, 'test_ap': test_ap,
                    'test_auc': test_auc, 'epoch': epoch, 'loss': loss,
                    'seconds': time.time() - te,
                    'hist_val_ap': hv[0], 'hist_val_auc': hv[1],
                    'hist_test_ap': ht[0], 'hist_test_auc': ht[1],
                    'hist_val_ap_lam': hvs[0], 'hist_val_auc_lam': hvs[1],
                    'hist_test_ap_lam': hts[0], 'hist_test_auc_lam': hts[1],
                    'ind_val_ap': iv[0], 'ind_val_auc': iv[1],
                    'ind_test_ap': it_[0], 'ind_test_auc': it_[1],
                    'ind_val_ap_lam': ivs[0], 'ind_val_auc_lam': ivs[1],
                    'ind_test_ap_lam': its[0], 'ind_test_auc_lam': its[1]}
            blob = {'state_dict': model.state_dict(), 'epoch': epoch,
                    'config': vars(a), 'lam': lam.tolist()}
            if ground is not None:
                blob['ground'] = ground.state_dict()
            torch.save(blob, ck_path)
            bad = 0
        else:
            bad += 1
        if epoch % a.print_every == 0 or improved or epoch == 1:
            print('[%s%s run%d %d/%d] L=%.4f val_ap=%.5f val_auc=%.5f '
                  'test_ap=%.5f hist=%.4f/%.4f ind=%.4f/%.4f '
                  '+lambda: hist=%.4f/%.4f ind=%.4f/%.4f '
                  'best@%d (%.1fs)' % (
                      dataset, a.tag, a.run, epoch, a.epochs, loss, val_ap,
                      val_auc, test_ap, hv[0], ht[0], iv[0], it_[0],
                      hvs[0], hts[0], ivs[0], its[0],
                      best['epoch'], time.time() - te), flush=True)
            if ground is not None:
                mu_, w_ = ground.rates()
                print('[grd] mu=%.3g w/lam=%s' % (float(mu_), np.array2string(
                    torch.exp(ground.theta).detach().cpu().numpy(),
                    precision=2, max_line_width=200)), flush=True)
            if torch.cuda.is_available():
                print('[mem] max_alloc=%.2fG reserved=%.2fG' % (
                    torch.cuda.max_memory_allocated() / 2**30,
                    torch.cuda.memory_reserved() / 2**30), flush=True)
        if bad >= a.patience:
            print('[sfs] early stop at epoch %d (patience %d)' % (
                epoch, a.patience), flush=True)
            break

    out = {'dataset': dataset, 'tag': a.tag, 'run': a.run,
           'test_ap': best['test_ap'], 'test_auc': best['test_auc'],
           'val_ap': best['val_ap'], 'val_auc': best['val_auc'],
           'epoch': best['epoch'], 'wall_s': time.time() - t0,
           'n_epochs_ran': n_ran, 'param_count': n_params,
           'hist_val_ap': best.get('hist_val_ap', float('nan')),
           'hist_test_ap': best.get('hist_test_ap', float('nan')),
           'hist_val_ap_lam': best.get('hist_val_ap_lam', float('nan')),
           'hist_test_ap_lam': best.get('hist_test_ap_lam', float('nan')),
           'ind_val_ap': best.get('ind_val_ap', float('nan')),
           'ind_test_ap': best.get('ind_test_ap', float('nan')),
           'ind_val_ap_lam': best.get('ind_val_ap_lam', float('nan')),
           'ind_test_ap_lam': best.get('ind_test_ap_lam', float('nan')),
           'best': best, 'lam': lam.tolist(), 'grid': ginfo,
           'config': vars(a)}
    with open(os.path.join(a.out_dir, '%s%s_run%d.json' % (
            dataset, a.tag, a.run)), 'w') as f:
        json.dump(out, f, indent=1)
    print('TRAIN-DONE ' + json.dumps({k: out[k] for k in (
        'dataset', 'tag', 'run', 'test_ap', 'test_auc', 'val_ap', 'val_auc',
        'hist_val_ap', 'hist_test_ap', 'hist_val_ap_lam', 'hist_test_ap_lam',
        'ind_val_ap', 'ind_test_ap', 'ind_val_ap_lam', 'ind_test_ap_lam',
        'epoch', 'wall_s')}), flush=True)


if __name__ == '__main__':
    main()
