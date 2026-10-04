# -*- coding: utf-8 -*-
"""Recompute a trained checkpoint's reported test numbers (random /
historical / inductive protocols on the whole test period, the latter two
bare and with + log lambda_u) with the current code on the given device,
and print them next to the values in the run's JSON (same basename), with
the differences.  Use it to confirm that a kernel change (or the device)
leaves every reported number unchanged.
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from tgmin import sfs                                  # noqa: E402
from tgmin.checkpoint import load_run                  # noqa: E402
from tgmin.vendor.utils import NegativeEdgeSampler     # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--data_root', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch_size', type=int, default=200)
    a = p.parse_args()
    run = load_run(a.ckpt, a.config, a.data_root, a.device)
    model, feeder, meta, ground, ts = (run['model'], run['feeder'],
                                       run['meta'], run['ground'], run['ts'])
    T_train, T_val, _ = meta['T']
    full = meta['full_data']
    last = meta['val_data'].node_interact_times[-1]
    test_sampler = NegativeEdgeSampler(src_node_ids=full.src_node_ids,
                                       dst_node_ids=full.dst_node_ids, seed=2)
    hist_test = NegativeEdgeSampler(
        src_node_ids=full.src_node_ids, dst_node_ids=full.dst_node_ids,
        interact_times=full.node_interact_times, last_observed_time=last,
        negative_sample_strategy='historical', seed=2)
    ind_test = NegativeEdgeSampler(
        src_node_ids=full.src_node_ids, dst_node_ids=full.dst_node_ids,
        interact_times=full.node_interact_times, last_observed_time=last,
        negative_sample_strategy='inductive', seed=2)
    got = {}
    ap, auc, _ = sfs.eval_period(model, feeder, meta, 2, test_sampler,
                                 a.batch_size, T_train + T_val, ts)
    got['test_ap'], got['test_auc'] = ap, auc
    r = sfs.eval_period_hist(model, feeder, meta, 2, hist_test, a.batch_size,
                             T_train + T_val, ts, ground=ground)
    got['hist_test_ap'] = r[0]
    if ground is not None:
        got['hist_test_ap_lam'] = r[3]
    r = sfs.eval_period_hist(model, feeder, meta, 2, ind_test, a.batch_size,
                             T_train + T_val, ts, ground=ground)
    got['ind_test_ap'] = r[0]
    if ground is not None:
        got['ind_test_ap_lam'] = r[3]
    rep = {}
    jp = os.path.splitext(a.ckpt)[0] + '.json'
    if os.path.exists(jp):
        j = json.load(open(jp))
        for k in got:
            v = j.get(k, j.get(k.replace('_lam', '_src')))
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                rep[k] = float(v)
    out = {'ckpt': os.path.basename(a.ckpt), 'device': a.device,
           'recomputed': got, 'reported': rep,
           'max_abs_diff': max([abs(got[k] - rep[k]) for k in rep]
                               or [float('nan')])}
    print(json.dumps(out))


if __name__ == '__main__':
    main()
