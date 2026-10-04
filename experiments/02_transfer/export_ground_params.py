# -*- coding: utf-8 -*-
"""Collect the fitted parameters of the activity term from our main checkpoints into one JSON file.

    python export_ground_params.py --ckpt_dir <dir of 01_main_comparison> --out ground_params.json
                                   [--seed 0] [--datasets ...]

Per stream: the rate grid (lam, K values), the fitted log-weights (theta, K values, and theta_mu), the epoch of
the checkpoint and the chunking it was trained with (t_train, chunk_mode).  The transfer readout
(model/tools/sfs_transfer_readout.py) recomputes log lambda_u from these alone; nothing else of our model is
used on the other backbones.  results/transfer/ground_params.json was made this way from the seed-0 checkpoints.
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common import MODEL, DATASETS, main_run_files  # noqa: E402


def export(ckpt, config):
    blob = torch.load(ckpt, map_location='cpu', weights_only=False)
    g = blob['ground']
    cfg = json.load(open(config))
    return {'lam': [float(x) for x in g['lam'].tolist()], 'theta': [float(x) for x in g['theta'].tolist()],
            'theta_mu': float(g['theta_mu']), 'epoch': int(blob['epoch']),
            't_train': int(blob['config'].get('t_train') or cfg['t_train']),
            'chunk_mode': blob['config'].get('chunk_mode') or cfg['chunk_mode']}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt_dir', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--datasets', default=','.join(k for k, _ in DATASETS))
    a = p.parse_args()
    out = {}
    for ds in a.datasets.split(','):
        ckpt, _ = main_run_files(a.ckpt_dir, ds, a.seed)
        out[ds] = export(ckpt, os.path.join(MODEL, 'configs', ds + '.json'))
        print('%-10s K=%2d epoch=%3d t_train=%d %s' % (ds, len(out[ds]['lam']), out[ds]['epoch'], out[ds]['t_train'], out[ds]['chunk_mode']))
    with open(a.out, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1)
    print('written', a.out)


if __name__ == '__main__':
    main()
