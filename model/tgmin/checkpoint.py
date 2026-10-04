# -*- coding: utf-8 -*-
"""Rebuild a trained run from its checkpoint -- data bundle, feeder, model
and ground process, exactly as scripts/train_sfs.py built them -- for the
evaluation script and the tools."""
import json

import numpy as np
import torch

from . import sfs
from .data import build_lp_bundle


def load_run(ckpt, config, data_root, device, dropout=0.0):
    """dict with cfg, c (the run's flags), blob, bundle, train_bundle,
    meta, feeder, model (eval mode), ground (or None), lam, ts, device."""
    cfg = json.load(open(config))
    blob = torch.load(ckpt, map_location='cpu', weights_only=False)
    c = blob['config']
    t_train = int(c.get('t_train') or cfg['t_train'])
    chunk_mode = c.get('chunk_mode') or cfg['chunk_mode']
    bundle, train_bundle, meta = build_lp_bundle(
        cfg['dataset'], data_root, device, t_train=t_train,
        chunk_mode=chunk_mode, want_feats=bool(c.get('feat', 0)))
    N = int(bundle['num_nodes'])
    edges = [e.cpu().numpy() for e in bundle['edges']]
    ts = [np.asarray(t, dtype=np.float64) for t in meta['ts_np']]
    lam = np.asarray(blob['lam'], dtype=np.float64)
    feeder = sfs.SFSFeeder(N, edges, ts, lam, device)
    feat_dim = 0
    if c.get('feat', 0) and meta['efeat'] is not None and meta['feat_dim'] > 0:
        eids = np.concatenate([np.asarray(e, dtype=np.int64)
                               for e in meta['eids_np'] if len(e)])
        feeder.set_features(meta['efeat'][torch.tensor(eids, device=device)])
        feat_dim = int(meta['feat_dim'])
    model = sfs.SFSModel(N, lam, d=int(c['d']), rank=int(c['rank']),
                         second_order=bool(c['second']),
                         e_train=bool(c.get('e_train', 1)),
                         static=bool(c.get('static', 1)),
                         squash=bool(c.get('squash', 0)), dropout=dropout,
                         feat_dim=feat_dim, sketch_dim=int(c.get('sketch', 0)),
                         set_M=int(c.get('set_M', 0)),
                         set_hidden=int(c.get('set_hidden', 64)),
                         fresh_state=bool(c.get('fresh_state', 0)),
                         fresh_records=bool(c.get('fresh_records', 1)),
                         count_channels=bool(c.get('count_channels', 1))).to(device)
    missing, unexpected = model.load_state_dict(blob['state_dict'],
                                                strict=False)
    if missing or unexpected:
        print('[load_run] state_dict missing=%s unexpected=%s' % (
            missing, unexpected), flush=True)
    model.eval()
    ground = None
    if 'ground' in blob:
        ground = sfs.GroundProcess(lam).to(device)
        ground.load_state_dict(blob['ground'])
    return {'cfg': cfg, 'c': c, 'blob': blob, 'bundle': bundle,
            'train_bundle': train_bundle, 'meta': meta, 'feeder': feeder,
            'model': model, 'ground': ground, 'lam': lam, 'ts': ts,
            'device': device}
