# -*- coding: utf-8 -*-
"""Data pipeline: dataset loading, chunking, edge features and bundle
assembly.

  * chunk   a contiguous block of the chronological edge stream of one
            period (train / val / test).  The trunk recomputes its entering
            state once per chunk; every read inside a chunk is exact at the
            query's own timestamp (tgmin/sfs.py).
  * bundle  the dict the trainer consumes: T (number of chunks), per-chunk
            edge tensors, and the training pair lists.

Chunk modes:
  equal   t_train equal-edge-count chunks over the train period; each eval
          period gets max(2, round(t_train * 15/70)) chunks.
  native  one chunk per unique timestamp of the period (for streams whose
          time axis is a discrete sequence of snapshots), merged evenly if
          the count exceeds MAX_NATIVE.

Alignment convention ("next-chunk alignment"): bundle['train_pos'][t] holds
chunk t+1's edges, so the training loop that scores pairs[t] against the
state entering chunk t+1 performs next-chunk prediction.  The last index
holds an empty tensor.
"""
import os

import numpy as np
import torch

from .vendor.dataloader import get_link_prediction_data

MAX_NATIVE = 96        # native chunk-count cap


# --------------------------------------------------------------------------
# dataset loading (the vendored loader reads ./processed_data/<name>/...)
# --------------------------------------------------------------------------
def load_lp_split(dataset, data_root):
    """(node_feat, edge_feat, full, train, val, test, nn_val, nn_test).

    Link-prediction split: 70/15/15 by timestamp quantiles; a fixed-seed
    10% node sample is held out as "new nodes", their edges removed from
    train; nn_val / nn_test are the val/test edges touching a new node
    (the inductive evaluation sets)."""
    cwd = os.getcwd()
    os.chdir(data_root)
    try:
        return get_link_prediction_data(dataset_name=dataset, val_ratio=0.15,
                                        test_ratio=0.15)
    finally:
        os.chdir(cwd)


# --------------------------------------------------------------------------
# chunking
# --------------------------------------------------------------------------
def _eval_chunks(t_train):
    return max(2, int(round(t_train * 15.0 / 70.0)))


def _equal_chunks(n, k):
    return [c for c in np.array_split(np.arange(n), k)]


def _native_chunks(ts, cap):
    """One chunk per unique timestamp (merged evenly if > cap)."""
    _, starts = np.unique(ts, return_index=True)
    starts = np.sort(starts)
    bounds = list(starts) + [len(ts)]
    spans = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
    if len(spans) > cap:
        merged = []
        for grp in np.array_split(np.arange(len(spans)), cap):
            merged.append((spans[grp[0]][0], spans[grp[-1]][1]))
        spans = merged
    return [np.arange(a, b) for a, b in spans]


def chunk_lists_for(periods, t_train, chunk_mode):
    """Per-period lists of index arrays.  Period 0 is the train period,
    every later one an eval period (see the chunk-mode rules above)."""
    if chunk_mode == 'native':
        return [_native_chunks(d.node_interact_times, MAX_NATIVE)
                for d in periods]
    t_ev = _eval_chunks(t_train)
    ks = [t_train] + [t_ev] * (len(periods) - 1)
    for d, k in zip(periods, ks):
        assert len(d.src_node_ids) >= k, 'period smaller than its chunk count'
    return [_equal_chunks(len(d.src_node_ids), k)
            for d, k in zip(periods, ks)]


# --------------------------------------------------------------------------
# edge features
# --------------------------------------------------------------------------
def load_edge_features(dataset, data_root, device, train_edge_ids,
                       n_edges_expected):
    """[E+1, F'] edge-feature tensor with a column mask AND standardisation
    statistics fitted on TRAIN-period rows only (no val/test rows enter the
    scaler).  Constant columns are dropped; returns None if no column
    varies on train.  Row 0 stays an exact zero pad."""
    p = os.path.join(data_root, 'processed_data', dataset,
                     'ml_%s.npy' % dataset)
    ef = np.load(p)
    assert ef.shape[0] == n_edges_expected + 1, (
        'ml npy rows %d != edges+pad %d' % (ef.shape[0], n_edges_expected + 1))
    tr = ef[train_edge_ids]                  # train rows only
    keep = tr.std(axis=0) > 0
    if not keep.any():
        return None
    mu = tr[:, keep].mean(axis=0)
    sd = tr[:, keep].std(axis=0)
    out = ((ef[:, keep] - mu) / sd).astype(np.float32)
    out[0] = 0.0                             # keep the zero pad exact
    return torch.tensor(out, device=device)


# --------------------------------------------------------------------------
# stream building and bundle assembly
# --------------------------------------------------------------------------
def build_stream(periods, chunk_lists):
    """Per-chunk edge / edge-id / timestamp arrays of a tuple of periods
    (period 0 = train), with the period index of every chunk."""
    edges_np, eids_np, ts_np, chunk_of_period = [], [], [], []
    for p_idx, (d, cl) in enumerate(zip(periods, chunk_lists)):
        for c in cl:
            edges_np.append(np.stack([d.src_node_ids[c],
                                      d.dst_node_ids[c]]).astype(np.int64))
            eids_np.append(d.edge_ids[c].astype(np.int64))
            ts_np.append(d.node_interact_times[c].astype(np.float64))
            chunk_of_period.append(p_idx)
    return {'edges_np': edges_np, 'eids_np': eids_np, 'ts_np': ts_np,
            'chunk_of_period': chunk_of_period,
            'chunk_sizes': [int(e.shape[1]) for e in edges_np]}


def assemble_bundle(blob, dataset, num_nodes, device):
    """The trainer-facing bundle dict, with the next-chunk alignment for
    train_pos (index t holds chunk t+1's edges, empty past the train
    period)."""
    T = len(blob['edges_np'])
    t_tr = blob['chunk_of_period'].count(0)
    edges = [torch.tensor(e, dtype=torch.long, device=device)
             for e in blob['edges_np']]
    empty = torch.zeros(2, 0, dtype=torch.long, device=device)

    def _pairs(t):
        if t + 1 >= t_tr:
            return empty
        e = blob['edges_np'][t + 1]
        if e.shape[1] == 0:
            return empty
        return torch.tensor(e, dtype=torch.long, device=device)

    return {'data': dataset, 'num_nodes': num_nodes, 'T': T, 'edges': edges,
            'train_pos': [_pairs(t) for t in range(T)]}


def build_lp_bundle(dataset, data_root, device, t_train=12,
                    chunk_mode='equal', want_feats=True):
    """Transductive link-prediction data: (bundle, train_bundle, meta).

    bundle spans train+val+test chunks; train_bundle is the length-t_train
    prefix view (training never sees val/test structure); meta carries the
    eval-side pieces (period data, chunk sizes, edge ids/features)."""
    (_nrf, _erf, full_data, train_data, val_data, test_data,
     nn_val, nn_test) = load_lp_split(dataset, data_root)

    num_nodes = int(max(full_data.src_node_ids.max(),
                        full_data.dst_node_ids.max())) + 1
    for nm, d in (('train', train_data), ('val', val_data),
                  ('test', test_data)):
        assert (np.diff(d.node_interact_times) >= 0).all(), nm + ' unsorted'

    periods = [train_data, val_data, test_data]
    chunk_lists = chunk_lists_for(periods, t_train, chunk_mode)
    blob = build_stream(periods, chunk_lists)
    bundle = assemble_bundle(blob, dataset, num_nodes, device)

    t_tr, t_v, t_te = [blob['chunk_of_period'].count(i) for i in (0, 1, 2)]
    train_bundle = {k: (v[:t_tr] if isinstance(v, list) else v)
                    for k, v in bundle.items()}
    train_bundle['T'] = t_tr

    efeat = load_edge_features(
        dataset, data_root, device, train_edge_ids=train_data.edge_ids,
        n_edges_expected=int(full_data.edge_ids.max())) if want_feats else None
    meta = {
        'dataset': dataset, 'num_nodes': num_nodes,
        'T': (t_tr, t_v, t_te), 'chunk_mode': chunk_mode,
        'edges_per_chunk': [int(e.shape[1]) for e in bundle['edges']],
        'eids_np': blob['eids_np'], 'ts_np': blob['ts_np'],
        'efeat': efeat,
        'feat_dim': 0 if efeat is None else int(efeat.shape[1]),
        'full_data': full_data, 'train_data': train_data,
        'val_data': val_data, 'test_data': test_data,
        # the inductive setting: val / test edges touching a new node
        'new_node_val_data': nn_val, 'new_node_test_data': nn_test,
        # eval chunk boundaries (period-local sizes) for eval_period
        'val_chunk_sizes': [int(s) for s, p in zip(
            blob['chunk_sizes'], blob['chunk_of_period']) if p == 1],
        'test_chunk_sizes': [int(s) for s, p in zip(
            blob['chunk_sizes'], blob['chunk_of_period']) if p == 2],
    }
    return bundle, train_bundle, meta


def train_neg_like(train_bundle, rng, device):
    """Per-epoch training negatives: keep the positive sources, draw
    destinations uniformly from the destinations OBSERVED IN TRAIN, no
    collision check (the same recipe as the evaluation sampler's random
    strategy).  Index t of train_pos holds chunk t+1's sources."""
    dsts = np.unique(np.concatenate(
        [train_bundle['edges'][t][1].cpu().numpy()
         for t in range(train_bundle['T'])]))
    out = []
    for t in range(train_bundle['T']):
        pos = train_bundle['train_pos'][t]
        n = int(pos.shape[1])
        if n == 0:
            out.append(torch.zeros(2, 0, dtype=torch.long, device=device))
            continue
        neg_dst = dsts[rng.randint(0, len(dsts), n)]
        out.append(torch.stack([
            pos[0],
            torch.tensor(neg_dst, dtype=torch.long, device=device)]))
    return out
