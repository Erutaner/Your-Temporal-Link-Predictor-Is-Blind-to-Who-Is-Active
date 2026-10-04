# -*- coding: utf-8 -*-
"""Preprocess a raw event table into the standard ml_* dataset format the
package's loaders consume.

Input: a CSV whose columns are

    u, i, ts, label[, f1, f2, ...]

  u      source node id (int)
  i      destination node id (int)
  ts     event timestamp (float; any monotone unit)
  label  per-interaction binary label (0/1; use 0 everywhere if the
         dataset has no node-classification task)
  f*     optional edge-feature columns (floats)

A header row is auto-detected (if the first field of the first line is not
numeric it is treated as a header).  Rows are sorted by ts (stable).

Output, under <out_root>/processed_data/<name>/:

    ml_<name>.csv        columns u, i, ts, label, idx
                         - node ids reindexed to 1..N (id 0 is a dead pad;
                           with --bipartite the destination ids get their
                           own range above the sources)
                         - idx = 1..E in ts order (strictly increasing)
    ml_<name>.npy        [E+1, F] float32 edge features; row 0 is an
                         all-zero pad row (edge id e -> row e).  If the
                         input has no feature columns a single zero column
                         is written.
    ml_<name>_node.npy   [N+1, 1] zeros (placeholder node features; the
                         models in this package do not consume node
                         features)

Usage:
    python scripts/preprocess.py --csv raw_events.csv --name mydata \
        --out_root <dir> [--bipartite]

Constraints checked: at most 172 feature columns (the loader's padding
cap); non-negative integer node ids; at least one edge.
"""
import argparse
import csv
import os

import numpy as np
import pandas as pd


def _has_header(csv_path):
    with open(csv_path, newline='', encoding='utf-8') as f:
        first = next(csv.reader(f))
    try:
        float(first[0])
        return False
    except ValueError:
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True, help='raw event table')
    p.add_argument('--name', required=True, help='dataset name')
    p.add_argument('--out_root', default='.',
                   help='processed_data/ is created under this directory')
    p.add_argument('--bipartite', action='store_true',
                   help='destinations get their own id range above sources')
    a = p.parse_args()

    header = 0 if _has_header(a.csv) else None
    df = pd.read_csv(a.csv, header=header)
    assert df.shape[1] >= 4, 'need at least u, i, ts, label columns'
    df.columns = ['u', 'i', 'ts', 'label'] + [
        'f%d' % k for k in range(df.shape[1] - 4)]
    n_feat = df.shape[1] - 4
    assert n_feat <= 172, 'at most 172 edge-feature columns supported'
    assert len(df) > 0, 'empty event table'

    df['u'] = df['u'].astype(np.int64)
    df['i'] = df['i'].astype(np.int64)
    df['ts'] = df['ts'].astype(np.float64)
    df['label'] = df['label'].astype(np.float64)
    assert (df['u'] >= 0).all() and (df['i'] >= 0).all(), \
        'node ids must be non-negative integers'

    # chronological order (stable, so equal-ts rows keep their input order)
    df = df.sort_values('ts', kind='stable').reset_index(drop=True)

    # reindex node ids to 1..N (id 0 stays a dead pad row)
    if a.bipartite:
        u_ids = {x: k + 1 for k, x in enumerate(np.unique(df['u']))}
        off = len(u_ids)
        i_ids = {x: off + k + 1 for k, x in enumerate(np.unique(df['i']))}
        df['u'] = df['u'].map(u_ids)
        df['i'] = df['i'].map(i_ids)
    else:
        all_ids = {x: k + 1 for k, x in enumerate(
            np.unique(np.concatenate([df['u'].values, df['i'].values])))}
        df['u'] = df['u'].map(all_ids)
        df['i'] = df['i'].map(all_ids)

    df['idx'] = np.arange(1, len(df) + 1, dtype=np.int64)

    out_dir = os.path.join(a.out_root, 'processed_data', a.name)
    os.makedirs(out_dir, exist_ok=True)

    df[['u', 'i', 'ts', 'label', 'idx']].to_csv(
        os.path.join(out_dir, 'ml_%s.csv' % a.name), index=False)

    if n_feat > 0:
        feats = df[['f%d' % k for k in range(n_feat)]].values.astype(
            np.float32)
    else:
        feats = np.zeros((len(df), 1), dtype=np.float32)
    edge_feat = np.vstack([np.zeros((1, feats.shape[1]), dtype=np.float32),
                           feats])                    # row 0 = zero pad
    np.save(os.path.join(out_dir, 'ml_%s.npy' % a.name), edge_feat)

    max_id = int(max(df['u'].max(), df['i'].max()))
    node_feat = np.zeros((max_id + 1, 1), dtype=np.float32)
    np.save(os.path.join(out_dir, 'ml_%s_node.npy' % a.name), node_feat)

    print('wrote %s: %d events, %d nodes, %d feature dims (+zero pad row)'
          % (out_dir, len(df), max_id, feats.shape[1]))


if __name__ == '__main__':
    main()
