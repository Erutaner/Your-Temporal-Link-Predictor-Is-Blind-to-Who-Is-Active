# -*- coding: utf-8 -*-
"""Brute-force verification of tgmin.sfs on a synthetic stream.

Checks (all against Python-loop oracles):
  1. raw first-order state and decayed count at random (node, t) queries,
     including queries AT an existing timestamp (tie rule: excluded);
  2. exact multi-scale pair counts;
  3. the entering + fresh split reproduces the oracle for queries inside
     every chunk (the split point is chunk-start on the rank axis);
  4. second-order path, scoring and backward run; zero-init channel head
     leaves the score equal to the plain readout at init.
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from tgmin import sfs  # noqa: E402


def synth(seed=0, N=24, E=360, n_chunks=6):
    rng = np.random.RandomState(seed)
    # timestamps with many ties and a broad spread
    t = np.sort(np.round(np.exp(rng.uniform(0, 8, E)), 0))
    u = rng.randint(0, N // 2, E)
    v = rng.randint(N // 2, N, E)
    # force repeats of a few pairs
    for i in range(0, E, 7):
        u[i], v[i] = 1, N // 2 + 1
    bounds = np.linspace(0, E, n_chunks + 1).astype(int)
    edges, ts = [], []
    for c in range(n_chunks):
        lo, hi = bounds[c], bounds[c + 1]
        edges.append(np.stack([u[lo:hi], v[lo:hi]]))
        ts.append(t[lo:hi])
    return edges, ts, N


def main():
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    edges, ts, N = synth()
    lam, ginfo = sfs.scale_grid(np.concatenate(ts), per_decade=1, kmax=6)
    print('grid', ginfo, lam)
    feeder = sfs.SFSFeeder(N, edges, ts, lam, dev)
    torch.manual_seed(0)
    model = sfs.SFSModel(N, lam, d=16, rank=4, second_order=True, sketch_dim=32, set_M=5, fresh_state=True).to(dev)
    E_np = model.E.weight.detach().cpu().numpy().astype(np.float64)
    rng = np.random.RandomState(1)
    worst = 0.0
    worst_n = 0.0
    worst_p = 0.0
    n_checked = 0
    for c in range(feeder.n_chunks):
        tc = ts[c]
        if tc.size == 0:
            continue
        ent = feeder.entering(c)
        states = model.entry_states(ent)
        # queries: random nodes at chunk timestamps (ties!) and between
        qs_t = np.concatenate([rng.choice(tc, 6), tc[:2] + 0.5])
        qs_t = qs_t[qs_t >= tc[0]]
        qs_n = rng.randint(0, N, qs_t.shape[0])
        x1n, x2n, n_tot, _, _ = model.read(feeder, c, ent, states, qs_n, qs_t)
        x1raw = (x1n * torch.sqrt(1.0 + n_tot).unsqueeze(-1)).detach().cpu().numpy()
        n_np = n_tot.detach().cpu().numpy()
        for i in range(qs_t.shape[0]):
            xo, no = sfs.oracle_state(feeder, E_np, int(qs_n[i]),
                                      float(qs_t[i]))
            worst = max(worst, float(np.abs(x1raw[i] - xo).max()))
            worst_n = max(worst_n, float(np.abs(n_np[i] - no).max()))
            n_checked += 1
        # pair counts
        pu = rng.randint(0, N // 2, qs_t.shape[0])
        pv = rng.randint(N // 2, N, qs_t.shape[0])
        pu[0], pv[0] = 1, N // 2 + 1          # the repeated pair
        pc, _ = feeder.pair_counts(pu, pv, qs_t)
        pc = pc.detach().cpu().numpy()
        for i in range(qs_t.shape[0]):
            po = sfs.oracle_pair_count(feeder, int(pu[i]), int(pv[i]),
                                       float(qs_t[i]))
            worst_p = max(worst_p, float(np.abs(pc[i] - po).max()))
        # score + backward
        z = model.score(feeder, c, ent, states, pu, pv, qs_t)
        assert z.shape == (qs_t.shape[0], 2)
        z.sum().backward()
        model.zero_grad(set_to_none=True)
    print('checked %d queries: max|x1 - oracle| = %.3e, max|n - oracle| = '
          '%.3e, max|pair - oracle| = %.3e' % (n_checked, worst, worst_n,
                                              worst_p))
    assert worst < 1e-4 and worst_n < 1e-4 and worst_p < 1e-4   # float32 reads
    # zero-init channel head: score == readout on [hu;hv]
    c = 2
    ent = feeder.entering(c)
    states = model.entry_states(ent)
    qs_t = ts[c][:5]
    pu = np.arange(5)
    pv = N // 2 + np.arange(5)
    with torch.no_grad():
        z = model.score(feeder, c, ent, states, pu, pv, qs_t)
        xu1, xu2, _, _, _ = model.read(feeder, c, ent, states, pu, qs_t)
        xv1, xv2, _, _, _ = model.read(feeder, c, ent, states, pv, qs_t)
        x = torch.cat([model.node_repr(xu1, xu2), model.node_repr(xv1, xv2)], 1)
        z0 = model.head(x)
    assert float((z - z0).abs().max()) == 0.0, 'zero-init head not inert'
    print('zero-init channel head inert: OK')
    print('SFS-SELFTEST-OK')


if __name__ == '__main__':
    main()
