# -*- coding: utf-8 -*-
"""Brute-force oracles for the SFS paths that tools/sfs_selftest.py does not
cover: the second-order state, the edge-feature state, the sketch states
(first and second order), the set encoder's last-M record selection, and
the train/eval symmetry of reads (the same (node, t) read inside a training
batch and inside an evaluation batch must be identical).  Every oracle is a
literal sum over the stream's records strictly before t (rank rule)."""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from tgmin import sfs                                # noqa: E402
from tools.sfs_selftest import synth                 # noqa: E402


def records_before(feeder, node, t):
    """(partner, t_e, eid) of every record of `node` with rank < rank(t)."""
    rq = np.searchsorted(feeder.tsu, t, side='left')
    sel = feeder.rank < rq
    out = []
    for msk, part in ((sel & (feeder.u == node), feeder.v),
                      (sel & (feeder.v == node), feeder.u)):
        for idx in np.where(msk)[0]:
            out.append((int(part[idx]), float(feeder.t[idx]), int(idx)))
    return out


def main():
    dev = sys.argv[1] if len(sys.argv) > 1 else (
        'cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)
    edges, ts, N = synth(seed=7, N=24, E=480, n_chunks=6)
    lam, _ = sfs.scale_grid(np.concatenate(ts), per_decade=1, kmax=6)
    feeder = sfs.SFSFeeder(N, edges, ts, lam, dev)
    Fd = 3
    rng = np.random.RandomState(5)
    feat = rng.randn(feeder.E_n, Fd).astype(np.float32)
    feeder.set_features(torch.tensor(feat))
    model = sfs.SFSModel(N, lam, d=16, rank=4, second_order=True, static=True,
                         squash=True, feat_dim=Fd, sketch_dim=32, set_M=4,
                         dropout=0.0, fresh_state=True).to(dev)
    model.eval()
    K = model.K
    worst = {'x2': 0.0, 'xf': 0.0, 's1': 0.0, 's2': 0.0, 'lastM': 0.0,
             'symmetry': 0.0, 'nofs': 0.0}
    model_nofs = sfs.SFSModel(N, lam, d=16, rank=4, second_order=True,
                              static=True, squash=True, feat_dim=Fd,
                              sketch_dim=32, set_M=4, dropout=0.0,
                              fresh_state=False).to(dev)
    model_nofs.load_state_dict(model.state_dict())
    model_nofs.eval()
    rq_rng = np.random.RandomState(11)
    with torch.no_grad():
        for c in range(1, feeder.n_chunks):
            ent = feeder.entering(c)
            states = model.entry_states(ent, feeder, c)
            x1, x2, m1, xf, sk = states
            m1_np = m1.cpu().numpy().astype(np.float64)
            s1e_np = sk[0].cpu().numpy().astype(np.float64)       # [N,K,ds]
            R_np = model.R.cpu().numpy().astype(np.float64)
            e = edges[c]
            t_all = np.asarray(ts[c], dtype=np.float64)
            # queries: chunk events' endpoints at their own times, plus a
            # few random nodes at random times inside the chunk
            qn = list(e[0][:12]) + list(e[1][:12]) + list(rq_rng.randint(0, N, 8))
            qt = list(t_all[:12]) + list(t_all[:12]) + list(
                rq_rng.uniform(t_all[0], t_all[-1], 8))
            qn = np.asarray(qn, dtype=np.int64)
            qt = np.asarray(qt, dtype=np.float64)
            xu1n, xu2n, n_tot, xufn, sku = model.read(
                feeder, c, ent, states, qn, qt)
            # default: same counts, state = decayed entering state
            xn = model_nofs.read(feeder, c, ent, model_nofs.entry_states(
                ent, feeder, c), qn, qt)
            dec = torch.exp(-model.lam.unsqueeze(0) * (
                torch.tensor(qt, device=dev) - ent['c_start']).unsqueeze(1)).float()
            x_ref = (dec.unsqueeze(-1) * x1[torch.tensor(qn, device=dev)]
                     / torch.sqrt(1.0 + n_tot).unsqueeze(-1))
            worst['nofs'] = max(worst['nofs'],
                                float((xn[2] - n_tot).abs().max()),
                                float((xn[0] - x_ref).abs().max()))
            for i in range(qn.shape[0]):
                recs = records_before(feeder, int(qn[i]), float(qt[i]))
                cnt = np.zeros(K)
                x2o = np.zeros((K, model.d))
                xfo = np.zeros((K, Fd))
                s1o = np.zeros((K, model.sketch_dim))
                s2o = np.zeros((K, model.sketch_dim))
                for p, te, eid in recs:
                    w = np.exp(-lam * (qt[i] - te))
                    cnt += w
                    x2o += w[:, None] * m1_np[p][None, :]
                    xfo += w[:, None] * feat[eid][None, :]
                    s1o += w[:, None] * R_np[p][None, :]
                    s2o += w[:, None] * s1e_np[p]
                x2o = x2o / (1.0 + cnt)[:, None]
                xfo = xfo / (1.0 + cnt)[:, None]
                worst['x2'] = max(worst['x2'], float(np.abs(
                    xu2n[i].cpu().numpy() - x2o).max()))
                worst['xf'] = max(worst['xf'], float(np.abs(
                    xufn[i].cpu().numpy() - xfo).max()))
                worst['s1'] = max(worst['s1'], float(np.abs(
                    sku[0][i].cpu().numpy() - s1o).max()))
                worst['s2'] = max(worst['s2'], float(np.abs(
                    sku[1][i].cpu().numpy() - s2o).max()))
                # last-M records: the M latest by rank, strictly before t
                recs_sorted = sorted(recs, key=lambda r: (
                    np.searchsorted(feeder.tsu, r[1], side='left'), r[2]))
                lastM = recs_sorted[-model.set_M:]
                q_idx, partner, dt, eids = feeder.last_events(
                    qn[i:i + 1], qt[i:i + 1], model.set_M)
                got = sorted(zip(partner.tolist(), eids.tolist()))
                exp = sorted((p, eid) for p, _, eid in lastM)
                worst['lastM'] = max(worst['lastM'], 0.0 if got == exp else 1.0)
            # train/eval symmetry: the same (node, t) inside a big batch and
            # alone
            for i in range(0, qn.shape[0], 5):
                alone = model.read(feeder, c, ent, states, qn[i:i + 1],
                                   qt[i:i + 1])
                worst['symmetry'] = max(
                    worst['symmetry'],
                    float((alone[0][0] - xu1n[i]).abs().max()),
                    float((alone[1][0] - xu2n[i]).abs().max()),
                    float((alone[2][0] - n_tot[i]).abs().max()))
    for k_, v_ in worst.items():
        print('%-9s max|diff| = %.3e' % (k_, v_))
    assert worst['x2'] < 1e-4 and worst['xf'] < 1e-4, worst
    assert worst['s1'] < 1e-3 and worst['s2'] < 1e-2, worst
    assert worst['lastM'] == 0.0 and worst['symmetry'] < 1e-5, worst
    assert worst['nofs'] < 1e-5, worst
    print('SFS-SELFTEST2-OK')


if __name__ == '__main__':
    main()
