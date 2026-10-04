# -*- coding: utf-8 -*-
"""SFS -- the scale-free continuous-time state trunk.

Per node u and scale k the state is

    x_u^k(t) = sum_{e ∋ u, t_e < t} exp(-lam_k (t - t_e)) psi(e)

with psi(e) = E[partner(e)] (first order) and, for the second order, the
mixed first-order state of the partner.  The scale grid lam_k is
log-spaced over the stream's own measured decades (dilation invariance:
the only grid invariant under t -> c t is log-uniform).

There is no recurrence: x is a closed-form linear function of the history
and of E.  Two exact pieces make up every read at a query (u, t) inside
chunk c (c_start = the chunk's first timestamp):

  entering part  exp(-lam_k (t - c_start)) . [A^k(c) E]_u,  A^k(c) the
                 sparse decayed adjacency of ALL events with t_e < c_start
                 (one spmm per scale; differentiable in E; no staleness),
  fresh part     sum over u's events with c_start <= t_e and
                 rank(t_e) < rank(t) of exp(-lam_k (t - t_e)) E[partner]
                 -- the strict-before tie rule on the exact timestamp rank.

Second order at entry: A^k(c) @ m1 with m1 the mixed first-order state;
its fresh part uses m1[partner] (the inner hop tolerates the chunk lag).

Exact multi-scale pair counts n_uv^k(t-) and endpoint counts n_u^k(t-)
ride alongside as scalar channels through a zero-init head.

Everything the feeder holds is index structure (no trained parameter);
everything learned lives in SFSModel.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .readout import make_readout, smooth_max_logits

FRESH_SLICE = 1 << 18        # (query, event) pairs per gather slice
PAIR_SLICE = 1 << 18


# --------------------------------------------------------------------------
# scale grid
# --------------------------------------------------------------------------
def scale_grid(ts_all, per_decade=2, kmax=16, kmin=2):
    """Log-spaced rates covering [1/span, 1/dt_min] of the stream's own
    timestamps, `per_decade` rates per decade.  Returns float64 [K]."""
    ts = np.unique(np.asarray(ts_all, dtype=np.float64))
    assert ts.shape[0] >= 2, 'a stream needs at least two timestamps'
    dts = np.diff(ts)
    dt_min = float(dts[dts > 0].min())
    span = float(ts[-1] - ts[0])
    lam_max = 1.0 / dt_min
    lam_min = 1.0 / span
    decades = math.log10(lam_max / lam_min)
    K = int(math.ceil(decades * per_decade)) + 1
    K = max(kmin, min(kmax, K))
    if K == 1:
        # a single scale: the geometric middle of the covered range
        return (np.array([math.sqrt(lam_max * lam_min)], dtype=np.float64),
                {'dt_min': dt_min, 'span': span, 'decades': decades, 'K': 1})
    lam = np.logspace(math.log10(lam_max), math.log10(lam_min), K)
    return lam.astype(np.float64), {'dt_min': dt_min, 'span': span,
                                    'decades': decades, 'K': K}


# --------------------------------------------------------------------------
# feeder: index structure of one stream (no parameter)
# --------------------------------------------------------------------------
class SFSFeeder(object):
    """Holds the flattened, time-sorted event stream and the sorted index
    structures needed for (i) the entering sparse adjacency of any chunk,
    (ii) exact strict-before fresh reads at raw query timestamps, (iii)
    exact multi-scale pair counts."""

    def __init__(self, num_nodes, edges_np, ts_np, lambdas, device):
        self.N = int(num_nodes)
        self.lam_np = np.asarray(lambdas, dtype=np.float64)
        self.K = int(self.lam_np.shape[0])
        self.device = torch.device(device)
        self.lam = torch.tensor(self.lam_np, device=self.device)
        us, vs, ts, cs, c_start = [], [], [], [], []
        self.n_chunks = len(edges_np)
        for c, (e, t) in enumerate(zip(edges_np, ts_np)):
            if e is None or np.asarray(e).size == 0:
                c_start.append(np.nan)
                continue
            e = np.asarray(e)
            t = np.asarray(t, dtype=np.float64)
            assert e.shape[1] == t.shape[0], (c, e.shape, t.shape)
            assert bool((np.diff(t) >= 0).all()), 'chunk %d unsorted' % c
            us.append(e[0].astype(np.int64))
            vs.append(e[1].astype(np.int64))
            ts.append(t)
            cs.append(np.full(t.shape[0], c, dtype=np.int64))
            c_start.append(float(t[0]))
        self.u = np.concatenate(us)
        self.v = np.concatenate(vs)
        self.t = np.concatenate(ts)
        self.c = np.concatenate(cs)
        assert bool((np.diff(self.t) >= 0).all()), 'stream not time-sorted'
        assert int(max(self.u.max(), self.v.max())) < self.N
        self.E_n = int(self.t.shape[0])
        # empty chunks take the next non-empty chunk's start (never queried)
        cs_arr = np.asarray(c_start, dtype=np.float64)
        for c in range(self.n_chunks - 1, -1, -1):
            if np.isnan(cs_arr[c]):
                cs_arr[c] = cs_arr[c + 1] if c + 1 < self.n_chunks else np.inf
        self.c_start = cs_arr
        # exact timestamp ranks (the strict-before rule lives on ranks)
        self.tsu = np.unique(self.t)
        self.M = int(self.tsu.shape[0]) + 1
        self.rank = np.searchsorted(self.tsu, self.t, side='left')
        # ---- directed pair id space (both directions) for the sparse A
        self.rows = np.concatenate([self.u, self.v])       # [2E]
        self.cols = np.concatenate([self.v, self.u])
        key = self.rows * np.int64(self.N) + self.cols
        uk, inv = np.unique(key, return_inverse=True)
        self.P = int(uk.shape[0])
        self.pair_rows = torch.tensor(uk // self.N, device=self.device)
        self.pair_cols = torch.tensor(uk % self.N, device=self.device)
        self.ev_pair = torch.tensor(inv.astype(np.int64), device=self.device)  # [2E]
        self.t_dir = torch.tensor(np.concatenate([self.t, self.t]),
                                  device=self.device)                   # [2E]
        # ---- (node, rank)-sorted endpoint records for fresh reads
        node = np.concatenate([self.u, self.v])
        partner = np.concatenate([self.v, self.u])
        rk = np.concatenate([self.rank, self.rank])
        tt = np.concatenate([self.t, self.t])
        eid = np.concatenate([np.arange(self.E_n), np.arange(self.E_n)])
        order = np.lexsort((rk, node))
        self.nr_node = node[order]
        self.nr_partner = partner[order]
        self.nr_t = tt[order]
        self.nr_eid = eid[order]
        self.nr_comp = self.nr_node * np.int64(self.M) + rk[order]
        # event-incidence rows for the optional edge-feature state:
        # directed record r <-> stream event r mod E_n
        self.ev_of_dir = torch.tensor(np.concatenate(
            [np.arange(self.E_n), np.arange(self.E_n)]), device=self.device)
        self.node_of_dir = torch.tensor(self.rows, device=self.device)
        self.feat = None                       # [E_n, F] float32 (optional)
        # ---- (pair, rank)-sorted records for exact pair counts
        a = np.minimum(self.u, self.v)
        b = np.maximum(self.u, self.v)
        pkey = a * np.int64(self.N) + b
        porder = np.lexsort((self.rank, pkey))
        self.pk_sorted = pkey[porder]
        self.pr_t = self.t[porder]
        self.pr_comp = self.pk_sorted * np.int64(self.M) + self.rank[porder]

    # ------------------------------------------------------------ entering
    def n_before(self, c):
        """Number of stream events with t < c_start(c)."""
        return int(np.searchsorted(self.t, self.c_start[c], side='left'))

    def entering(self, c):
        """(idx [2, P] long, w [K, P] float32, ncount [N, K] float32,
        c_start) -- the sparse decayed adjacency of every event strictly
        before chunk c's first timestamp, per scale, plus row sums."""
        n_c = self.n_before(c)
        cst = float(self.c_start[c])
        if n_c == 0 or not np.isfinite(cst):
            w = torch.zeros((self.K, self.P), device=self.device)
            ncount = torch.zeros((self.N, self.K), device=self.device)
            return {'idx': torch.stack([self.pair_rows, self.pair_cols]),
                    'w': w, 'ncount': ncount, 'c_start': cst, 'n_events': 0}
        sel = torch.cat([torch.arange(0, n_c, device=self.device),
                         torch.arange(self.E_n, self.E_n + n_c,
                                      device=self.device)])
        age = (cst - self.t_dir[sel]).clamp(min=0.0)            # [2n] f64
        wev = torch.exp(-self.lam.unsqueeze(1) * age.unsqueeze(0))  # [K,2n]
        pid = self.ev_pair[sel]
        w = torch.zeros((self.K, self.P), dtype=torch.float64,
                        device=self.device)
        w.index_add_(1, pid, wev)
        ncount = torch.zeros((self.N, self.K), dtype=torch.float64,
                             device=self.device)
        ncount.index_add_(0, self.pair_rows, w.t())
        return {'idx': torch.stack([self.pair_rows, self.pair_cols]),
                'w': w.float(), 'ncount': ncount.float(), 'c_start': cst,
                'n_events': n_c}

    # --------------------------------------------------------------- fresh
    def set_features(self, feat_events):
        """Per-event edge features in stream order, [E_n, F] float32."""
        feat_events = torch.as_tensor(feat_events, dtype=torch.float32)
        assert feat_events.shape[0] == self.E_n, (feat_events.shape, self.E_n)
        self.feat = feat_events.to(self.device)

    def entering_feat(self, c, ent):
        """[N, K, F]: decayed sums of the edge features of every event with
        t_e < c_start(c), both endpoints (constant: no parameter inside)."""
        n_c = ent['n_events']
        Fd = self.feat.shape[1]
        if n_c == 0:
            return torch.zeros((self.N, self.K, Fd), device=self.device)
        sel = torch.cat([torch.arange(0, n_c, device=self.device),
                         torch.arange(self.E_n, self.E_n + n_c,
                                      device=self.device)])
        age = (ent['c_start'] - self.t_dir[sel]).clamp(min=0.0)
        wev = torch.exp(-self.lam.unsqueeze(1) * age.unsqueeze(0)).float()
        idx = torch.stack([self.node_of_dir[sel], self.ev_of_dir[sel]])
        outs = []
        with torch.no_grad():
            for k in range(self.K):
                S = torch.sparse_coo_tensor(idx, wev[k], (self.N, self.E_n))
                outs.append(torch.sparse.mm(S, self.feat))
        return torch.stack(outs, dim=1)

    def fresh(self, c, nodes, t_query, with_eid=False):
        """Strict-before fresh records of the queried nodes inside chunk c:
        (q_idx [nnz] long, partner [nnz] long, dt [nnz] f64, kfresh [nq])
        [, eid [nnz]].  nodes/t_query are numpy arrays."""
        nodes = np.asarray(nodes, dtype=np.int64)
        tq = np.asarray(t_query, dtype=np.float64)
        cst = float(self.c_start[c])
        r_q = np.searchsorted(self.tsu, tq, side='left')
        r_c = int(np.searchsorted(self.tsu, cst, side='left'))
        assert bool((tq >= cst).all()), 'query before its chunk start'
        pos = np.searchsorted(self.nr_comp, nodes * np.int64(self.M) + r_q,
                              side='left')
        first = np.searchsorted(self.nr_comp, nodes * np.int64(self.M) + r_c,
                                side='left')
        k = pos - first
        assert bool((k >= 0).all())
        nnz = int(k.sum())
        if nnz == 0:
            e = np.zeros(0, dtype=np.int64)
            out = (e, e, np.zeros(0, dtype=np.float64), k)
            return out + (e,) if with_eid else out
        q_idx = np.repeat(np.arange(nodes.shape[0], dtype=np.int64), k)
        offs = np.arange(nnz, dtype=np.int64) - np.repeat(
            np.cumsum(k) - k, k)
        ev = np.repeat(first, k) + offs
        partner = self.nr_partner[ev]
        dt = tq[q_idx] - self.nr_t[ev]
        assert bool((dt >= 0).all())
        if with_eid:
            return q_idx, partner, dt, k, self.nr_eid[ev]
        return q_idx, partner, dt, k

    # ------------------------------------------------------ last-M events
    def last_events(self, nodes, t_query, M):
        """The last M records of each node strictly before t (rank rule),
        any chunk: (q_idx [nnz], partner [nnz], dt [nnz] f64, eid [nnz])."""
        nodes = np.asarray(nodes, dtype=np.int64)
        tq = np.asarray(t_query, dtype=np.float64)
        r_q = np.searchsorted(self.tsu, tq, side='left')
        pos = np.searchsorted(self.nr_comp, nodes * np.int64(self.M) + r_q,
                              side='left')
        start_node = np.searchsorted(self.nr_node, nodes, side='left')
        first = np.maximum(start_node, pos - int(M))
        k = pos - first
        nnz = int(k.sum())
        if nnz == 0:
            e = np.zeros(0, dtype=np.int64)
            return e, e, np.zeros(0, dtype=np.float64), e
        q_idx = np.repeat(np.arange(nodes.shape[0], dtype=np.int64), k)
        offs = np.arange(nnz, dtype=np.int64) - np.repeat(
            np.cumsum(k) - k, k)
        ev = np.repeat(first, k) + offs
        dt = tq[q_idx] - self.nr_t[ev]
        assert bool((dt >= 0).all())
        return q_idx, self.nr_partner[ev], dt, self.nr_eid[ev]

    # ---------------------------------------------------------- pair counts
    def pair_counts(self, u, v, t_query):
        """[nq, K] float32 tensor: exact decayed counts of the unordered
        pair's events strictly before t_query, per scale."""
        u = np.asarray(u, dtype=np.int64)
        v = np.asarray(v, dtype=np.int64)
        tq = np.asarray(t_query, dtype=np.float64)
        nq = int(u.shape[0])
        a = np.minimum(u, v)
        b = np.maximum(u, v)
        pkey = a * np.int64(self.N) + b
        r_q = np.searchsorted(self.tsu, tq, side='left')
        pos = np.searchsorted(self.pr_comp, pkey * np.int64(self.M) + r_q,
                              side='left')
        start = np.searchsorted(self.pk_sorted, pkey, side='left')
        k = pos - start
        k = np.where(k < 0, 0, k)
        out = torch.zeros((nq, self.K), dtype=torch.float64,
                          device=self.device)
        nnz = int(k.sum())
        if nnz > 0:
            q_idx = np.repeat(np.arange(nq, dtype=np.int64), k)
            offs = np.arange(nnz, dtype=np.int64) - np.repeat(
                np.cumsum(k) - k, k)
            ev = np.repeat(start, k) + offs
            dt = tq[q_idx] - self.pr_t[ev]
            assert bool((dt >= 0).all())
            for lo in range(0, nnz, PAIR_SLICE):
                hi = min(nnz, lo + PAIR_SLICE)
                dts = torch.tensor(dt[lo:hi], device=self.device)
                w = torch.exp(-dts.unsqueeze(1) * self.lam.unsqueeze(0))
                out.index_add_(0, torch.tensor(q_idx[lo:hi],
                                               device=self.device), w)
        return out.float(), k


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
class SFSModel(nn.Module):
    """The scale-free state trunk with its readout (module docstring)."""

    def __init__(self, num_nodes, lambdas, d=128, rank=16, hidden=64,
                 second_order=True, num_classes=2, e_train=True,
                 static=True, squash=False, dropout=0.0, feat_dim=0,
                 sketch_dim=0, sketch_seed=7, set_M=0, set_hidden=64,
                 fresh_state=False, fresh_records=True, count_channels=True):
        super().__init__()
        # fresh_records=True (the default): the within-chunk
        # records strictly before t enter the counts, the sketches and the
        # feature states (the event clock).  False is the chunking ablation: the
        # state is the entering state decayed to t only (the chunk clock);
        # the pair counts and the set encoder are unaffected.
        self.fresh_records = bool(fresh_records)
        # count_channels=False (trunk ablation): the exact pair / endpoint
        # count channels of the scoring head are zeroed; everything else,
        # including the counts the activity process reads, is unchanged.
        self.count_channels = bool(count_channels)
        # fresh_state=False (the default): the state lives on the
        # chunk clock (entering state decayed to t) and the within-chunk
        # records enter only the scalar channels (counts, pair counts,
        # sketch, set encoder).  True keeps the exact within-chunk partner
        # sums in the state (the sufficient-statistic form; a memorisation
        # channel on session streams).
        self.fresh_state = bool(fresh_state)
        self.feat_dim = int(feat_dim)
        self.sketch_dim = int(sketch_dim)
        self.set_M = int(set_M)
        self.set_hidden = int(set_hidden)
        self.e_train = bool(e_train)
        self.use_static = bool(static)
        self.squash = bool(squash)
        self.dropout = float(dropout)
        self.N = int(num_nodes)
        self.register_buffer('lam', torch.tensor(np.asarray(lambdas,
                                                            dtype=np.float64)))
        self.K = int(self.lam.shape[0])
        self.d = int(d)
        self.rank = int(rank)
        self.second = bool(second_order)
        K, d, r = self.K, self.d, self.rank
        self.E = nn.Embedding(self.N, d)
        nn.init.xavier_uniform_(self.E.weight)
        if not self.e_train:
            # fixed random partner codes: the state is then an unbiased
            # random-feature sketch of the exact decayed counts (no node
            # identity can be memorised through E)
            self.E.weight.requires_grad_(False)
        self.P1 = nn.Linear(d, d, bias=False)
        self.P2 = nn.Linear(d, d, bias=False)
        nn.init.xavier_uniform_(self.P1.weight)
        nn.init.xavier_uniform_(self.P2.weight)
        self.g1 = nn.Parameter(torch.full((K,), 1.0 / K))
        self.g2 = nn.Parameter(torch.zeros(K))
        self.W = nn.Linear(d, d)
        nn.init.xavier_uniform_(self.W.weight)
        self.head = make_readout(d, num_classes=num_classes, hidden=hidden)
        # per-scale low-rank bilinear channels: (1,1), (1,2), (2,1)
        def _bl():
            p = nn.Parameter(torch.empty(K, r, d))
            for k in range(K):
                nn.init.xavier_uniform_(p.data[k])
            return p
        self.A11, self.B11 = _bl(), _bl()
        self.A12, self.B12 = _bl(), _bl()
        self.A21, self.B21 = _bl(), _bl()
        self.n_feat = 6 * K + 1
        if self.sketch_dim > 0:
            # fixed Rademacher codes: the sketch states s1 = A R, s2 = A s1
            # give unbiased estimates of the decayed walk counts through
            # inner products (<R_p, R_p'> = 1 if p = p', ~N(0, 1/ds) else);
            # four count channels per scale, log1p, no parameter inside
            g = torch.Generator().manual_seed(int(sketch_seed))
            R = (torch.randint(0, 2, (self.N, self.sketch_dim), generator=g)
                 .float() * 2.0 - 1.0) / math.sqrt(self.sketch_dim)
            self.register_buffer('R', R)
            self.n_feat += 4 * K
        if self.feat_dim > 0:
            # edge-feature state: decayed per-scale means of the event
            # features; mixed over scales into the node representation
            # (zero-init map: the feature-free model is the zero point)
            # and a per-scale bilinear cross channel (K more scalars)
            self.gf = nn.Parameter(torch.full((K,), 1.0 / K))
            self.Wf = nn.Linear(self.feat_dim, d, bias=False)
            nn.init.zeros_(self.Wf.weight)
            self.Cf = nn.Parameter(torch.empty(K, r, self.feat_dim))
            for k in range(K):
                nn.init.xavier_uniform_(self.Cf.data[k])
            self.n_feat += K
        self.ch_pre = nn.Parameter(torch.zeros(hidden, self.n_feat))
        self.ch_cert = nn.Parameter(torch.zeros(num_classes, self.n_feat))
        # cold prior: a learned state vector for nodes with (almost) no
        # history, fading as 1/(1 + n) with the node's own decayed count at
        # the slowest scale (the Bayesian prior of the state; zero-init)
        self.cold = nn.Parameter(torch.zeros(d))
        if self.set_M > 0:
            # candidate-conditioned Deep-Sets encoder over the source's last
            # M events: per event [decays (K); log1p sketch co-occurrence of
            # the event's partner with the candidate (K); learned similarity
            # (1); edge features (F)] -> MLP -> sum -> zero-init heads
            assert self.sketch_dim > 0, 'the set encoder needs the sketch'
            in_dim = 2 * K + 1 + self.feat_dim
            self.se_mlp = nn.Sequential(
                nn.Linear(in_dim, self.set_hidden), nn.GELU(),
                nn.Linear(self.set_hidden, self.set_hidden), nn.GELU())
            self.se_sim = nn.Linear(d, d, bias=False)
            nn.init.xavier_uniform_(self.se_sim.weight)
            self.se_pre = nn.Parameter(torch.zeros(hidden, self.set_hidden))
            self.se_cert = nn.Parameter(torch.zeros(num_classes,
                                                    self.set_hidden))

    # ---------------------------------------------------------- utilities
    def hot_params(self):
        hot = [self.ch_pre, self.ch_cert]
        if self.set_M > 0:
            hot += [self.se_pre, self.se_cert]
        return hot

    def set_encode(self, feeder, u, v, t_query, states, xv_m1):
        """[nq, set_hidden]: Deep-Sets code of the source's last M events,
        conditioned on the candidate v (xv_m1 = m1-mixed candidate rows,
        [nq, d])."""
        x1, x2, m1, xf, sk = states
        dev = m1.device
        nq = int(u.shape[0])
        q_idx, partner, dt, eids = feeder.last_events(u, t_query, self.set_M)
        if q_idx.shape[0] == 0:
            return torch.zeros((nq, self.set_hidden), device=dev)
        qi = torch.tensor(q_idx, device=dev)
        pi = torch.tensor(partner, device=dev)
        dts = torch.tensor(dt, device=dev)
        vt = torch.tensor(np.asarray(v, dtype=np.int64), device=dev)
        w = torch.exp(-dts.unsqueeze(1) * self.lam.unsqueeze(0)).float()  # [nnz,K]
        cols = [w]
        with torch.no_grad():
            s1 = sk[0]                                          # [N,K,ds]
            co = torch.empty((qi.shape[0], self.K), device=dev)
            vq = vt[qi]
            for k in range(self.K):
                co[:, k] = (s1[pi, k] * s1[vq, k]).sum(-1)
            cols.append(0.1 * torch.log1p(co.clamp(min=0.0)))   # O(1)
        sim = (self.se_sim(m1[pi]) * xv_m1[qi]).sum(-1, keepdim=True)
        cols.append(sim / math.sqrt(self.d))
        if self.feat_dim > 0:
            cols.append(feeder.feat[torch.tensor(eids, device=dev)])
        z = self.se_mlp(torch.cat(cols, dim=1))                 # [nnz,H]
        out = torch.zeros((nq, self.set_hidden), device=dev)
        out.index_add_(0, qi, z)
        return out

    def _spmm_all(self, idx, w, dense):
        """[N, K, d]: per-scale sparse (coalesced, fixed index set) times
        dense [N, d]."""
        outs = []
        for k in range(self.K):
            A = torch.sparse_coo_tensor(idx, w[k], (self.N, self.N),
                                        is_coalesced=True)
            outs.append(torch.sparse.mm(A, dense))
        return torch.stack(outs, dim=1)

    def _records(self, q_idx, partner, dt, nq):
        """The within-chunk records of a read, coalesced by (query, key):
        (idx [2, m] long, vals [m, K] float32, uniq [n_u] long, cnt [nq, K]
        float64).  Duplicate (query, key) pairs are summed per scale; idx[1]
        indexes uniq; cnt is the per-query decayed count.  The decay
        weights are computed in slices of FRESH_SLICE records, so the peak
        memory is O(slice K + m K + nnz), never O(nnz K)."""
        dev = self.lam.device
        uniq, inv = np.unique(partner, return_inverse=True)
        n_u = int(uniq.shape[0])
        key = torch.tensor(q_idx.astype(np.int64) * n_u + inv.astype(np.int64),
                           device=dev)
        ukey, pos = torch.unique(key, return_inverse=True)   # sorted keys
        m = int(ukey.shape[0])
        idx = torch.stack([ukey // n_u, ukey % n_u])
        vals = torch.zeros((m, self.K), device=dev, dtype=torch.float64)
        cnt = torch.zeros((nq, self.K), device=dev, dtype=torch.float64)
        qi = torch.tensor(q_idx, device=dev)
        nnz = int(q_idx.shape[0])
        for lo in range(0, nnz, FRESH_SLICE):
            hi = min(nnz, lo + FRESH_SLICE)
            dts = torch.tensor(dt[lo:hi], device=dev)
            w = torch.exp(-dts.unsqueeze(1) * self.lam.unsqueeze(0))  # [s,K]
            vals.index_add_(0, pos[lo:hi], w)
            cnt.index_add_(0, qi[lo:hi], w)
        return idx, vals.float(), torch.tensor(uniq, device=dev), cnt

    def _segment_sum(self, rec, table, perscale=False):
        """[nq, K, d] = sum over the coalesced records of vals[:, k] *
        table[uniq[col]] (perscale: table[uniq[col], k]), as index_add
        segment sums in slices of FRESH_SLICE records.  No sparse-matrix
        kernel: the CUDA COO spmm faulted sporadically on tiny batches."""
        idx, vals, uniq, cnt = rec
        nq = int(cnt.shape[0])
        d = int(table.shape[-1])
        m = int(idx.shape[1])
        acc = [torch.zeros((nq, d), device=table.device, dtype=table.dtype)
               for _ in range(self.K)]
        for lo in range(0, m, FRESH_SLICE):
            hi = min(m, lo + FRESH_SLICE)
            r = idx[0, lo:hi]
            p = uniq[idx[1, lo:hi]]
            if perscale:
                for k in range(self.K):
                    acc[k] = acc[k].index_add(0, r, vals[lo:hi, k:k + 1]
                                              * table[p, k])
            else:
                g = table[p]                                          # [s,d]
                for k in range(self.K):
                    acc[k] = acc[k].index_add(0, r, vals[lo:hi, k:k + 1] * g)
        return torch.stack(acc, dim=1)

    def _direct_sum(self, q_idx, keys, dt, nq, table):
        """[nq, K, d] = sum_{(q,e)} exp(-lam dt) table[keys[e]] accumulated
        record by record in slices of FRESH_SLICE (no coalescing: for keys
        unique per record, e.g. event ids); memory O(slice (K + d))."""
        dev = table.device
        acc = [torch.zeros((nq, table.shape[-1]), device=dev,
                           dtype=table.dtype) for _ in range(self.K)]
        nnz = int(q_idx.shape[0])
        for lo in range(0, nnz, FRESH_SLICE):
            hi = min(nnz, lo + FRESH_SLICE)
            qi = torch.tensor(q_idx[lo:hi], device=dev)
            dts = torch.tensor(dt[lo:hi], device=dev)
            w = torch.exp(-dts.unsqueeze(1) * self.lam.unsqueeze(0)).float()
            g = table[torch.tensor(keys[lo:hi], device=dev)]           # [s,d]
            for k in range(self.K):
                acc[k].index_add_(0, qi, w[:, k:k + 1] * g)
        return torch.stack(acc, dim=1)

    def entry_states(self, ent, feeder=None, c=None):
        """Raw entering states from the feeder's sparse structure:
        x1 [N,K,d], x2 [N,K,d] (zeros if second order off), m1 [N,d],
        xf [N,K,F] or None (edge-feature state, constant)."""
        assert self.feat_dim == 0 or feeder is not None, \
            'the feature state needs the feeder'
        E = self.E.weight
        x1 = self._spmm_all(ent['idx'], ent['w'], E)               # raw
        n1 = ent['ncount']                                         # [N,K]
        x1n = x1 / torch.sqrt(1.0 + n1).unsqueeze(-1)
        m1 = (self.g1.view(1, -1, 1) * self.P1(x1n)).sum(dim=1)    # [N,d]
        if self.second:
            x2 = self._spmm_all(ent['idx'], ent['w'], m1)          # raw
        else:
            x2 = torch.zeros_like(x1)
        xf = feeder.entering_feat(c, ent) if self.feat_dim > 0 else None
        sk = None
        if self.sketch_dim > 0:
            with torch.no_grad():
                s1 = self._spmm_all(ent['idx'], ent['w'], self.R)   # [N,K,ds]
                s2 = torch.stack([torch.sparse.mm(torch.sparse_coo_tensor(
                    ent['idx'], ent['w'][k], (self.N, self.N),
                    is_coalesced=True), s1[:, k]) for k in range(self.K)],
                    dim=1)                                        # [N,K,ds]
            sk = (s1, s2)
        return x1, x2, m1, xf, sk

    def read(self, feeder, c, ent, states, nodes, t_query):
        """Exact reads at (nodes, t_query) inside chunk c: normalized
        first/second-order states [nq,K,d] and decayed counts [nq,K]."""
        x1, x2, m1, xf, sk = states
        E = self.E.weight
        nq = int(nodes.shape[0])
        dev = E.device
        nodes_t = torch.tensor(nodes, device=dev)
        tq = torch.tensor(np.asarray(t_query, dtype=np.float64), device=dev)
        dec = torch.exp(-self.lam.unsqueeze(0)
                        * (tq - ent['c_start']).clamp(min=0).unsqueeze(1))
        dec = dec.float()                                          # [nq,K]
        if self.fresh_records:
            q_idx, partner, dt, _k, eids = feeder.fresh(c, nodes, t_query,
                                                        with_eid=True)
        else:                                   # records off: chunk clock
            q_idx = np.zeros(0, dtype=np.int64)
            partner, eids = q_idx, q_idx
            dt = np.zeros(0, dtype=np.float64)
        nnz = int(q_idx.shape[0])
        # the within-chunk records, coalesced once per key type
        rec_p = self._records(q_idx, partner, dt, nq) if nnz else None

        def zeros_kd(d_):
            return torch.zeros((nq, self.K, d_), device=dev)

        cnt_f = (rec_p[3].float() if rec_p is not None
                 else torch.zeros((nq, self.K), device=dev))
        n_ent = ent['ncount'][nodes_t] * dec                       # [nq,K]
        n_tot = n_ent + cnt_f
        # fresh_state: within-chunk partner sums also enter the state
        # (the exact sufficient-statistic form; default: off)
        f1 = (self._segment_sum(rec_p, E) if self.fresh_state and rec_p
              is not None else zeros_kd(self.d))
        xu1 = dec.unsqueeze(-1) * x1[nodes_t] + f1
        xu1n = xu1 / torch.sqrt(1.0 + n_tot).unsqueeze(-1)
        if self.second:
            f2 = (self._segment_sum(rec_p, m1) if self.fresh_state and rec_p
                  is not None else zeros_kd(self.d))
            xu2 = dec.unsqueeze(-1) * x2[nodes_t] + f2
            xu2n = xu2 / (1.0 + n_tot).unsqueeze(-1)
        else:
            xu2n = torch.zeros_like(xu1n)
        xufn = None
        if self.feat_dim > 0:
            with torch.no_grad():
                ff = (self._direct_sum(q_idx, eids, dt, nq, feeder.feat)
                      if nnz else zeros_kd(self.feat_dim))
                xuf = dec.unsqueeze(-1) * xf[nodes_t] + ff
                xufn = xuf / (1.0 + n_tot).unsqueeze(-1)       # decayed mean
        sku = None
        if self.sketch_dim > 0:
            with torch.no_grad():
                s1, s2 = sk
                if rec_p is not None:
                    fs1 = self._segment_sum(rec_p, self.R)
                    fs2 = self._segment_sum(rec_p, s1, perscale=True)
                else:
                    fs1 = zeros_kd(self.sketch_dim)
                    fs2 = zeros_kd(self.sketch_dim)
                su1 = dec.unsqueeze(-1) * s1[nodes_t] + fs1       # raw counts
                su2 = dec.unsqueeze(-1) * s2[nodes_t] + fs2
            sku = (su1, su2)
        return xu1n, xu2n, n_tot, xufn, sku

    def node_repr(self, x1n, x2n, xfn=None, n_tot=None):
        h = (self.g1.view(1, -1, 1) * self.P1(x1n)).sum(dim=1)
        if self.second:
            h = h + (self.g2.view(1, -1, 1) * self.P2(x2n)).sum(dim=1)
        if self.feat_dim > 0 and xfn is not None:
            h = h + self.Wf((self.gf.view(1, -1, 1) * xfn).sum(dim=1))
        if n_tot is not None:
            h = h + self.cold.view(1, -1) / (1.0 + n_tot[:, -1:])
        return torch.tanh(self.W(h))

    def score(self, feeder, c, ent, states, u, v, t_query, xu=None):
        """[nq, 2] logits for the queried pairs at their raw timestamps.
        xu: optional precomputed read of the sources (shared by the
        positive and negative comparisons of one chunk)."""
        u = np.asarray(u, dtype=np.int64)
        v = np.asarray(v, dtype=np.int64)
        if xu is None:
            xu = self.read(feeder, c, ent, states, u, t_query)
        xu1, xu2, nu, xuf, sku = xu
        xv1, xv2, nv, xvf, skv = self.read(feeder, c, ent, states, v, t_query)
        hu = self.node_repr(xu1, xu2, xuf, nu)
        hv = self.node_repr(xv1, xv2, xvf, nv)
        x = torch.cat([hu, hv], dim=1)

        def bil(A, B, xa, xb):
            au = torch.einsum('qkd,krd->qkr', xa, A)
            bv = torch.einsum('qkd,krd->qkr', xb, B)
            return (au * bv).sum(-1)                               # [nq,K]
        q11 = bil(self.A11, self.B11, xu1, xv1)
        if self.second:
            q12 = bil(self.A12, self.B12, xu1, xv2)
            q21 = bil(self.A21, self.B21, xu2, xv1)
        else:
            q12 = torch.zeros_like(q11)
            q21 = torch.zeros_like(q11)
        if self.squash:
            # signed log link on the bilinear intensities: bounded inputs
            # to the channel head whatever the state magnitudes
            q11 = torch.sign(q11) * torch.log1p(q11.abs())
            q12 = torch.sign(q12) * torch.log1p(q12.abs())
            q21 = torch.sign(q21) * torch.log1p(q21.abs())
        npair, _ = feeder.pair_counts(u, v, t_query)
        E = self.E.weight
        ut = torch.tensor(u, device=E.device)
        vt = torch.tensor(v, device=E.device)
        if self.use_static:
            static = (E[ut] * E[vt]).sum(-1, keepdim=True) / math.sqrt(self.d)
        else:
            static = torch.zeros((u.shape[0], 1), device=E.device)
        if self.count_channels:
            cols = [q11, q12, q21, torch.log1p(npair), torch.log1p(nu),
                    torch.log1p(nv), static]
        else:
            cols = [q11, q12, q21, torch.zeros_like(npair), torch.zeros_like(nu),
                    torch.zeros_like(nv), static]
        if self.sketch_dim > 0:
            su1, su2 = sku
            sv1, sv2 = skv
            for a_, b_ in ((su1, sv1), (su1, sv2), (su2, sv1), (su2, sv2)):
                cnt = (a_ * b_).sum(-1).clamp(min=0.0)             # [nq,K]
                cols.append(0.1 * torch.log1p(cnt))   # O(1) inputs to the head
        if self.feat_dim > 0:
            qff = bil(self.Cf, self.Cf, xuf, xvf)
            if self.squash:
                qff = torch.sign(qff) * torch.log1p(qff.abs())
            cols.append(qff)
        feat = torch.cat(cols, dim=1)
        assert feat.shape[1] == self.n_feat
        if self.dropout > 0 and self.training:
            x = F.dropout(x, p=self.dropout, training=True)
            feat = F.dropout(feat, p=self.dropout, training=True)
        head = self.head
        s = head.pre(x) + F.linear(feat, self.ch_pre)
        cert_add = F.linear(feat, self.ch_cert)
        if self.set_M > 0:
            m1 = states[2]
            code = self.set_encode(feeder, u, v, t_query, states, m1[vt])
            s = s + F.linear(code, self.se_pre)
            cert_add = cert_add + F.linear(code, self.se_cert)
        phi = torch.cat([F.relu(s), F.relu(-s)], dim=-1)
        z = head.nl(phi) + head.cert(x) + cert_add
        return z


# --------------------------------------------------------------------------
# ground process: lambda_u(t) by its point-process likelihood
# --------------------------------------------------------------------------
class GroundProcess(nn.Module):
    """lambda_u(t) = mu + sum_k w_k n_u^k(t) with w_k = lam_k exp(theta_k)
    and mu = lam_min exp(theta_mu) (K + 1 parameters): the multi-exponential
    Hawkes ground process on the scale bank.  Because n_u^k is a sum of
    exponentials, its integral over a window is closed-form (compensator),
    so the process is fitted by exact maximum likelihood, no negatives."""

    def __init__(self, lam):
        super().__init__()
        lam = np.asarray(lam, dtype=np.float64)
        self.register_buffer('lam', torch.tensor(lam))
        self.theta = nn.Parameter(torch.zeros(lam.shape[0]))
        self.theta_mu = nn.Parameter(torch.zeros(()))

    def rates(self):
        w = self.lam.float() * torch.exp(self.theta)
        mu = float(self.lam.min()) * torch.exp(self.theta_mu)
        return mu, w

    def log_intensity(self, n):
        """[nq] log lambda_u(t-) from the sources' decayed incident counts
        n [nq, K] strictly before t."""
        mu, w = self.rates()
        return torch.log(mu + (w.unsqueeze(0) * n).sum(-1))

    def compensator(self, ent, pool_t, T, ev_dt_end):
        """sum_{u in pool} int_{c_start}^{c_end} lambda_u(t) dt, closed form:
        entering counts times (1 - e^{-lam T}) / lam plus, for every
        incident (event, endpoint-in-pool) record inside the window,
        (1 - e^{-lam (c_end - t_e)}) / lam."""
        mu, w = self.rates()
        lam = self.lam
        g = (1.0 - torch.exp(-lam * float(T))) / lam
        ent_term = (ent['ncount'][pool_t].double() * g.unsqueeze(0)).sum(0)
        if ev_dt_end.numel():
            fresh_term = ((1.0 - torch.exp(-lam.unsqueeze(0)
                                           * ev_dt_end.unsqueeze(1)))
                          / lam.unsqueeze(0)).sum(0)
        else:
            fresh_term = torch.zeros_like(g)
        tot = (ent_term + fresh_term).float()
        return mu * float(T) * float(pool_t.numel()) + (w * tot).sum()


def chunk_window(feeder, c):
    """(c_start, c_end) of chunk c: the next chunk's start, or the chunk's
    last event time for the final chunk."""
    cst = float(feeder.c_start[c])
    if c + 1 < feeder.n_chunks and np.isfinite(feeder.c_start[c + 1]):
        cend = float(feeder.c_start[c + 1])
    else:
        cend = float(feeder.t[feeder.c == c].max())
    return cst, cend


class WindowCache(object):
    """Static per-chunk quantities of the compensator: window length and
    c_end - t_e of every incident (event, endpoint) record inside the
    window (by TIME: events tied with the chunk start are records of this
    chunk under the rank rule) whose endpoint is a source of the pool."""

    def __init__(self, feeder, pool_ids, device):
        self.f = feeder
        mask = np.zeros(feeder.N, dtype=bool)
        mask[np.asarray(pool_ids, dtype=np.int64)] = True
        self.mask = mask
        self.pool_t = torch.tensor(np.asarray(pool_ids, dtype=np.int64),
                                   device=device)
        self.device = device
        self._cache = {}

    def get(self, c):
        if c in self._cache:
            return self._cache[c]
        f = self.f
        cst, cend = chunk_window(f, c)
        m = (f.t >= cst) & (f.t < cend)
        u, v, t = f.u[m], f.v[m], f.t[m]
        dts = np.concatenate([cend - t[self.mask[u]], cend - t[self.mask[v]]])
        dts = np.clip(dts, 0.0, None)
        out = (cend - cst, torch.tensor(dts, device=self.device,
                                        dtype=torch.float64))
        self._cache[c] = out
        return out


# --------------------------------------------------------------------------
# training / evaluation loops
# --------------------------------------------------------------------------
def chunk_loss(lg_pos, lg_neg):
    return (-F.logsigmoid(smooth_max_logits(lg_pos)).mean()
            - F.logsigmoid(-smooth_max_logits(lg_neg)).mean())


def train_epoch(model, feeder, bundle_train, ts_np, negs, optimizer,
                ground=None, wcache=None):
    """One pass over the train chunks: chunk t+1's positives
    (bundle_train['train_pos'][t]) and this epoch's negatives are scored at
    their raw timestamps against the state entering chunk t+1 plus the
    exact within-chunk part; one optimizer step per scored chunk."""
    T = bundle_train['T']
    terms, steps, skipped = 0.0, 0, 0
    optimizer.zero_grad(set_to_none=True)
    for t in range(T):
        pos, neg = bundle_train['train_pos'][t], negs[t]
        if pos.numel() == 0 or neg.numel() == 0:
            skipped += 1
            continue
        c = t + 1
        ts = np.asarray(ts_np[c], dtype=np.float64)
        pu = pos[0].cpu().numpy()
        pv = pos[1].cpu().numpy()
        nv = neg[1].cpu().numpy()
        assert ts.shape[0] == pu.shape[0], (c, ts.shape, pu.shape)
        ent = feeder.entering(c)
        states = model.entry_states(ent, feeder, c)
        xu = model.read(feeder, c, ent, states, pu, ts)
        lg_pos = model.score(feeder, c, ent, states, pu, pv, ts, xu=xu)
        lg_neg = model.score(feeder, c, ent, states, pu, nv, ts, xu=xu)
        term = chunk_loss(lg_pos, lg_neg)
        if ground is not None:
            # ground process by its likelihood over the chunk window:
            # -sum log lambda_u(t_i-) + closed-form compensator, per event
            Tw, dt_end = wcache.get(c)
            comp = ground.compensator(ent, wcache.pool_t, Tw, dt_end)
            term = term + (-ground.log_intensity(xu[2].detach()).sum()
                           + comp) / pu.shape[0]
        term.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        terms += float(term.detach())
        steps += 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()      # release per-chunk temporaries
    return terms / max(steps, 1), skipped, steps


def readout_term(term, ground, mlp, counts, feeder, nodes, tq):
    """The term b of the readout score f + beta * b for the scored sources
    `nodes` at times `tq`, from their decayed counts [nq, K].  'ground' is
    the default (log lambda_u); the others are the simpler terms of the ablation:
    the slowest / fastest scale's log(1 + count), minus log(1 + time since
    the source's last event), or a small classifier fitted on the training
    period (see scripts/evaluate_sfs.py)."""
    nq = int(len(nodes))
    if term == 'ground':
        if ground is None:
            return np.zeros(nq, dtype=np.float32)
        return ground.log_intensity(counts).cpu().numpy()
    if term == 'count_slow':       # lam is ordered fast -> slow
        return torch.log1p(counts[:, -1]).cpu().numpy()
    if term == 'count_fast':
        return torch.log1p(counts[:, 0]).cpu().numpy()
    if term == 'count_mid':        # the middle scale of the grid (the fastest one is degenerate on second-resolution streams)
        return torch.log1p(counts[:, counts.shape[1] // 2]).cpu().numpy()
    if term == 'recency':
        q_idx, _p, dt, _e = feeder.last_events(nodes, tq, 1)
        span = float(feeder.tsu[-1] - feeder.tsu[0])
        d = np.full(nq, span, dtype=np.float64)      # no event yet: the whole span
        d[np.asarray(q_idx, dtype=np.int64)] = np.asarray(dt, dtype=np.float64)
        return (-np.log1p(d)).astype(np.float32)
    if term == 'mlp':
        with torch.no_grad():
            return mlp(torch.log1p(counts)).squeeze(-1).float().cpu().numpy()
    raise ValueError('unknown readout term %r' % term)


@torch.no_grad()
def score_split(model, feeder, meta, period_idx, sampler, batch_size,
                first_chunk_global, ts_np, subset=None, ground=None,
                term='ground', mlp=None):
    """Scores of one evaluation split under the sampler's protocol, every
    read at the raw timestamp.  period_idx 1 = validation, 2 = test;
    subset = None scores the whole period (the transductive setting), a
    Data object holding a subset of its edges scores that subset (the
    inductive setting: the edges touching a new node).  Negatives are
    drawn batch by batch, in order, from the seeded reference sampler
    after a reset: one per positive, the source kept under the random
    strategy and both endpoints replaced under the historical / inductive
    strategies; every negative is scored at its positive's timestamp.

    Returns a dict: pos_f / neg_f (max over the two logits), pos_b / neg_b
    (log lambda_u of the scored source; zeros without a ground process),
    n (number of positives)."""
    data = meta['val_data'] if period_idx == 1 else meta['test_data']
    sizes = (meta['val_chunk_sizes'] if period_idx == 1
             else meta['test_chunk_sizes'])
    chunk_full = np.concatenate(
        [np.full(s_, i) for i, s_ in enumerate(sizes)]) + first_chunk_global
    if subset is None:
        src, dst = data.src_node_ids, data.dst_node_ids
        tq = np.asarray(data.node_interact_times, dtype=np.float64)
        chunk_global = chunk_full
    else:
        pos = np.searchsorted(data.edge_ids, subset.edge_ids)
        assert (data.edge_ids[pos] == subset.edge_ids).all(), \
            'subset edges not in the period'
        src, dst = subset.src_node_ids, subset.dst_node_ids
        tq = np.asarray(subset.node_interact_times, dtype=np.float64)
        chunk_global = chunk_full[pos]
    n = len(src)
    assert sampler.seed is not None
    sampler.reset_random_state()
    random = sampler.negative_sample_strategy == 'random'
    neg_src = np.empty(n, dtype=np.int64)
    neg_dst = np.empty(n, dtype=np.int64)
    for lo in range(0, n, batch_size):
        hi = min(lo + batch_size, n)
        if random:
            _, b = sampler.sample(size=hi - lo)
            neg_src[lo:hi] = src[lo:hi]
        else:
            a, b = sampler.sample(size=hi - lo, batch_src_node_ids=src[lo:hi],
                                  batch_dst_node_ids=dst[lo:hi],
                                  current_batch_start_time=tq[lo],
                                  current_batch_end_time=tq[hi - 1])
            neg_src[lo:hi] = a
        neg_dst[lo:hi] = b
    out = {k: np.zeros(n, dtype=np.float32)
           for k in ('pos_f', 'neg_f', 'pos_b', 'neg_b')}
    for c in np.unique(chunk_global):
        m_ = chunk_global == c
        ts = tq[m_]
        ent = feeder.entering(int(c))
        states = model.entry_states(ent, feeder, int(c))
        xu_p = model.read(feeder, int(c), ent, states, src[m_], ts)
        xu_n = xu_p if random else model.read(feeder, int(c), ent, states,
                                              neg_src[m_], ts)
        ps = model.score(feeder, int(c), ent, states, src[m_], dst[m_], ts,
                         xu=xu_p)
        ns = model.score(feeder, int(c), ent, states, neg_src[m_],
                         neg_dst[m_], ts, xu=xu_n)
        out['pos_f'][m_] = ps.max(dim=1).values.cpu().numpy()
        out['neg_f'][m_] = ns.max(dim=1).values.cpu().numpy()
        if ground is not None or term != 'ground':
            out['pos_b'][m_] = readout_term(term, ground, mlp, xu_p[2],
                                            feeder, src[m_], ts)
            out['neg_b'][m_] = (out['pos_b'][m_] if random else
                                readout_term(term, ground, mlp, xu_n[2],
                                             feeder, neg_src[m_], ts))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    out['n'] = n
    return out


def protocol_metrics(scores, beta, batch_size):
    """(AP, AUC) of a split's scores f + beta * log lambda_u, computed per
    batch of `batch_size` positives with their negatives and averaged
    over the batches (the reference metric)."""
    from .vendor.metrics import get_link_prediction_metrics
    pos = scores['pos_f'] + beta * scores['pos_b']
    neg = scores['neg_f'] + beta * scores['neg_b']
    n = scores['n']
    aps, aucs = [], []
    for lo in range(0, n, batch_size):
        hi = min(lo + batch_size, n)
        predicts = torch.tensor(np.concatenate([pos[lo:hi], neg[lo:hi]]))
        labels = torch.tensor(np.concatenate([np.ones(hi - lo),
                                              np.zeros(hi - lo)]))
        m = get_link_prediction_metrics(predicts=predicts, labels=labels)
        aps.append(m['average_precision'])
        aucs.append(m['roc_auc'])
    return float(np.mean(aps)), float(np.mean(aucs))


def eval_period(model, feeder, meta, period_idx, sampler, batch_size,
                first_chunk_global, ts_np):
    """The random-negative protocol on a whole period: (AP, AUC, batches)."""
    s = score_split(model, feeder, meta, period_idx, sampler, batch_size,
                    first_chunk_global, ts_np)
    ap, auc = protocol_metrics(s, 0.0, batch_size)
    return ap, auc, (s['n'] + batch_size - 1) // batch_size


def eval_period_hist(model, feeder, meta, period_idx, sampler, batch_size,
                     first_chunk_global, ts_np, ground=None):
    """A two-endpoint protocol (historical / inductive sampler) on a whole
    period: (AP, AUC, batches) of the bare readout f and, with a ground
    process, also (AP, AUC) of f + log lambda_u appended."""
    s = score_split(model, feeder, meta, period_idx, sampler, batch_size,
                    first_chunk_global, ts_np, ground=ground)
    bare = protocol_metrics(s, 0.0, batch_size)
    nb = (s['n'] + batch_size - 1) // batch_size
    if ground is None:
        return bare + (nb,)
    return bare + (nb,) + protocol_metrics(s, 1.0, batch_size)


# --------------------------------------------------------------------------
# brute-force oracle (verification)
# --------------------------------------------------------------------------
def oracle_state(feeder, E_np, node, t_query, second=False, m1_np=None):
    """Python-loop reference of the RAW first-order state and count of
    `node` strictly before t_query (rank rule), all scales.  If second, the
    raw second-order state with the mixed table m1_np frozen at the
    ENTERING value is not reproduced here (its fresh part uses the entry
    table by construction); callers verify first order + counts."""
    lam = feeder.lam_np
    K = lam.shape[0]
    d = E_np.shape[1]
    x = np.zeros((K, d))
    n = np.zeros(K)
    r_q = np.searchsorted(feeder.tsu, t_query, side='left')
    for e in range(feeder.E_n):
        if feeder.rank[e] >= r_q:
            continue
        if feeder.u[e] == node:
            p = feeder.v[e]
        elif feeder.v[e] == node:
            p = feeder.u[e]
        else:
            continue
        w = np.exp(-lam * (t_query - feeder.t[e]))
        x += w[:, None] * E_np[p][None, :]
        n += w
    return x, n


def oracle_pair_count(feeder, u, v, t_query):
    lam = feeder.lam_np
    n = np.zeros(lam.shape[0])
    r_q = np.searchsorted(feeder.tsu, t_query, side='left')
    a, b = min(u, v), max(u, v)
    for e in range(feeder.E_n):
        if feeder.rank[e] >= r_q:
            continue
        if min(feeder.u[e], feeder.v[e]) == a and max(feeder.u[e],
                                                      feeder.v[e]) == b:
            n += np.exp(-lam * (t_query - feeder.t[e]))
    return n
