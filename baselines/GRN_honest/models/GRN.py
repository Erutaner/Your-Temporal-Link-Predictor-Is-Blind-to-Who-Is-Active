# -*- coding: utf-8 -*-
"""GRN (Graph Retention Network) with a strict streaming-state protocol.

The layer parameters, the layer stack, the retention arithmetic, the
node-feature / state updates and the default configuration are those of the
released model.  What changes is *when* information becomes visible:

  1. strict-before visibility: a query at time t reads the state committed
     from events strictly earlier than the chunk's first timestamp and, inside
     the chunk, only positive events with the same destination and a strictly
     earlier timestamp.  (The released code let each row see every earlier row
     of its own call by row order, same-timestamp rows included, and committed
     the chunk's positives before the negative call of the same chunk.)
  2. no query-edge features: the scoring pass never reads the features of the
     edge being predicted.  (The released code fed the positive edge's features
     to the positive and the negative pair.)  The features enter the update of
     the positive event only.
  3. negatives are read-only: they write neither node features nor
     last-interaction times.  (The released code wrote the times of negatives.)
  4. real backups: backup_state / reset_state copy the state.  (The released
     backup returned references to tensors that were then mutated in place, so
     restoring did nothing and the final evaluation ran on a state that had
     already consumed the validation, and on test epochs the test, events.)
  5. the state is not part of the checkpoint; evaluation rebuilds it by
     streaming the training events through the selected weights.
"""
import math

import numpy as np
import torch
import torch.nn as nn

from models.GraphRetention import GraphRetention, TemporalEncoding
from utils.utils import NeighborSampler


class GRN(nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, channel_embedding_dim: int, patch_size: int = 1, num_layers: int = 2, num_heads: int = 2,
                 dropout: float = 0.1, max_input_sequence_length: int = 512, device: str = 'cpu'):
        super(GRN, self).__init__()
        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        assert channel_embedding_dim % num_heads == 0
        self.head_size = channel_embedding_dim // num_heads
        self.channel_embedding_dim = channel_embedding_dim
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.time_encoder = None
        self.device = device
        self.neighbor_sampler = neighbor_sampler
        # streaming state: not part of the checkpoint (rebuilt by streaming the events)
        self.register_buffer('node_features', None, persistent=False)
        self.register_buffer('state', None, persistent=False)
        self.register_buffer('time', None, persistent=False)
        self.edge_features = self.edge_raw_features
        self.pending = {}
        self._ctx = None
        self.setup_net()

    def setup_net(self):
        self.input_layer = nn.Linear(in_features=self.node_feat_dim, out_features=self.channel_embedding_dim)
        self.time_encoder = TemporalEncoding(out_channels=self.channel_embedding_dim)
        self.grns = nn.ModuleDict({
            f'grn{i}': GraphRetention(in_channels=self.channel_embedding_dim, out_channels=self.head_size, heads=self.num_heads,
                                      concat=True, dropout=self.dropout, edge_dim=self.edge_feat_dim, time_encoder=None)
            for i in range(self.num_layers)})
        if self.time_encoder is not None:
            self.grns['grn0'].time_encoder = self.time_encoder
        self.ffns = nn.ModuleDict({
            f'ffn{i}': nn.Sequential(nn.Linear(self.channel_embedding_dim, self.channel_embedding_dim * 2),
                                     nn.Dropout(self.dropout),
                                     HSwish(),
                                     nn.Linear(self.channel_embedding_dim * 2, self.channel_embedding_dim),
                                     nn.Dropout(self.dropout))
            for i in range(self.num_layers)})
        self.LayerNorm1 = nn.ModuleDict({f'grn_LayerNorm{i}': nn.LayerNorm(self.channel_embedding_dim) for i in range(self.num_layers)})
        self.LayerNorm2 = nn.ModuleDict({f'ffn_LayerNorm{i}': nn.LayerNorm(self.channel_embedding_dim) for i in range(self.num_layers)})
        self.LayerNorm3 = nn.LayerNorm(self.channel_embedding_dim)
        self.output_layer = nn.Linear(in_features=self.channel_embedding_dim, out_features=self.node_feat_dim, bias=True)
        self.reset_state()

    # ------------------------------------------------------------------ streaming state
    def reset_state(self, inputs=None):
        """Fresh state, or restore a backup made by backup_state (by value)."""
        if inputs is not None:
            state, time, node_features, pending = inputs
            self.state = state.to(self.device).clone()
            self.time = time.to(self.device).clone()
            self.node_features = node_features.to(self.device).clone()
            self.pending = {node: [dict(e) for e in entries] for node, entries in pending.items()}
            self._ctx = None
            return
        num_nodes = self.node_raw_features.shape[0]
        self.state = torch.zeros((self.num_layers, num_nodes, self.num_heads, self.head_size, self.head_size),
                                 dtype=torch.float32, device=self.device)
        self.time = torch.empty(num_nodes, dtype=torch.float64, device=self.device).fill_(math.pi / 2)
        self.node_features = self.node_raw_features.clone()
        self.pending = {}
        self._ctx = None

    def backup_state(self):
        """A copy of the streaming state (kept on the CPU)."""
        return (self.state.detach().cpu().clone(), self.time.detach().cpu().clone(), self.node_features.detach().cpu().clone(),
                {node: [dict(e) for e in entries] for node, entries in self.pending.items()})

    def flush_before(self, query_time: float):
        """Commit the pending positive events with timestamp < query_time: node features (last write of the node),
        retention states and last-interaction times (last event with the node as destination)."""
        if not self.pending:
            return
        feat_idx, feat_val, st_idx, st_val, tm_val = [], [], [], [], []
        for node in list(self.pending.keys()):
            entries = self.pending[node]
            done = [e for e in entries if e['t'] < query_time]
            if not done:
                continue
            feat_idx.append(node)
            feat_val.append(done[-1]['emb'])
            with_state = [e for e in done if e['s'] is not None]
            if with_state:
                st_idx.append(node)
                st_val.append(with_state[-1]['s'])
                tm_val.append(with_state[-1]['t'])
            keep = entries[len(done):]
            if keep:
                self.pending[node] = keep
            else:
                del self.pending[node]
        if feat_idx:
            self.node_features[torch.tensor(feat_idx, device=self.device)] = torch.stack(feat_val)
        if st_idx:
            idx = torch.tensor(st_idx, device=self.device)
            self.state[:, idx] = torch.stack(st_val, dim=1)
            self.time[idx] = torch.tensor(tm_val, dtype=torch.float64, device=self.device)

    def flush_all(self):
        self.flush_before(float('inf'))

    def _put(self, node: int, t: float, emb, s):
        entries = self.pending.get(node)
        if entries is not None and entries[-1]['t'] == t:
            entries[-1]['emb'] = emb
            if s is not None:
                entries[-1]['s'] = s
            return
        if entries is None:
            entries = self.pending[node] = []
        entries.append({'t': t, 'emb': emb, 's': s})

    def _submit(self, times: np.ndarray, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, states, src_emb, dst_emb):
        """Queue the updates of a chunk of positive events (released write order: sources' features, then the
        destinations' features and states, later rows overriding earlier ones)."""
        for i in range(len(src_node_ids)):
            self._put(int(src_node_ids[i]), float(times[i]), src_emb[i], None)
        for i in range(len(dst_node_ids)):
            self._put(int(dst_node_ids[i]), float(times[i]), dst_emb[i], states[:, i])

    # ------------------------------------------------------------------ forward
    def _intervals(self, t64, dst_ids, ctx_t64, ctx_dst):
        """Time intervals of the destinations and the visibility of the context rows."""
        last = self.time[dst_ids]
        if ctx_t64 is None:
            return t64 - last, None
        vis = (ctx_dst.unsqueeze(0) == dst_ids.unsqueeze(1)) & (ctx_t64.unsqueeze(0) < t64.unsqueeze(1))
        ctx_rep = ctx_t64.unsqueeze(0).expand(vis.shape)
        prev = torch.where(vis, ctx_rep, torch.full_like(ctx_rep, float('-inf'))).max(dim=1).values
        prev = torch.where(vis.any(dim=1), prev, last)
        return t64 - prev, vis

    def _run(self, src_ids, dst_ids, dt, edge_attr, ctx_layers, vis):
        """The released layer stack; returns src / dst embeddings, the per-layer states of the rows and the per-layer
        context entries of the rows."""
        n = src_ids.shape[0]
        x = torch.cat([self.node_features[src_ids], self.node_features[dst_ids]], dim=0)
        x = self.input_layer(x)
        dt = dt.to(torch.float32)
        states, entries = [], []
        for i in range(self.num_layers):
            s = self.state[i][dst_ids]
            y = self.LayerNorm1[f'grn_LayerNorm{i}'](x)
            src, dst = y[:n], y[n:]
            src, dst, s, entry = self.grns[f'grn{i}'].strict_forward(
                src=src, dst=dst, state=s, edge_attr=edge_attr, time=dt, vis=vis,
                ctx=None if ctx_layers is None else ctx_layers[i])
            states.append(s)
            entries.append(entry)
            y = torch.cat([src, dst], dim=0)
            y = y + x
            x = self.LayerNorm2[f'ffn_LayerNorm{i}'](y)
            x = self.ffns[f'ffn{i}'](y)
            x = x + y
        x = self.LayerNorm3(x)
        x = self.output_layer(x)
        return x[:n], x[n:], torch.stack(states), entries

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                                                 node_interact_times: np.ndarray, batch_edge_ids: np.ndarray,
                                                 edges_are_positive: bool = True, need_scores: bool = True):
        """
        Positive call (edges_are_positive=True, first for every chunk): commits the pending events strictly before the
        chunk, runs the update pass over the chunk's positive events (with their edge features; each row sees the
        strictly earlier same-destination rows), queues their updates, and scores the pairs.  Negative call: scores
        the pairs against the same committed state and the same chunk context.  Scoring never reads the query
        edge's features and never writes.
        """
        t64 = torch.from_numpy(node_interact_times.astype(np.float64)).to(self.device)
        src_ids = torch.from_numpy(src_node_ids.astype(np.int64)).to(self.device)
        dst_ids = torch.from_numpy(dst_node_ids.astype(np.int64)).to(self.device)
        self.flush_before(float(node_interact_times[0]))
        if edges_are_positive:
            dt, vis = self._intervals(t64, dst_ids, t64, dst_ids)
            edge_attr = self.edge_features[torch.from_numpy(batch_edge_ids.astype(np.int64)).to(self.device)]
            src_emb_u, dst_emb_u, states, entries = self._run(src_ids, dst_ids, dt, edge_attr, None, vis)
            self._ctx = {'t': t64, 'dst': dst_ids, 'layers': entries}
            self._submit(node_interact_times, src_node_ids, dst_node_ids, states.detach(), src_emb_u.detach(), dst_emb_u.detach())
            if not need_scores:
                return None, None
        ctx = self._ctx
        assert ctx is not None and ctx['t'].shape[0] == t64.shape[0] and bool(torch.equal(ctx['t'], t64)), \
            'the positive call of a chunk must precede its negative call'
        dt, vis = self._intervals(t64, dst_ids, ctx['t'], ctx['dst'])
        src_node_embeddings, dst_node_embeddings, _, _ = self._run(src_ids, dst_ids, dt, None, ctx['layers'], vis)
        return src_node_embeddings, dst_node_embeddings

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()


class HSwish(nn.Module):

    def __init__(self):
        super(HSwish, self).__init__()
        self.relu6 = nn.ReLU6()

    def forward(self, x):
        return x * self.relu6(x + 3) / 6
