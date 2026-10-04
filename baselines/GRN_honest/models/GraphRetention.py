# -*- coding: utf-8 -*-
"""Graph retention layer of GRN under the strict-before rule.

The parameters of the released layer are unchanged (lin_key / lin_query /
lin_value, the two GroupNorms, the two edge projections, the head decays
`gammas`, the optional time encoder).  `strict_forward` is the released
chunkwise paradigm with the visibility rule needed for a strict evaluation:

  * inside a chunk a query at time t attends only to positive events with the
    same destination whose timestamp is strictly earlier than t (the released
    code masked by row order, so same-timestamp rows were visible), and the
    same context serves positive and negative queries alike;
  * the retention state read for a query is the state committed before the
    chunk (see GRN.flush_before).

The retention arithmetic (decayed key-value state, self term, causal decay
matrix, its two normalisations, the output GroupNorms) is the released one.
"""
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Linear

OptTensor = Optional[Tensor]


class GraphRetention(nn.Module):

    def __init__(self, in_channels: Union[int, Tuple[int, int]], out_channels: int, heads: int = 1, concat: bool = False,
                 time_encoder: Optional[callable] = None, dropout: float = 0., edge_dim: Optional[int] = None,
                 bias: bool = True, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.register_buffer('gammas', (1 - torch.exp(torch.linspace(math.log(1 / 64), math.log(1 / 1024), heads))).detach())
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.lin_key = Linear(in_channels[0], heads * out_channels)
        self.lin_query = Linear(in_channels[1], heads * out_channels)
        self.lin_value = Linear(in_channels[0], heads * out_channels)
        self.norm_key = nn.GroupNorm(self.heads, self.heads)
        self.norm_out = nn.GroupNorm(self.heads, self.heads)
        self.time_encoder = time_encoder
        if edge_dim is not None:
            self.lin_edge_src = Linear(edge_dim, heads * out_channels, bias=True)
            self.lin_edge_dst = Linear(edge_dim, heads * out_channels, bias=True)
        else:
            self.register_parameter('lin_edge', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        if self.edge_dim is not None:
            self.lin_edge_src.reset_parameters()
            self.lin_edge_dst.reset_parameters()

    def message(self, src: Tensor, dst: Tensor, edge_attr: OptTensor, time: OptTensor):
        if edge_attr is not None:
            assert self.lin_edge_src is not None
            src = src + self.lin_edge_src(edge_attr)
            dst = dst + self.lin_edge_dst(edge_attr)
        if self.time_encoder is not None:
            assert time is not None
            src = src + self.time_encoder(time)
        return src, dst

    def strict_forward(self, src: Tensor, dst: Tensor, state: Tensor, edge_attr: OptTensor, time: Tensor,
                       vis: OptTensor, ctx: Optional[dict]):
        """
        :param src: [n, in], layer inputs of the query rows' sources
        :param dst: [n, in], layer inputs of the query rows' destinations
        :param state: [n, H, C, C], retention state of the destinations as committed before the chunk
        :param edge_attr: [n, edge_dim] or None (None while scoring: the query edge's features are not known at query time)
        :param time: [n], time intervals of the destinations
        :param vis: [n, m] bool or None; vis[i, j] = context row j is a strictly earlier positive event with the same
                    destination as query row i
        :param ctx: {'value': [m, H, C], 'alpha': [m, H]} of the context rows at this layer; None = the query rows
                    themselves are the context (update pass over a chunk of positive events)
        :return: src_out [n, H*C], dst_out [n, H*C], new_state [n, H, C, C], entry (value / alpha of the query rows)
        """
        H, C = self.heads, self.out_channels
        src, dst = self.message(src=src, dst=dst, edge_attr=edge_attr, time=time)
        query = self.lin_query(dst).view(-1, H, C)
        key = self.lin_key(src).view(-1, H, C)
        value = self.lin_value(src).view(-1, H, C)
        n = query.shape[0]
        # retention state after the row's own event (released chunkwise paradigm)
        state_ = self.gammas.view(1, H, 1, 1) * state + \
            (key.view(-1, H, 1, C).transpose(-1, -2) @ value.view(-1, H, 1, C)) / math.sqrt(C)
        alpha = (query * key).sum(dim=-1) / math.sqrt(C)                    # [n, H]
        entry = {'value': value, 'alpha': alpha}
        if ctx is None:
            ctx = entry
        if vis is not None and bool(vis.any()):
            m = vis.shape[1]
            visf = vis.to(query.dtype)
            # decay exponent of context row j for query row i: 1 + (number of visible rows after j), i.e. the released
            # occ_i - occ_j counted over visible rows only
            cum = visf.cumsum(dim=1)
            expo = 1.0 + (cum[:, -1:] - cum)                                  # [n, m]
            D = (self.gammas.view(H, 1, 1) ** expo.unsqueeze(0)) * visf.unsqueeze(0)          # [H, n, m]
            D = torch.cat([D, torch.ones(H, n, 1, device=query.device, dtype=query.dtype)], dim=-1)   # + the row itself
            D = D / D.sum(dim=-1, keepdim=True)
            a = torch.cat([ctx['alpha'].transpose(0, 1).unsqueeze(1).expand(H, n, m),
                           alpha.transpose(0, 1).unsqueeze(-1)], dim=-1)      # [H, n, m + 1]
            D = D * a
            D = D / D.detach().sum(dim=-1, keepdim=True).abs().clamp(min=1)
            out = torch.einsum('hnm,mhc->nhc', D[..., :m], ctx['value']) + \
                D[..., m].transpose(0, 1).unsqueeze(-1) * value
        else:
            w = alpha / alpha.detach().abs().clamp(min=1)
            out = w.unsqueeze(-1) * value
        out = out + (query.view(-1, H, 1, C) @ state_).squeeze(-2) * self.gammas.view(1, H, 1)
        out = self.norm_out(out)
        key = self.norm_key(key)
        if self.concat:
            return key.reshape(n, H * C), out.reshape(n, H * C), state_, entry
        return key.mean(dim=1), out.mean(dim=1), state_, entry

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


'''The implementation of PositionalEncoding and TemporalEncoding comes from
the torch_geometric team at https://pytorch-geometric.readthedocs.io/en/latest/.'''


class PositionalEncoding(torch.nn.Module):

    def __init__(self, out_channels: int, base_freq: float = 1e-4, granularity: float = 1.0):
        super().__init__()
        if out_channels % 2 != 0:
            raise ValueError(f"Cannot use sinusoidal positional encoding with odd 'out_channels' (got {out_channels}).")
        self.out_channels = out_channels
        self.base_freq = base_freq
        self.granularity = granularity
        frequency = torch.logspace(0, 1, out_channels // 2, base_freq)
        self.register_buffer('frequency', frequency)

    def forward(self, x: Tensor) -> Tensor:
        x = x / self.granularity if self.granularity != 1.0 else x
        out = x.view(-1, 1) * self.frequency.view(1, -1)
        return torch.cat([torch.sin(out), torch.cos(out)], dim=-1)


class TemporalEncoding(torch.nn.Module):

    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels
        sqrt = math.sqrt(out_channels)
        weight = 1.0 / sqrt**torch.linspace(0, sqrt, out_channels).view(1, -1)
        self.register_buffer('weight', weight)

    def forward(self, x: Tensor) -> Tensor:
        return torch.cos(x.view(-1, 1) @ self.weight)
