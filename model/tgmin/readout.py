# -*- coding: utf-8 -*-
"""Scoring head and the smooth surrogate of the evaluation statistic.

The link-prediction score of a pair (u, v) is head([h_u ; h_v]) (plus the
scalar-channel terms added in tgmin/sfs.py), with h the node
representation.  The head has a direct linear path alongside a
sign-split nonlinear path:

    z_c(x) = W2 . [ReLU(s), ReLU(-s)] + U[c,:] . x + b[c],   s = W1 x + b1

Two properties motivate this shape:
  * the sign split [ReLU(s), ReLU(-s)] keeps both half-spaces of every
    pre-activation coordinate;
  * the linear path U x + b passes every class's evidence through no
    rectifier, so its gradient dz_c/dU[c,:] = x is nonzero whenever x is.

The head emits 2 logits (class 0 = "no edge", class 1 = "edge"); the
evaluation statistic is the max over the two logits, and the training loss
uses a smooth (logsumexp) surrogate of that max.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

TAU_LSE = 1.0    # logsumexp temperature of the smooth-max surrogate


def smooth_max_logits(logits):
    """Smooth surrogate of the evaluator's max-over-logits statistic."""
    return TAU_LSE * torch.logsumexp(logits / TAU_LSE, dim=1)


class CertifiedReadout(nn.Module):
    """z_c(x) = W2 . [ReLU(s), ReLU(-s)]  +  U[c,:] . x  +  b[c],
    with s = W1 x + b1.  See the module docstring."""

    def __init__(self, in_dim, hidden, num_classes):
        super().__init__()
        self.in_dim, self.hidden, self.num_classes = in_dim, hidden, num_classes
        self.pre = nn.Linear(in_dim, hidden)                       # W1, b1
        self.nl = nn.Linear(2 * hidden, num_classes, bias=False)   # W2
        self.cert = nn.Linear(in_dim, num_classes, bias=True)      # U, b

    def forward(self, x):
        s = self.pre(x)
        phi = torch.cat([F.relu(s), F.relu(-s)], dim=-1)
        return self.nl(phi) + self.cert(x)


def make_readout(d, num_classes=2, hidden=64):
    """Fresh scoring head: Xavier init on matrices, and an antipodal
    certificate init (the two classes' linear directions are exact
    negatives, bias zero) so both classes start symmetric."""
    head = CertifiedReadout(2 * d, hidden, num_classes)
    for p in head.parameters():
        if p.dim() >= 2:
            nn.init.xavier_uniform_(p)
    with torch.no_grad():
        head.cert.weight[1] = -head.cert.weight[0]
        head.cert.bias.zero_()
    return head
