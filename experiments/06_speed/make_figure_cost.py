# -*- coding: utf-8 -*-
"""Appendix cost figure (S3), three panels in a row, one shared legend, every panel relative so the five datasets
share one axis:
  (a) training time per epoch versus the number of time scales K, relative to the main configuration's K;
  (b) the overhead of within-chunk records: epoch time with records on divided by records off, versus T;
  (c) peak GPU memory of the scoring pass versus the number of training chunks T, relative to T = 48.
Data: results/speed/results_s23.jsonl (one exclusive RTX 5090, median of 3 repeats).
Writes outputs/fig_cost_s3.pdf / .png.
"""
import collections
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common import RESULTS, OUTPUTS  # noqa: E402

RES = RESULTS
OUT = OUTPUTS
DATASETS = [('wikipedia', 'Wikipedia'), ('reddit', 'Reddit'), ('uci', 'UCI'), ('enron', 'Enron'), ('lastfm', 'LastFM')]
COLOR = {'wikipedia': '#0072B2', 'reddit': '#D55E00', 'uci': '#009E73', 'enron': '#E69F00', 'lastfm': '#CC79A7'}
MARK = {'wikipedia': 'o', 'reddit': 'D', 'uci': 's', 'enron': '^', 'lastfm': 'v'}
KVAR = ['K1', 'pd1', 'pd2', 'pd4']            # pd2 = the main configuration
TS = [48, 96, 192]


def load():
    recs = [json.loads(l) for l in open(os.path.join(RES, 'speed', 'results_s23.jsonl'), encoding='utf-8') if l.strip()]
    by = collections.defaultdict(list)
    K = {}
    for r in recs:
        if r['exp'] != 'S3' or r['measure'] is None:
            continue
        by[(r['dataset'], r['variant'], r['phase'])].append(r['measure'])
        if r['phase'] == 'train' and r.get('K'):
            K[(r['dataset'], r['variant'])] = r['K']
    med = lambda ds, v, ph, key: float(np.median([m[key] for m in by[(ds, v, ph)]]))
    return med, K


def style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=8.5)
    ax.set_ylabel(ylabel, fontsize=8.5)
    ax.set_title(title, fontsize=9, pad=5)
    ax.tick_params(labelsize=7.5, length=2.5)
    ax.grid(True, which='major', color='#E6E6E6', linewidth=0.6)
    ax.set_axisbelow(True)
    for sp_ in ('top', 'right'):
        ax.spines[sp_].set_visible(False)
    for sp_ in ('left', 'bottom'):
        ax.spines[sp_].set_color('#888888')


def line(ax, ds, x, y, label=None):
    ax.plot(x, y, color=COLOR[ds], marker=MARK[ds], markersize=4, markerfacecolor='white', markeredgewidth=1.1,
            linewidth=1.4, label=label)


def main():
    med, K = load()
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.45))
    # (a) epoch time vs the four scale grids (categorical): one point per dataset, black line through the means
    ax = axes[0]
    offs = np.linspace(-0.22, 0.22, len(DATASETS))
    means = []
    for vi, v in enumerate(KVAR):
        vals = []
        for (ds, name), off in zip(DATASETS, offs):
            ts = [med(ds, vv, 'train', 'train_seconds') for vv in KVAR]
            y = ts[vi] / ts[2]
            vals.append(y)
            ax.plot([vi + off], [y], color=COLOR[ds], marker=MARK[ds], markersize=4.5, markerfacecolor='white',
                    markeredgewidth=1.2, linestyle='none', label=name if vi == 0 else None, zorder=4)
        means.append(np.mean(vals))
    mean_line, = ax.plot(range(len(KVAR)), means, color='#222222', linewidth=1.6, marker='_', markersize=14, markeredgewidth=1.8, zorder=3)
    ax.legend([mean_line], ['mean of five'], fontsize=6.8, frameon=False, loc='upper left', handlelength=1.6, borderaxespad=0.2)
    ax.axhline(1.0, color='#BBBBBB', linewidth=0.8, zorder=0)
    ax.set_xticks(range(len(KVAR)))
    ax.set_xticklabels(['1', '8-11', '14-16', '27-32'])
    ax.set_xlim(-0.6, len(KVAR) - 0.4)
    ax.set_ylim(0.5, 1.7)
    style(ax, 'number of timescales $K$ (main: 14-16)', 'relative time per epoch', '(a) training time vs. $K$')
    # (b) overhead of within-chunk records vs T
    ax = axes[1]
    for ds, name in DATASETS:
        r = [med(ds, 'fr1_t%d' % t, 'train', 'train_seconds') / med(ds, 'fr0_t%d' % t, 'train', 'train_seconds') for t in TS]
        line(ax, ds, TS, r)
    ax.axhline(1.0, color='#BBBBBB', linewidth=0.8, zorder=0)
    ax.set_xscale('log', base=2)
    ax.set_xticks(TS)
    ax.set_xticklabels([str(t) for t in TS])
    ax.minorticks_off()
    ax.set_ylim(0.9, 3.3)
    style(ax, 'number of training blocks $C$', 'time per epoch, on / off', '(b) within-block update cost vs. $C$')
    # (c) scoring memory vs T, relative to T = 48
    ax = axes[2]
    for ds, name in DATASETS:
        ms = [med(ds, 'fr1_t%d' % t, 'eval', 'peak_mem_mb') for t in TS]
        line(ax, ds, TS, [m / ms[0] for m in ms])
    ax.axhline(1.0, color='#BBBBBB', linewidth=0.8, zorder=0)
    ax.set_xscale('log', base=2)
    ax.set_xticks(TS)
    ax.set_xticklabels([str(t) for t in TS])
    ax.minorticks_off()
    ax.set_ylim(0.2, 1.1)
    style(ax, 'number of training blocks $C$', 'relative peak memory', '(c) inference memory vs. $C$')
    handles, labels = axes[0].get_legend_handles_labels()
    handles, labels = [h for h, l in zip(handles, labels) if l != 'mean of five'], [l for l in labels if l != 'mean of five']
    fig.legend(handles, labels, loc='upper center', ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.02),
               handlelength=1.8, columnspacing=1.6)
    fig.tight_layout(pad=0.4, w_pad=1.8, rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(OUT, 'fig_cost_s3.pdf'))
    fig.savefig(os.path.join(OUT, 'fig_cost_s3.png'), dpi=200)
    for ds, name in DATASETS:
        ks = [K[(ds, v)] for v in KVAR]
        ts = [med(ds, v, 'train', 'train_seconds') for v in KVAR]
        ov = [med(ds, 'fr1_t%d' % t, 'train', 'train_seconds') / med(ds, 'fr0_t%d' % t, 'train', 'train_seconds') for t in TS]
        ms = [med(ds, 'fr1_t%d' % t, 'eval', 'peak_mem_mb') for t in TS]
        print('%-10s K=%s t=%s | overhead=%s | mem MB=%s' % (name, ks, [round(t, 1) for t in ts], [round(o, 2) for o in ov], [round(m) for m in ms]))


if __name__ == '__main__':
    main()
