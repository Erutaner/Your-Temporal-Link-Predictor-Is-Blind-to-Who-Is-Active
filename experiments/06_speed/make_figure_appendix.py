# -*- coding: utf-8 -*-
"""Appendix speed figure: one full page, five datasets (UCI, Wikipedia, Enron, Reddit, Flights) x two columns
(training time per epoch | test-split scoring time), same conventions as the main-text LastFM figure:
x = time relative to ours (log), y = transductive test AP under historical negative sampling.
Times: results/speed/results.jsonl (all-model benchmark). AP: main-table sources (TPNet paper for the paper baselines, our
DyGFormer fixed-padding retrain where it exists, our DSRD retrain, ours = f + beta log lambda).
Writes outputs/fig_speed_appendix.pdf / .png.
"""
import glob
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
DATASETS = [('Flights', 'Flights', 'Flights'), ('wikipedia', 'wikipedia', 'Wikipedia'), ('enron', 'enron', 'Enron'),
            ('reddit', 'reddit', 'Reddit'), ('uci', 'uci', 'UCI')]      # (key, key in the timing records, paper name)
MODELS = ['JODIE', 'DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer', 'EdgeBank', 'TPNet', 'DSRD', 'Ours']
STYLE = {
    'JODIE': ('o', '#0072B2'), 'DyRep': ('s', '#E69F00'), 'TGAT': ('^', '#009E73'), 'TGN': ('v', '#D55E00'),
    'CAWN': ('p', '#56B4E9'), 'TCL': ('D', '#CC79A7'), 'GraphMixer': ('*', '#8C564B'), 'DyGFormer': ('h', '#7F7F7F'),
    'EdgeBank': ('X', '#BCBD22'), 'TPNet': ('x', '#B8B000'), 'DSRD': ('P', '#17BECF'), 'Ours': ('*', '#C00000'),
}
DEFAULT_OFF = (7, 0)
# hand placement per (dataset, panel): model -> (dx, dy) points; missing = right of the marker
OFF = {
    ('uci', 'train'): {'Ours': (8, 3), 'DSRD': (8, 6), 'TPNet': (-8, 0), 'GraphMixer': (8, 0), 'DyGFormer': (8, -4), 'TCL': (0, -12),
                       'TGN': (-8, 2), 'JODIE': (8, -6), 'TGAT': (8, 0), 'CAWN': (0, -12), 'DyRep': (8, 3), 'TCL': (4, -12)},
    ('uci', 'eval'): {'Ours': (8, 3), 'DSRD': (-8, 6), 'TPNet': (8, -3), 'TGN': (-8, 3), 'TCL': (8, -3), 'DyGFormer': (8, -4),
                      'GraphMixer': (8, 3), 'JODIE': (8, -4), 'EdgeBank': (8, 0), 'TGAT': (8, 0), 'CAWN': (0, -12), 'DyRep': (8, 3)},
    ('wikipedia', 'train'): {'Ours': (8, 3), 'GraphMixer': (8, 3), 'DSRD': (-8, -4), 'TCL': (8, -1), 'TGN': (-8, -2), 'TGAT': (8, 4),
                             'JODIE': (8, 2), 'TPNet': (8, -3), 'DyRep': (0, -12), 'DyGFormer': (8, 0), 'CAWN': (8, 0)},
    ('wikipedia', 'eval'): {'Ours': (8, 3), 'GraphMixer': (8, 0), 'DSRD': (8, -3), 'TCL': (-8, -2), 'TGN': (8, 0), 'TGAT': (8, 0),
                            'JODIE': (8, 2), 'TPNet': (-8, -2), 'DyRep': (8, -3), 'DyGFormer': (8, 0), 'EdgeBank': (8, -2), 'CAWN': (8, 0)},
    ('enron', 'train'): {'Ours': (8, 3), 'DSRD': (8, 0), 'TPNet': (-8, 0), 'GraphMixer': (8, 0), 'DyGFormer': (8, 0), 'TGN': (8, 0),
                         'DyRep': (-8, 2), 'JODIE': (0, -10), 'TCL': (8, 0), 'TGAT': (-8, -2), 'CAWN': (8, 2), 'GraphMixer': (8, 3)},
    ('enron', 'eval'): {'Ours': (8, 3), 'DSRD': (-8, 0), 'TPNet': (-8, 0), 'TGN': (-8, 2), 'DyRep': (0, -11), 'TCL': (8, 0),
                        'JODIE': (-8, 0), 'EdgeBank': (6, -11), 'DyGFormer': (8, 0), 'GraphMixer': (8, 3), 'TGAT': (-8, -2), 'CAWN': (8, 2)},
    ('reddit', 'train'): {'Ours': (8, 3), 'DSRD': (8, 0), 'JODIE': ('abs', 1.9, 78.0, 'right'), 'DyRep': ('abs', 3.2, 75.8, 'right'), 'TGN': (-8, 4), 'TPNet': (0, 14),
                          'DyGFormer': (6, 11), 'GraphMixer': ('abs', 17, 75.5, 'left'), 'TCL': (0, -11), 'CAWN': (8, 4), 'TGAT': (8, -3)},
    ('reddit', 'eval'): {'Ours': (8, 3), 'DSRD': (8, 0), 'TPNet': (-8, 0), 'JODIE': ('abs', 1.4, 78.2, 'right'), 'TCL': (0, -11), 'DyRep': (8, 0),
                         'TGN': (0, 11), 'DyGFormer': (4, 10), 'GraphMixer': (0, -11), 'EdgeBank': (8, 0), 'TGAT': (8, -4), 'CAWN': (8, 2)},
    ('Flights', 'train'): {'Ours': (8, 3), 'JODIE': (-8, 0), 'DyRep': ('abs', 9, 69.3, 'right'), 'DSRD': ('abs', 9, 72.2, 'right'),
                           'TCL': ('abs', 24, 78.5, 'center'), 'GraphMixer': ('abs', 55, 76.5, 'left'), 'TGAT': (8, 0),
                           'DyGFormer': (8, -2), 'CAWN': (0, -11), 'TGN': (0, -11), 'TPNet': ('abs', 32, 68.3, 'left')},
    ('Flights', 'eval'): {'Ours': (8, 3), 'DSRD': ('abs', 4.5, 74.5, 'right'), 'TCL': (8, -2),
                          'TPNet': ('abs', 4.5, 69.3, 'right'), 'JODIE': ('abs', 5, 64.5, 'right'), 'TGN': ('abs', 17, 64.2, 'left'), 'DyRep': ('abs', 27, 69.0, 'left'),
                          'GraphMixer': (0, 10), 'TGAT': (8, 0), 'EdgeBank': (8, 0), 'DyGFormer': (8, -2), 'CAWN': (0, -11)},
}
LEADER_MIN = 10          # offsets at least this long get a leader line


def load_speed():
    """(scoring, training): model -> {stream: seconds}, medians of three runs."""
    from common import speed_medians
    return speed_medians()


def load_ap():
    paper = json.load(open(os.path.join(RES, 'main', 'tpnet_paper_all_baselines.json')))['trans_hist']['AP']
    summ = json.load(open(os.path.join(RES, 'baselines', 'summary.json')))
    gate = json.load(open(os.path.join(RES, 'baselines', 'tpnet_gate.json')))['leak']
    ap = {}
    for key, _, pname in DATASETS:
        d = {m: paper[pname][m][0] for m in MODELS if m in paper[pname]}
        if key in summ['DyGFormer_fixpad']:
            d['DyGFormer'] = summ['DyGFormer_fixpad'][key]['trans_ap/hist']['mean']
        if gate.get(key):
            d['TPNet'] = summ['TPNet_strict'][key]['trans_ap/hist']['mean']
        d['DSRD'] = summ['DSRD_honest'][key]['trans_ap/hist']['mean']
        evs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, 'main', key, '%s_final_run*_eval.json' % key)))]
        d['Ours'] = 100 * float(np.mean([e['selected']['transductive']['historical']['test_ap'] for e in evs]))
        ap[key] = d
    return ap


def panel(ax, key, col, times, ap, which, title, ylabel, xlabel):
    ours = times['Ours'][col]
    off = OFF.get((key, which), {})
    for m in MODELS:
        t = times.get(m, {}).get(col)
        if t is None or m not in ap:
            continue
        mk, colr = STYLE[m]
        rel = t / ours
        ax.scatter([rel], [ap[m]], marker=mk, s=150 if m == 'Ours' else 55, c=colr, zorder=3)
        label = 'Ours' if m == 'Ours' else ('%s $\\times$%.0f' % (m, rel) if rel >= 10 else '%s $\\times$%.1f' % (m, rel))
        o = off.get(m, DEFAULT_OFF)
        arrow_style = dict(arrowstyle='-', color='#8A8A8A', lw=0.5, shrinkA=1, shrinkB=4.5)
        if o[0] == 'abs':
            ax.annotate(label, (rel, ap[m]), xytext=(o[1], o[2]), textcoords='data', ha=o[3], va='center', fontsize=7,
                        color='#222222', zorder=4, arrowprops=arrow_style)
            continue
        dx, dy = o
        ha = 'left' if dx > 0 else ('right' if dx < 0 else 'center')
        va = 'center' if dx != 0 else ('bottom' if dy > 0 else 'top')
        arrow = arrow_style if (dx * dx + dy * dy) ** 0.5 >= LEADER_MIN else None
        ax.annotate(label, (rel, ap[m]), xytext=(dx, dy), textcoords='offset points', ha=ha, va=va, fontsize=7,
                    color='#222222', fontweight='bold' if m == 'Ours' else 'normal', zorder=4, arrowprops=arrow)
    ax.set_xscale('log')
    ax.set_title(title, fontsize=9.5)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8.5)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8.5)
    ax.grid(True, which='major', color='#E3E3E3', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp_ in ('top', 'right'):
        ax.spines[sp_].set_visible(False)
    ax.tick_params(labelsize=7.5)


# per-dataset axis limits (x: relative time; y: AP)
LIMS = {
    'uci': ((0.08, 30), (50, 100)), 'wikipedia': ((0.25, 120), (68.5, 96)), 'enron': ((0.03, 80), (58, 95)),
    'reddit': ((0.25, 300), (70, 94)), 'Flights': ((0.5, 2500), (60, 88)),
}


def main():
    ev, tr = load_speed()
    ap = load_ap()
    fig, axes = plt.subplots(len(DATASETS), 2, figsize=(7.4, 10.6))
    for i, (key, col, pname) in enumerate(DATASETS):
        last = i == len(DATASETS) - 1
        panel(axes[i, 0], key, col, tr, ap[key], 'train', '%s: training time per epoch' % pname,
              'AP, historical negatives (%)', 'time relative to ours (log scale)' if last else None)
        panel(axes[i, 1], key, col, ev, ap[key], 'eval', '%s: test-split scoring time' % pname,
              None, 'time relative to ours (log scale)' if last else None)
        (x0, x1), (y0, y1) = LIMS[key]
        for ax in axes[i]:
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
    fig.tight_layout(h_pad=1.2, w_pad=1.5)
    fig.savefig(os.path.join(OUT, 'fig_speed_appendix.pdf'))
    fig.savefig(os.path.join(OUT, 'fig_speed_appendix.png'), dpi=150)
    for key, col, _ in DATASETS:
        print(key, 'train', {m: round(tr[m][col] / tr['Ours'][col], 1) for m in MODELS if tr.get(m, {}).get(col) is not None},
              '| eval', {m: round(ev[m][col] / ev['Ours'][col], 1) for m in MODELS if ev.get(m, {}).get(col) is not None})
        print('   AP', {m: round(v, 1) for m, v in ap[key].items()})


if __name__ == '__main__':
    main()
