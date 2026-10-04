# -*- coding: utf-8 -*-
"""Speed figure (main text), LastFM: two panels sharing the y axis (transductive test AP under historical negative
sampling); x = time relative to ours on a log scale, left = training time per epoch, right = test-split scoring time.
Data: results/speed/results.jsonl (all-model benchmark, one exclusive RTX 5090, median of 3 repeats) and the main-table AP cells
(TPNet paper values; DyGFormer = our fixed-padding retrain; DSRD = our retraining with the corrected code; ours = f + beta log lambda).
Label positions are set by hand in the script.
Writes outputs/fig_speed_lastfm.pdf / .png.
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
# LastFM: training time per epoch and test-split scoring time, medians of three runs (results/speed/results.jsonl)
from common import speed_medians  # noqa: E402
_SCORING, _TRAINING = speed_medians()
TRAIN = {m: v['lastfm'] for m, v in _TRAINING.items() if 'lastfm' in v}
EVAL = {m: v['lastfm'] for m, v in _SCORING.items() if 'lastfm' in v}
# marker per model: identity is carried by the direct label and the shape, colour is secondary (Okabe-Ito hues)
STYLE = {
    'JODIE': ('o', '#0072B2'), 'DyRep': ('s', '#E69F00'), 'TGAT': ('^', '#009E73'), 'TGN': ('v', '#D55E00'),
    'CAWN': ('p', '#56B4E9'), 'TCL': ('D', '#CC79A7'), 'GraphMixer': ('*', '#8C564B'), 'DyGFormer': ('h', '#7F7F7F'),
    'EdgeBank': ('X', '#BCBD22'), 'TPNet': ('x', '#B8B000'), 'DSRD': ('P', '#17BECF'), 'Ours': ('*', '#C00000'),
}


def ap_hist():
    paper = json.load(open(os.path.join(RES, 'main', 'tpnet_paper_all_baselines.json')))['trans_hist']['AP']['LastFM']
    summ = json.load(open(os.path.join(RES, 'baselines', 'summary.json')))
    ap = {m: paper[m][0] for m in ('JODIE', 'DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'EdgeBank', 'TPNet')}
    ap['DyGFormer'] = summ['DyGFormer_fixpad']['lastfm']['trans_ap/hist']['mean']
    ap['DSRD'] = summ['DSRD_honest']['lastfm']['trans_ap/hist']['mean']
    evs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, 'main', 'lastfm', 'lastfm_final_run*_eval.json')))]
    ap['Ours'] = 100 * float(np.mean([e['selected']['transductive']['historical']['test_ap'] for e in evs]))
    return ap


# label placement per panel: (dx, dy) in points, ha / va derived from the sign
OFF = {
    'train': {'Ours': (8, 3), 'TPNet': (7, 5), 'DSRD': (7, -5), 'DyGFormer': (7, 0), 'TGN': (7, 4), 'JODIE': (-13, -5),
              'DyRep': (2, -15), 'GraphMixer': (0, -15), 'TGAT': (7, 2), 'CAWN': (0, -15), 'TCL': (7, 0)},
    'eval': {'Ours': (8, 3), 'TPNet': (7, 5), 'DSRD': (7, -5), 'DyGFormer': (7, 0), 'TGN': (12, 9), 'JODIE': (0, -14),
             'DyRep': (13, -6), 'GraphMixer': (0, -16), 'EdgeBank': (0, 12), 'TGAT': (13, 6), 'CAWN': (0, -15), 'TCL': (7, 0)},
}


# labels that get a thin leader line to their marker (crowded regions)
LEADER = {'train': {'JODIE', 'DyRep', 'GraphMixer', 'CAWN'},
          'eval': {'JODIE', 'DyRep', 'TGN', 'GraphMixer', 'EdgeBank', 'TGAT', 'CAWN'}}


def panel(ax, times, ap, key, title):
    ours = times['Ours']
    for m, t in times.items():
        mk, col = STYLE[m]
        rel = t / ours
        ax.scatter([rel], [ap[m]], marker=mk, s=160 if m == 'Ours' else 60, c=col, zorder=3)
        label = 'Ours' if m == 'Ours' else '%s $\\times$%.0f' % (m, rel) if rel >= 10 else '%s $\\times$%.1f' % (m, rel)
        dx, dy = OFF[key][m]
        ha = 'left' if dx > 0 else ('right' if dx < 0 else 'center')
        va = 'center' if dx != 0 and abs(dy) < 4 else ('bottom' if dy > 0 else 'top')
        if dx != 0 and abs(dy) >= 4:
            va = 'center'
        arrow = dict(arrowstyle='-', color='#8A8A8A', lw=0.5, shrinkA=1, shrinkB=4.5) if m in LEADER[key] else None
        ax.annotate(label, (rel, ap[m]), xytext=(dx, dy), textcoords='offset points', ha=ha, va=va, fontsize=7,
                    color='#222222', fontweight='bold' if m == 'Ours' else 'normal', zorder=4, arrowprops=arrow)
    ax.set_xscale('log')
    ax.set_xlim(0.2, 400)
    ax.set_ylim(54, 100)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('time relative to ours (log scale)', fontsize=9)
    ax.grid(True, which='major', color='#E3E3E3', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp_ in ('top', 'right'):
        ax.spines[sp_].set_visible(False)
    ax.tick_params(labelsize=8)


def main():
    ap = ap_hist()
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3), sharey=True)
    panel(axes[0], TRAIN, ap, 'train', 'LastFM: training time per epoch')
    panel(axes[1], EVAL, ap, 'eval', 'LastFM: test-split scoring time')
    axes[0].set_ylabel('AP, historical negatives (%)', fontsize=9)
    fig.tight_layout(w_pad=1.5)
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, 'fig_speed_lastfm.pdf'))
    fig.savefig(os.path.join(OUT, 'fig_speed_lastfm.png'), dpi=200)
    print('done')


if __name__ == '__main__':
    main()
