# -*- coding: utf-8 -*-
"""AP versus the readout coefficient beta (score = f + beta log lambda_u), transductive test split, three
negative-sampling protocols; mean over the five seeds with a +-1 std band; the validation-selected beta of each
protocol (the value most seeds selected) is marked with a ring.  Data: results/main/<ds>/<ds>_final_run*_eval.json.
Writes outputs/fig_beta_<ds>.pdf / .png for one dataset; with --all the twelve-panel appendix figure fig_beta_appendix instead.

    python make_figures.py [--dataset mooc | --all] [--setting transductive]
"""
import argparse
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
ORDER = [('wikipedia', 'Wikipedia'), ('reddit', 'Reddit'), ('mooc', 'MOOC'), ('lastfm', 'LastFM'), ('enron', 'Enron'),
         ('SocialEvo', 'Social Evo.'), ('uci', 'UCI'), ('Flights', 'Flights'), ('CanParl', 'Can. Parl.'), ('USLegis', 'US Legis.'),
         ('UNtrade', 'UN Trade'), ('UNvote', 'UN Vote'), ('Contacts', 'Contact')]
NAMES = dict(ORDER)
STRATS = [('random', 'random', '#0072B2', 'o'), ('historical', 'historical', '#D55E00', 's'), ('inductive', 'inductive', '#009E73', '^')]


def draw(ax, ds, setting, legend=True, title=True):
    evs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, 'main', ds, '%s_final_run*_eval.json' % ds)))]
    betas = [float(b) for b in evs[0]['betas']]
    for strat, label, col, mk in STRATS:
        ap = np.array([[100 * e['grid'][setting]['test'][strat]['by_beta']['%g' % b]['ap'] for b in betas] for e in evs])
        m, s = ap.mean(0), ap.std(0)
        ax.fill_between(betas, m - s, m + s, color=col, alpha=0.13, linewidth=0)
        ax.plot(betas, m, color=col, marker=mk, markersize=4.5, markeredgecolor='white', markeredgewidth=0.6,
                linewidth=1.5, label=label + ' negatives')
        sel = [e['selected'][setting][strat]['beta'] for e in evs]
        b_sel = max(sorted(set(sel)), key=sel.count)
        ax.plot([b_sel], [m[betas.index(b_sel)]], marker='o', markersize=10, markerfacecolor='none', markeredgecolor=col,
                markeredgewidth=1.3, linestyle='none', zorder=4)
    ax.set_xticks(betas)
    ax.set_xlim(-0.12, betas[-1] + 0.12)
    ax.tick_params(labelsize=7.5, length=2.5)
    ax.grid(True, color='#E6E6E6', linewidth=0.6)
    ax.set_axisbelow(True)
    for sp_ in ('top', 'right'):
        ax.spines[sp_].set_visible(False)
    for sp_ in ('left', 'bottom'):
        ax.spines[sp_].set_color('#888888')
    if title:
        ax.set_title(NAMES.get(ds, ds), fontsize=9.5, pad=4)
    if legend:
        ax.legend(fontsize=7, frameon=False, handlelength=1.8, borderaxespad=0.2, loc='best')
    return len(evs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='wikipedia')
    p.add_argument('--setting', default='transductive')
    p.add_argument('--all', action='store_true')
    a = p.parse_args()
    dss = [] if a.all else [a.dataset]
    for ds in dss:
        fig, ax = plt.subplots(figsize=(3.0, 2.8))
        n = draw(ax, ds, a.setting)
        ax.set_xlabel(r'$\beta$ in $f+\beta\log\lambda_u$', fontsize=8.5)
        ax.set_ylabel('test AP (%)', fontsize=8.5)
        fig.tight_layout(pad=0.4)
        fig.savefig(os.path.join(OUT, 'fig_beta_%s.pdf' % ds))
        fig.savefig(os.path.join(OUT, 'fig_beta_%s.png' % ds), dpi=200)
        plt.close(fig)
        print(ds, 'seeds', n)
    if a.all:
        # appendix figure: the twelve datasets other than MOOC (main text), 4 rows x 3 columns, shared axis labels
        rest = [(k, n) for k, n in ORDER if k != 'mooc']
        fig, axes = plt.subplots(4, 3, figsize=(7.2, 8.8))
        for i, (ax, (ds, _)) in enumerate(zip(axes.flat, rest)):
            draw(ax, ds, a.setting, legend=(i == 0))
            if i % 3 == 0:
                ax.set_ylabel('test AP (%)', fontsize=8.5)
            if i >= 9:
                ax.set_xlabel(r'$\beta$ in $f+\beta\log\lambda_u$', fontsize=8.5)
        fig.tight_layout(pad=0.5, h_pad=1.0, w_pad=1.0)
        fig.savefig(os.path.join(OUT, 'fig_beta_appendix.pdf'))
        fig.savefig(os.path.join(OUT, 'fig_beta_appendix.png'), dpi=150)
        print('appendix figure written')


if __name__ == '__main__':
    main()
