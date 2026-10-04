# -*- coding: utf-8 -*-
"""A1 (number of time scales K) main-text table for two datasets, sub-columns K | rnd | hist | ind, rows in
increasing K (K = 1 single rate at the centre of the covered range; 1 / 2 (ours) / 4 rates per decade),
transductive test AP x100, mean +- std over 3 seeds, full model (beta chosen on validation).

Layouts: 'stacked' (default) = one 4-column block per dataset, one under the other (narrow, pairs with another
table); 'wide' = the two datasets side by side (8 columns).  Size = LaTeX font switch for the tabular.
Writes outputs/table_a1_small.tex and .md.

    python make_table_main.py [--datasets uci,reddit] [--layout stacked|wide] [--size footnotesize]
"""
import argparse
import json
import os

import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common import RESULTS, OUTPUTS  # noqa: E402

RES = RESULTS
NAMES = {'uci': 'UCI', 'reddit': 'Reddit', 'wikipedia': 'Wikipedia', 'enron': 'Enron', 'lastfm': 'LastFM'}
# number of scales per dataset for the four grids, read from the training logs (decades=... K=...)
K = {'wikipedia': (1, 8, 14, 27), 'reddit': (1, 11, 16, 32), 'uci': (1, 8, 15, 28), 'enron': (1, 9, 16, 32), 'lastfm': (1, 10, 16, 32)}
VARIANTS = ['A1_K1', 'A1_pd1', None, 'A1_pd4']          # None = the main checkpoints
STRATS = [('rnd', 'random'), ('hist', 'historical'), ('ind', 'inductive')]
CAPTION = ('Number of timescales $K$: transductive test AP (\\%), mean{\\tiny$\\pm$std} over three seeds, under '
           'random (rnd), historical (hist) and inductive (ind) negative sampling. $\\star$: the configuration of the main results; '
           '$K=1$ keeps a single timescale at the centre of the covered range. Bold: best per column.')


def load(variant, ds):
    d = os.path.join(RES, 'main', ds) if variant is None else os.path.join(RES, 'ablation', variant, ds)
    fs = [os.path.join(d, '%s_final_run%d_eval.json' % (ds, s)) for s in range(3)]
    evs = [json.load(open(f)) for f in fs if os.path.exists(f)]
    assert len(evs) == 3, (variant, ds, len(evs))
    return evs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', default='uci,reddit')
    p.add_argument('--layout', default='stacked', choices=['stacked', 'wide'])
    p.add_argument('--size', default='footnotesize', help='small | footnotesize | scriptsize')
    p.add_argument('--out_dir', default=OUTPUTS)
    a = p.parse_args()
    dss = a.datasets.split(',')
    cells = {}
    for ds in dss:
        for vi, v in enumerate(VARIANTS):
            evs = load(v, ds)
            for blk, strat in STRATS:
                x = [100 * e['selected']['transductive'][strat]['test_ap'] for e in evs]
                cells[(ds, vi, blk)] = (float(np.mean(x)), float(np.std(x)))

    best = {(ds, blk): max(cells[(ds, vi, blk)][0] for vi in range(len(VARIANTS))) for ds in dss for blk, _ in STRATS}

    def tex_k(ds, vi):
        return ('%d$^\\star$' % K[ds][vi]) if VARIANTS[vi] is None else str(K[ds][vi])

    def tex_v(ds, vi, blk):
        m, s = cells[(ds, vi, blk)]
        return (('\\textbf{%.2f}' if round(m, 2) == round(best[(ds, blk)], 2) else '%.2f') % m) + '{\\tiny$\\pm$%.2f}' % s

    def md_k(ds, vi):
        return ('%d*' % K[ds][vi]) if VARIANTS[vi] is None else str(K[ds][vi])

    def md_v(ds, vi, blk):
        m, s = cells[(ds, vi, blk)]
        return (('**%.2f**' if round(m, 2) == round(best[(ds, blk)], 2) else '%.2f') % m) + '±%.2f' % s

    pre = ['% requires booktabs', '\\begin{table}[t]', '\\caption{%s}' % CAPTION, '\\label{tab:a1_scales}', '\\centering',
           '\\%s' % a.size, '\\setlength{\\tabcolsep}{3pt}', '\\renewcommand{\\arraystretch}{0.92}']
    if a.layout == 'wide':
        tex = pre + ['\\begin{tabular}{' + ' '.join('cccc' for _ in dss) + '}', '\\toprule',
                     ' & '.join('\\multicolumn{4}{c}{%s}' % NAMES[ds] for ds in dss) + ' \\\\',
                     ' '.join('\\cmidrule(lr){%d-%d}' % (4 * i + 1, 4 * i + 4) for i in range(len(dss))),
                     ' & '.join('$K$ & rnd & hist & ind' for _ in dss) + ' \\\\', '\\midrule']
        for vi in range(len(VARIANTS)):
            row = []
            for ds in dss:
                row += [tex_k(ds, vi)] + [tex_v(ds, vi, blk) for blk, _ in STRATS]
            tex.append(' & '.join(row) + ' \\\\')
        tex += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
        md = ['| ' + ' | '.join('%s | | | ' % NAMES[ds] for ds in dss) + '|', '|' + '---|' * (4 * len(dss)),
              '| ' + ' | '.join('K | rnd | hist | ind' for _ in dss) + ' |']
        for vi in range(len(VARIANTS)):
            row = []
            for ds in dss:
                row += [md_k(ds, vi)] + [md_v(ds, vi, blk) for blk, _ in STRATS]
            md.append('| ' + ' | '.join(row) + ' |')
    else:
        tex = pre + ['\\begin{tabular}{cccc}', '\\toprule', '$K$ & rnd & hist & ind \\\\']
        md = ['| K | rnd | hist | ind |', '|---|---|---|---|']
        for di, ds in enumerate(dss):
            tex.append('\\midrule')
            tex.append('\\multicolumn{4}{c}{\\textit{%s}} \\\\' % NAMES[ds])
            tex.append('\\midrule')
            md.append('| *%s* | | | |' % NAMES[ds])
            for vi in range(len(VARIANTS)):
                tex.append(' & '.join([tex_k(ds, vi)] + [tex_v(ds, vi, blk) for blk, _ in STRATS]) + ' \\\\')
                md.append('| ' + ' | '.join([md_k(ds, vi)] + [md_v(ds, vi, blk) for blk, _ in STRATS]) + ' |')
        tex += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    md.append('')
    md.append('Transductive test AP (x100), mean ± std over 3 seeds, full model (beta chosen on validation). * = the configuration of the '
              'main results; K = 1 keeps a single timescale at the centre of the covered range; the other rows use one, two and four intervals per decade of the '
              'covered range (upper bound 16, raised to 32 for the densest grid). Bold = best per column.')
    with open(os.path.join(a.out_dir, 'table_a1_small.tex'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(tex) + '\n')
    with open(os.path.join(a.out_dir, 'table_a1_small.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(md) + '\n')
    print('\n'.join(tex))


if __name__ == '__main__':
    main()
