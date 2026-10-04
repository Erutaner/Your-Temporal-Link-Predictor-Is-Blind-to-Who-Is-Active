# -*- coding: utf-8 -*-
"""Trunk-channel ablation table, one table for both settings, same columns as the time-scale appendix table:
variant | Transductive: rnd, hist f, hist f+lambda, ind f, ind f+lambda | Inductive: rnd, hist f, hist f+lambda.
Rows per stream: the main configuration (starred), then the model retrained without the set encoder, without the
random-projection channels (which also removes the set encoder), without the count channels, and with none of the
three (the bare trunk).  Test AP x100, mean +- std over 3 seeds; bold = best of the rows of a stream in that column.
Writes outputs/table_a4_appendix.tex and .md.

    python make_table.py [--size scriptsize] [--partial]     # --partial: rows with fewer than 3 seeds are shown with (n)
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common import RESULTS, OUTPUTS  # noqa: E402

RES = RESULTS
DATASETS = [('wikipedia', 'Wikipedia'), ('reddit', 'Reddit'), ('uci', 'UCI'), ('enron', 'Enron'), ('lastfm', 'LastFM')]
# (result folder or None = the main checkpoints, tex label, md label)
ROWS = [(None, 'all inputs$^\\star$', 'all inputs *'),
        ('A4_noset', 'no $c_{uv}$', 'no c_uv'),
        ('A4_nosketch', 'no projections', 'no projections'),
        ('A4_nocount', 'no counts', 'no counts'),
        ('A4_bare', 'none of the three', 'none of the three')]
COLS = [('transductive', 'random', 'full'), ('transductive', 'historical', 'trunk'), ('transductive', 'historical', 'full'),
        ('transductive', 'inductive', 'trunk'), ('transductive', 'inductive', 'full'),
        ('inductive', 'random', 'full'), ('inductive', 'historical', 'trunk'), ('inductive', 'historical', 'full')]
SUB_TEX = ['rnd', 'hist $f$', 'hist $f{+}\\lambda$', 'ind $f$', 'ind $f{+}\\lambda$', 'rnd', 'hist $f$', 'hist $f{+}\\lambda$']
SUB_MD = ['rnd', 'hist f', 'hist f+λ', 'ind f', 'ind f+λ', 'rnd', 'hist f', 'hist f+λ']
CAPTION = ('Which inputs of the relation branch carry the pair information: test AP (\\%), mean{\\scriptsize$\\pm$std} over three seeds, '
           'transductive and inductive link prediction under random (rnd), historical (hist) and inductive (ind) negative sampling; '
           '$f$ = the base score alone ($\\beta=0$), $f{+}\\lambda$ = with the source-activity term ($\\beta$ chosen on validation). '
           '$\\star$: the configuration of the main results. The other rows retrain the model without the recent interaction encoding '
           '$c_{uv}(t)$ of the source\'s last 20 events, without the structural features from random projections (which removes the recent '
           'interaction encoding as well, since it reads them), without the pair and endpoint counts $n_{uv}^k$, $n_u^k$, $n_v^k$, and without all three (the node embeddings, the bilinear '
           'matches and the static term only). Bold: best of the rows of a stream. In the inductive setting both samplers draw '
           'from the same new-node subset and agree to within 0.06 at a fixed $\\beta$, so only the historical columns are shown.')


def load(variant, ds, partial):
    d = os.path.join(RES, 'main', ds) if variant is None else os.path.join(RES, 'ablation', variant, ds)
    fs = [os.path.join(d, '%s_final_run%d_eval.json' % (ds, s)) for s in range(3)]
    evs = [json.load(open(f)) for f in fs if os.path.exists(f)]
    if not partial:
        assert len(evs) == 3, (variant, ds, len(evs))
    return evs


def value(evs, setting, strat, mode):
    if mode == 'trunk':
        x = [100 * e['grid'][setting]['test'][strat]['by_beta']['0']['ap'] for e in evs]
    else:
        x = [100 * e['selected'][setting][strat]['test_ap'] for e in evs]
    return float(np.mean(x)), float(np.std(x))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--size', default='scriptsize')
    p.add_argument('--out_dir', default=OUTPUTS)
    p.add_argument('--partial', action='store_true')
    a = p.parse_args()
    n = len(COLS)
    cells, counts = {}, {}
    for ds, _ in DATASETS:
        for variant, _, _ in ROWS:
            evs = load(variant, ds, a.partial)
            counts[(ds, variant)] = len(evs)
            if evs:
                cells[(ds, variant)] = [value(evs, s, st, m) for s, st, m in COLS]
    best = {}
    for ds, _ in DATASETS:
        for j in range(n):
            vals = [round(cells[(ds, v)][j][0], 2) for v, _, _ in ROWS if (ds, v) in cells]
            best[(ds, j)] = max(vals) if vals else None
    tex = ['% requires booktabs', '\\begin{table}[p]', '\\caption{%s}' % CAPTION, '\\label{tab:a4_trunk_channels}', '\\centering',
           '\\%s' % a.size, '\\setlength{\\tabcolsep}{2pt}', '\\renewcommand{\\arraystretch}{0.92}',
           '\\begin{tabular}{l' + 'c' * n + '}', '\\toprule',
           ' & \\multicolumn{5}{c}{Transductive} & \\multicolumn{3}{c}{Inductive} \\\\',
           '\\cmidrule(lr){2-6} \\cmidrule(lr){7-9}',
           'variant & ' + ' & '.join(SUB_TEX) + ' \\\\', '\\midrule']
    md = ['| stream | variant | ' + ' | '.join('T ' + s for s in SUB_MD[:5]) + ' | ' + ' | '.join('I ' + s for s in SUB_MD[5:]) + ' |',
          '|' + '---|' * (n + 2)]
    for bi, (ds, name) in enumerate(DATASETS):
        if bi:
            tex.append('\\midrule')
        tex.append('\\multicolumn{%d}{l}{\\textit{%s}} \\\\' % (n + 1, name))
        first = True
        for variant, lt, lm in ROWS:
            if (ds, variant) not in cells:
                continue
            row = cells[(ds, variant)]
            suffix = '' if counts[(ds, variant)] == 3 else ' (%d)' % counts[(ds, variant)]
            tcells, mcells = [], []
            for j, (m, s) in enumerate(row):
                txt = '%.2f' % m
                b = round(m, 2) == best[(ds, j)]
                tcells.append(('\\textbf{%s}' % txt if b else txt) + '{\\tiny$\\pm$%.2f}' % s)
                mcells.append(('**%s**' % txt if b else txt) + '±%.2f' % s)
            tex.append('%s%s & %s \\\\' % (lt, suffix, ' & '.join(tcells)))
            md.append('| %s | %s%s | %s |' % (name if first else '', lm, suffix, ' | '.join(mcells)))
            first = False
    tex += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    os.makedirs(a.out_dir, exist_ok=True)
    open(os.path.join(a.out_dir, 'table_a4_appendix.tex'), 'w', encoding='utf-8').write('\n'.join(tex) + '\n')
    open(os.path.join(a.out_dir, 'table_a4_appendix.md'), 'w', encoding='utf-8').write('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
