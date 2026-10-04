# -*- coding: utf-8 -*-
"""A2 (event clock) appendix table, one table for both settings, same columns as the A1 appendix table:
T | Transductive: rnd, hist f, hist f+lambda, ind f, ind f+lambda | Inductive: rnd, hist f, hist f+lambda.
Rows: five datasets, each with two sub-blocks (within-chunk records on / off) of T = 48 / 96 / 192; the main-configuration
cell (on, the dataset's T) is starred and comes from the main checkpoints; bold = best of the six rows of a
dataset in that column.  Test AP x100, mean +- std over 3 seeds.
Writes outputs/table_a2_appendix.tex and .md.
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
DATASETS = [('wikipedia', 'Wikipedia'), ('reddit', 'Reddit'), ('uci', 'UCI'), ('enron', 'Enron'), ('lastfm', 'LastFM')]
FINAL_T = {'wikipedia': 96, 'reddit': 48, 'uci': 192, 'enron': 192, 'lastfm': 192}
TS = (48, 96, 192)
COLS = [('transductive', 'random', 'full'), ('transductive', 'historical', 'trunk'), ('transductive', 'historical', 'full'),
        ('transductive', 'inductive', 'trunk'), ('transductive', 'inductive', 'full'),
        ('inductive', 'random', 'full'), ('inductive', 'historical', 'trunk'), ('inductive', 'historical', 'full')]
SUB_TEX = ['rnd', 'hist $f$', 'hist $f{+}\\lambda$', 'ind $f$', 'ind $f{+}\\lambda$', 'rnd', 'hist $f$', 'hist $f{+}\\lambda$']
SUB_MD = ['rnd', 'hist f', 'hist f+λ', 'ind f', 'ind f+λ', 'rnd', 'hist f', 'hist f+λ']
CAPTION = ('Number of training blocks and within-block updates, transductive and inductive link prediction: test AP (\\%), mean{\\scriptsize$\\pm$std} over three seeds. '
           '$C$ = number of training blocks; $\\star$ = the configuration of the main results. With within-block updates on, the '
           'events of the current block that precede the query time enter the counts, the feature states and the random '
           'projections; off, only the block-start state, decayed to the query time, is used. $f$ = the base score alone ($\\beta=0$), '
           '$f{+}\\lambda$ = with the source-activity term ($\\beta$ chosen on validation). Bold: best of the six rows of a dataset. '
           'In the inductive setting both samplers draw from the same new-node subset, so only the historical columns are shown.')


def load(fr, t, ds):
    if fr == 1 and t == FINAL_T[ds]:
        d = os.path.join(RES, 'main', ds)
    else:
        d = os.path.join(RES, 'ablation', 'A2_fr%d_t%d' % (fr, t), ds)
    fs = [os.path.join(d, '%s_final_run%d_eval.json' % (ds, s)) for s in range(3)]
    evs = [json.load(open(f)) for f in fs if os.path.exists(f)]
    assert len(evs) == 3, (fr, t, ds, len(evs))
    return evs


def value(evs, setting, strat, mode):
    if mode == 'trunk':
        x = [100 * e['grid'][setting]['test'][strat]['by_beta']['0']['ap'] for e in evs]
    else:
        x = [100 * e['selected'][setting][strat]['test_ap'] for e in evs]
    return float(np.mean(x)), float(np.std(x))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--size', default='footnotesize')
    p.add_argument('--out_dir', default=OUTPUTS)
    a = p.parse_args()
    n = len(COLS)
    rows, best = {}, {}
    for ds, _ in DATASETS:
        for fr in (1, 0):
            for t in TS:
                rows[(ds, fr, t)] = [value(load(fr, t, ds), *c) for c in COLS]
        best[ds] = [max(rows[(ds, fr, t)][j][0] for fr in (1, 0) for t in TS) for j in range(n)]
    tex = ['% requires booktabs', '\\begin{table}[p]', '\\caption{%s}' % CAPTION, '\\label{tab:a2_event_clock}', '\\centering',
           '\\%s' % a.size, '\\setlength{\\tabcolsep}{2.5pt}', '\\renewcommand{\\arraystretch}{0.92}',
           '\\begin{tabular}{c' + 'c' * n + '}', '\\toprule',
           ' & \\multicolumn{5}{c}{Transductive} & \\multicolumn{3}{c}{Inductive} \\\\', '\\cmidrule(lr){2-6} \\cmidrule(lr){7-9}',
           '$C$ & ' + ' & '.join(SUB_TEX) + ' \\\\']
    md = ['| | Transductive | | | | | Inductive | | |', '|' + '---|' * (n + 1), '| C | ' + ' | '.join(SUB_MD) + ' |']
    for ds, name in DATASETS:
        tex.append('\\midrule')
        for fr, word in ((1, 'on'), (0, 'off')):
            if fr == 0:
                tex.append('\\cmidrule(lr){1-%d}' % (n + 1))
            tex.append('\\multicolumn{%d}{l}{\\textit{%s, within-block updates %s}} \\\\' % (n + 1, name, word))
            md.append('| *%s, within-block updates %s* |' % (name, word) + ' |' * n)
            for t in TS:
                cells = rows[(ds, fr, t)]
                main_cfg = fr == 1 and t == FINAL_T[ds]
                tl = ('%d$^\\star$' % t) if main_cfg else str(t)
                ml = ('%d*' % t) if main_cfg else str(t)
                ct = [(('\\textbf{%.2f}' if round(c[0], 2) == round(best[ds][j], 2) else '%.2f') % c[0]) + '{\\scriptsize$\\pm$%.2f}' % c[1]
                      for j, c in enumerate(cells)]
                cm = [(('**%.2f**' if round(c[0], 2) == round(best[ds][j], 2) else '%.2f') % c[0]) + '±%.2f' % c[1] for j, c in enumerate(cells)]
                tex.append(' & '.join([tl] + ct) + ' \\\\')
                md.append('| ' + ' | '.join([ml] + cm) + ' |')
    tex += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    md += ['', 'Test AP (x100), mean ± std over 3 seeds. C = training blocks; * = main-results configuration. Within-block updates on = '
               'the events of the current block before the query time enter the counts, feature states and random projections; '
               'off = only the block-start state decayed to the query time is used. f = base score alone (beta = 0); f+λ = beta chosen on '
               'validation. Bold = best of the six rows of a dataset. In the inductive setting both samplers draw from the same new-node '
               'subset, so only the historical columns are shown.']
    open(os.path.join(a.out_dir, 'table_a2_appendix.tex'), 'w', encoding='utf-8').write('\n'.join(tex) + '\n')
    open(os.path.join(a.out_dir, 'table_a2_appendix.md'), 'w', encoding='utf-8').write('\n'.join(md) + '\n')
    print('\n'.join(md[:12]))


if __name__ == '__main__':
    main()
