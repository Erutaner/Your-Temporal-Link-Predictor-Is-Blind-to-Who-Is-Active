# -*- coding: utf-8 -*-
"""A1 (number of time scales K) appendix table, one table for both settings:
columns K | Transductive: rnd, hist f, hist f+lambda, ind f, ind f+lambda | Inductive: rnd, hist f, hist f+lambda
(in the inductive setting the two samplers draw from the same new-node subset and agree to within 0.06 at fixed beta,
so its ind columns are omitted); five datasets as row blocks, rows in increasing K; the main-configuration row is starred;
bold = best of the four rows of a dataset in that column.  Test AP x100, mean +- std over 3 seeds.
Writes outputs/table_a1_appendix.tex and .md.

    python make_table_appendix.py [--size footnotesize]
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
K = {'wikipedia': (1, 8, 14, 27), 'reddit': (1, 11, 16, 32), 'uci': (1, 8, 15, 28), 'enron': (1, 9, 16, 32), 'lastfm': (1, 10, 16, 32)}
VARIANTS = ['A1_K1', 'A1_pd1', None, 'A1_pd4']          # None = the main checkpoints
COLS = [('transductive', 'random', 'full'), ('transductive', 'historical', 'trunk'), ('transductive', 'historical', 'full'),
        ('transductive', 'inductive', 'trunk'), ('transductive', 'inductive', 'full'),
        ('inductive', 'random', 'full'), ('inductive', 'historical', 'trunk'), ('inductive', 'historical', 'full')]
SUB_TEX = ['rnd', 'hist $f$', 'hist $f{+}\\lambda$', 'ind $f$', 'ind $f{+}\\lambda$', 'rnd', 'hist $f$', 'hist $f{+}\\lambda$']
SUB_MD = ['rnd', 'hist f', 'hist f+λ', 'ind f', 'ind f+λ', 'rnd', 'hist f', 'hist f+λ']
CAPTION = ('Number of timescales $K$, transductive and inductive link prediction: test AP (\\%), mean{\\scriptsize$\\pm$std} '
           'over three seeds, under random (rnd), historical (hist) and inductive (ind) negative sampling; $f$ = the base score alone '
           '($\\beta=0$), $f{+}\\lambda$ = with the source-activity term ($\\beta$ chosen on validation). $\\star$: the configuration of '
           'the main results; $K=1$ keeps a single timescale at the centre of the covered range, the other rows use one, two and four intervals per decade '
           'of it. Bold: best of the four rows of a dataset. In the inductive setting both samplers draw from the same new-node subset and '
           'agree to within 0.06 at a fixed $\\beta$, so only the historical columns are shown.')


def load(variant, ds):
    d = os.path.join(RES, 'main', ds) if variant is None else os.path.join(RES, 'ablation', variant, ds)
    fs = [os.path.join(d, '%s_final_run%d_eval.json' % (ds, s)) for s in range(3)]
    evs = [json.load(open(f)) for f in fs if os.path.exists(f)]
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
    p.add_argument('--size', default='footnotesize')
    p.add_argument('--out_dir', default=OUTPUTS)
    a = p.parse_args()
    n = len(COLS)
    rows, best = {}, {}
    for ds, _ in DATASETS:
        for vi, v in enumerate(VARIANTS):
            rows[(ds, vi)] = [value(load(v, ds), *c) for c in COLS]
        best[ds] = [max(rows[(ds, vi)][j][0] for vi in range(len(VARIANTS))) for j in range(n)]
    tex = ['% requires booktabs', '\\begin{table}[p]', '\\caption{%s}' % CAPTION, '\\label{tab:a1_scales_full}', '\\centering',
           '\\%s' % a.size, '\\setlength{\\tabcolsep}{2.5pt}', '\\renewcommand{\\arraystretch}{0.92}',
           '\\begin{tabular}{c' + 'c' * n + '}', '\\toprule',
           ' & \\multicolumn{5}{c}{Transductive} & \\multicolumn{3}{c}{Inductive} \\\\', '\\cmidrule(lr){2-6} \\cmidrule(lr){7-9}',
           '$K$ & ' + ' & '.join(SUB_TEX) + ' \\\\']
    md = ['| | Transductive | | | | | Inductive | | |', '|' + '---|' * (n + 1), '| K | ' + ' | '.join(SUB_MD) + ' |']
    for ds, name in DATASETS:
        tex += ['\\midrule', '\\multicolumn{%d}{l}{\\textit{%s}} \\\\' % (n + 1, name)]
        md.append('| *%s* |' % name + ' |' * n)
        for vi, v in enumerate(VARIANTS):
            k = K[ds][vi]
            cells = rows[(ds, vi)]
            kl = ('%d$^\\star$' % k) if v is None else str(k)
            km = ('%d*' % k) if v is None else str(k)
            ct = [(('\\textbf{%.2f}' if round(c[0], 2) == round(best[ds][j], 2) else '%.2f') % c[0]) + '{\\scriptsize$\\pm$%.2f}' % c[1] for j, c in enumerate(cells)]
            cm = [(('**%.2f**' if round(c[0], 2) == round(best[ds][j], 2) else '%.2f') % c[0]) + '±%.2f' % c[1] for j, c in enumerate(cells)]
            tex.append(' & '.join([kl] + ct) + ' \\\\')
            md.append('| ' + ' | '.join([km] + cm) + ' |')
    tex += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    md += ['', 'Test AP (x100), mean ± std over 3 seeds. f = base score alone (beta = 0); f+λ = beta chosen on the validation split of the same '
               'setting and sampling. * = the configuration of the main results; K = 1 keeps a single timescale at the centre of the covered '
               'range; the other rows use one, two and four intervals per decade of the covered range (upper bound 16, raised to 32 for the densest '
               'grid). Bold = best of the four rows of a dataset. In the inductive setting both samplers draw from the same new-node subset '
               'and agree to within 0.06 at a fixed beta, so only the historical columns are shown.']
    open(os.path.join(a.out_dir, 'table_a1_appendix.tex'), 'w', encoding='utf-8').write('\n'.join(tex) + '\n')
    open(os.path.join(a.out_dir, 'table_a1_appendix.md'), 'w', encoding='utf-8').write('\n'.join(md) + '\n')
    print('\n'.join(md[:8]))


if __name__ == '__main__':
    main()
