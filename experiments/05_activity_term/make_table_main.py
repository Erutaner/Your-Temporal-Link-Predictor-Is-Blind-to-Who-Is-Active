# -*- coding: utf-8 -*-
"""A3 (form of the source-activity term) table: score = f + beta * b with b replaced, beta chosen on validation,
checkpoints untouched; five datasets, transductive test AP under one negative-sampling strategy (default historical).
Rows: f alone; b = log lambda_u (ground process, main results, starred); slowest-scale count; middle-scale count;
time since the source's last event; MLP on the counts fitted on the training period.  mean +- std over 3 seeds;
bold = best per column.  Writes outputs/table_a3_<setting>_<strategy>.tex / .md.

    python make_table_main.py [--setting transductive] [--strategy historical] [--size scriptsize]
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
ROWS = [  # (variant dir or None = main checkpoints, mode, tex label, md label)
    (None, 'trunk', 'none ($f$ alone, $\\beta=0$)', 'none (f alone, β=0)'),
    (None, 'full', '$\\log\\lambda_u$ (SNAM)$^\\star$', 'log λ_u (SNAM)*'),
    ('A3_count_slow', 'full', '$\\log(1+n_u^{K})$', 'log(1+n_u^K)'),
    ('A3_count_mid', 'full', '$\\log(1+n_u^{K/2})$', 'log(1+n_u^{K/2})'),
    ('A3_recency', 'full', '$-\\log(1+\\Delta t_u)$', '−log(1+Δt_u)'),
    ('A3_mlp', 'full', 'MLP on $\\log(1+n_u)$', 'MLP on log(1+n_u)'),
]


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
    p.add_argument('--setting', default='transductive', choices=['transductive', 'inductive'])
    p.add_argument('--strategy', default='historical', choices=['random', 'historical', 'inductive'])
    p.add_argument('--size', default='scriptsize')
    p.add_argument('--out_dir', default=OUTPUTS)
    a = p.parse_args()
    cells = {(ri, ds): value(load(v, ds), a.setting, a.strategy, mode) for ri, (v, mode, _, _) in enumerate(ROWS) for ds, _ in DATASETS}
    best = {ds: max(cells[(ri, ds)][0] for ri in range(len(ROWS))) for ds, _ in DATASETS}
    stitle = {'transductive': 'Transductive', 'inductive': 'Inductive'}[a.setting]
    cap = ('Form of the source-activity term: %s test AP (\\%%) under %s negative sampling, mean{\\tiny$\\pm$std} over three seeds, '
           'for the score $f+\\beta\\,b$ with the term $b$ replaced on the same checkpoints ($\\beta$ chosen on validation; under random '
           'negatives every variant selects $\\beta=0$). $n_u^k$ is the source\'s decayed event count at timescale $k$ ($K$ the slowest); '
           '$\\Delta t_u$ is the time elapsed since the source\'s previous event, so $-\\log(1+\\Delta t_u)$ is a recency score; the MLP is a '
           'two-layer network on $\\log(1+n_u)$ fitted on the training period. $\\star$: the main results. Bold: best.'
           % (stitle.lower(), a.strategy))
    tex = ['% requires booktabs', '\\begin{table}[t]', '\\caption{%s}' % cap, '\\label{tab:a3_%s_%s}' % (a.setting, a.strategy), '\\centering',
           '\\%s' % a.size, '\\setlength{\\tabcolsep}{2.5pt}', '\\renewcommand{\\arraystretch}{0.9}',
           '\\begin{tabular}{l' + 'c' * len(DATASETS) + '}', '\\toprule',
           'term $b$ & ' + ' & '.join(n for _, n in DATASETS) + ' \\\\', '\\midrule']
    md = ['| term b | ' + ' | '.join(n for _, n in DATASETS) + ' |', '|---|' + '---|' * len(DATASETS)]
    for ri, (v, mode, lt, lm) in enumerate(ROWS):
        if ri == 2:
            tex.append('\\midrule')
        ct = [(('\\textbf{%.2f}' if round(cells[(ri, ds)][0], 2) == round(best[ds], 2) else '%.2f') % cells[(ri, ds)][0]) + '{\\tiny$\\pm$%.2f}' % cells[(ri, ds)][1]
              for ds, _ in DATASETS]
        cm = [(('**%.2f**' if round(cells[(ri, ds)][0], 2) == round(best[ds], 2) else '%.2f') % cells[(ri, ds)][0]) + '±%.2f' % cells[(ri, ds)][1] for ds, _ in DATASETS]
        tex.append(lt + ' & ' + ' & '.join(ct) + ' \\\\')
        md.append('| %s | %s |' % (lm, ' | '.join(cm)))
    tex += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    md += ['', '%s test AP (x100) under %s negatives, mean ± std over 3 seeds; score f + β·b with b replaced on the same checkpoints, '
               'β chosen on validation (under random negatives every variant selects β = 0). * = main results. Bold = best.' % (stitle, a.strategy)]
    fn = 'table_a3_%s_%s' % (a.setting, a.strategy)
    open(os.path.join(a.out_dir, fn + '.tex'), 'w', encoding='utf-8').write('\n'.join(tex) + '\n')
    open(os.path.join(a.out_dir, fn + '.md'), 'w', encoding='utf-8').write('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
