# -*- coding: utf-8 -*-
"""A3 (form of the source-activity term) appendix table: the remaining protocols in one table, two row blocks
(transductive setting under inductive negatives; inductive setting under historical negatives), same six rows as the
main-text table.  Test AP x100, mean +- std over 3 seeds; bold = best per column within a block.
Writes outputs/table_a3_appendix.tex / .md.
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
ROWS = [
    (None, 'trunk', 'none ($f$ alone, $\\beta=0$)', 'none (f alone, β=0)'),
    (None, 'full', '$\\log\\lambda_u$ (SNAM)$^\\star$', 'log λ_u (SNAM)*'),
    ('A3_count_slow', 'full', '$\\log(1+n_u^{K})$', 'log(1+n_u^K)'),
    ('A3_count_mid', 'full', '$\\log(1+n_u^{K/2})$', 'log(1+n_u^{K/2})'),
    ('A3_recency', 'full', '$-\\log(1+\\Delta t_u)$', '−log(1+Δt_u)'),
    ('A3_mlp', 'full', 'MLP on $\\log(1+n_u)$', 'MLP on log(1+n_u)'),
]
BLOCKS = [('transductive', 'historical', 'Transductive setting, historical negatives'),
          ('inductive', 'historical', 'Inductive setting, historical negatives')]
CAPTION = ('Form of the source-activity term, remaining protocols: test AP (\\%), mean{\\scriptsize$\\pm$std} over three seeds, for the '
           'score $f+\\beta\\,b$ with the term $b$ replaced on the same checkpoints ($\\beta$ chosen on validation; under random negatives '
           'every variant selects $\\beta=0$). $n_u^k$ is the source\'s decayed event count at scale $k$ ($K$ the slowest); $\\Delta t_u$ '
           'is the time elapsed since the source\'s previous event, so $-\\log(1+\\Delta t_u)$ is a recency score; the MLP is a two-layer '
           'network on $\\log(1+n_u)$ fitted on the training period. In the inductive setting the historical and inductive samplers draw '
           'from the same new-node subset, so only the historical protocol is shown. $\\star$: the main results. Bold: best within a block.')


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
    n = len(DATASETS)
    tex = ['% requires booktabs', '\\begin{table}[t]', '\\caption{%s}' % CAPTION, '\\label{tab:a3_appendix}', '\\centering',
           '\\%s' % a.size, '\\setlength{\\tabcolsep}{3pt}', '\\renewcommand{\\arraystretch}{0.92}',
           '\\begin{tabular}{l' + 'c' * n + '}', '\\toprule', 'term $b$ & ' + ' & '.join(nm for _, nm in DATASETS) + ' \\\\']
    md = ['| term b | ' + ' | '.join(nm for _, nm in DATASETS) + ' |', '|---|' + '---|' * n]
    for setting, strat, title in BLOCKS:
        cells = {(ri, ds): value(load(v, ds), setting, strat, mode) for ri, (v, mode, _, _) in enumerate(ROWS) for ds, _ in DATASETS}
        best = {ds: max(cells[(ri, ds)][0] for ri in range(len(ROWS))) for ds, _ in DATASETS}
        tex += ['\\midrule', '\\multicolumn{%d}{l}{\\textit{%s}} \\\\' % (n + 1, title), '\\midrule']
        md.append('| *%s* |' % title + ' |' * n)
        for ri, (v, mode, lt, lm) in enumerate(ROWS):
            if ri == 2:
                tex.append('\\cmidrule(lr){1-%d}' % (n + 1))
            ct = [(('\\textbf{%.2f}' if round(cells[(ri, ds)][0], 2) == round(best[ds], 2) else '%.2f') % cells[(ri, ds)][0]) + '{\\scriptsize$\\pm$%.2f}' % cells[(ri, ds)][1]
                  for ds, _ in DATASETS]
            cm = [(('**%.2f**' if round(cells[(ri, ds)][0], 2) == round(best[ds], 2) else '%.2f') % cells[(ri, ds)][0]) + '±%.2f' % cells[(ri, ds)][1] for ds, _ in DATASETS]
            tex.append(lt + ' & ' + ' & '.join(ct) + ' \\\\')
            md.append('| %s | %s |' % (lm, ' | '.join(cm)))
    tex += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    md += ['', 'Test AP (x100), mean ± std over 3 seeds; score f + β·b with b replaced on the same checkpoints, β chosen on validation '
               '(under random negatives every variant selects β = 0). In the inductive setting the two samplers coincide, so only the '
               'historical protocol is shown. * = main results. Bold = best within a block.']
    open(os.path.join(a.out_dir, 'table_a3_appendix.tex'), 'w', encoding='utf-8').write('\n'.join(tex) + '\n')
    open(os.path.join(a.out_dir, 'table_a3_appendix.md'), 'w', encoding='utf-8').write('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
