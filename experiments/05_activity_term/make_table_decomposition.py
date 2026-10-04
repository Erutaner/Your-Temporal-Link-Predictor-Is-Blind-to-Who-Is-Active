# -*- coding: utf-8 -*-
"""I1 main-text table: what goes into the score, transductive test AP under historical negative sampling,
five datasets.  Rows: DyGFormer (our fixed-padding retrain), TPNet (paper), f alone (beta = 0), log lambda_u alone
(no pair information, f zeroed), f + beta log lambda_u (ours).  mean +- std over seeds.
Writes outputs/table_i1_small.tex and .md.

    python make_table_decomposition.py [--dsrd]
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


def ours(ds, variant, mode):
    d = os.path.join(RES, 'main', ds) if variant is None else os.path.join(RES, 'ablation', variant, ds)
    fs = [os.path.join(d, '%s_final_run%d_eval.json' % (ds, s)) for s in range(3)]
    evs = [json.load(open(f)) for f in fs if os.path.exists(f)]
    assert len(evs) == 3, (variant, ds)
    if mode == 'trunk':
        x = [100 * e['grid']['transductive']['test']['historical']['by_beta']['0']['ap'] for e in evs]
    else:
        x = [100 * e['selected']['transductive']['historical']['test_ap'] for e in evs]
    return float(np.mean(x)), float(np.std(x)), 3


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dsrd', action='store_true', help='add a DSRD row (our corrected-code retrain)')
    p.add_argument('--out_dir', default=OUTPUTS)
    a = p.parse_args()
    summ = json.load(open(os.path.join(RES, 'baselines', 'summary.json')))
    paper = json.load(open(os.path.join(RES, 'main', 'tpnet_paper_all_baselines.json')))
    rows = []      # (tex label, md label, {ds: (mean, std, n, dagger)})
    rows.append(('DyGFormer$^\\dagger$', 'DyGFormer†', {ds: (summ['DyGFormer_fixpad'][ds]['trans_ap/hist']['mean'], summ['DyGFormer_fixpad'][ds]['trans_ap/hist']['std'],
                                                          summ['DyGFormer_fixpad'][ds]['trans_ap/hist']['n']) for ds, _ in DATASETS}))
    rows.append(('TPNet', 'TPNet', {ds: (paper['trans_hist']['AP'][n]['TPNet'][0], paper['trans_hist']['AP'][n]['TPNet'][1], 5) for ds, n in DATASETS}))
    if a.dsrd:
        rows.append(('DSRD$^\\dagger$', 'DSRD†', {ds: (summ['DSRD_honest'][ds]['trans_ap/hist']['mean'], summ['DSRD_honest'][ds]['trans_ap/hist']['std'], 5) for ds, _ in DATASETS}))
    rows.append(('$f$ only', 'f only', {ds: ours(ds, None, 'trunk') for ds, _ in DATASETS}))
    rows.append(('$\\log\\lambda_u$ only', 'log λ_u only', {ds: ours(ds, 'I1_bonly', 'full') for ds, _ in DATASETS}))
    rows.append(('$f+\\beta\\log\\lambda_u$', 'f + β·log λ_u', {ds: ours(ds, None, 'full') for ds, _ in DATASETS}))

    def fmt(v, tex):
        m, s, n = v
        if n == 1:
            return '%.2f' % m
        return ('%.2f{\\tiny$\\pm$%.2f}' if tex else '%.2f±%.2f') % (m, s)

    best = {ds: max(r[2][ds][0] for r in rows) for ds, _ in DATASETS}
    cap = ('What the score contains, transductive test AP (\\%) under historical negative sampling, mean{\\tiny$\\pm$std} over seeds '
           '(three for our rows and DyGFormer$^\\dagger$, one on LastFM; five for the rest). $f$ only: the base score ($\\beta=0$); $\\log\\lambda_u$ only: the source\'s event '
           'rate only, with no pair information. $\\dagger$: retrained by us with the corrected released code. Bold: best.')
    tex = ['% requires booktabs', '\\begin{table}[t]', '\\caption{%s}' % cap, '\\label{tab:i1_decomposition}', '\\centering', '\\scriptsize',
           '\\setlength{\\tabcolsep}{2pt}', '\\renewcommand{\\arraystretch}{0.9}', '\\begin{tabular}{l' + 'c' * len(DATASETS) + '}', '\\toprule',
           ' & ' + ' & '.join(n for _, n in DATASETS) + ' \\\\', '\\midrule']
    md = ['| score | ' + ' | '.join(n for _, n in DATASETS) + ' |', '|---|' + '---|' * len(DATASETS)]
    for i, (lt, lm, vals) in enumerate(rows):
        if lt.startswith('$f$ only'):
            tex.append('\\midrule')
        ct = [('\\textbf{%s}' % fmt(vals[ds], True)) if round(vals[ds][0], 2) == round(best[ds], 2) else fmt(vals[ds], True) for ds, _ in DATASETS]
        cm = [('**%s**' % fmt(vals[ds], False)) if round(vals[ds][0], 2) == round(best[ds], 2) else fmt(vals[ds], False) for ds, _ in DATASETS]
        tex.append(lt + ' & ' + ' & '.join(ct) + ' \\\\')
        md.append('| %s | %s |' % (lm, ' | '.join(cm)))
    tex += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    md += ['', 'Transductive test AP (x100), historical negative sampling, mean ± std over seeds (3 for our rows and DyGFormer†, 1 on LastFM; '
               '5 otherwise). log λ_u alone = f zeroed, β = 1: the source\'s event rate with no pair information. † = our retrain with the corrected released '
               'code. Bold = best.']
    with open(os.path.join(a.out_dir, 'table_i1_small.tex'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(tex) + '\n')
    with open(os.path.join(a.out_dir, 'table_i1_small.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(md) + '\n')
    print('\n'.join(md))
    print('\n'.join(tex))


if __name__ == '__main__':
    main()
