# -*- coding: utf-8 -*-
"""Transfer table for the main text: four backbones x five datasets, transductive test AP under historical negative
sampling, mean +- std over the backbone seeds; per backbone two rows (alone / + our lambda readout), plus ours.
Reads results/transfer/<backbone>/<ds>/seed<S>_transfer.json and the evaluation files of our main runs.
Writes outputs/table_transfer_small.tex and .md.
"""
import glob
import json
import os

import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common import RESULTS, OUTPUTS  # noqa: E402

RES = RESULTS
DATASETS = [('wikipedia', 'Wikipedia'), ('reddit', 'Reddit'), ('uci', 'UCI'), ('enron', 'Enron'), ('lastfm', 'LastFM')]
BACKBONES = [('TPNet_transfer', 'TPNet'), ('TGN_transfer', 'TGN$^\\dagger$', ), ('DyGFormer_fixpad_transfer', 'DyGFormer$^\\dagger$'),
             ('DSRD_transfer', 'DSRD$^\\dagger$')]
MD_NAME = {'TPNet': 'TPNet', 'TGN$^\\dagger$': 'TGN†', 'DyGFormer$^\\dagger$': 'DyGFormer†', 'DSRD$^\\dagger$': 'DSRD†'}


def backbone_cells(key, ds):
    fs = sorted(glob.glob(os.path.join(RES, 'transfer', key, ds, 'seed*_transfer.json')))
    runs = [json.load(open(f)) for f in fs]
    alone = [100 * r['grid']['transductive']['test']['historical']['by_beta']['0']['ap'] for r in runs]
    plus = [100 * r['selected']['transductive']['historical']['test_ap'] for r in runs]
    return (np.mean(alone), np.std(alone), len(alone)), (np.mean(plus), np.std(plus), len(plus))


def ours(ds):
    evs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, 'main', ds, '%s_final_run*_eval.json' % ds)))]
    x = [100 * e['selected']['transductive']['historical']['test_ap'] for e in evs]
    return np.mean(x), np.std(x), len(x)


def fmt(v, tex):
    m, s, n = v
    if n == 1:
        return '%.2f' % m
    return ('%.2f{\\tiny$\\pm$%.2f}' if tex else '%.2f±%.2f') % (m, s)


def main():
    cap = ('Transfer of the source-activity term: transductive test AP (\\%) under historical negative sampling for four backbones, alone and with '
           '$\\beta\\log\\lambda_u$ added to their logits, mean{\\tiny$\\pm$std} over three seeds (DyGFormer on LastFM: one). The backbone '
           'is not retrained and $\\lambda_u$ keeps the parameters fitted for our model on the same dataset; only $\\beta$ is chosen on '
           'the backbone\'s validation split. $\\dagger$: retrained by us with the corrected released code.')
    tex = ['% requires booktabs', '\\begin{table}[t]', '\\caption{%s}' % cap, '\\label{tab:transfer}', '\\centering', '\\scriptsize',
           '\\setlength{\\tabcolsep}{2.5pt}', '\\renewcommand{\\arraystretch}{0.95}', '\\begin{tabular}{llccccc}', '\\toprule',
           'backbone & score & ' + ' & '.join(n for _, n in DATASETS) + ' \\\\', '\\midrule']
    md = ['| backbone | score | ' + ' | '.join(n for _, n in DATASETS) + ' |', '|---|---|' + '---|' * len(DATASETS)]
    for key, name in BACKBONES:
        cells = {ds: backbone_cells(key, ds) for ds, _ in DATASETS}
        tex.append('\\multirow{2}{*}{%s} & alone & ' % name + ' & '.join(fmt(cells[ds][0], True) for ds, _ in DATASETS) + ' \\\\')
        tex.append(' & $+\\beta\\log\\lambda_u$ & ' + ' & '.join('\\textbf{%s}' % fmt(cells[ds][1], True) for ds, _ in DATASETS) + ' \\\\')
        tex.append('\\midrule')
        md.append('| %s | alone | ' % MD_NAME[name] + ' | '.join(fmt(cells[ds][0], False) for ds, _ in DATASETS) + ' |')
        md.append('| | +β·log λ_u | ' + ' | '.join('**%s**' % fmt(cells[ds][1], False) for ds, _ in DATASETS) + ' |')
    o = {ds: ours(ds) for ds, _ in DATASETS}
    tex.append('\\multicolumn{2}{l}{Ours ($f+\\beta\\log\\lambda_u$)} & ' + ' & '.join(fmt(o[ds], True) for ds, _ in DATASETS) + ' \\\\')
    tex += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    md.append('| Ours (f+β·log λ_u) | | ' + ' | '.join(fmt(o[ds], False) for ds, _ in DATASETS) + ' |')
    out = OUTPUTS
    open(os.path.join(out, 'table_transfer_small.tex'), 'w', encoding='utf-8').write('\n'.join(tex) + '\n')
    open(os.path.join(out, 'table_transfer_small.md'), 'w', encoding='utf-8').write('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
