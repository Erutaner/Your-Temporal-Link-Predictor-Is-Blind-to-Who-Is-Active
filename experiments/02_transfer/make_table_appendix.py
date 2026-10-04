# -*- coding: utf-8 -*-
"""Transfer of the source-activity term, the protocols not shown in the main-text table: what each backbone gains over
itself in the transductive setting under inductive negatives and in the inductive setting under historical negatives
(its inductive protocol coincides with the historical one).  One row block per backbone, one row per stream and a mean
row; columns = the backbone alone, with the term, and the difference.  Under random negatives the validation split
selects beta = 0 for every backbone and seed, so those columns would show no change and are left out.  Test AP x100,
mean +- std over seeds (the difference is the mean and std of the per-seed differences).
Writes outputs/table_transfer_appendix.tex and .md.

    python make_table_appendix.py [--size scriptsize]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common import RESULTS, OUTPUTS  # noqa: E402

RES = RESULTS
DATASETS = [('wikipedia', 'Wikipedia'), ('reddit', 'Reddit'), ('uci', 'UCI'), ('enron', 'Enron'), ('lastfm', 'LastFM')]
BACKBONES = [('TPNet_transfer', 'TPNet', 'TPNet'), ('TGN_transfer', 'TGN$^\\dagger$', 'TGN†'),
             ('DyGFormer_fixpad_transfer', 'DyGFormer$^\\dagger$', 'DyGFormer†'), ('DSRD_transfer', 'DSRD$^\\dagger$', 'DSRD†')]
GROUPS = [('transductive', 'inductive', 'Transductive, ind'), ('inductive', 'historical', 'Inductive, hist')]
CAPTION = ('Transfer of the source-activity term, remaining protocols: what each backbone gains over itself. Test AP (\\%) of the '
           'backbone alone and with $\\beta\\log\\lambda_u$ added to its logits, and the difference $\\Delta$, in the transductive '
           'setting under inductive (ind) negatives and in the inductive setting under historical (hist) negatives; the transductive '
           'setting under historical negatives is in Table~\\ref{tab:transfer}. Mean{\\scriptsize$\\pm$std} over three seeds '
           '(DyGFormer$^\\dagger$ on LastFM: one), the std of $\\Delta$ over the per-seed differences; the mean row averages the five '
           'streams. $\\lambda_u$ keeps the parameters fitted for our model on the same stream and $\\beta$ is chosen on the '
           'backbone\'s validation split. Under random negatives the validation split selects $\\beta=0$ for every backbone and seed, so '
           'the backbone\'s score is unchanged there and those columns are omitted. In the inductive setting the two samplers draw '
           'from the same new-node subset, so only the historical protocol is shown. $\\dagger$: retrained by us with the corrected '
           'released code.')


def load(key, ds):
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, 'transfer', key, ds, 'seed*_transfer.json')))]


def cells(runs, setting, protocol):
    alone = np.array([100 * r['backbone_only'][setting][protocol]['test_ap'] for r in runs])
    plus = np.array([100 * r['selected'][setting][protocol]['test_ap'] for r in runs])
    return [(float(x.mean()), float(x.std()), len(x)) for x in (alone, plus, plus - alone)]


def fmt(v, tex, signed=False):
    m, s, n = v
    txt = ('%+.2f' if signed else '%.2f') % m
    if n == 1:
        return txt
    return txt + (('{\\tiny$\\pm$%.2f}' if tex else '±%.2f') % s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--size', default='scriptsize')
    p.add_argument('--out_dir', default=OUTPUTS)
    a = p.parse_args()
    n = 3 * len(GROUPS)
    tex = ['% requires booktabs', '\\begin{table}[t]', '\\caption{%s}' % CAPTION, '\\label{tab:transfer_appendix}', '\\centering',
           '\\%s' % a.size, '\\setlength{\\tabcolsep}{4pt}', '\\renewcommand{\\arraystretch}{0.95}',
           '\\begin{tabular}{l' + 'ccc' * len(GROUPS) + '}', '\\toprule',
           ' & ' + ' & '.join('\\multicolumn{3}{c}{%s}' % g[2] for g in GROUPS) + ' \\\\',
           ' '.join('\\cmidrule(lr){%d-%d}' % (2 + 3 * i, 4 + 3 * i) for i in range(len(GROUPS))),
           'backbone / stream & ' + ' & '.join('alone & $+\\lambda$ & $\\Delta$' for _ in GROUPS) + ' \\\\', '\\midrule']
    md = ['| backbone / stream | ' + ' | '.join('%s alone | +λ | Δ' % g[2] for g in GROUPS) + ' |', '|' + '---|' * (n + 1)]
    for bi, (key, bt, bm) in enumerate(BACKBONES):
        if bi:
            tex.append('\\midrule')
        tex.append('\\multicolumn{%d}{l}{\\textit{%s}} \\\\' % (n + 1, bt))
        md.append('| *%s* |' % bm + ' |' * n)
        per_stream = []
        for ds, name in DATASETS:
            runs = load(key, ds)
            assert runs, (key, ds)
            for r in runs:
                for setting in ('transductive', 'inductive'):
                    assert r['selected'][setting]['random']['beta'] == 0.0, (key, ds, setting)
            row = [c for setting, protocol, _ in GROUPS for c in cells(runs, setting, protocol)]
            per_stream.append(row)
            tex.append('%s & %s \\\\' % (name, ' & '.join(fmt(v, True, signed=(j % 3 == 2)) for j, v in enumerate(row))))
            md.append('| %s | %s |' % (name, ' | '.join(fmt(v, False, signed=(j % 3 == 2)) for j, v in enumerate(row))))
        means = [float(np.mean([row[j][0] for row in per_stream])) for j in range(n)]
        tex.append('\\textit{mean} & ' + ' & '.join(('%+.2f' if j % 3 == 2 else '%.2f') % m for j, m in enumerate(means)) + ' \\\\')
        md.append('| *mean* | ' + ' | '.join(('%+.2f' if j % 3 == 2 else '%.2f') % m for j, m in enumerate(means)) + ' |')
    tex += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    os.makedirs(a.out_dir, exist_ok=True)
    open(os.path.join(a.out_dir, 'table_transfer_appendix.tex'), 'w', encoding='utf-8').write('\n'.join(tex) + '\n')
    open(os.path.join(a.out_dir, 'table_transfer_appendix.md'), 'w', encoding='utf-8').write('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
