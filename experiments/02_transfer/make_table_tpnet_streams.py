# -*- coding: utf-8 -*-
"""Transfer of the source-activity term to TPNet on the eight streams outside the transfer study (one seed each):
test AP under historical and inductive negatives in the transductive setting and under historical negatives in the
inductive setting, the backbone alone and with the term.  Writes outputs/table_transfer_tpnet_streams.tex and .md.

    python make_table_tpnet_streams.py [--size scriptsize]
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
DATASETS = [('mooc', 'MOOC'), ('SocialEvo', 'Social Evo.'), ('Flights', 'Flights'), ('CanParl', 'Can. Parl.'),
            ('USLegis', 'US Legis.'), ('UNtrade', 'UN Trade'), ('UNvote', 'UN Vote'), ('Contacts', 'Contact')]
# (setting, protocol)
COLS = [('transductive', 'historical'), ('transductive', 'inductive'), ('inductive', 'historical')]
CAPTION = ('Transfer of the source-activity term to TPNet on the eight streams outside the transfer study: test AP (\\%) of the '
           'backbone alone and with $\\beta\\log\\lambda_u$ added, in the transductive setting under historical (hist) and inductive '
           '(ind) negatives and in the inductive setting under historical negatives (its inductive protocol coincides), one seed per '
           'stream. $\\lambda_u$ keeps the parameters fitted for our model on the same stream; $\\beta$ is chosen on TPNet\'s validation '
           'split, and $\\Delta$ is the change. %s')


def values(ds):
    fs = sorted(glob.glob(os.path.join(RES, 'transfer', 'TPNet_transfer', ds, 'seed*_transfer.json')))
    runs = [json.load(open(f)) for f in fs]
    assert len(runs) == 1, (ds, len(runs))
    r = runs[0]
    out = []
    for setting, protocol in COLS:
        alone = 100 * r['backbone_only'][setting][protocol]['test_ap']
        plus = 100 * r['selected'][setting][protocol]['test_ap']
        out.append((alone, plus, plus - alone, r['selected'][setting][protocol]['beta']))
    # random negatives are not shown: record the cases in which the validation split selected beta > 0 there
    exceptions = []
    for setting in ('transductive', 'inductive'):
        s = r['selected'][setting]['random']
        if s['beta'] != 0.0:
            exceptions.append((setting, s['beta'], 100 * r['backbone_only'][setting]['random']['test_ap'], 100 * s['test_ap']))
    return out, exceptions


def random_sentence(exc):
    """The caption sentence about random negatives, written from the data."""
    if not exc:
        return 'Under random negatives $\\beta=0$ is selected on every stream and nothing changes.'
    parts = ['%s in the %s setting ($\\beta=%g$, %.2f to %.2f)' % (name, setting, beta, al, pl) for name, setting, beta, al, pl in exc]
    return ('Under random negatives $\\beta=0$ is selected on every stream and setting except %s; the random columns are '
            'therefore omitted.' % '; '.join(parts))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--size', default='scriptsize')
    p.add_argument('--out_dir', default=OUTPUTS)
    a = p.parse_args()
    rows, exceptions = [], []
    for ds, name in DATASETS:
        v, exc = values(ds)
        rows.append((name, v))
        exceptions += [(name,) + e for e in exc]
    caption = CAPTION.replace('%s', random_sentence(exceptions), 1)
    tex = ['% requires booktabs', '\\begin{table}[t]', '\\caption{%s}' % caption, '\\label{tab:transfer_tpnet_streams}', '\\centering',
           '\\%s' % a.size, '\\setlength{\\tabcolsep}{4pt}', '\\renewcommand{\\arraystretch}{0.95}',
           '\\begin{tabular}{l' + 'ccc' * len(COLS) + '}', '\\toprule',
           ' & \\multicolumn{3}{c}{Transductive, hist} & \\multicolumn{3}{c}{Transductive, ind} & \\multicolumn{3}{c}{Inductive, hist} \\\\',
           '\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10}',
           'Dataset & alone & $+\\lambda$ & $\\Delta$ & alone & $+\\lambda$ & $\\Delta$ & alone & $+\\lambda$ & $\\Delta$ \\\\', '\\midrule']
    md = ['| Dataset | T hist alone | +λ | Δ | T ind alone | +λ | Δ | I hist alone | +λ | Δ |', '|' + '---|' * 10]
    for name, v in rows:
        tex.append('%s & %s \\\\' % (name, ' & '.join('%.2f & %.2f & %+.2f' % (al, pl, d) for al, pl, d, _ in v)))
        md.append('| %s | %s |' % (name, ' | '.join('%.2f | %.2f | %+.2f' % (al, pl, d) for al, pl, d, _ in v)))
    tex += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    md.append('')
    md.append(random_sentence(exceptions).replace('$\\beta', 'β').replace('$', '').replace('\\', ''))
    os.makedirs(a.out_dir, exist_ok=True)
    open(os.path.join(a.out_dir, 'table_transfer_tpnet_streams.tex'), 'w', encoding='utf-8').write('\n'.join(tex) + '\n')
    open(os.path.join(a.out_dir, 'table_transfer_tpnet_streams.md'), 'w', encoding='utf-8').write('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
