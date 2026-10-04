# -*- coding: utf-8 -*-
"""Aggregate the retrained baselines (results/baselines/<model>/<dataset>/seed<S>_<strategy>.json,
DyGLib result format) into mean / std over seeds per (model, dataset, table, strategy), the structure
make_tables.py reads (results/baselines/summary.json).

    python collect_baselines.py <baselines_dir> [--out summary.json] [--md summary.md]

Tables: trans_ap / trans_auc from "test metrics", ind_ap / ind_auc from "new node test metrics";
strategies rnd / hist / ind from the seed<S>_{random,historical,inductive}.json files.
"""
import argparse
import glob
import json
import os
import re

import numpy as np

STRATS = {'random': 'rnd', 'historical': 'hist', 'inductive': 'ind'}
TABLES = {'trans_ap': ('test metrics', 'average_precision'), 'trans_auc': ('test metrics', 'roc_auc'),
          'ind_ap': ('new node test metrics', 'average_precision'), 'ind_auc': ('new node test metrics', 'roc_auc')}


def collect(root):
    out = {}
    for model in sorted(os.listdir(root)):
        mdir = os.path.join(root, model)
        if not os.path.isdir(mdir):
            continue
        for ds in sorted(os.listdir(mdir)):
            ddir = os.path.join(mdir, ds)
            cells = {}
            for f in glob.glob(os.path.join(ddir, 'seed*_*.json')):
                m = re.match(r'seed(\d+)_(random|historical|inductive)\.json$', os.path.basename(f))
                if not m:
                    continue
                seed, strat = int(m.group(1)), STRATS[m.group(2)]
                r = json.load(open(f))
                for tab, (block, metric) in TABLES.items():
                    if block in r and metric in r[block]:
                        cells.setdefault((tab, strat), {})[seed] = float(r[block][metric])
            if cells:
                out.setdefault(model, {})[ds] = {
                    '%s/%s' % (tab, strat): {'mean': 100 * float(np.mean(list(v.values()))),
                                             'std': 100 * float(np.std(list(v.values()))),
                                             'n': len(v), 'seeds': sorted(v)}
                    for (tab, strat), v in cells.items()}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('baselines_dir')
    p.add_argument('--out', default=None)
    p.add_argument('--md', default=None)
    a = p.parse_args()
    summary = collect(a.baselines_dir)
    lines = []
    for model, dss in summary.items():
        lines.append('### %s' % model)
        lines.append('')
        lines.append('| dataset | n | trans AP rnd | hist | ind | ind-setting AP rnd | hist | ind |')
        lines.append('|---|---|---|---|---|---|---|---|')
        for ds, cells in dss.items():
            def c(k):
                v = cells.get(k)
                return ('%.2f ± %.2f' % (v['mean'], v['std'])) if v else '—'
            n = max(v['n'] for v in cells.values())
            lines.append('| %s | %d | %s | %s | %s | %s | %s | %s |' % (
                ds, n, c('trans_ap/rnd'), c('trans_ap/hist'), c('trans_ap/ind'), c('ind_ap/rnd'), c('ind_ap/hist'), c('ind_ap/ind')))
        lines.append('')
    text = '\n'.join(lines)
    print(text)
    if a.out:
        with open(a.out, 'w', encoding='utf-8') as fh:
            json.dump(summary, fh, indent=1)
    if a.md:
        with open(a.md, 'w', encoding='utf-8') as fh:
            fh.write(text + '\n')


if __name__ == '__main__':
    main()
