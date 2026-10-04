# -*- coding: utf-8 -*-
"""Speed tables from results/speed/results.jsonl (one exclusive RTX 5090, batch 200, median of 3 repeats).

Per (model, dataset):
  train epoch [s]      released models: ms per training batch (300 timed batches after warm-up) x batches per epoch;
                       ours: one full epoch measured (the second of two).
  test split [s]       released models: ms per evaluation batch (up to 500 timed batches, random negatives) x test
                       batches; ours: the full transductive test scoring pass measured (random negatives).
  pairs / s            scored (positive + negative) pairs per second during the test split.
  peak memory [MB]     torch.cuda.max_memory_allocated during the timed evaluation window (training window too).
    python make_table_all_models.py [--md out.md]
"""
import argparse
import collections
import json
import os

import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common import RESULTS, OUTPUTS  # noqa: E402

ROOT = os.path.join(RESULTS, 'speed')
DATASETS = ['uci', 'wikipedia', 'enron', 'reddit', 'lastfm', 'Flights']
NAMES = {'uci': 'UCI', 'wikipedia': 'Wikipedia', 'enron': 'Enron', 'reddit': 'Reddit', 'lastfm': 'LastFM', 'Flights': 'Flights'}
MODELS = ['JODIE', 'DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer', 'EdgeBank', 'TPNet', 'DSRD', 'SFS']
LABEL = {'SFS': 'ours'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--md', default=os.path.join(OUTPUTS, 'speed_table.md'))
    a = p.parse_args()
    recs = [json.loads(l) for l in open(os.path.join(ROOT, 'results.jsonl'), encoding='utf-8') if l.strip()]
    by = collections.defaultdict(list)
    for r in recs:
        if r['measure'] is not None:
            by[(r['model'], r['dataset'], r['phase'])].append(r['measure'])
    tables = {'train': {}, 'eval': {}, 'pairs': {}, 'mem_eval': {}, 'mem_train': {}, 'flag': {}}
    for m in MODELS:
        for ds in DATASETS:
            tr = by.get((m, ds, 'train'), [])
            ev = by.get((m, ds, 'eval'), [])
            if m == 'SFS':
                if tr:
                    tables['train'][(m, ds)] = float(np.median([x['train_seconds'] for x in tr]))
                    tables['mem_train'][(m, ds)] = float(np.median([x['peak_mem_mb'] for x in tr]))
                if ev:
                    sec = float(np.median([x['seconds'] for x in ev]))
                    tables['eval'][(m, ds)] = sec
                    tables['pairs'][(m, ds)] = 2.0 * ev[0]['positives'] / sec
                    tables['mem_eval'][(m, ds)] = float(np.median([x['peak_mem_mb'] for x in ev]))
                    tables['flag'][(m, ds)] = 'measured'
            else:
                if tr:
                    tables['train'][(m, ds)] = float(np.median([x['ms_per_batch'] * x['total_batches'] / 1000.0 for x in tr]))
                    tables['mem_train'][(m, ds)] = float(np.median([x['peak_mem_alloc_mb'] for x in tr]))
                if ev:
                    tables['eval'][(m, ds)] = float(np.median([x['ms_per_batch'] * x['total_batches'] / 1000.0 for x in ev]))
                    tables['pairs'][(m, ds)] = float(np.median([2.0 * x['batch_size'] / (x['ms_per_batch'] / 1000.0) for x in ev]))
                    tables['mem_eval'][(m, ds)] = float(np.median([x['peak_mem_alloc_mb'] for x in ev]))
                    tables['flag'][(m, ds)] = 'extrapolated' if ev[0]['total_batches'] > ev[0]['timed_batches'] + 25 else 'measured'
    out = ['# Speed benchmark: one exclusive RTX 5090, batch 200, median of 3 repeats', '',
           'Released models: per-batch times from 300 training / up to 500 evaluation timed batches after a warm-up, multiplied by the '
           'number of batches of the split (extrapolated where the split is longer than the timed window); ours: full epoch and full '
           'test-split pass measured. Evaluation = random negatives, transductive test split.', '']
    def table(title, key, fmt, note=''):
        out.append('## %s' % title)
        if note:
            out.append(''); out.append(note)
        out.append(''); out.append('| model | ' + ' | '.join(NAMES[d] for d in DATASETS) + ' |'); out.append('|' + '---|' * (len(DATASETS) + 1))
        for m in MODELS:
            row = []
            for ds in DATASETS:
                v = tables[key].get((m, ds))
                row.append('—' if v is None else fmt % v)
            out.append('| %s | %s |' % (LABEL.get(m, m), ' | '.join(row)))
        out.append('')
    table('Test-split scoring time [s]', 'eval', '%.1f')
    table('Training time per epoch [s]', 'train', '%.0f')
    table('Scored pairs per second during the test split', 'pairs', '%.0f')
    table('Peak GPU memory during evaluation [MB]', 'mem_eval', '%.0f')
    table('Peak GPU memory during training [MB]', 'mem_train', '%.0f')
    # speed-ups of ours over each model (test split)
    out.append('## Test-split time relative to ours (x slower)'); out.append('')
    out.append('| model | ' + ' | '.join(NAMES[d] for d in DATASETS) + ' |'); out.append('|' + '---|' * (len(DATASETS) + 1))
    for m in MODELS:
        if m == 'SFS':
            continue
        row = []
        for ds in DATASETS:
            a_, b_ = tables['eval'].get((m, ds)), tables['eval'].get(('SFS', ds))
            row.append('—' if a_ is None or b_ is None else '%.1fx' % (a_ / b_))
        out.append('| %s | %s |' % (m, ' | '.join(row)))
    out.append('')
    text = '\n'.join(out)
    open(a.md, 'w', encoding='utf-8').write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
