# -*- coding: utf-8 -*-
"""S2 (cost versus stream length) appendix table from results/speed/results_s23.jsonl: LastFM and Flights
prefixes 10 / 30 / 100 %, each split 70/15/15 on its own; training time per epoch and test-split scoring time
(median of 3), with per-100k normalisations.  Writes outputs/table_s2_stream_length.tex / .md.
"""
import collections
import json
import os

import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common import RESULTS, OUTPUTS  # noqa: E402

RES = RESULTS
OUT = OUTPUTS
ROWS = [('lastfm', 'LastFM'), ('Flights', 'Flights')]
PREF = [('p10', '10\\%', '10%'), ('p30', '30\\%', '30%'), ('p100', '100\\%', '100%')]
CAPTION = ('Cost as the stream gets longer. We cut LastFM and Flights to their first 10\\%, 30\\% and 100\\% of events, split each '
           'prefix 70/15/15 by time on its own, trained the main configuration on it and timed one training epoch and the scoring of '
           'its test split (one RTX~5090, median of three runs). The last two columns divide by the amount of work: both times grow '
           'at most linearly with the length of the stream, and on the short prefixes the fixed cost per block dominates (LastFM keeps '
           '$C=192$ training blocks at every length, so its 10\\% prefix pays about 12\\,s of block overhead per epoch; Flights uses '
           'one block per time step: 5 / 17 / 90 blocks).')


def fmt_n(n):
    return '%.2fM' % (n / 1e6) if n >= 1e6 else '%dk' % round(n / 1e3)


def main():
    recs = [json.loads(l) for l in open(os.path.join(RES, 'speed', 'results_s23.jsonl'), encoding='utf-8') if l.strip()]
    by = collections.defaultdict(list)
    for r in recs:
        if r['exp'] == 'S2' and r['measure'] is not None:
            by[(r['dataset'], r['variant'], r['phase'])].append(r['measure'])
    head = ['stream', 'prefix', 'train edges', 'test positives', 'epoch time (s)', 's per 100k edges', 'scoring time (s)', 's per 100k positives']
    tex = ['% requires booktabs, multirow', '\\begin{table}[t]', '\\caption{%s}' % CAPTION, '\\label{tab:s2_stream_length}', '\\centering',
           '\\footnotesize', '\\setlength{\\tabcolsep}{5pt}', '\\begin{tabular}{llrrrrrr}', '\\toprule',
           'stream & prefix & train & test & epoch & s per & scoring & s per \\\\',
           ' & & edges & positives & time (s) & 100k edges & time (s) & 100k positives \\\\', '\\midrule']
    md = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for ds, name in ROWS:
        for i, (v, ptex, pmd) in enumerate(PREF):
            tr, ev = by[(ds, v, 'train')], by[(ds, v, 'eval')]
            e, n = tr[0]['train_edges'], ev[0]['positives']
            ts = float(np.median([m['train_seconds'] for m in tr]))
            es = float(np.median([m['seconds'] for m in ev]))
            cells = [fmt_n(e), fmt_n(n), '%.1f' % ts, '%.1f' % (ts / e * 1e5), '%.1f' % es, '%.1f' % (es / n * 1e5)]
            lead = ('\\multirow{3}{*}{%s}' % name) if i == 0 else ''
            tex.append('%s & %s & ' % (lead, ptex) + ' & '.join(cells) + ' \\\\')
            md.append('| %s | %s | ' % (name if i == 0 else '', pmd) + ' | '.join(cells) + ' |')
        if ds != ROWS[-1][0]:
            tex.append('\\midrule')
    tex += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    md += ['', 'Prefixes of the stream, each split 70/15/15 on its own; one RTX 5090, median of 3 runs. LastFM keeps C = 192 training '
               'blocks at every length; Flights uses one block per time step (5 / 17 / 90).']
    open(os.path.join(OUT, 'table_s2_stream_length.tex'), 'w', encoding='utf-8').write('\n'.join(tex) + '\n')
    open(os.path.join(OUT, 'table_s2_stream_length.md'), 'w', encoding='utf-8').write('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
