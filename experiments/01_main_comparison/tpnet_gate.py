# -*- coding: utf-8 -*-
"""TPNet paper-value rule: a dataset's paper cells are kept if the corrected code, averaged over the seeds run
(seed 0 on every dataset; five seeds where the seed-0 run fell outside the tolerance), is not below paper - max(1.0, 2 x paper std) on all six
AP cells (transductive and inductive-setting x random / historical / inductive negatives); where the paper
reports no standard deviation the tolerance is 1.0 point.  A dataset that passes keeps its paper values in the
main tables; a dataset that drops is retrained for all five seeds and the retrained value replaces the paper cell.
Writes results/baselines/tpnet_gate.json, which make_tables.py reads.

    python tpnet_gate.py [--seeds 0,1,2,3,4]
"""
import argparse
import glob
import json
import os
import re

import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common import RESULTS  # noqa: E402
BASE = os.path.join(RESULTS, 'baselines', 'TPNet_strict')
PAPER = os.path.join(RESULTS, 'main', 'tpnet_paper.json')
NAMES = {'wikipedia': 'Wikipedia', 'reddit': 'Reddit', 'mooc': 'MOOC', 'lastfm': 'LastFM', 'enron': 'Enron',
         'SocialEvo': 'Social Evo.', 'uci': 'UCI', 'Flights': 'Flights', 'CanParl': 'Can. Parl.', 'USLegis': 'US Legis.',
         'UNtrade': 'UN Trade', 'UNvote': 'UN Vote', 'Contacts': 'Contact'}
TOL = 1.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', default='0,1,2,3,4', help='seeds whose mean is compared with the paper (those present)')
    a = p.parse_args()
    seeds = set(int(x) for x in a.seeds.split(','))
    paper = json.load(open(PAPER))
    # per-cell standard deviations parsed from the paper text (tolerance = max(TOL, 2 x std)); 1.0 where absent
    std_path = os.path.join(os.path.dirname(PAPER), 'tpnet_paper_std.json')
    paper_std = json.load(open(std_path)) if os.path.exists(std_path) else {}
    verdicts = {}
    for ds in sorted(os.listdir(BASE)) if os.path.isdir(BASE) else []:
        cells = {}
        for f in glob.glob(os.path.join(BASE, ds, 'seed*_*.json')):
            m = re.match(r'seed(\d+)_(random|historical|inductive)\.json$', os.path.basename(f))
            if not m or int(m.group(1)) not in seeds:
                continue
            r = json.load(open(f))
            blk = {'random': 'rnd', 'historical': 'hist', 'inductive': 'ind'}[m.group(2)]
            cells.setdefault(('trans_ap', blk), []).append(100 * float(r['test metrics']['average_precision']))
            cells.setdefault(('ind_ap', blk), []).append(100 * float(r['new node test metrics']['average_precision']))
        if len(cells) < 6:
            print('%-10s incomplete (%d of 6 cells)' % (ds, len(cells)))
            continue
        worst, rows, leak = 0.0, [], False
        for (tab, blk), vals in sorted(cells.items()):
            p = paper[tab][blk].get(NAMES[ds])
            if p is None:
                rows.append('%s/%s strict %.2f paper —' % (tab, blk, np.mean(vals)))
                continue
            gap = float(p) - float(np.mean(vals))
            worst = max(worst, gap)
            sd = paper_std.get('%s/%s/%s' % (tab, blk, NAMES[ds]))
            tol = max(TOL, 2 * sd) if sd is not None else TOL
            over = gap > tol
            leak = leak or over
            rows.append('%s/%s strict %.2f paper %.2f ± %s (gap %+.2f, tol %.2f)%s' % (
                tab, blk, np.mean(vals), float(p), ('%.2f' % sd) if sd is not None else '?', gap, tol, '  <-- over' if over else ''))
        n = max(len(v) for v in cells.values())
        verdicts[ds] = leak
        print('%-10s n=%d %s  worst gap %.2f\n    %s' % (ds, n, 'drops -> retrained values' if leak else 'passes -> paper values', worst, '\n    '.join(rows)))
    # outcome file read by make_tables.py: datasets that drop get the retrained value
    with open(os.path.join(os.path.dirname(BASE), 'tpnet_gate.json'), 'w', encoding='utf-8') as fh:
        json.dump({'tolerance': TOL, 'leak': verdicts}, fh, indent=1)


if __name__ == '__main__':
    main()
