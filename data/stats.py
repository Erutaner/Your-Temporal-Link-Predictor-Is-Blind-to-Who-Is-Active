# -*- coding: utf-8 -*-
"""Statistics of the thirteen streams, computed from the processed files (the table of the appendix).

    python stats.py --data_root <data_root> [--out_dir ../outputs] [--model_dir ../model] [--datasets ...]

Per stream: the standard columns of the DyGLib benchmark table (domain, nodes, events, feature dimensions, bipartite,
span, distinct timestamps, time unit), all recomputed from the released files except the domain and the time unit, plus
three quantities this paper's experiments depend on:
  decades, K     the range of inter-event times covered by the time-scale grid (log10 of span / smallest positive gap,
                 training events only) and the number of scales it yields with the default two per decade, at most 16;
  tied events    share of events that share their timestamp with at least one other event of the stream: what the
                 rule "only events strictly earlier than the query are visible" acts on;
  repeated pairs share of test-period events whose (unordered) node pair already occurred earlier in the stream.
The split is the one every model uses (model/tgmin/vendor/dataloader.py, DyGLib's code): chronological 70 / 15 / 15
by time quantiles, training events restricted to pairs of nodes that are not held out.
Writes <out_dir>/table_datasets.tex and .md and prints the markdown table.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ['wikipedia', 'reddit', 'mooc', 'lastfm', 'enron', 'SocialEvo', 'uci', 'Flights', 'CanParl', 'USLegis',
            'UNtrade', 'UNvote', 'Contacts']
# display name, domain and the unit of one time step as documented by the benchmark (DyGLib, Yu et al. 2023); the
# timestamps of the released files are in seconds except for Flights (days) and US Legis. (congresses)
INFO = {
    'wikipedia': ('Wikipedia', 'Social', 's'),
    'reddit': ('Reddit', 'Social', 's'),
    'mooc': ('MOOC', 'Interaction', 's'),
    'lastfm': ('LastFM', 'Interaction', 's'),
    'enron': ('Enron', 'Social', 's'),
    'SocialEvo': ('Social Evo.', 'Proximity', 's'),
    'uci': ('UCI', 'Social', 's'),
    'Flights': ('Flights', 'Transport', 'day'),
    'CanParl': ('Can. Parl.', 'Politics', 'yr'),
    'USLegis': ('US Legis.', 'Politics', 'congr.'),
    'UNtrade': ('UN Trade', 'Economics', 'yr'),
    'UNvote': ('UN Vote', 'Politics', 'yr'),
    'Contacts': ('Contact', 'Proximity', '5 min'),
}
SECONDS_PER_UNIT = {'s': 1.0, '5 min': 1.0, 'yr': 1.0, 'day': 86400.0}      # seconds per timestamp unit (years and 5-minute steps are counted, not converted)


def feature_dim(arr):
    """Number of informative feature columns (the benchmark pads every table to 172 columns; a column that is zero
    everywhere carries nothing), or None when there is no such column."""
    n = int(np.sum(np.any(arr != 0, axis=0))) if arr.ndim == 2 else 0
    return n if n > 0 else None


def span_text(span, unit):
    """The time between the first and the last event, in days or years; congresses are counted."""
    if unit == 'congr.':
        return '%.0f congr.' % span
    days = span * SECONDS_PER_UNIT[unit] / 86400.0
    return '%.0f d' % days if days < 400 else '%.1f yr' % (days / 365.25)


def stats(ds, data_root, scale_grid, loader):
    cwd = os.getcwd()
    os.chdir(data_root)                                   # the loader reads ./processed_data/<stream>/
    try:
        node_feat, edge_feat, full, train, _val, test, _nv, new_test = loader(ds, 0.15, 0.15)
    finally:
        os.chdir(cwd)
    src, dst, ts = full.src_node_ids, full.dst_node_ids, full.node_interact_times.astype(np.float64)
    order = np.argsort(ts, kind='stable')
    assert np.all(np.diff(ts[order]) >= 0)
    n_events = int(ts.shape[0])
    nodes = np.union1d(src, dst)
    bipartite = not np.intersect1d(src, dst).size
    counts = pd.Series(ts).value_counts()
    tied = float(counts[counts > 1].sum()) / n_events
    # unordered pairs in time order: an event repeats a pair when the same pair occurred at an earlier row
    pairs = pd.DataFrame({'a': np.minimum(src, dst)[order], 'b': np.maximum(src, dst)[order]})
    repeated = pairs.duplicated(keep='first').to_numpy()          # True after the pair's first occurrence
    test_time = float(np.quantile(ts, 0.85))
    is_test = ts[order] > test_time
    assert int(is_test.sum()) == int(test.node_interact_times.shape[0]), (ds, int(is_test.sum()), int(test.node_interact_times.shape[0]))
    repeated_test = float(repeated[is_test].mean())
    lam, ginfo = scale_grid(train.node_interact_times.astype(np.float64))
    return {'name': INFO[ds][0], 'domain': INFO[ds][1], 'unit': INFO[ds][2], 'nodes': int(nodes.shape[0]), 'events': n_events,
            'node_feat': feature_dim(node_feat), 'edge_feat': feature_dim(edge_feat), 'bipartite': bipartite,
            'span': float(ts.max() - ts.min()), 'steps': int(counts.shape[0]), 'decades': float(ginfo['decades']),
            'K': int(lam.shape[0]), 'dt_min': float(ginfo['dt_min']), 'tied': tied, 'repeated_test': repeated_test,
            'test_events': int(test.node_interact_times.shape[0]), 'new_node_test_events': int(new_test.node_interact_times.shape[0])}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', required=True)
    p.add_argument('--out_dir', default=os.path.join(HERE, '..', 'outputs'))
    p.add_argument('--model_dir', default=os.path.join(HERE, '..', 'model'))
    p.add_argument('--datasets', default=','.join(DATASETS))
    a = p.parse_args()
    sys.path.insert(0, os.path.abspath(a.model_dir))
    from tgmin.sfs import scale_grid                                   # noqa: E402
    from tgmin.vendor.dataloader import get_link_prediction_data       # noqa: E402
    rows = [stats(ds, os.path.abspath(a.data_root), scale_grid, get_link_prediction_data) for ds in a.datasets.split(',')]
    for r in rows:
        print('%-12s events %9d nodes %6d steps %8d span %.4g %s = %s (dt_min %.4g) decades %.2f K %2d node feat %s edge feat %s '
              'tied %.1f%% repeated test pairs %.1f%% (test events %d, new-node test events %d)'
              % (r['name'], r['events'], r['nodes'], r['steps'], r['span'], r['unit'], span_text(r['span'], r['unit']), r['dt_min'],
                 r['decades'], r['K'], r['node_feat'], r['edge_feat'], 100 * r['tied'], 100 * r['repeated_test'], r['test_events'],
                 r['new_node_test_events']), flush=True)
    assert all(r['node_feat'] is None for r in rows), 'a stream has node features: add the column back'
    head_tex = ['Dataset', 'Domain', '\\#Nodes', '\\#Events', 'Feat.', 'Bip.', 'Span', 'Steps / unit', 'Decades', '$K$',
                'Tied (\\%)', 'Rep. (\\%)']
    head_md = ['Dataset', 'Domain', '#Nodes', '#Events', 'Feat.', 'Bip.', 'Span', 'Steps / unit', 'Decades', 'K', 'Tied (%)',
               'Rep. (%)']
    caption = ('The thirteen streams, measured on the released files. Feat. = number of event feature columns that are not '
               'identically zero (no stream has node features; the benchmark pads every feature table to 172 columns; -- = none); '
               'Bip. = bipartite; span = time between the first and the last event; steps = number of distinct timestamps, in the '
               'unit of one time step. Decades = $\\log_{10}$ of the ratio between the span of the training events and their smallest '
               'positive inter-event gap, the range the timescale grid covers; $K$ = the number of timescales it yields (two intervals per decade, '
               'at most 16). Tied = share of events that share their timestamp with another event, which a query at that time cannot see '
               '(Appendix~\\ref{app:data_splits_visibility}). Rep. = share of test-period events whose node pair already occurred earlier in the '
               'stream. The spans of MOOC and LastFM differ from the durations given in the benchmark\'s own table (17 months and '
               '1 month): the timestamps of the released files, in seconds, cover 30 days and 4.3 years.')
    tex = ['% requires booktabs', '\\begin{table}[t]', '\\caption{%s}' % caption, '\\label{tab:datasets}', '\\centering', '\\scriptsize',
           '\\setlength{\\tabcolsep}{1.5pt}', '\\begin{tabular}{llrrcclrrrrr}', '\\toprule', ' & '.join(head_tex) + ' \\\\', '\\midrule']
    md = ['| ' + ' | '.join(head_md) + ' |', '|' + '---|' * len(head_md)]
    for r in rows:
        cells = [r['name'], r['domain'], '{:,}'.format(r['nodes']), '{:,}'.format(r['events']),
                 '--' if r['edge_feat'] is None else str(r['edge_feat']), 'yes' if r['bipartite'] else 'no',
                 span_text(r['span'], r['unit']), '{:,} / {}'.format(r['steps'], r['unit']), '%.1f' % r['decades'], str(r['K']),
                 '%.1f' % (100 * r['tied']), '%.1f' % (100 * r['repeated_test'])]
        tex.append(' & '.join(cells) + ' \\\\')
        md.append('| ' + ' | '.join(cells) + ' |')
    tex += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    os.makedirs(a.out_dir, exist_ok=True)
    open(os.path.join(a.out_dir, 'table_datasets.tex'), 'w', encoding='utf-8').write('\n'.join(tex) + '\n')
    open(os.path.join(a.out_dir, 'table_datasets.md'), 'w', encoding='utf-8').write('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
