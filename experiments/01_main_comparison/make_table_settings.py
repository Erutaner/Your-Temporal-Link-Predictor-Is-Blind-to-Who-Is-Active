# -*- coding: utf-8 -*-
"""The per-stream training settings of our model as a table, read from the same place the run scripts read them
(experiments/common.py and model/configs/<stream>.json); the shared settings of the caption are checked against the
defaults of model/scripts/train_sfs.py.  Writes outputs/table_training_settings.tex and .md.

    python make_table_settings.py
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common import MODEL, OUTPUTS, DATASETS, MAIN_FLAGS, PATIENCE  # noqa: E402

SHARED = ('$d=128$, $r=16$, $256$ random projection dimensions, $M=20$, learning rate $10^{-3}$, at most $500$ epochs, '
          'early stopping after $100$ epochs without an improvement of the validation AP and AUC ($50$ on LastFM and Flights), '
          'five seeds')
CAPTION = ('Per-stream training settings of our model. Blocks: the training period is processed in $C$ blocks of equal event counts '
           '(equal) or in one block per time step (step). Table: the node embedding table $E$ is trained (yes) or holds fixed random '
           'codes (no); weight decay applies to the table only and is therefore listed only where the table is trained. Negatives: '
           'training negatives are drawn from all destinations of the training period (all) or from the destinations observed before '
           'the current block (seen). Partner updates: within-block events also update the partner-sum state '
           '(Appendix~\\ref{app:query_complexity}); they always update the counts. All other settings are shared by every stream: '
           + SHARED + '; $K$ is given in Table~\\ref{tab:datasets}.')


def check_shared_defaults():
    """The architecture constants quoted in the caption are the defaults of the training script."""
    src = open(os.path.join(MODEL, 'scripts', 'train_sfs.py'), encoding='utf-8').read()
    for flag, value in (('d', 128), ('rank', 16), ('sketch', 256), ('set_M', 20)):
        assert re.search(r"'--%s', type=int, default=%d" % (flag, value), src), 'train_sfs.py default of --%s is not %d' % (flag, value)


def rows():
    out = []
    for ds, name in DATASETS:
        cfg = json.load(open(os.path.join(MODEL, 'configs', ds + '.json')))
        flags = dict(cfg)
        flags.update(MAIN_FLAGS[ds])                     # the run scripts pass MAIN_FLAGS on top of the config
        blocks = '%d equal' % flags['t_train'] if cfg['chunk_mode'] == 'equal' else 'step'
        table = 'yes' if flags['e_train'] else 'no'
        wd = ('%g' % flags['wd']) if flags['e_train'] else '--'
        out.append((name, blocks, table, '%g' % flags['dropout'], wd, {'train': 'all', 'seen': 'seen'}[flags['neg_pool']],
                    'yes' if flags.get('fresh_state', 0) else 'no', PATIENCE[ds]))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default=OUTPUTS)
    a = p.parse_args()
    check_shared_defaults()
    r = rows()
    # the caption's shared-settings sentence must agree with the patience values of common.py
    assert all(pat == (50 if ds in ('lastfm', 'Flights') else 100) for (ds, _), (*_, pat) in zip(DATASETS, r))
    head = ['Dataset', 'Blocks $C$', 'Table', 'Dropout', 'Weight decay', 'Negatives', 'Partner updates']
    tex = ['% requires booktabs', '\\begin{table}[t]', '\\caption{%s}' % CAPTION, '\\label{tab:training_settings}', '\\centering',
           '\\footnotesize', '\\setlength{\\tabcolsep}{5pt}', '\\begin{tabular}{lcccccc}', '\\toprule', ' & '.join(head) + ' \\\\', '\\midrule']
    md = ['| ' + ' | '.join(h.replace('$', '') for h in head) + ' |', '|' + '---|' * len(head)]
    for name, blocks, table, dropout, wd, negatives, partner, _ in r:
        tex.append('%s & %s & %s & %s & %s & %s & %s \\\\' % (name, blocks, table, dropout, wd, negatives, partner))
        md.append('| %s | %s | %s | %s | %s | %s | %s |' % (name, blocks, table, dropout, wd, negatives, partner))
    tex += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    md += ['', 'Shared by every stream: ' + SHARED.replace('$', '') + '.']
    os.makedirs(a.out_dir, exist_ok=True)
    open(os.path.join(a.out_dir, 'table_training_settings.tex'), 'w', encoding='utf-8').write('\n'.join(tex) + '\n')
    open(os.path.join(a.out_dir, 'table_training_settings.md'), 'w', encoding='utf-8').write('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
