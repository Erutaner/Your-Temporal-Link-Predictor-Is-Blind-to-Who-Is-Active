# -*- coding: utf-8 -*-
"""The four main comparison tables (transductive / inductive setting x AP / AUC-ROC), DSRD-style layout:
three horizontal blocks (random / historical / inductive negative sampling), 13 datasets per block, a
per-block Avg. Rank row and an Overall Avg. Rank row; bold = best, underline = second best (per row).

Columns: nine baselines of the TPNet paper (JODIE, TGN, CAWN, EdgeBank, GraphMixer, NAT, PINT, DyGFormer, TPNet;
DyRep / TGAT / TCL dropped), GRN, DSRD, Ours (trunk, beta = 0) and Ours+lambda (beta chosen on validation).

Cell sources
  paper       TPNet paper Tables 1, 6-10 (results/main/tpnet_paper_all_baselines.json, parsed + validated)
  retrain (+) our retraining with the corrected code, official configuration (results/baselines/summary.json):
              TPNet on the datasets where the corrected code, averaged over its seeds, does not reproduce the paper (tpnet_gate.json),
              DyGFormer (fixed-length padding) on every dataset we retrained, GRN and DSRD on every dataset.
  ours        results/main/<ds>/<ds>_final_run<S>_eval.json, mean +- std over the seeds present.

    python make_tables.py
"""
import argparse
import glob
import json
import os

import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common import RESULTS, OUTPUTS  # noqa: E402

RES = RESULTS
ORDER = [('wikipedia', 'Wikipedia'), ('reddit', 'Reddit'), ('mooc', 'MOOC'), ('lastfm', 'LastFM'),
         ('enron', 'Enron'), ('SocialEvo', 'Social Evo.'), ('uci', 'UCI'), ('Flights', 'Flights'),
         ('CanParl', 'Can. Parl.'), ('USLegis', 'US Legis.'), ('UNtrade', 'UN Trade'),
         ('UNvote', 'UN Vote'), ('Contacts', 'Contact')]
BLOCKS = [('rnd', 'random'), ('hist', 'historical'), ('ind', 'inductive')]
SETTINGS = [('trans', 'transductive', 'Transductive'), ('ind', 'inductive', 'Inductive')]
METRICS = [('ap', 'AP'), ('auc', 'AUC')]
PAPER_COLS = ['JODIE', 'TGN', 'CAWN', 'EdgeBank', 'GraphMixer', 'NAT', 'PINT', 'DyGFormer', 'TPNet']
# the inductive setting has no EdgeBank in the reference papers: its column goes to TGAT there (the classic inductive model)
COLS_BY_SETTING = {'trans': PAPER_COLS + ['GRN', 'DSRD', 'Ours', 'Ours+λ'],
                   'ind': [c if c != 'EdgeBank' else 'TGAT' for c in PAPER_COLS] + ['GRN', 'DSRD', 'Ours', 'Ours+λ']}
# the main-text table (transductive AP) is tight; the three appendix tables get normal spacing
ROW_SPACING = {('trans', 'ap'): '\\renewcommand{\\arraystretch}{0.85}\\setlength{\\tabcolsep}{3pt}%'}
ROW_DEFAULT = '\\renewcommand{\\arraystretch}{1.0}\\setlength{\\tabcolsep}{4pt}%'
TEX_HEAD = {'GRN': r'GRN$^\dagger$', 'DSRD': r'DSRD$^\dagger$', 'Ours+λ': r'Ours+$\lambda$'}
MD_HEAD = {'GRN': 'GRN†', 'DSRD': 'DSRD†'}
RETRAIN_MODEL = {'TPNet': 'TPNet_strict', 'DyGFormer': 'DyGFormer_fixpad', 'GRN': 'GRN_honest', 'DSRD': 'DSRD_honest'}


def load_ours():
    rows = {}
    for key, _ in ORDER:
        files = sorted(glob.glob(os.path.join(RES, 'main', key, '%s_final_run*_eval.json' % key)))
        evs = [json.load(open(f)) for f in files]
        r = {'n': len(evs)}
        for s_key, s_full, _ in SETTINGS:
            for m_key, _ in METRICS:
                for blk, strat in BLOCKS:
                    trunk = [100 * e['grid'][s_full]['test'][strat]['by_beta']['0'][m_key] for e in evs]
                    full = [100 * e['selected'][s_full][strat]['test_' + m_key] for e in evs]
                    r[('Ours', s_key, m_key, blk)] = (float(np.mean(trunk)), float(np.std(trunk)), len(trunk))
                    r[('Ours+λ', s_key, m_key, blk)] = (float(np.mean(full)), float(np.std(full)), len(full))
        rows[key] = r
    return rows


def avg_ranks(values_by_row, cols):
    """values_by_row: list of {col: mean}; average rank (1 = best, ties share the mean rank) per column over
    the rows where the column has a value; None if it never has one."""
    acc = {c: [] for c in cols}
    for vals in values_by_row:
        items = sorted(vals.items(), key=lambda kv: -kv[1])
        i = 0
        while i < len(items):
            j = i
            while j + 1 < len(items) and items[j + 1][1] == items[i][1]:
                j += 1
            r = (i + 1 + j + 1) / 2.0
            for k in range(i, j + 1):
                acc[items[k][0]].append(r)
            i = j + 1
    return {c: (float(np.mean(v)) if v else None) for c, v in acc.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default=OUTPUTS)
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    paper = json.load(open(os.path.join(RES, 'main', 'tpnet_paper_all_baselines.json')))
    summary = json.load(open(os.path.join(RES, 'baselines', 'summary.json')))
    gate = json.load(open(os.path.join(RES, 'baselines', 'tpnet_gate.json')))['leak']
    ours = load_ours()
    dyg_retrained = sorted(summary['DyGFormer_fixpad'].keys())
    tp_retrained = sorted(ds for ds, leak in gate.items() if leak)
    notes = {'seed_counts': {}}

    def cell(col, key, name, s_key, m_key, blk):
        """(mean, std, n, mark) or None."""
        tab = '%s_%s' % (s_key, m_key)
        if col in ('Ours', 'Ours+λ'):
            m, s, n = ours[key][(col, s_key, m_key, blk)]
            return (m, s, n, '')
        if col in ('GRN', 'DSRD') or (col == 'DyGFormer' and key in dyg_retrained) or (col == 'TPNet' and key in tp_retrained):
            c = summary[RETRAIN_MODEL[col]][key]['%s/%s' % (tab, blk)]
            notes['seed_counts'].setdefault(col, {})[name] = c['n']
            return (c['mean'], c['std'], c['n'], '†')
        v = paper['%s_%s' % (s_key, blk)][m_key.upper()][name].get(col)
        return None if v is None else (v[0], v[1], 5, '')

    md_all = []
    for s_key, s_full, s_title in SETTINGS:
        COLS = COLS_BY_SETTING[s_key]
        for m_key, m_title in METRICS:
            # ---- gather ----
            table = {}                      # (blk, key) -> {col: cell}
            for blk, _ in BLOCKS:
                for key, name in ORDER:
                    table[(blk, key)] = {c: cell(c, key, name, s_key, m_key, blk) for c in COLS}
            ranks = {blk: avg_ranks([{c: v[0] for c, v in table[(blk, k)].items() if v is not None} for k, _ in ORDER], COLS)
                     for blk, _ in BLOCKS}
            overall = {}
            for c in COLS:
                rs = [ranks[blk][c] for blk, _ in BLOCKS]
                overall[c] = float(np.mean(rs)) if all(r is not None for r in rs) else None
            # ---- render ----
            def marks(vals):
                """{col: 'b'|'u'} best / second best by mean (ties share)."""
                present = sorted({round(v[0], 2) for v in vals.values() if v is not None}, reverse=True)
                out = {}
                for c, v in vals.items():
                    if v is None:
                        continue
                    if round(v[0], 2) == present[0]:
                        out[c] = 'b'
                    elif len(present) > 1 and round(v[0], 2) == present[1]:
                        out[c] = 'u'
                return out

            def rank_marks(rk):
                present = sorted({v for v in rk.values() if v is not None})
                out = {}
                for c, v in rk.items():
                    if v is None:
                        continue
                    if v == present[0]:
                        out[c] = 'b'
                    elif len(present) > 1 and v == present[1]:
                        out[c] = 'u'
                return out

            def tex_cell(v, mk):
                if v is None:
                    return '--'
                s = ('%.2f' % v[0]) if v[2] == 1 else '%.2f{\\scriptsize$\\pm$%.2f}' % (v[0], v[1])
                if mk == 'b':
                    s = '\\textbf{%s}' % s
                elif mk == 'u':
                    s = '\\underline{%s}' % s
                if v[3]:
                    s += '$^\\dagger$'
                return s

            def md_cell(v, mk):
                if v is None:
                    return '—'
                s = ('%.2f' % v[0]) if v[2] == 1 else '%.2f±%.2f' % (v[0], v[1])
                if mk == 'b':
                    s = '**%s**' % s
                elif mk == 'u':
                    s = '<u>%s</u>' % s
                if v[3]:
                    s += '†'
                return s

            def tex_rank(v, mk):
                if v is None:
                    return '--'
                s = '%.2f' % v
                return '\\textbf{%s}' % s if mk == 'b' else ('\\underline{%s}' % s if mk == 'u' else s)

            def md_rank(v, mk):
                if v is None:
                    return '—'
                s = '%.2f' % v
                return '**%s**' % s if mk == 'b' else ('<u>%s</u>' % s if mk == 'u' else s)

            caption = ('%s link prediction, %s (\\%%), mean{\\scriptsize$\\pm$std} over seeds (five; DyGFormer$^\\dagger$ three, '
                       'LastFM one) under random (rnd), historical (hist) and inductive (ind) negative sampling. \\textbf{Bold}: best; '
                       '\\underline{underline}: second best. $\\dagger$: retrained by us with the corrected released code (released configuration).'
                       % (s_title, m_title))
            tex = ['% requires booktabs, multirow, graphicx', '\\begin{table}[t]', '\\caption{%s}' % caption, '\\label{tab:%s_%s}' % (s_key, m_key), '\\centering',
                   '\\resizebox{\\textwidth}{!}{%', ROW_SPACING.get((s_key, m_key), ROW_DEFAULT), '\\begin{tabular}{ll' + 'c' * len(COLS) + '}', '\\toprule',
                   'NSS & Dataset & ' + ' & '.join(TEX_HEAD.get(c, c) for c in COLS) + ' \\\\', '\\midrule']
            md = ['### %s link prediction, %s (x100), mean ± std over 5 seeds' % (s_title, m_title), '',
                  '| NSS | Dataset | ' + ' | '.join(MD_HEAD.get(c, c) for c in COLS) + ' |', '|---|---|' + '---|' * len(COLS)]
            for bi, (blk, _) in enumerate(BLOCKS):
                for di, (key, name) in enumerate(ORDER):
                    vals = table[(blk, key)]
                    mk = marks(vals)
                    lead = ('\\multirow{%d}{*}{\\rotatebox{90}{%s}}' % (len(ORDER) + 1, blk)) if di == 0 else ''
                    tex.append('%s & %s & ' % (lead, name) + ' & '.join(tex_cell(vals[c], mk.get(c)) for c in COLS) + ' \\\\')
                    md.append('| %s | %s | ' % (blk, name) + ' | '.join(md_cell(vals[c], mk.get(c)) for c in COLS) + ' |')
                rmk = rank_marks(ranks[blk])
                tex.append('\\cmidrule{2-%d}' % (len(COLS) + 2))
                tex.append(' & Avg. Rank & ' + ' & '.join(tex_rank(ranks[blk][c], rmk.get(c)) for c in COLS) + ' \\\\')
                md.append('| %s | **Avg. Rank** | ' % blk + ' | '.join(md_rank(ranks[blk][c], rmk.get(c)) for c in COLS) + ' |')
                tex.append('\\midrule')
            omk = rank_marks(overall)
            tex.append('\\multicolumn{2}{l}{Overall Avg. Rank} & ' + ' & '.join(tex_rank(overall[c], omk.get(c)) for c in COLS) + ' \\\\')
            md.append('| | **Overall Avg. Rank** | ' + ' | '.join(md_rank(overall[c], omk.get(c)) for c in COLS) + ' |')
            tex += ['\\bottomrule', '\\end{tabular}}', '\\end{table}']
            fn = 'table_%s_%s' % (s_key, m_key)
            with open(os.path.join(a.out_dir, fn + '.tex'), 'w', encoding='utf-8') as fh:
                fh.write('\n'.join(tex) + '\n')
            md_all += md + ['']
            print('wrote', fn + '.tex')
    foot = [
        '#### Notes', '',
        '- Cells without † are the values reported in the TPNet paper (its Tables 1 and 6-10). Transductive tables omit DyRep, TGAT and TCL; '
        'EdgeBank is not reported there for the inductive setting, so the inductive tables carry TGAT in its place (DyRep, EdgeBank and TCL omitted).',
        '- † = our retraining with the corrected released code (released configuration), mean ± std over seeds: TPNet on the datasets where the '
        'corrected code, averaged over its seeds, does not reproduce the paper (%s; 5 seeds each); DyGFormer with fixed-length padding on %s (seeds: %s); '
        'GRN and DSRD on every dataset (5 seeds; their released code is corrected).' % (
            ', '.join(dict(ORDER)[k] for k in tp_retrained), ', '.join(dict(ORDER)[k] for k in dyg_retrained),
            ', '.join('%s %d' % (n, c) for n, c in notes['seed_counts'].get('DyGFormer', {}).items())),
        '- Ours = the base score alone (beta = 0); Ours+λ = the base score plus the source-activity term with beta chosen on the validation split '
        'of the same setting and sampling; both mean ± std over 5 seeds.',
        '- Avg. Rank: mean rank over the 13 datasets of the block (1 = best; ties share the average rank); Overall = mean of the three blocks.',
        '- Under the inductive setting the historical and inductive samplers coincide by construction (both draw from the new-node subset), '
        'so the two blocks agree there for every model.',
    ]
    with open(os.path.join(a.out_dir, 'main_tables.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(md_all + foot) + '\n')
    print('wrote main_tables.md; retrained TPNet datasets:', tp_retrained, '; DyGFormer:', dyg_retrained,
          '; DyGFormer seeds:', notes['seed_counts'].get('DyGFormer'))


if __name__ == '__main__':
    main()
